import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.graph.DyGFormer import EdgeConv, MLPMixer
from utils.loss import ContrastiveLoss
from einops.layers.torch import Rearrange
from torch.nn import MultiheadAttention
from models.sparseatt import CoordAttOptimized
from models.sparsenolocal import NonLocalSparseAttention

__all__ = ['adg6']

def adg6(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = HierarchicalTemporalSpatialModel(**kwargs)
    return model

class StereoPrior(nn.Module):
    def __init__(self, num_nodes, channels, dropout=0.1):
        super(StereoPrior, self).__init__()
        self.channels = channels
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.fc1 = nn.Linear(num_nodes, num_nodes)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(num_nodes, num_nodes)
        self.gelu2 = nn.GELU()
    def forward(self, left_spatial, right_spatial, left_temporal, right_temporal):
        B, N, C = left_spatial.size()
        
        # 1. 视差特征提取
        disparity1 = torch.abs(left_spatial - right_spatial)  # (B, N, C)
        disparity2 = torch.abs(left_temporal - right_temporal)  # (B, N, C)
        disparity1 = self.fc1(disparity1.permute(0,2,1)).permute(0,2,1)
        disparity1 = self.gelu1(disparity1)
        disparity2 = self.fc2(disparity2.permute(0,2,1)).permute(0,2,1)
        disparity2 = self.gelu2(disparity2)
        return disparity1, disparity2
    

class SpatialEncoder(nn.Module):
    def __init__(self, num_nodes, channels, reduction=4):
        super(SpatialEncoder, self).__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        # 卷积层将单通道热图转换为指定通道数,下采样
        self.conv = nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=False)
        # CoordAttOptimized 模块
        self.coord_att = CoordAttOptimized(channels=channels, output_nodes=num_nodes, reduction=reduction)

    def forward(self, heatmap):
        # 通过卷积层处理热图
        features = self.conv(heatmap)  # [batch, channels, height, width]
        # 通过 CoordAttOptimized 处理特征
        output = self.coord_att(features)  # [batch, output_nodes, channels]
        return output
        

        
class TemporalEncoder(nn.Module):
    def __init__(self, num_nodes, out_channels):
        super(TemporalEncoder, self).__init__()
        self.gconv = EdgeConv(node_dim=2, edge_dim=2, time_dim=8, out_channels=out_channels, 
                              num_heads=2, dropout=0.1)
        self.fc = nn.Linear(940, num_nodes)
        self.mixer = MLPMixer(num_tokens=num_nodes, num_channels=out_channels, out_tokens=num_nodes, out_channels=out_channels)
        self.gelu = nn.GELU()

    def forward(self, graph):
        node_features = graph[0]
        timestamps = graph[1]
        edge_features = graph[2]
        edge_index = graph[3]
        temporal_feature = self.gconv(node_features, edge_index, edge_features, timestamps)
        temporal_feature = self.fc(temporal_feature.permute(0, 2, 1)).permute(0, 2, 1)
        temporal_feature = self.gelu(temporal_feature)
        return temporal_feature

class SubtaskFusion(nn.Module):
    def __init__(self, num_nodes, out_channels, stereo_prior, num_heads=4, dropout=0.1):
        super(SubtaskFusion, self).__init__()
        self.num_heads = num_heads
        self.SE_l = SpatialEncoder(num_nodes, out_channels)  # 左侧空间编码器
        self.SE_r = SpatialEncoder(num_nodes, out_channels)  # 右侧空间编码器
        self.TE_l = TemporalEncoder(num_nodes, out_channels)  # 左侧时间编码器
        self.TE_r = TemporalEncoder(num_nodes, out_channels)  # 右侧时间编码器
        self.stereo_prior = stereo_prior  # 共享的立体先验模块

        self.fc = nn.Linear(out_channels, out_channels)  # 全连接层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

        self.fc1 = nn.Linear(out_channels*3, out_channels)  # 全连接层
        self.fc2 = nn.Linear(out_channels*3, out_channels)  # 全连接层
        self.fc3 = nn.Linear(out_channels*2, out_channels)  # 全连接层
        self.gelu = nn.GELU()  # GELU激活函数

    def forward(self, left_heatmap, right_heatmap, left_graph, right_graph):
        left_spatial = self.SE_l(left_heatmap)  #（batchsize，num_nodes，channels）
        right_spatial = self.SE_r(right_heatmap)  #（batchsize，num_nodes，channels）
        left_temporal = self.TE_l(left_graph)  #（batchsize，num_nodes，channels）
        right_temporal = self.TE_r(right_graph)  #（batchsize，num_nodes，channels）
        disparity1, disparity2  = self.stereo_prior(left_spatial, right_spatial, left_temporal, right_temporal)
        fused_spatial_features= torch.cat([left_spatial, right_spatial, disparity1], dim=-1)  # (B, 3N, C)
        fused_temporal_features = torch.cat([left_temporal, right_temporal, disparity2], dim=-1)  # (B, 3N, C)
        fusion_query = self.fc1(fused_spatial_features)
        fusion_key = self.fc2(fused_temporal_features)
        fusion_value = self.fc3(torch.cat([fusion_query, fusion_key], dim=-1))
        fusion_value = self.gelu(fusion_value)
        return fusion_value


class IntraTaskFusion(nn.Module):
    def __init__(self, num_subtasks, num_nodes, out_channels, stereo_prior, dropout=0.1):
        super(IntraTaskFusion, self).__init__()
        self.num_subtasks = num_subtasks
        self.stereo_prior= stereo_prior
        self.subtask_fusions = nn.ModuleList([
            SubtaskFusion(num_nodes, out_channels, self.stereo_prior, num_heads=4, dropout=0.1) for _ in range(num_subtasks)
        ])
        self.fc = nn.Linear(out_channels * self.num_subtasks, out_channels)
        self.att = NonLocalSparseAttention(n_hashes=4, channels=out_channels, k_size=3, reduction=4, chunk_size=10, res_scale=1)
        self.fc2 = nn.Linear(out_channels * num_nodes, out_channels)
        self.gelu = nn.GELU() 
    def forward(self, heatmap, graph):
        subfeatures = []
        for i in range(self.num_subtasks):
            left_heatmap = heatmap[i * 2]
            right_heatmap = heatmap[i * 2 + 1]
            left_graph = graph[i * 2]
            right_graph = graph[i * 2 + 1]
            subfeature = self.subtask_fusions[i](left_heatmap, right_heatmap, left_graph, right_graph)
            subfeatures.append(subfeature)
        x = torch.cat(subfeatures, dim=-1)
        x = self.fc(x)
        x = self.att(x)+x
        x = self.fc2(x.reshape(x.size(0), -1))
        x = self.gelu(x)
        return x


class InterTaskIntegration(nn.Module):
    def __init__(self, num_nodes, out_channels):
        super(InterTaskIntegration, self).__init__()
        self.stereo_prior = StereoPrior(num_nodes=num_nodes, channels=out_channels)
        self.saccade_feature_extractor = IntraTaskFusion(4, num_nodes, out_channels, self.stereo_prior)
        self.sensitivity_feature_extractor = IntraTaskFusion(3, num_nodes, out_channels, self.stereo_prior)
        self.significance_feature_extractor = IntraTaskFusion(5, num_nodes, out_channels, self.stereo_prior)
        self.fc = nn.Linear(out_channels * 3, out_channels)
        self.gelu = nn.GELU()
        self.contrastive_loss_fn = ContrastiveLoss()
    
    def forward(self, heatmaps, graphs):
        saccade_heatmap, saccade_graph = heatmaps[:8], graphs[:8]
        sensitivity_heatmap, sensitivity_graph = heatmaps[8:14], graphs[8:14]
        significance_heatmap, significance_graph = heatmaps[14:24], graphs[14:24]

        saccade_feature = self.saccade_feature_extractor(saccade_heatmap, saccade_graph)          # (batch_size, out_channels)
        sensitivity_feature = self.sensitivity_feature_extractor(sensitivity_heatmap, sensitivity_graph)  # (batch_size, out_channels)
        significance_feature = self.significance_feature_extractor(significance_heatmap, significance_graph)  # (batch_size, out_channels)

        x = torch.cat([saccade_feature, sensitivity_feature, significance_feature], dim=-1)  # (batch_size, out_channels * 3)
        x = self.gelu(self.fc(x))  # (batch_size, out_channels)

        contrastive_loss = None
        features = torch.cat([saccade_feature, sensitivity_feature, significance_feature], dim=0)  # (3 * batch_size, out_channels)
        task_labels = torch.cat([
            torch.zeros(saccade_feature.size(0)),
            torch.ones(sensitivity_feature.size(0)),
            torch.full((significance_feature.size(0),), 2)
        ]).long().cuda()

        contrastive_loss = self.contrastive_loss_fn(features, task_labels)
        
        return x, contrastive_loss

class HierarchicalTemporalSpatialModel(nn.Module):
    def __init__(self, num_nodes=50, hidden_dim=32):
        super(HierarchicalTemporalSpatialModel, self).__init__()
        self.num_nodes = num_nodes
        self.inter_task_integration = InterTaskIntegration(self.num_nodes, hidden_dim)
        self.edu_age_adjuster = nn.Linear(2, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim+1, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
            
    def forward(self, heatmaps, graphs, edu_age):
        inter_task_feature, contrastive_loss = self.inter_task_integration(heatmaps, graphs) 
        output = self.fc1(inter_task_feature)+inter_task_feature
        output = self.gelu(output)
        ea = self.edu_age_adjuster(edu_age)
        output = self.fc2(torch.cat([output, ea], dim=-1))
        
        return output.squeeze(-1), contrastive_loss