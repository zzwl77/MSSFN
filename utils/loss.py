import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 ContrastiveLoss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().cuda()
        logits = similarity_matrix / self.temperature
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).cuda()
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss

class ConfidenceEnhancementLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ConfidenceEnhancementLoss, self).__init__()
        self.margin = margin

    def forward(self, logits, threshold):
        """
        计算远离阈值的损失，以增强模型输出的置信度。
        参数:
        - logits: 模型的预测输出。
        - threshold: 动态计算得到的阈值。

        返回:
        - loss: 置信度增强损失的均值。
        """
        # 定义正类和负类的边界
        margin_pos = threshold + self.margin
        margin_neg = threshold - self.margin

        # 计算正类和负类的间距损失
        pos_loss = F.relu(margin_pos - logits)  # 正类应大于阈值加边缘
        neg_loss = F.relu(logits - margin_neg)  # 负类应小于阈值减边缘

        # 返回损失的平均值
        return (pos_loss + neg_loss).mean()
# 假设 StereoPrior、IntraTaskFusion 等类已经定义
# 以下只展示 InterTaskIntegration 类的修改


