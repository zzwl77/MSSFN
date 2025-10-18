import torch
import torch.nn as nn
import torch.nn.functional as F

# 激活函数定义
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# 位置编码模块
class PositionEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super(PositionEncoding, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels

        # Create positional encodings
        y_embed = torch.arange(height).view(1, 1, height, 1).float()
        x_embed = torch.arange(width).view(1, 1, 1, width).float()

        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-torch.log(torch.tensor(10000.0)) / channels))
        pe = torch.zeros(1, channels, height, width)
        pe[:, 0::2, :, :] = torch.sin(x_embed * div_term).repeat(1, 1, height, 1)
        pe[:, 1::2, :, :] = torch.cos(y_embed * div_term).repeat(1, 1, 1, width)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

# 优化后的CoordAtt模块
class CoordAttOptimized(nn.Module):
    def __init__(self, channels, output_nodes, reduction=4, top_k=50):
        """
        Args:
            channels (int): 输入通道数
            output_nodes (int): 输出节点数（稀疏化后选择的像素点数）
            reduction (int): 通道数压缩比例
            top_k (int): 每个通道选择的Top-K像素点数
        """
        super(CoordAttOptimized, self).__init__()
        self.output_nodes = output_nodes
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.top_k = top_k

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入特征图，形状为 [batch, channels, height, width]
        Returns:
            Tensor: 输出特征，形状为 [batch, output_nodes, channels]
        """
        identity = x
        n, c, h, w = x.size()
        # Coordinate Attention
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, 1, w]

        y = torch.cat([x_h, x_w], dim=2)  # [n, c, h + w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [n, c, w, 1]

        a_h = self.conv_h(x_h).sigmoid()  # [n, c, h, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [n, c, 1, w]

        # Combine attention
        a_h = a_h.expand(-1, -1, -1, w)  # [n, c, h, w]
        a_w = a_w.expand(-1, -1, h, -1)  # [n, c, h, w]
        attention = a_h * a_w  # [n, c, h, w]

        # 综合每个空间位置的注意力，得到一个整体的空间重要性评分
        # 可以通过平均或求和，这里使用求和
        attention_sum = attention.sum(1)  # [n, h, w]

        # 扁平化空间维度
        attention_flat = attention_sum.view(n, -1)  # [n, h*w]

        # 选择Top-K重要的空间位置
        topk = min(self.output_nodes, attention_flat.size(1))  # 确保Top-K不超过总空间位置
        topk_vals, topk_idx = torch.topk(attention_flat, topk, dim=1)  # [n, topk]

        # 获取选中位置的特征
        selected_features = self._gather_features(identity, topk_idx, h, w)  # [n, topk, c]

        return selected_features  # [n, topk, c]

    def _gather_features(self, x, idx, h, w):
        """
        根据索引收集特征
        Args:
            x (Tensor): 输入特征图 [n, c, h, w]
            idx (Tensor): 索引 [n, topk]
            h (int): 高度
            w (int): 宽度
        Returns:
            Tensor: 选中的特征 [n, topk, c]
        """
        n, c, _, _ = x.size()
        # 将索引扩展到与通道数匹配
        idx = idx.unsqueeze(1).expand(n, c, self.output_nodes)  # [n, c, topk]
        # 将特征图展平
        x_flat = x.view(n, c, -1)  # [n, c, h*w]
        # 使用gather选择特定位置的特征
        selected = torch.gather(x_flat, 2, idx)  # [n, c, topk]
        # 转置为 [n, topk, c]
        return selected.permute(0, 2, 1)  # [n, topk, c]