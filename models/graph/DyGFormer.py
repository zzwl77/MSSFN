import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.nn import MultiheadAttention

class TimeEncoder(nn.Module): 
    def __init__(self, time_dim): 
        super(TimeEncoder, self).__init__()
        self.time_embedding = nn.Linear(1, time_dim)

    def forward(self, timestamps):
        # timestamps: [B, N]
        return torch.sin(self.time_embedding(timestamps.unsqueeze(-1)))  # [B, N, time_dim]

class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        return self.projection(features)
    
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=2, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.attention = MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_input, K_input, V_input):
        # Q_input, K_input, V_input: [B, E, in_channels]
        # index_target: [B, E]
        # num_nodes: int
        # Perform multi-head attention
        attn_output, _ = self.attention(Q_input, K_input, V_input)  # [B, E, in_channels]

        # Residual connection and layer normalization
        out1 = self.norm1(Q_input + attn_output)  # [B, E, in_channels]

        # Feed-Forward Network
        ffn_output = self.ffn(out1)  # [B, E, in_channels]

        # Second residual connection and layer normalization
        out2 = self.norm2(out1 + ffn_output)  # [B, E, in_channels]

        return out2  # [B, N, in_channels]
    

class EdgeConv(nn.Module):
    def __init__(self, node_dim, edge_dim, time_dim, out_channels, num_heads=2, dropout=0.1):
        super(EdgeConv, self).__init__()
        self.node_feature_encoder = FeatureProjection(node_dim * 2, out_channels)
        self.edge_feature_encoder = FeatureProjection(edge_dim, out_channels)
        self.time_encoder = TimeEncoder(time_dim)
        self.time_feature_encoder = FeatureProjection(time_dim * 2, out_channels)
        self.transformer = TransformerBlock(
            in_channels=out_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        # self.out_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, timestamps):
        # x: [B, N, node_in_channels]
        # edge_index: [B, 2, E]
        # edge_attr: [B, E, edge_in_channels]
        # timestamps: [B, N]

        batch_size, num_nodes, _ = x.size()

        # Add self-loops
        edge_index, edge_attr = self.add_self_loops_batch(edge_index, edge_attr, num_nodes)

        # Get source and target indices
        edge_index_source = edge_index[:, 0, :]  # [B, E]·
        edge_index_target = edge_index[:, 1, :]  # [B, E]

        # Gather node features and timestamps
        x_i = self.index_select_batch(x, edge_index_source)  # [B, E, node_in_channels]
        x_j = self.index_select_batch(x, edge_index_target)  # [B, E, node_in_channels]
        t_i = self.index_select_batch(timestamps, edge_index_source)  # [B, E]
        t_j = self.index_select_batch(timestamps, edge_index_target)  # [B, E]

        # Encode timestamps
        t_i_enc = self.time_encoder(t_i)  # [B, E, time_dim]
        t_j_enc = self.time_encoder(t_j)  # [B, E, time_dim]

        node_feature = self.node_feature_encoder(torch.cat([x_i, x_j], dim=-1))
        time_feature = self.time_feature_encoder(torch.cat([t_i_enc, t_j_enc], dim=-1))

        # Encode edge attributes
        edge_feature= self.edge_feature_encoder(edge_attr)  # [B, E, out_channels]

        # Transformer block
        out = self.transformer(node_feature, time_feature, edge_feature)  # [B, N, in_channels]

        return out  # [B, N, out_channels]

    def add_self_loops_batch(self, edge_index, edge_attr, num_nodes):
        batch_size = edge_index.size(0)
        device = edge_index.device

        # Create self-loop indices
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1)  # [B, N]
        self_loops = self_loops.unsqueeze(1).repeat(1, 2, 1)  # [B, 2, N]

        # Concatenate self-loops to edge_index
        edge_index = torch.cat([edge_index, self_loops], dim=2)  # [B, 2, E + N]

        # Create self-loop edge attributes (zeros)
        edge_attr_self_loops = torch.zeros(batch_size, num_nodes, edge_attr.size(-1), device=device)  # [B, N, edge_in_channels]

        # Concatenate self-loops to edge_attr
        edge_attr = torch.cat([edge_attr, edge_attr_self_loops], dim=1)  # [B, E + N, edge_in_channels]

        return edge_index, edge_attr

    def index_select_batch(self, x, indices):
        # x: [B, N, ...] or [B, N]
        # indices: [B, E]
        batch_size = x.size(0)
        E = indices.size(1)
        if x.dim() == 3:
            N, feat_dim = x.size(1), x.size(2)
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1).expand(batch_size, E)
            x_selected = x[batch_indices, indices]  # [B, E, feat_dim]
        elif x.dim() == 2:
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1).expand(batch_size, E)
            x_selected = x[batch_indices, indices]  # [B, E]
        else:
            raise ValueError("Input x must be 2D or 3D tensor")
        return x_selected

class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, out_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=out_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, 
                 out_tokens: int, out_channels: int,
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.1):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, out_dim = out_tokens,
                                                dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels, out_dim = out_channels,
                                                  dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = EdgeConv(node_dim=2, edge_dim=2, time_dim=8, out_channels=32, num_heads=2, dropout=0.1)
        self.classifier = nn.Linear(32, 3)

    def forward(self, x, edge_index, edge_attr, timestamps):
        x = self.conv1(x, edge_index, edge_attr, timestamps)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.mean(x, dim=1)  # Global average pooling over nodes
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

# # Initialize the model
# model = Net()

# # Sample data
# batch_size = 4          # Batch size
# num_nodes = 50          # Number of nodes per graph
# node_in_channels = 2    # Number of node features
# num_edges = 890         # Number of edges per graph
# edge_in_channels = 2    # Number of edge features

# # Generate sample batch data
# x = torch.randn(batch_size, num_nodes, node_in_channels)  # Node features
# edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges), dtype=torch.long)  # Edge indices
# edge_attr = torch.randn(batch_size, num_edges, edge_in_channels)  # Edge features
# timestamps = torch.rand(batch_size, num_nodes)  # Timestamps for nodes

# # Forward pass
# output = model(x, edge_index, edge_attr, timestamps)
# print(output.shape)  # Expected shape: [batch_size, num_classes]
