import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x
        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)
        return x2 + shortcut


class InteractionMask(nn.Module):
    """
    Generates sparse interaction masks via asymmetric convolutions.
    Paper Section 3.2.2: threshold ξ=0.5, values < threshold set to 0.
    """

    def __init__(self, number_asymmetric_conv_layer=2, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4
        assert len(dense_spatial_interaction.shape) == 4

        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask)
        temporal_zero = torch.zeros_like(temporal_interaction_mask)

        spatial_interaction_mask = torch.where(
            spatial_interaction_mask > threshold,
            spatial_interaction_mask,
            spatial_zero
        )
        temporal_interaction_mask = torch.where(
            temporal_interaction_mask > threshold,
            temporal_interaction_mask,
            temporal_zero
        )

        return spatial_interaction_mask, temporal_interaction_mask


class ZeroSoftmax(nn.Module):
    """Zero-preserving softmax (Paper Eq. 14)."""

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SelfAttention(nn.Module):
    """
    Multi-head self-attention for computing dense interaction scores.
    Paper Section 3.2.2: Eq. 2-5 (spatial), Eq. 6-8 (temporal).
    """

    def __init__(self, in_dims=4, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.register_buffer('scaled_factor', torch.sqrt(torch.tensor(float(d_model))))
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads
        self.d_model = d_model

    def split_heads(self, x):
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=False, multi_head=False):
        assert len(x.shape) == 3

        embeddings = self.embedding(x)
        query = self.query(embeddings)
        key = self.key(embeddings)

        if multi_head:
            query = self.split_heads(query)
            key = self.split_heads(key)
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))

        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:
            mask_mat = torch.ones_like(attention)
            attention = attention * torch.tril(mask_mat)

        return attention, embeddings


class SpatialTemporalFusion(nn.Module):
    """
    Fuses spatial attention scores across time dimension.
    Paper Section 3.2.2.
    """

    def __init__(self, obs_len=10):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class SparseWeightedAdjacency(nn.Module):
    """
    Generates normalized sparse spatial and temporal adjacency matrices.
    4 features [LON_rel, LAT_rel, SOG_rel, Heading_rel] — unchanged from baseline.
    """

    def __init__(self, spa_in_dims=4, tem_in_dims=4, embedding_dims=64, obs_len=10,
                 dropout=0, number_asymmetric_conv_layer=2, num_heads=4):
        super(SparseWeightedAdjacency, self).__init__()

        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims, num_heads=num_heads)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims, num_heads=num_heads)

        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)

        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            spatial_channels=num_heads,
            temporal_channels=num_heads
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()

    def forward(self, graph, identity):
        assert len(graph.shape) == 3

        spatial_graph = graph
        temporal_graph = graph.permute(1, 0, 2)

        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(
            spatial_graph, multi_head=True
        )
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(
            temporal_graph, multi_head=True
        )

        st_interaction = self.spa_fusion(
            dense_spatial_interaction.permute(1, 0, 2, 3)
        ).permute(1, 0, 2, 3)

        ts_interaction = dense_temporal_interaction

        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)

        spatial_mask = spatial_mask + identity[0].unsqueeze(1)
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)

        normalized_spatial_adjacency_matrix = self.zero_softmax(
            dense_spatial_interaction * spatial_mask, dim=-1
        )
        normalized_temporal_adjacency_matrix = self.zero_softmax(
            dense_temporal_interaction * temporal_mask, dim=-1
        )

        return (normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix,
                spatial_embeddings, temporal_embeddings)


class GraphConvolution(nn.Module):
    """Single graph convolution layer: H' = σ(A · H · W)"""

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout

    def forward(self, graph, adjacency):
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout, training=self.training)
        return gcn_features


class SparseGraphConvolution(nn.Module):
    """
    Two-path sparse GCN with layer 1 addition fusion — unchanged from baseline.
    """

    def __init__(self, in_dims=4, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):

        spa_graph = graph.permute(1, 0, 2, 3)      # [T, 1, N, 4]
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # [N, 1, T, 4]

        gcn_spatial_layer1 = self.spatial_temporal_sparse_gcn[0](
            spa_graph, normalized_spatial_adjacency_matrix
        )  # [T, num_heads, N, emb]

        gcn_spatial_layer1_perm = gcn_spatial_layer1.permute(2, 1, 0, 3)

        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn[1](
            gcn_spatial_layer1_perm, normalized_temporal_adjacency_matrix
        )  # not used in fusion

        gcn_temporal_layer1 = self.temporal_spatial_sparse_gcn[0](
            tem_graph, normalized_temporal_adjacency_matrix
        )  # [N, num_heads, T, emb]

        gcn_temporal_layer1_perm = gcn_temporal_layer1.permute(2, 1, 0, 3)

        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn[1](
            gcn_temporal_layer1_perm, normalized_spatial_adjacency_matrix
        )  # not used in fusion

        x = gcn_spatial_layer1 + gcn_temporal_layer1_perm  # [T, num_heads, N, emb]
        H = x.permute(2, 0, 1, 3)                          # [N, T, num_heads, emb]

        return H


class TCN(nn.Module):
    def __init__(self, fin, fout, layers=3, ksize=3):
        super(TCN, self).__init__()
        self.fin = fin
        self.fout = fout
        self.layers = layers
        self.ksize = ksize

        self.convs = nn.ModuleList()
        for i in range(self.layers):
            in_ch = fin if i == 0 else fout
            self.convs.append(nn.Conv2d(in_ch, fout, kernel_size=self.ksize))

    def forward(self, x):
        for conv in self.convs:
            x = F.pad(x, (self.ksize - 1, 0, self.ksize - 1, 0))
            x = conv(x)
        return x


class Encoder(nn.Module):
    """
    Gated TCN encoder (Paper Eq. 20) — unchanged from baseline.
    H(l+1) = residual + sigmoid(Wf * H) ⊗ tanh(Wg * H)
    """

    def __init__(self, fin, fout, layers=3, ksize=3):
        super(Encoder, self).__init__()
        self.tcnf = TCN(fin, fout, layers, ksize)
        self.tcng = TCN(fin, fout, layers, ksize)

        if fin != fout:
            self.residual_proj = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        f = torch.sigmoid(self.tcnf(x))
        g = torch.tanh(self.tcng(x))
        return residual + f * g


class TemporalSelfAttention(nn.Module):
    """
    Lightweight self-attention over the prediction horizon (pred_len steps).

    V9 change: applied AFTER the TCN Encoder and BEFORE the Linear output.
    The TCN acts as a feature extractor (embedding), and self-attention
    captures dependencies across pred_len steps — each predicted timestep
    can attend to all others.

    Motivated by:
    - EmbTCN-Transformer: TCN as input embedding + Transformer backbone
    - DualSTMA ablation: temporal attention > spatial for long-term prediction
    - GATransformer ablation: Temporal Encoder contributes 5-45% improvement
    - SGAformer: Skip-PAM attention improves mid-to-long term prediction

    Architecture:
      Input:  [N, pred_len, embedding_dims]
      Q, K, V projections over pred_len dimension
      Output: [N, pred_len, embedding_dims]  (same shape, residual connection)

    Uses causal masking (upper-triangular) so step t only attends to steps ≤ t.
    This is physically motivated: prediction at step t should not depend on
    future predicted steps t+1, t+2, ...

    num_heads=4, d_model=embedding_dims=64 — consistent with rest of model.
    """

    def __init__(self, embedding_dims=64, num_heads=4, dropout=0.0):
        super(TemporalSelfAttention, self).__init__()

        assert embedding_dims % num_heads == 0
        self.num_heads = num_heads
        self.d_model = embedding_dims
        self.head_dim = embedding_dims // num_heads

        self.q_proj = nn.Linear(embedding_dims, embedding_dims)
        self.k_proj = nn.Linear(embedding_dims, embedding_dims)
        self.v_proj = nn.Linear(embedding_dims, embedding_dims)
        self.out_proj = nn.Linear(embedding_dims, embedding_dims)

        self.register_buffer('scale', torch.sqrt(torch.tensor(float(self.head_dim))))

        self.norm = nn.LayerNorm(embedding_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [N, pred_len, embedding_dims]
        Returns:
            [N, pred_len, embedding_dims]
        """
        N, L, D = x.shape  # L = pred_len = 5

        # Multi-head projections
        Q = self.q_proj(x).reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # All: [N, num_heads, L, head_dim]

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [N, num_heads, L, L]

        # Causal mask: step t attends only to steps 0..t
        causal_mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)                             # [N, num_heads, L, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(N, L, D)         # [N, L, D]
        out = self.out_proj(out)                                # [N, L, D]

        # Residual + LayerNorm
        return self.norm(x + out)                               # [N, pred_len, embedding_dims]


class TrajectoryModel(nn.Module):
    """
    SMCHN V9: TCN Encoder + Temporal Self-Attention before Linear output.
    Baseline: True Baseline (4 features, layer 1 GCN fusion, Gated TCN encoder)

    V9 change: TemporalSelfAttention module inserted between Encoder and output.

    Pipeline:
      GCN → Gated TCN Encoder → TemporalSelfAttention → Linear → predictions

    The TCN acts as a local feature extractor (EmbTCN-Transformer role),
    and the self-attention captures global temporal dependencies across the
    pred_len prediction steps, allowing each step to condition on all others.

    Motivated by 4 papers:
    - EmbTCN-Transformer: TCN as embedding + Transformer = best combination
    - DualSTMA ablation (M3 vs M2): temporal-first attention > spatial-first
    - GATransformer: Temporal Encoder contributes 5-45% across horizons
    - SGAformer: attention improves mid-to-long term prediction most

    Causal masking ensures step t only attends to steps ≤ t (physically sound).
    Residual + LayerNorm for training stability.

    All other components unchanged from true baseline.
    """

    def __init__(self,
                 number_asymmetric_conv_layer=2, embedding_dims=64, number_gcn_layers=1,
                 dropout=0, obs_len=10, pred_len=5, out_dims=2, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dropout = dropout
        self.num_heads = num_heads
        self.embedding_dims = embedding_dims

        gcn_hidden = embedding_dims // num_heads  # 16

        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            spa_in_dims=4,
            tem_in_dims=4,
            embedding_dims=embedding_dims,
            obs_len=obs_len,
            dropout=dropout,
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            num_heads=num_heads
        )

        self.stsgcn = SparseGraphConvolution(
            in_dims=4,
            embedding_dims=gcn_hidden,
            dropout=dropout
        )

        # Unchanged from baseline
        self.encoder = Encoder(fin=obs_len, fout=pred_len)

        # V9: Temporal Self-Attention after encoder
        self.temporal_attention = TemporalSelfAttention(
            embedding_dims=embedding_dims,
            num_heads=num_heads,
            dropout=0.0
        )

        self.output = nn.Linear(embedding_dims, out_dims)

    def forward(self, graph, identity):
        """
        Args:
            graph:    [1, obs_len, N, 4]
            identity: [spatial (T,N,N), temporal (N,T,T)]

        Returns:
            [pred_len, N, 2] — predicted (LON_vel, LAT_vel)
        """
        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, \
            spatial_embeddings, temporal_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(0), identity)

        H = self.stsgcn(
            graph,
            normalized_spatial_adjacency_matrix,
            normalized_temporal_adjacency_matrix
        )

        features = self.encoder(H)       # [N, pred_len, num_heads, gcn_hidden]

        N = features.shape[0]
        features = features.contiguous().view(N, self.pred_len, -1)
        # [N, pred_len, embedding_dims]

        # V9: Temporal Self-Attention over pred_len steps
        features = self.temporal_attention(features)
        # [N, pred_len, embedding_dims]

        prediction = self.output(features)              # [N, pred_len, 2]

        return prediction.permute(1, 0, 2).contiguous() # [pred_len, N, 2]