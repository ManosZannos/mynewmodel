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

    NOTE: Original repo uses soft masking (keeps sigmoid values above threshold).
    This differs from paper Eq. 11-12 which describes binary masks.
    We follow the original repo implementation here.
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

        assert len(dense_temporal_interaction.shape) == 4   # (T, num_heads, N, N)
        assert len(dense_spatial_interaction.shape) == 4    # (N, num_heads, T, T)

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
    """
    Zero-preserving softmax (Paper Eq. 14).
    """

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
        # No squeeze — input (num_heads, T, N, N), all dims >= 2 in normal operation
        return self.conv(x) + self.shortcut(x)


class SparseWeightedAdjacency(nn.Module):
    """
    Generates normalized sparse spatial and temporal adjacency matrices.
    Paper Section 3.2.2: Eq. 2-15.

    Input graph shape: (T, N, 6) with features:
      [LON_abs, LAT_abs, LON_rel, LAT_rel, SOG_rel, Heading_rel]
    - spatial_graph uses graph[:, :, 1:] → (T, N, 5)
    - temporal_graph uses full graph     → (N, T, 6)
    """

    def __init__(self, spa_in_dims=5, tem_in_dims=6, embedding_dims=64, obs_len=10,
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
        """
        Args:
            graph: (T, N, 6) — [LON_abs, LAT_abs, LON_rel, LAT_rel, SOG_rel, Heading_rel]
            identity: [spatial_identity (T,N,N), temporal_identity (N,T,T)]
        """
        assert len(graph.shape) == 3

        # Spatial graph: skip LON_abs → (T, N, 5)
        spatial_graph = graph[:, :, 1:]

        # Temporal graph: full features → (N, T, 6)
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
    """
    Single graph convolution layer: H' = σ(A · H · W)
    Paper Section 3.2.3.
    """

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout

    def forward(self, graph, adjacency):
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout, training=self.training)
        return gcn_features


class HybridNetwork(nn.Module):
    """
    Hybrid Network for fusing spatial and temporal GCN features.
    Paper Section 3.2.3, Eq. 18-19.

    Learns dynamic weights w=[w1, w2] via a 2-layer MLP:
      w = softmax(W_hy * tanh(W_hid * [Hs ⊕ Ht] + b_hid) + b_hy)
      H = w1 * Hs + w2 * Ht

    Args:
        feature_dim: dimensionality of each feature vector (gcn_hidden * num_heads)
        hidden_dim:  hidden state dimension of MLP (paper sets to 64)
    """

    def __init__(self, feature_dim, hidden_dim=64):
        super(HybridNetwork, self).__init__()

        # 2-layer MLP: input = concat(Hs, Ht) → 2*feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),   # W_hid, b_hid
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),                  # W_hy, b_hy
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Hs, Ht):
        """
        Args:
            Hs: [N, T, num_heads, emb] — spatial-temporal GCN output (Eq. 16)
            Ht: [N, T, num_heads, emb] — temporal-spatial GCN output (Eq. 17)

        Returns:
            H: [N, T, num_heads, emb] — fused deep feature representation (Eq. 19)
        """
        # Concatenate along feature dim for MLP input: [N, T, num_heads, 2*emb]
        combined = torch.cat([Hs, Ht], dim=-1)

        # Compute weights: [N, T, num_heads, 2]
        w = self.softmax(self.mlp(combined))

        w1 = w[..., 0:1]  # [N, T, num_heads, 1]
        w2 = w[..., 1:2]  # [N, T, num_heads, 1]

        # Weighted fusion: H = w1*Hs + w2*Ht  (Eq. 19)
        H = w1 * Hs + w2 * Ht

        return H


class SparseGraphConvolution(nn.Module):
    """
    Two-path sparse GCN with Hybrid Network fusion.
    Paper Section 3.2.3, Eq. 16-19.

    FIX from original repo:
    - Uses layer2 outputs (gcn_spatial_temporal_features, gcn_temporal_spatial_features)
      as Hs and Ht — not layer1 outputs as in the original repo.
    - Replaces simple addition with HybridNetwork weighted fusion (Eq. 18-19).
    """

    def __init__(self, in_dims=5, embedding_dims=16, dropout=0, num_heads=4):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        # FIX: Hybrid Network for weighted fusion of Hs and Ht (paper Eq. 18-19)
        # feature_dim = embedding_dims (per head), hidden_dim = 64 (paper Section 4.1.2)
        self.hybrid_network = HybridNetwork(
            feature_dim=embedding_dims,
            hidden_dim=64
        )

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        """
        Args:
            graph: [1, T, N, 6] — full graph with absolute + relative features
            normalized_spatial_adjacency_matrix:  [T, num_heads, N, N]
            normalized_temporal_adjacency_matrix: [N, num_heads, T, T]

        Returns:
            H: [N, T, num_heads, emb] — deep feature representation of vessel trajectories
        """
        # Skip LON_abs (first feature), use remaining 5 features
        graph = graph[:, :, :, 1:]              # [1, T, N, 5]

        spa_graph = graph.permute(1, 0, 2, 3)   # [T, 1, N, 5]
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # [N, 1, T, 5]

        # ── Spatial path (paper Eq. 16) ──────────────────────────────────────
        # Layer 1: spatial adjacency on spatial graph
        gcn_spatial_layer1 = self.spatial_temporal_sparse_gcn[0](
            spa_graph, normalized_spatial_adjacency_matrix
        )  # [T, num_heads, N, emb]

        # Layer 2: temporal adjacency on spatially-encoded features
        gcn_spatial_layer1_perm = gcn_spatial_layer1.permute(2, 1, 0, 3)  # [N, num_heads, T, emb]
        Hs = self.spatial_temporal_sparse_gcn[1](
            gcn_spatial_layer1_perm, normalized_temporal_adjacency_matrix
        )  # [N, num_heads, T, emb]
        Hs = Hs.permute(0, 2, 1, 3)  # [N, T, num_heads, emb]

        # ── Temporal path (paper Eq. 17) ─────────────────────────────────────
        # Layer 1: temporal adjacency on temporal graph
        gcn_temporal_layer1 = self.temporal_spatial_sparse_gcn[0](
            tem_graph, normalized_temporal_adjacency_matrix
        )  # [N, num_heads, T, emb]

        # Layer 2: spatial adjacency on temporally-encoded features
        gcn_temporal_layer1_perm = gcn_temporal_layer1.permute(2, 1, 0, 3)  # [T, num_heads, N, emb]
        Ht = self.temporal_spatial_sparse_gcn[1](
            gcn_temporal_layer1_perm, normalized_spatial_adjacency_matrix
        )  # [T, num_heads, N, emb]
        Ht = Ht.permute(2, 0, 1, 3)  # [N, T, num_heads, emb]

        # ── Hybrid Network fusion (paper Eq. 18-19) ───────────────────────────
        # FIX: replaces simple addition of layer1 outputs from original repo
        H = self.hybrid_network(Hs, Ht)  # [N, T, num_heads, emb]

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
    Gated TCN encoder (Paper Eq. 20):
    H(l+1) = residual + sigmoid(Wf * H) ⊗ tanh(Wg * H)

    When fin != fout, a projection Conv2d maps residual to correct output shape.
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


class TrajectoryModel(nn.Module):
    """
    SMCHN: Spatial-Multi-Channel Heterogeneous Network
    Adapted from Wang et al. (Ocean Engineering 2023) for vessel trajectory prediction.

    Changes from v1:
    - out_dims: 5 → 2 (deterministic velocity prediction, no Gaussian parameters)
    - Input features: 5 → 6 (added absolute position LON_abs, LAT_abs)
    - spa_in_dims: 4 → 5, tem_in_dims: 5 → 6, GCN in_dims: 4 → 5
    - Loss: NLL → Huber position loss (DualSTMA-aligned)

    Fix vs original repo:
    - HybridNetwork (paper Eq. 18-19) replaces simple addition fusion
    - GCN layer2 outputs (Hs, Ht) correctly used as HybridNetwork inputs
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
            spa_in_dims=5,
            tem_in_dims=6,
            embedding_dims=embedding_dims,
            obs_len=obs_len,
            dropout=dropout,
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            num_heads=num_heads
        )

        # FIX: num_heads passed to SparseGraphConvolution for HybridNetwork
        self.stsgcn = SparseGraphConvolution(
            in_dims=5,
            embedding_dims=gcn_hidden,
            dropout=dropout,
            num_heads=num_heads
        )

        self.encoder = Encoder(fin=obs_len, fout=pred_len)

        self.output = nn.Linear(embedding_dims, out_dims)

    def forward(self, graph, identity):
        """
        Args:
            graph:    [1, obs_len, N, 6] — V_obs:
                      [LON_abs, LAT_abs, LON_rel, LAT_rel, SOG_rel, Heading_rel]
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
        )  # [N, T, num_heads, gcn_hidden]

        features = self.encoder(H)  # [N, pred_len, num_heads, gcn_hidden]

        b, l, _, _ = features.shape
        features = features.contiguous().view(b, self.pred_len, -1)  # [N, pred_len, embedding_dims]

        prediction = self.output(features)  # [N, pred_len, 2]

        return prediction.permute(1, 0, 2).contiguous()  # [pred_len, N, 2]