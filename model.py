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

    def __init__(self, number_asymmetric_conv_layer=2, num_heads=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(num_heads, num_heads)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(num_heads, num_heads)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4       # (N, num_heads, T, T)
        assert len(dense_spatial_interaction.shape) == 4        # (T, num_heads, N, N)

        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        # Paper-faithful: binary masks (0/1) after thresholding
        spatial_interaction_mask = (spatial_interaction_mask >= threshold).float()
        temporal_interaction_mask = (temporal_interaction_mask >= threshold).float()

        return spatial_interaction_mask, temporal_interaction_mask


class ZeroSoftmax(nn.Module):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SelfAttention(nn.Module):

    def __init__(self, in_dims=4, d_model=64, num_heads=4):    
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        # Register as buffer so it moves with model.to(device)
        self.register_buffer("scaled_factor", torch.sqrt(torch.tensor(float(d_model))))
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads
        self.d_model = d_model

    def positional_encoding(self, seq_len, d_model, device):
        """
        Generate positional encoding as described in the paper (Section 3.2.2, Eq. 6)
        This is added to temporal graph embeddings
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                             -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

    def split_heads(self, x):

      
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()   #contiguous：batch_size, seq_len, h, embeeding/h----------seq_len
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]   #permute, embeeding/nun_heads

    def forward(self, x, mask=False, multi_head=False, add_positional_encoding=False):
        """
        Forward pass with optional positional encoding for temporal graphs
        
        Args:
            x: input tensor [batch_size, seq_len, feature_dim]
            mask: whether to apply causal mask
            multi_head: whether to use multi-head attention
            add_positional_encoding: whether to add positional encoding (for temporal graphs, Eq. 6)
        """
        # batch_size seq_len 2      spatial_graph[8,57,2]

        assert len(x.shape) == 3

        embeddings = self.embedding(x)  # batch_size seq_len d_model
        
        # Add positional encoding for temporal graph (Eq. 6 in paper)
        if add_positional_encoding:
            seq_len = x.shape[1]
            pos_enc = self.positional_encoding(seq_len, self.d_model, device=x.device)
            embeddings = embeddings + pos_enc.unsqueeze(0)  # broadcast across batch
        
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model    [batch_size nun_heads seq_len depth] , depth=embeeding/num_head
            key = self.split_heads(key)  # B num_heads seq_len d_model
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)  ---[batch_size, num_heads,]
        else:                                                         #---[batch_size, num_heads,depth,seq_len] ---k
            scores = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)   ：[batch_size seq_len, seq_len]
                                                                     #[batch_size seq_len, embedding][batch_size， seq_len embedding]
        
        # Scale scores
        scores = scores / self.scaled_factor
        
        # Apply causal mask BEFORE softmax (paper-faithful: current independent of future)
        if mask is True:
            causal_mask = torch.tril(torch.ones_like(scores, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        
        attention = self.softmax(scores)     # [batch_size，num_heads, seq_len, seq_len],：[batch_size,seq_len,seq_len]

        return attention, embeddings


class SpatialTemporalFusion(nn.Module):

    def __init__(self, obs_len=10):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):

        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()


class SparseWeightedAdjacency(nn.Module):   #得到邻接矩阵

    def __init__(self, spa_in_dims=4, tem_in_dims=4, embedding_dims=64, obs_len=10, dropout=0,
                 number_asymmetric_conv_layer=2, num_heads=4):
        super(SparseWeightedAdjacency, self).__init__()
        # dense interaction
        # Paper-faithful: Both use 4D state vector (LON, LAT, SOG, Heading)
        # Positional encoding is added via sinusoidal PE in temporal attention (Eq. 6)
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims, num_heads=num_heads)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims, num_heads=num_heads)

        # [batch_size，num_heads, seq_len, seq_len],
        # [batch_size,seq_len,seq_len]
        # batch_size seq_len d_model

        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len) 

        # interaction mask
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            num_heads=num_heads
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()

    def forward(self, graph, identity):
        """
        Args:
            graph: Input shape (T, N, 4) with 4D state vector [LON, LAT, SOG, Heading]
                   Paper-faithful: no pos_enc feature, sinusoidal PE added in attention
            identity: [spatial_identity, temporal_identity] matrices
        """
        assert len(graph.shape) == 3
        
        # graph shape: (T, N, 4) where T=obs_len, N=num_vessels
        # Features: [LON, LAT, SOG, Heading] (paper's 4D state vector)
        
        # Spatial graph: Use all 4D features
        spatial_graph = graph  # (T, N, 4) - 4D state vector as per paper
        
        # Temporal graph: Use all 4D features, permute to (N, T, 4)
        temporal_graph = graph.permute(1, 0, 2)  # (N, T, 4)
        # (T num_heads N N)   (T N d_model)  -------（[batch_size，num_heads, seq_len, seq_len]）（ # ：batch_size seq_len d_model）,---T是batch_size
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True) #-----[batch_size，num_heads, seq_len, seq_len]

        # (N num_heads T T)   (N T d_model)
        # Add positional encoding for temporal graph as per Eq. 6 in paper
        # Apply causal mask to ensure current state is independent of future state (paper requirement)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, multi_head=True, add_positional_encoding=True, mask=True)

        # attention fusion   dense_spatial_interaction.permute(1, 0, 2, 3))=（num_heads,T,N,N）
        st_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)     #(T num_heads N N) ）
        ts_interaction = dense_temporal_interaction

        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)  #st_interaction
          #经过interaction_mask，spatial_mask:(T num_heads N N)  ;temporal_mask:(N num_heads T T)
        # self-connected
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)     #[8,N,N] ，identity
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)  # [N,8,8]

        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)   #dense_spatial_interaction
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix,\
               spatial_embeddings, temporal_embeddings


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer for SMCHN model
    
    Applies graph convolution: H' = σ(A · H · W) where:
    - A: adjacency matrix (captures interactions)
    - H: node features (vessel state vectors)
    - W: learnable weight matrix
    - σ: activation function (PReLU)
    """

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency):
        """
        Args:
            graph: Node features [batch_size, 1, seq_len, in_dims]
                   in_dims can be 4 (initial: LON,LAT,SOG,Heading) or embedding_dims (subsequent layers)
            adjacency: [batch_size, num_heads, seq_len, seq_len]
        Returns:
            gcn_features: [batch_size, num_heads, seq_len, embedding_dims]
        """
        # Graph convolution: A · H · W
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout, training=self.training)

        return gcn_features


class HybridNetwork(nn.Module):
    """
    Hybrid network for fusing spatial-temporal and temporal-spatial features
    as described in Eq. 18 of the paper.
    
    This MLP dynamically learns weights w = [w1, w2] to fuse features:
    w = softmax(Why*tanh(Whid*[Hs ⊕ Ht] + bhid) + bhy)
    H = w1*Hs + w2*Ht
    """
    
    def __init__(self, feature_dim=16, hidden_dim=64):
        super(HybridNetwork, self).__init__()
        
        # Two-layer MLP as described in paper (Section 3.2.3)
        # Input: concatenated features [Hs ⊕ Ht]
        # Output: 2D weight vector [w1, w2]
        self.fc_hidden = nn.Linear(feature_dim * 2, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, 2)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, spatial_temporal_features, temporal_spatial_features):
        """
        Args:
            spatial_temporal_features (Hs): features from spatial→temporal GCN path
            temporal_spatial_features (Ht): features from temporal→spatial GCN path
            
        Returns:
            fused_features (H): weighted combination H = w1*Hs + w2*Ht
        """
        # Concatenate features along last dimension [Hs ⊕ Ht]
        concatenated = torch.cat([spatial_temporal_features, temporal_spatial_features], dim=-1)
        
        # Two-layer MLP (Eq. 18)
        hidden = self.tanh(self.fc_hidden(concatenated))
        logits = self.fc_output(hidden)
        
        # Softmax to get weights w = [w1, w2]
        weights = self.softmax(logits)  # Shape: [..., 2]
        
        # Extract w1 and w2
        w1 = weights[..., 0:1]  # Keep dimension for broadcasting
        w2 = weights[..., 1:2]
        
        # Weighted fusion: H = w1*Hs + w2*Ht (Eq. 19)
        fused_features = w1 * spatial_temporal_features + w2 * temporal_spatial_features
        
        return fused_features


class SparseGraphConvolution(nn.Module):

    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))  #append()：在 ModuleList 
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        # Hybrid network for feature fusion (Eq. 18-19 in paper)
        self.hybrid_network = HybridNetwork(feature_dim=embedding_dims, hidden_dim=64)


    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        """
        Two-layer GCN with hybrid network fusion as described in paper Section 3.2.3
        
        Spatial path (Eq. 16): Â'st → GCN1 → Â'ts → GCN2 → Hs
        Temporal path (Eq. 17): Â'ts → GCN1 → Â'st → GCN2 → Ht
        Fusion (Eq. 18-19): H = w1*Hs + w2*Ht via hybrid network
        
        Args:
            graph: shape [batch_size, seq_len, num_vessels, 4]
                   Features: [LON, LAT, SOG, Heading] (paper's 4D state vector)
            normalized_spatial_adjacency_matrix: shape [batch, num_heads, seq_len, seq_len]
            normalized_temporal_adjacency_matrix: shape [batch, num_heads, seq_len, seq_len]
        """
        # graph is already 4D state vector [LON, LAT, SOG, Heading]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)

        # ===== SPATIAL PATH: Â'st → GCN1 → Â'ts → GCN2 → Hs (Eq. 16) =====
        # First layer: apply spatial adjacency
        gcn_spatial_layer1 = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        # Permute to prepare for temporal adjacency
        gcn_spatial_layer1_perm = gcn_spatial_layer1.permute(2, 1, 0, 3)  # (num_p, num_heads, seq_len, hidden_dim)
        # Second layer: apply temporal adjacency
        Hs = self.spatial_temporal_sparse_gcn[1](gcn_spatial_layer1_perm, normalized_temporal_adjacency_matrix)
        # Hs shape: (num_p, num_heads, seq_len, embedding_dims)

        # ===== TEMPORAL PATH: Â'ts → GCN1 → Â'st → GCN2 → Ht (Eq. 17) =====
        # First layer: apply temporal adjacency
        gcn_temporal_layer1 = self.temporal_spatial_sparse_gcn[0](tem_graph, normalized_temporal_adjacency_matrix)
        # Permute to prepare for spatial adjacency
        gcn_temporal_layer1_perm = gcn_temporal_layer1.permute(2, 1, 0, 3)  # (seq_len, num_heads, num_p, hidden_dim)
        # Second layer: apply spatial adjacency
        Ht = self.temporal_spatial_sparse_gcn[1](gcn_temporal_layer1_perm, normalized_spatial_adjacency_matrix)
        # Ht shape: (seq_len, num_heads, num_p, embedding_dims)

        # Align dimensions for hybrid network: both should be (num_p, seq_len, num_heads, embedding_dims)
        # or similar compatible shape
        Hs_aligned = Hs.permute(0, 2, 1, 3)  # (num_p, seq_len, num_heads, embedding_dims)
        Ht_aligned = Ht.permute(2, 0, 1, 3)  # (num_p, seq_len, num_heads, embedding_dims)

        # ===== HYBRID NETWORK FUSION: H = w1*Hs + w2*Ht (Eq. 18-19) =====
        H = self.hybrid_network(Hs_aligned, Ht_aligned)
        
        # Return in expected format: (num_p, seq_len, num_heads, embedding_dims)
        return H

class TCN(nn.Module):
    def __init__(self,fin,fout,layers=3,ksize=3):
        super(TCN, self).__init__()
        self.fin = fin
        self.fout = fout
        self.layers = layers
        self.ksize = ksize

        self.convs = nn.ModuleList()
        for i in range(self.layers):
            self.convs.append(nn.Conv2d(self.fin,self.fout,kernel_size=self.ksize))

    def forward(self,x):
        for _,conv_layer in enumerate(self.convs):
            x = nn.functional.pad(x,(self.ksize-1,0,self.ksize-1,0))
            x = conv_layer(x)
        return x



class Encoder(nn.Module):
    """
    TCN with gating mechanism as described in Eq. 20 of the paper:
    H(l+1) = g(Wg * H(l)) ⊗ σ(Wf * H(l))
    
    where g is tanh activation and σ is sigmoid activation
    """
    def __init__(self,fin,fout,layers=3,ksize=3):
        super(Encoder, self).__init__()
        self.fin = fin
        self.fout = fout
        self.layers = layers
        self.ksize = ksize

        self.tcnf = TCN(self.fin, self.fout, self.layers, self.ksize)  # σ(Wf * H)
        self.tcng = TCN(self.fin, self.fout, self.layers, self.ksize)  # g(Wg * H)

    def forward(self,x):
        """
        Gated TCN: H(l+1) = tanh(Wg * H) ⊗ sigmoid(Wf * H)
        Note: Paper formula (Eq. 20) does NOT include residual connection
        """
        f = torch.sigmoid(self.tcnf(x))  # σ(Wf * H(l))
        g = torch.tanh(self.tcng(x))      # g(Wg * H(l))
        
        # Element-wise multiplication (⊗) as per Eq. 20
        return f * g  # Removed residual connection to match paper


class TrajectoryModel(nn.Module):


    def __init__(self,
                 number_asymmetric_conv_layer=2, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            spa_in_dims=4,
            tem_in_dims=4,
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            obs_len=obs_len,
            num_heads=num_heads
        )

        # graph convolution
        # FIX: use full embedding_dims (64), not embedding_dims // num_heads (16).
        self.stsgcn = SparseGraphConvolution(
            in_dims=4, embedding_dims=embedding_dims, dropout=dropout
        )

        self.fusion_ = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        # self.tcns = nn.ModuleList()
        # self.tcns.append(nn.Sequential(
        #     nn.Conv2d(obs_len, pred_len, 3, padding=1),
        #     nn.PReLU()
        # ))
        #
        # for j in range(1, self.n_tcn):  #1,2,3,4
        #     self.tcns.append(nn.Sequential(
        #         nn.Conv2d(pred_len, pred_len, 3, padding=1),
        #         nn.PReLU()
        # ))
        self.encoder = Encoder(fin=self.obs_len,fout=pred_len)

        # self.output = nn.Linear(embedding_dims // num_heads, out_dims)
        self.output = nn.Linear(embedding_dims , out_dims)
        # self.tcnf = nn.Conv2d(obs_len, pred_len, 3, padding=1)
        # self.tcng = nn.Conv2d(obs_len, pred_len, 3, padding=1)


    def forward(self, graph, identity):
        """
        Args:
            graph: V_obs with shape [batch_size, obs_len, N, 4]
                   Features: [LON, LAT, SOG, Heading] (paper's 4D state vector)
            identity: [identity_spatial (obs_len, N, N), identity_temporal (N, obs_len, obs_len)]
        """
        # Expected input: V_obs shape [batch_size, obs_len, N, 4]
        # With 4 features: [LON, LAT, SOG, Heading] (paper-faithful)

        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(0), identity)   

        H = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )   
#batch_size,hidden_size
        # gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features
        # gcn_representation = self.fusion_(gcn_temporal_spatial_features) + self.fusion_(gcn_spatial_temporal_features)
        # gcn_representation = gcn_temporal_spatial_features + gcn_spatial_temporal_features  #[N,4,obs,16]

        gcn_representation = H  # (N, seq_len, num_heads, embedding_dims)

        features = self.encoder(gcn_representation)  # (N, pred_len, num_heads, embedding_dims)

        # Mean over num_heads dim → (N, pred_len, embedding_dims)
        # This matches the output Linear layer which expects embedding_dims (64)
        features = features.mean(dim=2)  # (N, pred_len, embedding_dims)

        prediction = self.output(features)  # (N, pred_len, 5)

        # f = torch.sigmoid(self.tcnf(gcn_representation))
        # g = torch.tanh(self.tcng(gcn_representation))
        # features = gcn_representation + f * g


        # prediction = torch.mean(self.output(features), dim=-2)
        #
        # f = torch.sigmoid(self.tcnf(prediction))
        # g = torch.tanh(self.tcng(prediction))
        # prediction = prediction + f * g

        #
        # for k in range(1, self.n_tcn):  #1,2,3,4
        #     features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        # prediction = torch.mean(self.output(features), dim=-2)   #dim=-2
        return prediction.permute(1, 0, 2).contiguous()