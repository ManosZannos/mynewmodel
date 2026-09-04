"""
model_cpagrn_uniontopk.py — CPA-GRN, Union Top-K variant (Πρόταση 1)

Motivation: the locked v4 architecture sparsifies neighbors purely by
Euclidean distance (top_k=10, or top_k=15 in the confirmed obs10 ablation).
A vessel that is currently far away but has a very low DCPA and a positive
TCPA (i.e. rapidly closing toward a collision-relevant encounter) can fall
outside this distance-based top_k, even though it may be the single most
decision-relevant vessel for a short-horizon prediction.

This is NOT the already-rejected "TCPA-based sparsification" (which fully
replaced the distance criterion and lost the geometric locality that was
already shown to help). This is additive: keep everything the distance-based
top_k already contributes, and ADD the top_k_risk closest-DCPA APPROACHING
vessels that the pure distance criterion might have missed.

Selection rule per vessel i, over candidate neighbors j:
  - keep_dist: the top_k_dist NEAREST neighbors by current Euclidean distance
               (identical logic to model_cpagrn.py)
  - keep_risk: among neighbors with TCPA > 0 (i.e. actually approaching —
               vessels moving apart are never selected via this branch),
               the top_k_risk neighbors with the LOWEST DCPA
  - final kept set = keep_dist  UNION  keep_risk

Everything else (CPA edge features, GRU encoder, final spatial refinement,
decoder, per-step aggregation during the observation window) is unchanged
from the locked v4 architecture — only the neighbor-selection mask inside
NeighborAggregation is modified. Checkpoint tag convention:
CPAGRN_v5_uniontopk_obs<N>_pred<N>_s<seed>.

Input:  obs  [B, N, obs_len, 4]    (LON, LAT, SOG, Heading — z-score)
Output: pred [B, N, pred_len, 2]   (displacement in z-score space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_cpagrn import CPAFeatures


# ─────────────────────────────────────────────────────────────────────────────
# Union Top-K Neighbor Aggregation (distance ∪ risk)
# ─────────────────────────────────────────────────────────────────────────────

class NeighborAggregationUnion(nn.Module):
    """
    Same attention/message-passing mechanics as model_cpagrn.NeighborAggregation,
    but the sparsification mask is the UNION of:
      - top_k_dist nearest neighbors by current distance
      - top_k_risk lowest-DCPA neighbors among those with TCPA > 0 (approaching)

    A neighbor with no approaching relationship (TCPA <= 0) can never enter
    the kept set via the risk branch, regardless of how few approaching
    neighbors exist — it may still enter via the distance branch as before.
    """

    def __init__(
        self,
        d_model:    int,
        edge_dim:   int = 7,
        top_k_dist: int = 12,
        top_k_risk: int = 3,
    ):
        super().__init__()
        self.top_k_dist = top_k_dist
        self.top_k_risk = top_k_risk

        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.msg_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        h:     torch.Tensor,         # [B, N, d_model]
        edges: torch.Tensor,         # [B, N, N, 7]  (tcpa, dcpa, dist, sin, cos, dhdg, |dhdg|)
        mask:  torch.Tensor | None,  # [B, N] bool
    ) -> torch.Tensor:
        B, N, D = h.shape

        h_j    = h.unsqueeze(1).expand(B, N, N, D)
        scores = self.attn_mlp(torch.cat([h_j, edges], dim=-1)).squeeze(-1)  # [B, N, N]

        mask_j = None
        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        tcpa = edges[..., 0]
        dcpa = edges[..., 1]
        dist = edges[..., 2]

        # ── Distance branch (identical to v4) ──────────────────────────
        dist_masked = dist.masked_fill(~mask_j, float('inf')) if mask_j is not None else dist
        k_dist = min(self.top_k_dist, N)
        kth_dist, _ = dist_masked.topk(k_dist, dim=-1, largest=False)
        thresh_dist = kth_dist[..., -1].unsqueeze(-1)
        keep_dist = dist_masked <= thresh_dist

        # ── Risk branch: lowest DCPA among TCPA>0 (approaching) only ───
        risk_score = dcpa.masked_fill(tcpa <= 0, float('inf'))
        if mask_j is not None:
            risk_score = risk_score.masked_fill(~mask_j, float('inf'))
        k_risk = min(self.top_k_risk, N)
        kth_risk, _ = risk_score.topk(k_risk, dim=-1, largest=False)
        thresh_risk = kth_risk[..., -1].unsqueeze(-1)
        keep_risk = (risk_score <= thresh_risk) & (risk_score < float('inf'))

        # ── Union ───────────────────────────────────────────────────────
        keep = keep_dist | keep_risk
        scores = scores.masked_fill(~keep, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        msgs = self.msg_proj(h)
        agg  = torch.einsum('bij,bjd->bid', weights, msgs)

        return self.norm(self.out_proj(agg))


# ─────────────────────────────────────────────────────────────────────────────
# Main Model (identical structure to v4, swapping in the union aggregation)
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRNUnionTopK(nn.Module):
    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 1,
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k_dist:   int   = 12,
        top_k_risk:   int   = 3,
    ):
        super().__init__()
        self.d_model    = d_model
        self.pred_len   = pred_len
        self.top_k_dist = top_k_dist
        self.top_k_risk = top_k_risk

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.cpa_features = CPAFeatures()
        self.neighbor_agg = NeighborAggregationUnion(
            d_model, edge_dim=7, top_k_dist=top_k_dist, top_k_risk=top_k_risk
        )

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.final_spatial = NeighborAggregationUnion(
            d_model, edge_dim=7, top_k_dist=top_k_dist, top_k_risk=top_k_risk
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len * 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs:   torch.Tensor,
        mask:  torch.Tensor | None = None,
        stats: dict | None         = None,
    ) -> torch.Tensor:
        B, N, T, _ = obs.shape

        x = self.embed(obs)  # [B, N, T, d_model]

        fused_steps = []
        for t in range(T):
            pos_t = obs[:, :, t, :2]
            hdg_t = obs[:, :, t, 3]
            vel_t = obs[:, :, t, :2] - obs[:, :, t-1, :2] if t > 0 \
                    else torch.zeros_like(pos_t)

            edges_t = self.cpa_features(pos_t, vel_t, hdg_t)
            x_t     = x[:, :, t, :]
            nbr_t   = self.neighbor_agg(x_t, edges_t, mask)
            fused_steps.append(x_t + nbr_t)

        fused_seq = torch.stack(fused_steps, dim=2)

        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)
        h      = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        pos_last   = obs[:, :, -1, :2]
        vel_last   = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 \
                     else torch.zeros_like(pos_last)
        hdg_last   = obs[:, :, -1, 3]
        edges_last = self.cpa_features(pos_last, vel_last, hdg_last)
        h = h + self.final_spatial(h, edges_last, mask)

        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


# Re-export the shared loss (identical to v4)
from model_cpagrn import cpagrn_loss  # noqa: E402,F401
