"""
model_cpagrn_smoothvel.py — CPA-Aware Graph Recurrent Network (v4 + smoothed velocity)

Identical to model_cpagrn.py (v4, locked-in final architecture) EXCEPT for
ONE change: the velocity estimate used to compute TCPA/DCPA edge features
is now a 2-frame moving average instead of a raw single-step finite
difference.

Motivation (evidence-based, not a blind hyperparameter sweep):
  - Documented finding: TCPA/DCPA help strongly at 10min, hurt at 20min,
    and become neutral at 30min. Interpretation on record: "instantaneous
    CPA is unreliable" at longer horizons because a single-frame velocity
    estimate is noisy and vessels change heading between the last
    observed step and the true future.
  - The raw finite-difference velocity (obs[t] - obs[t-1]) feeds this
    noise directly into TCPA/DCPA at every step.
  - This is NOT the same idea as the already-rejected "rate-of-change"
    experiment (dTCPA/dDCPA), which ADDED new derivative features
    (increasing EDGE_DIM, increasing overfitting risk). Here EDGE_DIM
    stays at 7 — we only clean the existing velocity input used to
    compute the same 7 features.

Change applied in exactly two places (both marked "SMOOTHED VELOCITY"):
  1. Per-step CPA aggregation loop (forward, step 2)
  2. Final spatial refinement, last-observed-step CPA features (forward, step 4)

Fallback logic for short windows:
  t >= 2  →  vel_t = (obs[t] - obs[t-2]) / 2.0      (2-frame average)
  t == 1  →  vel_t = obs[t] - obs[t-1]               (only one diff available)
  t == 0  →  vel_t = zeros                           (no motion info yet)

Everything else (embedding, attention, GRU, decoder, loss, parameter count)
is UNCHANGED from model_cpagrn.py. This keeps the ablation clean: exactly
one variable changes.

Recommended first test horizon: obs20/pred20 or obs30/pred30, where the
CPA contribution is currently negative or neutral — that is where this
fix, if the underlying hypothesis is correct, should show the clearest
effect. Testing on obs10/pred10 first is not recommended: CPA is already
close to optimal there and the effect size would be hard to distinguish
from run-to-run noise.

Input:  obs  [B, N, obs_len, 4]    (LON, LAT, SOG, Heading — z-score)
Output: pred [B, N, pred_len, 2]   (displacement in z-score space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPA Feature Computation (single timestep) — UNCHANGED from model_cpagrn.py
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    """
    Computes 7 pairwise edge features for a single timestep snapshot.

    For each vessel pair (i → j):
        TCPA             — time to closest point of approach
        DCPA             — distance at closest point of approach
        dist             — current Euclidean distance
        sin/cos(bearing) — direction from i to j
        dhdg             — relative heading (z-score space)
        dhdg.abs()       — absolute heading difference

    All computed in z-score normalised space.

    NOTE: this module itself is unchanged — it still just consumes
    whatever (pos, vel, hdg) it is given. The smoothing happens in the
    caller (CPAGRN.forward), where `vel` is now a 2-frame average
    instead of a raw single-step diff.
    """

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pos: torch.Tensor,   # [B, N, 2]  current positions (LON, LAT)
        vel: torch.Tensor,   # [B, N, 2]  current velocity
        hdg: torch.Tensor,   # [B, N]     current heading
    ) -> torch.Tensor:
        """Returns [B, N, N, 7]"""
        B, N, _ = pos.shape

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i          # relative position
        v = vel_j - vel_i          # relative velocity

        dist    = r.norm(dim=-1)
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i

        v_sq = (v * v).sum(dim=-1) + self.eps
        tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)
        dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

        return torch.stack([
            tcpa,
            dcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)  # [B, N, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sparse Neighbor Aggregation (distance-based top_k) — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

class NeighborAggregation(nn.Module):
    """
    Aggregates neighbor messages at a single timestep.

    Each vessel collects a weighted sum of neighbor embeddings, weighted by
    CPA-aware attention scores. Sparsified to the top_k NEAREST neighbors
    by current Euclidean distance (this was found to outperform TCPA-based
    or unrestricted attention in extensive ablation).
    """

    def __init__(self, d_model: int, edge_dim: int = 7, top_k: int = 10):
        super().__init__()
        self.top_k = top_k

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
        h:     torch.Tensor,         # [B, N, d_model]  current vessel embeddings
        edges: torch.Tensor,         # [B, N, N, 7]     CPA edge features
        mask:  torch.Tensor | None,  # [B, N] bool
    ) -> torch.Tensor:
        B, N, D = h.shape

        h_j    = h.unsqueeze(1).expand(B, N, N, D)   # neighbor embeddings
        scores = self.attn_mlp(
            torch.cat([h_j, edges], dim=-1)
        ).squeeze(-1)  # [B, N, N]

        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        dist = edges[..., 2]  # [B, N, N]
        if mask is not None:
            dist_masked = dist.masked_fill(~mask_j, float('inf'))
        else:
            dist_masked = dist

        if self.top_k < N:
            k = min(self.top_k, N)
            kth, _ = dist_masked.topk(k, dim=-1, largest=False)
            threshold = kth[..., -1].unsqueeze(-1)
            scores = scores.masked_fill(dist_masked > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)           # [B, N, N]
        weights = torch.nan_to_num(weights, nan=0.0)

        msgs = self.msg_proj(h)                        # [B, N, D]
        agg  = torch.einsum('bij,bjd->bid', weights, msgs)  # [B, N, D]

        return self.norm(self.out_proj(agg))           # [B, N, D]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """
    CPA-Aware Graph Recurrent Network — v4 + smoothed velocity for CPA features.

    The encoder processes each observation timestep with spatial context
    from neighbors, allowing the GRU hidden state to encode the temporal
    evolution of vessel interactions, not just individual vessel kinematics.

    ONLY DIFFERENCE from model_cpagrn.py (v4): velocity used to compute
    TCPA/DCPA is a 2-frame moving average instead of a 1-frame finite
    difference. See module docstring for the evidence-based motivation.
    """

    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 1,
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k:        int   = 10,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len
        self.top_k    = top_k

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.cpa_features = CPAFeatures()
        self.neighbor_agg = NeighborAggregation(d_model, edge_dim=7, top_k=top_k)

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.final_spatial = NeighborAggregation(d_model, edge_dim=7, top_k=top_k)

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

    @staticmethod
    def _smoothed_velocity(obs: torch.Tensor, t: int) -> torch.Tensor:
        """
        2-frame moving-average velocity estimate at timestep t.

        obs: [B, N, T, 4]  (LON, LAT, SOG, Heading — z-score)
        t:   current timestep index

        t >= 2  →  (obs[t] - obs[t-2]) / 2.0   (average of last 2 steps)
        t == 1  →  obs[t] - obs[t-1]           (only one diff available)
        t == 0  →  zeros                       (no motion info yet)

        SMOOTHED VELOCITY (Πρόταση 1): replaces the raw single-step
        finite difference used in model_cpagrn.py, to reduce per-frame
        noise fed into TCPA/DCPA — this is the ONLY functional change
        relative to v4.
        """
        pos_t = obs[:, :, t, :2]
        if t >= 2:
            return (obs[:, :, t, :2] - obs[:, :, t - 2, :2]) / 2.0
        elif t == 1:
            return obs[:, :, t, :2] - obs[:, :, t - 1, :2]
        else:
            return torch.zeros_like(pos_t)

    def forward(
        self,
        obs:   torch.Tensor,
        mask:  torch.Tensor | None = None,
        stats: dict | None         = None,   # unused, kept for API compatibility
    ) -> torch.Tensor:
        """
        obs:  [B, N, T_obs, 4]   (LON, LAT, SOG, Heading — z-score)
        mask: [B, N] bool        (True = real vessel, False = padding)
        Returns: [B, N, pred_len, 2]  (displacement in z-score space)
        """
        B, N, T, _ = obs.shape

        # 1. Per-step embedding
        x = self.embed(obs)  # [B, N, T, d_model]

        # 2. Per-step neighbor aggregation (temporally-aware encoder input)
        fused_steps = []
        for t in range(T):
            pos_t = obs[:, :, t, :2]
            hdg_t = obs[:, :, t, 3]

            # ── SMOOTHED VELOCITY (Πρόταση 1) ──────────────────────────
            vel_t = self._smoothed_velocity(obs, t)
            # ─────────────────────────────────────────────────────────

            edges_t = self.cpa_features(pos_t, vel_t, hdg_t)  # [B, N, N, 7]
            x_t     = x[:, :, t, :]                            # [B, N, d_model]
            nbr_t   = self.neighbor_agg(x_t, edges_t, mask)    # [B, N, d_model]
            fused_steps.append(x_t + nbr_t)

        fused_seq = torch.stack(fused_steps, dim=2)  # [B, N, T, d_model]

        # 3. GRU Encoder
        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)                     # [layers, B*N, d_model]
        h      = h_n[-1].reshape(B, N, self.d_model)  # [B, N, d_model]

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # 4. Final spatial refinement (last-observed-step CPA features)
        pos_last = obs[:, :, -1, :2]

        # ── SMOOTHED VELOCITY (Πρόταση 1) ──────────────────────────────
        vel_last = self._smoothed_velocity(obs, T - 1)
        # ─────────────────────────────────────────────────────────────

        hdg_last   = obs[:, :, -1, 3]
        edges_last = self.cpa_features(pos_last, vel_last, hdg_last)
        h = h + self.final_spatial(h, edges_last, mask)

        # 5. Decode
        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def cpagrn_loss(
    pred_disp:   torch.Tensor,
    target_disp: torch.Tensor,
    mask:        torch.Tensor,
) -> torch.Tensor:
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()
