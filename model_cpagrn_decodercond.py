"""
model_cpagrn_decodercond.py — CPA-GRN, Decoder-Level CPA Conditioning variant (Πρόταση 2)

Motivation: the locked v4/gru2 decoder is a single MLP that maps the final
encoder hidden state h -> ALL pred_len displacement steps at once
(Linear(d_model, pred_len*2)). It has no explicit signal about WHICH future
step it is currently producing, nor how close that step is to the vessel's
CPA (closest point of approach) moment. It can only infer this implicitly
from the compressed hidden state.

This is a direct response to an already-documented ablation finding: TCPA/DCPA
contribute proportionally MORE to FDE than to ADE (i.e. the CPA signal matters
most near the end of the prediction horizon, close to the moment of closest
approach) — but the decoder currently has no explicit mechanism to exploit
that per-step structure.

Change (decoder ONLY — encoder, neighbor selection, GRU are all identical to
the confirmed gru2 architecture):
  For each prediction step k = 1..pred_len:
    1. Take the single most-relevant real-world CPA context for this vessel:
       (TCPA, DCPA) to its closest APPROACHING neighbor (TCPA>0, lowest DCPA),
       computed once at the last observed timestep (same definition already
       used by evaluate_cpagrn_stratified.py for the risky/non-risky split).
       Vessels with no approaching neighbor get a neutral sentinel (0, 10) —
       10 matches the existing DCPA clamp/sentinel convention used elsewhere
       in this codebase (train_cpagrn_riskloss.py).
    2. cond_k = MLP([TCPA, DCPA, k/pred_len])   — small, cond_dim-wide vector
       that tells the decoder "how far into the horizon am I, and was this
       vessel in a real encounter when we started predicting?"
    3. output_k = MLP([h, cond_k])  ->  2D displacement for that step

This is a genuinely different mechanism from every previous combination
attempt (top_k+gru2, gru2+riskloss, gru2+smoothvel, union-topk): those all
changed WHAT the encoder sees or HOW the loss is weighted. This changes HOW
the decoder uses the CPA signal it already has access to, targeting the
FDE-specific benefit directly instead of hoping the encoder's compressed
hidden state already captures it.

Built on top of the gru2 backbone (gru_layers=2 default, top_k=10 distance-
based neighbor selection) — the current headline recipe (gru2 + lr=5e-4).

Input:  obs  [B, N, obs_len, 4]    (LON, LAT, SOG, Heading — z-score)
Output: pred [B, N, pred_len, 2]   (displacement in z-score space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_cpagrn import CPAFeatures, NeighborAggregation, cpagrn_loss


# ─────────────────────────────────────────────────────────────────────────────
# Per-vessel CPA context extraction (single closest APPROACHING neighbor)
# ─────────────────────────────────────────────────────────────────────────────

def nearest_approaching_cpa(
    edges: torch.Tensor,        # [B, N, N, 7]  (tcpa, dcpa, dist, ...)
    mask:  torch.Tensor | None, # [B, N] bool
    sentinel_dcpa: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each vessel i, finds the neighbor j with TCPA>0 (approaching) and the
    LOWEST DCPA — same definition as the risky/non-risky split in
    evaluate_cpagrn_stratified.py. Vessels with no approaching neighbor get
    (tcpa=0, dcpa=sentinel_dcpa) — a neutral "no imminent encounter" signal.

    Returns: (tcpa_ctx, dcpa_ctx), each [B, N]
    """
    B, N, _, _ = edges.shape
    tcpa = edges[..., 0]
    dcpa = edges[..., 1]

    invalid = tcpa <= 0
    if mask is not None:
        mask_j = (~mask).unsqueeze(1).expand(B, N, N)
        invalid = invalid | mask_j

    dcpa_for_min = dcpa.masked_fill(invalid, sentinel_dcpa)
    min_dcpa, min_idx = dcpa_for_min.min(dim=-1)          # [B, N]
    tcpa_at_min = torch.gather(
        tcpa, dim=-1, index=min_idx.unsqueeze(-1)
    ).squeeze(-1)                                         # [B, N]

    no_approach = min_dcpa >= (sentinel_dcpa - 1e-6)
    tcpa_ctx = torch.where(no_approach, torch.zeros_like(tcpa_at_min), tcpa_at_min)
    dcpa_ctx = min_dcpa

    return tcpa_ctx, dcpa_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRNDecoderCond(nn.Module):
    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 2,      # default matches current headline (gru2)
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k:        int   = 10,
        cond_dim:     int   = 16,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len
        self.top_k    = top_k
        self.cond_dim = cond_dim

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

        # ── Decoder conditioning (NEW — additive residual correction on top
        # of the full per-step decoder, NOT a replacement of it). The delta
        # projection is zero-initialized so at the start of training this
        # model behaves EXACTLY like the plain gru2 decoder; the conditioning
        # can only learn a correction, never destroy the base decoder's
        # already-good per-step differentiation. An earlier draft tied the
        # output projection across all pred_len steps, which collapsed the
        # decoder's per-step capacity into the tiny cond_dim pathway and
        # caused severe degradation especially at short horizons (near-term
        # prediction needs precise, largely-independent-per-step weights,
        # exactly what the base decoder already provides) — this fixes that. ──
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len * 2),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),      # [TCPA, DCPA, k/pred_len]
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
        )
        self.delta_proj = nn.Linear(cond_dim, 2)  # zero-initialized below

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the delta projection LAST (after the general loop above
        # would otherwise xavier-init it): at the start of training, the
        # conditioning branch contributes exactly zero, so this model is
        # numerically IDENTICAL to plain gru2 at init. Training can only
        # learn a correction from there — it cannot start by destabilizing
        # an already-good base decoder.
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(
        self,
        obs:   torch.Tensor,
        mask:  torch.Tensor | None = None,
        stats: dict | None         = None,
    ) -> torch.Tensor:
        B, N, T, _ = obs.shape

        # 1-4: identical to gru2 (embed -> per-step neighbor agg -> GRU -> final spatial)
        x = self.embed(obs)

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

        # 5. Decoder conditioning (NEW) — base decoder keeps its FULL
        # per-step capacity (identical to gru2); conditioning contributes
        # only a small additive correction, vectorized over all pred_len
        # steps at once (nn.Linear broadcasts over leading dims).
        tcpa_ctx, dcpa_ctx = nearest_approaching_cpa(edges_last, mask)  # [B, N] each

        base = self.decoder(h).reshape(B, N, self.pred_len, 2)  # [B,N,P,2], same as gru2

        t_idx = torch.arange(
            1, self.pred_len + 1, device=obs.device, dtype=h.dtype
        ) / self.pred_len                                        # [pred_len]
        t_frac   = t_idx.view(1, 1, -1).expand(B, N, self.pred_len)          # [B,N,P]
        tcpa_exp = tcpa_ctx.unsqueeze(-1).expand(B, N, self.pred_len)        # [B,N,P]
        dcpa_exp = dcpa_ctx.unsqueeze(-1).expand(B, N, self.pred_len)        # [B,N,P]

        cond_in  = torch.stack([tcpa_exp, dcpa_exp, t_frac], dim=-1)  # [B,N,P,3]
        cond_all = self.cond_mlp(cond_in)                             # [B,N,P,cond_dim]
        delta    = self.delta_proj(cond_all)                          # [B,N,P,2] — ≈0 at init

        out = base + delta

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


# Re-export the shared loss (identical to v4/gru2)
__all__ = ['CPAGRNDecoderCond', 'cpagrn_loss', 'nearest_approaching_cpa']