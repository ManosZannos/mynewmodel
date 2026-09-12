"""
model_cpagrn_auxrisk.py — CPA-GRN, Auxiliary Risk-Prediction Head variant (Πρόταση 4)

Motivation: add a second, small output head that predicts the REAL future
DCPA-at-closest-approach for each vessel (computed from ground-truth future
trajectories of ALL vessels in the scene — a free supervised signal, not an
extra label). A small auxiliary loss term forces the shared encoder to encode
risk-relevant information explicitly, with ZERO added inference cost (the aux
head is simply not used/dropped at deployment — only the main displacement
decoder matters at test time).

This targets a genuinely different mechanism from every prior attempt:
  - NOT neighbor selection (top_k=15, union-topk)
  - NOT loss reweighting by INSTANTANEOUS DCPA (riskloss, already rejected)
  - NOT decoder conditioning on already-known CPA at observation time
  This forces the ENCODER's representation itself to be predictive of a
  REALIZED future risk outcome, via multi-task learning — a standard
  technique in trajectory prediction literature (auxiliary intent/goal
  prediction).

Built on top of the gru2 backbone (identical encoder/GRU/neighbor selection
to the confirmed headline gru2 + lr=5e-4) — only two additions:
  1. A small aux_head: Linear(d_model -> d_model//2) -> ReLU -> Linear(-> 1),
     applied to the same final hidden state h used by the main decoder.
  2. compute_true_future_dcpa(): a pure-tensor helper (no model params) that
     computes, from the GROUND-TRUTH future absolute positions of all
     vessels in the scene, the realized minimum distance to any other real
     vessel across the whole prediction window — the target for the aux head.

Input:  obs  [B, N, obs_len, 4]    (LON, LAT, SOG, Heading — z-score)
Output: (pred, aux_pred)
    pred:     [B, N, pred_len, 2]  displacement in z-score space (main task)
    aux_pred: [B, N]               predicted realized future min-DCPA (aux task)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_cpagrn import CPAFeatures, NeighborAggregation, cpagrn_loss


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth future closest-approach distance (pure tensor ops, no params)
# ─────────────────────────────────────────────────────────────────────────────

def compute_true_future_dcpa(
    pred_abs_pos: torch.Tensor,  # [B, N, pred_len, 2]  ABSOLUTE future positions (z-score)
    mask:         torch.Tensor,  # [B, N] bool
    sentinel:     float = 10.0,
) -> torch.Tensor:
    """
    For each vessel i, the realized minimum distance to any OTHER real vessel
    j, minimized over every future timestep in the prediction window. This is
    the ground-truth target for the auxiliary risk head — NOT a TCPA/DCPA
    geometric estimate, but the actual closest approach that happens in the
    ground-truth future trajectories.

    Returns: [B, N] — sentinel (10.0) if no other real vessel exists (N=1
    after masking, shouldn't normally occur since min_vessels=3 upstream).
    """
    B, N, T, _ = pred_abs_pos.shape

    pos_i = pred_abs_pos.unsqueeze(2)  # [B, N, 1, T, 2]
    pos_j = pred_abs_pos.unsqueeze(1)  # [B, 1, N, T, 2]
    dist  = (pos_i - pos_j).norm(dim=-1)  # [B, N, N, T]

    min_dist_over_time = dist.min(dim=-1).values  # [B, N, N] — closest approach per pair

    # Mask out invalid neighbors (padding) and self (i == j)
    eye = torch.eye(N, device=pred_abs_pos.device, dtype=torch.bool).unsqueeze(0)  # [1,N,N]
    invalid = eye.expand(B, N, N).clone()
    if mask is not None:
        mask_j = (~mask).unsqueeze(1).expand(B, N, N)
        invalid = invalid | mask_j

    min_dist_over_time = min_dist_over_time.masked_fill(invalid, sentinel)
    true_dcpa = min_dist_over_time.min(dim=-1).values.clamp(0.0, sentinel)  # [B, N]

    return true_dcpa


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRNAuxRisk(nn.Module):
    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 2,      # default matches current headline (gru2)
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

        # ── Auxiliary risk head (NEW — dropped at deployment, zero extra
        # inference cost for the task that actually matters) ──
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
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
    ):
        B, N, T, _ = obs.shape

        # 1-4: identical to gru2
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

        # 5. Main decode (unchanged)
        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        # 6. Auxiliary risk prediction (NEW)
        aux_pred = self.aux_head(h).squeeze(-1)  # [B, N]

        return out, aux_pred


def auxrisk_loss(
    pred_disp:    torch.Tensor,
    target_disp:  torch.Tensor,
    aux_pred:     torch.Tensor,
    true_dcpa:    torch.Tensor,
    mask:         torch.Tensor,
    aux_weight:   float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, main_loss, aux_loss) for logging both terms."""
    main_loss = cpagrn_loss(pred_disp, target_disp, mask)

    aux_sq_err = (aux_pred - true_dcpa) ** 2
    aux_loss   = aux_sq_err[mask].mean()

    total = main_loss + aux_weight * aux_loss
    return total, main_loss, aux_loss


__all__ = ['CPAGRNAuxRisk', 'compute_true_future_dcpa', 'auxrisk_loss']
