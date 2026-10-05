"""
model_cpagrn_relvel_huberloss.py — Relative-velocity control ablation, matched
to the FINAL headline recipe (gru2, lr=5e-4, Huber delta=0.05).

Question this answers: the headline edge features are
    [TCPA, DCPA, dist, sin(bearing), cos(bearing), dhdg, |dhdg|]          (7)
and TCPA/DCPA are the ONLY features that carry relative-velocity information.
Is the measured gain over the No-CPA / No-graph models due to the specific
CPA construction, or simply to giving the edges access to relative velocity?

This model replaces exactly the two CPA channels by the raw relative-velocity
vector of the same pair, with the SAME sign convention (v = vel_j - vel_i):
    [dvx, dvy, dist, sin(bearing), cos(bearing), dhdg, |dhdg|]            (7)

Why this is a clean control:
  * Same edge_dim (7), same channel positions (`dist` stays at index 2, so the
    distance-based top-k sparsification is untouched), same parameter count.
  * (dist, bearing) fix the relative position r and (dvx, dvy) is the relative
    velocity v, so this model has access to EXACTLY the information from which
    TCPA/DCPA are computed. The only thing it lacks is the hand-engineered
    nonlinear transformation. If the headline still wins, the benefit comes
    from the explicit CPA feature construction; if it ties, the benefit is
    velocity information, not collision geometry specifically.
  * Everything else is inherited unchanged from model_cpagrn.CPAGRN (forward
    pass, per-step graph, final spatial refinement, decoder, init). The
    replaced module has no learnable parameters, so with the same seed this
    model starts from EXACTLY the same initial weights as the headline.

Scaling: velocity here is the per-step displacement in z-score position units,
which is ~100x smaller than the (clamped) TCPA/DCPA channels. Feeding it raw
would hand the control an avoidable scale handicap. It is therefore divided by
`vel_scale`, a constant set once before training from the training data
(per-component RMS of the pairwise relative velocity, see
estimate_rel_vel_scale) and stored as a buffer, so it is saved in the
checkpoint and restored automatically at evaluation time.
"""

import numpy as np
import torch
import torch.nn as nn

from model_cpagrn import CPAGRN as _HeadlineCPAGRN
from model_cpagrn_huberloss import huber_cpagrn_loss

__all__ = ['CPAGRN', 'huber_cpagrn_loss', 'estimate_rel_vel_scale']


class RelVelFeatures(nn.Module):
    """Edge features [dvx, dvy, dist, sin(bearing), cos(bearing), dhdg, |dhdg|]."""

    EDGE_DIM = 7

    def __init__(self, vel_scale: float = 1.0):
        super().__init__()
        self.register_buffer('vel_scale', torch.tensor(float(vel_scale)))

    def forward(
        self,
        pos: torch.Tensor,   # [B, N, 2]
        vel: torch.Tensor,   # [B, N, 2]
        hdg: torch.Tensor,   # [B, N]
    ) -> torch.Tensor:
        """Returns [B, N, N, 7]"""
        B, N, _ = pos.shape

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i                          # relative position (same as headline)
        v = (vel_j - vel_i) / self.vel_scale       # relative velocity (same sign as headline), standardized

        dist    = r.norm(dim=-1)
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i

        return torch.stack([
            v[..., 0],
            v[..., 1],
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)  # [B, N, N, 7]


class CPAGRN(_HeadlineCPAGRN):
    """Headline architecture with the CPA edge channels replaced by relative velocity."""

    def __init__(self, *args, vel_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # CPAFeatures has no parameters, so replacing it does not change the
        # RNG consumption of the base __init__ (identical initial weights).
        self.cpa_features = RelVelFeatures(vel_scale)


def estimate_rel_vel_scale(loader, n_batches: int = 100) -> float:
    """
    Per-component RMS of the pairwise relative velocity (vel_j - vel_i) over
    ordered pairs of VALID vessels in the same scene, using the per-step
    displacement of z-scored positions (the same velocity definition as the
    model). Padding is excluded via the mask. Uses the identity
        sum_{i != j} |v_j - v_i|^2 = 2 n sum_i |v_i|^2 - 2 |sum_i v_i|^2
    so the cost is O(N) per scene instead of O(N^2). The mean of the relative
    velocity over ordered pairs is exactly 0 (antisymmetry), so the RMS equals
    the standard deviation.
    """
    total_sq, total_cnt = 0.0, 0.0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        obs, mask = batch[0], batch[2]
        obs = obs.detach().cpu().numpy().astype(np.float64)    # [B, N, T, 4]
        m   = mask.detach().cpu().numpy().astype(np.float64)   # [B, N]

        vel  = obs[:, :, 1:, :2] - obs[:, :, :-1, :2]          # [B, N, T-1, 2]
        n    = m.sum(axis=1)                                    # [B] valid vessels per scene
        keep = n >= 2
        if not keep.any():
            continue

        mm = m[:, :, None, None]
        S2 = (mm * vel ** 2).sum(axis=(1, 3))                   # [B, T-1]   sum_i |v_i|^2
        S1 = (mm * vel).sum(axis=1)                             # [B, T-1, 2] sum_i v_i
        pair_sq = 2.0 * n[:, None] * S2 - 2.0 * (S1 ** 2).sum(axis=-1)   # [B, T-1]
        pairs   = n * (n - 1.0)                                 # ordered pairs per scene-step

        steps      = vel.shape[2]
        total_sq  += pair_sq[keep].sum()
        total_cnt += pairs[keep].sum() * steps * 2.0            # x2 velocity components

    if total_cnt <= 0:
        raise RuntimeError('estimate_rel_vel_scale: no scene with >= 2 valid vessels found')
    return float(max(np.sqrt(total_sq / total_cnt), 1e-8))
