"""
model_cpagrn_huberloss.py — CPA-GRN with Huber (magnitude-based) loss.

Motivation: cpagrn_loss (model_cpagrn.py) is plain MSE on displacement,
summed over x/y and meaned over valid (vessel, timestep) pairs. MSE's
minimizer is the conditional MEAN of the target distribution — in a genuinely
multimodal encounter scenario (e.g. "turns" vs "continues straight"), this
can pull the deterministic prediction toward an unrealistic in-between
trajectory that no real vessel would follow ("mode averaging").

Huber loss is quadratic for small errors and LINEAR for large ones — it
still can't recover multimodality (the model remains deterministic, single-
trajectory), but it weighs large-error samples (the ones most likely to come
from a mode the model didn't pick) less aggressively than squared error
does, which can reduce the incentive to compromise between modes.

This is NOT a replacement for genuine multimodal modeling (CVAE, mixture
density, K-head winner-takes-all) — it is a cheap, zero-architectural-risk
FIRST test of whether the loss SHAPE alone matters here, before investing in
a heavier multimodal mechanism. Must be validated on the 10min headline
BEFORE any 30min test (per project convention: nothing proceeds to 30min
until it is at least neutral-to-positive on the 10min priority horizon).

Architecture is 100% IDENTICAL to model_cpagrn.CPAGRN (gru2 backbone) — this
file only adds the alternative loss function. Same checkpoint format, so
evaluate_cpagrn.py works unmodified on checkpoints trained with this loss.

Usage: see train_cpagrn_huberloss.py.
"""

import torch

from model_cpagrn import CPAGRN  # re-export the identical architecture, unchanged


def huber_cpagrn_loss(
    pred_disp:   torch.Tensor,
    target_disp: torch.Tensor,
    mask:        torch.Tensor,
    delta:       float = 0.05,
) -> torch.Tensor:
    """
    Huber loss on the EUCLIDEAN MAGNITUDE of the per-timestep displacement
    error (not per-component x/y), so `delta` has a direct geometric
    interpretation: errors smaller than `delta` (z-score units) are
    penalized quadratically (like MSE); errors larger than `delta` are
    penalized only linearly beyond that point (like MAE) — reducing the
    incentive to compromise between plausible-but-different trajectories
    when the model is uncertain/wrong by a large margin.

    `delta` is in the same z-score coordinate space as the model's internal
    displacement predictions (NOT degrees, NOT meters) — calibrate it using
    the percentile print in train_cpagrn_huberloss.py before a full run.
    """
    diff = pred_disp - target_disp                # [B, N, T, 2]
    dist = diff.norm(dim=-1)                       # [B, N, T] — Euclidean magnitude

    quadratic_part = torch.clamp(dist, max=delta)
    linear_part    = dist - quadratic_part
    loss_per_step  = 0.5 * quadratic_part ** 2 + delta * linear_part  # [B, N, T]

    m = mask.unsqueeze(-1).expand_as(loss_per_step)
    return loss_per_step[m].mean()


__all__ = ['CPAGRN', 'huber_cpagrn_loss']
