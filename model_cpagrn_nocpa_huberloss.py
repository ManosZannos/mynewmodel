"""
model_cpagrn_nocpa_huberloss.py — No-CPA ablation matched to the FINAL headline recipe.

Purpose: the headline model is gru2 + lr=5e-4 + Huber(delta=0.05) with CPA edge
features [TCPA, DCPA, dist, sin(bearing), cos(bearing), dhdg, |dhdg|] (7 dims).
This file provides the ablation that differs from it in EXACTLY ONE factor:
the edge features are geometry only [dist, sin(bearing), cos(bearing), dhdg,
|dhdg|] (5 dims) — TCPA and DCPA are removed.

Verified with a line-by-line diff of model_cpagrn.py vs model_cpagrn_nocpa.py:
the only functional differences are (a) TCPA/DCPA removed from CPAFeatures,
(b) edge_dim 7 -> 5 in both NeighborAggregation modules, (c) the `dist` column
used for top-k sparsification moves from index 2 to index 0. The neighbor
selection itself (top-k by Euclidean distance), the per-timestep dynamic graph,
the GRU encoder, the final spatial refinement and the decoder are identical.

Note on interpretation: TCPA/DCPA are the ONLY edge features derived from
relative velocity, so this ablation removes the velocity information from the
edges as well. The GRU can still infer approach/recession indirectly from how
dist/bearing change across timesteps — which makes this a fair, non-trivial
baseline ("generic distance-based dynamic graph attention").

The Huber loss is imported from model_cpagrn_huberloss so that it is
guaranteed to be the SAME function used for the headline (no re-implementation).

Usage: see train_cpagrn_nocpa_huberloss.py.
"""

from model_cpagrn_nocpa import CPAGRN            # geometry-only architecture, unchanged
from model_cpagrn_huberloss import huber_cpagrn_loss  # identical loss as the headline

__all__ = ['CPAGRN', 'huber_cpagrn_loss']
