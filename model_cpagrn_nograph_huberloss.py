"""
model_cpagrn_nograph_huberloss.py — No-graph ablation, matched to the FINAL
headline recipe (gru2, lr=5e-4, Huber delta=0.05).

Purpose: the headline-vs-LSTM comparison (+38.8% ADE) conflates two changes at
once — LSTM -> GRU encoder, AND no-graph -> CPA-graph. This file isolates the
encoder choice alone: same embed -> GRU(2 layers) -> decoder skeleton as
model_cpagrn.py / model_cpagrn_nocpa.py, but with BOTH NeighborAggregation
calls (per-timestep and final-spatial) removed entirely. Every vessel is
encoded independently; there is no cross-vessel information flow whatsoever.

This gives the clean three-link chain for the paper:
    LSTM -> nograph   : effect of the encoder (LSTM vs GRU), no interaction in either
    nograph -> nocpa  : effect of adding a graph (geometry-only edges)
    nocpa -> headline : effect of adding CPA/TCPA edge features (already measured: +5.3% ADE)

Verified by construction (not by diffing against model_cpagrn_nocpa.py, since
large parts are deleted): forward() keeps exactly the embed/GRU/decoder path of
model_cpagrn_nocpa.py — same embed Sequential, same nn.GRU constructor
arguments, same decoder Sequential, same xavier_uniform_ init, same final
mask-multiply steps — and nothing else. No CPAFeatures module, no
NeighborAggregation module exists in this file at all.

Parameter count: d_model=64, gru_layers=2 -> embed (4*64+64=320) + GRU
(2-layer, 64->64: 3*64*(64+64+1)*2=49,536... see printed `Parameters:` line in
training log) + decoder (64*64+64 + 64*20+20=5,460). No NeighborAggregation
params (no attn_mlp/msg_proj/out_proj/norm) are removed relative to nocpa.

The Huber loss is imported from model_cpagrn_huberloss so that it is
guaranteed to be the SAME function used for the headline and the No-CPA
ablation (no re-implementation).
"""

import torch
import torch.nn as nn

from model_cpagrn_huberloss import huber_cpagrn_loss

__all__ = ['CPAGRN', 'huber_cpagrn_loss']


class CPAGRN(nn.Module):
    """Plain per-vessel GRU encoder-decoder. No neighbor aggregation at all —
    every vessel is encoded and decoded completely independently of every
    other vessel in the scene. This is the "no interaction mechanism"
    reference point for the ablation chain."""

    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 1,
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k:        int   = 10,   # unused; kept so callers built for the
                                     # other CPAGRN variants pass --top_k
                                     # without needing a special case.
    ):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
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

    def forward(self, obs, mask=None, stats=None):
        B, N, T, _ = obs.shape

        x = self.embed(obs)                          # [B, N, T, d_model]

        gru_in = x.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)
        h      = h_n[-1].reshape(B, N, self.d_model)  # last layer's final hidden state

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out
