#!/usr/bin/env python3
"""
phase3_sdf_habit.py — σκέλος: M0 + patch SDF (phase3_sdf.py) στη θέση του πολικού καναλιού ξηράς + κανάλι συνήθειας.

Ίδιο μοντέλο, εκπαίδευση, αξιολόγηση και ορίσματα με το phase3.py. Context = [patch SDF 32 × 32 | hab_kk, hab_n του
πεδίου (stageD/field.parquet ή --field_dir)]. Το patch περνά από το CNN του phase3_sdf.py, η συνήθεια από ένα γραμμικό
επίπεδο όπως στο phase3.py, και τα δύο ενώνονται σε ένα επίπεδο H. Με --load_model, το --eval_field_dir αλλάζει
μόνο τη συνήθεια (π.χ. stageD_hab στο Houston)· το raster SDF είναι πάντα του build αξιολόγησης.

    python phase3_sdf_habit.py --train_build dataset/socal_2025-03_b19c45d7 --seed 0
"""
from pathlib import Path

import numpy as np

import phase3 as P3
import phase3_sdf as PS
import stageD_field as D

N2 = PS.N ** 2
NH = D.DEFAULTS["K"] + 1          # hab_kk (K sectors) + hab_n


def ctx_module(H):
    import torch
    import torch.nn as nn

    class SdfHabit(nn.Module):
        def __init__(self):
            super().__init__()
            self.sdf = PS.ctx_module(H)
            self.hab = nn.Sequential(nn.Linear(NH, H), nn.ReLU())
            self.mix = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU())

        def forward(self, c):
            return self.mix(torch.cat([self.sdf(c[:, :N2]), self.hab(c[:, N2:])], 1))
    return SdfHabit()


def ctx_loader(build, d, field_dir="stageD"):
    hab = P3.load_field(Path(build) / field_dir / "field.parquet", d["meta"]["window_id"].to_numpy(), ["habit"])
    assert hab.shape[1] == NH, hab.shape
    return np.concatenate([PS.ctx_loader(build, d), hab], 1).astype(np.float32)


def main(argv=None):
    return P3.main(argv, "sdf+habit", ctx_module, ctx_loader)


if __name__ == "__main__":
    main()
