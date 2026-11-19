#!/usr/bin/env python3
"""
phase3_sdf.py — ΕΦΕΔΡΙΚΟ σκέλος: M0 + τοπικό patch προσημασμένης απόστασης από την ακτή (αντί για το κανάλι ξηράς).

Ίδιο μοντέλο, εκπαίδευση, αξιολόγηση και ορίσματα με το phase3.py· αλλάζει μόνο το context: patch 32 × 32 σε ±5 km
γύρω από το τελευταίο διαθέσιμο σημείο παρατήρησης, περιστραμμένο στην πορεία, κομμένο από το raster του
stageE_arms.py (sdf)· μικρό CNN με global pooling. Η ιδέα του SDF υπάρχει στη βιβλιογραφία (spec §13.7).
Τα patches υπολογίζονται στη μνήμη σε κάθε εκτέλεση (~4 MB ανά 1000 παράθυρα).

    python phase3_sdf.py --train_build dataset/socal_2025-03_b19c45d7 --seed 0
"""
import phase3 as P3
import stageE_arms as E

N = E.SDF["n"]


def ctx_module(H):
    import torch.nn as nn

    class Patch(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, H), nn.ReLU())

        def forward(self, c):
            return self.cnn(c.view(len(c), 1, N, N))
    return Patch()


def ctx_loader(build, d):
    m = d["meta"]
    _, th = P3.course(d["pos"], m["last_avail"], m["p_age_s"])
    return E.sdf_patches(E.load_sdf(build), m["lat_last"].to_numpy(), m["lon_last"].to_numpy(), th)


def main(argv=None):
    return P3.main(argv, "sdf", ctx_module, ctx_loader)


if __name__ == "__main__":
    main()
