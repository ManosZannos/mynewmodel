#!/usr/bin/env python3
"""
phase3_pool.py — ΕΦΕΔΡΙΚΟ σκέλος: M0 + pooling γειτόνων (αντί για το κανάλι πλοίων του πολικού πεδίου).

Ίδιο μοντέλο, εκπαίδευση, αξιολόγηση και ορίσματα με το phase3.py· αλλάζει μόνο το context: οι έως 8 πλησιέστεροι
γείτονες του stageE_arms.py (nb), κοινό MLP ανά γείτονα και max πάνω στους έγκυρους (αμετάβλητο στη σειρά τους).
Παράθυρο χωρίς γείτονα δίνει μηδενικό διάνυσμα. Η ιδέα του pooling υπάρχει στη βιβλιογραφία (spec §13.7).

    python phase3_pool.py --train_build dataset/socal_2025-03_b19c45d7 --seed 0
"""
import phase3 as P3
import stageE_arms as E


def ctx_module(H):
    import torch
    import torch.nn as nn

    class Pool(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(5, H), nn.ReLU(), nn.Linear(H, H))

        def forward(self, c):
            f, valid = c[:, :E.NB_K * 5].reshape(len(c), E.NB_K, 5), c[:, E.NB_K * 5:] > 0
            e = self.mlp(f).masked_fill(~valid[..., None], float("-inf")).amax(1)
            return torch.where(valid.any(1, keepdim=True), e, torch.zeros_like(e))
    return Pool()


def ctx_loader(build, d, field_dir=None):
    return E.load_neighbours(build, d["meta"]["window_id"].to_numpy())


def main(argv=None):
    return P3.main(argv, "pool", ctx_module, ctx_loader)


if __name__ == "__main__":
    main()
