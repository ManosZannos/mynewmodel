#!/usr/bin/env python3
"""
baseline_transformer.py — baseline: Transformer encoder πάνω στην ίδια αιτιατή είσοδο με το LSTM του baselines.py.

  είσοδος   obs του loader (11 βήματα × 11 χαρακτηριστικά, το 11ο = η τελευταία αναφορά), γραμμική προβολή σε d,
            μαθημένη θέση ανά βήμα· χωρίς περιστροφή και χωρίς διόρθωση πάνω στο CV, όπως συνήθως στη βιβλιογραφία
  encoder   pre-norm, d = 128, 4 κεφαλές, 2 επίπεδα, ff 256, dropout 0.1 (~ίδιες παράμετροι με το LSTM 2 × 128)
  έξοδος    MLP από το διάνυσμα του τελευταίου βήματος (τελευταία αναφορά) → 10 θέσεις, όπως το LSTM
Εκπαίδευση, early stopping, loss (μέση ευκλείδεια απόσταση), checkpoint και αποτελέσματα: ακριβώς του baselines.py
(--model transformer). Οι υπερπαράμετροι ορίστηκαν πριν από κάθε run και δεν ρυθμίζονται απέναντι στο μοντέλο μας.

    python baseline_transformer.py --train_build dataset/socal_2025-03_b19c45d7 --eval_split val --seed 0
"""
import sys

import baselines as BL

D_MODEL, HEADS, LAYERS, FF, DROPOUT = 128, 4, 2, 256, 0.1


def make_transformer(n_feat, pred, steps=11):
    import torch
    import torch.nn as nn

    class TransformerBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_feat, D_MODEL)
            self.pos = nn.Parameter(torch.zeros(1, steps, D_MODEL))
            layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, FF, DROPOUT, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(D_MODEL)
            self.head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, pred * 2))

        def forward(self, x):
            h = self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]]))
            return self.head(h[:, -1]).view(x.shape[0], pred, 2)     # km, like the LSTM
    return TransformerBaseline()


BL.MODELS["transformer"] = make_transformer


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--model" in argv:
        sys.exit("το μοντέλο είναι πάντα transformer: χωρίς --model")
    return BL.main(argv + ["--model", "transformer"])


if __name__ == "__main__":
    main()
