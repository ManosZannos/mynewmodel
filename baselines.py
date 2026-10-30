#!/usr/bin/env python3
"""
baselines.py — baselines για το V6 (ευαισθησία) και ως αναφορά για τη φάση μοντέλου.

  cv    constant velocity: ταχύτητα από τα 3 τελευταία λεπτά παρατήρησης (ίδιος ορισμός με το tier_cv)
  lstm  LSTM encoder (2 × 128) + MLP → 10 θέσεις· απώλεια = μέση ευκλείδεια απόσταση (ADE)·
        early stopping σε 10% των πλοίων του train (hash MMSI), ΟΧΙ στο split αξιολόγησης

Μετρικές (m): ADE/FDE σε όλα τα παράθυρα και στην κύρια μετρική (headline), και στην κύρια μετρική
ανά tier_cv, scene_group, vessel_cat, και για το υποσύνολο capped.

    python baselines.py --train_build dataset/socal_2025-03_210b4127 --model cv --eval_split val
    python baselines.py --train_build dataset/socal_2025-03_210b4127 --model lstm --eval_split val --seed 0
    python baselines.py --train_build dataset/socal_2025-03_210b4127 --model lstm --load_model reports/v6/lstm_s0.pt \
        --eval_build dataset/houston_2025-03_6e340c21 --eval_manifest alltest --eval_split test
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import loader as LD

BASELINES_VERSION = "baselines-v1.0"
STRATA = ["tier_cv", "scene_group", "vessel_cat"]


# ---------------------------------------------------------------------------
def cv_predict(pos, pred=LD.PRED):
    """pos (n, 10, 2) → (n, pred, 2): συνέχιση με την ταχύτητα των 3 τελευταίων λεπτών."""
    v = (pos[:, -1] - pos[:, -4]) / 3.0
    k = np.arange(1, pred + 1, dtype=np.float64)[None, :, None]
    return pos[:, -1:].astype(np.float64) + v[:, None, :].astype(np.float64) * k


def errors(pred, fut):
    d = np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(fut, np.float64), axis=-1)
    return d.mean(1), d[:, -1]


def metrics(ade, fde, meta):
    def agg(m):
        return {"n": int(m.sum()), "ADE": round(float(ade[m].mean()), 2) if m.any() else None,
                "FDE": round(float(fde[m].mean()), 2) if m.any() else None,
                "FDE_p90": round(float(np.percentile(fde[m], 90)), 1) if m.any() else None}
    head = meta["headline"].to_numpy() == 1
    out = {"all": agg(np.ones(len(meta), bool)), "headline": agg(head)}
    if "capped" in meta:
        out["headline_capped"] = agg(head & (meta["capped"].to_numpy() == 1))
    for s in STRATA:
        if s in meta:
            vals = meta[s].astype(str).to_numpy()
            out[f"by_{s}"] = {v: agg(head & (vals == v)) for v in sorted(set(vals[head]))}
    if "seen_mmsi" in meta:
        sm = meta["seen_mmsi"].to_numpy()
        out["by_seen"] = {"seen": agg(head & (sm == 1)), "unseen": agg(head & (sm == 0))}
    return out


def print_metrics(tag, m):
    h = m["headline"]
    print(f"  [{tag}] κύρια μετρική: ADE {h['ADE']} m | FDE {h['FDE']} m | FDE p90 {h['FDE_p90']} m | n={h['n']:,}"
          f"   (όλα: ADE {m['all']['ADE']}, FDE {m['all']['FDE']}, n={m['all']['n']:,})")
    if "headline_capped" in m:
        c = m["headline_capped"]
        print(f"     capped: ADE {c['ADE']} | FDE {c['FDE']} | n={c['n']:,}")
    for k in [k for k in m if k.startswith("by_")]:
        print(f"     {k[3:]:<12} " + " | ".join(f"{v}: {r['FDE']} ({r['n']:,})" for v, r in m[k].items()))


def shore_dir_of(build):
    c = sorted(glob.glob(f"{build}/stageC/shore_*"))
    return c[-1] if c else None


def _hash01(mmsi):
    return np.array([int(hashlib.sha1(f"es:{m}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for m in mmsi])


# ---------------------------------------------------------------------------
def make_lstm(n_feat, pred, hidden=128, layers=2):
    import torch.nn as nn

    class LSTMBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, pred * 2))

        def forward(self, x):
            _, (h, _) = self.enc(x)
            return self.head(h[-1]).view(x.shape[0], pred, 2)      # σε km
    return LSTMBaseline()


def lstm_predict(model, obs, device, bs=8192):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(obs), bs):
            out.append(model(torch.from_numpy(obs[a:a + bs]).to(device)).cpu().numpy())
    return np.concatenate(out) * LD.SCALES["pos_m"]


def train_lstm(tr, es, a, device):
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = make_lstm(tr["obs"].shape[-1], LD.PRED).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    X = torch.from_numpy(tr["obs"])
    Y = torch.from_numpy(tr["fut"] / LD.SCALES["pos_m"])
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(X), generator=g)
        for b in range(0, len(X), a.batch):
            i = perm[b:b + a.batch]
            xb, yb = X[i].to(device), Y[i].to(device)
            loss = torch.linalg.norm(model(xb) - yb, dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(i)
        ade_es = errors(lstm_predict(model, es["obs"], device), es["fut"])[0].mean()
        sched.step(ade_es)
        print(f"    epoch {ep:>2}: train ADE {tot / len(X) * LD.SCALES['pos_m']:.1f} m | early-stop ADE {ade_es:.1f} m | "
              f"lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f} s")
        if ade_es < best - 0.1:
            best, best_state, bad = ade_es, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_early_stop_ADE": round(float(best), 2), "epochs_run": ep}


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_build", required=True)
    ap.add_argument("--manifest", default="vessel")
    ap.add_argument("--model", choices=["cv", "lstm"], required=True)
    ap.add_argument("--eval_build", default=None, help="προεπιλογή: το train_build")
    ap.add_argument("--eval_manifest", default=None, help="προεπιλογή: το manifest")
    ap.add_argument("--eval_split", default="val")
    ap.add_argument("--train_capped", action="store_true", help="εκπαίδευση μόνο σε παράθυρα capped (V6)")
    ap.add_argument("--load_model", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default="reports/v6")
    a = ap.parse_args(argv)
    eb, em = a.eval_build or a.train_build, a.eval_manifest or a.manifest
    tag = a.tag or f"{a.model}_{Path(a.train_build).name}_{Path(eb).name}_{em}_{a.eval_split}" + \
        (f"_s{a.seed}" if a.model == "lstm" else "") + ("_trcapped" if a.train_capped else "")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[baselines] {tag}")

    sd_e = shore_dir_of(eb)
    print(f"  αξιολόγηση: {eb} / {em} / {a.eval_split} | στάδιο Γ: {sd_e}")
    ev = LD.load_split(eb, em, a.eval_split, shore_dir=sd_e)
    res = {"tag": tag, "versions": {"baselines": BASELINES_VERSION, "loader": LD.LOADER_VERSION,
                                    "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
           "args": vars(a), "eval": {"build": eb, "manifest": em, "split": a.eval_split, "shore_dir": sd_e}}
    try:
        res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        res["commit"] = None

    if a.model == "cv":
        pred = cv_predict(ev["pos"])
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        res["versions"]["torch"] = torch.__version__
        if a.load_model:
            model = make_lstm(ev["obs"].shape[-1], LD.PRED).to(device)
            model.load_state_dict(torch.load(a.load_model, map_location=device))
            res["model_path"] = a.load_model
        else:
            x_tr = LD.load_rows(a.train_build, shore_dir_of(a.train_build))
            tr = LD.load_split(a.train_build, a.manifest, "train", x=x_tr)
            if a.train_capped:
                keep = tr["meta"]["capped"].to_numpy() == 1
                tr = {**tr, "obs": tr["obs"][keep], "fut": tr["fut"][keep], "meta": tr["meta"][keep].reset_index(drop=True)}
            esm = _hash01(tr["meta"]["mmsi"].to_numpy()) < 0.1
            es = {"obs": tr["obs"][esm], "fut": tr["fut"][esm]}
            tr = {"obs": tr["obs"][~esm], "fut": tr["fut"][~esm]}
            print(f"  εκπαίδευση: {len(tr['obs']):,} παράθυρα | early stopping: {len(es['obs']):,} | {device}")
            model, info = train_lstm(tr, es, a, device)
            res["train"] = {**info, "n_train": int(len(tr["obs"])), "n_early_stop": int(len(es["obs"]))}
            res["model_path"] = str(out / f"{tag}.pt")
            torch.save(model.state_dict(), res["model_path"])
        pred = lstm_predict(model, ev["obs"], device)

    ade, fde = errors(pred, ev["fut"])
    res["metrics"] = metrics(ade, fde, ev["meta"])
    print_metrics(tag, res["metrics"])
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[baselines] → {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
