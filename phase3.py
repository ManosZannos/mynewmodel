#!/usr/bin/env python3
"""
phase3.py — μοντέλο της Φάσης 3. Έκδοση M0: μόνο το ίδιο το πλοίο (χωρίς ακτή, χωρίς γείτονες).

  είσοδος   obs του loader (11 βήματα × 11 χαρακτηριστικά), περιστραμμένη ώστε η ταχύτητα του CV να είναι στον +x
  έξοδος    διόρθωση πάνω στο αιτιατό CV του baselines.py, όλος ο ορίζοντας με ένα πέρασμα
  loss      Huber στην ευκλείδεια απόσταση (τετραγωνική ως HUBER_M, γραμμική πιο πάνω)
  δοκιμές   --no_rotate, --no_cv, --encoder gru (μία αλλαγή τη φορά απέναντι στην προεπιλογή)

Πρωτόκολλο εκπαίδευσης, μετρικές, strata και μορφή αποτελεσμάτων: ίδια με το baselines.py (spec §12),
ώστε τα JSON να συγκρίνονται με compare_runs.py / scorecard.py. Η εκπαίδευση αξιολογεί μόνο σε val·
τα test θέλουν --load_model (χωριστό, χειροκίνητο βήμα). Με --load_model η αρχιτεκτονική διαβάζεται
από το checkpoint, όχι από τη γραμμή εντολών.

    python phase3.py --train_build dataset/socal_2025-03_b19c45d7 --seed 0
    python phase3.py --train_build dataset/socal_2025-03_b19c45d7 --load_model reports/phase3/<tag>.pt \
        --eval_build dataset/houston_2025-03_7bfeb419 --eval_manifest alltest --eval_split test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as BL
import loader as LD

PHASE3_VERSION = "phase3-v0.1"
RES_SCALE = LD.SCALES["pos_m"]      # network output unit (m); provisional, like the loader scales (spec §4.3)
HUBER_M = 100.0                     # fixed before any result: errors above 100 m are penalised linearly
ARCH = ("encoder", "hidden", "no_cv", "no_rotate")
DILATIONS = (1, 2, 4)               # kernel 3 -> receptive field 15 >= the 11 input steps
_I = {f: i for i, f in enumerate(LD.FEATURES)}
# feature pairs that are (east, north) vectors; (sin_cog, cos_cog) is the unit course vector
VEC = [[_I["x"], _I["y"]], [_I["dx"], _I["dy"]], [_I["sin_cog"], _I["cos_cog"]]]


# ---------------------------------------------------------------------------
def rot(v, th):
    """Rotate (n, T, 2) vectors by -th (n,): a vector at angle th lands on +x."""
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], axis=-1)


def prepare(d, rotate=True):
    """Loader split -> model arrays. cv: causal CV of baselines.py (m); th: direction of its velocity;
    x: obs with every vector feature rotated by -th, so the input does not depend on the compass course."""
    if list(d["features"]) != LD.FEATURES:
        raise AssertionError("το phase3 θέλει την αιτιατή παρατήρηση του loader (FEATURES)")
    m = d["meta"]
    cv = BL.cv_predict(d["pos"], last=m["last_avail"].to_numpy(), age_s=m["p_age_s"].to_numpy())
    v = cv[:, 1] - cv[:, 0]
    th = np.arctan2(v[:, 1], v[:, 0]) if rotate else np.zeros(len(cv))
    x = d["obs"].astype(np.float64)
    for p in VEC:
        x[..., p] = rot(x[..., p], th)
    return {"x": x.astype(np.float32), "cv": cv, "th": th, "fut": d["fut"]}


def target(p, use_cv=True):
    """What the network must output: (future - CV) in the rotated frame, in RES_SCALE units."""
    return (rot(p["fut"].astype(np.float64) - p["cv"] * use_cv, p["th"]) / RES_SCALE).astype(np.float32)


def predict(f, p, use_cv=True):
    """f: x -> network output (n, PRED, 2). Returns positions in metres in the loader frame."""
    return rot(np.asarray(f(p["x"]), np.float64) * RES_SCALE, -p["th"]) + p["cv"] * use_cv


# ---------------------------------------------------------------------------
def make_net(arch, n_feat=len(LD.FEATURES), pred=LD.PRED):
    import torch
    import torch.nn as nn
    H = arch["hidden"]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if arch["encoder"] == "gru":
                self.enc = nn.GRU(n_feat, H, num_layers=2, batch_first=True, dropout=0.1)
            else:
                self.inp = nn.Conv1d(n_feat, H, 1)
                self.convs = nn.ModuleList(nn.Conv1d(H, H, 3, dilation=k) for k in DILATIONS)
            self.head = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, pred * 2))
            nn.init.zeros_(self.head[-1].weight)      # untrained network = exactly the CV baseline
            nn.init.zeros_(self.head[-1].bias)

        def forward(self, x):
            if arch["encoder"] == "gru":
                h = self.enc(x)[1][-1]
            else:
                h = self.inp(x.transpose(1, 2))
                for k, conv in zip(DILATIONS, self.convs):
                    h = h + torch.relu(conv(nn.functional.pad(h, (2 * k, 0))))   # left padding only
                h = h[:, :, -1]
            return self.head(h).view(len(x), pred, 2)
    return Net()


def net_fn(model, device, bs=8192):
    import torch

    def f(x):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.from_numpy(x[a:a + bs]).to(device)).cpu().numpy()
                                   for a in range(0, len(x), bs)])
    return f


def train(model, tr, es, a, device):
    """Protocol of spec §12, identical to baselines.train_lstm: early stopping and ReduceLROnPlateau on the
    early-stop ADE (m) with improvement > 0.1 m, return to the best epoch, stop reason recorded."""
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    use_cv = not a.no_cv
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=a.lr_patience,
                                                       threshold=0.1, threshold_mode="abs")
    X, Y = torch.from_numpy(tr["x"]), torch.from_numpy(target(tr, use_cv))
    best, best_state, bad, best_ep, curve, stop = np.inf, None, 0, 0, [], "max_epochs"
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(X), generator=g)
        for b in range(0, len(X), a.batch):
            i = perm[b:b + a.batch]
            e = torch.linalg.norm(model(X[i].to(device)) - Y[i].to(device), dim=-1)
            loss = torch.nn.functional.huber_loss(e, torch.zeros_like(e), delta=HUBER_M / RES_SCALE)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(i)
        ade_es = BL.errors(predict(net_fn(model, device), es, use_cv), es["fut"])[0].mean()
        lr = opt.param_groups[0]["lr"]
        sched.step(ade_es)
        curve.append({"epoch": ep, "train_loss": round(tot / len(X), 6), "es_ADE": round(float(ade_es), 2), "lr": lr})
        print(f"    epoch {ep:>3}: train loss {tot / len(X):.5f} | early-stop ADE {ade_es:.1f} m | "
              f"lr {lr:.1e} | {time.time() - t0:.0f} s")
        if ade_es < best - 0.1:
            best, best_ep, bad = ade_es, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                stop = "patience"
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if stop == "max_epochs":
        print(f"  ΠΡΟΣΟΧΗ: δεν συνέκλινε (όριο {a.epochs} epochs) — το run δεν χρησιμοποιείται για απόφαση")
    return model, {"best_early_stop_ADE": round(float(best), 2), "epochs_run": ep, "best_epoch": best_ep,
                   "stop": stop, "curve": curve}


def save_checkpoint(path, model, arch, args):
    import torch
    n = sum(p.numel() for p in model.parameters())
    torch.save({"state": model.state_dict(), "arch": arch, "args": args, "n_params": n,
                "version": PHASE3_VERSION}, path)
    return n


def load_checkpoint(path, device):
    """Model rebuilt from the architecture saved in the checkpoint (CLI flags are not consulted)."""
    import torch
    ck = torch.load(path, map_location=device)
    model = make_net(ck["arch"]).to(device)
    model.load_state_dict(ck["state"])
    return model, ck


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_build", required=True)
    ap.add_argument("--manifest", default="vessel")
    ap.add_argument("--eval_build", default=None, help="προεπιλογή: το train_build")
    ap.add_argument("--eval_manifest", default=None, help="προεπιλογή: το manifest")
    ap.add_argument("--eval_split", default="val")
    ap.add_argument("--load_model", default=None, help="checkpoint: μόνο αξιολόγηση, αρχιτεκτονική από το αρχείο")
    ap.add_argument("--encoder", choices=["tcn", "gru"], default="tcn")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no_cv", action="store_true", help="δοκιμή: απευθείας θέσεις αντί για διόρθωση πάνω στο CV")
    ap.add_argument("--no_rotate", action="store_true", help="δοκιμή: χωρίς περιστροφή στην πορεία")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr_patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default="reports/phase3")
    a = ap.parse_args(argv)
    eb, em = a.eval_build or a.train_build, a.eval_manifest or a.manifest
    if not a.load_model and a.eval_split != "val":
        sys.exit("η εκπαίδευση αξιολογεί μόνο σε val· τα test θέλουν --load_model (χωριστό βήμα)")
    import torch
    dev = BL.device_info(torch)
    device = dev["device"]
    if a.load_model:
        model, ck = load_checkpoint(a.load_model, device)
        arch, seed = ck["arch"], ck["args"]["seed"]
    else:
        arch, seed = {k: getattr(a, k) for k in ARCH}, a.seed
    use_cv = not arch["no_cv"]
    tag = a.tag or (f"m0_{Path(a.train_build).name}_{Path(eb).name}_{em}_{a.eval_split}_s{seed}"
                    + ("_gru" if arch["encoder"] == "gru" else "") + (f"_h{arch['hidden']}" if arch["hidden"] != 128 else "")
                    + ("_nocv" if arch["no_cv"] else "") + ("_norot" if arch["no_rotate"] else ""))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[phase3] {tag} | συσκευή: {device}" + (f" ({dev['gpu']})" if dev["gpu"] else " — ΧΩΡΙΣ GPU"))

    sd_e = BL.shore_dir_of(eb)
    print(f"  αξιολόγηση: {eb} / {em} / {a.eval_split} | στάδιο Γ: {sd_e}")
    x_ev = LD.load_rows(eb, sd_e)
    ev = LD.load_split(eb, em, a.eval_split, x=x_ev)
    res = {"tag": tag, "versions": {"phase3": PHASE3_VERSION, "baselines": BL.BASELINES_VERSION,
                                    "loader": LD.LOADER_VERSION, "python": sys.version.split()[0],
                                    "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
           "args": {**vars(a), "seed": seed}, "arch": arch, "device": dev,
           "eval": {"build": eb, "manifest": em, "split": a.eval_split, "shore_dir": sd_e}}
    try:
        res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        res["commit"] = None

    if a.load_model:
        res["model_path"], res["n_params"] = a.load_model, ck["n_params"]
    else:
        tr = LD.load_split(a.train_build, a.manifest, "train", x=BL.train_rows(a.train_build, eb, x_ev))
        esm = BL._hash01(tr["meta"]["mmsi"].to_numpy()) < 0.1          # same early-stopping vessels as the LSTM
        p = prepare(tr, not arch["no_rotate"])
        tr_p, es_p = ({k: v[m] for k, v in p.items()} for m in (~esm, esm))
        print(f"  εκπαίδευση: {len(tr_p['x']):,} παράθυρα | early stopping: {len(es_p['x']):,}")
        model, info = train(make_net(arch).to(device), tr_p, es_p, a, device)
        res["train"] = {**info, "n_train": int(len(tr_p["x"])), "n_early_stop": int(len(es_p["x"]))}
        res["model_path"] = str(out / f"{tag}.pt")
        res["n_params"] = save_checkpoint(res["model_path"], model, arch, res["args"])
    print(f"  παράμετροι: {res['n_params']:,}")

    pred = predict(net_fn(model, device), prepare(ev, not arch["no_rotate"]), use_cv)
    ade, fde = BL.errors(pred, ev["fut"])
    res["metrics"] = BL.metrics(ade, fde, ev["meta"])
    BL.print_metrics(tag, res["metrics"])
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[phase3] → {out / (tag + '.json')}")
    return res


if __name__ == "__main__":
    main()
