#!/usr/bin/env python3
"""
a6_noise_tol.py — sensitivity of A6/A7 to the position-noise tolerance (despike.pos_noise_m).

Measurement only: stage A is not changed. Re-runs cleaning up to segmentation (v5_visual.scan)
with the config of an existing build, once per tolerance, and reports spikes, jump cuts and the
share of cuts at small Δt (the signature of GPS noise). Saves PNG pages of cuts that disappear
and spikes that are no longer removed at the largest tolerance.

    python a6_noise_tol.py --raw_dir data/2025_03 --month 2025-03 \
        --build dataset/socal_2025-03_70a880d6 --out reports/a6_noise_tol
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

import v5_visual as V

DT_BINS = [0, 10, 30, 60, 180, np.inf]


def cut_dt(rep):
    """Δt (s) from the previous kept report of the same vessel, for every jump cut."""
    k = rep[~rep["spike"]]
    dt = k["t"].diff().where(k["mmsi"].eq(k["mmsi"].shift()))
    return dt[k["cut_before"]].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--tols", default="0,10,30")
    ap.add_argument("--out", default="reports/a6_noise_tol")
    ap.add_argument("--n_show", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tols = [float(x) for x in a.tols.split(",")]

    cfg0 = json.load(open(f"{a.build}/build_meta.json", encoding="utf-8"))["config"]
    print(f"[A6] build {a.build} ({cfg0['config_version']}), ανοχές {tols}")
    raw = V.load_raw(a.raw_dir, a.month, cfg0)
    reps, rows = {}, []
    for tol in tols:
        cfg = copy.deepcopy(cfg0)
        cfg["despike"]["pos_noise_m"] = tol
        rep = V.scan(raw, cfg)
        reps[tol] = rep if tol == tols[0] else rep[["mmsi", "t", "spike", "cut_before"]]
        dt = cut_dt(rep)
        hist = np.histogram(dt, DT_BINS)[0]
        row = {"tol_m": tol, "spikes": int(rep["spike"].sum()), "cuts": len(dt),
               "cut_vessels": int(rep.loc[rep["cut_before"], "mmsi"].nunique()),
               "cuts_dt_lt30": int((dt < 30).sum()),
               "cuts_dt_lt30_pct": round(100 * float(np.mean(dt < 30)), 1) if len(dt) else None}
        row.update({f"dt_{lo:g}-{hi:g}s": int(n) for lo, hi, n in zip(DT_BINS[:-1], DT_BINS[1:], hist)})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    # visual check: what the largest tolerance stops cutting / removing (positions from tol 0)
    del raw
    os.makedirs(a.out, exist_ok=True)
    full0 = reps[tols[0]]
    hi = reps[tols[-1]].set_index(["mmsi", "t"])
    lo = full0.set_index(["mmsi", "t"])
    grid = V.load_grid(a.build)
    for col, stem in (("cut_before", "cut_gone"), ("spike", "spike_kept")):
        gone = lo.index[lo[col].to_numpy() & ~hi[col].reindex(lo.index, fill_value=False).to_numpy()]
        print(f"{stem}: {len(gone):,} (υπάρχουν στο {tols[0]:g} m, όχι στο {tols[-1]:g} m)")
        if len(gone):
            pick = pd.DataFrame(list(gone), columns=["mmsi", "t"]).sample(
                min(a.n_show, len(gone)), random_state=a.seed).sort_values(["mmsi", "t"])
            cases = [(*V.window(full0, grid, r.mmsi, r.t, 600), f"{stem} {r.mmsi} {V.fmt_t(r.t)}")
                     for r in pick.itertuples()]
            for p in V.pages(cases, a.out, stem):
                print("  ", p)
    pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False)


if __name__ == "__main__":
    main()
