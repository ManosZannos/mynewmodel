#!/usr/bin/env python3
"""
v8_spikes.py — V8: A6 despike of stage A vs docs/v8_reference.py on the SAME post-A5 reports.

Both despikes run on the reports produced by v5_visual.scan (identical funnel up to A5), so every
difference is an A6 difference. Prints counts (both / only ours / only V8 / blocked in V8), splits the
differences by SOG and by deviation from the predicted position, counts how many are explained by the
two chord tolerances (AMB-08), and draws PNG samples:

  only_v8_*.png  only_ours_*.png  blocked_v8_*.png   (red × ours, purple + V8)

    python v8_spikes.py --raw_dir data/2025_03 --month 2025-03 --days 1 2 3 \
        --build dataset_v8ref/socal_2025-03_b19c45d7 --out reports/v8_spikes
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import v5_visual as V5

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs"))
import v8_reference as R  # noqa: E402


def add_v8(rep):
    """spike_v8, blocked_v8 from the reference despike (rep is contiguous per mmsi, sorted by t)."""
    sp, bl = [], []
    for _, g in rep.groupby("mmsi", sort=False):
        keep, blocked = R.despike(g["t"].to_numpy(np.int64), g["lat"].to_numpy(float), g["lon"].to_numpy(float),
                                  g["sog"].to_numpy(float), g["cog"].to_numpy(float), R.CFG)
        sp.append(~keep)
        bl.append(blocked)
    return rep.assign(spike_v8=np.concatenate(sp), blocked_v8=np.concatenate(bl))


def add_triplet(rep):
    """err = distance of i from the position predicted from i-1, i+1 (as in V8: Hermite / chord);
    tol_ours = max(150, 0.25·mean(SOG_a, SOG_c)·Δt_ac), tol_v8 = max(150, 0.25·max(SOG_a,b,c)·min(Δt1, Δt2)).
    shortcut: neighbours are the raw i-1, i+1, not the current ones after earlier removals; fine for
    isolated spikes, approximate inside chains."""
    n = len(rep)
    m, t = rep["mmsi"].to_numpy(), rep["t"].to_numpy(float)
    la, lo = rep["lat"].to_numpy(float), rep["lon"].to_numpy(float)
    sg, cg = rep["sog"].to_numpy(float), rep["cog"].to_numpy(float)
    err, tol_o, tol_v = (np.full(n, np.nan) for _ in range(3))
    b = np.arange(1, n - 1)
    b = b[(m[b - 1] == m[b]) & (m[b + 1] == m[b])]
    a, c = b - 1, b + 1
    dt1, dt2 = t[b] - t[a], t[c] - t[b]
    dt13 = dt1 + dt2
    xb, yb = R.enu(la[a], lo[a], la[b], lo[b])
    xc, yc = R.enu(la[a], lo[a], la[c], lo[c])
    f = dt1 / dt13
    px, py = f * xc, f * yc
    cv = ~np.isnan(cg) & ~(sg < R.CFG["cog_valid_min_sog"])
    h = (sg[a] >= R.CFG["a6_hermite_min_sog"]) & (sg[c] >= R.CFG["a6_hermite_min_sog"]) & cv[a] & cv[c]
    px[h], py[h] = R.hermite_xy(f[h], dt13[h], xc[h], yc[h], sg[a][h], cg[a][h], sg[c][h], cg[c][h])
    err[b] = np.hypot(xb - px, yb - py)
    chord_kn = np.hypot(xc, yc) / dt13 / R.KN
    s_mean = np.where(np.isnan(sg[a]), sg[c], np.where(np.isnan(sg[c]), sg[a], (sg[a] + sg[c]) / 2))
    s_mean = np.where(np.isnan(s_mean), chord_kn, s_mean)
    s_max = R.nanmax_n(sg[a], sg[b], sg[c])
    s_max = np.where(np.isnan(s_max), chord_kn, s_max)
    lo_m, k = R.CFG["a6_chord_min_m"], R.CFG["a6_chord_frac"]
    tol_o[b] = np.maximum(lo_m, k * s_mean * R.KN * dt13)
    tol_v[b] = np.maximum(lo_m, k * s_max * R.KN * np.minimum(dt1, dt2))
    return rep.assign(err=err, tol_ours=tol_o, tol_v8=tol_v)


def summary(rep):
    sp, v8 = rep["spike"].to_numpy(), rep["spike_v8"].to_numpy()
    cls = {"both": sp & v8, "only_ours": sp & ~v8, "only_v8": ~sp & v8}
    out = {k: int(v.sum()) for k, v in cls.items()}
    out["blocked_v8"] = int(rep["blocked_v8"].sum())
    sog_bin = pd.cut(rep["sog"], [-np.inf, 10, 20, np.inf], right=False, labels=["<10", "10-20", ">=20"])
    err_bin = pd.cut(rep["err"], [0, 150, 300, 1000, np.inf], right=False,
                     labels=["<150", "150-300", "300-1000", ">=1000"])
    amb08 = (rep["err"] > rep["tol_v8"]) & (rep["err"] <= rep["tol_ours"])
    for k in ("only_ours", "only_v8"):
        mk = cls[k]
        out[k + "_by_sog"] = {str(i): int(v) for i, v in sog_bin[mk].value_counts(dropna=False).sort_index().items()}
        out[k + "_by_err_m"] = {str(i): int(v) for i, v in err_bin[mk].value_counts(dropna=False).sort_index().items()}
        out[k + "_tol_v8_lt_err_le_tol_ours"] = int((mk & amb08).sum())
        out[k + "_vessels"] = int(rep.loc[mk, "mmsi"].nunique())
    return out, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--days", type=int, nargs="+", required=True)
    ap.add_argument("--build", required=True, help="stage A build of the same days (grid for the plots)")
    ap.add_argument("--out", default="reports/v8_spikes")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meta = json.load(open(f"{a.build}/build_meta.json", encoding="utf-8"))
    cfg = meta["config"]
    print(f"[v8_spikes] {cfg['config_version']} vs {R.VERSION}, μέρες {a.days}")
    rep = V5.scan(V5.load_raw(a.raw_dir, a.month, cfg, days=set(a.days)), cfg)
    rep = add_triplet(add_v8(rep))
    out, cls = summary(rep)
    out.update(config_version=cfg["config_version"], v8_version=R.VERSION, days=a.days,
               build_despike=meta["funnel"]["dropped"].get("despike"), scan_despike=int(rep["spike"].sum()))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    grid = V5.load_grid(a.build)
    made = []
    for stem, mask, n in (("only_v8", cls["only_v8"], a.n), ("only_ours", cls["only_ours"], a.n),
                          ("blocked_v8", rep["blocked_v8"].to_numpy(), a.n // 2)):
        d = rep[mask]
        if d.empty:
            continue
        smp = d.sample(min(n, len(d)), random_state=a.seed).sort_values(["mmsi", "t"])
        cases = [(*V5.window(rep, grid, r.mmsi, r.t, 600),
                  f"{stem} {r.mmsi} {V5.fmt_t(r.t)} sog {r.sog:.1f}\n"
                  f"err {r.err:.0f} m, tol ours {r.tol_ours:.0f} / V8 {r.tol_v8:.0f} m") for r in smp.itertuples()]
        made += V5.pages(cases, a.out, stem)
    print(f"\n[v8_spikes] {len(made)} σελίδες στο {a.out} (κόκκινο × εμείς, μοβ + V8)")
    for p in made:
        print("  ", p)


if __name__ == "__main__":
    main()
