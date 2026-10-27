#!/usr/bin/env python3
"""
stageB_radius.py — μέτρηση της ακτίνας γειτόνων R από τα δεδομένα (στάδιο Β, §4.2).

Για ζεύγη κινούμενων πλοίων (SOG ≥ --min_sog) στη στιγμή t0 (= τελευταίο λεπτό παρατήρησης):
  α) προβλεπόμενες συναντήσεις: CPA < --cpa_nm και 0 ≤ TCPA ≤ --horizon_min (από θέση/ταχύτητα στο t0)
  β) πραγματικές συναντήσεις: ελάχιστη απόσταση < --cpa_nm μέσα στα επόμενα --horizon_min λεπτά
και τυπώνει την κατανομή της απόστασης τους στο t0. R = η απόσταση που καλύπτει το 95%.

    python stageB_radius.py dataset/socal_2025-03_210b4127
    python stageB_radius.py dataset/houston_2025-03_6e340c21 --every 10

Χρησιμοποιεί μόνο τα δεδομένα θέσης/ταχύτητας του build (καμία πληροφορία μοντέλου).
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0
NM = 1852.0
COLS = ["mmsi", "t", "lat", "lon", "sog", "vx_sog", "vy_sog", "vx_pos", "vy_pos",
        "STATIONARY", "IN_CORE", "vessel_cat"]


def load(d):
    files = sorted(glob.glob(f"{d}/day_*.parquet"))
    if files:
        x = pd.concat((pd.read_parquet(f, columns=COLS) for f in files), ignore_index=True)
    else:
        files = sorted(glob.glob(f"{d}/day_*.csv.gz"))
        x = pd.concat((pd.read_csv(f, usecols=COLS) for f in files), ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True)
    x["tm"] = ((x["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    return x


def encounters(x, min_sog=0.5, cpa_nm=1.0, horizon_min=10, every=5, max_range_km=25.0):
    vx = x["vx_sog"].fillna(x["vx_pos"]).to_numpy()
    vy = x["vy_sog"].fillna(x["vy_pos"]).to_numpy()
    lat, lon = x["lat"].to_numpy(), x["lon"].to_numpy()
    mmsi, tm = x["mmsi"].to_numpy(), x["tm"].to_numpy()
    moving = (x["sog"].fillna(0).to_numpy() >= min_sog) & (x["STATIONARY"].to_numpy() == 0)
    core = x["IN_CORE"].to_numpy() == 1
    key = pd.MultiIndex.from_arrays([mmsi, tm])
    lookup = pd.Series(np.arange(len(x)), index=key)

    t_all = np.unique(tm)
    t_sel = t_all[(t_all - t_all.min()) % every == 0]
    order = np.argsort(tm, kind="stable")
    tm_sorted = tm[order]
    rows = []
    for t0 in t_sel:
        lo, hi = np.searchsorted(tm_sorted, [t0, t0 + 1])
        idx = order[lo:hi]
        idx = idx[moving[idx] & ~np.isnan(vx[idx]) & ~np.isnan(vy[idx])]
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        a, b = idx[i], idx[j]
        a_core = core[a] | core[b]
        latm = np.radians((lat[a] + lat[b]) / 2)
        dx = np.radians(lon[b] - lon[a]) * R_EARTH * np.cos(latm)
        dy = np.radians(lat[b] - lat[a]) * R_EARTH
        d0 = np.hypot(dx, dy)
        keep = a_core & (d0 <= max_range_km * 1000)
        if not keep.any():
            continue
        a, b, dx, dy, d0 = a[keep], b[keep], dx[keep], dy[keep], d0[keep]
        dvx, dvy = vx[b] - vx[a], vy[b] - vy[a]
        v2 = dvx ** 2 + dvy ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            tcpa = np.where(v2 > 1e-9, -(dx * dvx + dy * dvy) / v2, 0.0)
        cpa = np.hypot(dx + dvx * tcpa, dy + dvy * tcpa)
        # πραγματική ελάχιστη απόσταση στα επόμενα λεπτά
        dmin = d0.copy()
        for k in range(1, horizon_min + 1):
            ia = lookup.reindex(pd.MultiIndex.from_arrays([mmsi[a], np.full(len(a), t0 + k)])).to_numpy()
            ib = lookup.reindex(pd.MultiIndex.from_arrays([mmsi[b], np.full(len(b), t0 + k)])).to_numpy()
            ok = ~np.isnan(ia) & ~np.isnan(ib)
            if not ok.any():
                continue
            ia_, ib_ = ia[ok].astype(int), ib[ok].astype(int)
            lm = np.radians((lat[ia_] + lat[ib_]) / 2)
            ddx = np.radians(lon[ib_] - lon[ia_]) * R_EARTH * np.cos(lm)
            ddy = np.radians(lat[ib_] - lat[ia_]) * R_EARTH
            dmin[ok] = np.minimum(dmin[ok], np.hypot(ddx, ddy))
        rows.append(pd.DataFrame({
            "t0": t0, "d0": d0, "cpa": cpa, "tcpa_s": tcpa, "dmin": dmin,
            "cat_a": x["vessel_cat"].to_numpy()[a], "cat_b": x["vessel_cat"].to_numpy()[b]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def report(e, cpa_nm, horizon_min):
    thr = cpa_nm * NM
    pred = e[(e["cpa"] < thr) & (e["tcpa_s"] >= 0) & (e["tcpa_s"] <= horizon_min * 60)]
    real = e[e["dmin"] < thr]
    q = [0.5, 0.75, 0.9, 0.95, 0.99]
    print(f"ζεύγη κινούμενων (≤ 25 km) σε {e['t0'].nunique():,} στιγμές: {len(e):,}")
    for name, g in (("α) προβλεπόμενες (CPA)", pred), ("β) πραγματικές (ελάχ. απόσταση)", real)):
        if g.empty:
            print(f"{name}: καμία")
            continue
        qs = (g["d0"].quantile(q) / 1000).round(2).tolist()
        print(f"{name}: {len(g):,} ζεύγη | απόσταση στο t0 (km) p50/p75/p90/p95/p99 = {qs}")
        for R in (3, 5, 6, 10):
            print(f"     R = {R:>2} km καλύπτει {100 * (g['d0'] <= R * 1000).mean():.1f}%")
    both = real.merge(pred[["t0", "d0"]], on=["t0", "d0"], how="left", indicator=True)
    if len(real):
        print(f"πραγματικές που ΔΕΝ προβλέπονταν από το CPA στο t0: {100 * (both['_merge'] == 'left_only').mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--min_sog", type=float, default=0.5)
    ap.add_argument("--cpa_nm", type=float, default=1.0)
    ap.add_argument("--horizon_min", type=int, default=10)
    ap.add_argument("--every", type=int, default=5, help="δειγματοληψία στιγμών t0 ανά Ν λεπτά")
    a = ap.parse_args()
    x = load(a.build_dir)
    print(f"[R] {a.build_dir}: {len(x):,} γραμμές")
    e = encounters(x, a.min_sog, a.cpa_nm, a.horizon_min, a.every)
    report(e, a.cpa_nm, a.horizon_min)


if __name__ == "__main__":
    main()
