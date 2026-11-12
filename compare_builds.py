#!/usr/bin/env python3
"""
compare_builds.py — σύγκριση δύο builds του σταδίου Α (π.χ. stageA-v3.5 έναντι v3.6).

    python compare_builds.py dataset/socal_2025-03_6dcfa1ee dataset/socal_2025-03_70a880d6

Ελέγχει (1) ότι το σύνολο (mmsi, t) είναι ίδιο, (2) ποιες στήλες άλλαξαν και σε πόσες γραμμές,
(3) τη μετατόπιση lat/lon σε μέτρα και σε πόσα segments/πλοία. Γράφει <new>/compare_<old-hash>.json.
Στήλες που υπάρχουν μόνο στο ένα build αναφέρονται, δεν συγκρίνονται.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0
KEY = ["mmsi", "t"]


def compare_days(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Compare two frames of the same day. Returns counts; per-row details are only kept as aggregates."""
    old = old.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    new = new.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    k_old = pd.MultiIndex.from_frame(old[KEY])
    k_new = pd.MultiIndex.from_frame(new[KEY])
    only_old, only_new = int((~k_old.isin(k_new)).sum()), int((~k_new.isin(k_old)).sum())
    res = {"rows_old": len(old), "rows_new": len(new), "only_old": only_old, "only_new": only_new,
           "changed": {}, "disp_m": np.zeros(0), "changed_vessels": set(), "changed_segments": set()}
    if only_old or only_new:                 # compare the common (mmsi, t) rows; both frames are sorted by KEY
        old, new = old[k_old.isin(k_new)].reset_index(drop=True), new[k_new.isin(k_old)].reset_index(drop=True)
    changed_any = np.zeros(len(old), bool)
    for c in [c for c in old.columns if c in new.columns and c not in KEY]:
        a, b = old[c], new[c]
        if c in ("lat", "lon"):
            continue
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            diff = ~((a == b) | (a.isna() & b.isna())).to_numpy()
        else:
            diff = (a.astype(object) != b.astype(object)).to_numpy()
        if diff.any():
            res["changed"][c] = int(diff.sum())
            changed_any |= diff
    if {"lat", "lon"} <= set(old.columns) & set(new.columns):
        lat0 = np.radians((old["lat"].to_numpy() + new["lat"].to_numpy()) / 2)
        dy = np.radians(new["lat"].to_numpy() - old["lat"].to_numpy()) * R_EARTH
        dx = np.radians(new["lon"].to_numpy() - old["lon"].to_numpy()) * R_EARTH * np.cos(lat0)
        d = np.hypot(dx, dy)
        moved = d > 1e-3                                     # 1 mm
        res["disp_m"] = d[moved]
        if moved.any():
            res["changed"]["lat/lon"] = int(moved.sum())
        changed_any |= moved
    if changed_any.any():
        res["changed_vessels"] = set(old.loc[changed_any, "mmsi"].unique().tolist())
        if "seg_id" in old.columns:
            res["changed_segments"] = set(old.loc[changed_any, "seg_id"].unique().tolist())
    res["changed_rows"] = int(changed_any.sum())
    res["old_cog_eq_360"] = int((old.get("cog", pd.Series(dtype=float)) >= 360).sum())
    res["old_heading_eq_360"] = int((old.get("heading", pd.Series(dtype=float)) >= 360).sum())
    return res


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 2:
        sys.exit(__doc__)
    d_old, d_new = Path(a[0]), Path(a[1])
    f_old = {Path(f).name: f for f in glob.glob(str(d_old / "day_*.parquet"))}
    f_new = {Path(f).name: f for f in glob.glob(str(d_new / "day_*.parquet"))}
    if set(f_old) != set(f_new):
        sys.exit(f"διαφορετικές μέρες: μόνο old {sorted(set(f_old) - set(f_new))}, μόνο new {sorted(set(f_new) - set(f_old))}")
    tot = {"rows_old": 0, "rows_new": 0, "only_old": 0, "only_new": 0, "changed_rows": 0,
           "old_cog_eq_360": 0, "old_heading_eq_360": 0}
    changed, disp, vessels, segs = {}, [], set(), set()
    cols_old = cols_new = None
    for name in sorted(f_old):
        o, n = pd.read_parquet(f_old[name]), pd.read_parquet(f_new[name])
        cols_old, cols_new = list(o.columns), list(n.columns)
        r = compare_days(o, n)
        for k in tot:
            tot[k] += r.get(k, 0)
        for c, v in r["changed"].items():
            changed[c] = changed.get(c, 0) + v
        disp.append(r["disp_m"])
        vessels |= r["changed_vessels"]
        segs |= r["changed_segments"]
        print(f"  {name}: changed_rows={r.get('changed_rows', 0):,}", flush=True)
    disp = np.concatenate(disp) if disp else np.zeros(0)
    out = {"old": str(d_old), "new": str(d_new), **tot, "same_keys": tot["only_old"] == 0 and tot["only_new"] == 0,
           "changed_columns": changed, "changed_vessels": len(vessels), "changed_segments": len(segs),
           "columns_only_old": sorted(set(cols_old) - set(cols_new)),
           "columns_only_new": sorted(set(cols_new) - set(cols_old))}
    if len(disp):
        out["disp_m"] = {"n": int(len(disp)), "p50": float(np.percentile(disp, 50)),
                         "p90": float(np.percentile(disp, 90)), "p99": float(np.percentile(disp, 99)),
                         "max": float(disp.max())}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (d_new / f"compare_{d_old.name.split('_')[-1]}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.exit(0 if out["same_keys"] else 1)


if __name__ == "__main__":
    main()
