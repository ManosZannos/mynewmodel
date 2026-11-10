#!/usr/bin/env python3
"""
v8_compare.py — V8: compare a stage A build with the independent reference (docs/v8_reference.py).

    python v8_compare.py dataset_v8ref/socal_2025-03_<hash> dataset_v8/socal_2025-03 --out reports/v8_compare.json

Unlike compare_builds.py (same code, exact equality) this aligns two different implementations:
t as minutes from the start of the month, seg_id compared as a partition, floats with tolerances,
angles circularly. Every difference is then explained as a bug or a spec ambiguity (AMB-xx).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0
FLOAT_TOL = {"sog": 0.01, "cog": 0.1, "heading": 0.1, "vx_pos": 0.01, "vy_pos": 0.01,
             "P_SOG": 1e-6, "P_COG": 1e-6, "N_SOG": 1e-6, "P_LAT": 1e-9, "P_LON": 1e-9,
             "INTERP_SPAN": 0.5, "R_AFTER_S": 0.5, "P_AGE_S": 0.5}
ANGLES = {"cog", "heading", "P_COG"}
EXACT = ["COG_VALID", "HDG_SRC", "vessel_cat", "transceiver", "GAP_IMP", "STATIONARY",
         "QC_SOG", "QC_COG", "IN_CORE", "MMSI_QC"]
N_EX = 5


def load(d):
    files = sorted(glob.glob(os.path.join(d, "day_*.parquet")))
    if not files:
        raise SystemExit(f"κανένα day_*.parquet στο {d}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def to_minutes(t):
    """Datetime (UTC, minute grid) -> int minutes from the start of its month; ints pass through."""
    if not pd.api.types.is_datetime64_any_dtype(t):
        return t.astype(np.int64)
    t0 = t.min()
    start = pd.Timestamp(year=t0.year, month=t0.month, day=1, tz=t0.tz)
    return ((t - start) // pd.Timedelta(minutes=1)).astype(np.int64)


def _examples(m, mask, c):
    ex = m.loc[mask, ["mmsi", "t", f"{c}_ref", f"{c}_v8"]].head(N_EX)
    return [[None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r] for r in ex.to_numpy()]


def compare(ref, v8):
    ref, v8 = ref.assign(t=to_minutes(ref["t"])), v8.assign(t=to_minutes(v8["t"]))
    m = ref.merge(v8, on=["mmsi", "t"], how="outer", suffixes=("_ref", "_v8"), indicator=True)
    side = m["_merge"]
    res = {"rows_ref": len(ref), "rows_v8": len(v8), "common": int((side == "both").sum()),
           "only_ref": int((side == "left_only").sum()), "only_v8": int((side == "right_only").sum())}
    per = (m.loc[side != "both"].groupby("mmsi")["_merge"].value_counts().unstack(fill_value=0)
           .reindex(columns=["left_only", "right_only"], fill_value=0))
    per = per.assign(n=per.sum(axis=1)).sort_values("n", ascending=False)
    res["only_vessels"] = len(per)
    res["only_top"] = [[int(i), int(r.left_only), int(r.right_only)] for i, r in per.head(20).iterrows()]

    b = m.loc[side == "both"].reset_index(drop=True)
    lat0 = np.radians((b["lat_ref"] + b["lat_v8"]) / 2)
    d = np.hypot(np.radians(b["lon_v8"] - b["lon_ref"]) * R_EARTH * np.cos(lat0),
                 np.radians(b["lat_v8"] - b["lat_ref"]) * R_EARTH)
    res["pos_m"] = {"gt_1m": int((d > 1).sum()), "p99": round(float(np.quantile(d, 0.99)), 3) if len(d) else None,
                    "max": round(float(d.max()), 3) if len(d) else None}
    b["pos_m_ref"], b["pos_m_v8"] = d, 0.0
    res["pos_m"]["examples"] = _examples(b, d > 1, "pos_m")

    cols = {}
    for c, tol in FLOAT_TOL.items():
        if f"{c}_ref" not in b or f"{c}_v8" not in b:
            cols[c] = "λείπει από " + ("ref" if f"{c}_ref" not in b else "v8")
            continue
        x, y = b[f"{c}_ref"].astype(float), b[f"{c}_v8"].astype(float)
        diff = (y - x).abs()
        if c in ANGLES:
            diff = np.minimum(diff, 360 - diff)
        bad = (diff > tol) | (x.isna() != y.isna())
        cols[c] = {"diff": int(bad.sum()), "nan_mismatch": int((x.isna() != y.isna()).sum()),
                   "examples": _examples(b, bad, c)}
    for c in EXACT:
        if f"{c}_ref" not in b or f"{c}_v8" not in b:
            cols[c] = "λείπει από " + ("ref" if f"{c}_ref" not in b else "v8")
            continue
        x, y = b[f"{c}_ref"].astype(str), b[f"{c}_v8"].astype(str)
        bad = x != y
        cols[c] = {"diff": int(bad.sum()), "examples": _examples(b, bad, c)}
    res["columns"] = cols

    # seg_id as a partition: a segment of one side that maps to > 1 segment of the other is a split
    pairs = b[["mmsi", "seg_id_ref", "seg_id_v8"]].drop_duplicates()
    split_ref = pairs.groupby(["mmsi", "seg_id_ref"]).size()
    split_v8 = pairs.groupby(["mmsi", "seg_id_v8"]).size()
    res["segments"] = {"ref": int(len(split_ref)), "v8": int(len(split_v8)),
                       "ref_split_in_v8": int((split_ref > 1).sum()), "v8_split_in_ref": int((split_v8 > 1).sum()),
                       "examples_ref_split": [list(map(str, k)) for k in split_ref[split_ref > 1].index[:N_EX]]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_build")
    ap.add_argument("v8_build")
    ap.add_argument("--out", default="reports/v8_compare.json")
    a = ap.parse_args()
    res = compare(load(a.ref_build), load(a.v8_build))
    fr = json.load(open(os.path.join(a.ref_build, "build_meta.json"), encoding="utf-8"))["funnel"]
    fv = json.load(open(os.path.join(a.v8_build, "funnel.json"), encoding="utf-8"))
    res["funnel"] = {"ref": {"dropped": fr["dropped"], "implied_cuts": fr.get("implied_cuts"),
                             "forced_cuts": fr.get("forced_cuts"),
                             "segments": fr["segments"], "mmsi_qc": len(fr["mmsi_qc_flagged"])},
                     "v8": {"reports": fv["reports"], "grid": fv["grid"], "counters": fv["counters"],
                            "mmsi_qc": fv["mmsi_qc"]["n_flagged"]}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    print(f"γραμμές ref {res['rows_ref']:,} | v8 {res['rows_v8']:,} | κοινές {res['common']:,} | "
          f"μόνο ref {res['only_ref']:,} | μόνο v8 {res['only_v8']:,} ({res['only_vessels']} πλοία)")
    print(f"θέση > 1 m: {res['pos_m']['gt_1m']:,} (p99 {res['pos_m']['p99']} m, max {res['pos_m']['max']} m)")
    for c, r in res["columns"].items():
        print(f"  {c:12s} {r if isinstance(r, str) else format(r['diff'], ',')}")
    print("segments:", {k: v for k, v in res["segments"].items() if k != "examples_ref_split"})
    print("→", a.out)


if __name__ == "__main__":
    main()
