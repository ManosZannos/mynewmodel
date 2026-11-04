#!/usr/bin/env python3
"""
v4_invariants.py — V4 της προδιαγραφής: invariants σε όλη την έξοδο ενός build του σταδίου Α.

    python v4_invariants.py dataset/socal_2025-03_<hash>

Διαβάζει το config από το build_meta.json, ελέγχει όλες τις γραμμές και γράφει
<build>/v4_invariants.json. Exit code 1 αν αποτύχει οποιοσδήποτε έλεγχος.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import preprocess_noaa as PA

V4_VERSION = "v4-v1.0"
COLS = ["mmsi", "t", "seg_id", "lat", "lon", "sog", "cog", "COG_VALID", "heading", "HDG_SRC", "GAP_IMP",
        "INTERP_SPAN", "STATIONARY", "QC_SOG", "QC_COG", "IN_CORE", "MMSI_QC", "R_AFTER_S",
        "P_AGE_S", "P_LAT", "P_LON", "P_SOG", "P_COG"]
FLAGS = ["COG_VALID", "GAP_IMP", "STATIONARY", "QC_SOG", "QC_COG", "IN_CORE", "MMSI_QC"]


def check(df: pd.DataFrame, cfg: dict) -> dict:
    """df sorted by (mmsi, t). Returns {name: {"ok": bool, "bad": int, ...}}."""
    B, G = cfg["bridge_max_s"], cfg["grid_s"]
    res = {}

    def put(name, bad, **kw):
        res[name] = {"ok": int(bad) == 0, "bad": int(bad), **kw}

    tm = (df["t"] - PA.EPOCH) // pd.Timedelta(seconds=1)
    tm = tm.to_numpy(np.int64)
    mm, seg = df["mmsi"].to_numpy(), df["seg_id"].to_numpy()
    put("no_nan_key_position", df[["mmsi", "t", "seg_id", "lat", "lon"]].isna().any(axis=1).sum())
    put("unique_mmsi_t", df.duplicated(["mmsi", "t"]).sum())
    put("t_on_grid", (tm % G != 0).sum())
    put("in_box_with_margin", (~PA.box_with_margin(df, cfg["region"], cfg["margin_km"])).sum())

    # ranges
    sog, cog, hdg = (df[c].to_numpy(float) for c in ("sog", "cog", "heading"))
    put("sog_range", ((sog < 0) | (sog > cfg["speed_caps_kn"]["global"])).sum())
    put("cog_range", ((cog < 0) | (cog >= 360)).sum())
    put("heading_range", ((hdg < 0) | (hdg >= 360)).sum())
    put("flags_binary", sum(int((~df[c].isin([0, 1])).sum()) for c in FLAGS))
    put("hdg_src_values", (~(df["HDG_SRC"].isin([0, 1, 2]) | df["HDG_SRC"].isna())).sum())

    # interpolation bookkeeping (A8) and last report (v3.5)
    span, ra, age = (df[c].to_numpy(float) for c in ("INTERP_SPAN", "R_AFTER_S", "P_AGE_S"))
    put("no_point_in_gap_gt_B", ((span < 0) | (span > B)).sum())
    put("gap_imp_matches_span", (df["GAP_IMP"].to_numpy() != (span > cfg["gap_imp_flag_s"])).sum())
    put("r_after_s_range", ((ra < 0) | (ra > B)).sum())
    put("p_age_s_range", ((age < 0) | (age > B)).sum())
    put("age_plus_after_eq_span", (age + ra != span).sum())
    psog, pcog = df["P_SOG"].to_numpy(float), df["P_COG"].to_numpy(float)
    put("p_cog_invalid_when_slow", ((psog < cfg["cog_valid_min_sog_kn"]) & ~np.isnan(pcog)).sum())
    put("p_position_not_nan", df[["P_LAT", "P_LON"]].isna().any(axis=1).sum())

    # consecutive rows
    same_v = mm[1:] == mm[:-1]
    same_s = same_v & (seg[1:] == seg[:-1])
    dt = (tm[1:] - tm[:-1]).astype(float)
    put("time_increasing_in_vessel", (same_v & (dt <= 0)).sum())
    put("hole_after_cut", (same_v & ~same_s & (dt < 2 * G)).sum())
    lat, lon = df["lat"].to_numpy(float), df["lon"].to_numpy(float)
    d, _, _ = PA._pair_dist_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d / dt / PA.KN
    step = same_s & (dt == G)
    thr = PA._pair_threshold(sog[:-1], sog[1:], cfg["speed_caps_kn"]["global"], cfg["despike"])
    put("no_step_above_A6_threshold", (step & (v > thr)).sum(), steps=int(step.sum()))

    # implied speed / SOG within ±10% on ≥ 90% of moving steps (SOG ≥ 3 kn at both ends)
    mv = step & (sog[:-1] >= 3) & (sog[1:] >= 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = v[mv] / ((sog[:-1] + sog[1:]) / 2)[mv]
    frac = float(np.mean(np.abs(ratio - 1) <= 0.10)) if mv.any() else float("nan")
    res["implied_over_sog_within_10pct"] = {"ok": bool(frac >= 0.90), "frac": round(frac, 4), "steps": int(mv.sum())}
    return res


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 1:
        sys.exit(__doc__)
    d = Path(a[0])
    cfg = json.loads((d / "build_meta.json").read_text(encoding="utf-8"))["config"]
    files = sorted(glob.glob(str(d / "day_*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df.sort_values(["mmsi", "t"], kind="mergesort").reset_index(drop=True)
    res = check(df, cfg)
    ok = all(r["ok"] for r in res.values())
    for k, r in res.items():
        print(f"  {'✓' if r['ok'] else '✗'} {k}: " + ", ".join(f"{a}={b}" for a, b in r.items() if a != "ok"))
    out = {"v4_version": V4_VERSION, "build": str(d), "config_version": cfg.get("config_version"),
           "rows": int(len(df)), "files": len(files), "all_ok": ok, "checks": res,
           "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
           "commit": PA._git_commit()}
    (d / "v4_invariants.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V4] {len(df):,} γραμμές | όλα OK: {ok} → {d / 'v4_invariants.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
