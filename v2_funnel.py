#!/usr/bin/env python3
"""
v2_funnel.py — V2 της προδιαγραφής: έλεγχος ενός build του σταδίου Α σε όλο τον μήνα.

    python v2_funnel.py dataset/socal_2025-03_ed7b0cbd
    python v2_funnel.py dataset/socal_2025-03_ed7b0cbd --east_lon -117.6

Τυπώνει:
  1. funnel ως ποσοστό των αναφορών στο κουτί (και αν κλείνει)
  2. κάλυψη ανά μέρα / ώρες με χαμηλή κάλυψη
  3. γραμμές πλέγματος ανά μέρα, κινούμενα, GAP_IMP, σημαίες QC, MMSI με QC
  4. region split: μερίδιο ανατολικά του ορίου, πλοία σε κάθε πλευρά και στις δύο
  5. εκτίμηση όγκου παραθύρων 10+10 λεπτών (stride 5) ανά κατηγορία πλοίου
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

COLS = ["mmsi", "t", "seg_id", "lat", "lon", "sog", "STATIONARY", "IN_CORE",
        "GAP_IMP", "QC_SOG", "QC_COG", "vessel_cat", "transceiver", "MMSI_QC"]


def load(d):
    files = sorted(glob.glob(f"{d}/day_*.parquet")) or sorted(glob.glob(f"{d}/day_*.csv.gz"))
    if not files:
        raise SystemExit(f"δεν βρέθηκαν day_* στο {d}")
    if files[0].endswith(".parquet"):
        x = pd.concat((pd.read_parquet(f, columns=COLS) for f in files), ignore_index=True)
    else:
        x = pd.concat((pd.read_csv(f, usecols=COLS) for f in files), ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True)
    return x.sort_values(["mmsi", "t"], kind="mergesort").reset_index(drop=True)


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "—"


def windows(x, obs=10, pred=10, stride=5):
    """Αρχές παραθύρων: obs+pred συνεχόμενα λεπτά μέσα στο ίδιο segment."""
    new_run = (x["seg_id"] != x["seg_id"].shift()) | (x["t"].diff() != pd.Timedelta(minutes=1))
    run = new_run.cumsum()
    pos = x.groupby(run).cumcount()
    run_len = run.map(run.value_counts())
    L = obs + pred
    start = (pos % stride == 0) & (pos + L <= run_len)
    cs = x["sog"].fillna(0).groupby(run).cumsum()
    prev = cs.groupby(run).shift(1).fillna(0)
    end_obs = cs.groupby(run).shift(-(obs - 1))
    mean_obs = (end_obs - prev) / obs
    core_last = x["IN_CORE"].groupby(run).shift(-(obs - 1))
    lon_last = x["lon"].groupby(run).shift(-(obs - 1))
    w = pd.DataFrame({"mmsi": x["mmsi"], "cat": x["vessel_cat"], "trx": x["transceiver"],
                      "mean_sog": mean_obs, "core": core_last, "lon": lon_last})[start]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--east_lon", type=float, default=-117.6)
    a = ap.parse_args()

    m = json.load(open(f"{a.build_dir}/build_meta.json", encoding="utf-8"))
    f = m["funnel"]
    d = f["dropped"]
    inbox = f["raw_rows"] - d.get("outside_box", 0) - d.get("invalid_position", 0)
    print("=" * 70, f"\nV2 — {a.build_dir}\n" + "=" * 70)
    print(f"\n1. FUNNEL (ως % των {inbox:,} αναφορών στο κουτί + περιθώριο)")
    for k in ("invalid_mmsi", "duplicate", "despike", "short_segment"):
        print(f"   {k:<16} {d.get(k, 0):>10,}  {pct(d.get(k, 0), inbox)}")
    print(f"   {'kept_reports':<16} {f['kept_reports']:>10,}  {pct(f['kept_reports'], inbox)}")
    for k, v in f["nulled"].items():
        if v:
            print(f"   nulled {k:<9} {v:>10,}  {pct(v, inbox)}")
    print(f"   implied cuts {f.get('implied_cuts', 0):,} | segments {f['segments']:,} | "
          f"funnel κλείνει: {m.get('funnel_closes')}")

    print("\n2. ΚΑΛΥΨΗ")
    cov = m.get("coverage_rows_per_hour", {})
    low_total = 0
    for fname, hours in sorted(cov.items()):
        v = pd.Series(hours, dtype=float)
        v.index = pd.to_datetime(v.index, utc=True)
        day0 = pd.Timestamp(fname.split("ais-")[1][:10], tz="UTC")
        v = v.reindex(pd.date_range(day0, periods=24, freq="h"), fill_value=0.0)  # ώρες με 0 αναφορές
        low = v[v < 0.5 * v.median()]
        low_total += len(low)
        flag = "" if low.empty else f"  ΧΑΜΗΛΕΣ ΩΡΕΣ: {[pd.Timestamp(h).hour for h in low.index]}"
        print(f"   {fname}: {int(v.sum()):>9,}{flag}")
    print(f"   ώρες με χαμηλή κάλυψη: {low_total} από {24 * len(cov)}")

    x = load(a.build_dir)
    mov = x["STATIONARY"] == 0
    print(f"\n3. ΠΛΕΓΜΑ: {len(x):,} γραμμές, {x['mmsi'].nunique():,} πλοία, {x['seg_id'].nunique():,} segments")
    print(f"   κινούμενα {pct(mov.sum(), len(x))} | GAP_IMP όλα {pct(x['GAP_IMP'].sum(), len(x))}, "
          f"κινούμενα {pct(x.loc[mov, 'GAP_IMP'].sum(), mov.sum())}")
    print(f"   QC_SOG {pct(x['QC_SOG'].sum(), len(x))} | QC_COG {pct(x['QC_COG'].sum(), len(x))} | "
          f"IN_CORE {pct(x['IN_CORE'].sum(), len(x))}")
    print(f"   MMSI με QC: {f['mmsi_qc_flagged']}")
    per_day = x.groupby(x["t"].dt.day).agg(rows=("t", "size"), moving=("STATIONARY", lambda s: (s == 0).sum()))
    print("   γραμμές ανά μέρα (min / median / max):",
          int(per_day["rows"].min()), int(per_day["rows"].median()), int(per_day["rows"].max()))

    print(f"\n4. REGION SPLIT στο {a.east_lon}° (κινούμενα, IN_CORE)")
    mc = x[mov & (x["IN_CORE"] == 1)]
    east = mc["lon"] > a.east_lon
    ves_e, ves_w = set(mc.loc[east, "mmsi"]), set(mc.loc[~east, "mmsi"])
    print(f"   ανατολικά: {pct(east.sum(), len(mc))} των κινούμενων γραμμών")
    print(f"   πλοία: ανατολικά {len(ves_e)}, δυτικά {len(ves_w)}, και στις δύο {len(ves_e & ves_w)} "
          f"({pct(len(ves_e & ves_w), len(ves_e | ves_w))})")

    print("\n5. ΠΑΡΑΘΥΡΑ 10+10 λεπτών (stride 5, στόχος IN_CORE στο τελευταίο λεπτό παρατήρησης)")
    w = windows(x)
    w = w[w["core"] == 1]
    w_mov = w[w["mean_sog"] >= 0.5]
    w_head = w[w["mean_sog"] >= 3.0]
    print(f"   όλα {len(w):,} | κινούμενα (≥0.5 kn) {len(w_mov):,} | κύρια μετρική (≥3 kn) {len(w_head):,}")
    print(f"   πλοία με ≥1 παράθυρο κύριας μετρικής: {w_head['mmsi'].nunique():,}")
    print(f"   κύρια μετρική ανατολικά του {a.east_lon}°: {pct((w_head['lon'] > a.east_lon).sum(), len(w_head))}")
    tab = pd.DataFrame({"κινούμενα": w_mov["cat"].value_counts(), "κύρια": w_head["cat"].value_counts()}).fillna(0)
    tab["κύρια %"] = (100 * tab["κύρια"] / tab["κύρια"].sum()).round(1)
    print(tab.sort_values("κύρια", ascending=False).astype({"κινούμενα": int, "κύρια": int}).to_string())
    print(f"   class B στην κύρια μετρική: {pct((w_head['trx'] == 'B').sum(), len(w_head))}")


if __name__ == "__main__":
    main()