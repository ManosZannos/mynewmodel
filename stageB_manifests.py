#!/usr/bin/env python3
"""
stageB_manifests.py — Στάδιο Β της προδιαγραφής: ευρετήριο παραθύρων-στόχων και manifests.

Ανεξάρτητο από μοντέλο. Διαβάζει ένα build του σταδίου Α και γράφει στο <build>/stageB/:
  windows.parquet            ένα δείγμα ανά (πλοίο, έναρξη παραθύρου obs+pred) με strata
  targets_<manifest>.parquet window_id → split, μετά τα όρια ανά πλοίο/segment
  manifest_<manifest>.json   κανόνας, παράμετροι, salt, μεγέθη, επικάλυψη MMSI, commit

Manifests:
  vessel   hash(MMSI) → 80/10/10, εξισορρόπηση στους κινούμενους στόχους (κύριο)
  region   τομή σε μήκος (--east_lon): ανατολικά test, δυτικά train/val (hash)
  time     χρονολογικά 70/15/15 με κενό ≥ --gap_min στα όρια
  alltest  όλα test (Houston, Δεκέμβριος)· seen/unseen ως προς το train ενός άλλου manifest

    python stageB_manifests.py dataset/socal_2025-03_210b4127 --manifests vessel region time
    python stageB_manifests.py dataset/houston_2025-03_6e340c21 --manifests alltest \
        --ref_manifest dataset/socal_2025-03_210b4127/stageB/manifest_vessel.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0

DEFAULTS = {
    "stageB_version": "stageB-v1.0",
    "obs_min": 10, "pred_min": 10, "stride_min": 5,
    "moving_min_sog_kn": 0.5,        # στόχος: μέσο SOG παρατήρησης
    "moving_min_disp_m": 100.0,      # στόχος: καθαρή μετατόπιση στην παρατήρηση (V5)
    "headline_min_sog_kn": 3.0,      # κύρια μετρική
    "difficulty_tiers": [5.0, 12.0], # EnvShip: easy < 5 ≤ medium < 12 ≤ hard
    "caps": {"train": [55, 12], "val": [70, 20], "test": [70, 20]},   # [ανά πλοίο, ανά segment]
    "vessel_split": [0.8, 0.1, 0.1], "balance_tol": 0.03, "salts": list(range(20)),
    "time_split": [0.7, 0.15, 0.15], "gap_min": 20,
    "east_lon": -117.6, "region_val_frac": 0.1,
}

COLS = ["mmsi", "t", "seg_id", "lat", "lon", "sog", "cog", "COG_VALID", "GAP_IMP",
        "IN_CORE", "MMSI_QC", "vessel_cat", "transceiver"]


# ---------------------------------------------------------------------------
def load_build(d):
    files = sorted(glob.glob(f"{d}/day_*.parquet"))
    if files:
        x = pd.concat((pd.read_parquet(f, columns=COLS) for f in files), ignore_index=True)
    else:
        files = sorted(glob.glob(f"{d}/day_*.csv.gz"))
        x = pd.concat((pd.read_csv(f, usecols=COLS) for f in files), ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True)
    x = x.sort_values(["mmsi", "t"], kind="mergesort").reset_index(drop=True)
    x["tm"] = ((x["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    return x


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def build_windows(x, cfg, chunk=400_000):
    """Όλα τα παράθυρα obs+pred συνεχόμενων λεπτών μέσα στο ίδιο segment, με strata."""
    O, P, S = cfg["obs_min"], cfg["pred_min"], cfg["stride_min"]
    L = O + P
    new_run = (x["seg_id"] != x["seg_id"].shift()) | (x["tm"].diff() != 1)
    run = new_run.cumsum().to_numpy()
    pos = x.groupby(run).cumcount().to_numpy()
    run_len = pd.Series(run).map(pd.Series(run).value_counts()).to_numpy()
    starts = np.flatnonzero((pos % S == 0) & (pos + L <= run_len))

    lat, lon = x["lat"].to_numpy(float), x["lon"].to_numpy(float)
    sog, cog = x["sog"].to_numpy(float), x["cog"].to_numpy(float)
    cogv, gimp = x["COG_VALID"].to_numpy(), x["GAP_IMP"].to_numpy()
    core = x["IN_CORE"].to_numpy()
    out = []
    off = np.arange(L)
    for c0 in range(0, len(starts), chunk):
        s = starts[c0:c0 + chunk]
        ix = s[:, None] + off[None, :]
        la, lo = lat[ix], lon[ix]
        lat0 = la[:, :1]
        xx = np.radians(lo - lo[:, :1]) * R_EARTH * np.cos(np.radians(lat0))
        yy = np.radians(la - lat0) * R_EARTH
        dx, dy = np.diff(xx, axis=1), np.diff(yy, axis=1)
        step = np.hypot(dx, dy)
        brg = np.where(step > 1.0, np.degrees(np.arctan2(dx, dy)), np.nan)      # (n, L-1)
        turn = np.abs(_wrap(np.diff(brg, axis=1)))                               # (n, L-2)
        th, tf = turn[:, :O - 2], turn[:, O:]          # βήματα μέσα στην παρατήρηση / στο μέλλον
        bridge = np.abs(_wrap(brg[:, O - 1] - brg[:, O - 2]))                    # last_obs→fut0 έναντι τελευταίου βήματος obs
        mb = np.concatenate([brg, brg[:, -1:]], axis=1)                          # κατεύθυνση κίνησης ανά σημείο
        ecog = np.where(cogv[ix] == 1, np.abs(_wrap(cog[ix] - mb)), np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # παράθυρα χωρίς έγκυρο COG / βήματα
            D = (np.nan_to_num(np.nanmean(th, 1)) + np.nan_to_num(np.nanmean(tf, 1)) +
                 0.5 * (np.nan_to_num(np.nanmax(th, 1)) + np.nan_to_num(np.nanmax(tf, 1))) +
                 np.nan_to_num(bridge) + np.nan_to_num(np.nanmean(ecog, 1)))
            mean_sog_obs = np.nanmean(sog[ix[:, :O]], 1)
        disp_obs = np.hypot(xx[:, O - 1], yy[:, O - 1])
        out.append(pd.DataFrame({
            "row0": s, "mmsi": x["mmsi"].to_numpy()[s], "seg_id": x["seg_id"].to_numpy()[s],
            "t_start": x["tm"].to_numpy()[s],
            "lat_last": la[:, O - 1], "lon_last": lo[:, O - 1],
            "lon_min": lo.min(1), "lon_max": lo.max(1),
            "core_last": core[ix[:, O - 1]].astype(np.int8),
            "mean_sog_obs": mean_sog_obs, "disp_obs_m": disp_obs,
            "interp_ratio": gimp[ix].mean(1), "difficulty": D,
        }))
    w = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    w["vessel_cat"] = x["vessel_cat"].to_numpy()[w["row0"]]
    w["transceiver"] = x["transceiver"].to_numpy()[w["row0"]]
    w["mmsi_qc"] = x["MMSI_QC"].to_numpy()[w["row0"]].astype(np.int8)
    w["moving"] = ((w["mean_sog_obs"] >= cfg["moving_min_sog_kn"]) &
                   (w["disp_obs_m"] >= cfg["moving_min_disp_m"])).astype(np.int8)
    w["headline"] = (w["moving"].astype(bool) & (w["mean_sog_obs"] >= cfg["headline_min_sog_kn"])).astype(np.int8)
    t1, t2 = cfg["difficulty_tiers"]
    w["tier"] = np.where(w["difficulty"] < t1, "easy", np.where(w["difficulty"] < t2, "medium", "hard"))
    w["eligible"] = ((w["moving"] == 1) & (w["core_last"] == 1) & (w["mmsi_qc"] == 0)).astype(np.int8)
    w.insert(0, "window_id", np.arange(len(w), dtype=np.int64))
    return w


# ---------------------------------------------------------------------------
def _hash01(mmsi, salt):
    return np.array([int(hashlib.sha1(f"{salt}:{int(m)}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                     for m in mmsi])


def assign_by_hash(mmsi_unique, salt, fracs):
    h = _hash01(mmsi_unique, salt)
    a, b = fracs[0], fracs[0] + fracs[1]
    return pd.Series(np.where(h < a, "train", np.where(h < b, "val", "test")), index=mmsi_unique)


def apply_caps(t, caps):
    """Ντετερμινιστικά όρια: ομοιόμορφα κατανεμημένα παράθυρα ανά segment, μετά ανά πλοίο."""
    keep = []
    for split, g in t.groupby("split"):
        per_v, per_s = caps[split]
        g = g.sort_values(["mmsi", "t_start"], kind="mergesort")
        sel = []
        for _, gs in g.groupby("seg_id", sort=False):
            if len(gs) > per_s:
                gs = gs.iloc[np.unique(np.round(np.linspace(0, len(gs) - 1, per_s)).astype(int))]
            sel.append(gs)
        g = pd.concat(sel) if sel else g.iloc[:0]
        sel = []
        for _, gv in g.sort_values(["mmsi", "t_start"], kind="mergesort").groupby("mmsi", sort=False):
            if len(gv) > per_v:
                gv = gv.iloc[np.unique(np.round(np.linspace(0, len(gv) - 1, per_v)).astype(int))]
            sel.append(gv)
        keep.append(pd.concat(sel) if sel else g.iloc[:0])
    return pd.concat(keep).sort_values("window_id").reset_index(drop=True)


def manifest_vessel(w, cfg):
    el = w[w["eligible"] == 1]
    vessels = np.sort(el["mmsi"].unique())
    tgt = np.array(cfg["vessel_split"])
    for salt in cfg["salts"]:
        sp = assign_by_hash(vessels, salt, cfg["vessel_split"])
        share = el["mmsi"].map(sp).value_counts(normalize=True).reindex(["train", "val", "test"]).fillna(0).to_numpy()
        if np.all(np.abs(share - tgt) <= cfg["balance_tol"]):
            break
    t = el[["window_id", "mmsi", "seg_id", "t_start"]].copy()
    t["split"] = t["mmsi"].map(sp)
    info = {"rule": "sha1(salt:mmsi) → 80/10/10", "salt": int(salt),
            "eligible_share_before_caps": dict(zip(["train", "val", "test"], np.round(share, 4).tolist()))}
    return t, info


def manifest_region(w, cfg):
    e = cfg["east_lon"]
    el = w[(w["eligible"] == 1) & ~((w["lon_min"] < e) & (w["lon_max"] > e))]   # χωρίς παράθυρα που περνούν το όριο
    t = el[["window_id", "mmsi", "seg_id", "t_start", "lon_last"]].copy()
    west = t["lon_last"] <= e
    vs = np.sort(t.loc[west, "mmsi"].unique())
    h = pd.Series(_hash01(vs, 0), index=vs)
    t["split"] = np.where(~west, "test", np.where(t["mmsi"].map(h) < cfg["region_val_frac"], "val", "train"))
    info = {"rule": f"test: lon_last > {e}; train/val: δυτικά, hash(mmsi) {cfg['region_val_frac']:.0%} val",
            "dropped_crossing": int(((w["eligible"] == 1) & (w["lon_min"] < e) & (w["lon_max"] > e)).sum())}
    return t.drop(columns="lon_last"), info


def manifest_time(w, cfg):
    el = w[w["eligible"] == 1]
    L = cfg["obs_min"] + cfg["pred_min"]
    q = np.cumsum(cfg["time_split"])[:2]
    b1, b2 = np.quantile(el["t_start"], q).astype(np.int64)
    g = cfg["gap_min"]
    t = el[["window_id", "mmsi", "seg_id", "t_start"]].copy()
    end = t["t_start"] + L
    t["split"] = np.select([end <= b1, (t["t_start"] >= b1 + g) & (end <= b2), t["t_start"] >= b2 + g],
                           ["train", "val", "test"], default="")
    t = t[t["split"] != ""]
    info = {"rule": "χρονολογικά 70/15/15", "boundaries_utc": [str(pd.to_datetime(int(b) * 60, unit="s", utc=True))
                                                              for b in (b1, b2)], "gap_min": g}
    return t, info


def manifest_alltest(w, cfg, ref_manifest):
    el = w[w["eligible"] == 1]
    t = el[["window_id", "mmsi", "seg_id", "t_start"]].copy()
    t["split"] = "test"
    info = {"rule": "όλα test"}
    if ref_manifest:
        ref = json.load(open(ref_manifest, encoding="utf-8"))
        tp = Path(ref_manifest).with_name(Path(ref_manifest).name.replace("manifest_", "targets_").replace(".json", ".parquet"))
        tc = Path(str(tp).replace(".parquet", ".csv.gz"))
        rw = pd.read_parquet(tp) if tp.exists() else (pd.read_csv(tc) if tc.exists() else None)
        train_mmsi = set(rw.loc[rw["split"] == "train", "mmsi"]) if rw is not None else set(ref.get("train_mmsi", []))
        t["seen_mmsi"] = t["mmsi"].isin(train_mmsi).astype(np.int8)
        info["ref_manifest"] = ref_manifest
    return t, info


# ---------------------------------------------------------------------------
def summarize(name, t, w):
    tw = t.merge(w[["window_id", "headline", "tier", "vessel_cat", "transceiver"]], on="window_id")
    rows = {}
    for sp, g in tw.groupby("split"):
        rows[sp] = {"windows": int(len(g)), "headline": int(g["headline"].sum()), "vessels": int(g["mmsi"].nunique()),
                    "tiers_headline": g.loc[g["headline"] == 1, "tier"].value_counts(normalize=True).round(3).to_dict(),
                    "class_B_headline": round(float((g.loc[g["headline"] == 1, "transceiver"] == "B").mean()), 3)
                    if g["headline"].sum() else None}
    sets = {sp: set(g["mmsi"]) for sp, g in tw.groupby("split")}
    overlap = None
    if "train" in sets and "test" in sets:
        te = tw[tw["split"] == "test"]
        overlap = round(float(te["mmsi"].isin(sets["train"]).mean()), 4)
    return rows, overlap


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir")
    ap.add_argument("--manifests", nargs="+", default=["vessel", "region", "time"],
                    choices=["vessel", "region", "time", "alltest"])
    ap.add_argument("--ref_manifest", default=None)
    ap.add_argument("--east_lon", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(json.dumps(DEFAULTS))
    if a.east_lon is not None:
        cfg["east_lon"] = a.east_lon

    out = Path(a.build_dir) / "stageB"
    out.mkdir(exist_ok=True)
    print(f"[stageB] {a.build_dir}")
    x = load_build(a.build_dir)
    w = build_windows(x, cfg)
    w.to_parquet(out / "windows.parquet", index=False) if _has_pyarrow() else w.to_csv(out / "windows.csv.gz", index=False)
    print(f"  παράθυρα: {len(w):,} | επιλέξιμα (κινούμενα, IN_CORE, χωρίς QC): {int(w['eligible'].sum()):,} | "
          f"κύρια μετρική: {int((w['eligible'] & w['headline']).sum()):,}")
    print("  δυσκολία (κύρια μετρική):", w.loc[(w["eligible"] == 1) & (w["headline"] == 1), "tier"]
          .value_counts(normalize=True).round(3).to_dict())

    for name in a.manifests:
        if name == "vessel":
            t, info = manifest_vessel(w, cfg)
        elif name == "region":
            t, info = manifest_region(w, cfg)
        elif name == "time":
            t, info = manifest_time(w, cfg)
        else:
            t, info = manifest_alltest(w, cfg, a.ref_manifest)
        n_before = t.groupby("split").size().to_dict()
        t = apply_caps(t, cfg["caps"])
        rows, overlap = summarize(name, t, w)
        meta = {"manifest": name, "build": str(a.build_dir), "config": cfg, "commit": _git(), **info,
                "windows_before_caps": {k: int(v) for k, v in n_before.items()}, "splits": rows,
                "test_rows_from_train_mmsi": overlap}
        if "seen_mmsi" in t.columns:
            meta["seen_mmsi_share_test"] = round(float(t["seen_mmsi"].mean()), 4)
        keep_cols = ["window_id", "mmsi", "seg_id", "t_start", "split"] + (["seen_mmsi"] if "seen_mmsi" in t.columns else [])
        if _has_pyarrow():
            t[keep_cols].to_parquet(out / f"targets_{name}.parquet", index=False)
        else:
            t[keep_cols].to_csv(out / f"targets_{name}.csv.gz", index=False)
        with open(out / f"manifest_{name}.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
        print(f"\n  [{name}] {info.get('rule')}")
        for sp in ("train", "val", "test"):
            if sp in rows:
                r = rows[sp]
                print(f"    {sp:<5} {r['windows']:>8,} παράθυρα ({r['headline']:,} κύρια) σε {r['vessels']:,} πλοία | "
                      f"δυσκολία {r['tiers_headline']} | class B {r['class_B_headline']}")
        if overlap is not None:
            print(f"    test γραμμές από MMSI του train: {overlap:.1%}")
        if "seen_mmsi_share_test" in meta:
            print(f"    seen MMSI (ως προς το train αναφοράς): {meta['seen_mmsi_share_test']:.1%}")
        for k in ("salt", "eligible_share_before_caps", "boundaries_utc", "dropped_crossing"):
            if k in info:
                print(f"    {k}: {info[k]}")
    print(f"\n[stageB] έξοδος: {out}")


def _has_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
