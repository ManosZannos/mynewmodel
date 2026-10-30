#!/usr/bin/env python3
"""
loader.py — παράθυρα του σταδίου Β → πίνακες για μοντέλα (ανεξάρτητο από μοντέλο).

Για ένα build, ένα manifest και ένα split επιστρέφει:
  obs   (n, 10, F) float32   χαρακτηριστικά των 10 λεπτών παρατήρησης (μόνο παρελθόν)
  fut   (n, 10, 2) float32   θέσεις των 10 λεπτών πρόβλεψης (m), ως στόχος
  pos   (n, 10, 2) float32   θέσεις παρατήρησης (m) — για baselines όπως το CV
  meta  DataFrame            window_id, mmsi, split, vessel_cat, transceiver, headline, capped,
                             tier_cv, cv_fde_m, scene_group, lat_last, lon_last [, seen_mmsi]

Σύστημα αναφοράς: ισοαπέχουσα προβολή με κλίμακα στο πρώτο σημείο του παραθύρου (όπως το στάδιο Β),
με αρχή το τελευταίο σημείο παρατήρησης. Η μετατόπιση δεν αλλάζει διαφορές, άρα το CV του loader
συμπίπτει με το cv_fde_m του σταδίου Β (έλεγχος στο test_loader.py).

Κανονικοποίηση: σταθερές κλίμακες (SCALES), όχι στατιστικά δεδομένων → ίδιες σε SoCal/Houston/Δεκέμβριο.
Είναι για τα baselines· οι τελικές τιμές ορίζονται στη φάση μοντέλου (§9 ανοιχτά σημεία).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stageB_manifests as SB

LOADER_VERSION = "loader-v1.0"
R_EARTH = SB.R_EARTH
OBS, PRED = SB.DEFAULTS["obs_min"], SB.DEFAULTS["pred_min"]
SCALES = {"pos_m": 1000.0, "step_m": 100.0, "sog_kn": 20.0}
FEATURES = ["x", "y", "dx", "dy", "sog", "sin_cog", "cos_cog", "cog_valid", "gap_imp"]
SHORE_COLS = ["ON_LAND", "ON_LAND_DEPTH_M", "D_SHORE", "D_WATERFRONT"]
# scene groups τύπου EnvShip, μόνο με κριτήρια απόστασης (τα κριτήρια πλήθους τμημάτων εξαρτώνται από το OSM)
SCENE = {"harbor_wf_m": 1200.0, "nearshore_m": 900.0, "coastal_m": 2200.0}


# ---------------------------------------------------------------------------
def load_rows(build_dir, shore_dir=None):
    """Γραμμές του build με την ΙΔΙΑ σειρά με το stageB.load_build (άρα ισχύει το row0 των παραθύρων)."""
    files = sorted(glob.glob(f"{build_dir}/day_*.parquet"))
    if files:
        parts = [pd.read_parquet(f, columns=SB.COLS) for f in files]
    else:
        files = sorted(glob.glob(f"{build_dir}/day_*.csv.gz"))
        parts = [pd.read_csv(f, usecols=SB.COLS) for f in files]
        for p in parts:
            p["t"] = pd.to_datetime(p["t"], utc=True)
    if shore_dir:
        out = []
        for f, b in zip(files, parts):
            s = pd.read_parquet(Path(shore_dir) / Path(f).name.replace(".csv.gz", ".parquet"), columns=["mmsi"] + SHORE_COLS)
            if len(s) != len(b) or not (s["mmsi"].to_numpy() == b["mmsi"].to_numpy()).all():
                raise AssertionError(f"{Path(f).name}: το στάδιο Γ δεν ευθυγραμμίζεται με το build")
            out.append(pd.concat([b.reset_index(drop=True), s[SHORE_COLS].reset_index(drop=True)], axis=1))
        parts = out
    x = pd.concat(parts, ignore_index=True)
    x = x.sort_values(["mmsi", "t"], kind="mergesort").reset_index(drop=True)
    x["tm"] = ((x["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    return x


def window_rows(x, w):
    """(n, 20) δείκτες γραμμών· ελέγχει ότι κάθε παράθυρο είναι συνεχόμενα λεπτά του ίδιου πλοίου/segment."""
    L = OBS + PRED
    ix = w["row0"].to_numpy()[:, None] + np.arange(L)[None, :]
    if ix.max(initial=-1) >= len(x):
        raise AssertionError("row0 εκτός ορίων: το build ή το στάδιο Γ άλλαξαν μετά το στάδιο Β")
    tm, mm, sg = x["tm"].to_numpy(), x["mmsi"].to_numpy(), x["seg_id"].to_numpy()
    ok = ((tm[ix[:, 0]] == w["t_start"].to_numpy()) & (mm[ix[:, 0]] == w["mmsi"].to_numpy())
          & (tm[ix[:, -1]] - tm[ix[:, 0]] == L - 1) & (mm[ix[:, -1]] == mm[ix[:, 0]]) & (sg[ix[:, -1]] == sg[ix[:, 0]]))
    if not ok.all():
        raise AssertionError(f"{int((~ok).sum())} παράθυρα δεν ταιριάζουν με το build (σειρά γραμμών/έκδοση)")
    return ix


def local_xy(lat, lon):
    """(n, L) → x, y (m): ισοαπέχουσα με κλίμακα στο 1ο σημείο, αρχή στο τελευταίο σημείο παρατήρησης."""
    lat0 = lat[:, :1]
    xx = np.radians(lon - lon[:, :1]) * R_EARTH * np.cos(np.radians(lat0))
    yy = np.radians(lat - lat0) * R_EARTH
    return xx - xx[:, OBS - 1:OBS], yy - yy[:, OBS - 1:OBS]


def scene_group(d_shore, d_wf):
    g = np.full(len(d_shore), "offshore", dtype=object)
    g[d_shore <= SCENE["coastal_m"]] = "coastal"
    g[d_shore <= SCENE["nearshore_m"]] = "nearshore"
    g[d_wf <= SCENE["harbor_wf_m"]] = "harbor"
    return g


def make_arrays(x, w):
    """Πίνακες obs/fut/pos για τα παράθυρα w (μόνο οι γραμμές obs μπαίνουν στα χαρακτηριστικά)."""
    ix = window_rows(x, w)
    lat, lon = x["lat"].to_numpy(float)[ix], x["lon"].to_numpy(float)[ix]
    xx, yy = local_xy(lat, lon)
    ob = ix[:, :OBS]
    px, py = xx[:, :OBS], yy[:, :OBS]
    dx = np.diff(px, axis=1, prepend=px[:, :1])
    dy = np.diff(py, axis=1, prepend=py[:, :1])
    cog = np.radians(x["cog"].to_numpy(float)[ob])
    cv = (x["COG_VALID"].to_numpy()[ob] == 1)
    feats = np.stack([px / SCALES["pos_m"], py / SCALES["pos_m"], dx / SCALES["step_m"], dy / SCALES["step_m"],
                      np.nan_to_num(x["sog"].to_numpy(float)[ob]) / SCALES["sog_kn"],
                      np.where(cv, np.sin(cog), 0.0), np.where(cv, np.cos(cog), 0.0), cv.astype(float),
                      x["GAP_IMP"].to_numpy(float)[ob]], axis=-1)
    obs = feats.astype(np.float32)
    pos = np.stack([px, py], axis=-1).astype(np.float32)
    fut = np.stack([xx[:, OBS:], yy[:, OBS:]], axis=-1).astype(np.float32)
    last = ix[:, OBS - 1]
    extra = {"lat_last": lat[:, OBS - 1], "lon_last": lon[:, OBS - 1]}
    if "D_SHORE" in x:
        extra["scene_group"] = scene_group(x["D_SHORE"].to_numpy()[last], x["D_WATERFRONT"].to_numpy()[last])
    return obs, fut, pos, extra


# ---------------------------------------------------------------------------
def _read_tab(path_parquet):
    p = Path(path_parquet)
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(p.with_name(p.stem + ".csv.gz"))


def load_split(build_dir, manifest, split, shore_dir=None, headline_only=False, x=None, windows=None):
    """Πίνακες για ένα split ενός manifest. Τα x/windows μπορούν να δοθούν για επαναχρησιμοποίηση."""
    sb = Path(build_dir) / "stageB"
    w = windows if windows is not None else _read_tab(sb / "windows.parquet")
    t = _read_tab(sb / f"targets_{manifest}.parquet")
    t = t[t["split"] == split]
    keep = [c for c in ("window_id", "split", "capped", "seen_mmsi") if c in t]
    ws = t[keep].merge(w, on="window_id", how="left", validate="one_to_one")
    if ws["row0"].isna().any():
        raise AssertionError("targets με window_id που λείπουν από το windows.parquet")
    if headline_only:
        ws = ws[ws["headline"] == 1]
    ws = ws.sort_values("window_id").reset_index(drop=True)
    if x is None:
        x = load_rows(build_dir, shore_dir)
    obs, fut, pos, extra = make_arrays(x, ws)
    cols = [c for c in ("window_id", "mmsi", "split", "vessel_cat", "transceiver", "headline", "capped",
                        "tier_cv", "cv_fde_m", "seen_mmsi", "on_land") if c in ws]
    meta = ws[cols].copy()
    for k, v in extra.items():
        meta[k] = v
    return {"obs": obs, "fut": fut, "pos": pos, "meta": meta, "features": FEATURES, "scales": SCALES,
            "loader_version": LOADER_VERSION, "manifest": manifest, "split": split}


def manifest_info(build_dir, manifest):
    return json.loads((Path(build_dir) / "stageB" / f"manifest_{manifest}.json").read_text(encoding="utf-8"))
