#!/usr/bin/env python3
"""
loader.py — παράθυρα του σταδίου Β → πίνακες για μοντέλα (ανεξάρτητο από μοντέλο).

Για ένα build, ένα manifest και ένα split επιστρέφει:
  obs   (n, 11, F) float32   10 λεπτά παρατήρησης + η τελευταία αναφορά ≤ t0 (αιτιατό· noncausal: (n, 10, F-1))
  fut   (n, 10, 2) float32   θέσεις των 10 λεπτών πρόβλεψης (m), ως στόχος
  pos   (n, 10, 2) float32   θέσεις παρατήρησης (m) — για baselines όπως το CV
  meta  DataFrame            window_id, mmsi, split, vessel_cat, transceiver, headline, capped,
                             tier_cv, cv_fde_m, scene_group, lat_last, lon_last [, seen_mmsi]

Σύστημα αναφοράς: ισοαπέχουσα προβολή με κλίμακα στο πρώτο σημείο του παραθύρου (όπως το στάδιο Β),
με αρχή την τελευταία αναφορά ≤ t0 (αιτιατό, v1.5) ή το τελευταίο σημείο παρατήρησης (noncausal). Η μετατόπιση
δεν αλλάζει διαφορές, άρα το CV του loader συμπίπτει με το cv_fde_m του σταδίου Β (έλεγχος στο test_loader.py).

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

LOADER_VERSION = "loader-v1.5"
R_EARTH = SB.R_EARTH
OBS, PRED = SB.DEFAULTS["obs_min"], SB.DEFAULTS["pred_min"]
SCALES = {"pos_m": 1000.0, "step_m": 100.0, "sog_kn": 20.0}
FEATURES_NONCAUSAL = ["x", "y", "dx", "dy", "sog", "sin_cog", "cos_cog", "cog_valid", "gap_imp", "available"]
# v1.5: dt_min = minutes before t0 (grid point k: OBS-1-k; last-report token: P_AGE_S / 60)
FEATURES = FEATURES_NONCAUSAL + ["dt_min"]
CAUSAL_COLS = ["R_AFTER_S", "P_AGE_S", "P_LAT", "P_LON", "P_SOG", "P_COG"]
SHORE_COLS = ["ON_LAND", "ON_LAND_DEPTH_M", "D_SHORE", "D_WATERFRONT"]
# scene groups τύπου EnvShip, μόνο με κριτήρια απόστασης (τα κριτήρια πλήθους τμημάτων εξαρτώνται από το OSM)
SCENE = {"harbor_wf_m": 1200.0, "nearshore_m": 900.0, "coastal_m": 2200.0}


# ---------------------------------------------------------------------------
def load_rows(build_dir, shore_dir=None):
    """Γραμμές του build με την ΙΔΙΑ σειρά με το stageB.load_build (άρα ισχύει το row0 των παραθύρων)."""
    files = sorted(glob.glob(f"{build_dir}/day_*.parquet"))
    if files:
        import pyarrow.parquet as pq
        names = pq.read_schema(files[0]).names
        cols = SB.COLS + [c for c in CAUSAL_COLS if c in names]
        parts = [pd.read_parquet(f, columns=cols) for f in files]
    else:
        files = sorted(glob.glob(f"{build_dir}/day_*.csv.gz"))
        head = pd.read_csv(files[0], nrows=0).columns
        parts = [pd.read_csv(f, usecols=SB.COLS + [c for c in CAUSAL_COLS if c in head]) for f in files]
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


def resample_obs(obs, step_s):
    """Πυκνότητα παρατήρησης για το πείραμα 20/60/120 s (οι στόχοι μένουν στα 60 s).
    120: κάθε δεύτερο σημείο (καταλήγει στο t0). 20: γραμμική παρεμβολή ανάμεσα στα σημεία των 60 s — ίδια
    πληροφορία, περισσότερα σημεία (όπως ένα πλέγμα 20 s πάνω σε αναφορές ~70 s). dx, dy ξαναϋπολογίζονται."""
    if step_s == 60:
        return obs
    if step_s == 120:
        out = obs[:, (OBS - 1) % 2::2].copy()
    elif step_s == 20:
        n, T, F = obs.shape
        tq = np.arange(0, T - 1 + 1e-9, 1 / 3)                      # 0, 1/3, ..., T-1
        i0 = np.minimum(np.floor(tq).astype(int), T - 2)
        a = (tq - i0)[None, :, None]
        out = ((1 - a) * obs[:, i0] + a * obs[:, i0 + 1]).astype(np.float32)
    else:
        raise ValueError(f"obs step {step_s} s: μόνο 20, 60, 120")
    xy = out[..., :2] * SCALES["pos_m"]
    d = np.diff(xy, axis=1, prepend=xy[:, :1])
    out[..., 2:4] = d / SCALES["step_m"]
    return out


def scene_group(d_shore, d_wf):
    g = np.full(len(d_shore), "offshore", dtype=object)
    g[d_shore <= SCENE["coastal_m"]] = "coastal"
    g[d_shore <= SCENE["nearshore_m"]] = "nearshore"
    g[d_wf <= SCENE["harbor_wf_m"]] = "harbor"
    return g


def _nonoverlap(w):
    """Παράθυρα με έναρξη σε πολλαπλάσιο των 20 λεπτών από το πρώτο παράθυρο του ίδιου (mmsi, seg_id)."""
    t = w["t_start"].to_numpy()
    first = w.groupby(["mmsi", "seg_id"])["t_start"].transform("min").to_numpy()
    return ((t - first) % (OBS + PRED) == 0).astype(np.int8)


def available_mask(x, ix, causal=True):
    """(n, OBS) bool: obs point k is available at t0 only if its later report arrived by t0
    (R_AFTER_S <= 60 * (OBS-1-k)). Monotone in k, so availability is a prefix of the window."""
    n = len(ix)
    if not causal:
        return np.ones((n, OBS), bool)
    if any(c not in x for c in CAUSAL_COLS):
        raise AssertionError("build χωρίς R_AFTER_S/P_* (στάδιο Α < v3.5): η αιτιατή παρατήρηση δεν είναι δυνατή")
    ra = x["R_AFTER_S"].to_numpy()[ix[:, :OBS]]
    return ra <= 60 * (OBS - 1 - np.arange(OBS))[None, :]


def make_arrays(x, w, causal=True):
    """Πίνακες obs/fut/pos για τα παράθυρα w. Causal: σημεία που εξαρτώνται από αναφορά μετά το t0
    μηδενίζονται (available = 0), προστίθεται ως 11ο βήμα η τελευταία αναφορά ≤ t0 (P_* της γραμμής t0)
    και η αρχή των συντεταγμένων είναι αυτή η αναφορά (v1.5· v1.4: το τελευταίο διαθέσιμο σημείο)."""
    ix = window_rows(x, w)
    n = len(ix)
    lat, lon = x["lat"].to_numpy(float)[ix], x["lon"].to_numpy(float)[ix]
    av = available_mask(x, ix, causal)
    c = av.sum(1) - 1                                   # last available obs index
    if (c < 3).any():
        raise AssertionError(f"{int((c < 3).sum())} παράθυρα με < 4 διαθέσιμα σημεία παρατήρησης")
    xx, yy = local_xy(lat, lon)
    rows = np.arange(n)
    t0r = ix[:, OBS - 1]
    if causal:
        age = x["P_AGE_S"].to_numpy(float)[t0r]
        if (age > 60 * (OBS - 1 - c)).any():
            raise AssertionError("τελευταία αναφορά πριν το τελευταίο διαθέσιμο σημείο: ασυνεπή P_AGE_S/R_AFTER_S")
        # last report in the same projection as local_xy (scale at the first window point), then origin there
        plat, plon = x["P_LAT"].to_numpy(float)[t0r], x["P_LON"].to_numpy(float)[t0r]
        ox = np.radians(plon - lon[:, OBS - 1]) * R_EARTH * np.cos(np.radians(lat[:, 0]))
        oy = np.radians(plat - lat[:, OBS - 1]) * R_EARTH
    else:
        ox, oy = xx[rows, c], yy[rows, c]
    xx, yy = xx - ox[:, None], yy - oy[:, None]       # origin = last report (causal) / last obs point
    ob = ix[:, :OBS]
    px, py = np.where(av, xx[:, :OBS], 0.0), np.where(av, yy[:, :OBS], 0.0)
    dx = np.diff(px, axis=1, prepend=px[:, :1])
    dy = np.diff(py, axis=1, prepend=py[:, :1])
    cog = np.radians(x["cog"].to_numpy(float)[ob])
    # v1.4: COG only where SOG exists — when SOG is missing, stage A falls back to a position-based speed
    # with central differences (uses the next minute), so COG_VALID there is not causal
    sog_obs = x["sog"].to_numpy(float)[ob]
    cv = (x["COG_VALID"].to_numpy()[ob] == 1) & ~np.isnan(sog_obs)
    cols = [px / SCALES["pos_m"], py / SCALES["pos_m"], dx / SCALES["step_m"], dy / SCALES["step_m"],
            np.nan_to_num(sog_obs) / SCALES["sog_kn"],
            np.where(cv, np.sin(cog), 0.0), np.where(cv, np.cos(cog), 0.0), cv.astype(float),
            x["GAP_IMP"].to_numpy(float)[ob], np.ones((n, OBS))]
    if causal:
        cols.append(np.broadcast_to((OBS - 1 - np.arange(OBS)).astype(float), (n, OBS)))
    feats = np.where(av[..., None], np.stack(cols, axis=-1), 0.0)   # unavailable points: all channels 0
    if causal:
        # token = last report: at the origin, step from the last available point, raw SOG/COG, age
        psog, pcog = x["P_SOG"].to_numpy(float)[t0r], x["P_COG"].to_numpy(float)[t0r]
        pcv = ~np.isnan(psog) & ~np.isnan(pcog)
        pc = np.radians(np.nan_to_num(pcog))
        tok = np.stack([np.zeros(n), np.zeros(n), -px[rows, c] / SCALES["step_m"], -py[rows, c] / SCALES["step_m"],
                        np.nan_to_num(psog) / SCALES["sog_kn"], np.where(pcv, np.sin(pc), 0.0),
                        np.where(pcv, np.cos(pc), 0.0), pcv.astype(float), np.zeros(n), np.ones(n), age / 60.0], axis=-1)
        feats = np.concatenate([feats, tok[:, None, :]], axis=1)
    obs = feats.astype(np.float32)
    pos = np.stack([px, py], axis=-1).astype(np.float32)
    fut = np.stack([xx[:, OBS:], yy[:, OBS:]], axis=-1).astype(np.float32)
    last = ix[rows, c]
    extra = {"lat_last": lat[rows, c], "lon_last": lon[rows, c], "last_avail": c,
             "p_age_s": age if causal else np.zeros(n),
             # v1.1: ποσοστό σημείων στόχου από μεγάλα κενά (GAP_IMP) — stratum παρεμβολής (πρακτική EnvShip core/full)
             "pred_gap_frac": x["GAP_IMP"].to_numpy(float)[ix[:, OBS:]].mean(1),
             # μη επικαλυπτόμενο υποσύνολο: κάθε 4ο παράθυρο ανά segment (v1.1 μετρούσε σε απόλυτο χρόνο → ~5% αντί ~25%)
             "nonoverlap": _nonoverlap(w)}
    if "D_SHORE" in x:
        extra["scene_group"] = scene_group(x["D_SHORE"].to_numpy()[last], x["D_WATERFRONT"].to_numpy()[last])
    return obs, fut, pos, extra


# ---------------------------------------------------------------------------
def _read_tab(path_parquet):
    p = Path(path_parquet)
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(p.with_name(p.stem + ".csv.gz"))


def load_split(build_dir, manifest, split, shore_dir=None, headline_only=False, x=None, windows=None, causal=True):
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
    obs, fut, pos, extra = make_arrays(x, ws, causal)
    cols = [c for c in ("window_id", "mmsi", "split", "vessel_cat", "transceiver", "headline", "capped",
                        "tier_cv", "cv_fde_m", "seen_mmsi", "on_land") if c in ws]
    meta = ws[cols].copy()
    for k, v in extra.items():
        meta[k] = v
    return {"obs": obs, "fut": fut, "pos": pos, "meta": meta, "features": FEATURES if causal else FEATURES_NONCAUSAL,
            "scales": SCALES,
            "loader_version": LOADER_VERSION, "manifest": manifest, "split": split, "causal": causal}


def manifest_info(build_dir, manifest):
    return json.loads((Path(build_dir) / "stageB" / f"manifest_{manifest}.json").read_text(encoding="utf-8"))