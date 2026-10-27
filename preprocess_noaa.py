#!/usr/bin/env python3
"""
preprocess_noaa.py — Στάδιο Α της προδιαγραφής preprocessing (NOAA MarineCadastre 2025).

Καθαρισμός και resampling σε πλέγμα 1 λεπτού, σε φυσικές μονάδες, χωρίς split και
χωρίς κανονικοποίηση. Τα splits (στάδιο Β) και η ακτογραμμή (στάδιο Γ) είναι
ξεχωριστά βήματα.

Χρήση:
    python preprocess_noaa.py --raw_dir data/2025_03 --month 2025-03 --region socal
    python preprocess_noaa.py --raw_dir data/2025_03 --month 2025-03 --region houston
    python preprocess_noaa.py ... --max_days 2          # γρήγορη δοκιμή σε 2 μέρες
    python preprocess_noaa.py ... --config my.yaml      # αλλαγή παραμέτρων

Έξοδος: dataset/<region>_<YYYY-MM>_<config-hash>/
    day_YYYY-MM-DD.parquet (ή .csv.gz αν λείπει το pyarrow)
    build_meta.json  (config, commit, εκδόσεις, sha256 εισόδου/εξόδου, funnel, κάλυψη)

Τα κανόνες A1–A12 αντιστοιχούν στο §3 της προδιαγραφής. Τα tests: test_synthetic.py.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0
KN = 0.514444  # m/s ανά κόμβο
EPOCH = pd.Timestamp(0, tz="UTC")   # ανεξάρτητο από την ανάλυση χρόνου του pandas (ns/us)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGIONS = {
    "socal":   {"name": "socal",   "lat_min": 30.0, "lat_max": 35.0, "lon_min": -120.0, "lon_max": -115.0},
    "houston": {"name": "houston", "lat_min": 26.5, "lat_max": 31.5, "lon_min": -97.0,  "lon_max": -92.0},
}

DEFAULT_CONFIG = {
    "config_version": "stageA-v3.2",
    "region": REGIONS["socal"],
    "margin_km": 10.0,              # κουτί + περιθώριο για γείτονες (στόχοι μόνο IN_CORE)
    "grid_s": 60,
    "seg_gap_s": 600,               # A7  🔬 {300, 600, 900}
    "bridge_max_s": 180,            # A8  🔬 {120, 180, 360}
    "gap_imp_flag_s": 120,
    "interp_method": "hermite",     # A8  κινηματική Hermite (SOG/COG των άκρων), αλλιώς linear· V3
    "min_segment_s": 300,           # A12 🔬
    # A4 μόνο γενικό όριο αληθοφάνειας. Τα όρια ανά τύπο του EnvShip (35/45/25/20 kn) καταργήθηκαν:
    # στον Μάρτιο 2025 τα εμπορικά δεν τα ξεπερνούν ποτέ, ενώ μικρά σκάφη με λάθος κωδικό τύπου
    # (sport fishing 31–43 kn, «ρυμουλκό» στα 50 kn, SOG = implied speed) κόβονταν σε κάθε βήμα.
    "speed_caps_kn": {"global": 55.0},
    "despike": {                    # A6
        "factor": 1.5,              # κατώφλι = max(SOG) * factor + margin, ≤ όριο τύπου + margin
        "margin_kn": 5.0,
        "fallback_kn": 27.0,        # όταν λείπει το SOG (επαληθευμένο κατώφλι του v2)
        "pos_noise_m": 0.0,         # 🔬 {0, 10, 30}
        "chord_min_m": 150.0,       # δεύτερο κριτήριο: απόκλιση από τη θέση της χορδής i-1 → i+1 στον χρόνο του i
        "chord_k": 0.25,            # κατώφλι χορδής = max(chord_min_m, k * SOG * Δt)
        "cog_agree_deg": 30.0,      # το COG των i-1, i+1 πρέπει να συμφωνεί με τη χορδή
        "max_passes": 10,
        "max_consecutive": 2,       # V5: όχι αλυσίδες αφαιρέσεων (366991470)
    },
    "hermite_min_sog_kn": 3.0,      # Hermite μόνο όπου επικυρώθηκε (V3: SOG ≥ 3 kn), αλλιώς linear
    "sog_interp_max_diff_kn": 5.0,  # A9
    "cog_valid_min_sog_kn": 0.5,    # A10
    "stationary_sog_kn": 0.5,
    "qc_sog_rel": 0.20,             # A11
    "qc_sog_min_kn": 3.0,
    "qc_cog_max_deg": 30.0,
    "mmsi_qc_max_anomaly_frac": 0.05,
    "mmsi_qc_min_reports": 50,      # V5: όχι σημαία από 1 ανωμαλία σε λίγες αναφορές
}

DROP_KEYS = ["invalid_position", "outside_box", "invalid_mmsi", "duplicate",
             "despike", "short_segment"]


def load_config(path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                user = yaml.safe_load(f) or {}
            else:
                user = json.load(f)
        _deep_update(cfg, user)
    return cfg


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Κατηγορίες πλοίων (κωδικοί AIS, §3 της προδιαγραφής)
# ---------------------------------------------------------------------------
def vessel_category(code) -> str:
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "unknown"
    c = int(code)
    if c == 0:
        return "unknown"
    if 70 <= c <= 79:
        return "cargo"
    if 80 <= c <= 89:
        return "tanker"
    if 60 <= c <= 69:
        return "passenger"
    if c == 30:
        return "fishing"
    if c in (31, 32, 52):
        return "tug"
    if c in (36, 37):
        return "pleasure"
    if 40 <= c <= 49:
        return "hsc"
    if c == 35:
        return "military"
    if 50 <= c <= 59 or c in (33, 34):
        return "service"          # pilot, SAR, law enforcement κ.λπ. → γενικό όριο
    return "other"


def _mode(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    vc = s.value_counts()
    top = vc[vc == vc.max()].index
    return sorted(top)[0]          # ντετερμινιστικά στις ισοπαλίες


# ---------------------------------------------------------------------------
# Γεωμετρία
# ---------------------------------------------------------------------------
def _pair_dist_m(lat1, lon1, lat2, lon2):
    latm = np.radians((lat1 + lat2) / 2)
    dy = np.radians(lat2 - lat1) * R_EARTH
    dx = np.radians(lon2 - lon1) * R_EARTH * np.cos(latm)
    return np.hypot(dx, dy), dx, dy


def _to_local(lat, lon, lat0, lon0):
    x = np.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return x, y


def _from_local(x, y, lat0, lon0):
    lat = lat0 + np.degrees(y / R_EARTH)
    lon = lon0 + np.degrees(x / (R_EARTH * math.cos(math.radians(lat0))))
    return lat, lon


def _ang_diff(a, b):
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)


def _circ_interp(a0, a1, alpha):
    a0r, a1r = np.radians(a0), np.radians(a1)
    x = (1 - alpha) * np.cos(a0r) + alpha * np.cos(a1r)
    y = (1 - alpha) * np.sin(a0r) + alpha * np.sin(a1r)
    return np.degrees(np.arctan2(y, x)) % 360.0


# ---------------------------------------------------------------------------
# A1–A5: adapter, εγκυρότητα, κουτί, sentinels, αληθοφάνεια SOG, διπλότυπα
# ---------------------------------------------------------------------------
NOAA_COLS = ["mmsi", "base_date_time", "longitude", "latitude", "sog", "cog",
             "heading", "vessel_type", "status", "transceiver"]


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """A1: σχήμα NOAA 2025 → κανονικό σχήμα, μετονομασία κατά όνομα."""
    df = pd.DataFrame({
        "mmsi": pd.to_numeric(raw["mmsi"], errors="coerce"),
        "t": pd.to_datetime(raw["base_date_time"], utc=True, errors="coerce"),
        "lat": pd.to_numeric(raw["latitude"], errors="coerce"),
        "lon": pd.to_numeric(raw["longitude"], errors="coerce"),
        "sog": pd.to_numeric(raw["sog"], errors="coerce"),
        "cog": pd.to_numeric(raw["cog"], errors="coerce"),
        "heading": pd.to_numeric(raw["heading"], errors="coerce"),
        "vessel_type": pd.to_numeric(raw.get("vessel_type"), errors="coerce"),
        "transceiver": raw.get("transceiver", pd.Series(np.nan, index=raw.index)).astype("object"),
    })
    return df


def box_with_margin(df, region, margin_km):
    dlat = math.degrees(margin_km * 1000 / R_EARTH)
    coslat = np.cos(np.radians(df["lat"].clip(-89, 89)))
    dlon = np.degrees(margin_km * 1000 / (R_EARTH * coslat))
    return ((df["lat"] >= region["lat_min"] - dlat) & (df["lat"] <= region["lat_max"] + dlat) &
            (df["lon"] >= region["lon_min"] - dlon) & (df["lon"] <= region["lon_max"] + dlon))


def valid_mmsi(m: pd.Series) -> pd.Series:
    """9 ψηφία, MID 201–775 (σταθμοί πλοίων), όχι placeholder MID + 000000 (π.χ. 345000000)."""
    ok = m.notna() & (m >= 100_000_000) & (m <= 999_999_999) & (m == np.floor(m))
    mid = (m // 1_000_000)
    placeholder = (m % 1_000_000) == 0
    return ok & (mid >= 201) & (mid <= 775) & ~placeholder


# ---------------------------------------------------------------------------
# A6: despike
# ---------------------------------------------------------------------------
def _pair_threshold(sog_a, sog_b, cap, dcfg):
    """Κατώφλι implied speed. Το SOG μόνο ΑΝΕΒΑΖΕΙ το κατώφλι (γρήγορα σκάφη)· κατώτατο το
    fallback (27 kn, v2), ώστε λάθος χαμηλό SOG να μην κόβει τροχιές (V5: 338090233, 368121190)."""
    s = np.fmax(sog_a, sog_b)                       # fmax: αγνοεί NaN αν υπάρχει η άλλη τιμή
    thr = np.where(np.isnan(s), dcfg["fallback_kn"],
                   np.maximum(dcfg["fallback_kn"], s * dcfg["factor"] + dcfg["margin_kn"]))
    return np.minimum(thr, cap + dcfg["margin_kn"])


def _implied_kn(lat, lon, t, i, j, tol):
    d, dx, dy = _pair_dist_m(lat[i], lon[i], lat[j], lon[j])
    dt = (t[j] - t[i]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.maximum(0.0, d - tol) / dt / KN
    return v, dx, dy, dt


def despike_vessel(lat, lon, t, sog, cog, cap, cfg):
    """Επιστρέφει μάσκα αναφορών που κρατιούνται και πλήθος αφαιρέσεων."""
    dc = cfg["despike"]
    keep = np.ones(len(t), bool)
    removed = 0
    cvm = cfg["cog_valid_min_sog_kn"]
    for _ in range(dc["max_passes"]):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        la, lo, tt, sg, cg = lat[idx], lon[idx], t[idx], sog[idx], cog[idx]
        a, b, c = np.arange(0, len(idx) - 2), np.arange(1, len(idx) - 1), np.arange(2, len(idx))
        dt_ab = (tt[b] - tt[a]).astype(float)
        dt_bc = (tt[c] - tt[b]).astype(float)
        near = (dt_ab <= cfg["seg_gap_s"]) & (dt_bc <= cfg["seg_gap_s"])

        v_ab, *_ = _implied_kn(la, lo, tt, a, b, dc["pos_noise_m"])
        v_bc, *_ = _implied_kn(la, lo, tt, b, c, dc["pos_noise_m"])
        v_ac, dx_ac, dy_ac, dt_ac = _implied_kn(la, lo, tt, a, c, dc["pos_noise_m"])
        thr_ab = _pair_threshold(sg[a], sg[b], cap, dc)
        thr_bc = _pair_threshold(sg[b], sg[c], cap, dc)
        thr_ac = _pair_threshold(sg[a], sg[c], cap, dc)
        crit_speed = (v_ab > thr_ab) & (v_bc > thr_bc) & (v_ac <= thr_ac)
        score_speed = np.minimum(v_ab / thr_ab, v_bc / thr_bc)

        # δεύτερο κριτήριο: απόσταση του b από τη θέση που προβλέπεται στον χρόνο του b από τα a, c
        # (πιάνει πλευρικά και διαμήκη spikes). Πρόβλεψη Hermite όταν a, c έχουν SOG ≥ hermite_min
        # (λαμβάνει υπόψη στροφή και αλλαγή ταχύτητας), αλλιώς χορδή σταθερής ταχύτητας.
        _, dx_ab, dy_ab = _pair_dist_m(la[a], lo[a], la[b], lo[b])
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = dt_ab / dt_ac
        px, py = frac * dx_ac, frac * dy_ac
        hmin = cfg.get("hermite_min_sog_kn", 3.0)
        hok = (sg[a] >= hmin) & (sg[c] >= hmin) & ~np.isnan(cg[a]) & ~np.isnan(cg[c])
        if hok.any():
            f2, f3 = frac ** 2, frac ** 3
            h10, h01, h11 = f3 - 2 * f2 + frac, -2 * f3 + 3 * f2, f3 - f2
            vxa = np.nan_to_num(sg[a]) * KN * np.sin(np.radians(np.nan_to_num(cg[a])))
            vya = np.nan_to_num(sg[a]) * KN * np.cos(np.radians(np.nan_to_num(cg[a])))
            vxc = np.nan_to_num(sg[c]) * KN * np.sin(np.radians(np.nan_to_num(cg[c])))
            vyc = np.nan_to_num(sg[c]) * KN * np.cos(np.radians(np.nan_to_num(cg[c])))
            hx = h10 * dt_ac * vxa + h01 * dx_ac + h11 * dt_ac * vxc
            hy = h10 * dt_ac * vya + h01 * dy_ac + h11 * dt_ac * vyc
            px, py = np.where(hok, hx, px), np.where(hok, hy, py)
        dev = np.hypot(dx_ab - px, dy_ab - py)
        bearing = np.degrees(np.arctan2(dx_ac, dy_ac)) % 360.0
        n_s = (~np.isnan(sg[a])).astype(float) + (~np.isnan(sg[c])).astype(float)
        sum_s = np.nan_to_num(sg[a]) + np.nan_to_num(sg[c])
        with np.errstate(divide="ignore", invalid="ignore"):
            s_mean = np.where(n_s > 0, sum_s / n_s, v_ac) * KN
        chord_thr = np.maximum(dc["chord_min_m"], dc["chord_k"] * s_mean * dt_ac)
        cog_ok_a = ~np.isnan(cg[a]) & (sg[a] >= cvm)
        cog_ok_c = ~np.isnan(cg[c]) & (sg[c] >= cvm)
        agree = (cog_ok_a & cog_ok_c &
                 (_ang_diff(cg[a], bearing) <= dc["cog_agree_deg"]) &
                 (_ang_diff(cg[c], bearing) <= dc["cog_agree_deg"]))
        crit_chord = agree & (dev > chord_thr)
        score_chord = dev / chord_thr

        cand = near & (crit_speed | crit_chord)
        if not cand.any():
            break
        score = np.where(crit_speed, score_speed, 0.0)
        score = np.maximum(score, np.where(crit_chord, score_chord, 0.0))
        order = sorted(np.flatnonzero(cand), key=lambda k: (-score[k], k))
        chosen = set()
        removed_mask = ~keep
        max_run = dc.get("max_consecutive", 2)
        for k in order:
            if (k - 1) in chosen or (k + 1) in chosen:
                continue
            pos = idx[k + 1]
            left = 0
            while pos - left - 1 >= 0 and removed_mask[pos - left - 1]:
                left += 1
            right = 0
            while pos + right + 1 < len(keep) and removed_mask[pos + right + 1]:
                right += 1
            if left + right + 1 > max_run:      # αλυσίδα αφαιρέσεων → τομή, όχι διαγραφή
                continue
            chosen.add(k)
        if not chosen:
            break
        drop_pos = idx[np.array(sorted(chosen)) + 1]
        keep[drop_pos] = False
        removed += len(drop_pos)
    return keep, removed


# ---------------------------------------------------------------------------
# A7: segmentation
# ---------------------------------------------------------------------------
def segment_vessel(lat, lon, t, sog, cap, cfg):
    n = len(t)
    if n == 0:
        return np.zeros(0, int), 0
    dc = cfg["despike"]
    i, j = np.arange(n - 1), np.arange(1, n)
    v, *_ = _implied_kn(lat, lon, t, i, j, dc["pos_noise_m"])
    dt = (t[j] - t[i]).astype(float)
    thr = _pair_threshold(sog[i], sog[j], cap, dc)
    jump = v > thr
    cut = (dt > cfg["seg_gap_s"]) | jump
    seg = np.concatenate([[0], np.cumsum(cut)])
    return seg, int(jump.sum())


# ---------------------------------------------------------------------------
# A8–A11: resampling ενός segment
# ---------------------------------------------------------------------------
def resample_segment(t, lat, lon, sog, cog, hdg, cfg):
    G = cfg["grid_s"]
    t0, t1 = int(t[0]), int(t[-1])
    g = np.arange(-(-t0 // G) * G, (t1 // G) * G + 1, G, dtype=np.int64)
    if len(g) == 0:
        return None
    n = len(t)
    r = np.searchsorted(t, g, side="left")
    r = np.minimum(r, n - 1)
    exact = t[r] == g
    l = np.where(exact, r, r - 1)
    span = np.where(exact, 0, t[r] - t[np.maximum(l, 0)]).astype(float)
    ok = exact | ((l >= 0) & (span <= cfg["bridge_max_s"]))
    g, r, l, span, exact = g[ok], r[ok], l[ok], span[ok], exact[ok]
    if len(g) == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(exact, 0.0, (g - t[l]) / np.where(span > 0, span, 1.0))

    lat0, lon0 = float(lat[0]), float(lon[0])
    x, y = _to_local(lat, lon, lat0, lon0)
    gx = (1 - alpha) * x[l] + alpha * x[r]
    gy = (1 - alpha) * y[l] + alpha * y[r]
    if cfg.get("interp_method", "linear") == "hermite":
        # κινηματική Hermite: ταχύτητες των άκρων από SOG/COG (V3: p90 27.8 έναντι 48.1 m στα 120–180 s)
        hmin = cfg.get("hermite_min_sog_kn", 3.0)   # V5: σε χαμηλές ταχύτητες οι εφαπτόμενες κάνουν βρόχους
        vel_ok = ~np.isnan(sog) & ~np.isnan(cog) & (sog >= hmin)
        vxr = np.where(vel_ok, np.nan_to_num(sog) * KN * np.sin(np.radians(np.nan_to_num(cog))), 0.0)
        vyr = np.where(vel_ok, np.nan_to_num(sog) * KN * np.cos(np.radians(np.nan_to_num(cog))), 0.0)
        use = ~exact & vel_ok[l] & vel_ok[r]
        h = span
        a2, a3 = alpha ** 2, alpha ** 3
        h00, h10, h01, h11 = 2 * a3 - 3 * a2 + 1, a3 - 2 * a2 + alpha, -2 * a3 + 3 * a2, a3 - a2
        hx = h00 * x[l] + h10 * h * vxr[l] + h01 * x[r] + h11 * h * vxr[r]
        hy = h00 * y[l] + h10 * h * vyr[l] + h01 * y[r] + h11 * h * vyr[r]
        gx = np.where(use, hx, gx)
        gy = np.where(use, hy, gy)

    # SOG (A9)
    sa, sb = sog[l], sog[r]
    both = ~np.isnan(sa) & ~np.isnan(sb) & (np.abs(sb - sa) <= cfg["sog_interp_max_diff_kn"])
    nearest = np.where(alpha < 0.5, sa, sb)
    other = np.where(alpha < 0.5, sb, sa)
    nearest = np.where(np.isnan(nearest), other, nearest)
    gsog = np.where(both, (1 - alpha) * sa + alpha * sb, nearest)

    # COG (A10): άκυρο αν λείπει ή αν το SOG της αναφοράς < όριο
    cvm = cfg["cog_valid_min_sog_kn"]
    cvalid = ~np.isnan(cog) & ~(~np.isnan(sog) & (sog < cvm))
    ca, cb, va, vb = cog[l], cog[r], cvalid[l], cvalid[r]
    gcog = np.where(va & vb, _circ_interp(np.nan_to_num(ca), np.nan_to_num(cb), alpha),
                    np.where(va, ca, np.where(vb, cb, np.nan)))
    gcog = np.where(exact, np.where(cvalid[r], cog[r], np.nan), gcog)

    # heading
    ha, hb = hdg[l], hdg[r]
    ghdg = np.where(~np.isnan(ha) & ~np.isnan(hb),
                    _circ_interp(np.nan_to_num(ha), np.nan_to_num(hb), alpha),
                    np.where(~np.isnan(ha) & (alpha < 0.5 + 1e-12), ha,
                             np.where(~np.isnan(hb), hb, ha)))
    ghdg = np.where(exact, hdg[r], ghdg)

    out = pd.DataFrame({"tg": g, "x": gx, "y": gy, "sog": gsog, "cog": gcog, "heading": ghdg,
                        "INTERP_SPAN": span})
    # ταχύτητα από θέσεις, μέσα σε συνεχόμενα λεπτά
    run_id = np.concatenate([[0], np.cumsum(np.diff(g) != G)])
    vx = np.full(len(g), np.nan)
    vy = np.full(len(g), np.nan)
    for rid in np.unique(run_id):
        m = run_id == rid
        if m.sum() >= 2:
            vx[m] = np.gradient(gx[m], G)
            vy[m] = np.gradient(gy[m], G)
    out["vx_pos"], out["vy_pos"] = vx, vy
    vpos_kn = np.hypot(vx, vy) / KN
    speed = np.where(np.isnan(gsog), vpos_kn, gsog)

    cog_valid = ~np.isnan(gcog) & (speed >= cvm)
    out["COG_VALID"] = cog_valid.astype(np.int8)
    # heading: αναφορά (0) → από COG (1) → κρατημένο (2)
    src = np.where(~np.isnan(ghdg), 0, np.nan)
    use_cog = np.isnan(ghdg) & cog_valid
    ghdg = np.where(use_cog, gcog, ghdg)
    src = np.where(use_cog, 1, src)
    h = pd.Series(ghdg)
    held = h.isna()
    h = h.ffill().bfill()
    src = np.where(held & h.notna(), 2, src)
    out["heading"] = h.to_numpy()
    out["HDG_SRC"] = src

    rad = np.radians(np.nan_to_num(gcog))
    out["vx_sog"] = np.where(cog_valid & ~np.isnan(gsog), gsog * KN * np.sin(rad),
                             np.where(~np.isnan(gsog) & (gsog < cvm), 0.0, np.nan))
    out["vy_sog"] = np.where(cog_valid & ~np.isnan(gsog), gsog * KN * np.cos(rad),
                             np.where(~np.isnan(gsog) & (gsog < cvm), 0.0, np.nan))
    out["STATIONARY"] = (speed < cfg["stationary_sog_kn"]).astype(np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        qc_sog = ((gsog >= cfg["qc_sog_min_kn"]) & ~np.isnan(vpos_kn) &
                  (np.abs(gsog - vpos_kn) / gsog > cfg["qc_sog_rel"]))
    out["QC_SOG"] = qc_sog.astype(np.int8)
    bearing_pos = np.degrees(np.arctan2(vx, vy)) % 360.0
    qc_cog = cog_valid & (vpos_kn >= cvm) & (_ang_diff(gcog, bearing_pos) > cfg["qc_cog_max_deg"])
    out["QC_COG"] = qc_cog.astype(np.int8)
    out["GAP_IMP"] = (span > cfg["gap_imp_flag_s"]).astype(np.int8)
    out["lat"], out["lon"] = _from_local(gx, gy, lat0, lon0)
    return out


# ---------------------------------------------------------------------------
# Κύρια συνάρτηση σταδίου Α
# ---------------------------------------------------------------------------
def run_stage_a(raw: pd.DataFrame, cfg: dict):
    funnel = {"raw_rows": int(len(raw)), "kept_reports": 0,
              "dropped": {k: 0 for k in DROP_KEYS},
              "nulled": {"heading_sentinel": 0, "cog_sentinel": 0, "sog_sentinel": 0,
                         "sog_implausible": 0},
              "mmsi_qc_flagged": [], "segments": 0, "grid_rows": 0, "implied_cuts": 0}
    df = standardize(raw)

    def drop(mask, key):
        nonlocal df
        funnel["dropped"][key] += int((~mask).sum())
        df = df.loc[mask]

    drop(df["t"].notna() & df["lat"].between(-90, 90) & df["lon"].between(-180, 180), "invalid_position")
    drop(box_with_margin(df, cfg["region"], cfg["margin_km"]), "outside_box")
    drop(valid_mmsi(df["mmsi"]), "invalid_mmsi")
    df = df.copy()
    df["mmsi"] = df["mmsi"].astype(np.int64)

    # A3 sentinels → NaN
    for col, bad, key in [("heading", (df["heading"] >= 360) | (df["heading"] < 0), "heading_sentinel"),
                          ("cog", (df["cog"] >= 360) | (df["cog"] < 0), "cog_sentinel"),
                          ("sog", (df["sog"] >= 102.2) | (df["sog"] < 0), "sog_sentinel")]:
        funnel["nulled"][key] += int(bad.sum())
        df.loc[bad, col] = np.nan

    # τύπος και transceiver ανά MMSI (συχνότερη τιμή του μήνα)
    vt = df.loc[df["vessel_type"] > 0].groupby("mmsi")["vessel_type"].agg(_mode)
    cat = vt.map(vessel_category)
    df["vessel_cat"] = df["mmsi"].map(cat).fillna("unknown")
    trx = df.groupby("mmsi")["transceiver"].agg(_mode)
    caps = cfg["speed_caps_kn"]
    df["cap"] = df["vessel_cat"].map(lambda c: caps.get(c, caps["global"])).astype(float)

    # A4 αληθοφάνεια SOG
    bad = df["sog"] > df["cap"]
    funnel["nulled"]["sog_implausible"] += int(bad.sum())
    df.loc[bad, "sog"] = np.nan

    # A5 διπλότυπα: κρατιέται η πληρέστερη εγγραφή (ντετερμινιστικά)
    df["_ncomp"] = df[["sog", "cog", "heading"]].notna().sum(axis=1)
    df = df.sort_values(["mmsi", "t", "_ncomp", "lat", "lon", "sog", "cog", "heading"],
                        ascending=[True, True, False, True, True, True, True, True],
                        kind="mergesort", na_position="last")
    dup = df.duplicated(["mmsi", "t"], keep="first")
    funnel["dropped"]["duplicate"] += int(dup.sum())
    df = df.loc[~dup]

    reg = cfg["region"]
    outs = []
    for mmsi, g in df.groupby("mmsi", sort=True):
        t = ((g["t"] - EPOCH) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        sog, cog, hdg = (g[c].to_numpy(float) for c in ("sog", "cog", "heading"))
        cap = float(g["cap"].iloc[0])
        n_rep = len(t)

        keep, n_spk = despike_vessel(lat, lon, t, sog, cog, cap, cfg)
        funnel["dropped"]["despike"] += n_spk
        t, lat, lon, sog, cog, hdg = (a[keep] for a in (t, lat, lon, sog, cog, hdg))
        seg, n_cut = segment_vessel(lat, lon, t, sog, cap, cfg)
        funnel["implied_cuts"] += n_cut
        if n_rep >= cfg.get("mmsi_qc_min_reports", 50) and (n_spk + n_cut) / n_rep > cfg["mmsi_qc_max_anomaly_frac"]:
            funnel["mmsi_qc_flagged"].append(int(mmsi))

        prev_end = None
        k_seg = 0
        for s in np.unique(seg):
            m = seg == s
            ts = t[m]
            if ts[-1] - ts[0] < cfg["min_segment_s"]:
                funnel["dropped"]["short_segment"] += int(m.sum())
                continue
            funnel["kept_reports"] += int(m.sum())
            o = resample_segment(ts, lat[m], lon[m], sog[m], cog[m], hdg[m], cfg)
            if o is None:
                continue
            # υποχρεωτική τρύπα ≥ 1 λεπτού μετά από τομή
            if prev_end is not None:
                o = o[o["tg"] - prev_end >= 2 * cfg["grid_s"]]
                if o.empty:
                    continue
            prev_end = int(o["tg"].iloc[-1])
            o["seg_id"] = f"{int(mmsi)}_{k_seg:04d}"
            k_seg += 1
            o["mmsi"] = int(mmsi)
            outs.append(o)

    cols = ["mmsi", "t", "seg_id", "lat", "lon", "sog", "cog", "COG_VALID", "heading", "HDG_SRC",
            "vessel_cat", "transceiver", "GAP_IMP", "INTERP_SPAN", "STATIONARY", "QC_SOG", "QC_COG",
            "IN_CORE", "MMSI_QC", "vx_sog", "vy_sog", "vx_pos", "vy_pos"]
    if not outs:
        empty = pd.DataFrame(columns=cols)
        empty["t"] = pd.to_datetime(empty["t"], utc=True)
        return empty, funnel
    out = pd.concat(outs, ignore_index=True)
    out["t"] = pd.to_datetime(out["tg"], unit="s", utc=True)
    out["vessel_cat"] = out["mmsi"].map(cat).fillna("unknown")
    out["transceiver"] = out["mmsi"].map(trx)
    out["IN_CORE"] = (out["lat"].between(reg["lat_min"], reg["lat_max"]) &
                      out["lon"].between(reg["lon_min"], reg["lon_max"])).astype(np.int8)
    out["MMSI_QC"] = out["mmsi"].isin(funnel["mmsi_qc_flagged"]).astype(np.int8)
    out = out[cols].sort_values(["mmsi", "t"], kind="mergesort").reset_index(drop=True)
    funnel["segments"] = int(out["seg_id"].nunique())
    funnel["grid_rows"] = int(len(out))
    funnel["mmsi_qc_flagged"] = sorted(funnel["mmsi_qc_flagged"])
    return out, funnel


# ---------------------------------------------------------------------------
# CLI: όλος ο μήνας, partitions ανά MMSI, έξοδος ανά μέρα
# ---------------------------------------------------------------------------
def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _merge_funnel(acc, f):
    if acc is None:
        return copy.deepcopy(f)
    for k in ("raw_rows", "kept_reports", "segments", "grid_rows", "implied_cuts"):
        acc[k] += f[k]
    for d in ("dropped", "nulled"):
        for k, v in f[d].items():
            acc[d][k] = acc[d].get(k, 0) + v
    acc["mmsi_qc_flagged"] = sorted(set(acc["mmsi_qc_flagged"]) | set(f["mmsi_qc_flagged"]))
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--month", required=True, help="YYYY-MM")
    ap.add_argument("--region", required=True, choices=sorted(REGIONS))
    ap.add_argument("--out_root", default="dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n_parts", type=int, default=8)
    ap.add_argument("--max_days", type=int, default=None)
    ap.add_argument("--skip_input_sha", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    cfg["region"] = REGIONS[a.region]
    files = sorted(glob.glob(os.path.join(a.raw_dir, f"ais-{a.month}-*.csv.zst")))
    if a.max_days:
        files = files[: a.max_days]
    if not files:
        sys.exit(f"Δεν βρέθηκαν αρχεία ais-{a.month}-*.csv.zst στο {a.raw_dir}")

    out_dir = Path(a.out_root) / f"{a.region}_{a.month}_{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stageA] {len(files)} αρχεία → {out_dir}")

    # ανάγνωση: φίλτρο κουτιού ήδη στο διάβασμα (μνήμη), κάλυψη ανά ώρα
    parts, coverage, raw_rows, pre_outside, pre_invalid = [], {}, 0, 0, 0
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in NOAA_COLS, low_memory=False)
        raw_rows += len(d)
        lat = pd.to_numeric(d["latitude"], errors="coerce")
        lon = pd.to_numeric(d["longitude"], errors="coerce")
        tmp = pd.DataFrame({"lat": lat, "lon": lon})
        pos_ok = lat.between(-90, 90) & lon.between(-180, 180)
        inbox = pos_ok & box_with_margin(tmp, cfg["region"], cfg["margin_km"])
        pre_invalid += int((~pos_ok).sum())
        pre_outside += int((pos_ok & ~inbox).sum())
        d = d.loc[inbox]
        hours = pd.to_datetime(d["base_date_time"], utc=True, errors="coerce").dt.floor("h")
        coverage[Path(f).name] = {str(k): int(v) for k, v in hours.value_counts().sort_index().items()}
        parts.append(d)
        print(f"  {Path(f).name}: {len(d):,} γραμμές στο κουτί")
    data = pd.concat(parts, ignore_index=True)
    del parts

    mm = pd.to_numeric(data["mmsi"], errors="coerce").fillna(-1).astype(np.int64)
    funnel = None
    outs = []
    for p in range(a.n_parts):
        sub = data.loc[(mm % a.n_parts) == p]
        o, f = run_stage_a(sub, cfg)
        outs.append(o)
        funnel = _merge_funnel(funnel, f)
        print(f"  partition {p + 1}/{a.n_parts}: {len(o):,} γραμμές πλέγματος")
    funnel["raw_rows"] = raw_rows
    funnel["dropped"]["outside_box"] += pre_outside
    funnel["dropped"]["invalid_position"] += pre_invalid
    out = pd.concat(outs, ignore_index=True).sort_values(["mmsi", "t"], kind="mergesort")

    try:
        import pyarrow  # noqa: F401
        ext, writer = "parquet", lambda df, p: df.to_parquet(p, index=False)
    except ImportError:
        print("  ⚠️ λείπει το pyarrow: έξοδος σε .csv.gz (pip install pyarrow για Parquet)")
        ext, writer = "csv.gz", lambda df, p: df.to_csv(p, index=False, compression="gzip")
    out_files = {}
    for day, g in out.groupby(out["t"].dt.strftime("%Y-%m-%d")):
        p = out_dir / f"day_{day}.{ext}"
        writer(g.reset_index(drop=True), p)
        out_files[p.name] = _sha256(p)

    closes = funnel["kept_reports"] + sum(funnel["dropped"].values()) == funnel["raw_rows"]
    meta = {
        "stage": "A", "config": cfg, "config_hash": config_hash(cfg), "commit": _git_commit(),
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "inputs": {Path(f).name: (None if a.skip_input_sha else _sha256(f)) for f in files},
        "outputs": out_files, "funnel": funnel, "funnel_closes": bool(closes),
        "coverage_rows_per_hour": coverage,
    }
    with open(out_dir / "build_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[stageA] {funnel['grid_rows']:,} γραμμές, {funnel['segments']:,} segments, "
          f"funnel κλείνει: {closes}")
    print(json.dumps({"dropped": funnel["dropped"], "nulled": funnel["nulled"],
                      "mmsi_qc_flagged": len(funnel["mmsi_qc_flagged"])}, indent=2))


if __name__ == "__main__":
    main()