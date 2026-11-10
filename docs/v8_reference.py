#!/usr/bin/env python3
"""
v8_reference.py -- independent reference implementation of Stage A (check V8).

Source of truth: specification.md (sections 2, 3, 6, 7) in this folder ONLY.
Every place where the spec admits more than one reading is tagged in the code
with an ID "AMB-xx"; the full list (section, what the spec says, chosen
reading, reason) is in v8_ambiguities.md next to this file.

Usage
-----
  python v8_reference.py --raw_dir ~/PLAN/data/2025_03 --days 1-31 \
      --out_dir ./v8ref_socal_2025-03 --workers 16
  python v8_reference.py --selftest

Output
------
  <out_dir>/day_YYYY-MM-DD.parquet   one row per (mmsi, minute)
  <out_dir>/funnel.json              report-level and grid-level funnels
                                     (both must close), counters, MMSI QC,
                                     V4-lite invariants, inputs/outputs sha256.

Requires: numpy, pandas, pyarrow (pyarrow reads .zst natively; no zstandard pkg).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

VERSION = "v8ref-1.0"

R_EARTH = 6371008.8          # mean Earth radius (m)
KN = 1852.0 / 3600.0         # 1 knot in m/s

CFG = dict(
    # A2
    margin_m=10000.0, mid_min=201, mid_max=775,
    # A3 / A4
    hdg_sentinel=360.0, cog_sentinel=360.0, sog_sentinel=102.2, sog_max=55.0,
    # A6 (threshold and despike)
    a6_min_kn=27.0, a6_sog_factor=1.5, a6_sog_add=5.0, a6_max_kn=60.0,   # 60 = 55 + 5
    a6_noise_tol_m=0.0,
    a6_chord_min_m=150.0, a6_chord_frac=0.25,
    a6_cog_agree_deg=30.0,          # AMB-09: tolerance not stated, reuse A11's 30 deg
    a6_hermite_min_sog=3.0,
    a6_max_consecutive=2,
    # A7 / A8 / A9 / A10
    G_s=600, B_s=180, hermite_min_sog=3.0, sog_lin_max_diff=5.0, cog_valid_min_sog=0.5,
    # A11
    qc_sog_min=3.0, qc_sog_rel=0.20, qc_cog_deg=30.0, qc_cog_min_speed=0.5,
    # misc
    gap_imp_s=120, min_seg_s=300, hole_min_minutes=1,
    mmsi_qc_rate=0.05, mmsi_qc_min_reports=50,
    v4_grid_box_tol_m=50.0,
)

REGIONS = {
    "socal": dict(lat=(30.0, 35.0), lon=(-120.0, -115.0), split_lon=-117.6),
    "houston": dict(lat=(26.5, 31.5), lon=(-97.0, -92.0), split_lon=None),
}

# Report status codes (report-level funnel)
ST_KEPT, ST_DUP, ST_SPIKE, ST_SHORT = 0, 1, 2, 3

OUT_COLUMNS = [
    ("mmsi", pa.int64()), ("t", pa.int32()), ("seg_id", pa.int32()),
    ("lat", pa.float64()), ("lon", pa.float64()), ("sog", pa.float64()),
    ("cog", pa.float64()), ("COG_VALID", pa.int8()),
    ("heading", pa.float64()), ("HDG_SRC", pa.int8()),
    ("vessel_cat", pa.string()), ("transceiver", pa.string()),
    ("GAP_IMP", pa.int8()), ("INTERP_SPAN", pa.int32()), ("STATIONARY", pa.int8()),
    ("QC_SOG", pa.int8()), ("QC_COG", pa.int8()), ("IN_CORE", pa.int8()), ("MMSI_QC", pa.int8()),
    ("vx_pos", pa.float64()), ("vy_pos", pa.float64()),
    ("R_AFTER_S", pa.int32()), ("P_AGE_S", pa.int32()),
    ("P_LAT", pa.float64()), ("P_LON", pa.float64()), ("P_SOG", pa.float64()), ("P_COG", pa.float64()),
    ("N_SOG", pa.float64()),
]
OUT_SCHEMA = pa.schema(OUT_COLUMNS)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def hav(lat1, lon1, lat2, lon2):
    """Great-circle distance (m)."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def enu(lat0, lon0, lat, lon):
    """Local equirectangular plane centred at (lat0, lon0): x east, y north (m). AMB-14."""
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return x, y


def enu_inv(lat0, lon0, x, y):
    lat = lat0 + np.degrees(y / R_EARTH)
    lon = lon0 + np.degrees(x / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


def wrap360(a):
    """Normalise to [0, 360). np.mod can return exactly 360.0 for tiny negatives."""
    r = np.mod(a, 360.0)
    return np.where(r >= 360.0, 0.0, r)


def angdiff(a, b):
    return np.abs(np.mod(a - b + 180.0, 360.0) - 180.0)


def circ_interp(c0, c1, s):
    d = np.mod(c1 - c0 + 180.0, 360.0) - 180.0
    return wrap360(c0 + s * d)


def bearing_xy(x, y):
    return wrap360(np.degrees(np.arctan2(x, y)))


def nanmax_n(*arrs):
    m = np.asarray(arrs[0], dtype=float)
    for a in arrs[1:]:
        m = np.fmax(m, a)
    return m


def a6_thr(*sogs, cfg=CFG):
    """A6 implied-speed threshold (kn): max(27, maxSOG*1.5+5), capped at 55+5; no SOG -> 27."""
    m = nanmax_n(*sogs)
    th = np.minimum(m * cfg["a6_sog_factor"] + cfg["a6_sog_add"], cfg["a6_max_kn"])
    return np.fmax(th, cfg["a6_min_kn"])        # NaN (no SOG) -> 27


def implied_kn(d, dt, cfg=CFG):
    return np.maximum(0.0, d - cfg["a6_noise_tol_m"]) / dt / KN


def hermite_xy(s, span, dx, dy, sog0, cog0, sog1, cog1):
    """Kinematic cubic Hermite from p0=(0,0) to p1=(dx,dy), end velocities from SOG/COG."""
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    v0 = sog0 * KN
    v1 = sog1 * KN
    c0 = np.radians(cog0)
    c1 = np.radians(cog1)
    px = h10 * span * v0 * np.sin(c0) + h01 * dx + h11 * span * v1 * np.sin(c1)
    py = h10 * span * v0 * np.cos(c0) + h01 * dy + h11 * span * v1 * np.cos(c1)
    return px, py


# ---------------------------------------------------------------------------
# A1 adapter: read one daily NOAA .csv.zst by column NAME
# ---------------------------------------------------------------------------
ALIASES = {
    "mmsi": ["mmsi"],
    "base_date_time": ["base_date_time", "basedatetime"],
    "lon": ["longitude", "lon"],
    "lat": ["latitude", "lat"],
    "sog": ["sog"],
    "cog": ["cog"],
    "heading": ["heading"],
    "vessel_type": ["vessel_type", "vesseltype"],
    "transceiver": ["transceiver", "transceiver_class", "transceiverclass"],
}
TYPES = {
    "mmsi": pa.int64(), "base_date_time": pa.timestamp("s"),
    "lon": pa.float64(), "lat": pa.float64(), "sog": pa.float64(), "cog": pa.float64(),
    "heading": pa.float64(), "vessel_type": pa.float64(), "transceiver": pa.string(),
}


def _norm(c):
    return c.strip().strip('"').strip().lower().replace(" ", "_")


def _header(path):
    with pa.input_stream(path, compression="zstd") as f:
        buf = f.read(1 << 16)
    first = buf.decode("utf-8", "replace").splitlines()[0]
    return [c.strip().strip('"') for c in first.split(",")]


def read_day(path, month_start_epoch):
    """Return DataFrame with canonical columns; ts = seconds since month start (float, NaN if bad)."""
    cols = _header(path)
    by_norm = {}
    for c in cols:
        by_norm.setdefault(_norm(c), c)
    mapping = {}
    for canon, aliases in ALIASES.items():
        hit = [by_norm[a] for a in aliases if a in by_norm]
        if not hit:
            raise ValueError(f"A1: column '{canon}' not found in {path}; header={cols}")
        mapping[hit[0]] = canon

    def _read(types):
        with pa.input_stream(path, compression="zstd") as f:
            return pacsv.read_csv(
                f, convert_options=pacsv.ConvertOptions(
                    include_columns=list(mapping), column_types=types,
                    strings_can_be_null=True))

    try:
        tbl = _read({orig: TYPES[canon] for orig, canon in mapping.items()})
        robust = False
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        types = {orig: TYPES[canon] for orig, canon in mapping.items()}
        for orig, canon in mapping.items():
            if canon in ("mmsi", "base_date_time"):
                types[orig] = pa.string()
        tbl = _read(types)
        robust = True
    tbl = tbl.rename_columns([mapping[c] for c in tbl.column_names])
    if not robust:
        ep = tbl.column("base_date_time").cast(pa.int64()).to_numpy(zero_copy_only=False)
        ep = np.asarray(ep, dtype=float)
        tbl = tbl.drop(["base_date_time"])
        df = tbl.to_pandas()
        df["ts"] = ep - month_start_epoch
    else:
        df = tbl.to_pandas()
        dt = pd.to_datetime(df.pop("base_date_time"), utc=True, format="ISO8601", errors="coerce")
        sec = (dt - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1s")
        df["ts"] = sec.to_numpy(dtype=float) - month_start_epoch
        df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df["transceiver"] = df["transceiver"].astype(object)
    return df


# ---------------------------------------------------------------------------
# A1-A4 (row-wise, per day)
# ---------------------------------------------------------------------------
def build_box(region, cfg=CFG):
    r = REGIONS[region]
    return dict(core_lat=r["lat"], core_lon=r["lon"], margin_m=cfg["margin_m"])


def in_build_box(lat, lon, box):
    """Core box expanded by margin_m: lat by margin/R, lon by margin/(R cos lat) at the
    point's own (clipped) latitude. AMB-03."""
    mlat = np.degrees(box["margin_m"] / R_EARTH)
    latc = np.clip(lat, box["core_lat"][0], box["core_lat"][1])
    mlon = np.degrees(box["margin_m"] / (R_EARTH * np.cos(np.radians(latc))))
    return ((lat >= box["core_lat"][0] - mlat) & (lat <= box["core_lat"][1] + mlat)
            & (lon >= box["core_lon"][0] - mlon) & (lon <= box["core_lon"][1] + mlon))


def in_core(lat, lon, box):
    return ((lat >= box["core_lat"][0]) & (lat <= box["core_lat"][1])
            & (lon >= box["core_lon"][0]) & (lon <= box["core_lon"][1]))


def prefilter_day(df, day, box, cfg=CFG, lenient=False):
    """A1 (validity asserts), A2 (box + MMSI), A3 (sentinels), A4 (SOG plausibility)."""
    c = {}
    n0 = len(df)
    c["raw"] = n0
    # A1: rows without the mandatory key fields cannot be used at all (AMB-01)
    miss = df["mmsi"].isna() | df["ts"].isna() | df["lat"].isna() | df["lon"].isna()
    c["A1_missing_key"] = int(miss.sum())
    df = df[~miss]
    lo_t, hi_t = (day - 1) * 86400, day * 86400
    bad_range = ~df["lat"].between(-90, 90) | ~df["lon"].between(-180, 180)
    bad_time = (df["ts"] < lo_t) | (df["ts"] >= hi_t)
    if not lenient and (bad_range.any() or bad_time.any()):
        raise AssertionError(
            f"A1 assert failed on day {day}: {int(bad_range.sum())} rows lat/lon out of range, "
            f"{int(bad_time.sum())} rows outside the file's UTC day. Re-run with --a1_lenient "
            f"to drop and count them instead.")
    c["A1_out_of_range"] = int(bad_range.sum())
    c["A1_wrong_day"] = int((bad_time & ~bad_range).sum())
    df = df[~(bad_range | bad_time)].copy()
    df["mmsi"] = df["mmsi"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)
    # A2: box (+margin) first, then MMSI (AMB-02)
    inbox = in_build_box(df["lat"].to_numpy(), df["lon"].to_numpy(), box)
    c["A2_outside_box"] = int((~inbox).sum())
    df = df[inbox]
    m = df["mmsi"].to_numpy()
    mid = m // 1_000_000
    ok = (m >= 100_000_000) & (m <= 999_999_999) & (mid >= cfg["mid_min"]) & (mid <= cfg["mid_max"])
    ok &= (m % 1_000_000) != 0           # MID + 000000 placeholder (spec §8 V5 table). AMB-04
    c["A2_invalid_mmsi"] = int((~ok).sum())
    df = df[ok].copy()
    # A3: sentinels -> NaN, row kept. Negative values treated the same (AMB-05)
    for col, lim, key in (("heading", cfg["hdg_sentinel"], "A3_heading_nan"),
                          ("cog", cfg["cog_sentinel"], "A3_cog_nan"),
                          ("sog", cfg["sog_sentinel"], "A3_sog_nan")):
        v = df[col].to_numpy(dtype=float)
        bad = (v >= lim) | (v < 0)
        c[key] = int(bad.sum())
        v = np.where(bad, np.nan, v)
        df[col] = v
    # A4: SOG > 55 kn -> NaN (position judged in A6)
    v = df["sog"].to_numpy(dtype=float)
    bad = v > cfg["sog_max"]
    c["A4_sog_nan"] = int(bad.sum())
    df["sog"] = np.where(bad, np.nan, v)
    c["after_A4"] = len(df)
    df["day"] = np.int16(day)
    return df, c


# ---------------------------------------------------------------------------
# vessel category (static, monthly mode)
# ---------------------------------------------------------------------------
def vessel_cat_of(code):
    """AIS type code -> category (§3 'Αντιστοίχιση τύπων'). AMB-16."""
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "unknown"
    c = int(code)
    if c == 0:
        return "unknown"
    if c == 30:
        return "fishing"
    if c in (31, 32, 52):
        return "tug"
    if 50 <= c <= 59:
        return "service"
    if 60 <= c <= 69:
        return "passenger"
    if 70 <= c <= 79:
        return "cargo"
    if 80 <= c <= 89:
        return "tanker"
    if c in (36, 37):
        return "pleasure"
    if 40 <= c <= 49:
        return "hsc"
    if c == 35:
        return "military"
    return "other"


def monthly_mode(df, col):
    """Most frequent non-null value per MMSI; ties -> smallest value. AMB-15."""
    sub = df[["mmsi", col]].dropna()
    if col == "transceiver":
        sub = sub[sub[col].astype(str).str.strip() != ""]
    if sub.empty:
        return {}
    cnt = sub.groupby(["mmsi", col]).size().reset_index(name="n")
    cnt = cnt.sort_values(["mmsi", "n", col], ascending=[True, False, True], kind="mergesort")
    cnt = cnt.drop_duplicates("mmsi")
    return dict(zip(cnt["mmsi"].to_numpy(), cnt[col].to_numpy()))


# ---------------------------------------------------------------------------
# A6 despike
# ---------------------------------------------------------------------------
def despike(t, lat, lon, sog, cog, cfg=CFG):
    """Iterative single-point despike on the time-sorted reports of one vessel.

    Spike i (neighbours = current kept i-1, i+1, both gaps <= G, AMB-07):
      crit1: v(i-1,i) > thr and v(i,i+1) > thr while v(i-1,i+1) <= thr   (AMB-06)
      crit2: |p_i - pred_i| > max(150 m, 0.25*SOG*dt), pred from i-1,i+1 (Hermite when both
             SOG >= 3 kn, else constant-speed chord), only when the COG of i-1 and of i+1
             both agree with the chord bearing (AMB-08, AMB-09).
    At most a6_max_consecutive consecutive removed reports; a point whose removal would
    extend a run beyond that is 'blocked': kept, and a cut is forced before it (AMB-10).
    Per pass, flagged points are removed greedily left-to-right, never two adjacent (AMB-11).
    Returns keep mask, blocked mask.
    """
    n = len(t)
    keep = np.ones(n, bool)
    blocked = np.zeros(n, bool)
    if n < 3:
        return keep, blocked
    G = cfg["G_s"]
    cval = ~np.isnan(cog) & ~(sog < cfg["cog_valid_min_sog"])
    while True:
        k = np.flatnonzero(keep)
        if len(k) < 3:
            break
        a, b, c = k[:-2], k[1:-1], k[2:]
        dt1 = (t[b] - t[a]).astype(float)
        dt2 = (t[c] - t[b]).astype(float)
        dt13 = dt1 + dt2
        elig = (dt1 <= G) & (dt2 <= G) & ~blocked[b]
        if not elig.any():
            break
        sa, sb, sc = sog[a], sog[b], sog[c]
        v1 = implied_kn(hav(lat[a], lon[a], lat[b], lon[b]), dt1, cfg)
        v2 = implied_kn(hav(lat[b], lon[b], lat[c], lon[c]), dt2, cfg)
        v13 = implied_kn(hav(lat[a], lon[a], lat[c], lon[c]), dt13, cfg)
        plaus13 = v13 <= a6_thr(sa, sc, cfg=cfg)   # 'while i-1 -> i+1 is plausible' (both criteria, AMB-06b)
        crit1 = (v1 > a6_thr(sa, sb, cfg=cfg)) & (v2 > a6_thr(sb, sc, cfg=cfg)) & plaus13

        # crit2
        xb, yb = enu(lat[a], lon[a], lat[b], lon[b])
        xc, yc = enu(lat[a], lon[a], lat[c], lon[c])
        chord = np.hypot(xc, yc)
        brg = bearing_xy(xc, yc)
        ca, cc = cog[a], cog[c]
        with np.errstate(invalid="ignore"):
            agree = (cval[a] & cval[c] & (chord > 0)
                     & (angdiff(ca, brg) <= cfg["a6_cog_agree_deg"])
                     & (angdiff(cc, brg) <= cfg["a6_cog_agree_deg"]))
            s = dt1 / dt13
            herm = (sa >= cfg["a6_hermite_min_sog"]) & (sc >= cfg["a6_hermite_min_sog"]) & cval[a] & cval[c]
        px, py = s * xc, s * yc
        if herm.any():
            hx, hy = hermite_xy(s[herm], dt13[herm], xc[herm], yc[herm], sa[herm], ca[herm], sc[herm], cc[herm])
            px = px.copy(); py = py.copy()
            px[herm] = hx; py[herm] = hy
        err = np.hypot(xb - px, yb - py)
        sref = nanmax_n(sa, sb, sc)                               # AMB-08
        sref = np.where(np.isnan(sref), chord / dt13 / KN, sref)
        tol = np.maximum(cfg["a6_chord_min_m"], cfg["a6_chord_frac"] * sref * KN * np.minimum(dt1, dt2))
        crit2 = agree & plaus13 & (err > tol)

        flag = elig & (crit1 | crit2)
        if not flag.any():
            break
        chain_ok = (c - a - 1) <= cfg["a6_max_consecutive"]
        newly_blocked = flag & ~chain_ok
        blocked[b[newly_blocked]] = True
        sel = []
        last = -10
        for p in np.flatnonzero(flag & chain_ok):
            if p > last + 1:
                sel.append(p)
                last = p
        if sel:
            keep[b[np.asarray(sel)]] = False
        elif not newly_blocked.any():
            break
    return keep, blocked


# ---------------------------------------------------------------------------
# A8-A10 resampling of one segment
# ---------------------------------------------------------------------------
def resample_segment(t, lat, lon, sog, cog, hdg, st, cfg=CFG):
    n = len(t)
    g0 = -(-t[0] // 60) * 60
    g1 = (t[-1] // 60) * 60
    if g1 < g0:
        return None
    g = np.arange(g0, g1 + 1, 60, dtype=np.int64)
    st["grid_candidates"] += len(g)
    j = np.searchsorted(t, g, side="right") - 1
    exact = t[j] == g
    jn = np.where(exact, j, np.minimum(j + 1, n - 1))
    span = t[jn] - t[j]
    ok = span <= cfg["B_s"]                                     # no point inside a gap > B
    st["grid_drop_gap_gt_B"] += int((~ok).sum())
    if not ok.any():
        return None
    g, j, jn, exact, span = g[ok], j[ok], jn[ok], exact[ok], span[ok]
    s = np.where(exact, 0.0, (g - t[j]) / np.where(span > 0, span, 1).astype(float))

    cval = ~np.isnan(cog) & ~(sog < cfg["cog_valid_min_sog"])       # report-level COG validity
    herm_iv = np.zeros(n, bool)                                      # interval i = (t_i, t_i+1)
    if n > 1:
        hm = cfg["hermite_min_sog"]
        with np.errstate(invalid="ignore"):
            herm_iv[:-1] = (sog[:-1] >= hm) & (sog[1:] >= hm) & ~np.isnan(cog[:-1]) & ~np.isnan(cog[1:])
    dx, dy = enu(lat[j], lon[j], lat[jn], lon[jn])

    def positions():
        px, py = s * dx, s * dy
        h = herm_iv[j] & ~exact
        if h.any():
            hx, hy = hermite_xy(s[h], span[h].astype(float), dx[h], dy[h],
                                sog[j[h]], cog[j[h]], sog[jn[h]], cog[jn[h]])
            px = px.copy(); py = py.copy()
            px[h] = hx; py[h] = hy
        return enu_inv(lat[j], lon[j], px, py)

    # A8 guard (v3.6): linearise intervals touched by a grid step above the A6 threshold
    while True:
        la, lo = positions()
        if len(g) < 2:
            break
        cons = np.diff(g) == 60
        v = hav(la[:-1], lo[:-1], la[1:], lo[1:]) / 60.0 / KN
        thr = a6_thr(sog[j[:-1]], sog[jn[:-1]], sog[j[1:]], sog[jn[1:]], cfg=cfg)  # AMB-13
        bad = np.flatnonzero(cons & (v > thr))
        if len(bad) == 0:
            break
        changed = False
        for p in bad:
            lo_i = j[p]
            hi_i = j[p + 1] - (1 if exact[p + 1] else 0)
            sl = herm_iv[lo_i:hi_i + 1]
            if sl.any():
                st["guard_linearised_intervals"] += int(sl.sum())
                herm_iv[lo_i:hi_i + 1] = False
                changed = True
        if not changed:
            st["guard_unfixed_steps"] += len(bad)
            break

    # A9 SOG
    s0, s1 = sog[j], sog[jn]
    after_closer = (g - t[j]) > (t[jn] - g)                      # tie -> earlier report (AMB-17)
    near = np.where(after_closer, s1, s0)
    other = np.where(after_closer, s0, s1)
    near = np.where(np.isnan(near), other, near)                 # AMB-18
    with np.errstate(invalid="ignore"):
        lin = ~np.isnan(s0) & ~np.isnan(s1) & (np.abs(s1 - s0) <= cfg["sog_lin_max_diff"])
    sog_g = np.where(exact, s0, np.where(lin, s0 + s * (s1 - s0), near))

    # A10 COG: circular, invalid side replaced by the other, none -> NaN (AMB-19)
    c0v, c1v = cval[j], cval[jn]
    c0, c1 = cog[j], cog[jn]
    cog_g = np.full(len(g), np.nan)
    both = c0v & c1v
    cog_g[both] = circ_interp(c0[both], c1[both], s[both])
    m0 = c0v & ~c1v
    cog_g[m0] = c0[m0]
    m1 = ~c0v & c1v
    cog_g[m1] = c1[m1]
    cog_g = np.where(np.isnan(cog_g), np.nan, wrap360(np.nan_to_num(cog_g)))

    # heading: both bracketing reports -> circular interpolation, HDG_SRC 0 (rest in finalize)
    h0, h1 = hdg[j], hdg[jn]
    hb = ~np.isnan(h0) & ~np.isnan(h1)
    hdg_g = np.full(len(g), np.nan)
    hdg_g[hb] = circ_interp(h0[hb], h1[hb], s[hb])
    src = np.where(hb, 0, -1).astype(np.int8)

    return dict(
        g=g, lat=la, lon=lo, sog=sog_g, cog=cog_g, heading=hdg_g, HDG_SRC=src,
        INTERP_SPAN=span.astype(np.int64), R_AFTER_S=(t[jn] - g).astype(np.int64),
        P_AGE_S=(g - t[j]).astype(np.int64), P_LAT=lat[j], P_LON=lon[j], P_SOG=sog[j], P_COG=cog[j],
        N_SOG=sog[jn],
    )


def finalize_segment(r, hk_t, hk_v, cfg=CFG):
    """Position-difference velocity (central differences), COG_VALID, STATIONARY, A11 QC,
    heading completion (COG, then past-only hold)."""
    g = r["g"]
    m = len(g)
    idx = np.arange(m)
    step = np.diff(g) == 60
    prev_ok = np.r_[False, step]
    next_ok = np.r_[step, False]
    ip = np.where(prev_ok, idx - 1, idx)
    inx = np.where(next_ok, idx + 1, idx)
    dtv = (g[inx] - g[ip]).astype(float)
    x, y = enu(r["lat"][ip], r["lon"][ip], r["lat"][inx], r["lon"][inx])
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = np.where(dtv > 0, x / dtv, np.nan)
        vy = np.where(dtv > 0, y / dtv, np.nan)
    spd = np.hypot(vx, vy) / KN
    sog = r["sog"]
    cog = r["cog"]
    has_sog = ~np.isnan(sog)
    with np.errstate(invalid="ignore"):
        moving_ref = np.where(has_sog, sog >= cfg["cog_valid_min_sog"], spd >= cfg["cog_valid_min_sog"])
        cog_valid = ~np.isnan(cog) & moving_ref
        stationary = np.where(has_sog, sog < cfg["cog_valid_min_sog"], spd < cfg["cog_valid_min_sog"])
        qc_sog = (sog >= cfg["qc_sog_min"]) & ~np.isnan(spd) & (np.abs(spd / sog - 1.0) > cfg["qc_sog_rel"])
        brg = bearing_xy(vx, vy)
        qc_cog = cog_valid & (spd >= cfg["qc_cog_min_speed"]) & (angdiff(cog, brg) > cfg["qc_cog_deg"])
    hdg = r["heading"].copy()
    src = r["HDG_SRC"].copy()
    use_cog = (src < 0) & cog_valid
    hdg[use_cog] = cog[use_cog]
    src[use_cog] = 1
    rest = src < 0
    if rest.any() and len(hk_t):
        ii = np.searchsorted(hk_t, g[rest], side="right") - 1    # past (<= g) reports only
        okh = ii >= 0
        rr = np.flatnonzero(rest)[okh]
        hdg[rr] = hk_v[ii[okh]]
        src[rr] = 2
    r.update(
        heading=hdg, HDG_SRC=src, vx_pos=vx, vy_pos=vy, COG_VALID=cog_valid.astype(np.int8),
        STATIONARY=stationary.astype(np.int8), QC_SOG=qc_sog.astype(np.int8),
        QC_COG=qc_cog.astype(np.int8), GAP_IMP=(r["INTERP_SPAN"] > cfg["gap_imp_s"]).astype(np.int8),
    )
    return r


GRID_KEYS = ["g", "seg_id", "lat", "lon", "sog", "cog", "COG_VALID", "heading", "HDG_SRC",
             "GAP_IMP", "INTERP_SPAN", "STATIONARY", "QC_SOG", "QC_COG", "vx_pos", "vy_pos",
             "R_AFTER_S", "P_AGE_S", "P_LAT", "P_LON", "P_SOG", "P_COG", "N_SOG"]

STAT_KEYS = ["spikes", "despike_blocked", "cuts_gap", "cuts_jump", "cuts_forced", "segments",
             "segments_short", "segments_kept", "grid_candidates", "grid_drop_gap_gt_B",
             "grid_drop_hole_after_cut", "guard_linearised_intervals", "guard_unfixed_steps"]


def process_vessel(t, lat, lon, sog, cog, hdg, cfg=CFG):
    """A6-A12 for one vessel (all requested days together)."""
    n = len(t)
    st = {k: 0 for k in STAT_KEYS}
    status = np.zeros(n, np.int8)
    keep, blocked = despike(t, lat, lon, sog, cog, cfg)
    status[~keep] = ST_SPIKE
    st["spikes"] = int((~keep).sum())
    st["despike_blocked"] = int(blocked.sum())
    k = np.flatnonzero(keep)
    tk, la, lo, so, co, hd = t[k], lat[k], lon[k], sog[k], cog[k], hdg[k]
    # A7 segmentation
    cut = np.zeros(len(k), bool)                 # cut[i]: new segment starts at kept report i
    if len(k) > 1:
        dt = np.diff(tk).astype(float)
        gap = dt > cfg["G_s"]
        v = implied_kn(hav(la[:-1], lo[:-1], la[1:], lo[1:]), dt, cfg)
        jump = ~gap & (v > a6_thr(so[:-1], so[1:], cfg=cfg))
        forced = ~gap & ~jump & blocked[k[1:]]
        cut[1:] = gap | jump | forced
        st["cuts_gap"] = int(gap.sum())
        st["cuts_jump"] = int(jump.sum())
        st["cuts_forced"] = int(forced.sum())
    seg = np.cumsum(cut)
    # heading hold source: all kept (post-A6) reports of the vessel, past only (AMB-21)
    hmask = ~np.isnan(hd)
    hk_t, hk_v = tk[hmask], hd[hmask]

    bounds = np.flatnonzero(np.r_[True, np.diff(seg) != 0, True])
    out = []
    last_out = None
    seg_out = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        st["segments"] += 1
        if tk[b - 1] - tk[a] < cfg["min_seg_s"]:              # A12 (AMB-23)
            status[k[a:b]] = ST_SHORT
            st["segments_short"] += 1
            continue
        st["segments_kept"] += 1
        r = resample_segment(tk[a:b], la[a:b], lo[a:b], so[a:b], co[a:b], hd[a:b], st, cfg)
        sid = seg_out
        seg_out += 1
        if r is None:
            continue
        if last_out is not None:                              # A7 mandatory hole (AMB-22)
            keep_g = r["g"] >= last_out + 60 * (cfg["hole_min_minutes"] + 1)
            nd = int((~keep_g).sum())
            if nd:
                st["grid_drop_hole_after_cut"] += nd
                r = {kk: vv[keep_g] for kk, vv in r.items()}
            if len(r["g"]) == 0:
                continue
        r = finalize_segment(r, hk_t, hk_v, cfg)
        r["seg_id"] = np.full(len(r["g"]), sid, np.int32)
        last_out = int(r["g"][-1])
        out.append(r)
    grid = None
    if out:
        grid = {kk: np.concatenate([o[kk] for o in out]) for kk in GRID_KEYS}
    return grid, status, st


def process_chunk(payload):
    """Worker: a contiguous block of vessels (rows sorted by mmsi, ts)."""
    mmsi, t, lat, lon, sog, cog, hdg, starts, cfg = payload
    grids, statuses, vstats = [], [], []
    agg = {k: 0 for k in STAT_KEYS}
    for a, b in zip(starts[:-1], starts[1:]):
        grid, status, st = process_vessel(t[a:b], lat[a:b], lon[a:b], sog[a:b], cog[a:b], hdg[a:b], cfg)
        statuses.append(status)
        for kk in STAT_KEYS:
            agg[kk] += st[kk]
        vstats.append((int(mmsi[a]), b - a, st["spikes"], st["cuts_jump"] + st["cuts_forced"]))
        if grid is not None:
            grid["mmsi"] = np.full(len(grid["g"]), mmsi[a], np.int64)
            grids.append(grid)
    gcat = None
    if grids:
        gcat = {kk: np.concatenate([gg[kk] for gg in grids]) for kk in GRID_KEYS + ["mmsi"]}
    return gcat, np.concatenate(statuses) if statuses else np.zeros(0, np.int8), vstats, agg


# ---------------------------------------------------------------------------
# whole build
# ---------------------------------------------------------------------------
def run_stage_a(day_frames, region="socal", cfg=CFG, workers=1, lenient=False, n_days_in_month=31):
    """day_frames: iterable of (day, raw DataFrame from read_day or synthetic, raw_hourly or None).
    Returns grid DataFrame (sorted mmsi, t) and funnel dict."""
    box = build_box(region, cfg)
    frames, per_day_pre, raw_hourly = [], {}, {}
    for day, df, hourly in day_frames:
        f, c = prefilter_day(df, day, box, cfg, lenient)
        frames.append(f)
        per_day_pre[day] = c
        if hourly is not None:
            raw_hourly[day] = hourly
    days = sorted(per_day_pre)
    cols = ["mmsi", "ts", "lat", "lon", "sog", "cog", "heading", "vessel_type", "transceiver", "day"]
    df = pd.concat([f[cols] for f in frames], ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df["order"] = np.arange(len(df))

    # A5 duplicates: same (mmsi, ts) -> most complete record; tie -> first in file order (AMB-12)
    tr = df["transceiver"]
    comp = (df["sog"].notna().astype(int) + df["cog"].notna().astype(int) + df["heading"].notna().astype(int)
            + df["vessel_type"].notna().astype(int) + (tr.notna() & (tr.astype(str).str.strip() != "")).astype(int))
    df["_negcomp"] = -comp
    df = df.sort_values(["mmsi", "ts", "_negcomp", "order"], kind="mergesort").reset_index(drop=True)
    dup = df.duplicated(["mmsi", "ts"], keep="first").to_numpy()
    dup_by_day = pd.Series(df["day"].to_numpy()[dup]).value_counts().to_dict()
    df = df[~dup].reset_index(drop=True)

    cat_code = monthly_mode(df, "vessel_type")
    trx = monthly_mode(df, "transceiver")

    mmsi = df["mmsi"].to_numpy(np.int64)
    arrs = [df[c].to_numpy(dtype=float) for c in ("lat", "lon", "sog", "cog", "heading")]
    t = df["ts"].to_numpy(np.int64)
    rep_day = df["day"].to_numpy()
    vstarts = np.flatnonzero(np.r_[True, mmsi[1:] != mmsi[:-1]]) if len(mmsi) else np.zeros(0, int)
    vstarts = np.r_[vstarts, len(mmsi)]

    # chunks of whole vessels, ~balanced by rows
    nchunks = max(1, workers * 8)
    target = max(1, len(mmsi) // nchunks)
    payloads = []
    i = 0
    while i < len(vstarts) - 1:
        a = vstarts[i]
        jj = i + 1
        while jj < len(vstarts) - 1 and vstarts[jj] - a < target:
            jj += 1
        b = vstarts[jj]
        sl = slice(a, b)
        payloads.append((mmsi[sl], t[sl], *[x[sl] for x in arrs], vstarts[i:jj + 1] - a, cfg))
        i = jj
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_chunk, payloads))
    else:
        results = [process_chunk(p) for p in payloads]

    status = np.concatenate([r[1] for r in results]) if results else np.zeros(0, np.int8)
    agg = {k: 0 for k in STAT_KEYS}
    vstats = []
    for r in results:
        vstats.extend(r[2])
        for kk in STAT_KEYS:
            agg[kk] += r[3][kk]
    gparts = [r[0] for r in results if r[0] is not None]
    if gparts:
        G = {kk: np.concatenate([gp[kk] for gp in gparts]) for kk in GRID_KEYS + ["mmsi"]}
    else:
        G = {kk: np.zeros(0) for kk in GRID_KEYS + ["mmsi"]}

    # MMSI quality (AMB-24)
    qc_list, qc_set = [], set()
    for m_, nrep, nsp, ncut in vstats:
        rate = (nsp + ncut) / nrep if nrep else 0.0
        if nrep >= cfg["mmsi_qc_min_reports"] and rate > cfg["mmsi_qc_rate"]:
            qc_set.add(m_)
            qc_list.append(dict(mmsi=m_, n_reports=nrep, spikes=nsp, jump_cuts=ncut, rate=round(rate, 4)))

    grid = pd.DataFrame({
        "mmsi": G["mmsi"].astype(np.int64),
        "t": (G["g"] // 60).astype(np.int32),
        "seg_id": G["seg_id"].astype(np.int32),
        "lat": G["lat"], "lon": G["lon"], "sog": G["sog"], "cog": G["cog"],
        "COG_VALID": G["COG_VALID"].astype(np.int8),
        "heading": G["heading"], "HDG_SRC": G["HDG_SRC"].astype(np.int8),
    })
    gm = grid["mmsi"].to_numpy()
    grid["vessel_cat"] = pd.Series(gm).map(lambda m_: vessel_cat_of(cat_code.get(m_))).astype(object).to_numpy()
    grid["transceiver"] = pd.Series(gm).map(lambda m_: trx.get(m_)).astype(object).to_numpy()
    grid["GAP_IMP"] = G["GAP_IMP"].astype(np.int8)
    grid["INTERP_SPAN"] = G["INTERP_SPAN"].astype(np.int32)
    grid["STATIONARY"] = G["STATIONARY"].astype(np.int8)
    grid["QC_SOG"] = G["QC_SOG"].astype(np.int8)
    grid["QC_COG"] = G["QC_COG"].astype(np.int8)
    grid["IN_CORE"] = in_core(G["lat"], G["lon"], box).astype(np.int8)
    grid["MMSI_QC"] = np.isin(gm, np.fromiter(qc_set, np.int64, len(qc_set))).astype(np.int8)
    for kk in ("vx_pos", "vy_pos"):
        grid[kk] = G[kk]
    for kk in ("R_AFTER_S", "P_AGE_S"):
        grid[kk] = G[kk].astype(np.int32)
    for kk in ("P_LAT", "P_LON", "P_SOG", "P_COG", "N_SOG"):
        grid[kk] = G[kk]
    grid = grid[[c for c, _ in OUT_COLUMNS]]

    funnel = make_funnel(days, per_day_pre, dup_by_day, rep_day, status, agg, grid, qc_list,
                         raw_hourly, region, cfg, len(vstarts) - 1)
    return grid, funnel


def make_funnel(days, pre, dup_by_day, rep_day, status, agg, grid, qc_list, raw_hourly, region, cfg, n_vessels):
    pre_keys = ["A1_missing_key", "A1_out_of_range", "A1_wrong_day", "A2_outside_box", "A2_invalid_mmsi"]
    rep = {"raw": 0, "dropped": {k: 0 for k in pre_keys + ["A5_duplicate", "A6_spike", "A12_short_segment"]},
           "kept_in_output_segments": 0}
    per_day = {}
    grid_day = (grid["t"].to_numpy() // 1440 + 1) if len(grid) else np.zeros(0, int)
    for d in days:
        c = pre[d]
        sd = status[rep_day == d]
        row = {"raw": c["raw"]}
        for k in pre_keys:
            row[k] = c[k]
        row["A5_duplicate"] = int(dup_by_day.get(d, 0))
        row["A6_spike"] = int((sd == ST_SPIKE).sum())
        row["A12_short_segment"] = int((sd == ST_SHORT).sum())
        row["kept_in_output_segments"] = int((sd == ST_KEPT).sum())
        row["closes"] = row["raw"] == sum(row[k] for k in pre_keys + ["A5_duplicate", "A6_spike",
                                                                        "A12_short_segment",
                                                                        "kept_in_output_segments"])
        row["grid_rows"] = int((grid_day == d).sum())
        gh = grid["t"].to_numpy()[grid_day == d]
        row["grid_rows_per_hour"] = np.bincount((gh % 1440) // 60, minlength=24).tolist()
        if d in raw_hourly:
            row["raw_rows_per_hour"] = raw_hourly[d]
        per_day[d] = row
        rep["raw"] += row["raw"]
        for k in rep["dropped"]:
            rep["dropped"][k] += row[k]
        rep["kept_in_output_segments"] += row["kept_in_output_segments"]
    rep["closes"] = rep["raw"] == sum(rep["dropped"].values()) + rep["kept_in_output_segments"]
    gf = {"candidates": agg["grid_candidates"],
          "dropped": {"A8_inside_gap_gt_B": agg["grid_drop_gap_gt_B"],
                      "A7_hole_after_cut": agg["grid_drop_hole_after_cut"]},
          "output_rows": int(len(grid))}
    gf["closes"] = gf["candidates"] == sum(gf["dropped"].values()) + gf["output_rows"]
    counters = {k: sum(pre[d][k] for d in days) for k in ("A3_heading_nan", "A3_cog_nan", "A3_sog_nan", "A4_sog_nan")}
    counters.update({k: agg[k] for k in STAT_KEYS})
    counters["vessels_after_A5"] = int(n_vessels)
    counters["vessels_in_output"] = int(grid["mmsi"].nunique()) if len(grid) else 0
    out = dict(version=VERSION, region=region, config=cfg, days=days,
               reports=rep, grid=gf, counters=counters, per_day=per_day,
               mmsi_qc=dict(n_flagged=len(qc_list), flagged=qc_list))
    split_lon = REGIONS[region]["split_lon"]
    if split_lon is not None and len(grid):
        mv = (grid["STATIONARY"] == 0) & (grid["IN_CORE"] == 1)
        east = mv & (grid["lon"] > split_lon)
        west = mv & (grid["lon"] <= split_lon)
        me, mw = set(grid.loc[east, "mmsi"]), set(grid.loc[west, "mmsi"])
        out["region_split"] = dict(split_lon=split_lon, moving_core_rows=int(mv.sum()),
                                   east_share=float(east.sum() / max(1, mv.sum())),
                                   vessels_east=len(me), vessels_west=len(mw), vessels_both=len(me & mw))
    return out


# ---------------------------------------------------------------------------
# V4-lite invariants on the output
# ---------------------------------------------------------------------------
def v4_check(grid, region="socal", cfg=CFG):
    box = build_box(region, cfg)
    res = {}
    res["dup_mmsi_t"] = int(grid.duplicated(["mmsi", "t"]).sum())
    res["nan_position"] = int(grid["lat"].isna().sum() + grid["lon"].isna().sum())
    for c in ("cog", "heading", "P_COG"):
        v = grid[c].to_numpy()
        res[f"{c}_out_of_0_360"] = int(((v < 0) | (v >= 360)).sum())
    res["sog_out_of_0_55"] = int(((grid["sog"] < 0) | (grid["sog"] > cfg["sog_max"])).sum())
    res["inside_gap_gt_B"] = int((grid["INTERP_SPAN"] > cfg["B_s"]).sum())
    res["reports_outside_build_box"] = int((~in_build_box(grid["P_LAT"].to_numpy(), grid["P_LON"].to_numpy(), box)).sum())
    bb = dict(box, margin_m=box["margin_m"] + cfg["v4_grid_box_tol_m"])
    res["grid_outside_build_box_50m"] = int((~in_build_box(grid["lat"].to_numpy(), grid["lon"].to_numpy(), bb)).sum())
    g = grid.sort_values(["mmsi", "t"], kind="mergesort")
    m, tt = g["mmsi"].to_numpy(), g["t"].to_numpy()
    same = (m[1:] == m[:-1]) & (g["seg_id"].to_numpy()[1:] == g["seg_id"].to_numpy()[:-1]) & (tt[1:] - tt[:-1] == 1)
    la, lo = g["lat"].to_numpy(), g["lon"].to_numpy()
    v = hav(la[:-1], lo[:-1], la[1:], lo[1:]) / 60.0 / KN
    ps, ns = g["P_SOG"].to_numpy(), g["N_SOG"].to_numpy()
    thr = a6_thr(ps[:-1], ns[:-1], ps[1:], ns[1:], cfg=cfg)
    res["steps_above_A6"] = int((same & (v > thr)).sum())
    # adjacent minutes across a segment boundary (mandatory hole)
    diffseg = (m[1:] == m[:-1]) & (g["seg_id"].to_numpy()[1:] != g["seg_id"].to_numpy()[:-1]) & (tt[1:] - tt[:-1] < 2)
    res["no_hole_between_segments"] = int(diffseg.sum())
    # statistical: implied/SOG within +-10% on >= 90% of moving steps (informative only)
    s = g["sog"].to_numpy()
    mv = same & (s[:-1] >= 3) & (s[1:] >= 3)
    if mv.any():
        ratio = v[mv] / (0.5 * (s[:-1][mv] + s[1:][mv]))
        res["implied_over_sog_within_10pct_share"] = float(np.mean(np.abs(ratio - 1) <= 0.10))
    res["all_hard_checks_zero"] = all(v_ == 0 for k_, v_ in res.items() if isinstance(v_, int) and not isinstance(v_, bool))
    return res


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_days(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def write_days(grid, out_dir, year, month, days):
    os.makedirs(out_dir, exist_ok=True)
    gd = grid["t"].to_numpy() // 1440 + 1
    files = {}
    for d in days:
        sub = grid[gd == d]
        path = os.path.join(out_dir, f"day_{year:04d}-{month:02d}-{d:02d}.parquet")
        pq.write_table(pa.Table.from_pandas(sub, schema=OUT_SCHEMA, preserve_index=False), path,
                       compression="zstd")
        files[os.path.basename(path)] = sha256_file(path)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir")
    ap.add_argument("--days", default="1-31", help="e.g. 1-31 or 1,2,5-7")
    ap.add_argument("--month", default="2025-03")
    ap.add_argument("--region", default="socal", choices=sorted(REGIONS))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--a1_lenient", action="store_true", help="drop+count A1 violations instead of asserting")
    ap.add_argument("--no_hash_inputs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.raw_dir:
        ap.error("--raw_dir is required")
    year, month = map(int, a.month.split("-"))
    days = parse_days(a.days)
    out_dir = a.out_dir or f"v8ref_{a.region}_{a.month}"
    mstart = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    mstart_ep = int(mstart.timestamp())
    ndim = pd.Timestamp(year=year, month=month, day=1).days_in_month
    inputs = {}
    t0 = time.time()

    def frames():
        for d in days:
            p = os.path.join(os.path.expanduser(a.raw_dir), f"ais-{a.month}-{d:02d}.csv.zst")
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            df = read_day(p, mstart_ep)
            ts = df["ts"].to_numpy()
            ok = ~np.isnan(ts)
            hrs = np.floor((ts[ok] - (d - 1) * 86400) / 3600).astype(int)
            hrs = hrs[(hrs >= 0) & (hrs < 24)]
            hourly = np.bincount(hrs, minlength=24).tolist()
            if not a.no_hash_inputs:
                inputs[os.path.basename(p)] = sha256_file(p)
            print(f"[read] {os.path.basename(p)}: {len(df):,} rows ({time.time() - t0:.0f}s)", flush=True)
            yield d, df, hourly

    grid, funnel = run_stage_a(frames(), a.region, CFG, a.workers, a.a1_lenient, ndim)
    print(f"[stageA] {len(grid):,} grid rows ({time.time() - t0:.0f}s)", flush=True)
    outs = write_days(grid, out_dir, year, month, days)
    funnel["v4_lite"] = v4_check(grid, a.region)
    funnel["meta"] = dict(inputs_sha256=inputs, outputs_sha256=outs, python=sys.version.split()[0],
                          numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__,
                          platform=platform.platform(), elapsed_s=round(time.time() - t0, 1),
                          argv=sys.argv[1:])
    with open(os.path.join(out_dir, "funnel.json"), "w", encoding="utf-8") as f:
        json.dump(funnel, f, indent=1, default=_json_default)
    print(json.dumps(dict(reports=funnel["reports"], grid=funnel["grid"], v4_lite=funnel["v4_lite"]),
                     indent=1, default=_json_default))
    ok = funnel["reports"]["closes"] and funnel["grid"]["closes"]
    print("FUNNEL CLOSES" if ok else "FUNNEL DOES NOT CLOSE", flush=True)
    return 0 if ok else 1


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# self-tests with synthetic trajectories
# ---------------------------------------------------------------------------
LAT0, LON0 = 33.0, -118.5


def xy2ll(x, y):
    return enu_inv(LAT0, LON0, np.asarray(x, float), np.asarray(y, float))


def track(times, fx, fy, mmsi=367000001, vtype=70, trx="A", sog=None, cog=None, heading=None, eps=0.5):
    """Synthetic raw reports. fx, fy: callables t -> metres. SOG/COG from true velocity."""
    times = np.asarray(times, dtype=float)
    x, y = fx(times), fy(times)
    vx = (fx(times + eps) - fx(times - eps)) / (2 * eps)
    vy = (fy(times + eps) - fy(times - eps)) / (2 * eps)
    lat, lon = xy2ll(x, y)
    df = pd.DataFrame(dict(mmsi=mmsi, ts=times, lat=lat, lon=lon,
                           sog=np.hypot(vx, vy) / KN if sog is None else sog,
                           cog=bearing_xy(vx, vy) if cog is None else cog,
                           heading=np.nan if heading is None else heading,
                           vessel_type=float(vtype), transceiver=trx))
    df["transceiver"] = df["transceiver"].astype(object)
    return df


def run_synth(df, cfg=CFG, region="socal"):
    df = df.reset_index(drop=True)
    days = (np.floor(df["ts"].to_numpy() / 86400) + 1).astype(int)
    fr = [(int(d), df[days == d].copy(), None) for d in sorted(set(days))]
    return run_stage_a(fr, region, cfg, workers=1)


def phased_times(t0, t1, dt, rng, jitter=5):
    ts = np.arange(t0 + rng.uniform(0, dt), t1, dt) + rng.uniform(-jitter, jitter, size=None)
    return np.round(ts)


def lin(v, x0=0.0):
    return lambda t: x0 + v * t


def zero(t):
    return np.zeros_like(np.asarray(t, float))


def _true_err(grid, fx, fy):
    gs = grid["t"].to_numpy() * 60.0
    la, lo = xy2ll(fx(gs), fy(gs))
    return hav(grid["lat"].to_numpy(), grid["lon"].to_numpy(), la, lo)


def test_a1_adapter_roundtrip(tmp):
    rows = ("vessel_name,LATITUDE,base_date_time,mmsi,longitude,sog,cog,heading,imo,vessel_type,status,transceiver\n"
            "X,33.0,2025-03-02T00:00:05,367000001,-118.5,10.0,90.0,,IMO1,70,,A\n"
            "Y,33.1,2025-03-02 12:00:00,367000002,-118.6,,511,511,,,0,B\n")
    p = os.path.join(tmp, "ais-2025-03-02.csv.zst")
    with pa.CompressedOutputStream(p, "zstd") as f:
        f.write(rows.encode())
    df = read_day(p, int(pd.Timestamp("2025-03-01", tz="UTC").timestamp()))
    assert list(df["mmsi"]) == [367000001, 367000002]
    assert df["ts"].tolist() == [86400 + 5, 86400 + 43200]
    assert abs(df["lat"].iloc[1] - 33.1) < 1e-12 and abs(df["lon"].iloc[0] + 118.5) < 1e-12
    assert np.isnan(df["heading"].iloc[0]) and np.isnan(df["sog"].iloc[1])
    assert df["transceiver"].tolist() == ["A", "B"]
    box = build_box("socal")
    f, c = prefilter_day(df, 2, box)
    assert c["A3_cog_nan"] == 1 and c["A3_heading_nan"] == 1 and len(f) == 2
    try:
        prefilter_day(df, 3, box)
        raise RuntimeError("A1 day assert did not fire")
    except AssertionError:
        pass


def test_a2_box_and_mmsi(tmp):
    rng = np.random.default_rng(1)
    ts = phased_times(3600, 3600 + 1800, 70, rng)
    lat_north = 35.0 + 5000 / 111195.0           # 5 km north of the core: inside margin
    parts = []
    for mmsi, latc in ((367000010, 33.0), (12345678, 33.0), (100000001, 33.0), (345000000, 33.0),
                       (367000011, lat_north), (367000012, 35.0 + 22000 / 111195.0)):
        d = track(ts, lin(5.0), zero, mmsi=mmsi)
        d["lat"] = d["lat"] - LAT0 + latc
        parts.append(d)
    grid, fun = run_synth(pd.concat(parts))
    dr = fun["reports"]["dropped"]
    n = len(ts)
    assert dr["A2_invalid_mmsi"] == 3 * n, dr
    assert dr["A2_outside_box"] == n, dr
    assert set(grid["mmsi"]) == {367000010, 367000011}
    assert grid.loc[grid.mmsi == 367000011, "IN_CORE"].eq(0).all()
    assert grid.loc[grid.mmsi == 367000010, "IN_CORE"].eq(1).all()
    assert fun["reports"]["closes"] and fun["grid"]["closes"]


def test_a3_a4_sentinels_keep_rows(tmp):
    ts = np.arange(0, 1200, 60) + 30.0
    d = track(ts, lin(5.0), zero)
    d.loc[3, "heading"] = 511.0
    d.loc[4, "cog"] = 360.0
    d.loc[5, "sog"] = 102.3
    d.loc[6, "sog"] = 60.0
    grid, fun = run_synth(d)
    c = fun["counters"]
    assert (c["A3_heading_nan"], c["A3_cog_nan"], c["A3_sog_nan"], c["A4_sog_nan"]) == (1, 1, 1, 1), c
    assert fun["reports"]["kept_in_output_segments"] == len(ts)
    assert grid["sog"].max() < 55 and grid["P_SOG"].max(skipna=True) < 55


def test_a5_duplicates_keep_most_complete(tmp):
    ts = np.arange(0, 1200, 60.0)               # reports exactly on the minute
    d = track(ts, lin(5.0), zero)
    dupe = d.iloc[[5]].copy()
    d.loc[5, ["sog", "cog"]] = np.nan           # first-in-file copy is less complete
    dupe["sog"] = 9.7                            # distinctive value
    grid, fun = run_synth(pd.concat([d, dupe]))
    assert fun["reports"]["dropped"]["A5_duplicate"] == 1
    assert abs(grid.loc[grid.t == 5, "P_SOG"].iloc[0] - 9.7) < 1e-9


def test_straight_line_exact_and_r_after(tmp):
    rng = np.random.default_rng(2)
    ts = phased_times(0, 7200, 70, rng)
    fx = lin(10 * KN * math.sin(math.radians(60)))
    fy = lin(10 * KN * math.cos(math.radians(60)))
    d = track(ts, fx, fy)
    grid, fun = run_synth(d)
    err = _true_err(grid, fx, fy)
    assert err.max() < 0.5, err.max()
    assert grid["seg_id"].nunique() == 1 and fun["counters"]["spikes"] == 0
    assert np.allclose(grid["sog"], 10, atol=1e-6) and (angdiff(grid["cog"].to_numpy(), 60) < 1e-6).all()
    rt = set(ts.astype(int))
    g = grid["t"].to_numpy() * 60
    assert all(int(x) in rt for x in g + grid["R_AFTER_S"].to_numpy())
    assert all(int(x) in rt for x in g - grid["P_AGE_S"].to_numpy())
    ex = grid["R_AFTER_S"] == 0
    assert (grid.loc[ex, "INTERP_SPAN"] == 0).all() and (grid.loc[~ex, "INTERP_SPAN"] > 0).all()
    assert (grid["QC_SOG"] == 0).all() and (grid["QC_COG"] == 0).all()
    assert (grid["HDG_SRC"] == 1).all()          # no heading reported -> from COG


def test_turn_known_bound(tmp):
    rng = np.random.default_rng(3)
    v, rad = 10 * KN, 500.0
    w = v / rad
    ts = phased_times(0, 1800, 70, rng)
    fx = lambda t: rad * np.sin(w * t)
    fy = lambda t: rad * (1 - np.cos(w * t))
    grid, fun = run_synth(track(ts, fx, fy))
    err = _true_err(grid, fx, fy)
    assert err.max() < 5.0, err.max()            # Hermite on a 500 m circle, ~70 s sampling
    assert fun["counters"]["spikes"] == 0 and grid["seg_id"].nunique() == 1


def test_gaps_B_and_G(tmp):
    # 10 kn east, 60 s reporting at :30, with gaps of 150 s, 300 s, 480 s, 900 s
    t = list(np.arange(30, 1230, 60.0))          # ... 1170
    t += list(np.arange(1170 + 150, 2400, 60.0))  # 150 s gap
    last = t[-1]
    t += list(np.arange(last + 300, last + 300 + 1200, 60.0))  # 300 s gap (> B)
    last = t[-1]
    t += list(np.arange(last + 480, last + 480 + 1200, 60.0))  # 480 s (> B, < G)
    last = t[-1]
    t += list(np.arange(last + 900, last + 900 + 1200, 60.0))  # 900 s (> G): new segment
    t = np.array(t)
    fx = lin(10 * KN)
    grid, fun = run_synth(track(t, fx, zero))
    assert grid["seg_id"].nunique() == 2, grid["seg_id"].unique()
    assert (grid["INTERP_SPAN"] <= 180).all()
    assert grid["GAP_IMP"].sum() >= 2               # the 150 s gap is bridged and flagged
    gs = grid["t"].to_numpy() * 60
    for a, b in zip(t[:-1], t[1:]):
        if b - a > 180:
            assert not ((gs > a) & (gs < b)).any(), (a, b)
    assert fun["counters"]["cuts_gap"] == 1 and fun["grid"]["closes"]
    assert np.max(_true_err(grid, fx, zero)) < 0.5


def test_spikes_removed_200m_and_2km(tmp):
    rng = np.random.default_rng(4)
    ts = phased_times(0, 3600, 70, rng)
    fx = lin(10 * KN)
    d = track(ts, fx, zero)
    lat_off = lambda m: m / 111195.0
    d.loc[10, "lat"] += lat_off(2000)            # crit 1 (impossible in and out)
    d.loc[30, "lat"] += lat_off(200)             # crit 2 (off the chord, COG agrees)
    grid, fun = run_synth(d)
    assert fun["counters"]["spikes"] == 2 and fun["reports"]["dropped"]["A6_spike"] == 2, fun["counters"]
    assert grid["seg_id"].nunique() == 1 and fun["counters"]["cuts_jump"] == 0
    assert np.max(_true_err(grid, fx, zero)) < 0.5


def test_tight_turns_not_cut(tmp):
    # pleasure craft 6 kn: 180 deg in 60 s, reports every 10 s; then 10 kn 90 deg in 120 s
    v1 = 6 * KN
    r1 = v1 * 60 / math.pi
    w1 = v1 / r1
    ts = np.arange(0, 600, 10.0)
    fx = lambda t: np.where(t < 300, v1 * t, 300 * v1 + r1 * np.sin(w1 * (t - 300)) * (t < 360)
                            + (t >= 360) * (-v1 * (t - 360)))
    fy = lambda t: np.where(t < 300, 0.0, np.where(t < 360, r1 * (1 - np.cos(w1 * (t - 300))), 2 * r1))
    g1, f1 = run_synth(track(ts, fx, fy, vtype=37, trx="B"))
    assert f1["counters"]["spikes"] == 0 and f1["counters"]["cuts_jump"] == 0, f1["counters"]
    assert g1["seg_id"].nunique() == 1 and (g1["vessel_cat"] == "pleasure").all()
    v2 = 10 * KN
    r2 = v2 * 120 / (math.pi / 2)
    w2 = v2 / r2
    fx2 = lambda t: np.where(t < 300, v2 * t, 300 * v2 + r2 * np.sin(np.minimum(w2 * (t - 300), math.pi / 2)))
    fy2 = lambda t: np.where(t < 300, 0.0, r2 * (1 - np.cos(np.minimum(w2 * (t - 300), math.pi / 2)))
                             + np.maximum(0, t - 420) * v2)
    g2, f2 = run_synth(track(np.arange(0, 900, 10.0), fx2, fy2, mmsi=367000002))
    assert f2["counters"]["spikes"] == 0 and f2["counters"]["cuts_jump"] == 0, f2["counters"]


def test_jump_cut_and_mandatory_hole(tmp):
    # straight 8 kn, last report of part 1 at 610 s, teleport 20 km, first of part 2 at 650 s
    t1 = np.arange(10, 611, 30.0)
    t2 = np.arange(650, 1300, 30.0)
    fx = lin(8 * KN)
    d1 = track(t1, fx, zero)
    d2 = track(t2, lambda t: 20000 + 8 * KN * t, zero)
    grid, fun = run_synth(pd.concat([d1, d2]))
    c = fun["counters"]
    assert c["cuts_jump"] == 1 and c["spikes"] == 0 and grid["seg_id"].nunique() == 2, c
    assert fun["grid"]["dropped"]["A7_hole_after_cut"] == 1     # minute 11 (660 s) removed
    s0 = grid[grid.seg_id == 0]["t"].max()
    s1 = grid[grid.seg_id == 1]["t"].min()
    assert s1 - s0 >= 2, (s0, s1)


def test_short_segment_removed(tmp):
    ts = np.arange(0, 241, 30.0)                  # 4 minutes
    grid, fun = run_synth(track(ts, lin(5 * KN), zero))
    assert len(grid) == 0
    assert fun["reports"]["dropped"]["A12_short_segment"] == len(ts) and fun["reports"]["closes"]


def test_cog_wrap_359_0(tmp):
    ts = np.arange(30, 1800, 60.0)
    d = track(ts, zero, lin(5 * KN))
    d["cog"] = np.where(np.arange(len(d)) % 2 == 0, 359.9, 0.1)
    grid, _ = run_synth(d)
    c = grid["cog"].to_numpy()
    assert ((c >= 0) & (c < 360)).all() and (angdiff(c, 0.0) <= 0.1 + 1e-9).all()
    h = grid["heading"].to_numpy()
    assert ((h >= 0) & (h < 360)).all()


def test_stationary_no_heading_kept_and_ffill_past_only(tmp):
    # class B at anchor, reports every 180 s, heading missing in the first hour, 90 deg after
    ts = np.arange(0, 7200, 180.0)
    d = track(ts, zero, zero, trx="B", vtype=37, sog=0.0, cog=np.nan)
    d.loc[d.ts >= 3600, "heading"] = 90.0
    grid, fun = run_synth(d)
    assert len(grid) == (7020 // 60) + 1          # every minute between first and last report
    early = grid[grid.t * 60 < 3600 - 180]
    assert early["heading"].isna().all() and (early["HDG_SRC"] == -1).all()  # no bfill
    late = grid[grid.t * 60 >= 3600]
    assert (late["heading"] == 90.0).all() and late["HDG_SRC"].isin([0, 2]).all()
    assert (grid["STATIONARY"] == 1).all() and (grid["COG_VALID"] == 0).all()


def test_midnight_continuity(tmp):
    rng = np.random.default_rng(5)
    ts = phased_times(86400 - 1800, 86400 + 1800, 70, rng)
    fx = lin(8 * KN, -8 * KN * 86400)
    grid, fun = run_synth(track(ts, fx, zero))
    assert grid["seg_id"].nunique() == 1
    days = grid["t"] // 1440 + 1
    assert set(days) == {1, 2}
    assert (np.diff(grid["t"].to_numpy()) == 1).all()
    assert fun["per_day"][1]["closes"] and fun["per_day"][2]["closes"]
    out = os.path.join(tmp, "out")
    files = write_days(grid, out, 2025, 3, [1, 2])
    back = pd.concat([pd.read_parquet(os.path.join(out, f)) for f in sorted(files)])
    assert len(back) == len(grid)


def test_two_transmitters_mmsi_qc(tmp):
    ts = np.arange(0, 120 * 35, 35.0)
    d = track(ts, lin(10 * KN), zero)
    odd = np.arange(len(d)) % 2 == 1
    d.loc[odd, "lat"] += 10000 / 111195.0         # second transmitter 10 km north
    grid, fun = run_synth(d)
    assert fun["mmsi_qc"]["n_flagged"] == 1, fun["mmsi_qc"]
    assert (grid["MMSI_QC"] == 1).all()


def test_hermite_guard(tmp):
    # SOG says 30 kn (thr 50 kn) but positions move 4500 m / 180 s (48.6 kn): Hermite middle
    # step would be ~57 kn; the guard must linearise.
    ts = np.array([0.0, 180.0, 360.0, 540.0])
    d = track(ts, lin(4500 / 180), zero, sog=30.0, cog=90.0)
    grid, fun = run_synth(d)
    assert fun["counters"]["guard_linearised_intervals"] >= 1, fun["counters"]
    assert v4_check(grid)["steps_above_A6"] == 0
    assert np.max(_true_err(grid, lin(4500 / 180), zero)) < 0.5


def test_funnel_closes_and_v4(tmp):
    rng = np.random.default_rng(6)
    parts = []
    for i in range(6):
        ts = phased_times(i * 4000, i * 4000 + 3 * 3600, 70, rng)
        d = track(ts, lin(rng.uniform(2, 15) * KN), lin(rng.uniform(-3, 3) * KN), mmsi=367100000 + i,
                  vtype=[30, 52, 60, 70, 80, 37][i])
        parts.append(d)
        parts.append(d.iloc[::17])               # duplicates
    d = pd.concat(parts, ignore_index=True)
    d.loc[50, "lat"] += 0.02                      # one spike
    grid, fun = run_synth(d)
    assert fun["reports"]["closes"] and fun["grid"]["closes"]
    assert all(r["closes"] for r in fun["per_day"].values())
    v4 = v4_check(grid)
    assert v4["all_hard_checks_zero"], v4
    assert set(grid["vessel_cat"]) == {"fishing", "tug", "passenger", "cargo", "tanker", "pleasure"}


def selftest():
    import tempfile
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in tests:
            try:
                fn(tmp)
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
    print(f"{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
