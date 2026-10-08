"""
preprocess_dec2021_v2.py — NOAA MarineCadastre Dec 2021 (San Diego box) -> 1-min multi-vessel scenes.

Why v2 (see diag_resampling.md): v1 snapped every report to the NEXT whole minute
(dt.ceil + keep first + resample().interpolate, inherited from the public Sekhon & Fleming
code). With ~70 s report intervals this compresses time and injects along-track jitter of
~120 m (median) into inputs AND ground truth, and makes constant-velocity extrapolation from
positions 2x worse than dead reckoning. v1 also DROPPED every row with Heading = 511
(65% of the rows in the box).

What changes (design follows EnvShip-Bench, build/stages 05/07/08 + pipeline_utils.py;
the numeric thresholds are ours because EnvShip's config file is not in their repo):
  1. Time-aware resampling: positions are linearly interpolated in a local metric plane AT
     the 1-min grid instants from the bracketing reports (grid start = ceil, end = floor);
     no timestamp snapping.  [EnvShip stage 08]
  2. Sentinels become missing instead of dropping the row: Heading 511, COG >= 360,
     SOG >= 102.2. Heading gaps are filled from COG (when moving) and then held within the
     segment; HDG_SRC records where each value came from.  [EnvShip pipeline_utils]
  3. Angles are interpolated circularly (no 359->0 artefact).  [EnvShip circular_interp_deg]
  4. Glitches: one-point position spikes are removed (despike); remaining implausible jumps
     (implied speed) cut the segment as in EnvShip stage 05, and the cut always leaves a hole of
     at least one minute so a track never "teleports" between consecutive minutes.
  5. GAP_IMP flag = grid point bridged across a reporting gap > 120 s.  [EnvShip 07: short gap]
  6. frame_id = minutes since 2021-12-01 00:00 UTC, so "consecutive frame_id" in dataset.py
     means consecutive minutes (v1 numbered the distinct timestamps instead).

Kept from v1 (for comparability): region box, SOG <= 22 kn rows, keep vessels that exceed
1 kn at some point in the month, timestamps with > MIN_VESSELS vessels, split by day
(1-19 / 20-25 / 26-31), z-score stats from train days, columns
frame_id, vessel_id, LON, LAT, SOG, Heading (Heading in degrees, z-scored, as in v1) so the
existing dataset.py works unchanged.

Extra columns (NaN-free, so dataset.py's dropna() keeps the rows; ignored by the loader):
  HDG_SIN, HDG_COS, COG_SIN, COG_COS   raw (not z-scored) — for a later sin/cos feature set
  HDG_SRC   0 heading report, 1 COG (heading missing, SOG >= 1 kn), 2 held from a neighbour value
  GAP_IMP   1 if the grid point bridges a reporting gap > GAP_IMP_S

Two variants (run both; they isolate the two changes):
  --heading_policy drop : timing fix ONLY, same vessel population as v1 (rows with Heading 511
                          removed, as v1). Clean ablation of the snapping artefact.
  --heading_policy fill : full v2 (rows with Heading 511 kept). Larger scenes (more vessels).

Usage (on the RTX, where the raw zips are; CPU only; writes to a NEW folder):
    python preprocess_dec2021_v2.py --raw_dir ~/projects/trajectory_prediction_old/data/raw/2021_12 \
        --heading_policy drop --out_base dataset/noaa_dec2021_1min_v2a
    python preprocess_dec2021_v2.py --raw_dir ~/projects/trajectory_prediction_old/data/raw/2021_12 \
        --heading_policy fill --out_base dataset/noaa_dec2021_1min_v2b
    python preprocess_dec2021_v2.py --raw_dir ... --days 26,27 --out_base /tmp/v2_test   # quick test
"""

from __future__ import annotations

import os
import io
import re
import glob
import json
import zipfile
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---- kept from v1 ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 30.0, 35.0
LON_MIN, LON_MAX = -120.0, -115.0
SOG_MAX = 22.0             # knots, rows above are dropped (as v1)
SOG_MOVING = 1.0           # keep vessels with SOG >= 1 kn at some point
MIN_VESSELS = 3            # v1 keeps timestamps with count > MIN_VESSELS (i.e. >= 4); kept as is
TRAIN_DAYS = list(range(1, 20))
VAL_DAYS = list(range(20, 26))
TEST_DAYS = list(range(26, 32))
FEATURE_COLS = ['LON', 'LAT', 'SOG', 'Heading']
MONTH_START = pd.Timestamp('2021-12-01 00:00:00')

# ---- v2 parameters (ours; EnvShip design) -------------------------------------------------
GRID_S = 60                # 1-min grid
SEG_GAP_S = 600            # cut a segment when reports are > 10 min apart (v1 MAX_GAP_MIN)
BRIDGE_MAX_S = 360         # do not create grid points inside a reporting gap longer than this
                           # (v1: up to 5 interpolated minutes inside gaps <= 10 min)
GAP_IMP_S = 120            # flag grid points that bridge a gap > 120 s (EnvShip short-gap limit)
IMPLIED_CAP_KN = SOG_MAX + 5.0   # implied speed between consecutive reports > 27 kn -> glitch.
                                 # A fixed cap (not EnvShip's max(factor*SOG, cap)) because rows with
                                 # SOG > 22 kn are dropped, so nothing legitimate exceeds it. With every
                                 # kept report pair <= 27 kn, the piecewise-linear 1-min track can never
                                 # step faster than 27 kn either (checked at the end of main).
SOG_INTERP_DIFF_KN = 5.0   # interpolate SOG only if the two reports differ by <= 5 kn, else hold
SOG_SENTINEL = 102.2
R_EARTH = 6371008.8
KNOTS_PER_MPS = 1.9438444924406


def load_zip(path):
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.endswith('.csv')][0]
        with zf.open(name) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'),
                               usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading'],
                               low_memory=False)


def clean_raw(df, heading_policy='fill'):
    df = df.copy()
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
    df = df.dropna(subset=['BaseDateTime', 'MMSI', 'LAT', 'LON'])
    df['MMSI'] = df['MMSI'].astype(str).str.strip()
    df = df[df['MMSI'].str.match(r'^\d{9}$')]
    df = df[(df.LAT >= LAT_MIN) & (df.LAT <= LAT_MAX) & (df.LON >= LON_MIN) & (df.LON <= LON_MAX)]
    # sentinels -> missing (row kept)
    df.loc[df['SOG'] >= SOG_SENTINEL, 'SOG'] = np.nan
    df.loc[(df['COG'] < 0) | (df['COG'] >= 360), 'COG'] = np.nan
    df.loc[(df['Heading'] < 0) | (df['Heading'] >= 360), 'Heading'] = np.nan
    # genuine speeds above SOG_MAX: dropped as in v1
    df = df[~(df['SOG'] > SOG_MAX) & ~(df['SOG'] < 0)]
    if heading_policy == 'drop':
        # v1 behaviour: rows without a valid heading are removed (same vessel population as v1)
        df = df[df['Heading'].notna()]
    return df


def circ_interp(a0, a1, alpha):
    """Vectorised circular interpolation in degrees; if one side is missing, hold the other."""
    r0, r1 = np.deg2rad(a0), np.deg2rad(a1)
    s = (1 - alpha) * np.sin(r0) + alpha * np.sin(r1)
    c = (1 - alpha) * np.cos(r0) + alpha * np.cos(r1)
    out = np.rad2deg(np.arctan2(s, c)) % 360.0
    out = np.where(np.isnan(a0), a1, out)
    out = np.where(np.isnan(a1), a0, out)
    return out


def hold_fill(v):
    """forward fill then backward fill a 1-D float array (within one segment)."""
    s = pd.Series(v)
    return s.ffill().bfill().to_numpy()


def _implied_and_thr(t, lat, lon, sog, i, j):
    """implied speed (kn) and its threshold between report indices i and j (arrays)."""
    k = np.pi / 180.0
    dt = (t[j] - t[i]).astype(float)
    dy = (lat[j] - lat[i]) * k * R_EARTH
    dx = (lon[j] - lon[i]) * k * R_EARTH * np.cos(0.5 * (lat[j] + lat[i]) * k)
    implied = np.hypot(dx, dy) / np.maximum(dt, 1.0) * KNOTS_PER_MPS
    thr = np.full_like(implied, IMPLIED_CAP_KN)
    return implied, thr


def despike(t, lat, lon, sog, passes=3):
    """Remove one-point position spikes: i -> i+1 and i+1 -> i+2 implausible, i -> i+2 plausible.
    Returns a boolean keep-mask. (Persistent shifts are left to segment_reports.)"""
    keep = np.ones(len(t), dtype=bool)
    for _ in range(passes):
        idx = np.where(keep)[0]
        if len(idx) < 3:
            break
        a, b, c = idx[:-2], idx[1:-1], idx[2:]
        v_ab, t_ab = _implied_and_thr(t, lat, lon, sog, a, b)
        v_bc, t_bc = _implied_and_thr(t, lat, lon, sog, b, c)
        v_ac, t_ac = _implied_and_thr(t, lat, lon, sog, a, c)
        spike = (v_ab > t_ab) & (v_bc > t_bc) & (v_ac <= t_ac) & ((t[c] - t[a]) <= SEG_GAP_S)
        if not spike.any():
            break
        keep[b[spike]] = False
    return keep


def segment_reports(t, lat, lon, sog):
    """Segment ids on raw reports: time gap or implausible implied speed (EnvShip stage 05)."""
    k = np.pi / 180.0
    dt = np.diff(t)
    dy = (lat[1:] - lat[:-1]) * k * R_EARTH
    dx = (lon[1:] - lon[:-1]) * k * R_EARTH * np.cos(0.5 * (lat[1:] + lat[:-1]) * k)
    implied = np.hypot(dx, dy) / dt * KNOTS_PER_MPS
    thr = np.full_like(implied, IMPLIED_CAP_KN)
    cut = (dt > SEG_GAP_S) | (implied > thr)
    return np.concatenate([[0], np.cumsum(cut)]), int(((implied > thr) & (dt <= SEG_GAP_S)).sum())


def resample_segment(t, lat, lon, sog, cog, hdg):
    """Time-aware resampling of one segment onto the 1-min grid (EnvShip stage 08 logic)."""
    g0 = int(np.ceil(t[0] / GRID_S) * GRID_S)
    g1 = int(np.floor(t[-1] / GRID_S) * GRID_S)
    if g1 < g0:
        return None
    grid = np.arange(g0, g1 + 1, GRID_S, dtype=np.int64)
    right = np.searchsorted(t, grid, side='left')
    right = np.clip(right, 0, len(t) - 1)
    exact = t[right] == grid
    left = np.where(exact, right, right - 1)
    left = np.clip(left, 0, len(t) - 1)
    span = (t[right] - t[left]).astype(float)
    alpha = np.where(span > 0, (grid - t[left]) / np.where(span > 0, span, 1.0), 0.0)
    keep = span <= BRIDGE_MAX_S
    if not keep.any():
        return None

    k = np.pi / 180.0
    lat0, lon0 = lat[0], lon[0]
    x = (lon - lon0) * k * R_EARTH * np.cos(lat0 * k)
    y = (lat - lat0) * k * R_EARTH
    xg = (1 - alpha) * x[left] + alpha * x[right]
    yg = (1 - alpha) * y[left] + alpha * y[right]
    latg = lat0 + yg / (R_EARTH * k)
    long_ = lon0 + xg / (R_EARTH * k * np.cos(lat0 * k))

    sl, sr = sog[left], sog[right]
    both = ~np.isnan(sl) & ~np.isnan(sr)
    sogg = np.where(both & (np.abs(sr - sl) <= SOG_INTERP_DIFF_KN), (1 - alpha) * sl + alpha * sr,
                    np.where(np.isnan(sl), sr, sl))
    cogg = circ_interp(cog[left], cog[right], alpha)
    hdgg = circ_interp(hdg[left], hdg[right], alpha)

    out = pd.DataFrame({
        't': grid, 'LAT': latg, 'LON': long_, 'SOG': sogg, 'COG': cogg, 'HDG': hdgg,
        'GAP_IMP': (span > GAP_IMP_S).astype(np.int8),
    })[keep].reset_index(drop=True)
    # a hole created by `keep` splits the grid run: that is fine, dataset.py checks continuity
    out['SOG'] = hold_fill(out['SOG'].to_numpy())
    if out['SOG'].isna().all():
        return None
    # heading: report -> COG when moving -> held within segment
    src = np.zeros(len(out), dtype=np.int8)
    h = out['HDG'].to_numpy().copy()
    use_cog = np.isnan(h) & ~np.isnan(out['COG'].to_numpy()) & (out['SOG'].to_numpy() >= SOG_MOVING)
    h[use_cog] = out['COG'].to_numpy()[use_cog]
    src[use_cog] = 1
    miss = np.isnan(h)
    h = hold_fill(h)
    src[miss & ~np.isnan(h)] = 2
    out['HDG'] = h
    out['COG'] = hold_fill(out['COG'].to_numpy())
    out['COG'] = np.where(np.isnan(out['COG']), out['HDG'], out['COG'])
    out['HDG_SRC'] = src
    return out.dropna(subset=['HDG'])


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', type=str, required=True)
    p.add_argument('--out_base', type=str, default='dataset/noaa_dec2021_1min_v2')
    p.add_argument('--days', type=str, default=None, help='subset of days for a quick test, e.g. 26,27')
    p.add_argument('--heading_policy', type=str, default='fill', choices=['fill', 'drop'],
                   help="fill: keep rows with Heading 511 and fill from COG/hold (full v2). "
                        "drop: remove them as v1 did -> timing fix only, v1 vessel population")
    return p.parse_args()


def main():
    args = get_args()
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.raw_dir), 'AIS_2021_12_*.zip')))
    if args.days:
        want = {int(d) for d in args.days.split(',')}
        files = [f for f in files if int(re.search(r'_(\d{2})\.zip$', f).group(1)) in want]
    assert files, f'no AIS_2021_12_*.zip in {args.raw_dir}'
    print(f'{len(files)} raw files')

    frames, n_raw_box = [], 0
    for i, fp in enumerate(files, 1):
        print(f'  [{i:02d}/{len(files)}] {os.path.basename(fp)}', end='\r')
        c = clean_raw(load_zip(fp), args.heading_policy)
        n_raw_box += len(c)
        frames.append(c)
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.groupby('MMSI').filter(lambda g: g['SOG'].max() >= SOG_MOVING)
    print(f'\nrows in box after cleaning: {n_raw_box:,} | moving vessels: {df.MMSI.nunique():,} '
          f'| Heading missing: {100 * df.Heading.isna().mean():.1f}% (kept, filled later)')

    out_parts, n_speed_cuts, n_seg, n_spikes, n_boundary = [], 0, 0, 0, 0
    for mmsi, vd in df.groupby('MMSI', sort=False):
        vd = vd.sort_values('BaseDateTime')
        t = vd['BaseDateTime'].values.astype('datetime64[s]').astype(np.int64)
        # duplicates: keep the most complete record per timestamp
        vd = vd.assign(_t=t, _nn=vd[['SOG', 'COG', 'Heading']].notna().sum(axis=1))
        vd = vd.sort_values(['_t', '_nn']).drop_duplicates('_t', keep='last')
        t = vd['_t'].to_numpy()
        lat, lon = vd['LAT'].to_numpy(float), vd['LON'].to_numpy(float)
        sog, cog, hdg = (vd[c].to_numpy(float) for c in ('SOG', 'COG', 'Heading'))
        if len(t) < 2:
            continue
        kp = despike(t, lat, lon, sog)
        n_spikes += int((~kp).sum())
        t, lat, lon, sog, cog, hdg = t[kp], lat[kp], lon[kp], sog[kp], cog[kp], hdg[kp]
        if len(t) < 2:
            continue
        seg, cuts = segment_reports(t, lat, lon, sog)
        n_speed_cuts += cuts
        vparts = []
        for s in np.unique(seg):
            m = seg == s
            if m.sum() < 2:
                continue
            r = resample_segment(t[m], lat[m], lon[m], sog[m], cog[m], hdg[m])
            if r is None or r.empty:
                continue
            r['seg'] = s
            vparts.append(r)
        if not vparts:
            continue
        v = pd.concat(vparts, ignore_index=True).sort_values('t')
        # a speed cut must leave a hole: if the first minute of a new segment directly follows the
        # last minute of the previous one, drop it (otherwise the track "teleports" between minutes)
        adj = (v['seg'].values[1:] != v['seg'].values[:-1]) & (np.diff(v['t'].values) <= GRID_S)
        if adj.any():
            n_boundary += int(adj.sum())
            v = v.drop(v.index[1:][adj])
        v = v.drop(columns='seg')
        v['MMSI'] = mmsi
        out_parts.append(v)
        n_seg += len(vparts)
    res = pd.concat(out_parts, ignore_index=True)
    del out_parts, df
    res = res.drop_duplicates(['MMSI', 't'], keep='first')   # safety; should be a no-op
    print(f'one-point spikes removed: {n_spikes:,} | segments resampled: {n_seg:,} | '
          f'implied-speed cuts (after despiking): {n_speed_cuts:,} | boundary minutes dropped: {n_boundary:,} | '
          f'1-min rows: {len(res):,}')

    # scene filter (as v1): timestamps with > MIN_VESSELS vessels
    counts = res.groupby('t')['MMSI'].transform('count')
    res = res[counts > MIN_VESSELS].copy()

    ts = pd.to_datetime(res['t'], unit='s')
    res['frame_id'] = ((ts - MONTH_START).dt.total_seconds() // 60).astype(np.int64)
    res['day'] = ts.dt.day
    mmsis = sorted(res['MMSI'].unique())
    res['vessel_id'] = res['MMSI'].map({m: i for i, m in enumerate(mmsis)})
    res = res.rename(columns={'HDG': 'Heading'})
    for name, col in (('HDG', 'Heading'), ('COG', 'COG')):
        rad = np.deg2rad(res[col].to_numpy())
        res[f'{name}_SIN'], res[f'{name}_COS'] = np.sin(rad), np.cos(rad)

    # diagnostics on the output (raw units): implied speed / SOG for moving vessels
    r2 = res.sort_values(['vessel_id', 'frame_id'])
    same = (r2['vessel_id'].values[1:] == r2['vessel_id'].values[:-1]) & (np.diff(r2['frame_id'].values) == 1)
    k = np.pi / 180.0
    la, lo = r2['LAT'].to_numpy(), r2['LON'].to_numpy()
    d = R_EARTH * k * np.hypot(la[1:] - la[:-1], (lo[1:] - lo[:-1]) * np.cos(la[:-1] * k))
    sg = r2['SOG'].to_numpy()
    sel = same & (sg[:-1] >= 3) & (sg[1:] >= 3)
    ratio = d[sel] / (0.5 * (sg[:-1] + sg[1:])[sel] * 1852 / 60)
    print(f'output check — implied speed / SOG (>= 3 kn): p10/p50/p90 = '
          f'{np.percentile(ratio, 10):.2f} / {np.median(ratio):.2f} / {np.percentile(ratio, 90):.2f}, '
          f'within ±10%: {100 * np.mean(np.abs(ratio - 1) <= 0.1):.1f}%')
    tele = same & (d > IMPLIED_CAP_KN / KNOTS_PER_MPS * GRID_S)
    print(f'output check — consecutive-minute steps faster than {IMPLIED_CAP_KN:.0f} kn (any SOG): '
          f'{int(tele.sum()):,} (should be ~0)')
    print(f'HDG_SRC share: report {100 * (res.HDG_SRC == 0).mean():.1f}% | COG {100 * (res.HDG_SRC == 1).mean():.1f}% '
          f'| held {100 * (res.HDG_SRC == 2).mean():.1f}%   GAP_IMP: {100 * res.GAP_IMP.mean():.1f}%')

    train = res[res['day'].isin(TRAIN_DAYS)]
    src_for_stats = train if len(train) else res
    stats = {}
    for c in FEATURE_COLS:
        mu, sd = float(src_for_stats[c].mean()), float(src_for_stats[c].std())
        stats[c] = {'mean': mu, 'std': sd if sd > 0 and np.isfinite(sd) else 1.0}
    if not len(train):
        print('!! no train days in this run: stats computed on the subset (test mode only)')
    for c, s in stats.items():
        res[c] = (res[c] - s['mean']) / s['std']

    out_cols = ['frame_id', 'vessel_id', 'LON', 'LAT', 'SOG', 'Heading',
                'HDG_SIN', 'HDG_COS', 'COG_SIN', 'COG_COS', 'HDG_SRC', 'GAP_IMP']
    totals = {}
    for split, days in (('train', TRAIN_DAYS), ('val', VAL_DAYS), ('test', TEST_DAYS)):
        os.makedirs(os.path.join(args.out_base, split), exist_ok=True)
        for day in days:
            dd = res[res['day'] == day]
            if dd.empty:
                continue
            dd = dd.sort_values(['frame_id', 'vessel_id'])[out_cols]
            dd.to_csv(os.path.join(args.out_base, split, f'day_2021_12_{day:02d}.csv'), index=False)
            totals[split] = totals.get(split, 0) + len(dd)
    with open(os.path.join(args.out_base, 'global_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    meta = dict(version='v2', heading_policy=args.heading_policy, grid_s=GRID_S, seg_gap_s=SEG_GAP_S, bridge_max_s=BRIDGE_MAX_S,
                gap_imp_s=GAP_IMP_S, implied_cap_kn=IMPLIED_CAP_KN,
                sog_interp_diff_kn=SOG_INTERP_DIFF_KN, sog_max=SOG_MAX, min_vessels_gt=MIN_VESSELS,
                rows=totals, vessels=len(mmsis))
    with open(os.path.join(args.out_base, 'build_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'rows per split: {totals} | vessels: {len(mmsis):,}\nsaved -> {args.out_base}/')


if __name__ == '__main__':
    main()