"""
diag_resampling.py — does the 1-min resampling in preprocess_dec2021_merged.py inject
along-track position noise? (CPU only, read-only: writes nothing except an optional report.)

Hypothesis: resample_vessel() snaps every AIS report to the NEXT whole minute
(dt.ceil("1min")) and keeps the FIRST report per minute, so a position is labelled with
a time that is 0-60 s later than when it was measured. If the report phase drifts, the
shift changes from step to step -> along-track jitter of up to one minute of travel,
in the inputs AND in the ground truth. SOG/Heading are instantaneous and unaffected,
which would explain why dead reckoning (SOG+Heading) beats constant velocity from
positions by 2x on moving vessels.

Part A  raw NOAA reports (selected days): report intervals, second-of-minute, ceil shift,
        reports discarded by drop_duplicates, rows dropped by the Heading filter (511).
Part B  the same vessels resampled two ways and compared on the same 1-min grid:
          (i)  CURRENT : resample_vessel() imported from preprocess_dec2021_merged.py
          (ii) TIME-AWARE: linear interpolation at the true report times (np.interp)
        -> position difference (m), implied-speed / SOG ratio, and single-vessel
           cv3 / dr errors on 10->10 min windows (inputs and GT from the same method).
Part C  the PROCESSED split actually used for training/evaluation: implied-speed / SOG
        ratio between consecutive frames for moving vessels (direct check, no raw needed).

Usage (repo root on the DGX; limit threads, the server runs out of them otherwise):
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        python diag_resampling.py --days 26,27 --out diag_resampling.md
    python diag_resampling.py --skip_raw          # only Part C (fast)
"""

from __future__ import annotations
import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

import preprocess_dec2021_merged as P   # reuse the exact cleaning + resampling code

KN = 1852.0 / 3600.0                    # knots -> m/s
R_EARTH = 6371008.8


def metres(lat1, lon1, lat2, lon2):
    k = np.pi / 180.0
    dy = (lat2 - lat1) * k * R_EARTH
    dx = (lon2 - lon1) * k * R_EARTH * np.cos(0.5 * (lat1 + lat2) * k)
    return np.hypot(dx, dy)


def q(x, ps=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 'n/a'
    return ' / '.join(f'{v:.2f}' for v in np.percentile(x, ps)) + f'  (n={len(x):,})'


def ratio_summary(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 'n/a'
    within = 100 * np.mean(np.abs(r - 1) <= 0.1)
    return f'p10/p25/p50/p75/p90 = {q(r)} ; within ±10% of SOG: {within:.1f}%'


# ---------------------------------------------------------------------------
# (ii) time-aware resampling — same gap rules as resample_vessel()
# ---------------------------------------------------------------------------

def resample_timeaware(vd: pd.DataFrame) -> pd.DataFrame:
    vd = vd.sort_values('BaseDateTime').drop_duplicates('BaseDateTime', keep='last')
    t = vd['BaseDateTime'].values.astype('datetime64[s]').astype(np.int64).astype(float)
    lat, lon = vd['LAT'].values, vd['LON'].values
    sog, hdg = vd['SOG'].values, vd['Heading'].values
    seg_id = np.concatenate([[0], np.cumsum(np.diff(t) > P.MAX_GAP_MIN * 60)])
    out = []
    for s in np.unique(seg_id):
        m = seg_id == s
        if m.sum() < 2:
            continue
        ts = t[m]
        g0, g1 = np.ceil(ts[0] / 60) * 60, np.floor(ts[-1] / 60) * 60
        if g1 < g0:
            continue
        grid = np.arange(g0, g1 + 1, 60.0)
        prev = np.searchsorted(ts, grid, side='right') - 1
        ok = (grid - ts[prev]) <= P.INTERP_LIMIT * 60          # same "limit" spirit
        df = pd.DataFrame({
            'BaseDateTime': pd.to_datetime(grid[ok].astype(np.int64), unit='s'),
            'LAT': np.interp(grid, ts, lat[m])[ok],
            'LON': np.interp(grid, ts, lon[m])[ok],
            'SOG': sog[m][prev][ok],                             # last report (ffill)
            'Heading': hdg[m][prev][ok],
        })
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# single-vessel physics on a regular 1-min track
# ---------------------------------------------------------------------------

def windows_errors(lat, lon, sog, hdg, obs=10, pred=10, stride=20, min_kn=3.0):
    """cv3 and dr ADE/FDE (m) on non-overlapping 10->10 windows of one contiguous track."""
    res = []
    L = obs + pred
    for s in range(0, len(lat) - L + 1, stride):
        sl = slice(s, s + L)
        la, lo = lat[sl], lon[sl]
        if np.mean(sog[s:s + obs]) < min_kn:
            continue
        lat0, lon0 = la[obs - 1], lo[obs - 1]
        k = np.pi / 180.0
        x = (lo - lon0) * k * R_EARTH * np.cos(lat0 * k)
        y = (la - lat0) * k * R_EARTH
        P_ = np.stack([x, y], -1)
        Pobs, G = P_[:obs], P_[obs:]
        steps = np.arange(1, pred + 1)[:, None]
        v3 = (Pobs[-1] - Pobs[-4]) / 3.0
        cv3 = Pobs[-1] + steps * v3
        sp = sog[s + obs - 1] * KN * 60.0
        h = np.deg2rad(hdg[s + obs - 1])
        dr = Pobs[-1] + steps * np.array([sp * np.sin(h), sp * np.cos(h)])
        e_cv = np.linalg.norm(cv3 - G, axis=-1)
        e_dr = np.linalg.norm(dr - G, axis=-1)
        res.append((e_cv.mean(), e_cv[-1], e_dr.mean(), e_dr[-1]))
    return res


def contiguous_runs(times_min):
    """Split an increasing integer-minute array into runs of consecutive minutes."""
    br = np.where(np.diff(times_min) != 1)[0] + 1
    return np.split(np.arange(len(times_min)), br)


def implied_ratio(df):
    """|displacement| per minute / (SOG in m/min) for consecutive minutes with SOG >= 3 kn."""
    tm = (df['BaseDateTime'].values.astype('datetime64[m]').astype(np.int64))
    consec = np.diff(tm) == 1
    d = metres(df['LAT'].values[:-1], df['LON'].values[:-1], df['LAT'].values[1:], df['LON'].values[1:])
    sog = df['SOG'].values
    sog_avg = 0.5 * (sog[:-1] + sog[1:])
    sel = consec & (sog[:-1] >= 3) & (sog[1:] >= 3)
    return d[sel] / (sog_avg[sel] * KN * 60.0)


# ---------------------------------------------------------------------------

def part_raw(args, lines):
    days = [int(d) for d in args.days.split(',')]
    lines.append(f'## Part A — raw reports (days {days}, region of preprocess_dec2021_merged.py)\n')
    intervals, secs, shifts, dup_frac = [], [], [], []
    n_rows_region, n_511 = 0, 0
    vessels = []
    for d in days:
        zp = os.path.join(args.raw_dir, f'AIS_2021_12_{d:02d}.zip')
        if not os.path.exists(zp):
            lines.append(f'- missing {zp}, skipped')
            continue
        raw = P.load_zip(zp)
        raw['BaseDateTime'] = pd.to_datetime(raw['BaseDateTime'], errors='coerce')
        reg = raw[(raw.LAT >= P.LAT_MIN) & (raw.LAT <= P.LAT_MAX) &
                  (raw.LON >= P.LON_MIN) & (raw.LON <= P.LON_MAX) &
                  (raw.SOG >= 0) & (raw.SOG <= P.SOG_MAX)]
        n_rows_region += len(reg)
        n_511 += int((reg.Heading > P.HDG_MAX).sum())
        clean = P.clean_raw(raw)
        del raw, reg
        clean = P.keep_moving(clean)
        for mmsi, vd in clean.groupby('MMSI'):
            vd = vd.sort_values('BaseDateTime')
            t = vd['BaseDateTime'].values.astype('datetime64[s]').astype(np.int64)
            if len(t) < 3:
                continue
            intervals.append(np.diff(t))
            secs.append(t % 60)
            shifts.append((-t) % 60)                     # ceil(t) - t, seconds
            ceiled = np.ceil(t / 60.0)
            dup_frac.append(1 - len(np.unique(ceiled)) / len(ceiled))
            vessels.append(vd[['BaseDateTime', 'LON', 'LAT', 'SOG', 'Heading']].copy())
    if not vessels:
        lines.append('No raw data found — check --raw_dir.\n')
        return []
    iv = np.concatenate(intervals)
    iv = iv[iv > 0]
    sc = np.concatenate(secs)
    sh = np.concatenate(shifts)
    lines.append(f'- moving vessels analysed: {len(vessels):,}')
    lines.append(f'- report interval (s) p10/p25/p50/p75/p90: {q(iv)}')
    lines.append(f'- share of intervals in [55, 65] s: {100 * np.mean((iv >= 55) & (iv <= 65)):.1f}%  '
                 f'| < 30 s: {100 * np.mean(iv < 30):.1f}%')
    hist = np.histogram(sc, bins=6, range=(0, 60))[0]
    lines.append(f'- second-of-minute histogram (10 s bins): {", ".join(str(h) for h in hist)} '
                 f'(flat = random phase)')
    lines.append(f'- ceil shift (s) p10/p25/p50/p75/p90: {q(sh)}')
    lines.append(f'- mean share of reports discarded by drop_duplicates after ceil: {100 * np.mean(dup_frac):.1f}%')
    lines.append(f'- rows in region with Heading > 360 (e.g. 511) DROPPED by clean_raw: '
                 f'{n_511:,} / {n_rows_region:,} ({100 * n_511 / max(n_rows_region, 1):.1f}%)\n')
    return vessels


def part_compare(vessels, lines):
    lines.append('## Part B — current vs time-aware resampling on the same vessels\n')
    pos_diff, r_cur, r_ta = [], [], []
    err = {'current': [], 'timeaware': []}
    for vd in vessels:
        a = P.resample_vessel(vd)
        b = resample_timeaware(vd)
        if a.empty or b.empty:
            continue
        r_cur.append(implied_ratio(a))
        r_ta.append(implied_ratio(b))
        m = a.merge(b, on='BaseDateTime', suffixes=('_a', '_b'))
        mv = m['SOG_b'] >= 3
        if mv.any():
            pos_diff.append(metres(m['LAT_a'][mv].values, m['LON_a'][mv].values,
                                   m['LAT_b'][mv].values, m['LON_b'][mv].values))
        for name, df in (('current', a), ('timeaware', b)):
            tm = df['BaseDateTime'].values.astype('datetime64[m]').astype(np.int64)
            for run in contiguous_runs(tm):
                if len(run) < 20:
                    continue
                sub = df.iloc[run]
                err[name] += windows_errors(sub['LAT'].values, sub['LON'].values,
                                            sub['SOG'].values, sub['Heading'].values)
    pdiff = np.concatenate(pos_diff) if pos_diff else np.array([])
    lines.append(f'- |position current − time-aware| at the same minute, SOG ≥ 3 kn (m) '
                 f'p10/p25/p50/p75/p90: {q(pdiff)}')
    lines.append(f'- implied speed / SOG, CURRENT   : {ratio_summary(np.concatenate(r_cur))}')
    lines.append(f'- implied speed / SOG, TIME-AWARE: {ratio_summary(np.concatenate(r_ta))}\n')
    lines.append('Single-vessel physics on 10→10 min windows (mean obs SOG ≥ 3 kn, stride 20; inputs AND '
                 'ground truth from the same resampling; not the same window set as the model evaluation):\n')
    lines.append('| resampling | windows | cv3 ADE | cv3 FDE | dr ADE | dr FDE |')
    lines.append('|---|---|---|---|---|---|')
    for name in ('current', 'timeaware'):
        e = np.array(err[name])
        if len(e) == 0:
            lines.append(f'| {name} | 0 | — | — | — | — |')
            continue
        lines.append(f'| {name} | {len(e):,} | {e[:, 0].mean():.1f} | {e[:, 1].mean():.1f} | '
                     f'{e[:, 2].mean():.1f} | {e[:, 3].mean():.1f} |')
    lines.append('')


def part_processed(args, lines):
    lines.append(f'## Part C — processed split "{args.split}" used by the models\n')
    with open(os.path.join(args.data_dir, 'global_stats.json')) as f:
        st = json.load(f)
    ratios, jumps = [], []
    for fp in sorted(glob.glob(os.path.join(args.data_dir, args.split, '*.csv'))):
        df = pd.read_csv(fp).dropna()
        for c in ('LON', 'LAT', 'SOG'):
            df[c] = df[c] * st[c]['std'] + st[c]['mean']
        df = df.sort_values(['vessel_id', 'frame_id'])
        same = (df['vessel_id'].values[1:] == df['vessel_id'].values[:-1]) & \
               (np.diff(df['frame_id'].values) == 1)
        d = metres(df['LAT'].values[:-1], df['LON'].values[:-1], df['LAT'].values[1:], df['LON'].values[1:])
        sog = df['SOG'].values
        sel = same & (sog[:-1] >= 3) & (sog[1:] >= 3)
        ratios.append(d[sel] / (0.5 * (sog[:-1] + sog[1:])[sel] * KN * 60.0))
        still = same & (sog[:-1] < 0.5) & (sog[1:] < 0.5)
        jumps.append(d[still])
    r = np.concatenate(ratios)
    lines.append(f'- implied speed / SOG between consecutive frames, SOG ≥ 3 kn: {ratio_summary(r)}')
    lines.append(f'- share with ratio < 0.5 (position barely moved): {100 * np.mean(r < 0.5):.1f}%  '
                 f'| > 1.5 (position jumped): {100 * np.mean(r > 1.5):.1f}%')
    lines.append(f'- per-minute displacement of near-stationary vessels (SOG < 0.5 kn), m: '
                 f'{q(np.concatenate(jumps))}\n')
    lines.append('Reading: clean 1-min tracks give a ratio tightly around 1 (most within ±10%). '
                 'A wide spread with mass near 0 and 2 = timestamp-snapping jitter.\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir',  type=str, default=P.RAW_DIR)
    ap.add_argument('--days',     type=str, default='26,27')
    ap.add_argument('--data_dir', type=str, default=P.OUT_BASE)
    ap.add_argument('--split',    type=str, default='test')
    ap.add_argument('--skip_raw', action='store_true')
    ap.add_argument('--out',      type=str, default=None)
    args = ap.parse_args()

    lines = ['# Resampling diagnostic\n']
    print(f'pandas {pd.__version__}  (resample().interpolate() semantics depend on the version)')
    lines.append(f'pandas {pd.__version__}\n')
    part_processed(args, lines)
    print('\n'.join(lines))
    if not args.skip_raw:
        n0 = len(lines)
        vessels = part_raw(args, lines)
        if vessels:
            part_compare(vessels, lines)
        print('\n'.join(lines[n0:]))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
