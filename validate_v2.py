"""
validate_v2.py — validate preprocess_dec2021_v2.py on a few raw days BEFORE the full run.
CPU only, no torch. Writes only into --work_dir (default /tmp/v2_validation).

Checks
  1. Hold-out of reports (tests the time handling of the ACTUAL resampling code):
     every 2nd raw report of moving vessels is removed, the rest is resampled with
       v1 = preprocess_dec2021_merged.resample_vessel   and   v2 = preprocess_dec2021_v2.resample_segment,
     and each 1-min track is evaluated at the exact time of every held-out report.
     'direct' = interpolating the kept reports themselves (the floor from curvature).
     Pass: v2 ≈ direct, v1 much worse.
  2. Builds the v2 dataset for these days with BOTH heading policies (drop = v2a, fill = v2b)
     by running preprocess_dec2021_v2.py, and compares with v1 for the same days:
     rows, vessels, 20-min windows (dataset.py logic), vessels per window.
  3. Implied speed / SOG ratio on v1, v2a, v2b (moving vessels).
  4. Physics on the windows of v1 / v2a / v2b (same functions as dump_physics.py):
     stay, cv1, cv3, ca, dr — ADE by speed stratum and the per-horizon error curve
     (pass: cv3 ≈ dr on moving, cv1 not worse than stay, ca not exploding, smooth curve).
  5. COG vs Heading (raw reports with both, SOG ≥ 3 kn): is COG a sane stand-in?
  6. Figure: error of the v1 / v2 tracks vs the raw reports at the same true instant, 6 vessels.

Usage (on the RTX, repo root, after git pull):
    OMP_NUM_THREADS=4 python validate_v2.py \
        --raw_dir ~/projects/trajectory_prediction_old/data/raw/2021_12 \
        --v1_dir dataset/noaa_dec2021_1min --days 26,27
"""

from __future__ import annotations
import os
import sys
import json
import glob
import argparse
import subprocess

import numpy as np
import pandas as pd

import preprocess_dec2021_merged as P1
import preprocess_dec2021_v2 as P2
import dump_physics as DP            # numpy-only predictors (torch imported lazily there)

R = 6371008.8
K = np.pi / 180.0
OBS, PRED = 10, 10


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def pct(x, ps=(50, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return ' / '.join(f'{np.percentile(x, p):.1f}' for p in ps) if len(x) else 'n/a'


def dist_m(lat1, lon1, lat2, lon2):
    return R * K * np.hypot(lat2 - lat1, (lon2 - lon1) * np.cos(0.5 * (lat1 + lat2) * K))


def interp_track_at(t_grid, lat_g, lon_g, t_query):
    """Linear interpolation of a 1-min track at arbitrary times, only inside consecutive minutes."""
    idx = np.searchsorted(t_grid, t_query, side='right') - 1
    ok = (idx >= 0) & (idx < len(t_grid) - 1)
    idx = np.clip(idx, 0, len(t_grid) - 2)
    ok &= (t_grid[idx + 1] - t_grid[idx]) == 60
    a = (t_query - t_grid[idx]) / 60.0
    lat = (1 - a) * lat_g[idx] + a * lat_g[idx + 1]
    lon = (1 - a) * lon_g[idx] + a * lon_g[idx + 1]
    return lat, lon, ok


# ---------------------------------------------------------------------------- 1. hold-out
def holdout(raw, lines):
    err = {'direct': [], 'v1': [], 'v2': []}
    n_v = 0
    for mmsi, vd in raw.groupby('MMSI'):
        vd = vd.sort_values('BaseDateTime').drop_duplicates('BaseDateTime')
        if len(vd) < 20 or vd['SOG'].median() < 3:
            continue
        t = vd['BaseDateTime'].values.astype('datetime64[s]').astype(np.int64)
        seg, _ = P2.segment_reports(t, vd.LAT.to_numpy(float), vd.LON.to_numpy(float), vd.SOG.to_numpy(float))
        for s in np.unique(seg):
            m = seg == s
            if m.sum() < 10:
                continue
            sub = vd[m]
            ts = t[m]
            keep, held = sub.iloc[::2], sub.iloc[1::2]
            tk, th = ts[::2], ts[1::2]
            hl_lat, hl_lon = held.LAT.to_numpy(float), held.LON.to_numpy(float)
            mv = held.SOG.to_numpy(float) >= 3
            # direct interpolation of the kept reports (floor)
            ok_d = (th > tk[0]) & (th < tk[-1])
            dl = np.interp(th, tk, keep.LAT.to_numpy(float))
            do = np.interp(th, tk, keep.LON.to_numpy(float))
            # v2 grid
            r2 = P2.resample_segment(tk, keep.LAT.to_numpy(float), keep.LON.to_numpy(float),
                                     keep.SOG.to_numpy(float), keep.COG.to_numpy(float),
                                     keep.Heading.to_numpy(float))
            # v1 grid (needs a non-NaN Heading column, as v1 required)
            k1 = keep[['BaseDateTime', 'LON', 'LAT', 'SOG', 'Heading']].copy()
            k1['Heading'] = k1['Heading'].fillna(k1['Heading'].mean() if k1['Heading'].notna().any() else 0.0)
            r1 = P1.resample_vessel(k1)
            if r2 is None or r2.empty or r1.empty:
                continue
            g2 = r2['t'].to_numpy()
            l2, o2, ok2 = interp_track_at(g2, r2.LAT.to_numpy(), r2.LON.to_numpy(), th)
            g1 = r1['BaseDateTime'].values.astype('datetime64[s]').astype(np.int64)
            l1, o1, ok1 = interp_track_at(g1, r1.LAT.to_numpy(), r1.LON.to_numpy(), th)
            sel = ok_d & ok1 & ok2 & mv
            if sel.sum() == 0:
                continue
            n_v += 1
            err['direct'].append(dist_m(dl, do, hl_lat, hl_lon)[sel])
            err['v2'].append(dist_m(l2, o2, hl_lat, hl_lon)[sel])
            err['v1'].append(dist_m(l1, o1, hl_lat, hl_lon)[sel])
    lines.append('## 1. Hold-out of reports (moving vessels, SOG ≥ 3 kn at the held-out report)\n')
    if not n_v:
        lines.append('no eligible vessels\n')
        return
    rows = []
    for k in ('direct', 'v2', 'v1'):
        e = np.concatenate(err[k])
        rows.append([k, f'{len(e):,}', f'{e.mean():.1f}', pct(e, (50, 90, 99))])
    lines.append(f'{n_v} vessel segments. Error (m) of the track at the exact time of each held-out report:\n')
    lines.append(md_table(['track', 'n', 'mean', 'p50 / p90 / p99'], rows))
    lines.append('\nPass: v2 ≈ direct (only curvature error remains); v1 ≫ direct (timestamp snapping).\n')


# ---------------------------------------------------------------------------- 2-4. datasets
def load_split_days(base, days, stats=None):
    """All CSVs of the given days from base/{train,val,test}, de-normalised to raw units."""
    if stats is None:
        with open(os.path.join(base, 'global_stats.json')) as f:
            stats = json.load(f)
    out = []
    for d in days:
        hits = glob.glob(os.path.join(base, '*', f'*_{d:02d}.csv'))
        for h in hits:
            df = pd.read_csv(h).dropna()
            for c in ('LON', 'LAT', 'SOG', 'Heading'):
                df[c] = df[c] * stats[c]['std'] + stats[c]['mean']
            df['day'] = d
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def windows(df):
    """dataset.py windowing (stride = OBS+PRED, >= 3 vessels complete in obs and pred), per day."""
    W = []
    for _, dd in df.groupby('day'):
        ts = np.sort(dd['frame_id'].unique())
        vs = np.sort(dd['vessel_id'].unique())
        ti = {t: i for i, t in enumerate(ts)}
        vi = {v: i for i, v in enumerate(vs)}
        arr = np.full((len(ts), len(vs), 4), np.nan)
        a, b = dd['frame_id'].map(ti).values, dd['vessel_id'].map(vi).values
        for j, c in enumerate(('LON', 'LAT', 'SOG', 'Heading')):
            arr[a, b, j] = dd[c].values
        dif = np.diff(ts)
        L = OBS + PRED
        for s in range(0, len(ts) - L + 1, L):
            if np.any(dif[s:s + L - 1] != 1):
                continue
            o, p = arr[s:s + OBS], arr[s + OBS:s + L]
            pres = ~np.isnan(o[:, :, :2]).any(axis=(0, 2))
            if pres.sum() < 3:
                continue
            hp = ~np.isnan(p[:, pres, :2]).any(axis=(0, 2))
            idx = np.where(pres)[0][hp]
            if len(idx) < 3:
                continue
            W.append((o[:, idx].transpose(1, 0, 2), p[:, idx, :2].transpose(1, 0, 2)))
    return W


def physics(W):
    if not W:
        return None
    obs = np.concatenate([w[0] for w in W])
    gt = np.concatenate([w[1] for w in W])
    lat0, lon0 = obs[:, -1, 1], obs[:, -1, 0]
    ox, oy = DP.to_local(obs[..., 1], obs[..., 0], lat0, lon0)
    gx, gy = DP.to_local(gt[..., 1], gt[..., 0], lat0, lon0)
    Pm, G = np.stack([ox, oy], -1), np.stack([gx, gy], -1)
    aux = dict(sog_last=obs[:, -1, 2], hdg_last=obs[:, -1, 3] % 360, sog_mean=obs[..., 2].mean(1))
    res = {}
    for m in ('stay', 'cv1', 'cv3', 'ca', 'dr'):
        pr = DP.predict(m, Pm, PRED, aux)
        res[m] = np.linalg.norm(pr - G, axis=-1)          # [M, T]
    return res, aux['sog_mean']


def implied_ratio(df):
    d = df.sort_values(['vessel_id', 'frame_id'])
    same = (d.vessel_id.values[1:] == d.vessel_id.values[:-1]) & (np.diff(d.frame_id.values) == 1)
    dm = dist_m(d.LAT.values[:-1], d.LON.values[:-1], d.LAT.values[1:], d.LON.values[1:])
    s = d.SOG.values
    sel = same & (s[:-1] >= 3) & (s[1:] >= 3)
    return dm[sel] / (0.5 * (s[:-1] + s[1:])[sel] * 1852 / 60)


def build_v2(args, policy):
    out = os.path.join(args.work_dir, f'v2_{policy}')
    cmd = [sys.executable, 'preprocess_dec2021_v2.py', '--raw_dir', args.raw_dir,
           '--days', args.days, '--heading_policy', policy, '--out_base', out]
    print('$', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out


def datasets_section(args, days, lines):
    sets = {}
    if args.v1_dir and os.path.exists(os.path.join(args.v1_dir, 'global_stats.json')):
        sets['v1'] = load_split_days(args.v1_dir, days)
    for pol, name in (('drop', 'v2a'), ('fill', 'v2b')):
        sets[name] = load_split_days(build_v2(args, pol), days)

    lines.append('## 2. Dataset size on the same days\n')
    rows, phys, ratios = [], {}, {}
    for name, df in sets.items():
        if df.empty:
            rows.append([name, 0, 0, 0, '—'])
            continue
        W = windows(df)
        nv = [len(w[0]) for w in W]
        rows.append([name, f'{len(df):,}', f'{df.vessel_id.nunique():,}', f'{len(W):,}',
                     f'{np.mean(nv):.0f} / {np.max(nv)}' if nv else '—'])
        phys[name] = physics(W)
        ratios[name] = implied_ratio(df)
    lines.append(md_table(['data', 'rows', 'vessels', '20-min windows', 'vessels/window mean / max'], rows))
    lines.append('\nv2a should be close to v1 (same vessel population); v2b larger (Heading-511 vessels kept). '
                 'Scene size drives the N×N cost of the graph models.\n')

    lines.append('## 3. Implied speed / SOG between consecutive minutes (SOG ≥ 3 kn)\n')
    rows = []
    for name, r in ratios.items():
        rows.append([name, f'{len(r):,}', ' / '.join(f'{np.percentile(r, p):.2f}' for p in (10, 50, 90)),
                     f'{100 * np.mean(np.abs(r - 1) <= 0.1):.1f}%'])
    lines.append(md_table(['data', 'n', 'p10 / p50 / p90', 'within ±10%'], rows))
    lines.append('\nPass: v2 centred on 1 with most mass within ±10% (v1: 0.58 / 1.10 / 1.20, 22.7%).\n')

    lines.append('## 4. Physics on the dataset windows (same predictors as dump_physics.py)\n')
    rows = []
    for name, pr in phys.items():
        if pr is None:
            continue
        res, sog = pr
        for m, e in res.items():
            ade = e.mean(1)
            rows.append([name, m, f'{ade.mean():.1f}', f'{ade[sog < 0.5].mean():.1f}',
                         f'{ade[(sog >= 0.5) & (sog < 3)].mean():.1f}', f'{ade[sog >= 3].mean():.1f}'])
    lines.append(md_table(['data', 'method', 'ADE', '<0.5 kn', '0.5–3 kn', '≥3 kn'], rows))
    lines.append('\nPass for v2: cv3 ≈ dr on ≥3 kn; cv1 not worse than stay; ca not exploding.\n')
    lines.append('Per-horizon mean error of cv3 on moving vessels (m) — v1 showed jumps at steps 4 and 7:\n')
    hdr = ['data'] + [str(s + 1) for s in range(PRED)]
    rows = []
    for name, pr in phys.items():
        if pr is None:
            continue
        res, sog = pr
        e = res['cv3'][sog >= 3].mean(0)
        rows.append([name] + [f'{v:.0f}' for v in e])
        inc = np.diff(e)
        rows.append([f'{name} step increase'] + ['—'] + [f'{v:+.0f}' for v in inc])
    lines.append(md_table(hdr, rows))
    lines.append('\nPass: smooth, roughly constant increments for v2.\n')
    return sets


# ---------------------------------------------------------------------------- 5. COG vs heading
def cog_vs_heading(raw, lines):
    m = raw.Heading.notna() & raw.COG.notna() & (raw.SOG >= 3)
    d = np.abs(((raw.loc[m, 'COG'] - raw.loc[m, 'Heading']) + 180) % 360 - 180).to_numpy()
    lines.append('## 5. |COG − Heading| on raw reports with both, SOG ≥ 3 kn (degrees)\n')
    lines.append(f'n = {len(d):,}; p50 / p90 / p99 = {pct(d, (50, 90, 99))}; share ≤ 10°: {100 * np.mean(d <= 10):.1f}%\n')
    lines.append(f'Rows with Heading missing but COG present: {100 * (raw.Heading.isna() & raw.COG.notna()).mean():.1f}% of all rows.\n')


# ---------------------------------------------------------------------------- 6. figure
def figure(raw, args):
    """Position error of each 1-min track vs the raw reports interpolated at the SAME true time.
    (The snapping error is along-track, so it is invisible on a map; this plot shows it.)"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None
    cand = []
    for mmsi, vd in raw.groupby('MMSI'):
        vd = vd.sort_values('BaseDateTime').drop_duplicates('BaseDateTime')
        if len(vd) < 60:
            continue
        cog = vd.COG.to_numpy(float)
        turn = np.nanmean(np.abs(np.diff(np.unwrap(np.deg2rad(np.nan_to_num(cog))))))
        cand.append((mmsi, vd.SOG.median(), turn))
    if not cand:
        return None
    c = pd.DataFrame(cand, columns=['mmsi', 'sog', 'turn'])
    pick = list(c[c.sog >= 8].nlargest(2, 'sog').mmsi) + list(c[c.sog >= 3].nlargest(2, 'turn').mmsi) + \
        list(c[(c.sog > 0.3) & (c.sog < 3)].head(2).mmsi)
    pick = list(dict.fromkeys(pick))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for ax, mmsi in zip(axes.ravel(), pick):
        vd = raw[raw.MMSI == mmsi].sort_values('BaseDateTime').drop_duplicates('BaseDateTime')
        t = vd['BaseDateTime'].values.astype('datetime64[s]').astype(np.int64)
        seg, _ = P2.segment_reports(t, vd.LAT.to_numpy(float), vd.LON.to_numpy(float), vd.SOG.to_numpy(float))
        big = np.bincount(seg).argmax()
        m = seg == big
        sub, ts = vd[m], t[m]
        w = ts <= ts[0] + 3600
        sub, ts = sub[w], ts[w]
        if len(ts) < 5:
            continue
        r2 = P2.resample_segment(ts, sub.LAT.to_numpy(float), sub.LON.to_numpy(float), sub.SOG.to_numpy(float),
                                 sub.COG.to_numpy(float), sub.Heading.to_numpy(float))
        k1 = sub[['BaseDateTime', 'LON', 'LAT', 'SOG', 'Heading']].copy()
        k1['Heading'] = k1['Heading'].fillna(0.0)
        r1 = P1.resample_vessel(k1)
        for r, tcol, col, lab in ((r1, 'BaseDateTime', 'r', 'v1 (ceil)'), (r2, 't', 'b', 'v2 (time-aware)')):
            if r is None or r.empty:
                continue
            tg = r[tcol].to_numpy() if tcol == 't' else r[tcol].values.astype('datetime64[s]').astype(np.int64)
            ok = (tg >= ts[0]) & (tg <= ts[-1])
            la = np.interp(tg[ok], ts, sub.LAT.to_numpy(float))
            lo = np.interp(tg[ok], ts, sub.LON.to_numpy(float))
            e = dist_m(r.LAT.to_numpy()[ok], r.LON.to_numpy()[ok], la, lo)
            ax.plot((tg[ok] - ts[0]) / 60.0, e, col + '.-', ms=4, lw=0.8, label=lab)
        ax.set_title(f'MMSI {mmsi}  median SOG {sub.SOG.median():.1f} kn', fontsize=9)
        ax.set_xlabel('minutes')
        ax.set_ylabel('error vs raw at true time (m)')
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle('1-min track minus raw reports interpolated at the same true instant (60 min per vessel)')
    fig.tight_layout()
    path = os.path.join(args.work_dir, 'validate_v2_tracks.png')
    fig.savefig(path, dpi=120)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', type=str, required=True)
    ap.add_argument('--v1_dir', type=str, default='dataset/noaa_dec2021_1min')
    ap.add_argument('--days', type=str, default='26,27')
    ap.add_argument('--work_dir', type=str, default='/tmp/v2_validation')
    args = ap.parse_args()
    args.raw_dir = os.path.expanduser(args.raw_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    days = [int(d) for d in args.days.split(',')]

    raw = []
    for d in days:
        zp = os.path.join(args.raw_dir, f'AIS_2021_12_{d:02d}.zip')
        raw.append(P2.clean_raw(P2.load_zip(zp), 'fill'))
    raw = pd.concat(raw, ignore_index=True)
    raw = raw.groupby('MMSI').filter(lambda g: g['SOG'].max() >= P2.SOG_MOVING)
    print(f'raw rows (fill policy): {len(raw):,}, vessels: {raw.MMSI.nunique():,}')

    lines = [f'# validate_v2 — days {days}\n']
    holdout(raw, lines)
    print('\n'.join(lines))
    datasets_section(args, days, lines)
    cog_vs_heading(raw, lines)
    fig = figure(raw, args)
    if fig:
        lines.append(f'## 6. Figure\n\n{fig}\n')
    text = '\n'.join(lines)
    print(text)
    out = os.path.join(args.work_dir, 'validate_v2.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
