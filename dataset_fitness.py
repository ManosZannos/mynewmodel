"""
dataset_fitness.py — is the dataset able to show what we want a model to learn?
CPU only, no torch. Reads a processed dataset (e.g. the v2a build), writes a markdown report.

It uses the evaluation windows (10 obs + 10 pred minutes, stride 20, dataset.py rules) and asks,
for the MOVING vessel-samples (mean obs SOG >= 3 kn), which carry ~half of every model's error:

  A. Where are they?  Approximate sub-areas inside the 30-35N / 115-120W box
     (LA/Long Beach port complex, San Diego Bay, Santa Barbara Channel, rest).
  B. How often is there a real ENCOUNTER?  Another vessel whose CPA (from the last observed
     positions and velocities) is within --dcpa_nm, 0 <= TCPA <= 10 min, current range <= --range_nm.
     Counted against moving neighbours and against any neighbour.
  C. How much NON-constant-velocity behaviour is there (the headroom for any learned model)?
     "manoeuvre" = constant-velocity (last 3 min) final-position error > --manoeuvre_frac of the
     distance actually travelled in the 10 future minutes.
  D. Is non-CV behaviour more frequent / larger in encounters or in port areas?
     If yes, interaction / geography is plausibly worth modelling; if no, it is not, on this data.

Usage (repo root; after the final v2a build):
    OMP_NUM_THREADS=4 python dataset_fitness.py --data_dir dataset/noaa_dec2021_1min_v2a --out fitness_v2a.md
"""

from __future__ import annotations
import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

R = 6371008.8
K = np.pi / 180.0
NM = 1852.0
OBS, PRED = 10, 10

# approximate boxes (lat_min, lat_max, lon_min, lon_max); labels are descriptive, not official limits
AREAS = [
    ('LA / Long Beach ports', 33.60, 33.85, -118.35, -118.05),
    ('San Diego Bay & entrance', 32.60, 32.75, -117.30, -117.08),
    ('Santa Barbara Channel', 34.00, 34.50, -120.00, -119.00),
]


def load_split(base, split, stats):
    out = []
    for fp in sorted(glob.glob(os.path.join(base, split, '*.csv'))):
        df = pd.read_csv(fp, usecols=['frame_id', 'vessel_id', 'LON', 'LAT', 'SOG', 'Heading']).dropna()
        for c in ('LON', 'LAT', 'SOG', 'Heading'):
            df[c] = df[c] * stats[c]['std'] + stats[c]['mean']
        df['file'] = os.path.basename(fp)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def windows(df):
    """Same rules as dataset.py (per file/day, stride OBS+PRED, >= 3 complete vessels)."""
    W = []
    for _, dd in df.groupby('file'):
        ts = np.sort(dd['frame_id'].unique())
        vs = np.sort(dd['vessel_id'].unique())
        ti = {t: i for i, t in enumerate(ts)}
        vi = {v: i for i, v in enumerate(vs)}
        arr = np.full((len(ts), len(vs), 3), np.nan)
        a, b = dd['frame_id'].map(ti).values, dd['vessel_id'].map(vi).values
        for j, c in enumerate(('LON', 'LAT', 'SOG')):
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


def area_of(lat, lon):
    lab = np.full(len(lat), 'rest of box (coast & open sea)', dtype=object)
    for name, a, b, c, d in AREAS:
        lab[(lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)] = name
    return lab


def analyse(W, args):
    rows = []
    for obs, gt in W:
        N = obs.shape[0]
        lat0 = np.nanmean(obs[:, -1, 1])
        x = (obs[..., 0] - obs[0, -1, 0]) * K * R * np.cos(lat0 * K)       # [N, T] metres
        y = (obs[..., 1] - obs[0, -1, 1]) * K * R
        gx = (gt[..., 0] - obs[0, -1, 0]) * K * R * np.cos(lat0 * K)
        gy = (gt[..., 1] - obs[0, -1, 1]) * K * R
        p = np.stack([x[:, -1], y[:, -1]], -1)                              # [N, 2]
        v = np.stack([x[:, -1] - x[:, -4], y[:, -1] - y[:, -4]], -1) / 3.0  # m/min
        sog = obs[:, :, 2].mean(1)
        # CPA between every pair (i = subject, j = neighbour)
        dp = p[None, :, :] - p[:, None, :]
        dv = v[None, :, :] - v[:, None, :]
        dv2 = (dv ** 2).sum(-1)
        tcpa = np.where(dv2 > 1e-9, -(dp * dv).sum(-1) / np.maximum(dv2, 1e-9), 0.0)   # minutes
        dcpa = np.linalg.norm(dp + tcpa[..., None] * dv, axis=-1)
        rng = np.linalg.norm(dp, axis=-1)
        np.fill_diagonal(rng, np.inf)
        enc = (dcpa <= args.dcpa_nm * NM) & (tcpa >= 0) & (tcpa <= PRED) & (rng <= args.range_nm * NM)
        np.fill_diagonal(enc, False)
        moving_j = sog >= args.moving_kn
        # constant velocity (last 3 min) vs ground truth
        steps = np.arange(1, PRED + 1)[None, :, None]
        cv = p[:, None, :] + steps * v[:, None, :]
        G = np.stack([gx, gy], -1)
        err = np.linalg.norm(cv - G, axis=-1)                               # [N, PRED]
        G0 = np.concatenate([p[:, None, :], G], axis=1)
        travelled = np.linalg.norm(np.diff(G0, axis=1), axis=-1).sum(1)
        for i in range(N):
            rows.append((obs[i, -1, 1], obs[i, -1, 0], sog[i], err[i].mean(), err[i, -1], travelled[i],
                         int(enc[i].any()), int((enc[i] & moving_j).any()),
                         int((rng[i] <= 1 * NM).sum()), int((rng[i] <= 3 * NM).sum()),
                         int(((rng[i] <= 3 * NM) & moving_j).sum())))
    cols = ['lat', 'lon', 'sog', 'cv_ade', 'cv_fde', 'travelled', 'enc_any', 'enc_moving',
            'n_1nm', 'n_3nm', 'n_moving_3nm']
    df = pd.DataFrame(rows, columns=cols)
    df['area'] = area_of(df.lat.values, df.lon.values)
    df['moving'] = df.sog >= args.moving_kn
    df['manoeuvre'] = df.cv_fde > args.manoeuvre_frac * np.maximum(df.travelled, 1.0)
    return df


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='dataset/noaa_dec2021_1min_v2a')
    ap.add_argument('--splits', type=str, default='train,val,test')
    ap.add_argument('--moving_kn', type=float, default=3.0)
    ap.add_argument('--dcpa_nm', type=float, default=0.5)
    ap.add_argument('--range_nm', type=float, default=3.0)
    ap.add_argument('--manoeuvre_frac', type=float, default=0.2)
    ap.add_argument('--out', type=str, default='fitness.md')
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, 'global_stats.json')) as f:
        stats = json.load(f)
    parts = []
    for sp in args.splits.split(','):
        W = windows(load_split(args.data_dir, sp, stats))
        d = analyse(W, args)
        d['split'] = sp
        parts.append(d)
        print(f'{sp}: {len(W):,} windows, {len(d):,} vessel-samples')
    df = pd.concat(parts, ignore_index=True)
    mv = df[df.moving]

    L = [f'# Dataset fitness — {args.data_dir}\n',
         f'Windows: 10+10 min, stride 20 (dataset.py rules), splits {args.splits}. Moving = mean obs SOG ≥ '
         f'{args.moving_kn:g} kn. Encounter = DCPA ≤ {args.dcpa_nm:g} nm, 0 ≤ TCPA ≤ 10 min, range ≤ '
         f'{args.range_nm:g} nm. Manoeuvre = CV final error > {100 * args.manoeuvre_frac:.0f}% of the distance '
         f'travelled.\n']

    rows = []
    for sp, g in df.groupby('split', sort=False):
        m = g[g.moving]
        rows.append([sp, f'{len(g):,}', f'{100 * g.moving.mean():.1f}%', f'{len(m):,}',
                     f'{100 * m.enc_moving.mean():.1f}%', f'{100 * m.manoeuvre.mean():.1f}%'])
    L.append('## 0. Overview\n')
    L.append(md_table(['split', 'vessel-samples', 'moving', 'moving samples', 'moving with a moving-vessel encounter',
                       'moving with a manoeuvre'], rows))
    L.append('')

    rows = []
    for a, g in mv.groupby('area'):
        rows.append([a, f'{len(g):,} ({100 * len(g) / len(mv):.0f}%)',
                     f'{100 * (df.area == a).mean():.0f}%',
                     f'{g.cv_ade.mean():.0f}', f'{100 * g.cv_ade.sum() / mv.cv_ade.sum():.0f}%',
                     f'{100 * g.manoeuvre.mean():.1f}%', f'{100 * g.enc_moving.mean():.1f}%',
                     f'{g.n_moving_3nm.median():.0f}'])
    L.append('## A. Where are the moving samples? (approximate areas)\n')
    L.append(md_table(['area', 'moving samples (share)', 'all samples in area', 'CV ADE (m)',
                       'share of moving CV error', 'manoeuvre rate', 'encounter rate (moving nb)',
                       'median moving nb ≤ 3 nm'], rows))
    L.append('')

    rows = []
    for lab, sel in (('moving neighbour encounter', mv.enc_moving == 1), ('any-neighbour encounter', mv.enc_any == 1),
                     ('no encounter (any)', mv.enc_any == 0)):
        g = mv[sel]
        if len(g) == 0:
            rows.append([lab, '0', '—', '—', '—'])
            continue
        rows.append([lab, f'{len(g):,} ({100 * sel.mean():.1f}%)', f'{g.cv_ade.mean():.0f}',
                     f'{100 * g.manoeuvre.mean():.1f}%', f'{100 * g.cv_ade.sum() / mv.cv_ade.sum():.0f}%'])
    L.append('## B/D. Encounters among moving samples\n')
    L.append(md_table(['group', 'moving samples', 'CV ADE (m)', 'manoeuvre rate', 'share of moving CV error'], rows))
    L.append('\nReading: if encounters are rare AND their manoeuvre rate is close to the no-encounter rate, the data '
             'cannot reward interaction modelling much, whatever the model.\n')

    rows = []
    for lab, sel in (('manoeuvre', mv.manoeuvre), ('no manoeuvre', ~mv.manoeuvre)):
        g = mv[sel]
        rows.append([lab, f'{len(g):,} ({100 * sel.mean():.1f}%)', f'{g.cv_ade.mean():.0f}',
                     f'{100 * g.cv_ade.sum() / mv.cv_ade.sum():.0f}%', f'{100 * g.enc_moving.mean():.1f}%',
                     f'{100 * g.area.ne("rest of box (coast & open sea)").mean():.0f}%'])
    L.append('## C. How much of the moving error is non-constant-velocity behaviour?\n')
    L.append(md_table(['group', 'moving samples', 'CV ADE (m)', 'share of moving CV error',
                       'in an encounter', 'inside a named port/channel area'], rows))
    L.append('\nReading: the manoeuvre rows are the headroom of any learned model over CV. Where they sit '
             '(encounters vs port areas) points to interaction vs geography.\n')

    rows = []
    for a, g in mv.groupby('area'):
        for lab, col in (('moving nb', 'enc_moving'), ('any nb', 'enc_any')):
            e, n = g[g[col] == 1], g[g[col] == 0]
            rows.append([a, lab, f'{len(e):,}', f'{100 * e.manoeuvre.mean():.1f}%' if len(e) else '—',
                         f'{len(n):,}', f'{100 * n.manoeuvre.mean():.1f}%' if len(n) else '—',
                         f'{e.cv_ade.mean():.0f} / {n.cv_ade.mean():.0f}' if len(e) and len(n) else '—'])
    L.append('## D2. Encounter vs no encounter WITHIN each area (removes the location confound)\n')
    L.append(md_table(['area', 'encounter with', 'n encounter', 'manoeuvre rate (enc)', 'n no-enc',
                       'manoeuvre rate (no enc)', 'CV ADE enc / no-enc (m)'], rows))
    L.append('\nReading: encounters happen mostly in ports, where manoeuvres happen anyway. Only a gap that '
             'survives inside the same area points to interaction rather than geography.\n')

    q = mv[['n_1nm', 'n_3nm', 'n_moving_3nm']].quantile([0.5, 0.9]).round(0)
    L.append('## Neighbourhood of moving samples (counts of other vessels)\n')
    L.append(md_table(['quantile', 'any ≤ 1 nm', 'any ≤ 3 nm', 'moving ≤ 3 nm'],
                      [[f'p{int(100 * i)}'] + [int(v) for v in r] for i, r in q.iterrows()]))
    text = '\n'.join(L)
    print(text)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()