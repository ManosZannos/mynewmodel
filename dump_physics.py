"""
dump_physics.py — physics / kinematic baselines on the SAME windows as dump_errors.py,
saved in the SAME .npz format, so analyze_errors.py can compare them with the
learned models directly (scene bootstrap, speed strata, per-horizon, ...).

No training, no GPU. Runs on CPU in the trajpred env (torch is needed only to
reuse collate_fn / compute_min_dcpa, so the metadata is identical by construction).

Methods (all deterministic, all predict ABSOLUTE positions for every vessel):
    stay      : stay at the last observed position
    cv1       : constant velocity, velocity = last displacement (1 min)
    cv3       : constant velocity, velocity = mean displacement of the last 3 minutes
    ca        : constant acceleration from the last 3 positions
    ct        : constant speed + constant turn rate (CTRV-like) from positions
    dr        : dead reckoning from the reported last SOG + Heading
                (NB: Heading is the true heading, not COG; interpolated in preprocessing)
    cv_gated  : stay if mean obs SOG < --gate_kn, else cv3  (a "hand-made" moving/stationary switch)
    kf        : constant-velocity Kalman filter over the 10 observations, then open-loop prediction.
                q (process noise) and r (measurement noise) are TUNED ON THE VAL SPLIT ONLY
                (grid search on overall ADE), never on test.

All kinematics are done in a local tangent plane (metres) centred on each vessel's last
observed position; errors use l2_meters / l2_degrees imported from evaluate_cpagrn.py,
exactly as dump_errors.py does.

Extra keys saved (ignored by analyze_errors.py, useful for the error anatomy step):
    obs_path_m : path length travelled during the observation window (m)
    gt_path_m  : path length travelled during the ground-truth future (m)
    gt_net_m   : straight-line distance last obs -> last GT position (m)

Usage (from the repo root on the DGX, CPU is enough):
    python dump_physics.py --split test
    python dump_physics.py --split test --methods stay,cv3,kf
    python dump_physics.py --split val           # optional: same dumps on val

Then e.g.:
    python analyze_errors.py \
      --model headline=error_dumps/<HL_s42>__test.npz,error_dumps/<HL_s123>__test.npz,error_dumps/<HL_s456>__test.npz \
      --model cv3=error_dumps/PHYS_cv3__test.npz \
      --model kf=error_dumps/PHYS_kf__test.npz \
      --model stay=error_dumps/PHYS_stay__test.npz \
      --pairs headline:kf,headline:cv3,kf:stay --out analysis_physics.md
"""

from __future__ import annotations
import os
import json
import argparse

import numpy as np

R_EARTH = 6371008.8          # m (mean radius)
KN_TO_M_PER_MIN = 1852.0 / 60.0
ALL_METHODS = ['stay', 'cv1', 'cv3', 'ca', 'ct', 'dr', 'cv_gated', 'kf']


# ---------------------------------------------------------------------------
# geometry: local tangent plane around (lat0, lon0)
# ---------------------------------------------------------------------------

def to_local(lat, lon, lat0, lon0):
    """lat/lon [..., T] (deg) -> x (east), y (north) in metres, relative to lat0/lon0 [...]."""
    k = np.pi / 180.0
    x = (lon - lon0[..., None]) * k * R_EARTH * np.cos(lat0[..., None] * k)
    y = (lat - lat0[..., None]) * k * R_EARTH
    return x, y


def from_local(x, y, lat0, lon0):
    k = np.pi / 180.0
    lat = lat0[..., None] + y / (R_EARTH * k)
    lon = lon0[..., None] + x / (R_EARTH * k * np.cos(lat0[..., None] * k))
    return lat, lon


# ---------------------------------------------------------------------------
# kinematic predictors — inputs in metres: P [M, T_obs, 2] (x, y); output [M, T_pred, 2]
# the last observed position is the origin for every vessel (P[:, -1] == 0)
# ---------------------------------------------------------------------------

def pred_stay(P, H):
    return np.repeat(P[:, -1:, :], H, axis=1)


def pred_cv(P, H, k=1):
    v = (P[:, -1] - P[:, -1 - k]) / k                       # [M, 2] m/min
    steps = np.arange(1, H + 1, dtype=np.float64)[None, :, None]
    return P[:, -1:, :] + steps * v[:, None, :]


def pred_ca(P, H):
    v = P[:, -1] - P[:, -2]
    a = v - (P[:, -2] - P[:, -3])
    k = np.arange(1, H + 1, dtype=np.float64)[None, :, None]
    return P[:, -1:, :] + k * v[:, None, :] + 0.5 * k * (k + 1) * a[:, None, :]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pred_ct(P, H, n=3, min_speed=0.5 * KN_TO_M_PER_MIN, max_turn_deg=20.0):
    """Constant speed + constant turn rate, estimated from the last n displacements."""
    d = np.diff(P[:, -(n + 1):, :], axis=1)                  # [M, n, 2]
    speed = np.linalg.norm(d, axis=-1).mean(axis=1)          # [M]
    phi = np.arctan2(d[..., 1], d[..., 0])                   # [M, n] math angle
    omega = wrap(np.diff(phi, axis=1)).mean(axis=1)          # [M] rad/min
    omega = np.clip(omega, -np.deg2rad(max_turn_deg), np.deg2rad(max_turn_deg))
    omega[speed < min_speed] = 0.0                           # heading is noise when (almost) still
    phi0 = phi[:, -1]
    out = np.empty((P.shape[0], H, 2))
    pos = P[:, -1].copy()
    for t in range(H):
        ang = phi0 + (t + 1) * omega
        pos = pos + speed[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        out[:, t] = pos
    return out


def pred_dr(P, H, sog_kn_last, hdg_deg_last):
    """Dead reckoning: reported SOG along the reported heading (clockwise from north)."""
    s = np.clip(sog_kn_last, 0.0, None) * KN_TO_M_PER_MIN    # m/min
    h = np.deg2rad(hdg_deg_last)
    v = np.stack([s * np.sin(h), s * np.cos(h)], axis=-1)    # east, north
    k = np.arange(1, H + 1, dtype=np.float64)[None, :, None]
    return P[:, -1:, :] + k * v[:, None, :]


def kf_gains(T_obs, q, r, v0_std=700.0):
    """Gain sequence of a 1-D constant-velocity KF (dt = 1 min).

    Every vessel is observed at the same T_obs instants with no gaps, so the
    covariance recursion (and therefore the gains) is the same for all vessels and
    both coordinates: compute it once, then filter all vessels with vector ops.
    """
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])         # white-noise acceleration
    Hm = np.array([[1.0, 0.0]])
    Pc = np.diag([r ** 2, v0_std ** 2])                      # after initialising on z0
    gains = []
    for _ in range(1, T_obs):
        Pc = F @ Pc @ F.T + Q
        S = (Hm @ Pc @ Hm.T)[0, 0] + r ** 2
        K = (Pc @ Hm.T)[:, 0] / S                            # [2]
        gains.append(K)
        Pc = (np.eye(2) - np.outer(K, Hm[0])) @ Pc
    return gains


def pred_kf(P, H, q, r):
    T_obs = P.shape[1]
    gains = kf_gains(T_obs, q, r)
    pos = P[:, 0].copy()                                     # [M, 2]
    vel = np.zeros_like(pos)
    for t, K in zip(range(1, T_obs), gains):
        pos, vel = pos + vel, vel                            # predict
        innov = P[:, t] - pos
        pos, vel = pos + K[0] * innov, vel + K[1] * innov    # update
    k = np.arange(1, H + 1, dtype=np.float64)[None, :, None]
    return pos[:, None, :] + k * vel[:, None, :]


def predict(method, P, H, aux, kf_qr=None, gate_kn=0.5):
    if method == 'stay':
        return pred_stay(P, H)
    if method == 'cv1':
        return pred_cv(P, H, 1)
    if method == 'cv3':
        return pred_cv(P, H, 3)
    if method == 'ca':
        return pred_ca(P, H)
    if method == 'ct':
        return pred_ct(P, H)
    if method == 'dr':
        return pred_dr(P, H, aux['sog_last'], aux['hdg_last'])
    if method == 'cv_gated':
        out = pred_cv(P, H, 3)
        still = aux['sog_mean'] < gate_kn
        out[still] = pred_stay(P[still], H)
        return out
    if method == 'kf':
        return pred_kf(P, H, *kf_qr)
    raise ValueError(method)


# ---------------------------------------------------------------------------
# data pass (mirrors dump_errors.py so the metadata is identical)
# ---------------------------------------------------------------------------

def iterate_split(args, split, stats):
    """Yield per-batch dicts of de-normalised numpy arrays + metadata, exactly like dump_errors.py."""
    from dataset import collate_fn, denorm
    from dataset_meta import AISDatasetMeta, verify_against_parent
    from evaluate_cpagrn_stratified import compute_min_dcpa

    stride = args.obs_len + args.pred_len
    csv_dir = os.path.join(args.data_dir, split)
    if not args.skip_verify:
        verify_against_parent(csv_dir, args.obs_len, args.pred_len, stride)
    ds = AISDatasetMeta(csv_dir, args.obs_len, args.pred_len, stride=stride)

    st = {c: (stats[c]['mean'], stats[c]['std']) for c in ('LON', 'LAT', 'SOG', 'Heading')}

    for start in range(0, len(ds), args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, len(ds))))
        obs, pred_gt, mask, counts = collate_fn([ds[i] for i in idxs])
        dcpa = compute_min_dcpa(obs, mask).numpy()            # same call as dump_errors (CPU)
        obs_np, gt_np, m = obs.numpy(), pred_gt.numpy(), mask.numpy()

        # identical SOG feature to dump_errors.py (float32 denorm, mean over obs)
        sog_mean = denorm(obs_np[..., 2], *st['SOG']).mean(axis=-1)[m]

        o = obs_np[m].astype(np.float64)                      # [M, T_obs, 4]
        g = gt_np[m].astype(np.float64)                       # [M, T_pred, 2]
        meta = dict(window_idx=[], day=[], frame_start=[], vessel_id=[], n_in_window=[])
        for b, w in enumerate(idxs):
            c = int(counts[b])
            assert int(m[b].sum()) == c == len(ds.vessel_ids_list[w]), f'count mismatch, window {w}'
            meta['window_idx'].append(np.full(c, w, dtype=np.int64))
            meta['day'].append(np.full(c, ds.day_list[w], dtype=np.int64))
            meta['frame_start'].append(np.full(c, ds.frame_start_list[w], dtype=np.int64))
            meta['vessel_id'].append(ds.vessel_ids_list[w])
            meta['n_in_window'].append(np.full(c, c, dtype=np.int64))

        yield dict(
            obs_lon=denorm(o[..., 0], *st['LON']), obs_lat=denorm(o[..., 1], *st['LAT']),
            sog_last=denorm(o[:, -1, 2], *st['SOG']), hdg_last=denorm(o[:, -1, 3], *st['Heading']) % 360.0,
            true_lon=denorm(g[..., 0], *st['LON']), true_lat=denorm(g[..., 1], *st['LAT']),
            sog_mean=sog_mean, min_dcpa=dcpa[m], meta=meta, n_windows=len(ds),
        )


def load_split(args, split, stats):
    """Whole split in memory (it is small: ~83k vessels x 20 steps)."""
    parts = list(iterate_split(args, split, stats))
    cat = lambda k: np.concatenate([p[k] for p in parts], axis=0)
    D = {k: cat(k) for k in ('obs_lon', 'obs_lat', 'sog_last', 'hdg_last', 'true_lon', 'true_lat',
                             'sog_mean', 'min_dcpa')}
    D['meta'] = {k: np.concatenate(sum((p['meta'][k] for p in parts), []), axis=0)
                 for k in parts[0]['meta']}
    D['n_windows'] = parts[0]['n_windows']
    lat0, lon0 = D['obs_lat'][:, -1], D['obs_lon'][:, -1]
    ox, oy = to_local(D['obs_lat'], D['obs_lon'], lat0, lon0)
    gx, gy = to_local(D['true_lat'], D['true_lon'], lat0, lon0)
    D['P'] = np.stack([ox, oy], axis=-1)                       # [M, T_obs, 2]
    D['G'] = np.stack([gx, gy], axis=-1)                       # [M, T_pred, 2]
    D['lat0'], D['lon0'] = lat0, lon0
    return D


def errors(D, pred_xy):
    from evaluate_cpagrn import l2_meters, l2_degrees
    plat, plon = from_local(pred_xy[..., 0], pred_xy[..., 1], D['lat0'], D['lon0'])
    err_m = l2_meters(plat, plon, D['true_lat'], D['true_lon'])
    err_deg = l2_degrees(plat, plon, D['true_lat'], D['true_lon'])
    return np.asarray(err_m, dtype=np.float32), np.asarray(err_deg, dtype=np.float32)


def tune_kf(args, stats, q_grid, r_grid):
    print('\n[kf] tuning q, r on the VAL split (overall ADE, metres)')
    V = load_split(args, 'val', stats)
    H = V['G'].shape[1]
    best, table = None, []
    for q in q_grid:
        for r in r_grid:
            e, _ = errors(V, pred_kf(V['P'], H, q, r))
            ade = float(e.mean())
            mov = float(e[V['sog_mean'] >= 3.0].mean())
            table.append((q, r, ade, mov))
            if best is None or ade < best[2]:
                best = (q, r, ade, mov)
    print('   q (m^2/min^3)   r (m)    val ADE    val ADE (>=3 kn)')
    for q, r, ade, mov in table:
        flag = '  <- best' if (q, r) == best[:2] else ''
        print(f'   {q:>12g} {r:>7g} {ade:>10.2f} {mov:>14.2f}{flag}')
    on_edge = best[0] in (q_grid[0], q_grid[-1]) or best[1] in (r_grid[0], r_grid[-1])
    if on_edge:
        print('   !! best value is on the edge of the grid — extend the grid before trusting kf.')
    return best[0], best[1]


def summary_row(name, e, sog):
    ade = e.mean(axis=1)
    sel = [sog < 0.5, (sog >= 0.5) & (sog < 3.0), sog >= 3.0]
    return [name, f'{e.mean():.2f}', f'{e[:, -1].mean():.2f}'] + [f'{ade[s].mean():.1f}' for s in sel] + \
           [f'{np.median(ade):.2f}']


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split',      type=str, default='test', choices=['val', 'test'])
    p.add_argument('--methods',    type=str, default=','.join(ALL_METHODS))
    p.add_argument('--data_dir',   type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',    type=int, default=10)
    p.add_argument('--pred_len',   type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gate_kn',    type=float, default=0.5)
    p.add_argument('--kf_q',       type=str, default='0.01,0.1,1,10,100,1000')
    p.add_argument('--kf_r',       type=str, default='1,3,10,30,100')
    p.add_argument('--out_dir',    type=str, default='error_dumps')
    p.add_argument('--skip_verify', action='store_true')
    return p.parse_args()


def main():
    args = get_args()
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    for m in methods:
        assert m in ALL_METHODS, f'unknown method {m}'

    with open(os.path.join(args.data_dir, 'global_stats.json')) as f:
        stats = json.load(f)
    assert 'Heading' in stats, 'global_stats.json has no Heading entry'

    from dataset_meta import train_vessel_ids
    D = load_split(args, args.split, stats)
    H = D['G'].shape[1]
    seen_ids = np.fromiter(train_vessel_ids(args.data_dir), dtype=np.int64)
    seen = np.isin(D['meta']['vessel_id'], seen_ids)

    # sanity: the GT reconstruction must be exact (dump_errors builds GT the same way)
    assert np.allclose(D['P'][:, -1], 0.0), 'last obs is not the local origin'
    print(f'\n{args.split}: {len(D["sog_mean"]):,} vessel-samples, {D["n_windows"]} windows')

    kf_qr = None
    if 'kf' in methods:
        kf_qr = tune_kf(args, stats, [float(x) for x in args.kf_q.split(',')],
                        [float(x) for x in args.kf_r.split(',')])
        print(f'[kf] using q={kf_qr[0]:g}, r={kf_qr[1]:g}')

    aux = dict(sog_last=D['sog_last'], hdg_last=D['hdg_last'], sog_mean=D['sog_mean'])
    obs_path = np.linalg.norm(np.diff(D['P'], axis=1), axis=-1).sum(axis=1)
    G0 = np.concatenate([np.zeros_like(D['G'][:, :1]), D['G']], axis=1)
    gt_path = np.linalg.norm(np.diff(G0, axis=1), axis=-1).sum(axis=1)
    gt_net = np.linalg.norm(D['G'][:, -1], axis=-1)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for m in methods:
        pred = predict(m, D['P'], H, aux, kf_qr=kf_qr, gate_kn=args.gate_kn)
        err_m, err_deg = errors(D, pred)
        rows.append(summary_row(m, err_m, D['sog_mean']))
        out = dict(D['meta'])
        out.update(err_m=err_m, err_deg=err_deg,
                   mean_sog_kn=D['sog_mean'].astype(np.float32), min_dcpa=D['min_dcpa'],
                   seen_in_train=seen,
                   obs_path_m=obs_path.astype(np.float32), gt_path_m=gt_path.astype(np.float32),
                   gt_net_m=gt_net.astype(np.float32))
        extra = {}
        if m == 'kf':
            extra = dict(kf_q=np.array(kf_qr[0]), kf_r=np.array(kf_qr[1]))
        tag = f'PHYS_{m}'
        path = os.path.join(args.out_dir, f'{tag}__{args.split}.npz')
        np.savez_compressed(path, **out, **extra,
                            arch=np.array(f'physics_{m}'), tag=np.array(tag), split=np.array(args.split),
                            obs_len=np.array(args.obs_len), pred_len=np.array(args.pred_len))
        print(f'saved -> {path}')

    hdr = ['method', 'ADE (m)', 'FDE (m)', 'ADE <0.5 kn', 'ADE 0.5–3 kn', 'ADE ≥3 kn', 'median ADE']
    print('\n| ' + ' | '.join(hdr) + ' |\n|' + '|'.join(['---'] * len(hdr)) + '|')
    for r in rows:
        print('| ' + ' | '.join(r) + ' |')
    print('\nReference (test, 3 seeds, from analysis_report_4models.md): headline ADE 85.78, FDE 134.40; '
          'moving (>=3 kn) ADE ~337 m. Use analyze_errors.py for paired CIs.')


if __name__ == '__main__':
    main()
