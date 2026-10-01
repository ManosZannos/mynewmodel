"""
analyze_errors.py — publication statistics from dump_errors.py outputs (CPU only).

Each --model NAME=dump1.npz,dump2.npz,... is one model with several seeds. All
dumps must come from the SAME split (checked: metadata arrays must be identical).

What it reports (markdown, printed and optionally saved with --out):
  1. Overall error per model: mean +- std ACROSS SEEDS, median, p90 (per-vessel ADE).
  2. Paired comparisons with a SCENE-CLUSTERED bootstrap (resample windows, not
     vessels). The test split has ~430 windows with ~190 vessels each, so
     vessel-level tests (e.g. Wilcoxon on 83k samples) treat nested samples as
     independent; the cluster bootstrap uses the real unit of replication.
  3. Strata: speed (near-stationary vs moving), vessel seen in TRAIN vs unseen,
     risky vs non-risky (min DCPA), each with the model means, the share of the
     total error and the paired difference + CI for every requested pair.
  4. Per-day table (only 6 test days -> point estimates, no CI) and per-horizon
     mean error.

Convention: pair "A:B" means A compared with B. A positive difference means A
has the LOWER error (A is better). Errors are the seed-averaged per-vessel
values, in metres.

Usage:
  python analyze_errors.py \
    --model headline=error_dumps/HL_s42__test.npz,error_dumps/HL_s123__test.npz,error_dumps/HL_s456__test.npz \
    --model nocpa=error_dumps/NC_s42__test.npz,error_dumps/NC_s123__test.npz,error_dumps/NC_s456__test.npz \
    --model lstm=error_dumps/LSTM_s42__test.npz \
    --pairs headline:nocpa,headline:lstm,nocpa:lstm --out analysis_report.md
"""

from __future__ import annotations
import argparse
import itertools

import numpy as np

META_KEYS = ['window_idx', 'day', 'frame_start', 'vessel_id', 'n_in_window',
             'mean_sog_kn', 'min_dcpa', 'seen_in_train']


def load_dump(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def parse_models(specs):
    models = {}
    for s in specs:
        name, paths = s.split('=', 1)
        models[name] = [load_dump(p.strip()) for p in paths.split(',') if p.strip()]
    return models


def check_alignment(models):
    ref_name = next(iter(models))
    ref = models[ref_name][0]
    for name, dumps in models.items():
        for i, d in enumerate(dumps):
            for k in META_KEYS:
                a, b = ref[k], d[k]
                ok = (a.shape == b.shape) and (np.allclose(a, b, equal_nan=True)
                                               if a.dtype.kind == 'f' else np.array_equal(a, b))
                if not ok:
                    raise SystemExit(f'ALIGNMENT ERROR: {name}[{i}] differs from {ref_name}[0] in "{k}". '
                                     f'All dumps must come from the same split and the same dataset build.')
            if d['err_m'].shape != ref['err_m'].shape:
                raise SystemExit(f'ALIGNMENT ERROR: err_m shape differs for {name}[{i}]')
    return ref


class Cluster:
    """Scene-clustered bootstrap machinery (shared resamples for all comparisons)."""

    def __init__(self, window_idx, B, rng):
        uniq, self.wid = np.unique(window_idx, return_inverse=True)
        self.nW = len(uniq)
        self.B = B
        self.cnt = np.stack([np.bincount(rng.integers(0, self.nW, self.nW), minlength=self.nW)
                             for _ in range(B)]).astype(np.float64)   # [B, nW]

    def sums(self, values, sel):
        s = np.bincount(self.wid[sel], weights=values[sel], minlength=self.nW)
        n = np.bincount(self.wid[sel], minlength=self.nW).astype(np.float64)
        return s, n

    def paired(self, vals_a, vals_b, sel):
        """Mean(B) - Mean(A) over selected vessels; positive => A better (lower error)."""
        sa, n = self.sums(vals_a, sel)
        sb, _ = self.sums(vals_b, sel)
        if n.sum() == 0:
            return None
        point_a, point_b = sa.sum() / n.sum(), sb.sum() / n.sum()
        den = self.cnt @ n
        with np.errstate(invalid='ignore', divide='ignore'):
            ma, mb = (self.cnt @ sa) / den, (self.cnt @ sb) / den
        diff = mb - ma
        diff = diff[np.isfinite(diff)]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p_better = float((diff > 0).mean())
        return dict(mean_a=point_a, mean_b=point_b, diff=point_b - point_a,
                    pct=100 * (point_b - point_a) / point_b if point_b else np.nan,
                    lo=lo, hi=hi, p=min(1.0, 2 * min(p_better, 1 - p_better) + 1.0 / (len(diff) + 1)))


def fmt_ci(r):
    if r is None:
        return '—'
    sig = '' if (r['lo'] > 0 or r['hi'] < 0) else ' (n.s.)'
    return f"{r['diff']:+.1f} m [{r['lo']:+.1f}, {r['hi']:+.1f}] ({r['pct']:+.1f}%){sig}"


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True, help='NAME=dump1.npz,dump2.npz,...')
    ap.add_argument('--pairs', type=str, default=None, help='A:B,C:D (default: first model vs each other)')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--risk_threshold', type=float, default=0.0056)
    ap.add_argument('--speed_edges', type=str, default='0.5,3.0', help='knots; near-stationary / slow / moving')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    models = parse_models(args.model)
    names = list(models)
    meta = check_alignment(models)
    nV, T = meta['err_m'].shape
    rng = np.random.default_rng(args.seed)

    pairs = ([tuple(p.split(':')) for p in args.pairs.split(',')] if args.pairs
             else [(names[0], n) for n in names[1:]])
    for a, b in pairs:
        assert a in models and b in models, f'unknown model in pair {a}:{b}'

    # per-vessel seed-averaged errors
    E = {n: np.mean([d['err_m'] for d in ds], axis=0) for n, ds in models.items()}   # [nV, T]
    ADE = {n: E[n].mean(axis=1) for n in names}
    FDE = {n: E[n][:, -1] for n in names}
    cl = Cluster(meta['window_idx'], args.boot, rng)

    out = []
    out.append(f'# Error analysis — {nV:,} vessel-samples, {cl.nW} windows (scenes), '
               f'{len(np.unique(meta["day"]))} days, horizon {T} steps\n')
    out.append(f'Bootstrap: {args.boot} scene-clustered resamples, 95% percentile CI. '
               f'Pair "A:B": positive = A has lower error.\n')

    # 1. overall
    rows = []
    for n in names:
        ade_seed = [d['err_m'].mean() for d in models[n]]
        fde_seed = [d['err_m'][:, -1].mean() for d in models[n]]
        sd = lambda x: f'{np.std(x, ddof=1):.2f}' if len(x) > 1 else '—'
        rows.append([n, len(models[n]), f'{np.mean(ade_seed):.2f} ± {sd(ade_seed)}',
                     f'{np.mean(fde_seed):.2f} ± {sd(fde_seed)}',
                     f'{np.median(ADE[n]):.2f}', f'{np.percentile(ADE[n], 90):.2f}'])
    out.append('## 1. Overall (metres; ± = std across seeds)\n')
    out.append(md_table(['model', 'seeds', 'ADE mean', 'FDE mean', 'ADE median (per vessel)', 'ADE p90'], rows))
    out.append('\nMean much larger than median = heavy-tailed errors dominated by a minority of vessels.\n')

    # 2. paired overall
    allsel = np.ones(nV, dtype=bool)
    rows = []
    for a, b in pairs:
        rows.append([f'{a}:{b}', 'ADE', fmt_ci(cl.paired(ADE[a], ADE[b], allsel))])
        rows.append([f'{a}:{b}', 'FDE', fmt_ci(cl.paired(FDE[a], FDE[b], allsel))])
    out.append('## 2. Paired comparison, scene-clustered bootstrap\n')
    out.append(md_table(['pair', 'metric', 'difference [95% CI] (relative)'], rows))
    out.append('')

    # 3. strata
    e1, e2 = [float(x) for x in args.speed_edges.split(',')]
    strata = {
        'speed (mean SOG over the observation window)': [
            (f'< {e1:g} kn (near-stationary)', meta['mean_sog_kn'] < e1),
            (f'{e1:g}–{e2:g} kn', (meta['mean_sog_kn'] >= e1) & (meta['mean_sog_kn'] < e2)),
            (f'≥ {e2:g} kn (moving)', meta['mean_sog_kn'] >= e2)],
        'vessel seen in the training days?': [
            ('seen in train', meta['seen_in_train']), ('NOT seen in train', ~meta['seen_in_train'])],
        f'encounter risk (min DCPA ≤ {args.risk_threshold:g})': [
            ('risky', meta['min_dcpa'] <= args.risk_threshold), ('non-risky', meta['min_dcpa'] > args.risk_threshold)],
    }
    tot_err = {n: ADE[n].sum() for n in names}
    for title, groups in strata.items():
        header = ['stratum', 'n (%)'] + [f'{n}: ADE (% of total error)' for n in names] + [f'{a}:{b}' for a, b in pairs]
        rows = []
        for label, sel in groups:
            if sel.sum() == 0:
                rows.append([label, '0'] + ['—'] * (len(names) + len(pairs)))
                continue
            cells = [label, f'{sel.sum():,} ({100 * sel.mean():.1f}%)']
            cells += [f'{ADE[n][sel].mean():.1f} ({100 * ADE[n][sel].sum() / tot_err[n]:.0f}%)' for n in names]
            cells += [fmt_ci(cl.paired(ADE[a], ADE[b], sel)) for a, b in pairs]
            rows.append(cells)
        out.append(f'## 3. Strata — {title}\n')
        out.append(md_table(header, rows))
        out.append('')

    # 4. per day and per horizon
    days = np.unique(meta['day'])
    header = ['day', 'n'] + [f'{n} ADE' for n in names] + [f'{a}:{b} (m, %)' for a, b in pairs]
    rows = []
    for d in days:
        sel = meta['day'] == d
        cells = [int(d), f'{sel.sum():,}'] + [f'{ADE[n][sel].mean():.1f}' for n in names]
        for a, b in pairs:
            ma, mb = ADE[a][sel].mean(), ADE[b][sel].mean()
            cells.append(f'{mb - ma:+.1f} ({100 * (mb - ma) / mb:+.1f}%)')
        rows.append(cells)
    out.append('## 4a. Per test day (point estimates; too few days for a CI)\n')
    out.append(md_table(header, rows))
    out.append('')
    rows = [[t + 1] + [f'{E[n][:, t].mean():.1f}' for n in names] for t in range(T)]
    out.append('## 4b. Mean error per horizon step (m)\n')
    out.append(md_table(['step'] + names, rows))

    text = '\n'.join(out)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
