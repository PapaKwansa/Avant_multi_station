"""Repeated-noise calibration for AVANT nine-zone fixed-geometry Bayesian inversion.

Run from repository root. Uses existing pressure_history.csv and analytical NPZ.
No field data; no DREAM; no hydraulic rerun. Python >=3.9.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PERM = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=int)
ZCOLS = [f'zone_row{j}_col{i}_mpa' for j in range(3) for i in range(3)]


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(args.history)
    with np.load(args.kernel) as f:
        G = np.asarray(f['G_ns_per_pa'], float)
        B = np.asarray(f['basis'], float)
        mask = np.asarray(f['training_channel_mask'], bool)
        channels = np.asarray(f['channels']).astype(str)
    H = G @ B
    if H.shape != (32, 9) or B.shape != (9, 9) or not np.allclose(B, np.eye(9)):
        raise ValueError('Expected 32x9 sensitivity matrix and 9x9 identity pressure basis.')
    if mask.shape != (32,) or mask.sum() != 24 or len(channels) != 32:
        raise ValueError('Expected 24 training and 8 withheld channels.')
    if not all(c in history for c in ZCOLS):
        raise ValueError('Missing hydraulic zone pressure columns.')
    if args.noise_ns <= 0 or args.prior_sd_mpa <= 0 or args.repeats < 2:
        raise ValueError('Noise/prior SD must be positive; repeats must be >=2.')
    time = history['time_h'].to_numpy(float)
    p_all = history[ZCOLS].to_numpy(float)[:, PERM] * 1e6
    if not np.isfinite(p_all).all():
        raise ValueError('Non-finite pressures.')
    ht = H[mask]
    hh = H[~mask]
    sd0 = args.prior_sd_mpa * 1e6
    sigma = args.noise_ns
    # Stable posterior via scaled 9x9 precision; p expressed in MPa.
    A = ht * 1e6
    P = (A.T @ A) / sigma**2 + np.eye(9) / args.prior_sd_mpa**2
    L = np.linalg.cholesky(P)
    C_mpa2 = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(9)))
    C_mpa2 = (C_mpa2 + C_mpa2.T) / 2
    post_sd = np.sqrt(np.diag(C_mpa2))
    gain = C_mpa2 @ A.T / sigma**2
    rng = np.random.default_rng(args.seed)
    all_rows = []
    detail_rows = []
    selected = [float(v) for v in args.times]
    for target in selected:
        k = int(np.argmin(np.abs(time - target)))
        if abs(time[k] - target) > 1e-7:
            raise ValueError(f'No pressure history row for {target:g} hours.')
        truth = p_all[k] / 1e6
        clean = H @ p_all[k]
        noise = rng.normal(0, sigma, size=(args.repeats, len(channels)))
        observed = clean[None, :] + noise
        estimates = observed[:, mask] @ gain.T  # prior mean zero
        predicted = estimates @ (H * 1e6).T
        train_rmse = np.sqrt(np.mean((predicted[:, mask] - observed[:, mask])**2, axis=1))
        hold_rmse = np.sqrt(np.mean((predicted[:, ~mask] - observed[:, ~mask])**2, axis=1))
        errors = estimates - truth[None, :]
        inside = np.abs(errors) <= 1.95996398454 * post_sd[None, :]
        for j in range(9):
            all_rows.append(dict(time_h=target, zone=j, true_mpa=truth[j],
                mean_estimate_mpa=float(estimates[:, j].mean()),
                bias_mpa=float(errors[:, j].mean()),
                rmse_mpa=float(np.sqrt(np.mean(errors[:, j]**2))),
                posterior_sd_mpa=float(post_sd[j]),
                empirical_95_coverage=float(inside[:, j].mean()),
                empirical_sd_of_estimates_mpa=float(estimates[:, j].std(ddof=1))))
        for r in range(args.repeats):
            detail_rows.append(dict(time_h=target, realization=r,
                pressure_rmse_mpa=float(np.sqrt(np.mean(errors[r]**2))),
                training_rmse_ns=float(train_rmse[r]),
                holdout_rmse_ns=float(hold_rmse[r]),
                zones_in_95=int(inside[r].sum())))
        if abs(target - 19.0) < 1e-7:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.boxplot([estimates[:, j] for j in range(9)], tick_labels=[f'P{j}' for j in range(9)])
            ax.scatter(np.arange(1, 10), truth, marker='x', s=65, label='Hydraulic truth')
            ax.set(xlabel='Analytical pressure zone', ylabel='Recovered pressure (MPa)',
                   title=f'Repeated-noise nine-zone pressure recovery at {target:g} h')
            ax.legend(); ax.grid(axis='y', alpha=.25); fig.tight_layout()
            fig.savefig(out / 'repeated_noise_boxplots_19h.png', dpi=250); plt.close(fig)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.errorbar(np.arange(9), estimates.mean(axis=0),
                        yerr=1.95996398454 * post_sd, fmt='o', capsize=4,
                        label='Mean of posterior means; nominal 95% half-width')
            ax.scatter(np.arange(9), truth, marker='x', s=65, label='Hydraulic truth')
            ax.set(xlabel='Analytical pressure zone', ylabel='Pressure (MPa)',
                   title='Noise calibration at 19 h (interval width per realization)')
            ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
            fig.savefig(out / 'posterior_intervals_19h.png', dpi=250); plt.close(fig)
    summary = pd.DataFrame(all_rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(out / 'repeated_noise_summary.csv', index=False)
    detail.to_csv(out / 'realization_diagnostics.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(9):
        part = summary[summary.zone == j].sort_values('time_h')
        ax.plot(part.time_h, part.empirical_95_coverage, marker='o', label=f'P{j}')
    ax.axhline(.95, color='black', linestyle='--', label='Nominal 0.95')
    ax.set(xlabel='Time (h)', ylabel='Empirical coverage', ylim=(0, 1.05),
           title='Repeated-noise marginal interval coverage')
    ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / 'coverage_by_time.png', dpi=250); plt.close(fig)
    report = dict(scope='Fixed-truth same-model synthetic repeated-noise calibration',
                  repeats=args.repeats, noise_ns=sigma, prior_mean_mpa=0,
                  prior_sd_mpa=args.prior_sd_mpa, times_h=selected,
                  mean_empirical_coverage=float(summary.empirical_95_coverage.mean()),
                  note='Fixed-truth frequentist coverage need not equal nominal Bayesian coverage; no field validation or temporal coupling.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(summary.round(4).to_string(index=False))
    print('\n', json.dumps(report, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', required=True)
    ap.add_argument('--kernel', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--repeats', type=int, default=500)
    ap.add_argument('--noise-ns', type=float, default=2.)
    ap.add_argument('--prior-sd-mpa', type=float, default=1.)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--times', nargs='+', type=float, default=[5, 19, 19.5, 20, 24, 48, 96])
    run(ap.parse_args())
