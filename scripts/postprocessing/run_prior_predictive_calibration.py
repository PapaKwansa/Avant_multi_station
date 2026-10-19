"""Prior-predictive calibration for the AVANT nine-zone linear Gaussian pressure model.

Run from repository root. Uses the saved analytical Green matrix, not field data.
Gaussian priors here are unconstrained and may generate negative pressure changes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kernel', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--repeats', type=int, default=1000)
    ap.add_argument('--noise-ns', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--length-x-m', type=float, default=430.0)
    ap.add_argument('--length-y-m', type=float, default=800.0)
    args = ap.parse_args()
    if args.repeats < 10 or args.noise_ns <= 0 or args.length_x_m <= 0 or args.length_y_m <= 0:
        ap.error('Require repeats >= 10 and positive noise/domain dimensions.')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.kernel, allow_pickle=False) as f:
        G = np.asarray(f['G_ns_per_pa'], float)
        B = np.asarray(f['basis'], float)
        mask = np.asarray(f['training_channel_mask'], bool)
    if G.shape != (32, 9) or B.shape != (9, 9) or not np.allclose(B, np.eye(9)) or mask.shape != (32,) or mask.sum() != 24:
        raise ValueError('Expected verified 32-channel, nine-zone identity-basis kernel with 24 training channels.')
    H = (G @ B)[mask] * 1e6  # nanostrain / MPa
    if not np.isfinite(H).all():
        raise ValueError('Nonfinite sensitivity matrix.')
    xs = np.linspace(-args.length_x_m / 3, args.length_x_m / 3, 3)
    ys = np.linspace(-args.length_y_m / 3, args.length_y_m / 3, 3)
    xy = np.array([(x, y) for x in xs for y in ys])  # analytical x-major, y-minor
    dist2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
    candidates = [(f'independent_sd_{s:g}', np.eye(9) * s**2) for s in (0.25, 0.5, 1.0, 2.0)]
    candidates += [(f'spatial_l_{ell:g}m', np.exp(-dist2 / (2 * ell**2))) for ell in (100.0, 250.0, 500.0)]
    rng = np.random.default_rng(args.seed)
    prior_mean = np.zeros(9)
    z = 1.959963984540054
    rows = []
    zone_rows = []
    for name, C0 in candidates:
        # Generate both the truth and observations from the same prior/likelihood.
        truth = rng.multivariate_normal(prior_mean, C0, size=args.repeats)
        noise = rng.normal(0, args.noise_ns, size=(args.repeats, H.shape[0]))
        obs = truth @ H.T + noise
        S = H @ C0 @ H.T + args.noise_ns**2 * np.eye(H.shape[0])
        gain = np.linalg.solve(S, H @ C0).T
        means = prior_mean + (obs - prior_mean @ H.T) @ gain.T
        Cpost = C0 - gain @ H @ C0
        Cpost = 0.5 * (Cpost + Cpost.T)
        sd = np.sqrt(np.maximum(np.diag(Cpost), 0))
        inside = np.abs(truth - means) <= z * sd
        errors = means - truth
        for j in range(9):
            zone_rows.append(dict(prior=name, zone=j, coverage=float(inside[:, j].mean()),
                                  bias_mpa=float(errors[:, j].mean()),
                                  rmse_mpa=float(np.sqrt(np.mean(errors[:, j]**2))),
                                  posterior_sd_mpa=float(sd[j])))
        rows.append(dict(prior=name, repeats=args.repeats, mean_coverage=float(inside.mean()),
                         min_zone_coverage=float(inside.mean(axis=0).min()),
                         max_zone_coverage=float(inside.mean(axis=0).max()),
                         central_coverage=float(inside[:, 4].mean()),
                         pressure_rmse_mpa=float(np.sqrt(np.mean(errors**2))),
                         central_posterior_sd_mpa=float(sd[4])))
    summary = pd.DataFrame(rows)
    zones = pd.DataFrame(zone_rows)
    summary.to_csv(out / 'prior_predictive_summary.csv', index=False)
    zones.to_csv(out / 'prior_predictive_by_zone.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [s.replace('independent_sd_', 'Independent SD ').replace('spatial_l_', 'Spatial ').replace('m', ' m') for s in summary.prior]
    x = np.arange(len(summary))
    ax.bar(x, summary.mean_coverage * 100, label='Mean marginal coverage')
    ax.scatter(x, summary.central_coverage * 100, marker='D', color='black', label='Central-zone coverage', zorder=4)
    ax.axhline(95, color='tab:red', linestyle='--', label='Nominal 95%')
    ax.set_xticks(x, labels, rotation=30, ha='right')
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_ylim(85, 100)
    ax.set_title(f'Prior-predictive calibration ({args.repeats} realizations, {args.noise_ns:g} ns noise)')
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'prior_predictive_coverage.png', dpi=250)
    plt.close(fig)
    matrix = zones.pivot(index='prior', columns='zone', values='coverage').loc[summary.prior].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix, vmin=90, vmax=100, cmap='viridis', aspect='auto')
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(9), [f'P{i}' for i in range(9)])
    ax.set_title('Marginal 95% interval coverage by prior and pressure zone')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=8,
                    color='white' if matrix[i,j] < 94 else 'black')
    fig.colorbar(im, ax=ax, label='Empirical coverage (%)')
    fig.tight_layout()
    fig.savefig(out / 'prior_predictive_zone_coverage.png', dpi=250)
    plt.close(fig)
    metadata = dict(scope='Prior-predictive linear Gaussian calibration, not fixed hydraulic truth or field validation',
                    kernel=str(args.kernel), repeats=args.repeats, noise_ns=args.noise_ns,
                    seed=args.seed, length_x_m=args.length_x_m, length_y_m=args.length_y_m,
                    prior_mean_mpa=0.0, notes='Unconstrained Gaussian prior allows negative pressures; pressure fields drawn anew for every realization.')
    (out / 'calibration_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(summary.round(4).to_string(index=False))
    print('Saved:', out)


if __name__ == '__main__':
    main()
