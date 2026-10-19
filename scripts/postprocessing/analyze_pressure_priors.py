"""Compare nine-zone pressure priors with publication-ready figures.

Run from AVANT repository root. Uses SAVED hydraulic histories and SAVED
analytical Green matrix; no hydraulic PDE or cuboid kernel is recomputed.
Synthetic, fixed-geometry demonstration; not field validation.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import MaxNLocator

PERM = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
Z = 1.959963984540054


def grid(v):
    """Analytical x-major nine-vector to imshow [y, x]."""
    return np.asarray(v).reshape(3, 3).T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', required=True)
    p.add_argument('--kernel', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--time-h', type=float, default=19.)
    p.add_argument('--noise-ns', type=float, default=2.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=500)
    p.add_argument('--length-x-m', type=float, default=430.)
    p.add_argument('--length-y-m', type=float, default=800.)
    args = p.parse_args()
    if args.noise_ns <= 0 or args.repeats < 2 or args.length_x_m <= 0 or args.length_y_m <= 0:
        p.error('Noise, domain lengths must be positive and repeats >= 2.')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(args.history)
    ix = int(np.argmin(np.abs(history.time_h.to_numpy(float) - args.time_h)))
    row = history.iloc[ix]
    if abs(float(row.time_h) - args.time_h) > 1e-8:
        raise ValueError('Requested time not in history.')
    cols = [f'zone_row{j}_col{i}_mpa' for j in range(3) for i in range(3)]
    truth = row[cols].to_numpy(float)[PERM]
    with np.load(args.kernel) as f:
        G = f['G_ns_per_pa']
        B = f['basis']
        mask = f['training_channel_mask'].astype(bool)
    if G.shape != (32, 9) or B.shape != (9, 9) or not np.allclose(B, np.eye(9)) or mask.sum() != 24:
        raise ValueError('Unexpected kernel shape, basis or training mask.')
    ht = ((G @ B) * 1e6)[mask]
    rng = np.random.default_rng(args.seed)
    ytrue = ht @ truth  # Forward prediction: pressure (MPa) -> strain (ns).
    observations = ytrue[None, :] + rng.normal(0, args.noise_ns, (args.repeats, len(ytrue)))
    # Tile centers in analytical x-major order. Verify against source geometry if changed.
    xcenters = (np.arange(3) + .5) * args.length_x_m / 3 - args.length_x_m / 2
    ycenters = (np.arange(3) + .5) * args.length_y_m / 3 - args.length_y_m / 2
    xy = np.array([(x, y) for x in xcenters for y in ycenters])
    dist2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    candidates = [(f'independent_sd_{s:g}', f'Independent\nSD {s:g} MPa', np.eye(9) * s * s) for s in (.25, .5, 1., 2.)]
    candidates += [(f'spatial_l_{ell:g}m', f'Spatial\nlength {ell:g} m', np.exp(-dist2 / (2 * ell * ell))) for ell in (100., 250., 500.)]
    rows, maps = [], []
    for name, label, C0 in candidates:
        S = ht @ C0 @ ht.T + args.noise_ns ** 2 * np.eye(ht.shape[0])
        A = np.linalg.solve(S, ht @ C0).T
        means = observations @ A.T  # Prior mean is zero.
        C = C0 - A @ ht @ C0
        C = (C + C.T) / 2
        sd = np.sqrt(np.maximum(np.diag(C), 0))
        error = means - truth
        coverage = np.mean(np.abs(error) <= Z * sd, axis=0)
        draws = rng.multivariate_normal(means.mean(axis=0), C, size=10000, check_valid='warn')
        center_max = float(np.mean(draws[:, 4] > np.max(np.delete(draws, 4, axis=1), axis=1)))
        rows.append(dict(prior=name, rmse_mpa=float(np.sqrt(np.mean(error ** 2))), central_truth_mpa=float(truth[4]),
                         central_mean_mpa=float(means[:, 4].mean()), central_bias_mpa=float(error[:, 4].mean()),
                         central_posterior_sd_mpa=float(sd[4]), mean_empirical_coverage=float(coverage.mean()),
                         central_coverage=float(coverage[4]), central_is_max_probability_at_mean_data=center_max))
        maps.append((name, label, means.mean(axis=0), sd))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'prior_sensitivity_summary.csv', index=False)
    print(result.round(4).to_string(index=False))

    # Main pressure comparison: consistent, legible prior labels.
    fig, ax = plt.subplots(figsize=(11, 6.2))
    xx = np.arange(len(rows))
    means_c = np.array([r['central_mean_mpa'] for r in rows])
    sd_c = np.array([r['central_posterior_sd_mpa'] for r in rows])
    ax.errorbar(xx, means_c, yerr=Z * sd_c, fmt='o', capsize=4,
                label='Average posterior mean; nominal 95% posterior half-width')
    ax.axhline(truth[4], linestyle='--', color='black', label=f'Hydraulic truth ({truth[4]:.3f} MPa)')
    ax.set_xticks(xx, [m[1].replace('\n', ' ') for m in maps], rotation=25, ha='right')
    ax.set_ylabel('Central-zone pressure (MPa)')
    ax.set_title(f'Prior sensitivity at {args.time_h:g} h: central pressure')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'central_prior_comparison.png', dpi=250)
    plt.close(fig)

    # Shared scales: divergent map explicitly shows negative posterior estimates.
    all_mean = np.concatenate([truth] + [m[2] for m in maps])
    span = max(float(np.max(np.abs(all_mean))), 1e-8)
    mean_norm = TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span)
    sdmax = max(float(m[3].max()) for m in maps)
    sd_norm = Normalize(vmin=0, vmax=max(sdmax, 1e-8))
    extent = [-args.length_x_m / 2, args.length_x_m / 2, -args.length_y_m / 2, args.length_y_m / 2]

    def heat(ax, values, norm, cmap, annotate=True):
        im = ax.imshow(grid(values), origin='lower', extent=extent, aspect='equal', cmap=cmap, norm=norm)
        if annotate:
            for i, xc in enumerate(xcenters):
                for j, yc in enumerate(ycenters):
                    v = grid(values)[j, i]
                    rgba = im.cmap(im.norm(v))
                    lum = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                    ax.text(xc, yc, f'{v:.3f}', ha='center', va='center', fontsize=8,
                            color='black' if lum > .55 else 'white')
        ax.set_xticks([-200, 0, 200])
        ax.set_yticks([-400, 0, 400])
        ax.tick_params(labelsize=8)
        return im

    # Full comparison: 8 columns including hydraulic truth.
    fig, axes = plt.subplots(2, 8, figsize=(22, 8.2), constrained_layout=True)
    im_mean = heat(axes[0, 0], truth, mean_norm, 'RdBu_r')
    axes[0, 0].set_title('Hydraulic truth', fontsize=10, weight='bold')
    axes[1, 0].axis('off')
    axes[1, 0].text(.5, .5, 'Truth has no\nposterior SD', transform=axes[1, 0].transAxes,
                    ha='center', va='center', fontsize=10)
    for j, (name, label, mean, sd) in enumerate(maps, start=1):
        heat(axes[0, j], mean, mean_norm, 'RdBu_r')
        im_sd = heat(axes[1, j], sd, sd_norm, 'viridis')
        axes[0, j].set_title(label, fontsize=10)
    axes[0, 0].set_ylabel('Posterior mean / truth\nLocal y (m)')
    axes[1, 1].set_ylabel('Posterior SD\nLocal y (m)')
    for ax in axes[1, 1:]:
        ax.set_xlabel('Local x (m)', fontsize=9)
    fig.colorbar(im_mean, ax=axes[0, :].tolist(), location='right', shrink=.7, pad=.012, label='Pressure change (MPa)')
    fig.colorbar(im_sd, ax=axes[1, 1:].tolist(), location='right', shrink=.7, pad=.012, label='Posterior SD (MPa)')
    fig.suptitle(f'Nine-zone pressure recovery: prior sensitivity at {args.time_h:g} h', fontsize=15)
    fig.savefig(out / 'prior_pressure_maps.png', dpi=200)
    plt.close(fig)

    # Compact figure for a paper: truth + three illustrative priors.
    chosen = [maps[0], maps[2], maps[5], maps[6]]
    fig, axes = plt.subplots(2, 5, figsize=(14, 7.5), constrained_layout=True)
    im_mean = heat(axes[0, 0], truth, mean_norm, 'RdBu_r')
    axes[0, 0].set_title('Hydraulic truth', fontsize=10)
    axes[1, 0].axis('off')
    for j, (_, label, mean, sd) in enumerate(chosen, start=1):
        heat(axes[0, j], mean, mean_norm, 'RdBu_r')
        im_sd = heat(axes[1, j], sd, sd_norm, 'viridis')
        axes[0, j].set_title(label, fontsize=10)
    axes[0, 0].set_ylabel('Pressure (MPa)')
    axes[1, 1].set_ylabel('Posterior SD (MPa)')
    for ax in axes[1, 1:]:
        ax.set_xlabel('Local x (m)', fontsize=9)
    fig.colorbar(im_mean, ax=axes[0, :].tolist(), shrink=.7, pad=.01, label='Pressure (MPa)')
    fig.colorbar(im_sd, ax=axes[1, 1:].tolist(), shrink=.7, pad=.01, label='SD (MPa)')
    fig.suptitle(f'Prior effects on recovered pressure at {args.time_h:g} h', fontsize=14)
    fig.savefig(out / 'prior_pressure_maps_manuscript.png', dpi=250)
    plt.close(fig)
    print('Saved:', out)
    print('Figures: central_prior_comparison.png, prior_pressure_maps.png, prior_pressure_maps_manuscript.png')


if __name__ == '__main__':
    main()
