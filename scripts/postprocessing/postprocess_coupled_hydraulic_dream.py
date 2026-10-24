#!/usr/bin/env python3
"""Stage 3 DREAM diagnostics, posterior intervals and held-out predictive checks.

Requires prior Stage 3 sampling output, but does NOT rerun DREAM.
Run from the AVANT repository root on the same machine as Stage 1/2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.coupled_dream import (
    build_likelihood, converged, load_stage1, rank_diagnostics, sha256_file,
)
from avant_model.inversion.coupled_parameters import HydraulicBounds, PARAMETER_NAMES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--posterior-draws', type=int, default=150)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--rhat-max', type=float, default=1.01)
    p.add_argument('--ess-min', type=float, default=400)
    return p


def save_traces(chains, burn, output):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.3), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        for k, chain in enumerate(chains):
            ax.plot(chain[:, j], lw=0.7, alpha=0.72, label=f'Chain {k+1}')
        ax.axvline(burn, color='black', ls='--', lw=1, label='Burn-in cutoff')
        ax.set_ylabel(PARAMETER_NAMES[j]); ax.grid(alpha=0.2)
    axes[0].legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel('DREAM iteration per chain')
    fig.suptitle('Continuous chain histories; assess mixing AFTER burn-in')
    fig.savefig(output / 'dream_trace.png', dpi=200)
    fig.savefig(output / 'dream_trace.pdf')
    plt.close(fig)


def save_posteriors(samples, center, truth, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for j, ax in enumerate(axes):
        ax.hist(samples[:, j], bins=45, density=True, alpha=0.8, color='steelblue')
        ax.axvline(center[j], color='black', ls='--', lw=1.5, label='Stage 2 MLE')
        if truth is not None:
            ax.axvline(truth[j], color='firebrick', lw=1.5, label='Synthetic truth')
        ax.set_xlabel(PARAMETER_NAMES[j]); ax.set_ylabel('Posterior density')
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=8)
    fig.suptitle('Stage 3 marginal distributions (after burn-in; conditional on convergence)')
    fig.savefig(output / 'dream_posterior_marginals.png', dpi=220)
    fig.savefig(output / 'dream_posterior_marginals.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5.3), constrained_layout=True)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.18, rasterized=True)
    ax.plot(center[0], center[1], 'kx', ms=9, label='Stage 2 MLE')
    if truth is not None:
        ax.plot(truth[0], truth[1], '*', ms=12, color='firebrick', label='Synthetic truth')
    ax.set(xlabel=PARAMETER_NAMES[0], ylabel=PARAMETER_NAMES[1], title='Joint posterior samples')
    ax.grid(alpha=0.15); ax.legend()
    fig.savefig(output / 'dream_joint_posterior.png', dpi=220)
    fig.savefig(output / 'dream_joint_posterior.pdf')
    plt.close(fig)


def posterior_predictive(likelihood, samples, n_draws, seed, injection_h, output):
    obs = likelihood.observations
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)
    latent = np.stack([likelihood.predict(samples[i]) for i in indices], axis=0)
    # Replicate noisy observations (not just latent/parameter bands).
    replicated = latent + rng.normal(0., obs.sigma_ns, size=latent.shape)
    latent_mean = np.mean(latent, axis=0)
    latent_lo, latent_hi = np.percentile(latent, (5, 95), axis=0)
    obs_lo, obs_hi = np.percentile(replicated, (5, 95), axis=0)
    holdout = obs.holdout_mask
    measured = obs.observed_ns[:, holdout]
    rmse = float(np.sqrt(np.mean((latent_mean[:, holdout] - measured)**2)))
    coverage90 = float(np.mean((measured >= obs_lo[:, holdout]) &
                               (measured <= obs_hi[:, holdout])))
    records = []
    for it, t in enumerate(obs.times_h):
        for k, channel in enumerate(obs.channels):
            records.append({
                'time_h': float(t), 'channel': channel,
                'partition': 'holdout' if holdout[k] else 'training',
                'observed_ns': float(obs.observed_ns[it, k]),
                'posterior_mean_latent_ns': float(latent_mean[it, k]),
                'latent_p05_ns': float(latent_lo[it, k]),
                'latent_p95_ns': float(latent_hi[it, k]),
                'replicated_observation_p05_ns': float(obs_lo[it, k]),
                'replicated_observation_p95_ns': float(obs_hi[it, k]),
            })
    pd.DataFrame(records).to_csv(output / 'posterior_predictive.csv', index=False)
    holdout_stations = sorted({c.rsplit('_', 1)[1]
                               for c, flag in zip(obs.channels, holdout) if flag})
    components = ('eXX', 'eYY', 'eXY', 'eZZ')
    fig, axes = plt.subplots(len(components), len(holdout_stations),
                             figsize=(5.4 * len(holdout_stations), 10.2),
                             squeeze=False, sharex=True, constrained_layout=True)
    for j, station in enumerate(holdout_stations):
        for i, comp in enumerate(components):
            ax = axes[i, j]; channel = f'{comp}_{station}'
            if channel not in obs.channels:
                ax.set_visible(False); continue
            k = obs.channels.index(channel)
            ax.fill_between(obs.times_h, obs_lo[:, k], obs_hi[:, k],
                            color='gray', alpha=0.25, label='90% noisy observation band')
            ax.fill_between(obs.times_h, latent_lo[:, k], latent_hi[:, k],
                            color='steelblue', alpha=0.35, label='90% latent strain band')
            ax.plot(obs.times_h, latent_mean[:, k], color='navy', lw=1.6,
                    label='Posterior mean prediction')
            ax.scatter(obs.times_h, obs.observed_ns[:, k], color='darkorange',
                       s=17, label='Held-out synthetic data', zorder=4)
            ax.axvline(injection_h, color='black', ls='--', lw=0.9)
            ax.set_title(channel); ax.set_ylabel('Strain (ns)'); ax.grid(alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')
    for ax in axes[-1]: ax.set_xlabel('Hours since injection start')
    fig.suptitle('Held-out S07/S08: parameter bands vs noisy predictive bands')
    fig.savefig(output / 'heldout_posterior_predictive.png', dpi=200)
    fig.savefig(output / 'heldout_posterior_predictive.pdf')
    plt.close(fig)
    return dict(posterior_predictive_draws=int(len(indices)),
                heldout_rmse_posterior_mean_ns=rmse,
                heldout_empirical_90pct_observation_coverage=coverage90,
                uncertainty_note=f'Latent bands contain parameter uncertainty only; replicated-observation bands add independent {obs.sigma_ns:g}-ns noise.')


def main(argv=None):
    args = parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    for key, hash_key in (
        ('hydraulic_config', 'hydraulic_config_sha256'),
        ('mechanical_config', 'mechanical_config_sha256')):
        if sha256_file(manifest[key]) != manifest[hash_key]:
            raise ValueError(f'Configuration changed: {manifest[key]}')
    if sha256_file(manifest['station_file']) != manifest['station_file_sha256']:
        raise ValueError('Mechanical station metadata changed since DREAM')
    source = Path(manifest['source_dir'])
    if sha256_file(source / 'synthetic_observations.npz') != manifest['stage1_observations_sha256']:
        raise ValueError('Stage 1 synthetic observations changed since DREAM')
    if sha256_file(source / 'summary.json') != manifest['stage1_summary_sha256']:
        raise ValueError('Stage 1 summary changed since DREAM')
    if sha256_file(Path(manifest['map_dir']) / 'summary.json') != manifest['stage2_summary_sha256']:
        raise ValueError('Stage 2 optimizer summary changed since DREAM')

    chains = np.load(run_dir / 'chains_log10.npy', allow_pickle=False)
    logps = np.load(run_dir / 'log_posterior.npy', allow_pickle=False)
    n_chain = int(manifest['nchains']); n_iter = int(manifest['niterations_per_chain'])
    if chains.shape != (n_chain, n_iter, 2) or logps.shape != (n_chain, n_iter):
        raise ValueError('Unexpected chain shapes')
    if not np.isfinite(chains).all() or not np.isfinite(logps).all():
        raise ValueError('Chains or posterior log probabilities contain nonfinite values')
    bounds = HydraulicBounds(*[tuple(v) for v in manifest['bounds_log10']])
    for item in chains.reshape(-1, 2):
        if not bounds.contains(item):
            raise ValueError('MCMC state outside declared uniform prior support')
    burn = int(manifest['burn_in_per_chain'])
    if args.posterior_draws < 10 or args.rhat_max <= 1 or args.ess_min < 1:
        raise ValueError('Invalid postprocessing choices')
    post = chains[:, burn:, :]
    samples = post.reshape(-1, 2)
    diag = rank_diagnostics(chains, burn)
    status = converged(diag, threshold=args.rhat_max,
                       min_ess=args.ess_min, retained_per_chain=post.shape[1])
    stage1, obs = load_stage1(source, manifest['hydraulic_config'], manifest['mechanical_config'])
    theta_map = np.asarray(manifest['start_center_log10'], float)
    truth = np.asarray(stage1.get('truth_log10_K_log10_Ss'), float) if 'truth_log10_K_log10_Ss' in stage1 else None
    # Posterior credible intervals conditional on all stated fixed-model assumptions.
    quantiles = np.quantile(samples, [0.025, 0.5, 0.975], axis=0)
    posterior = {}
    for j, name in enumerate(PARAMETER_NAMES):
        posterior[name] = dict(mean=float(np.mean(samples[:, j])), sd=float(np.std(samples[:, j], ddof=1)),
                               q025=float(quantiles[0, j]), median=float(quantiles[1, j]),
                               q975=float(quantiles[2, j]),
                               physical_median=float(10**quantiles[1, j]),
                               physical_95pct_interval=[float(10**quantiles[0, j]),
                                                        float(10**quantiles[2, j])])
    corr = float(np.corrcoef(samples.T)[0, 1])
    df = []
    for chain in range(n_chain):
        for iteration in range(burn, n_iter):
            t = chains[chain, iteration]
            df.append({'chain': chain+1, 'iteration': iteration+1,
                       PARAMETER_NAMES[0]: float(t[0]), PARAMETER_NAMES[1]: float(t[1]),
                       'K_m_s': float(10**t[0]), 'Ss_per_m': float(10**t[1]),
                       'log_posterior_including_prior': float(logps[chain, iteration])})
    pd.DataFrame(df).to_csv(run_dir / 'posterior_samples.csv', index=False)
    save_traces(chains, burn, run_dir)
    save_posteriors(samples, theta_map, truth, run_dir)
    print('Building forward model for held-out posterior predictive evaluation...', flush=True)
    likelihood, hc = build_likelihood(manifest['hydraulic_config'], manifest['mechanical_config'],
                                      stage1, obs, bounds)
    predictive = posterior_predictive(likelihood, samples, args.posterior_draws, args.seed,
                                      float(hc['injection_hours']), run_dir)
    summary = {
        'scope': 'Conditional synthetic DREAM posterior: homogeneous K/Ss, fixed geometry/mechanics, independent strain errors; NOT field validation',
        'nchains': n_chain, 'iterations_per_chain': n_iter, 'burn_in_per_chain': burn,
        'retained_samples': int(len(samples)), 'prior_bounds_log10': manifest['bounds_log10'],
        'posterior_log10': posterior, 'posterior_log10_correlation': corr,
        'stage2_mle_log10': theta_map.tolist(),
        'synthetic_truth_log10': truth.tolist() if truth is not None else None,
        'diagnostics': diag, 'rhat_threshold': args.rhat_max, 'minimum_bulk_tail_ess': args.ess_min,
        'diagnostic_pass': bool(status),
        'diagnostic_message': ('PASS: rank-split Rhat and bulk/tail ESS thresholds satisfied'
                               if status else 'NOT CONVERGED / INSUFFICIENT SAMPLING: do not treat credible intervals as reliable'),
        **predictive,
        'limitations': [
            'Synthetic same-model observations, not field or independent COMSOL validation',
            'Narrow conditional posterior assumes fixed geometry, elastic constants and hydraulic boundaries',
            'Independent Gaussian 2-ns errors may substantially understate real field uncertainty',
            'Warm-started chains may miss distant secondary modes; investigate wider starts and prior sensitivity',
            'Observed held-out stations have never entered the likelihood',
            'Monte Carlo R-hat/ESS are necessary, not sufficient, evidence of posterior exploration',
        ],
    }
    (run_dir / 'posterior_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('Saved Stage 3 diagnostics and figures to:', run_dir, flush=True)
    return summary


if __name__ == '__main__':
    main()
