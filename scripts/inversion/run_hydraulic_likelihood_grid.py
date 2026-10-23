#!/usr/bin/env python3
"""First physics-constrained synthetic AVANT K-Ss likelihood experiment.

Run from repository root. This uses the verified, fixed-geometry one-way
hydraulic-to-cuboid strain model and independent Gaussian strain errors.
NO DREAM, NO field inversion, NO model validation, NO pressure-gauge likelihood.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.model.pressure_superposition import Cuboid, ElasticProperties
from avant_model.model.coupled_hydromechanics import CoupledHydroMechanicalForward
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.coupled_observations import make_synthetic_observations
from avant_model.inversion.coupled_likelihood import CoupledStrainLikelihood


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hydraulic-config', required=True)
    ap.add_argument('--mechanical-config', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--grid-size', type=int, default=11)
    ap.add_argument('--fine-n', type=int, default=18)
    ap.add_argument('--noise-ns', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--true-K', type=float, default=1e-6)
    ap.add_argument('--true-Ss', type=float, default=1e-6)
    ap.add_argument('--logK-min', type=float, default=-7.0)
    ap.add_argument('--logK-max', type=float, default=-4.0)
    ap.add_argument('--logSs-min', type=float, default=-7.0)
    ap.add_argument('--logSs-max', type=float, default=-4.0)
    return ap.parse_args()


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_times(cfg):
    """A modest set of independent *synthetic* measurements in injection/recovery."""
    proposed = np.array([0.5, 1, 2, 4, 8, 12, 16, 19, 19.5, 20,
                         21, 24, 30, 36, 48, 72, 96], dtype=float)
    dt = float(cfg['dt_hours'])
    tend = float(cfg['total_hours'])
    times = proposed[(proposed > 0) & (proposed <= tend + 1e-9)]
    if times.size < 4 or not np.allclose(times / dt, np.rint(times / dt), rtol=0, atol=1e-9):
        raise ValueError('Default observation times must coincide with model timesteps. Adjust config or script.')
    return times


def save_likelihood_figure(xk, xs, delta, true_theta, best_theta, out):
    fig, ax = plt.subplots(figsize=(8.2, 6.5), constrained_layout=True)
    shown = np.clip(delta, 0, 50)
    im = ax.pcolormesh(xk, xs, shown, shading='nearest', cmap='viridis_r', vmin=0, vmax=50)
    ax.plot(true_theta[0], true_theta[1], marker='*', color='#e31a1c', ms=17,
            mec='white', mew=0.9, linestyle='none', label='Synthetic truth')
    ax.plot(best_theta[0], best_theta[1], marker='X', color='white', ms=10,
            mec='black', mew=0.9, linestyle='none', label='Best point on grid')
    ax.set(xlabel=r'$\log_{10}(K\;[\mathrm{m/s}])$',
           ylabel=r'$\log_{10}(S_s\;[\mathrm{m^{-1}}])$',
           title='Synthetic strain likelihood: homogeneous aquifer')
    ax.legend(loc='best', framealpha=0.94)
    bar = fig.colorbar(im, ax=ax)
    bar.set_label(r'$-2\,[\log L-\log L_{\max}]$ (clipped at 50)')
    fig.savefig(out / 'hydraulic_likelihood_surface.png', dpi=250)
    fig.savefig(out / 'hydraulic_likelihood_surface.pdf')
    plt.close(fig)


def save_holdout_figure(obs, predicted, out, injection_h):
    channels = obs.channels
    holdout_stations = sorted({c.rsplit('_', 1)[1] for c, flag in zip(channels, obs.holdout_mask) if flag})
    comps = ('eXX', 'eYY', 'eXY', 'eZZ')
    fig, axes = plt.subplots(len(comps), len(holdout_stations),
                             figsize=(5.4 * len(holdout_stations), 10.0), sharex=True,
                             squeeze=False, constrained_layout=True)
    for j, station in enumerate(holdout_stations):
        for i, comp in enumerate(comps):
            ax = axes[i, j]
            label = f'{comp}_{station}'
            if label not in channels:
                ax.set_visible(False)
                continue
            k = channels.index(label)
            ax.plot(obs.times_h, predicted[:, k], color='#2166ac', lw=2,
                    label='Best grid prediction')
            ax.scatter(obs.times_h, obs.observed_ns[:, k], s=16, color='#d95f02',
                       alpha=0.8, label='Held-out synthetic observations')
            ax.axvline(injection_h, ls='--', color='0.35', lw=1)
            ax.set_title(label)
            ax.set_ylabel('Strain (nanostrain)')
            ax.grid(alpha=0.2)
            if i == 0:
                ax.legend(loc='best', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time since injection start (h)')
    fig.suptitle('Held-out S07/S08 strain: not used in likelihood', fontsize=14)
    fig.savefig(out / 'heldout_strain_predictions.png', dpi=250)
    fig.savefig(out / 'heldout_strain_predictions.pdf')
    plt.close(fig)


def sensitivity_diagnostics(model, times, theta, train_mask, sigma_ns, out):
    """Centered differences with respect to each base-10 LOG parameter."""
    step = 0.05
    derivatives = []
    for k in range(2):
        a = np.array(theta, float)
        b = np.array(theta, float)
        a[k] += step
        b[k] -= step
        ya = model.predict(10**a[0], 10**a[1], times).strain_ns
        yb = model.predict(10**b[0], 10**b[1], times).strain_ns
        derivatives.append((ya-yb)/(2*step))
    J = np.column_stack([v[:, train_mask].ravel() / sigma_ns for v in derivatives])
    singular = np.linalg.svd(J, compute_uv=False)
    corr = float(J[:, 0] @ J[:, 1] / (np.linalg.norm(J[:, 0])*np.linalg.norm(J[:, 1])))
    rms_by_time = [np.sqrt(np.mean(v[:, train_mask]**2, axis=1)) for v in derivatives]
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    ax.plot(times, rms_by_time[0], lw=2, marker='o', ms=3, label=r'$\partial\epsilon/\partial\log_{10} K$')
    ax.plot(times, rms_by_time[1], lw=2, marker='s', ms=3, label=r'$\partial\epsilon/\partial\log_{10} S_s$')
    ax.set(xlabel='Time since injection start (h)', ylabel='Training-channel RMS sensitivity (ns / decade)',
           title='Local hydraulic-parameter sensitivity at synthetic truth')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(out / 'hydraulic_parameter_sensitivity.png', dpi=250)
    plt.close(fig)
    return {
        'whitened_J_singular_values': [float(s) for s in singular],
        'whitened_J_condition_number': float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
        'normalized_sensitivity_column_dot': corr,
        'finite_difference_step_decades': step,
        'note': 'Local sensitivity/identifiability diagnostic, not a posterior uncertainty estimate',
    }


def main():
    a = parse_args()
    if a.grid_size < 3 or not np.isfinite(a.noise_ns) or a.noise_ns <= 0:
        raise ValueError('grid-size must be >= 3 and noise-ns must be positive')
    bounds = HydraulicBounds((a.logK_min, a.logK_max), (a.logSs_min, a.logSs_max))
    if a.true_K <= 0 or a.true_Ss <= 0:
        raise ValueError('Synthetic true K/Ss must be positive')
    true_theta = np.log10([a.true_K, a.true_Ss])
    if not bounds.contains(true_theta):
        raise ValueError('Synthetic truth is outside log-parameter bounds')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    hc = load_json(a.hydraulic_config)
    mc = load_json(a.mechanical_config)
    stations = pd.read_csv(mc['station_file'])
    times = choose_times(hc)
    print('Building fixed-geometry mechanical operator...', flush=True)
    model = CoupledHydroMechanicalForward(
        hc, Cuboid(**mc['geometry']), ElasticProperties(**mc['elastic']), stations,
        mechanical_splits=(a.fine_n, a.fine_n, 1), components=tuple(mc['components']))
    print('Generating same-model synthetic observations...', flush=True)
    truth_result = model.predict(a.true_K, a.true_Ss, times)
    obs = make_synthetic_observations(truth_result, sigma_ns=a.noise_ns,
                                      seed=a.seed, holdout_stations=('S07', 'S08'))
    likelihood = CoupledStrainLikelihood(model, obs, bounds)
    xk = np.linspace(*bounds.log10_K, a.grid_size)
    xs = np.linspace(*bounds.log10_Ss, a.grid_size)
    LL = np.empty((len(xs), len(xk)))
    start = time.perf_counter()
    for j, ss in enumerate(xs):
        for i, k in enumerate(xk):
            LL[j, i] = likelihood([k, ss])
        print(f'Likelihood grid row {j+1}/{len(xs)} complete', flush=True)
    runtime = time.perf_counter() - start
    if not np.isfinite(LL).any():
        raise RuntimeError('No finite likelihood evaluations')
    ibest = np.unravel_index(np.nanargmax(LL), LL.shape)
    best_theta = [float(xk[ibest[1]]), float(xs[ibest[0]])]
    best_prediction = likelihood.predict(best_theta)
    train = obs.training_mask
    holdout = obs.holdout_mask
    train_rmse = float(np.sqrt(np.mean((obs.observed_ns[:, train] - best_prediction[:, train])**2)))
    holdout_rmse = float(np.sqrt(np.mean((obs.observed_ns[:, holdout] - best_prediction[:, holdout])**2)))
    delta = 2 * (np.nanmax(LL) - LL)
    records = [{'log10_K': float(k), 'log10_Ss': float(ss), 'K_m_s': float(10**k),
                'Ss_per_m': float(10**ss), 'log_likelihood': float(LL[j, i]),
                'delta_deviance': float(delta[j, i])}
               for j, ss in enumerate(xs) for i, k in enumerate(xk)]
    pd.DataFrame(records).to_csv(out / 'likelihood_grid.csv', index=False)
    np.savez_compressed(out / 'synthetic_observations.npz', times_h=obs.times_h,
                        channels=np.array(obs.channels), synthetic_truth_ns=truth_result.strain_ns,
                        observed_ns=obs.observed_ns, train_mask=train, holdout_mask=holdout)
    save_likelihood_figure(xk, xs, delta, true_theta, best_theta, out)
    save_holdout_figure(obs, best_prediction, out, float(hc['injection_hours']))
    sensitivity = sensitivity_diagnostics(model, times, true_theta, train, a.noise_ns, out)
    report = {
        'scope': 'Same-model synthetic fixed-geometry K/Ss likelihood grid; not DREAM or field validation',
        'truth_log10_K_log10_Ss': [float(v) for v in true_theta],
        'best_grid_log10_K_log10_Ss': best_theta,
        'best_grid_K_m_s': float(10**best_theta[0]),
        'best_grid_Ss_per_m': float(10**best_theta[1]),
        'grid_size': a.grid_size,
        'n_grid_evaluations': int(LL.size),
        'grid_runtime_s': runtime,
        'likelihood_at_truth': float(likelihood(true_theta)),
        'max_grid_likelihood': float(np.max(LL)),
        'training_rmse_ns': train_rmse,
        'holdout_rmse_ns': holdout_rmse,
        'noise_std_ns': a.noise_ns,
        'synthetic_noise_independent_across_time_and_channels': True,
        'n_observation_times': int(len(times)),
        'n_training_channels': int(train.sum()),
        'n_holdout_channels': int(holdout.sum()),
        'fine_n': a.fine_n,
        'model_H_shape': list(model.H.shape),
        'maximum_mass_balance_residual_m3_s_at_truth': float(truth_result.max_mass_balance_residual_m3_s),
        'hydraulic_config_sha256': digest(a.hydraulic_config),
        'mechanical_config_sha256': digest(a.mechanical_config),
        'seed': a.seed,
        'sensitivity': sensitivity,
        'limitations': [
            'Same forward model used to generate and invert synthetic observations',
            'Uses fixed-head boundaries, constant thickness, and one-way mechanical coupling',
            'Hydraulic and mechanical local axes are assumed aligned',
            'Only strain observations are fit; pressure gauges not implemented in this runner',
            'Likelihood grid maximum is NOT a converged Bayesian posterior',
        ],
    }
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved figures and tables to:', out)


if __name__ == '__main__':
    main()
