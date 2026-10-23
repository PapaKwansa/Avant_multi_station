#!/usr/bin/env python3
"""Stage 2: continuous fixed-geometry K/Ss maximum-likelihood refinement.

Consumes the SAME saved observations from Stage 1; does not resimulate truth or noise.
No DREAM, no field calibration, no pressure-gauge likelihood, no full posterior.
Run from the AVANT repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from avant_model.model.pressure_superposition import Cuboid, ElasticProperties
from avant_model.model.coupled_hydromechanics import CoupledHydroMechanicalForward
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.coupled_observations import StrainObservations
from avant_model.inversion.coupled_likelihood import CoupledStrainLikelihood


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', required=True,
                   help='Existing Stage 1 grid output, preferably hydraulic_likelihood_local_003')
    p.add_argument('--hydraulic-config', required=True)
    p.add_argument('--mechanical-config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--logK-min', type=float, default=-7.0)
    p.add_argument('--logK-max', type=float, default=-4.0)
    p.add_argument('--logSs-min', type=float, default=-7.0)
    p.add_argument('--logSs-max', type=float, default=-4.0)
    p.add_argument('--max-nfev', type=int, default=80)
    return p.parse_args()


def read_observations(source_dir, sigma_ns):
    """Load stored observations; never redraw the original synthetic noise."""
    with np.load(Path(source_dir) / 'synthetic_observations.npz', allow_pickle=False) as data:
        obs = StrainObservations(
            times_h=data['times_h'],
            channels=tuple(map(str, data['channels'])),
            observed_ns=data['observed_ns'],
            sigma_ns=float(sigma_ns),
            training_mask=data['train_mask'],
            holdout_mask=data['holdout_mask'],
        )
    return obs


def optimization_residuals(theta, likelihood):
    """Whitened training-only residuals; holdout is never used for parameter fitting."""
    predicted = likelihood.predict(theta)
    observations = likelihood.observations
    return ((predicted[:, observations.training_mask]
             - observations.observed_ns[:, observations.training_mask])
            / observations.sigma_ns).ravel()


def local_laplace_summary(jac, theta):
    """Linearized uncertainty using fixed known Gaussian sigma and flat log-prior.

    Not a full posterior interval; ignores prior truncation, model error, and time correlation.
    """
    jac = np.asarray(jac, dtype=float)
    singular = np.linalg.svd(jac, compute_uv=False)
    info = jac.T @ jac
    cov = np.linalg.pinv(info)
    sd = np.sqrt(np.diag(cov))
    return {
        'jacobian_singular_values': singular.tolist(),
        'jacobian_condition_number': float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        'covariance_log10_linearized': cov.tolist(),
        'standard_deviations_log10_linearized': sd.tolist(),
        'approx_95pct_intervals_log10': [[float(v - 1.96*s), float(v + 1.96*s)]
                                        for v, s in zip(theta, sd)],
        'meaning': 'Local Gaussian approximation ONLY; not a DREAM posterior or field uncertainty',
    }


def save_holdout_plot(obs, predictions, injection_h, output):
    components = ('eXX', 'eYY', 'eXY', 'eZZ')
    holdout_stations = sorted({channel.rsplit('_', 1)[1]
                               for channel, is_holdout in zip(obs.channels, obs.holdout_mask)
                               if is_holdout})
    fig, axes = plt.subplots(len(components), len(holdout_stations),
                             figsize=(5.4*len(holdout_stations), 10),
                             sharex=True, squeeze=False, constrained_layout=True)
    for j, station in enumerate(holdout_stations):
        for i, component in enumerate(components):
            ax = axes[i, j]
            channel = f'{component}_{station}'
            if channel not in obs.channels:
                ax.set_visible(False)
                continue
            k = obs.channels.index(channel)
            ax.plot(obs.times_h, predictions[:, k], lw=2,
                    label='Optimized prediction')
            ax.scatter(obs.times_h, obs.observed_ns[:, k], s=18,
                       label='Held-out synthetic observations', zorder=3)
            ax.axvline(injection_h, linestyle='--', color='0.4', lw=1)
            ax.set_title(channel)
            ax.set_ylabel('Strain (ns)')
            ax.grid(alpha=0.2)
            if i == 0:
                ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Hours since injection start')
    fig.suptitle('Continuous fit — held-out S07/S08 (not used in likelihood)')
    fig.savefig(output / 'heldout_predictions_optimized.png', dpi=220)
    fig.savefig(output / 'heldout_predictions_optimized.pdf')
    plt.close(fig)


def main():
    a = parse_args()
    source_dir = Path(a.source_dir)
    saved = json.loads((source_dir / 'summary.json').read_text(encoding='utf-8'))
    if 'Same-model synthetic' not in saved.get('scope', ''):
        raise ValueError('This Stage 2 runner only accepts Stage 1 synthetic grid outputs')

    hc_path, mc_path = Path(a.hydraulic_config), Path(a.mechanical_config)
    for filename, expected_hash in ((hc_path, saved['hydraulic_config_sha256']),
                                    (mc_path, saved['mechanical_config_sha256'])):
        actual_hash = hashlib.sha256(filename.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f'Configuration changed since original synthetic observations: {filename}')

    bounds = HydraulicBounds((a.logK_min, a.logK_max), (a.logSs_min, a.logSs_max))
    obs = read_observations(source_dir, saved['noise_std_ns'])
    hc = json.loads(hc_path.read_text(encoding='utf-8'))
    mc = json.loads(mc_path.read_text(encoding='utf-8'))
    stations = pd.read_csv(mc['station_file'])
    print('Building fixed mechanical operator once...', flush=True)
    model = CoupledHydroMechanicalForward(
        hc, Cuboid(**mc['geometry']), ElasticProperties(**mc['elastic']), stations,
        mechanical_splits=(int(saved['fine_n']), int(saved['fine_n']), 1),
        components=tuple(mc['components']))
    ll = CoupledStrainLikelihood(model, obs, bounds)

    x0 = np.asarray(saved['best_grid_log10_K_log10_Ss'], dtype=float)
    lo = np.array((a.logK_min, a.logSs_min), float)
    hi = np.array((a.logK_max, a.logSs_max), float)
    if not bounds.contains(x0):
        raise ValueError('Stage 1 best point outside Stage 2 optimization bounds')
    if not np.all((x0 > lo) & (x0 < hi)):
        raise ValueError('Starting point must be strictly within optimization bounds')

    initial_ll = float(ll(x0))
    print(f'Initial parameters: {x0.tolist()}; log L={initial_ll:.6f}', flush=True)
    trace = []
    def fun(theta):
        residuals = optimization_residuals(theta, ll)
        trace.append({'evaluation': len(trace)+1, 'log10_K': float(theta[0]),
                      'log10_Ss': float(theta[1]),
                      'sum_squared_whitened_residual': float(residuals @ residuals)})
        return residuals

    fit = least_squares(fun, x0, bounds=(lo, hi), method='trf', jac='3-point',
                        max_nfev=a.max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-9)
    if not fit.success:
        raise RuntimeError(f'Continuous optimization did not converge: {fit.message}')
    theta = np.asarray(fit.x, float)
    predicted = ll.predict(theta)
    final_ll = float(ll(theta))
    if final_ll < initial_ll - 1e-7:
        raise RuntimeError('Optimization decreased log likelihood; investigate fit')
    tr = obs.training_mask
    ho = obs.holdout_mask
    train_rmse = float(np.sqrt(np.mean((predicted[:, tr] - obs.observed_ns[:, tr])**2)))
    holdout_rmse = float(np.sqrt(np.mean((predicted[:, ho] - obs.observed_ns[:, ho])**2)))

    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(trace).to_csv(output / 'optimization_trace.csv', index=False)
    records = []
    for j, t in enumerate(obs.times_h):
        for k, channel in enumerate(obs.channels):
            records.append({'time_h': float(t), 'channel': channel,
                            'partition': 'training' if tr[k] else 'holdout',
                            'observed_ns': float(obs.observed_ns[j, k]),
                            'predicted_ns': float(predicted[j, k]),
                            'residual_ns': float(obs.observed_ns[j, k]-predicted[j, k])})
    pd.DataFrame(records).to_csv(output / 'prediction_residuals.csv', index=False)
    save_holdout_plot(obs, predicted, float(hc['injection_hours']), output)

    result = {
        'scope': 'Continuous maximum likelihood on SAME saved Stage 1 synthetic observations; not DREAM or field validation',
        'source_dir': str(source_dir),
        'start_log10_K_log10_Ss': x0.tolist(),
        'fit_log10_K_log10_Ss': theta.tolist(),
        'fit_K_m_s': float(10**theta[0]),
        'fit_Ss_per_m': float(10**theta[1]),
        'start_log_likelihood': initial_ll,
        'optimized_log_likelihood': final_ll,
        'truth_log_likelihood_from_stage1': saved['likelihood_at_truth'],
        'training_rmse_ns': train_rmse,
        'holdout_rmse_ns': holdout_rmse,
        'sigma_ns': float(obs.sigma_ns),
        'num_training_data': int(obs.times_h.size * tr.sum()),
        'num_heldout_data': int(obs.times_h.size * ho.sum()),
        'optimizer_success': bool(fit.success),
        'optimizer_message': str(fit.message),
        'optimizer_nfev': int(fit.nfev),
        'optimizer_njev': int(fit.njev) if fit.njev is not None else None,
        'local_laplace': local_laplace_summary(fit.jac, theta),
        'limitations': [
            'Synthetic observations generated with same one-way coupled forward model',
            'Fixed geometry, fixed material mechanics, homogeneous K/Ss and fixed-head boundaries',
            'Independent Gaussian 2-ns noise is simulated, not a field-derived covariance',
            'Local Gaussian/Laplace uncertainty is not the full Bayesian posterior',
            'Held-out station data are used only for evaluation, never for optimization',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print('Saved output files to:', output)


if __name__ == '__main__':
    main()
