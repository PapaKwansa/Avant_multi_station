#!/usr/bin/env python3
"""Stage 4A: synthetic joint FV-pressure / analytical-strain inversion.

Reuse the SAME saved Stage 1 strain observations, with checked config hashes.
Generate ONE separate, seeded synthetic gauge dataset from the full FV pressure
field. Stage 4 fits only log10 K and log10 Ss, with fixed geometry/mechanics.
Compare Stage 2 strain-only MLE against pressure-only and joint continuous MLE.

This is not DREAM, independent COMSOL validation, or field inversion.
Run from the repository root with the (vrugt) environment.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from avant_model.inversion.coupled_dream import build_likelihood, load_stage1, sha256_file
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.joint_pressure_strain import (
    JointPressureStrainLikelihood, synthetic_pressure_observations,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', required=True, help='Saved Stage 1 results with synthetic_observations.npz')
    p.add_argument('--map-dir', required=True, help='Stage 2 strain-only MLE directory')
    p.add_argument('--hydraulic-config', required=True)
    p.add_argument('--mechanical-config', required=True)
    p.add_argument('--gauges-csv', default='configs/pressure_superposition/experiments/stage4_gauges_local.csv')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--pressure-noise-pa', type=float, default=20000.0)
    p.add_argument('--seed', type=int, default=314159)
    p.add_argument('--logK-min', type=float, default=-7.0)
    p.add_argument('--logK-max', type=float, default=-4.0)
    p.add_argument('--logSs-min', type=float, default=-7.0)
    p.add_argument('--logSs-max', type=float, default=-4.0)
    p.add_argument('--max-nfev', type=int, default=80)
    return p


def read_gauges(path):
    df = pd.read_csv(path)
    required = ['gauge', 'x_local_m', 'y_local_m', 'role']
    if list(df.columns) != required:
        raise ValueError(f'Gauge CSV columns must be exactly {required}')
    if df.empty or df[['gauge', 'x_local_m', 'y_local_m', 'role']].isna().any().any():
        raise ValueError('Gauge CSV must not contain missing entries')
    if len(set(df.gauge)) != len(df):
        raise ValueError('Gauge names must be unique')
    roles = df.role.astype(str).str.lower().tolist()
    if any(r not in ('train', 'holdout') for r in roles):
        raise ValueError('Gauge roles must be train or holdout')
    if roles.count('train') < 1 or roles.count('holdout') < 1:
        raise ValueError('At least one train and one holdout pressure gauge are required')
    return df, tuple(df.gauge.astype(str)), tuple(df.loc[df.role == 'holdout', 'gauge'].astype(str))


def fit_likelihood(ll, start, bounds, max_nfev):
    lower = np.array([bounds.log10_K[0], bounds.log10_Ss[0]], float)
    upper = np.array([bounds.log10_K[1], bounds.log10_Ss[1]], float)
    out = least_squares(ll.training_residuals, np.asarray(start, float),
                        bounds=(lower, upper), jac='3-point', method='trf',
                        xtol=1e-10, ftol=1e-10, gtol=1e-9, max_nfev=max_nfev)
    if not out.success:
        raise RuntimeError(f'Joint/pressure optimization failed: {out.message}')
    return out


def comparison_record(name, theta, likelihood):
    strain, pressure = likelihood.predict(theta)
    sobs, pobs = likelihood.strain, likelihood.pressure
    st, sh = sobs.training_mask, sobs.holdout_mask
    pt, ph = pobs.training_mask, pobs.holdout_mask
    log_strain, log_pressure = likelihood.log_likelihood_blocks(theta)
    return {
        'method': str(name), 'log10_K': float(theta[0]), 'log10_Ss': float(theta[1]),
        'K_m_s': float(10**theta[0]), 'Ss_per_m': float(10**theta[1]),
        'log_likelihood_strain_training': log_strain,
        'log_likelihood_pressure_training': log_pressure,
        'log_likelihood_joint_training': log_strain + log_pressure,
        'strain_training_rmse_ns': float(np.sqrt(np.mean((strain[:,st] - sobs.observed_ns[:,st])**2))),
        'strain_heldout_rmse_ns': float(np.sqrt(np.mean((strain[:,sh] - sobs.observed_ns[:,sh])**2))),
        'pressure_training_rmse_pa': float(np.sqrt(np.mean((pressure[:,pt] - pobs.observed_pa[:,pt])**2))),
        'pressure_heldout_rmse_pa': float(np.sqrt(np.mean((pressure[:,ph] - pobs.observed_pa[:,ph])**2))),
    }


def make_pressure_plot(ll, all_predictions, truth, output, injection_h):
    obs = ll.pressure
    n = len(obs.gauge_names)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.7*n), sharex=True, squeeze=False,
                             constrained_layout=True)
    for i, name in enumerate(obs.gauge_names):
        ax = axes[i, 0]
        role = 'calibration' if obs.training_mask[i] else 'HELD OUT'
        ax.plot(obs.times_h, truth[:,i] / 1.e6, '--', lw=1.2, label='Noise-free synthetic truth')
        for method, pred in all_predictions.items():
            ax.plot(obs.times_h, pred[:,i] / 1.e6, lw=1.6, label=f'{method} prediction')
        ax.scatter(obs.times_h, obs.observed_pa[:,i] / 1.e6, s=14, c='k',
                   alpha=.7, label='Synthetic gauge observations', zorder=5)
        ax.axvline(injection_h, ls=':', c='0.35')
        ax.set(title=f'{name} ({role}) at ({obs.gauge_x_m[i]:g},{obs.gauge_y_m[i]:g}) m local',
               ylabel='Pressure change (MPa)')
        ax.grid(alpha=.18)
        if i == 0: ax.legend(fontsize=8, loc='best')
    axes[-1,0].set_xlabel('Hours from injection start')
    fig.savefig(output / 'joint_pressure_gauges.png', dpi=200)
    fig.savefig(output / 'joint_pressure_gauges.pdf')
    plt.close(fig)


def make_strain_plot(ll, predictions, injection_h, output):
    obs = ll.strain
    channels = obs.channels
    stations = sorted({ch.rsplit('_',1)[-1] for ch, held in zip(channels, obs.holdout_mask) if held})
    components = ('eXX', 'eYY', 'eXY', 'eZZ')
    fig, axes = plt.subplots(len(components), len(stations), figsize=(5.3*len(stations),10),
                             sharex=True, squeeze=False, constrained_layout=True)
    for j, station in enumerate(stations):
        for i, comp in enumerate(components):
            ch = f'{comp}_{station}'
            ax = axes[i,j]
            if ch not in channels:
                ax.set_visible(False); continue
            k = channels.index(ch)
            for method, p in predictions.items():
                ax.plot(obs.times_h, p[:,k], lw=1.8, label=method)
            ax.scatter(obs.times_h, obs.observed_ns[:,k], c='k', s=12, zorder=3,
                       label='Held-out synthetic strain')
            ax.axvline(injection_h, ls=':', c='0.35')
            ax.set_title(ch); ax.set_ylabel('Strain (ns)'); ax.grid(alpha=.2)
            if i == 0 and j == 0: ax.legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('Hours from injection start')
    fig.suptitle('Held-out S07/S08: strain-only MLE versus joint pressure-strain MLE')
    fig.savefig(output / 'joint_heldout_strain.png', dpi=200)
    fig.savefig(output / 'joint_heldout_strain.pdf')
    plt.close(fig)


def main(argv=None):
    a = args_parser().parse_args(argv)
    if not np.isfinite(a.pressure_noise_pa) or a.pressure_noise_pa <= 0 or a.max_nfev < 1:
        raise ValueError('pressure noise must be positive; max-nfev >= 1')
    hc_path, mc_path = Path(a.hydraulic_config), Path(a.mechanical_config)
    stage1, strain_obs = load_stage1(a.source_dir, hc_path, mc_path)
    bounds = HydraulicBounds((a.logK_min, a.logK_max), (a.logSs_min, a.logSs_max))
    print('Building fixed geometry/mechanics once...', flush=True)
    old_ll, hc = build_likelihood(hc_path, mc_path, stage1, strain_obs, bounds)
    model = old_ll.model
    gauge_df, names, holdout_names = read_gauges(a.gauges_csv)
    truth_theta = np.asarray(stage1['truth_log10_K_log10_Ss'], float)
    if not bounds.contains(truth_theta):
        raise ValueError('Synthetic truth falls outside prior bounds')
    print('Generating separate synthetic FV pressure gauge measurements...', flush=True)
    pressure_obs, pressure_truth = synthetic_pressure_observations(
        model, strain_obs.times_h, 10**truth_theta[0], 10**truth_theta[1],
        names, gauge_df.x_local_m.to_numpy(float), gauge_df.y_local_m.to_numpy(float),
        sigma_pa=a.pressure_noise_pa, seed=a.seed, holdout_names=holdout_names)
    joint = JointPressureStrainLikelihood(model, strain_obs, pressure_obs, bounds)
    pressure_only = JointPressureStrainLikelihood(model, strain_obs, pressure_obs, bounds,
                                                 include_strain=False)

    # Regression: the two routes must produce IDENTICAL strain from identical FV pressure.
    independent_strain = model.predict(10**truth_theta[0], 10**truth_theta[1], strain_obs.times_h).strain_ns
    shared_strain, _ = joint.predict(truth_theta)
    diff = float(np.max(np.abs(independent_strain - shared_strain)))
    if diff > 1e-8:
        raise RuntimeError(f'Stage 4 strain does not reproduce Stage 3 kernel: max difference {diff} ns')
    strain_ll_diff = float(abs(joint.log_likelihood_blocks(truth_theta)[0] - old_ll(truth_theta)))
    if strain_ll_diff > 1e-7:
        raise RuntimeError('Stage 4 strain likelihood differs from established Stage 3 likelihood')

    previous_map = json.loads((Path(a.map_dir) / 'summary.json').read_text(encoding='utf-8'))
    if Path(previous_map['source_dir']).resolve() != Path(a.source_dir).resolve():
        raise ValueError('Stage 2 MLE source directory differs from Stage 1 observations')
    start = np.asarray(previous_map['fit_log10_K_log10_Ss'], float)
    if not bounds.contains(start):
        raise ValueError('Stage 2 MLE outside requested prior bounds')
    if abs(float(old_ll(start)) - previous_map['optimized_log_likelihood']) > 1e-5:
        raise RuntimeError('Stage 2 MLE strain likelihood does not reproduce')

    print('Fitting pressure-only parameters...', flush=True)
    pressure_fit = fit_likelihood(pressure_only, start, bounds, a.max_nfev)
    print('Fitting JOINT strain + pressure parameters...', flush=True)
    joint_fit = fit_likelihood(joint, start, bounds, a.max_nfev)
    thetas = {'synthetic_truth': truth_theta, 'strain_only_stage2_MLE': start,
              'pressure_only_MLE': pressure_fit.x, 'joint_MLE': joint_fit.x}
    records = [comparison_record(name, np.asarray(theta, float), joint)
               for name, theta in thetas.items()]
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'synthetic_pressure_observations.npz',
                        times_h=pressure_obs.times_h, gauge_names=np.asarray(pressure_obs.gauge_names),
                        gauge_x_m=pressure_obs.gauge_x_m, gauge_y_m=pressure_obs.gauge_y_m,
                        observed_pa=pressure_obs.observed_pa, truth_pa=pressure_truth,
                        train_mask=pressure_obs.training_mask, holdout_mask=pressure_obs.holdout_mask,
                        sigma_pa=pressure_obs.sigma_pa)
    pd.DataFrame(records).to_csv(output / 'parameter_fit_comparison.csv', index=False)
    gauge_predictions = {name: joint.predict(theta)[1] for name,theta in thetas.items()
                         if name in ('strain_only_stage2_MLE','joint_MLE')}
    strain_predictions = {name: joint.predict(theta)[0] for name,theta in thetas.items()
                          if name in ('strain_only_stage2_MLE','joint_MLE')}
    make_pressure_plot(joint, gauge_predictions, pressure_truth, output, float(hc['injection_hours']))
    make_strain_plot(joint, strain_predictions, float(hc['injection_hours']), output)
    report = {
        'scope': 'Stage 4A same-model synthetic joint FV pressure/analytical strain MLE; NOT DREAM, COMSOL, or field validation',
        'source_dir': str(Path(a.source_dir).resolve()),
        'stage2_map_dir': str(Path(a.map_dir).resolve()),
        'strain_observations_sha256': sha256_file(Path(a.source_dir)/'synthetic_observations.npz'),
        'hydraulic_config_sha256': sha256_file(hc_path),
        'mechanical_config_sha256': sha256_file(mc_path),
        'gauge_csv_sha256': sha256_file(a.gauges_csv),
        'pressure_synthetic_seed': a.seed,
        'pressure_sigma_pa': a.pressure_noise_pa,
        'strain_sigma_ns': strain_obs.sigma_ns,
        'pressure_gauges': [{'gauge': row.gauge, 'x_local_m': float(row.x_local_m),
                             'y_local_m': float(row.y_local_m), 'role': row.role}
                            for row in gauge_df.itertuples()],
        'n_training_strain': int(strain_obs.times_h.size * strain_obs.training_mask.sum()),
        'n_heldout_strain': int(strain_obs.times_h.size * strain_obs.holdout_mask.sum()),
        'n_training_pressure': int(pressure_obs.times_h.size * pressure_obs.training_mask.sum()),
        'n_heldout_pressure': int(pressure_obs.times_h.size * pressure_obs.holdout_mask.sum()),
        'max_new_vs_existing_strain_prediction_abs_ns': diff,
        'max_new_vs_existing_strain_loglikelihood_difference': strain_ll_diff,
        'fit_results': records,
        'optimizer_success': {'pressure_only': bool(pressure_fit.success), 'joint': bool(joint_fit.success)},
        'optimizer_nfev': {'pressure_only': int(pressure_fit.nfev), 'joint': int(joint_fit.nfev)},
        'limitations': [
            'Pressure gauge positions are local FV x/y; they are not automatically transformed from global survey coordinates.',
            'Gauge pressures are bilinear interpolations of full FV cell-centered pressure_change_pa, not regional mean/gradient coefficients.',
            'No extrapolation outside FV center rectangle; no explicit wellbore/skin pressure model.',
            'Pressure and strain errors independent Gaussian with fixed assumed standard deviations.',
            'Pressure and strain observations are generated by the same homogeneous forward model used for inversion.',
            'Joint likelihood uses one FV solve per candidate and existing fixed H = G @ A for strain.',
            'No posterior credibility until separate DREAM sampling and convergence checks.',
        ],
    }
    (output / 'joint_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('Saved Stage 4A joint results to:', output, flush=True)
    return output


if __name__ == '__main__':
    main()
