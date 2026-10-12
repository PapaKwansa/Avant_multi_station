#!/usr/bin/env python3
"""Exploratory conditional pressure inversion of Larry's COMSOL strain observations.

Adds no mechanics or hydraulic physics. The COMSOL strain observations are not
synthetic. Fixed geometry/mechanics and a user-specified Gaussian prior/error
model yield *conditional equivalent pressure changes*, not permeability.

The already-converted station snapshot is required: do not convert z twice.
Each selected time is inverted independently relative to the first sample;
shared reference errors imply cross-time dependence NOT modeled here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from avant_model.model.pressure_superposition import (
    Cuboid, ElasticProperties, build_pressure_kernel, coarse_block_basis,
    checked_splits,
)
from avant_model.inversion.linear_pressure_inversion import linear_gaussian_posterior


def require_file(path: Path) -> Path:
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Required file missing: {path}')
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_times(n: int, indices, all_times: bool) -> np.ndarray:
    if all_times:
        if indices is not None:
            raise ValueError('Specify either all_times=true or time_indices, not both.')
        return np.arange(1, n, dtype=int)
    if indices is None or not isinstance(indices, list) or not indices:
        raise ValueError('Provide nonempty time_indices or all_times=true.')
    if any(isinstance(i, bool) or not isinstance(i, int) or i <= 0 or i >= n for i in indices):
        raise ValueError(f'time_indices must be integer indices between 1 and {n-1}.')
    if len(set(indices)) != len(indices):
        raise ValueError('time_indices must be unique.')
    return np.asarray(sorted(indices), dtype=int)


def validate_settings(config: dict, benchmark: dict):
    fine = checked_splits(config['fine_splits'])
    models = config['pressure_models']
    if not isinstance(models, dict) or not models:
        raise ValueError('pressure_models must be a nonempty mapping.')
    for name, splits in models.items():
        if not name or any(f % c for f, c in zip(fine, checked_splits(splits))):
            raise ValueError(f'Model {name}: coarse splits must divide fine splits.')
    for field in ('prior_sd_pa', 'assumed_error_std_ns'):
        val = float(config[field])
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f'{field} must be finite and positive.')
    if not np.isfinite(float(config['prior_mean_pa'])):
        raise ValueError('prior_mean_pa must be finite.')
    if config.get('reference_mode') != 'first_sample':
        raise ValueError('Only first_sample reference_mode is supported.')
    if tuple(config['fine_splits']) != fine:
        raise ValueError('fine_splits must be an integer triple.')
    if not benchmark.get('station_file'):
        raise ValueError('Benchmark must identify the already-converted station file.')
    return fine


def channel_masks(channels, holdout_stations):
    holdout = tuple(holdout_stations)
    if not holdout or len(holdout) != len(set(holdout)):
        raise ValueError('Specify unique nonempty holdout_stations.')
    all_stations = {channel.split('_', 1)[1] for channel in channels}
    if not set(holdout) < all_stations:
        raise ValueError('Holdouts must be a proper subset of known stations.')
    mask = np.array([channel.split('_', 1)[1] not in holdout for channel in channels])
    return mask, ~mask


def load_inputs(settings: dict, benchmark: dict):
    station_path = require_file(Path(benchmark['station_file']))
    data_path = require_file(Path(settings['strain_file']))
    stations = pd.read_csv(station_path)
    expected_station_cols = ('station', 'x', 'y', 'z')
    if any(c not in stations.columns for c in expected_station_cols):
        raise ValueError('Converted station snapshot is missing columns.')
    if stations.empty or stations['station'].duplicated().any():
        raise ValueError('Station names must be unique and nonempty.')
    names = stations['station'].astype(str).tolist()
    if any(not n.strip() for n in names):
        raise ValueError('Station names must be nonempty.')
    xyz = stations[['x', 'y', 'z']].to_numpy(float)
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] < 0):
        raise ValueError('Station snapshot must contain finite positive-downward depths.')
    prov = benchmark.get('preparation_provenance', {})
    if prov.get('station_z_in_this_configuration') != 'positive-downward depth, already converted; do not convert again':
        raise ValueError('Benchmark lacks the established converted-depth provenance.')
    frame = pd.read_csv(data_path)
    components = tuple(benchmark['components'])
    channels = tuple(f'{comp}_{name}' for comp in components for name in names)
    if 'time_s' not in frame or any(c not in frame for c in channels):
        raise ValueError('Observed strain is missing time_s or required named channels.')
    if set(frame.columns) != set(('time_s',) + channels):
        raise ValueError('Unexpected columns in observed strain file.')
    time = frame['time_s'].to_numpy(float)
    observed = frame.loc[:, channels].to_numpy(float)
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Observed time_s must be finite, strictly increasing and have >=2 samples.')
    if not np.isfinite(observed).all():
        raise ValueError('Observed strain must be finite.')
    return station_path, data_path, names, xyz, components, channels, time, observed


def run(settings_path: Path, output_dir: Path):
    settings_path = require_file(settings_path)
    settings = json.loads(settings_path.read_text(encoding='utf-8-sig'))
    benchmark_path = require_file(Path(settings['benchmark_config']))
    benchmark = json.loads(benchmark_path.read_text(encoding='utf-8-sig'))
    fine = validate_settings(settings, benchmark)
    station_path, data_path, names, xyz, components, channels, time, observed = load_inputs(settings, benchmark)
    selected = choose_times(len(time), settings.get('time_indices'), bool(settings.get('all_times', False)))
    train, holdout = channel_masks(channels, settings['holdout_stations'])
    parent = Cuboid(**benchmark['geometry'])
    elastic = ElasticProperties(**benchmark['elastic'])
    kernel = build_pressure_kernel(parent, elastic, x_prime=xyz[:, 0], y_prime=xyz[:, 1],
                                   z=xyz[:, 2], station_names=names, splits=fine, components=components)
    if tuple(kernel.channels) != channels:
        raise RuntimeError('Named channel order mismatch between observations and kernel.')
    noise = float(settings['assumed_error_std_ns'])
    prior_sd = float(settings['prior_sd_pa'])
    prior_mean = float(settings['prior_mean_pa'])
    # Difference from first sample, NOT absolute pressure. Same reference is
    # shared by all times. Per-time noise assumes independent sample errors.
    # If both samples have std sigma, differencing gives sqrt(2)*sigma.
    effective_sd = np.sqrt(2) * noise
    out = (output_dir if output_dir.is_absolute() else ROOT / output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Output exists, refusing overwrite: {out}')
    results = {}
    summary = {}
    for model_name, coarse in settings['pressure_models'].items():
        B = coarse_block_basis(fine, coarse)
        H = kernel.G @ B
        npar = H.shape[1]
        prior_cov = np.eye(npar) * prior_sd**2
        posterior_rows, prediction_rows = [], []
        train_sq, hold_sq = [], []
        all_covariances = []
        for idx in selected:
            data = observed[idx] - observed[0]
            posterior = linear_gaussian_posterior(
                H[train], data[train], prior_mean_pa=np.full(npar, prior_mean),
                prior_covariance_pa2=prior_cov, noise_std_ns=effective_sd)
            mean = posterior.mean_pa
            sd = posterior.std_pa
            predicted = H @ mean
            residual = data - predicted
            train_sq.extend((residual[train] ** 2).tolist())
            hold_sq.extend((residual[holdout] ** 2).tolist())
            all_covariances.append(posterior.covariance_pa2)
            for j in range(npar):
                posterior_rows.append(dict(model=model_name, time_index=int(idx), time_s=float(time[idx]),
                                           zone=j, mean_pa=float(mean[j]), sd_pa=float(sd[j]),
                                           lower95_pa=float(mean[j] - 1.95996398454*sd[j]),
                                           upper95_pa=float(mean[j] + 1.95996398454*sd[j])))
            for m, channel in enumerate(channels):
                prediction_rows.append(dict(model=model_name, time_index=int(idx), time_s=float(time[idx]),
                                            channel=channel, set='training' if train[m] else 'holdout',
                                            observed_delta_ns=float(data[m]), predicted_delta_ns=float(predicted[m]),
                                            residual_ns=float(residual[m])))
        results[model_name] = (pd.DataFrame(posterior_rows), pd.DataFrame(prediction_rows),
                               np.asarray(all_covariances), H)
        summary[model_name] = dict(n_unknown_pressure_zones=npar, coarse_splits=list(coarse),
                                   training_rmse_ns=float(np.sqrt(np.mean(train_sq))),
                                   holdout_rmse_ns=float(np.sqrt(np.mean(hold_sq))),
                                   n_training_channels=int(train.sum()), n_holdout_channels=int(holdout.sum()))
    provenance = dict(scope='Exploratory COMSOL strain inversion; fixed benchmark mechanics; '
                             'assumed Gaussian prior/noise; NOT field validation or permeability inference.',
                      git_commit=None, input_hashes={str(p.relative_to(ROOT)): sha256(p) for p in
                              (settings_path, benchmark_path, station_path, data_path)},
                      time_reference_s=float(time[0]), time_indices=selected.tolist(),
                      assumed_per_sample_noise_std_ns=noise,
                      assumed_differenced_noise_std_ns=float(effective_sd),
                      caveats=['Time estimates are conditional and independent; shared first-sample error '
                               'induces cross-time correlation that is not modeled.',
                               'Fixed geometry and elastic parameters; no model discrepancy estimation.',
                               'Holdout error is descriptive; not independent field validation.',
                               'No COMSOL spatial pressure truth is used or recovered.'])
    import subprocess
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        provenance['git_commit'] = commit.stdout.strip()
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model_name, (posterior, predictions, covariances, H) in results.items():
        posterior.to_csv(out / f'{model_name}_pressure_posterior.csv', index=False)
        predictions.to_csv(out / f'{model_name}_channel_predictions.csv', index=False)
        np.savez_compressed(out / f'{model_name}_matrix_and_covariance.npz',
                            design_ns_per_pa=H, posterior_covariance_pa2=covariances,
                            channel_names=np.asarray(channels), time_indices=selected)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for zone, group in posterior.groupby('zone', sort=True):
            t = group['time_s'].to_numpy(float) / 86400.0
            mean = group['mean_pa'].to_numpy(float) / 1e6
            lower = group['lower95_pa'].to_numpy(float) / 1e6
            upper = group['upper95_pa'].to_numpy(float) / 1e6
            line, = ax.plot(t, mean, marker='o', label=f'Zone {zone}')
            ax.fill_between(t, lower, upper, alpha=0.16, color=line.get_color())
        ax.set(xlabel='Time (days)', ylabel='Pressure change from reference (MPa)',
               title=f'{model_name}: conditional equivalent pressure')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{model_name}_pressure_history.png', dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        for label, group in predictions.groupby('set'):
            ax.scatter(group['observed_delta_ns'], group['predicted_delta_ns'],
                       alpha=0.65, label=label, s=22)
        values = np.r_[predictions['observed_delta_ns'].to_numpy(float),
                       predictions['predicted_delta_ns'].to_numpy(float)]
        lo, hi = float(values.min()), float(values.max())
        ax.plot([lo, hi], [lo, hi], linestyle='--', color='gray')
        ax.set(xlabel='Observed COMSOL strain change (nanostrain)',
               ylabel='Predicted strain change (nanostrain)',
               title=f'{model_name}: selected times')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{model_name}_strain_parity.png', dpi=180)
        plt.close(fig)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Observed COMSOL strain changes inverted (relative to first sample).')
    print(f'Times: {selected.tolist()} ; reference time: {time[0]:g} s')
    for name, values in summary.items():
        print(f"{name}: zones={values['n_unknown_pressure_zones']}, training RMSE={values['training_rmse_ns']:.4g} ns, holdout RMSE={values['holdout_rmse_ns']:.4g} ns")
    print(f'Results: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == '__main__':
    main()
