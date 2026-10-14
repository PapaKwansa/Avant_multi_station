"""Small synthetic DREAM smoke test. NOT a converged inversion or COMSOL fit.

Run from the repository root. This runner does not modify the production DREAM
workflow. Inputs are already-converted analytical depths from the Larry config.
"""
import argparse
import hashlib
import json
from functools import partial
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from avant_model.model.pressure_superposition import (
    Cuboid, ElasticProperties, build_pressure_kernel, coarse_block_basis,
)
from avant_model.inversion.geometry_pressure_adapter import evaluate_geometry_pressure


def geometry_from_vector(x, fixed):
    """Enforce b>a using positive width difference rather than rejection."""
    a, delta_b, theta, x0, y0, log_sigma = map(float, x)
    if a <= 0 or delta_b <= 0:
        raise ValueError('a and b-a must be positive')
    return Cuboid(a=a, b=a + delta_b, c=float(fixed['c']),
                  h=float(fixed['h']), x0_prime=x0, y0_prime=y0,
                  theta_deg=theta), 10.0 ** log_sigma


def load_setup(cfg):
    benchmark = json.loads((ROOT / cfg['benchmark_config']).read_text())
    stations = pd.read_csv(ROOT / benchmark['station_file'])
    required = {'station', 'x', 'y', 'z'}
    if not required.issubset(stations.columns):
        raise ValueError(f'Station snapshot requires {sorted(required)}; found {stations.columns.tolist()}')
    selected = stations[~stations.station.isin(cfg['holdout_stations'])].copy()
    if selected.empty or selected.station.duplicated().any():
        raise ValueError('Invalid training station selection')
    return benchmark, selected


def make_problem(cfg):
    benchmark, stations = load_setup(cfg)
    g = benchmark['geometry']
    fixed = {'c': g['c'], 'h': g['h']}
    elastic = ElasticProperties(**benchmark['elastic'])
    x = stations.x.to_numpy(float)
    y = stations.y.to_numpy(float)
    z = stations.z.to_numpy(float)
    names = tuple(stations.station.astype(str))
    components = tuple(benchmark['components'])
    fine = tuple(cfg['fine_splits'])
    coarse = tuple(cfg['pressure_splits'])
    truth = np.asarray(cfg['truth_vector'], float)
    truth_geometry, _ = geometry_from_vector(truth, fixed)
    kernel = build_pressure_kernel(truth_geometry, elastic, x_prime=x,
                                   y_prime=y, z=z, station_names=names,
                                   splits=fine, components=components)
    basis = np.asarray(coarse_block_basis(fine, coarse), float)
    H = kernel.G @ basis
    pressures = np.asarray(cfg['synthetic_pressure_pa'], float)
    if H.shape[1] != len(pressures):
        raise ValueError('Pressure truth length does not match pressure zones')
    rng = np.random.default_rng(int(cfg['seed']))
    noise = float(cfg['synthetic_noise_std_ns'])
    if noise <= 0:
        raise ValueError('Synthetic noise must be positive')
    observed = H @ pressures + rng.normal(0, noise, H.shape[0])
    return dict(fixed=fixed, elastic=elastic, x=x, y=y, z=z,
                names=names, components=components, fine=fine, coarse=coarse,
                observed=observed, channels=kernel.channels, H_truth=H)


def log_likelihood(x, problem, cfg):
    try:
        geom, sigma = geometry_from_vector(x, problem['fixed'])
        if not np.isfinite(sigma) or sigma <= 0:
            return -np.inf
        n = int(np.prod(cfg['pressure_splits']))
        result = evaluate_geometry_pressure(
            geom, problem['elastic'], x_prime=problem['x'],
            y_prime=problem['y'], z=problem['z'], station_names=problem['names'],
            observed_ns=problem['observed'], observed_channels=problem['channels'],
            fine_splits=problem['fine'], pressure_splits=problem['coarse'],
            components=problem['components'],
            prior_mean_pa=np.full(n, float(cfg['prior_mean_pa'])),
            prior_covariance_pa2=np.eye(n) * float(cfg['prior_std_pa'])**2,
            noise_covariance_ns2=np.eye(len(problem['observed'])) * sigma**2,
        )
        return float(result.log_likelihood)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
        return -np.inf


def make_dream_likelihood(problem, cfg):
    """Create a picklable PyDREAM likelihood for Windows spawn workers."""
    return partial(log_likelihood, problem=problem, cfg=cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Verify geometry-dependent likelihood without DREAM')
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    problem = make_problem(cfg)
    bounds = np.asarray(cfg['bounds'], float)
    if bounds.shape != (6, 2) or not np.all(bounds[:, 1] > bounds[:, 0]):
        raise ValueError('Expected six finite lower/upper bounds')
    truth = np.asarray(cfg['truth_vector'], float)
    if not np.all((truth >= bounds[:, 0]) & (truth <= bounds[:, 1])):
        raise ValueError('Truth is outside bounds')
    score_truth = log_likelihood(truth, problem, cfg)
    score_mid = log_likelihood(bounds.mean(axis=1), problem, cfg)
    if not np.isfinite(score_truth) or not np.isfinite(score_mid):
        raise ValueError('Non-finite synthetic likelihood at truth or midpoint')
    summary = {'scope': 'same-model synthetic smoke only',
               'truth_log_likelihood': score_truth,
               'midpoint_log_likelihood': score_mid,
               'n_training_channels': len(problem['observed']),
               'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
               'pressure_coefficients_marginalized': int(np.prod(cfg['pressure_splits']))}
    np.savez(out / 'synthetic_problem.npz', observed_ns=problem['observed'],
             truth_design_ns_per_pa=problem['H_truth'], truth_vector=truth)
    if not args.evaluate_only:
        try:
            from pydream.core import run_dream
            from pydream.parameters import SampledParam
            from scipy.stats import uniform
        except ImportError as exc:
            raise RuntimeError('Install PyDREAM and scipy in the vrugt environment') from exc
        params = [SampledParam(uniform, loc=float(lo), scale=float(hi-lo)) for lo, hi in bounds]
        likelihood = make_dream_likelihood(problem, cfg)
        chains, logps = run_dream(parameters=params, likelihood=likelihood,
                                  niterations=int(cfg['niterations']),
                                  nchains=int(cfg['nchains']),
                                  model_name=str(out / 'synthetic_geometry_pressure'),
                                  verbose=True, restart=False, multitry=False, adapt_gamma=True,)
        for i, (chain, logp) in enumerate(zip(chains, logps)):
            np.save(out / f'chain_{i:02d}.npy', chain)
            np.save(out / f'logp_{i:02d}.npy', logp)
        summary['nchains'] = len(chains)
        summary['niterations'] = int(cfg['niterations'])
        summary['warning'] = 'Smoke run only; do not interpret chains as converged posterior.'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
