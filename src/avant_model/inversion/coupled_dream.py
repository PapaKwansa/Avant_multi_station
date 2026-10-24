"""Shared, testable Stage 3 DREAM tools for fixed-geometry homogeneous K/Ss.

All priors are uniform in base-10 logarithmic coordinates. Diagnostics use
rank-normalized split R-hat, bulk ESS, and quantile-tail ESS (Geyer IPS).
No field validation or pressure-gauge likelihood is implied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import norm, rankdata

from avant_model.inversion.coupled_likelihood import CoupledStrainLikelihood
from avant_model.inversion.coupled_observations import StrainObservations
from avant_model.inversion.coupled_parameters import HydraulicBounds, PARAMETER_NAMES


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_stage1(source_dir, hydraulic_config, mechanical_config):
    """Use ONLY Stage 1 saved observations, with SHA256 config verification."""
    source_dir = Path(source_dir)
    summary = json.loads((source_dir / 'summary.json').read_text(encoding='utf-8'))
    if 'Same-model synthetic' not in summary.get('scope', ''):
        raise ValueError('Stage 3 requires saved Stage 1 same-model synthetic observations')
    for path, key in ((hydraulic_config, 'hydraulic_config_sha256'),
                      (mechanical_config, 'mechanical_config_sha256')):
        if sha256_file(path) != summary[key]:
            raise ValueError(f'Configuration differs from Stage 1: {path}')
    with np.load(source_dir / 'synthetic_observations.npz', allow_pickle=False) as data:
        obs = StrainObservations(
            times_h=data['times_h'],
            channels=tuple(map(str, data['channels'])),
            observed_ns=data['observed_ns'],
            sigma_ns=float(summary['noise_std_ns']),
            training_mask=data['train_mask'],
            holdout_mask=data['holdout_mask'],
        )
    if obs.times_h.size * obs.training_mask.sum() != 408:
        raise ValueError('Expected the verified 408 training observations')
    if obs.times_h.size * obs.holdout_mask.sum() != 136:
        raise ValueError('Expected 136 held-out strain observations')
    return summary, obs


def build_likelihood(hydraulic_config, mechanical_config, summary, observations, bounds):
    """Construct fixed G, H, P once. Calling likelihood only reruns hydraulic FV."""
    import pandas as pd
    from avant_model.model.pressure_superposition import Cuboid, ElasticProperties
    from avant_model.model.coupled_hydromechanics import CoupledHydroMechanicalForward

    hc = json.loads(Path(hydraulic_config).read_text(encoding='utf-8'))
    mc = json.loads(Path(mechanical_config).read_text(encoding='utf-8'))
    station_path = Path(mc['station_file'])
    if not station_path.is_absolute():
        station_path = Path.cwd() / station_path
    stations = pd.read_csv(station_path)
    model = CoupledHydroMechanicalForward(
        hc, Cuboid(**mc['geometry']), ElasticProperties(**mc['elastic']), stations,
        mechanical_splits=(int(summary['fine_n']), int(summary['fine_n']), 1),
        components=tuple(mc['components']))
    return CoupledStrainLikelihood(model, observations, bounds), hc


def generate_starts(center, bounds, nchains, width=(0.02, 0.05), seed=42):
    """Independent dispersed warm starts; prior remains the full uniform domain."""
    center = np.asarray(center, float)
    widths = np.asarray(width, float)
    if nchains < 3 or center.shape != (2,) or widths.shape != (2,) or np.any(widths <= 0):
        raise ValueError('Need >=3 chains and two positive initialization widths')
    if not bounds.contains(center):
        raise ValueError('Initial center outside prior support')
    lo = np.array([bounds.log10_K[0], bounds.log10_Ss[0]])
    hi = np.array([bounds.log10_K[1], bounds.log10_Ss[1]])
    rng = np.random.default_rng(seed)
    starts = rng.normal(center, widths, (nchains, 2))
    starts = np.clip(starts, lo + 1.e-6, hi - 1.e-6)
    if np.unique(starts, axis=0).shape[0] != nchains:
        raise ValueError('Start positions collapsed to identical values')
    return starts


def normalize_chains(chains, logps, niterations, nchains):
    """Validate PyDREAM's list-of-arrays output before saving any samples."""
    if len(chains) != nchains or len(logps) != nchains:
        raise ValueError('Unexpected number of chains returned from PyDREAM')
    states = np.stack([np.asarray(c, float) for c in chains], axis=0)
    lp = np.stack([np.asarray(c, float).reshape(-1) for c in logps], axis=0)
    if states.shape != (nchains, niterations, 2) or lp.shape != (nchains, niterations):
        raise ValueError(f'Unexpected sampler shapes: {states.shape}, {lp.shape}')
    if not np.isfinite(states).all() or not np.isfinite(lp).all():
        raise ValueError('Non-finite DREAM chain states or posterior log probabilities')
    return states, lp


def _split_chains(samples):
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite chains of eight retained draws')
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def _rank_normalize(x):
    flat = np.asarray(x, float).ravel()
    ranks = rankdata(flat, method='average')
    u = (ranks - 0.375) / (len(ranks) + 0.25)
    return norm.ppf(u).reshape(np.asarray(x).shape)


def _rhat(x):
    x = np.asarray(x, float)
    m, n = x.shape
    W = np.mean(np.var(x, axis=1, ddof=1))
    B = n * np.var(np.mean(x, axis=1), ddof=1)
    if W <= 0:
        return 1.0 if B <= 0 else float('inf')
    return float(np.sqrt(((n - 1) * W / n + B / n) / W))


def _ess(x):
    """Split-chain effective sample size with paired positive autocorrelation truncation."""
    x = np.asarray(x, float)
    m, n = x.shape
    centered = x - np.mean(x, axis=1, keepdims=True)
    # FFT unbiased autocovariances, computed for each chain.
    fft_length = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(centered, n=fft_length, axis=1)
    acov = np.fft.irfft(f * np.conjugate(f), n=fft_length, axis=1)[:, :n]
    acov /= np.arange(n, 0, -1)[None, :]
    W = float(np.mean(acov[:, 0] * n / (n - 1)))
    B = n * float(np.var(np.mean(x, axis=1), ddof=1))
    var_plus = (n - 1) * W / n + B / n
    if not np.isfinite(var_plus) or var_plus <= 0:
        return float('nan')
    rho = np.ones(n)
    rho[1:] = 1.0 - (W - np.mean(acov[:, 1:], axis=0)) / var_plus
    # Geyer initial positive sequence, with adjacent pair monotonicity.
    pairs = []
    for t in range(0, n - 1, 2):
        pair = float(rho[t] + rho[t + 1])
        if not np.isfinite(pair) or pair <= 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    if not pairs:
        return float(m * n)
    tau = max(1.0, -1.0 + 2.0 * sum(pairs))
    return float(min(m * n, m * n / tau))


def rank_diagnostics(chains, burn_in):
    """Rank-normalized folded split R-hat, bulk and tail ESS per parameter."""
    x = np.asarray(chains, float)
    if x.ndim != 3 or x.shape[0] < 3 or x.shape[2] != 2:
        raise ValueError('Expected chain array (>=3 chains, draws, two parameters)')
    if not 0 <= burn_in < x.shape[1]:
        raise ValueError('Invalid burn-in')
    post = x[:, burn_in:, :]
    result = {}
    for j, name in enumerate(PARAMETER_NAMES):
        split = _split_chains(post[:, :, j])
        bulk = _rank_normalize(split)
        folded = _rank_normalize(np.abs(split - np.median(split)))
        rhat = max(_rhat(bulk), _rhat(folded))
        tail_ess = min(_ess((split <= np.quantile(split, q)).astype(float))
                       for q in (0.05, 0.95))
        result[name] = dict(rank_folded_split_rhat=float(rhat),
                            bulk_ess=float(_ess(bulk)), tail_ess=float(tail_ess))
    return result


def converged(diagnostics, threshold=1.01, min_ess=400, retained_per_chain=None):
    if retained_per_chain is not None and retained_per_chain < 100:
        return False
    return all(np.isfinite(d['rank_folded_split_rhat'])
               and d['rank_folded_split_rhat'] < threshold
               and d['bulk_ess'] >= min_ess and d['tail_ess'] >= min_ess
               for d in diagnostics.values())


@lru_cache(maxsize=8)
def _worker_likelihood(source, hydraulic_config, mechanical_config, bounds_k, bounds_ss):
    """Build mechanics once PER worker process, not once per DREAM proposal."""
    summary, obs = load_stage1(source, hydraulic_config, mechanical_config)
    ll, _ = build_likelihood(hydraulic_config, mechanical_config, summary, obs,
                             HydraulicBounds(bounds_k, bounds_ss))
    return ll


@dataclass(frozen=True)
class WorkerHydraulicLikelihood:
    """Small Windows-spawn-safe, pickleable callable passed to PyDREAM.

    Avoids transmitting the heavy 18x18 mechanical kernel and FV projection
    matrices with each candidate. A process-local LRU cache holds its model.
    """
    source: str
    hydraulic_config: str
    mechanical_config: str
    bounds_k: tuple
    bounds_ss: tuple

    def __call__(self, theta):
        likelihood = _worker_likelihood(
            self.source, self.hydraulic_config, self.mechanical_config,
            self.bounds_k, self.bounds_ss)
        return likelihood(theta)
