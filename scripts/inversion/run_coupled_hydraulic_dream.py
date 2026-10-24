#!/usr/bin/env python3
"""Stage 3: PyDREAM homogeneous K/Ss inference on SAME saved Stage 1 strain data.

Run from AVANT repository root. Stage 2 MLE supplies *initial states only*;
prior support remains user-defined, and S07/S08 are withheld entirely.
Postprocess separately to inspect convergence BEFORE using posterior intervals.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path
import time

import numpy as np
from scipy.stats import uniform

from avant_model.inversion.coupled_dream import (
    WorkerHydraulicLikelihood, build_likelihood, generate_starts, load_stage1, normalize_chains,
    sha256_file,
)
from avant_model.inversion.coupled_parameters import HydraulicBounds, PARAMETER_NAMES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', required=True)
    p.add_argument('--map-dir', required=True, help='Stage 2 MLE result directory')
    p.add_argument('--hydraulic-config', required=True)
    p.add_argument('--mechanical-config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--niterations', type=int, default=2000, help='Draws PER chain, includes burn-in')
    p.add_argument('--nchains', type=int, default=4)
    p.add_argument('--burn-in', type=int, default=None, help='Default: half of niterations')
    p.add_argument('--logK-min', type=float, default=-7.0)
    p.add_argument('--logK-max', type=float, default=-4.0)
    p.add_argument('--logSs-min', type=float, default=-7.0)
    p.add_argument('--logSs-max', type=float, default=-4.0)
    p.add_argument('--init-logK-width', type=float, default=0.02)
    p.add_argument('--init-logSs-width', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mp-start', choices=('spawn', 'fork', 'forkserver'), default='spawn')
    return p


def checked_map(map_dir, source_dir, likelihood, bounds):
    summary = json.loads((Path(map_dir) / 'summary.json').read_text(encoding='utf-8'))
    if not summary.get('scope', '').startswith('Continuous maximum likelihood on SAME saved'):
        raise ValueError('Expected Stage 2 continuous fit as warm-start reference')
    if Path(summary['source_dir']).resolve() != Path(source_dir).resolve():
        raise ValueError('MAP and DREAM must use the same Stage 1 observation directory')
    center = np.asarray(summary['fit_log10_K_log10_Ss'], dtype=float)
    if not bounds.contains(center):
        raise ValueError('Stage 2 optimum is outside DREAM prior bounds')
    actual = float(likelihood(center))
    if abs(actual - float(summary['optimized_log_likelihood'])) > 1.e-5:
        raise RuntimeError('Stage 2 MLE likelihood differs from Stage 3; data/config mismatch')
    return summary, center


def main(argv=None):
    a = parser().parse_args(argv)
    if a.niterations < 40 or a.nchains < 3:
        raise ValueError('Need >=40 draws per chain and >=3 chains')
    burn = a.niterations // 2 if a.burn_in is None else int(a.burn_in)
    if burn < 0 or a.niterations - burn < 16:
        raise ValueError('Burn-in leaves too few retained draws')
    if a.mp_start not in mp.get_all_start_methods():
        raise ValueError(f'Multiprocessing start method unavailable: {a.mp_start}')

    bounds = HydraulicBounds((a.logK_min, a.logK_max), (a.logSs_min, a.logSs_max))
    hc = Path(a.hydraulic_config).resolve()
    mc = Path(a.mechanical_config).resolve()
    source = Path(a.source_dir).resolve()
    map_dir = Path(a.map_dir).resolve()
    mc_meta = json.loads(mc.read_text(encoding='utf-8'))
    station_file = Path(mc_meta['station_file']).resolve()
    output = Path(a.output_dir).resolve()
    stage1_summary, obs = load_stage1(source, hc, mc)
    print('Building fixed-geometry mechanical operator once in the parent process...', flush=True)
    likelihood, _ = build_likelihood(hc, mc, stage1_summary, obs, bounds)
    stage2_summary, center = checked_map(map_dir, source, likelihood, bounds)
    starts = generate_starts(center, bounds, a.nchains,
                             (a.init_logK_width, a.init_logSs_width), a.seed)
    # Pass only lightweight paths/bounds to workers. Each Windows spawned
    # process builds its fixed G/H operator on its first likelihood call,
    # then uses a process-local cache for the remaining proposals.
    worker_likelihood = WorkerHydraulicLikelihood(
        str(source), str(hc), str(mc), tuple(bounds.log10_K), tuple(bounds.log10_Ss))
    try:
        pickle.dumps(worker_likelihood)
    except Exception as exc:
        raise RuntimeError('Worker likelihood adapter is not picklable for Windows spawn') from exc
    print(f'Using {a.nchains} DREAM chains, {a.niterations} iterations each.', flush=True)
    print(f'Full uniform-log prior: {bounds.log10_K}, {bounds.log10_Ss}', flush=True)
    print('Stage 2 MLE is used for dispersed starts only:', starts.tolist(), flush=True)

    # Delayed import helps unit tests and avoids spawning PyDREAM workers when
    # merely reading this runner's command-line help.
    from pydream.core import run_dream
    from pydream.parameters import SampledParam

    priors = [SampledParam(uniform, loc=float(l), scale=float(h-l)) for l, h in
              (bounds.log10_K, bounds.log10_Ss)]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'scope': 'Same-model synthetic fixed-geometry, strain-only PyDREAM posterior; NOT field or COMSOL validation',
        'source_dir': str(source), 'map_dir': str(map_dir),
        'hydraulic_config': str(hc), 'mechanical_config': str(mc),
        'hydraulic_config_sha256': sha256_file(hc),
        'mechanical_config_sha256': sha256_file(mc),
        'station_file': str(station_file),
        'station_file_sha256': sha256_file(station_file),
        'stage1_observations_sha256': sha256_file(source / 'synthetic_observations.npz'),
        'stage1_summary_sha256': sha256_file(source / 'summary.json'),
        'stage2_summary_sha256': sha256_file(map_dir / 'summary.json'),
        'niterations_per_chain': a.niterations, 'nchains': a.nchains,
        'burn_in_per_chain': burn, 'seed_parent': a.seed, 'mp_start': a.mp_start,
        'parameter_names': list(PARAMETER_NAMES),
        'prior': 'independent uniform in log10 K and log10 Ss',
        'bounds_log10': [list(bounds.log10_K), list(bounds.log10_Ss)],
        'start_center_log10': center.tolist(), 'start_positions_log10': starts.tolist(),
        'training_data_count': int(obs.training_mask.sum() * len(obs.times_h)),
        'heldout_data_count': int(obs.holdout_mask.sum() * len(obs.times_h)),
        'sigma_ns': obs.sigma_ns, 'stage2_mle_log_likelihood': stage2_summary['optimized_log_likelihood'],
        'note': 'Initialization near MLE is a computational choice; it does not restrict the uniform prior. Check multiple modes and convergence.',
        'worker_forward_cache': 'Each spawned process constructs the fixed mechanical model once; subsequent hydraulic evaluations reuse it.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    np.random.seed(a.seed)
    started = time.perf_counter()
    # Run ONCE: do not append batches as if they formed continuous chains.
    states_list, logp_list = run_dream(
        parameters=priors, likelihood=worker_likelihood,
        niterations=a.niterations, nchains=a.nchains,
        start=[np.asarray(row, dtype=float) for row in starts],
        start_random=False,
        mp_context=mp.get_context(a.mp_start),
        snooker=True, adapt_gamma=True, save_history=False,
        model_name=str(output / 'dream_hydraulic'), verbose=False)
    runtime = time.perf_counter() - started
    states, logps = normalize_chains(states_list, logp_list, a.niterations, a.nchains)
    np.save(output / 'chains_log10.npy', states)
    np.save(output / 'log_posterior.npy', logps)
    manifest['sampler_runtime_s'] = float(runtime)
    manifest['sampling_completed'] = True
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Sampling completed. Saved:', output, flush=True)
    print('NEXT: run scripts/postprocessing/postprocess_coupled_hydraulic_dream.py', flush=True)
    print('Do NOT quote credible intervals before checking R-hat, ESS and traces.', flush=True)
    return output


if __name__ == '__main__':
    main()
