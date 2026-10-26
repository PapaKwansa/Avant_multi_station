#!/usr/bin/env python3
"""M1 joint pressure/strain Bayesian DREAM using SAME saved MAP observations.

All nine parameters are sampled; none is held at synthetic truth. Geological
prior constraints are soft and illustrative. Does NOT claim COMSOL/field data.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from avant_model.inversion.m1_bayesian import (
    WorkerM1Likelihood, assert_worker_pickles, build_forward,
    generate_chain_starts, load_saved_m1, make_pydream_priors,
    normalize_dream, read_prior_config, sha256, validate_map_likelihood,
    M1BayesianLikelihood, M1GammaCachedForward,
)
from avant_model.inversion.m1_map_recovery import PARAMETER_NAMES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map-dir', required=True, help='Existing immutable M1 synthetic MAP directory')
    p.add_argument('--config', required=True, help='Same M1 pilot JSON used to create the observations')
    p.add_argument('--prior-config', required=True, help='Explicit geological priors; all 9 remain free')
    p.add_argument('--output-dir', required=True, help='NEW directory; never overwrite previous runs')
    p.add_argument('--niterations', type=int, default=1600, help='Per chain, including burn-in')
    p.add_argument('--burn-in', type=int, default=None)
    p.add_argument('--nchains', type=int, default=4)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--mp-start', choices=('spawn','fork','forkserver'), default='spawn')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--verify-only', action='store_true', help='Only check MAP likelihood and prior; no DREAM')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.nchains < 4 or a.niterations < 40:
        raise ValueError('At least four chains and 40 iterations per chain are required')
    burn = a.niterations // 2 if a.burn_in is None else a.burn_in
    if burn < 0 or a.niterations-burn < 20:
        raise ValueError('Burn-in must leave at least 20 samples per chain')
    if a.mp_start not in mp.get_all_start_methods():
        raise ValueError('Unsupported multiprocessing method')
    inputs = {k:str(Path(v).resolve()) for k,v in (
        ('map_dir',a.map_dir),('config',a.config),('prior_config',a.prior_config))}
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError('Output already exists; choose a NEW directory: '+str(out))
    saved = load_saved_m1(inputs['map_dir'],inputs['config'])
    priors = read_prior_config(inputs['prior_config'])
    print('Building unchanged M1 analytical kernel to verify saved MAP likelihood...', flush=True)
    model = build_forward(inputs['config'],saved)
    likelihood = M1BayesianLikelihood(saved,M1GammaCachedForward(model,saved.times_h))
    start_center, start_ll = validate_map_likelihood(likelihood,saved.map_report)
    starts = generate_chain_starts(start_center, priors, nchains=a.nchains, seed=a.seed)
    worker = WorkerM1Likelihood(inputs['map_dir'],inputs['config'])
    assert_worker_pickles(worker)
    manifest = {
        'scope':'M1 nine-parameter, synthetic joint pressure-strain DREAM; not COMSOL/field validation',
        'model':'one-way 2D FV two-zone K; fixed mechanical support; unchanged cuboid half-space',
        'sampler':'PyDREAM with all nine parameters free and explicit bounded priors',
        'prior_is_illustrative_not_measured':True,
        **inputs,
        'hashes':{
            'map_report':saved.map_report_sha256,
            'saved_observations':saved.observations_sha256,
            'config':sha256(inputs['config']),
            'prior_config':sha256(inputs['prior_config'])},
        'parameters':list(PARAMETER_NAMES),
        'grid_n':int(saved.map_report['grid_n']),
        'nchains':int(a.nchains),
        'niterations':int(a.niterations),
        'burn_in':int(burn),
        'start_seed':int(a.seed),
        'start_positions':starts.tolist(),
        'full_MAP':start_center.tolist(),
        'MAP_train_chi2':float(-2*start_ll),
        'noise_model':saved.map_report['noise_model'],
        'training_strain_count':int(saved.map_report['n_training_strain']),
        'training_pressure_count':int(saved.map_report['n_training_pressure']),
        'heldout_strain_count':int(saved.map_report['n_heldout_strain']),
        'heldout_pressure_count':int(saved.map_report['n_heldout_pressure']),
        'prior_provenance':priors['prior_provenance'],
        'mp_start':a.mp_start,
        'sampling_completed':False,
        'truth_not_used_in_priors_likelihood_or_fixed_params':True,
    }
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if a.verify_only:
        (out/'preflight.json').write_text(json.dumps({
            'result':'M1 MAP likelihood and illustrative prior configuration verified',
            'map_training_chi2_reproduced':float(-2*start_ll),
            'prior_means':{name:priors['parameters'][name].get('mean',None) for name in PARAMETER_NAMES},
            'no_DREAM_run':True},indent=2),encoding='utf-8')
        print('M1 Bayesian preflight PASS. No samples generated:',out,flush=True)
        return
    from pydream.core import run_dream
    print(f'Running PyDREAM with {a.nchains} chains x {a.niterations} iterations; '
          'all 9 M1 parameters free. Convergence NOT assumed.', flush=True)
    sampled = make_pydream_priors(priors)
    np.random.seed(a.seed)
    t0 = time.perf_counter()
    chains, logps = run_dream(
        parameters=sampled, likelihood=worker, nchains=a.nchains,
        niterations=a.niterations, start=[v.copy() for v in starts],
        start_random=False, mp_context=mp.get_context(a.mp_start),
        snooker=True, adapt_gamma=True, nCR=3, save_history=False,
        model_name=str(out/'m1_dream'), verbose=a.verbose,
    )
    states, lp = normalize_dream(chains,logps,a.niterations,a.nchains)
    np.save(out/'chains_physical_parameters.npy',states)
    np.save(out/'sampler_log_posterior.npy',lp)
    manifest['sampling_completed'] = True
    manifest['sampler_runtime_seconds'] = float(time.perf_counter()-t0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Sampling saved. Run postprocessor to assess Rhat/ESS; samples are NOT automatically converged:',out)


if __name__ == '__main__':
    main()
