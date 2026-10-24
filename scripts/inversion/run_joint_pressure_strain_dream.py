#!/usr/bin/env python3
"""Stage 4B: fixed-observation, fixed-geometry, joint pressure+strain PyDREAM.

One FV solve per candidate generates FULL FV pressures for gauges and regional
pressure coefficients for analytical cuboid strain. Reuses the Stage 4A NPZ;
no random pressure or strain noise is created in MCMC evaluations.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from scipy.stats import uniform

from avant_model.inversion.coupled_dream import (
    generate_starts, normalize_chains, sha256_file)
from avant_model.inversion.coupled_parameters import HydraulicBounds, PARAMETER_NAMES
from avant_model.inversion.joint_dream import (
    WorkerJointLikelihood, build_joint_likelihood, load_stage4a,
    validate_joint_mle, worker_is_pickleable)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage4a-dir', required=True)
    p.add_argument('--hydraulic-config', required=True)
    p.add_argument('--mechanical-config', required=True)
    p.add_argument('--gauges-csv', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--niterations', type=int, default=2000, help='Iterations per chain INCLUDING burn-in')
    p.add_argument('--burn-in', type=int, default=None)
    p.add_argument('--nchains', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--logK-min', type=float, default=-7.)
    p.add_argument('--logK-max', type=float, default=-4.)
    p.add_argument('--logSs-min', type=float, default=-7.)
    p.add_argument('--logSs-max', type=float, default=-4.)
    p.add_argument('--init-logK-width', type=float, default=.02)
    p.add_argument('--init-logSs-width', type=float, default=.05)
    p.add_argument('--mp-start', choices=('spawn','fork','forkserver'), default='spawn')
    p.add_argument('--verbose', action='store_true', help='Enable potentially chatty PyDREAM output')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.niterations < 40 or a.nchains < 3:
        raise ValueError('At least 40 iterations/chain and three chains are required')
    burn = a.niterations//2 if a.burn_in is None else int(a.burn_in)
    if burn < 0 or a.niterations-burn < 16:
        raise ValueError('Invalid burn-in leaves fewer than 16 draws')
    if a.mp_start not in mp.get_all_start_methods():
        raise ValueError('Multiprocessing method not supported on this machine')
    bounds = HydraulicBounds((a.logK_min, a.logK_max),
                             (a.logSs_min, a.logSs_max))
    stage4a, hc, mc, gauges, output = (
        Path(a.stage4a_dir).resolve(), Path(a.hydraulic_config).resolve(),
        Path(a.mechanical_config).resolve(), Path(a.gauges_csv).resolve(),
        Path(a.output_dir).resolve())
    report, stage1, strain, pressure = load_stage4a(stage4a, hc, mc, gauges)
    print('Building fixed mechanical operator for likelihood verification...', flush=True)
    joint, _ = build_joint_likelihood(hc, mc, stage1, strain, pressure, bounds)
    mle, mle_ll = validate_joint_mle(joint, report, bounds)
    starts = generate_starts(mle, bounds, a.nchains,
                             (a.init_logK_width, a.init_logSs_width), a.seed)
    worker = WorkerJointLikelihood(str(stage4a), str(hc), str(mc), str(gauges),
                                   tuple(bounds.log10_K), tuple(bounds.log10_Ss))
    worker_is_pickleable(worker)
    print(f'Using {a.nchains} chains, {a.niterations} iterations per chain; nCR=2.', flush=True)
    print('Log-uniform prior bounds:', bounds.log10_K, bounds.log10_Ss, flush=True)
    print('Stage 4A joint MLE (starts only):', mle.tolist(), 'log L:', mle_ll, flush=True)
    print('Individual progress is quiet unless --verbose; worker calculations may be CPU intensive.', flush=True)
    # Delay import so --help and tests do not require launching PyDREAM workers.
    from pydream.core import run_dream
    from pydream.parameters import SampledParam
    priors = [SampledParam(uniform, loc=float(lo), scale=float(hi-lo))
              for lo, hi in (bounds.log10_K, bounds.log10_Ss)]
    if output.exists():
        raise FileExistsError(f'Choose a NEW Stage 4B directory; already exists: {output}')
    output.mkdir(parents=True)
    from json import loads
    station_file = Path(loads(mc.read_text(encoding='utf-8'))['station_file']).resolve()
    source = Path(report['source_dir']).resolve()
    map_dir = Path(report['stage2_map_dir']).resolve()
    manifest = {
        'scope': 'Stage 4B same-model synthetic joint strain+gauge pressure DREAM; not field/COMSOL validation',
        'stage4a_dir': str(stage4a), 'hydraulic_config': str(hc),
        'mechanical_config': str(mc), 'gauges_csv': str(gauges),
        'stage1_dir': str(source), 'stage2_map_dir': str(map_dir),
        'stage4a_report_sha256': sha256_file(stage4a/'joint_summary.json'),
        'stage4a_pressure_observations_sha256': sha256_file(stage4a/'synthetic_pressure_observations.npz'),
        'stage1_summary_sha256': sha256_file(source/'summary.json'),
        'stage1_strain_observations_sha256': sha256_file(source/'synthetic_observations.npz'),
        'stage2_map_summary_sha256': sha256_file(map_dir/'summary.json'),
        'hydraulic_config_sha256': sha256_file(hc),
        'mechanical_config_sha256': sha256_file(mc),
        'gauges_csv_sha256': sha256_file(gauges),
        'station_file': str(station_file), 'station_file_sha256': sha256_file(station_file),
        'nchains': a.nchains, 'niterations_per_chain': a.niterations,
        'burn_in_per_chain': burn, 'seed_parent': a.seed,
        'mp_start': a.mp_start, 'crossover_nCR': 2,
        'prior': 'independent uniform in log10 K and log10 Ss',
        'bounds_log10': [list(bounds.log10_K),list(bounds.log10_Ss)],
        'parameter_names': list(PARAMETER_NAMES),
        'start_center_log10': mle.tolist(), 'start_positions_log10': starts.tolist(),
        'stage4a_joint_mle_loglikelihood': mle_ll,
        'train_strain_count': int(strain.training_mask.sum()*len(strain.times_h)),
        'holdout_strain_count': int(strain.holdout_mask.sum()*len(strain.times_h)),
        'train_pressure_count': int(pressure.training_mask.sum()*len(pressure.times_h)),
        'holdout_pressure_count': int(pressure.holdout_mask.sum()*len(pressure.times_h)),
        'sigma_strain_ns': float(strain.sigma_ns), 'sigma_pressure_pa': float(pressure.sigma_pa),
        'sampling_completed': False,
        'worker_cache_note': 'Fixed H, P and gauge weights built once per worker; new hydraulic solve per MCMC proposal',
    }
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    np.random.seed(a.seed)
    start_time = time.perf_counter()
    chains, logps = run_dream(
        parameters=priors, likelihood=worker, niterations=a.niterations,
        nchains=a.nchains, start=[np.asarray(v,float) for v in starts],
        start_random=False, mp_context=mp.get_context(a.mp_start),
        snooker=True, adapt_gamma=True, nCR=2, save_history=False,
        model_name=str(output/'dream_joint_pressure_strain'), verbose=a.verbose)
    runtime = time.perf_counter()-start_time
    states, lp = normalize_chains(chains,logps,a.niterations,a.nchains)
    np.save(output/'chains_log10.npy',states)
    np.save(output/'log_posterior.npy',lp)
    manifest['sampler_runtime_s'] = float(runtime)
    manifest['sampling_completed'] = True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Sampling completed:',output,flush=True)
    print('NEXT: python scripts/postprocessing/postprocess_joint_pressure_strain_dream.py --run-dir '+str(output),flush=True)
    return output


if __name__ == '__main__':
    main()
