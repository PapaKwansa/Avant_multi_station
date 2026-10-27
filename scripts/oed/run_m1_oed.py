#!/usr/bin/env python3
"""M1 nine-parameter local and scenario-robust pressure/strain OED pilot.

This runner reuses the M1 forward solver and the existing Fisher-information
implementation. All input measurements are hypothetical design candidates.
This is NOT field-validated OED, posterior-informed Bayesian OED, or a new
inverse parameter recovery. A non-converged DREAM posterior is NOT used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.m1_bayesian import (
    M1GammaCachedForward, build_forward, load_saved_m1,
)
from avant_model.inversion.m1_map_recovery import PARAMETER_NAMES, check_bounds
from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.model.stage6_coordinates import SourceRectangle
from avant_model.model.stage6_m1_forward import Stage6M1Forward
from avant_model.oed.m1_joint_design import (
    M1ObservationCatalog, compare_designs, greedy_add_sensors,
    make_m1_jacobian,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Original immutable stage6_m1_pilot_001.json')
    p.add_argument('--map-dir', required=True, help='Verified existing 81-grid synthetic MAP directory')
    p.add_argument('--design-config', required=True, help='M1 OED design definitions and uncertainty assumptions')
    p.add_argument('--output-dir', required=True, help='New directory; no overwriting')
    p.add_argument('--derivative-half-step-check', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Validate saved inputs/designs without full forward solves')
    return p


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_design(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('schema_version') != 'avant-m1-oed-1':
        raise ValueError('Unrecognized M1 OED design schema')
    if not isinstance(doc.get('designs'), list) or not doc['designs']:
        raise ValueError('No M1 observation designs supplied')
    return doc


def make_candidate_forward(config_path, saved, design):
    """Build exactly one fixed candidate-network model per M1 OED experiment.

    Existing synthetic sensor coordinates are used ONLY as illustrative design
    candidates. Extra candidates must be explicitly labeled hypothetical.
    """
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    extra_s = design.get('extra_strain_stations', [])
    extra_g = design.get('extra_pressure_gauges', [])
    if not isinstance(extra_s, list) or not isinstance(extra_g, list):
        raise ValueError('Candidate stations and gauges must be lists')
    names = [f'P{i+1:02d}' for i in range(len(saved.gauges_prime_m))]
    if not extra_s and not extra_g:
        model = build_forward(config_path, saved)
        if tuple(model.channels) != saved.channel_names:
            raise ValueError('Unexpected M1 channel ordering')
        return model, names
    contract = Stage6Contract.from_json(Path(cfg['stage6a_contract']))
    hc = json.loads(Path(cfg['hydraulic_config']).read_text(encoding='utf-8'))
    hc['nx'] = hc['ny'] = int(saved.map_report['grid_n'])
    stations = pd.read_csv(cfg['stations_csv'])
    expected = {'station', 'x', 'y', 'z'}
    if not expected.issubset(stations.columns):
        raise ValueError('Station CSV missing expected fields')
    for record in extra_s:
        if set(record) != expected:
            raise ValueError('Extra station must have station,x,y,z')
        if str(record['station']) in set(stations['station']):
            raise ValueError('Duplicate station name')
        stations = pd.concat([stations, pd.DataFrame([record])], ignore_index=True)
    gauges = saved.gauges_prime_m.tolist()
    for record in extra_g:
        if set(record) != {'gauge','x','y'}:
            raise ValueError('Extra gauge must have gauge,x,y')
        gauge = str(record['gauge'])
        if gauge in names or not gauge:
            raise ValueError('Duplicate/empty gauge label')
        names.append(gauge)
        gauges.append([float(record['x']), float(record['y'])])
    model = Stage6M1Forward(
        hc, contract.domain, contract.mechanics, stations.to_dict(orient='list'),
        SourceRectangle(**cfg['mechanical_support']),
        splits=tuple(cfg['mechanical_tile_splits']),
        quadrature_order=int(cfg['quadrature_order']),
        gauges_prime_m=np.asarray(gauges, dtype=float),
    )
    if tuple(channel for channel in model.channels if channel in saved.channel_names) != saved.channel_names:
        raise ValueError('Existing strain channels changed when appending candidates')
    return model, names


def _evaluate(predict, reference, catalog, spec, config, *, half_check=False):
    h = np.asarray(config['fd_steps'], float)
    scales = np.asarray(config['parameter_scales'], float)
    if scales.shape != (9,) or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError('Exactly nine positive parameter scales required')
    J = make_m1_jacobian(predict, reference, catalog, h)
    scores = compare_designs(J, catalog, spec, parameter_scales=scales,
                             prior_precision=float(config.get('prior_precision', 1.0)),
                             rank_rtol=float(config.get('rank_rtol', 1e-3)))
    stability = None
    if half_check:
        J_half = make_m1_jacobian(predict, reference, catalog, h/2)
        # Compute relative sensitivity difference across EVERY parameter.
        delta = np.linalg.norm(J - J_half, axis=0)
        baseline = np.maximum(np.linalg.norm(J, axis=0), 1e-12)
        stability = dict(zip(PARAMETER_NAMES, (delta/baseline).tolist()))
    return J, scores, stability


def _plot(scores, out):
    names = [r['name'] for r in scores]
    inf = np.array([r['information_gain_logdet'] for r in scores])
    rank = np.array([r['rank'] for r in scores])
    fig, ax = plt.subplots(figsize=(max(9, len(names)*1.0), 5), constrained_layout=True)
    xx = np.arange(len(names))
    ax.bar(xx, inf)
    ax.set_xticks(xx, names, rotation=40, ha='right')
    ax.set_ylabel('log det(I + scaled Fisher information)')
    ax.set_title('Local nine-parameter M1 OED: illustrative independent-error assumptions')
    for i, (value, rr) in enumerate(zip(inf, rank)):
        ax.text(i, value, f' rank {rr}/9', ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=.25)
    for ext in ('png','pdf'):
        fig.savefig(out/f'm1_oed_design_comparison.{ext}', dpi=180)
    plt.close(fig)


def main(argv=None):
    args = args_parser().parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Output already exists; refusing to overwrite: '+str(out))
    cfg_path = Path(args.config)
    design_path = Path(args.design_config)
    design = load_design(design_path)
    saved = load_saved_m1(args.map_dir, cfg_path)  # verifies immutable baseline hashes
    best = saved.map_report['fits']['full']['best']
    if best is None:
        raise ValueError('Existing M1 full MAP fit unavailable')
    reference = np.asarray(best['fitted_parameters'], float)
    bounds = check_bounds()
    if reference.shape != (9,) or not np.isfinite(reference).all() or np.any(reference < bounds[:,0]) or np.any(reference > bounds[:,1]):
        raise ValueError('Full MAP reference outside M1 verified parameter bounds')
    model, gauge_names = make_candidate_forward(cfg_path, saved, design)
    catalog = M1ObservationCatalog(
        saved.times_h, model.channels, gauge_names,
        strain_sigma_ns=float(design['strain_sigma_ns']),
        pressure_sigma_pa=float(design['pressure_sigma_pa']),
    )
    # Validate names/time indices for EVERY design before any forward calculation.
    for d in design['designs']:
        catalog.indices(d)
    greedy = design.get('greedy')
    if greedy is not None:
        catalog.indices(greedy['base_design'])
    provenance = {
        'm1_config_sha256':digest(cfg_path),
        'design_config_sha256':digest(design_path),
        'saved_map_sha256':saved.map_report_sha256,
        'saved_observations_sha256':saved.observations_sha256,
    }
    if args.dry_run:
        print('M1 OED dry-run PASS: compatible model, named candidate network, nine parameters and design masks.')
        print('Candidate rows:',len(catalog.rows),'Designs:',len(design['designs']))
        return
    predict = M1GammaCachedForward(model, saved.times_h)
    print('Computing one shared full candidate-network M1 Jacobian...',flush=True)
    J, scores, stability = _evaluate(predict, reference, catalog, design['designs'], design,
                                     half_check=args.derivative_half_step_check)
    augmented = None
    if greedy is not None:
        augmented = greedy_add_sensors(
            J, catalog,
            base_design=greedy['base_design'],
            candidate_stations=greedy.get('candidate_stations', []),
            candidate_gauges=greedy.get('candidate_gauges', []),
            max_additions=int(greedy.get('max_additions', 2)),
            parameter_scales=design['parameter_scales'],
            prior_precision=float(design.get('prior_precision',1.0)))
    # Optional explicitly declared nominal scenarios, NOT DREAM posterior draws.
    scenario_summaries = []
    for i, theta in enumerate(design.get('scenario_parameters', [])):
        x = np.asarray(theta,float)
        Jscenario, recs, _ = _evaluate(predict, x, catalog, design['designs'], design)
        scenario_summaries.append({'scenario_index': i, 'parameters':x.tolist(), 'design_scores':[
            {'name':r['name'], 'rank':r['rank'], 'information_gain_logdet':r['information_gain_logdet']}
            for r in recs]})
    # Equal-weight, explicitly declared scenarios; NOT an inferred posterior.
    robust_summary = []
    if scenario_summaries:
        for r in scores:
            values = [r['information_gain_logdet']] + [
                next(item['information_gain_logdet'] for item in scenario['design_scores']
                     if item['name'] == r['name']) for scenario in scenario_summaries]
            ranks = [r['rank']] + [
                next(item['rank'] for item in scenario['design_scores']
                     if item['name'] == r['name']) for scenario in scenario_summaries]
            robust_summary.append({'name':r['name'], 'scenario_count':len(values),
                                   'mean_information_gain':float(np.mean(values)),
                                   'worst_information_gain':float(np.min(values)),
                                   'minimum_effective_rank':int(min(ranks))})
    report = {
        'scope':'M1 nine-parameter local OED; hypothetical candidate instruments, not COMSOL or field validation',
        'parameter_names':list(PARAMETER_NAMES),
        'reference_source':'saved full nine-free-parameter synthetic MAP; NOT known generating truth',
        'reference_parameters':reference.tolist(),
        'n_times':len(catalog.times_h), 'n_channels':len(catalog.channels),
        'n_gauges':len(catalog.gauges), 'gauge_labels':list(catalog.gauges),
        'candidate_strain_stations':list(catalog.stations),
        'independent_error_assumptions':{'strain_sigma_ns':catalog.strain_sigma_ns,
                                         'pressure_sigma_pa':catalog.pressure_sigma_pa},
        'parameter_scales':list(map(float,design['parameter_scales'])),
        'fd_steps':list(map(float,design['fd_steps'])),
        'prior_precision_dimensionless':float(design.get('prior_precision',1.0)),
        'design_scores':scores, 'greedy_augmentation':augmented,
        'scenario_scores':scenario_summaries, 'robust_design_summary_equal_scenario_weights':robust_summary,
        'derivative_half_step_relative_differences':stability,
        'provenance':provenance,
        'limitations':[
            'Local linearized Fisher information, not global identifiability.',
            'MAP reference from same-model synthetic data; no independently validated geology.',
            'Candidate station/gauge positions are hypothetical, not measured instruments.',
            'Independent Gaussian error sigmas are assumptions, not field covariance.',
            'Ranking uses log det(I + F) with illustrative unit precision in scaled coordinates.',
            'Prior-regularized uncertainties are design proxies, NOT calibrated credible intervals.',
            'Finite-difference geometry sensitivity may depend on FV resolution and step size.',
            'Greedy augmentation is not globally optimal and compares gains per observation.',
            'Non-converged DREAM samples are not used for posterior-informed OED.',
            'Original analytical mechanics and M1 hydraulic numerical approximations unchanged.'
        ]
    }
    out.mkdir(parents=True)
    (out/'m1_oed_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(out/'m1_oed_jacobian.npz',
                        J=J, reference_parameters=reference, times_h=catalog.times_h,
                        sigma_per_observation=catalog.sigma,
                        row_kind=np.array([r.kind for r in catalog.rows]),
                        row_time_index=np.array([r.time_index for r in catalog.rows]),
                        row_station=np.array([r.station or '' for r in catalog.rows]),
                        row_component=np.array([r.component or '' for r in catalog.rows]),
                        row_gauge=np.array([r.gauge or '' for r in catalog.rows]))
    with (out/'m1_oed_design_scores.csv').open('w',newline='',encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['name','n_observations','n_strain','n_pressure','rank',
                                                'condition_number_unregularized', 'information_gain_logdet',
                                                'a_score_prior_regularized','e_score_prior_regularized'])
        writer.writeheader()
        for row in scores:
            writer.writerow({k:row[k] for k in writer.fieldnames})
    _plot(scores, out)
    for r in scores:
        print(f"{r['name']:<24} rank={r['rank']}/9  measurements={r['n_observations']} "
              f"information_gain={r['information_gain_logdet']:.3f}",flush=True)
    print('Saved M1 OED results:',out)
    print('No scientific field/posterior interpretation until independent validation.')


if __name__ == '__main__':
    main()
