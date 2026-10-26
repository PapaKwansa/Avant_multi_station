#!/usr/bin/env python3
"""One bounded, same-model synthetic M1 MAP recovery. No COMSOL/field data.

Use this only AFTER the M1 local identifiability and numerical gate has passed.
Outputs a complete train/holdout split, multistart results, parameter errors,
and prediction diagnostics. No posterior or global uniqueness claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.inversion.m1_map_recovery import (
    CONSTRAINED_FIXED, PARAMETER_NAMES, ParameterSubset, SEARCH_BOUNDS,
    ObservationLayout, station_train_holdout_masks, add_independent_noise,
    fit_multi_start, parameter_error_table, check_bounds,
)
from avant_model.model.stage6_coordinates import SourceRectangle, source_corners_global
from avant_model.model.stage6_m1_forward import Stage6M1Forward, M1State

TIMES_H = np.array([2., 4., 6., 8., 10., 12., 14., 16., 18., 19.,
                    20., 22., 24., 36., 48., 72., 96.])
# First two gauges reuse hypothetical off-well sites from the M1 gate.
# The third is a NEW, explicitly invented synthetic held-out location.
GAUGES = np.array([[-185., 25.], [300., 150.], [0., -350.]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_state(theta):
    p = np.asarray(theta, dtype=float)
    if p.shape != (9,) or not np.isfinite(p).all():
        raise ValueError('M1 state must be a finite 9-element parameter vector')
    return M1State(10**p[0], 10**p[1], 10**p[2], 10**p[3],
                   SourceRectangle(*p[4:]))


def extract_config_reference(cfg, contract):
    zone = contract.reference().source
    return np.array([
        np.log10(cfg['K_inside_m_s']),
        np.log10(cfg['K_outside_m_s']),
        np.log10(cfg['Ss_per_m']),
        np.log10(cfg['gamma_rel']),
        zone.a_m, zone.b_m, zone.theta_deg,
        zone.x0_prime_m, zone.y0_prime_m,
    ], dtype=float)


def create_model(cfg, contract, *, grid_n):
    hydraulic = json.loads(Path(cfg['hydraulic_config']).read_text(encoding='utf-8'))
    hydraulic['nx'] = int(grid_n)
    hydraulic['ny'] = int(grid_n)
    stations = pd.read_csv(cfg['stations_csv']).to_dict(orient='list')
    support = SourceRectangle(**cfg['mechanical_support'])
    return Stage6M1Forward(hydraulic, contract.domain, contract.mechanics,
                            stations, support,
                            splits=tuple(cfg['mechanical_tile_splits']),
                            quadrature_order=int(cfg['quadrature_order']),
                            gauges_prime_m=GAUGES)


class GammaCachedForward:
    """Cache an unchanged FV solution when only mechanical gamma changes.

    This is a one-entry, exact key cache: it does not interpolate parameters or
    modify the numerical response. Pressure is independent of gamma in M1.
    """
    def __init__(self, model, times):
        self.model = model
        self.times = np.asarray(times, dtype=float)
        self.last_key = None
        self.last_result = None
        self.full_forward_solves = 0

    def __call__(self, theta):
        p = np.asarray(theta, dtype=float)
        key = tuple(np.r_[p[:3], p[4:]])
        if self.last_key != key:
            base = p.copy()
            base[3] = 0.  # gamma=1: exact scaling in fixed linear elastic kernel
            self.last_result = self.model.predict(make_state(base), self.times)
            self.full_forward_solves += 1
            self.last_key = key
        r = self.last_result
        return np.asarray(r.strain_ns) * 10**p[3], np.asarray(r.pressure_gauges_pa)


def print_fit(label, best, truth):
    if best is None:
        print(f'{label}: NO successful optimizations (do not use for inference)')
        return
    est = np.asarray(best['fitted_parameters'])
    print(f'{label}: chi2={best["chi2"]:.3f}, start={best["start_id"]}, '
          f'nfev={best["nfev"]}, bound_hit={best["at_bounds"]}')
    for nm, a, b in zip(PARAMETER_NAMES, truth, est):
        print(f'  {nm:<19s} truth={a:>11.5f} fitted={b:>11.5f} diff={b-a:+.5f}')


def plot_results(out, modes, truth, observed, predict, layout):
    modes = {k:v for k,v in modes.items() if v.get('best') is not None}
    if not modes:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False)
    axes = axes.ravel()
    span = check_bounds()[:, 1] - check_bounds()[:, 0]
    for i, (nm, ax) in enumerate(zip(PARAMETER_NAMES, axes)):
        for j,(label, data) in enumerate(modes.items()):
            p = np.asarray(data['best']['fitted_parameters'])
            ax.scatter(j, (p[i]-truth[i])/span[i], s=55, label=label)
        ax.axhline(0, linewidth=1, linestyle='--')
        ax.set_xticks(range(len(modes)), list(modes), rotation=20)
        ax.set_title(nm)
        ax.set_ylabel('(fitted - truth) / search-range width')
    fig.suptitle('Same-model synthetic M1 MAP recovery; not field validation')
    fig.tight_layout()
    fig.savefig(out/'m1_map_parameter_errors.png', dpi=180)
    fig.savefig(out/'m1_map_parameter_errors.pdf')
    plt.close(fig)
    # Plot only *held-out* channels to avoid hiding overfit behind good training fits.
    s_obs, p_obs = observed
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    all_chan = list(predict.model.channels) if hasattr(predict, 'model') else []
    channel = 'eXX_S07'
    idx = all_chan.index(channel) if channel in all_chan else int(np.flatnonzero(layout.channel_holdout_mask)[0])
    axes[0].plot(TIMES_H, s_obs[:,idx], 'o', label='Synthetic held-out strain observations')
    axes[1].plot(TIMES_H, p_obs[:,2]/1000, 'o', label='Synthetic held-out pressure observations')
    for label, data in modes.items():
        theta=np.asarray(data['best']['fitted_parameters'])
        s,p=predict(theta)
        axes[0].plot(TIMES_H,s[:,idx],label=f'{label} prediction')
        axes[1].plot(TIMES_H,p[:,2]/1000,label=f'{label} prediction')
    axes[0].set(title=channel, xlabel='Time (h)', ylabel='Strain (nanostrain)')
    axes[1].set(title='Held-out invented gauge at (0,-350) m', xlabel='Time (h)',ylabel='Pressure change (kPa)')
    for ax in axes: ax.legend(fontsize=8)
    fig.suptitle('Held-out synthetic predictions; independent Gaussian noise')
    fig.tight_layout()
    fig.savefig(out/'m1_map_heldout_predictions.png', dpi=180)
    fig.savefig(out/'m1_map_heldout_predictions.pdf')
    plt.close(fig)
    # Explicitly label K maps as prescribed or fitted, not inverse uncertainty maps.
    models=list(modes.items())
    fig, axes = plt.subplots(1, len(models)+1, figsize=(5*(len(models)+1),4))
    axes = np.atleast_1d(axes)
    true_field = predict.model.predict(make_state(truth), TIMES_H[:1]).conductivity_field_m_s
    fields=[('Generating K field', true_field)]
    for label, data in models:
        fit=np.asarray(data['best']['fitted_parameters'])
        fields.append((f'{label} fitted K field', predict.model.predict(make_state(fit), TIMES_H[:1]).conductivity_field_m_s))
    for ax,(name,field) in zip(axes,fields):
        im=ax.imshow(np.log10(field),origin='lower',aspect='auto',extent=predict.model.hydraulic.domain.extent(),vmin=-7,vmax=-5.7)
        ax.set_title(name);ax.set_xlabel("x' (m)");ax.set_ylabel("y' (m)")
    fig.colorbar(im, ax=axes.tolist(), label='log10 K [m/s]', shrink=.75)
    fig.savefig(out/'m1_map_k_fields.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--grid-n',type=int,default=81,help='Use 41 for cheap smoke; final benchmark should use 81')
    p.add_argument('--starts',type=int,default=2)
    p.add_argument('--max-nfev',type=int,default=50)
    p.add_argument('--mode',choices=('both','constrained','full'),default='both')
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--strain-sigma-ns',type=float,default=2.0)
    p.add_argument('--pressure-sigma-pa',type=float,default=20000.)
    args=p.parse_args(argv)
    if args.grid_n < 21 or args.grid_n > 301 or args.grid_n % 2 == 0:
        p.error('--grid-n must be odd and between 21 and 301')
    if args.strain_sigma_ns<=0 or args.pressure_sigma_pa<=0:
        p.error('Noise sigmas must be strictly positive')
    out=Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Output folder exists; will not overwrite: '+str(out))
    cfg_path=Path(args.config)
    cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
    contract_path=Path(cfg['stage6a_contract'])
    contract=Stage6Contract.from_json(contract_path)
    truth=extract_config_reference(cfg,contract)
    check_bounds()
    ParameterSubset(truth)
    t0=time.perf_counter()
    print('Constructing unchanged analytical kernel and fixed global hydraulic model...')
    model=create_model(cfg,contract,grid_n=args.grid_n)
    predict=GammaCachedForward(model,TIMES_H)
    true_strain,true_pressure=predict(truth)
    mask_train,mask_holdout=station_train_holdout_masks(model.channels)
    layout=ObservationLayout(mask_train,mask_holdout,(0,1),(2,),
                             args.strain_sigma_ns,args.pressure_sigma_pa)
    observations=add_independent_noise(true_strain,true_pressure,layout,args.seed)
    output={'scope':'Same-model synthetic bounded Gaussian MAP; no COMSOL or field observations, no DREAM posterior',
            'branch':'feature/full-hydromechanical-spatial-inversion',
            'generating_parameters':dict(zip(PARAMETER_NAMES,truth.tolist())),
            'parameter_search_bounds':SEARCH_BOUNDS,
            'synthetic_fixed_oracle_names':list(CONSTRAINED_FIXED),
            'training_stations':['S01','S02','S03','S04','S05','S06'],
            'held_out_stations':['S07','S08'],
            'training_gauges_global_prime_m':GAUGES[:2].tolist(),
            'held_out_gauge_global_prime_m':GAUGES[2:].tolist(),
            'all_gauge_locations_are_synthetic':True,
            'synthetic_seed':args.seed,
            'noise_model':{'kind':'independent Gaussian','strain_sigma_ns':args.strain_sigma_ns,
                           'pressure_sigma_pa':args.pressure_sigma_pa},
            'n_training_strain':int(len(TIMES_H)*mask_train.sum()),
            'n_training_pressure':int(len(TIMES_H)*2),
            'n_heldout_strain':int(len(TIMES_H)*mask_holdout.sum()),
            'n_heldout_pressure':int(len(TIMES_H)),
            'grid_n':args.grid_n,'n_starts':args.starts,'max_nfev':args.max_nfev,
            'time_h':TIMES_H.tolist(),
            'provenance_sha256':{'config':sha(cfg_path),'contract':sha(contract_path),
                                'hydraulic_config':sha(cfg['hydraulic_config']),
                                'stations_csv':sha(cfg['stations_csv'])},
            'limitations':['The generating truth and fitted model use the exact same code and same grid.',
                           'Constrained mode fixes three properties to their EXACT synthetic generating values; optimistic oracle benchmark.',
                           'Bounds are chosen for a synthetic pilot, not inferred from a field survey.',
                           'Independent measurement errors only; Stage 5D showed that real temporal correlations affect coverage.',
                           'Held-out locations and all gauge measurements are hypothetical, not COMSOL/field measurements.',
                           'MAP point is bounded uniform-prior Gaussian MLE, not Bayesian posterior or proof of global uniqueness.',
                           'Mechanical integration support and homogeneous half-space kernel are fixed, approximated, not independently validated.',
                           'The mixed-cell hydraulic interface is a numerical approximation; field boundary interpretation remains conditional.',
                           'Full M1 DREAM must not be launched solely because these MAP fits complete.']}
    modes={}
    requested = ('constrained','full') if args.mode=='both' else (args.mode,)
    for i,label in enumerate(requested):
        fixed=CONSTRAINED_FIXED if label=='constrained' else ()
        subset=ParameterSubset(truth,fixed)
        print(f'Fitting {label}: {len(subset.free_indices)} free params, {args.starts} start(s), max_nfev={args.max_nfev}...')
        best, all_starts, n_calls=fit_multi_start(predict, observations,layout,subset,
                                               n_starts=args.starts,max_nfev=args.max_nfev,seed=args.seed+100+i)
        record={'free_parameters':list(subset.free_names), 'fixed_parameters':list(fixed),
                'successful_starts':sum(r['success'] for r in all_starts),
                'total_starts':len(all_starts), 'total_objective_calls':n_calls,
                'start_results':all_starts, 'best':best}
        if best is not None:
            theta_fit=np.asarray(best['fitted_parameters'],float)
            s_hat,p_hat=predict(theta_fit)
            record['parameter_errors']=parameter_error_table(truth,theta_fit)
            train_s,train_p=layout.rmse(s_hat,p_hat,*observations,subset='train')
            hold_s,hold_p=layout.rmse(s_hat,p_hat,*observations,subset='holdout')
            record['train_strain_rmse_ns']=train_s
            record['train_pressure_rmse_kpa']=train_p
            record['heldout_strain_rmse_ns']=hold_s
            record['heldout_pressure_rmse_kpa']=hold_p
            record['heldout_vs_noisy_data_only']=True
            print_fit(label,best,truth)
            print(f'  held-out strain RMSE={hold_s:.3f} ns; pressure RMSE={hold_p:.3f} kPa')
        modes[label]=record
    output['fits']=modes
    output['total_full_forward_solves']=predict.full_forward_solves
    output['elapsed_seconds']=time.perf_counter()-t0
    output['pilot_decision'] = ('MAP FITS AVAILABLE FOR REVIEW; NOT M1 DREAM GO'
                               if all(r['best'] is not None for r in modes.values()) else
                               'ONE OR MORE MAP MODES DID NOT CONVERGE; INVESTIGATE BEFORE DREAM')
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'m1_synthetic_observations.npz',
                        times_h=TIMES_H,channels=np.array(model.channels),
                        strain_true_ns=true_strain,pressure_true_pa=true_pressure,
                        strain_observed_ns=observations[0],pressure_observed_pa=observations[1],
                        gauges_global_prime_m=GAUGES,train_channel_mask=mask_train,
                        heldout_channel_mask=mask_holdout,parameter_truth=truth)
    (out/'m1_map_recovery_report.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    rows=[]
    for label,r in modes.items():
        if r['best'] is not None:
            for entry in r['parameter_errors']:
                rows.append({'fit_mode':label,**entry})
    pd.DataFrame(rows,columns=['fit_mode','parameter','generating_value','fitted_value','difference']).to_csv(
        out/'m1_map_parameter_errors.csv',index=False)
    if any(r['best'] is not None for r in modes.values()):
        plot_results(out,modes,truth,observations,predict,layout)
    print('Saved:',out)
    print('Model full FV solves:',predict.full_forward_solves)
    print('Decision:',output['pilot_decision'])
    print('This is same-model synthetic MAP, not independent validation or posterior inference.')

if __name__=='__main__':
    main()
