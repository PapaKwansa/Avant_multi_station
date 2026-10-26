#!/usr/bin/env python3
"""One bounded M1 numerical/identifiability pilot; NOT an inversion or field validation.

Works only with the original Stage 6 M1 configuration and unchanged legacy
strain equations. The output folder must not exist.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.m1_identifiability_gate import (
    PARAMETERS, scaled_information, central_difference,
    relative_column_changes, rms_difference, relative_change,
)
from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.model.stage6_coordinates import SourceRectangle
from avant_model.model.stage6_m1_forward import M1State, Stage6M1Forward

# Relative to a *hypothetical* sensor noise model. Not a measured M1 covariance.
DEFAULT_STRAIN_SIGMA_NS=2.0
DEFAULT_PRESSURE_SIGMA_PA=20000.0
TIMES=np.array([2.,4.,6.,8.,10.,12.,14.,16.,18.,19.,20.,22.,24.,36.,48.,72.,96.])
PARAMETER_SCALES=np.array([.5,.5,.5,.4,60.,100.,12.,75.,75.])
# Central-difference steps are numerical, not posterior/prior widths.
FD_STEPS=np.array([.006,.006,.006,.006,2.,2.,.5,2.,2.])


def state_from_vector(theta):
    p=np.asarray(theta,dtype=float)
    if p.shape != (9,) or not np.isfinite(p).all():
        raise ValueError('M1 parameter vector requires nine finite entries')
    return M1State(10**p[0],10**p[1],10**p[2],10**p[3],
                   SourceRectangle(*p[4:9]))


def make_reference_state(cfg, contract):
    geo=contract.reference().source
    return np.array([np.log10(cfg['K_inside_m_s']),np.log10(cfg['K_outside_m_s']),
        np.log10(cfg['Ss_per_m']),np.log10(cfg['gamma_rel']),geo.a_m,geo.b_m,
        geo.theta_deg,geo.x0_prime_m,geo.y0_prime_m], dtype=float)


def make_model(hydraulic_config, contract, stations, config, *, nx=None, ny=None,
               support=None, splits=None, gauges=None):
    hc=dict(hydraulic_config)
    if nx is not None:hc['nx']=int(nx)
    if ny is not None:hc['ny']=int(ny)
    sup=SourceRectangle(**config['mechanical_support']) if support is None else support
    tile_splits=tuple(config['mechanical_tile_splits']) if splits is None else splits
    return Stage6M1Forward(hc,contract.domain,contract.mechanics,stations,sup,
         splits=tile_splits,quadrature_order=int(config['quadrature_order']),
         gauges_prime_m=gauges)


def flatten_whitened(result, *, strain_sigma_ns, pressure_sigma_pa):
    if result.pressure_gauges_pa is None:
        raise ValueError('Pressure gauges required for joint identifiability')
    return np.concatenate([(result.strain_ns/strain_sigma_ns).ravel(),
                           (result.pressure_gauges_pa/pressure_sigma_pa).ravel()])


def checkpoint(mean_strain_ref, mean_strain_new, p_ref, p_new, times_h):
    mask=np.asarray(times_h)>0
    return {
        'strain_rms_ns':rms_difference(mean_strain_ref[mask],mean_strain_new[mask]),
        'strain_relative_l2':relative_change(mean_strain_ref[mask],mean_strain_new[mask]),
        'pressure_rms_kpa':rms_difference(p_ref[mask],p_new[mask])/1000.,
        'pressure_relative_l2':relative_change(p_ref[mask],p_new[mask]),
    }


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--refine-grid-factor',type=float,default=1.5)
    ap.add_argument('--skip-accuracy',action='store_true',help='Diagnostic only; no GO decision permitted')
    ap.add_argument('--strain-sigma-ns',type=float,default=2.0)
    ap.add_argument('--pressure-sigma-pa',type=float,default=20000.0)
    args=ap.parse_args(argv)
    if args.refine_grid_factor<=1 or not np.isfinite(args.refine_grid_factor):
        ap.error('refine-grid-factor must be greater than one')
    if args.strain_sigma_ns<=0 or args.pressure_sigma_pa<=0:
        ap.error('observation sigmas must be positive')
    root=Path(args.config)
    cfg=json.loads(root.read_text(encoding='utf-8'))
    out=Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Refusing to overwrite: '+str(out))
    contract_path=Path(cfg['stage6a_contract']);hc_path=Path(cfg['hydraulic_config']);st_path=Path(cfg['stations_csv'])
    contract=Stage6Contract.from_json(contract_path)
    hc=json.loads(hc_path.read_text(encoding='utf-8'))
    stations=pd.read_csv(st_path).to_dict(orient='list')
    theta=make_reference_state(cfg,contract)
    well=np.array([contract.domain.well_x_prime_m,contract.domain.well_y_prime_m])
    g=np.asarray(cfg['gauges_global_prime_m'],dtype=float)
    if g.ndim!=2 or g.shape[1]!=2:
        raise ValueError('gauge coordinates invalid')
    near_dist=2*max((contract.domain.xmax_prime_m-contract.domain.xmin_prime_m)/hc['nx'],
                    (contract.domain.ymax_prime_m-contract.domain.ymin_prime_m)/hc['ny'])
    keep=np.linalg.norm(g-well,axis=1)>=near_dist
    if not np.any(keep):
        raise ValueError('No off-well gauges; cannot assess pressure-field numerical accuracy')
    gauges=g[keep]
    t0=time.perf_counter()
    model=make_model(hc,contract,stations,cfg,gauges=gauges)
    base=model.predict(state_from_vector(theta),TIMES)
    model_sigma=(float(args.strain_sigma_ns),float(args.pressure_sigma_pa))
    def whiten(t):
        return flatten_whitened(model.predict(state_from_vector(t),TIMES),
                                strain_sigma_ns=model_sigma[0],pressure_sigma_pa=model_sigma[1])
    J=central_difference(whiten,theta,FD_STEPS)
    information=scaled_information(J,PARAMETER_SCALES)
    # Additional finite-difference check focuses on 5 zone-geometry columns:
    geom_ix=np.arange(4,9)
    Jhalf=central_difference(whiten,theta,FD_STEPS/2,indices=geom_ix)
    step_err=relative_column_changes(J[:,geom_ix],Jhalf)
    results={
        'scope': 'M1 LOCAL geometry identifiability and one observed-output numerical check; not global recovery, DREAM, or COMSOL validation',
        'parameters':list(PARAMETERS), 'reference_theta':theta.tolist(),
        'scale_log_or_physical_units':PARAMETER_SCALES.tolist(),
        'fd_step_log_or_physical_units':FD_STEPS.tolist(),
        'assumed_noise_strain_ns':args.strain_sigma_ns,
        'assumed_noise_pressure_pa':args.pressure_sigma_pa,
        'n_strain_channels':len(base.channels),'n_gauges_used':len(gauges),
        'gauges_used_global_prime_m':gauges.tolist(),
        'gauges_excluded_near_well_global_prime_m':g[~keep].tolist(),
        'n_times':len(TIMES), 'times_h':TIMES.tolist(),
        'local_singular_values':information.singular_values.tolist(),
        'local_rank_threshold_relative':1e-3,
        'local_effective_rank':information.sensitivity_rank,
        'local_scaled_condition_number':information.numerical_condition,
        'scaled_column_norms':information.column_norms.tolist(),
        'scaled_cosines':information.column_cosines.tolist(),
        'weakest_direction_scaled_coordinates':information.weakest_direction.tolist(),
        'conditional_sd_units_of_scale':None if information.conditional_sd_scaled is None else information.conditional_sd_scaled.tolist(),
        'geometry_jacobian_half_step_relative_change':dict(zip([PARAMETERS[i] for i in geom_ix],step_err.tolist())),
        'note': 'Sensitive to chosen time/sensor set, scales, assumed covariance, derivative step, baseline contrast and numerical resolution.',
    }
    # No contrast => geometric information must disappear, even if zone changes.
    no_contrast=theta.copy();no_contrast[1]=no_contrast[0]
    F0=whiten(no_contrast)
    geom_no_contrast=central_difference(whiten,no_contrast,FD_STEPS,indices=geom_ix)
    results['no_contrast_geometry_jacobian_max_abs']=float(np.max(np.abs(geom_no_contrast)))
    results['no_contrast_geometry_response_norm']=float(np.linalg.norm(geom_no_contrast))
    if not np.isfinite(F0).all():
        raise AssertionError('Invalid homogeneous-limit prediction')
    if not args.skip_accuracy:
        # Geometry-independent fixed mechanical loading: same supports in both.
        refined=make_model(hc,contract,stations,cfg,nx=int(round(hc['nx']*args.refine_grid_factor)),
                           ny=int(round(hc['ny']*args.refine_grid_factor)),gauges=gauges)
        field_ref=refined.predict(state_from_vector(theta),TIMES)
        results['grid_refinement']={
            'coarse_grid':[hc['nx'],hc['ny']],
            'refined_grid':[refined.hydraulic.nx,refined.hydraulic.ny],
            **checkpoint(base.strain_ns,field_ref.strain_ns,
                base.pressure_gauges_pa,field_ref.pressure_gauges_pa,TIMES),
        }
        # A modest *fixed* increase, not an attempt to estimate the source area.
        old=cfg['mechanical_support']
        newsup=SourceRectangle(old['a_m']+150.,old['b_m']+150.,old['theta_deg'],
                               old['x0_prime_m'],old['y0_prime_m'])
        # Tile size approximately equal to baseline (about 300 x 312.5m).
        tile_dx=2*old['a_m']/cfg['mechanical_tile_splits'][0]
        tile_dy=2*old['b_m']/cfg['mechanical_tile_splits'][1]
        splits=(round(2*newsup.a_m/tile_dx),round(2*newsup.b_m/tile_dy))
        enlarged=make_model(hc,contract,stations,cfg,support=newsup,splits=splits,gauges=gauges)
        enlarged_res=enlarged.predict(state_from_vector(theta),TIMES)
        results['mechanical_support_refinement']={
            'baseline_half_dimensions_m':[old['a_m'],old['b_m']],
            'expanded_half_dimensions_m':[newsup.a_m,newsup.b_m],
            'baseline_tile_splits':cfg['mechanical_tile_splits'],
            'expanded_tile_splits':list(splits),
            'strain_rms_ns':rms_difference(base.strain_ns,enlarged_res.strain_ns),
            'strain_relative_l2':relative_change(base.strain_ns,enlarged_res.strain_ns),
        }
        results['numerical_budget']={
            'predeclared_strain_rms_ns':0.5*args.strain_sigma_ns,
            'predeclared_pressure_rms_kpa':0.5*args.pressure_sigma_pa/1000.,
            'meaning': 'Engineering accuracy screening target (50% of hypothetical sensor sigma), not physics validation',
        }
        grid=results['grid_refinement']
        support=results['mechanical_support_refinement']
        results['numerical_budget_pass'] = bool(
            grid['strain_rms_ns'] <= .5*args.strain_sigma_ns and
            grid['pressure_rms_kpa'] <= .5*args.pressure_sigma_pa/1000. and
            support['strain_rms_ns']<=.5*args.strain_sigma_ns)
    else:
        results['numerical_budget_pass']=None
    results['derivative_stability_screen_pass']=bool(np.all(step_err < .1))
    results['local_rank_screen_pass']=bool(information.sensitivity_rank == len(PARAMETERS))
    results['pilot_decision']='NOT READY FOR FULL M1 DREAM' if not (
        results['numerical_budget_pass'] and results['derivative_stability_screen_pass']
        and results['local_rank_screen_pass']) else 'ELIGIBLE FOR BOUNDED SYNTHETIC MAP; NOT YET FIELD INFERENCE'
    results['limitations']=[
        'A numerical-rank screen is not proof of unique boundary recovery or of global identifiability.',
        'The sigma assumptions are from prior same-model simulations, not independently estimated M1 field noise.',
        'Cut-cell K uses an approximate arithmetic area mixture, not an exact oblique interface flux law.',
        'A one-step grid comparison is a screen, not formal order-of-accuracy or asymptotic convergence.',
        'Fixed mechanical loading support is a finite approximation and must not be conflated with geological zone.',
        'The original homogeneous half-space strain kernel remains unchanged; no elastic heterogeneity or 3D flow.',
        'Near-well gauges excluded from numerical screening because the point-well pressure is grid dependent.',
        'No independent COMSOL comparison or measured M1 observations are used.',
        'Only a handful of proposed sensor locations are used; this is not a field station network design.',
        'The pass/fail screens depend on declared arbitrary engineering budgets and cannot prove field validity.',
    ]
    results['elapsed_seconds']=time.perf_counter()-t0
    out.mkdir(parents=True,exist_ok=False)
    (out/'m1_go_no_go_report.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'m1_go_no_go_arrays.npz', J_white=J,theta=theta,scales=PARAMETER_SCALES,
                        geometry_fd_half_step=Jhalf, times_h=TIMES,
                        baseline_strain_ns=base.strain_ns,baseline_offwell_pressure_pa=base.pressure_gauges_pa)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axs[0].semilogy(np.arange(1,len(information.singular_values)+1),
                     information.singular_values,'o-')
    axs[0].set(xlabel='Singular value index',ylabel='Singular value',title='Local information: selected sensor layout')
    axs[0].grid(alpha=.25)
    h=axs[1].imshow(information.column_cosines,vmin=-1,vmax=1,cmap='coolwarm')
    axs[1].set(xticks=np.arange(9),yticks=np.arange(9),
        xticklabels=PARAMETERS,yticklabels=PARAMETERS,title='Scaled sensitivity column cosines')
    plt.setp(axs[1].get_xticklabels(),rotation=90)
    fig.colorbar(h,ax=axs[1],fraction=.046)
    fig.savefig(out/'m1_information.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'m1_information.pdf',bbox_inches='tight')
    plt.close(fig)
    print('M1 go/no-go pilot saved:',out)
    print('Local effective rank:',information.sensitivity_rank,'/',len(PARAMETERS))
    print('Geometric derivative step sensitivity:',dict(zip([PARAMETERS[i] for i in geom_ix],np.round(step_err,3))))
    if not args.skip_accuracy:
        print('Grid refinement strain RMS ns:',results['grid_refinement']['strain_rms_ns'])
        print('Support refinement strain RMS ns:',results['mechanical_support_refinement']['strain_rms_ns'])
    print('Decision:',results['pilot_decision'])
    print('Not an inverse parameter recovery or COMSOL validation.')


if __name__ == '__main__':
    main()
