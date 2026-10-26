#!/usr/bin/env python3
"""One-run Stage 6B-3 forward assembly and finite-pressure-support sensitivity.

This uses the saved Larry station coordinates as receiver locations, NOT the
observed COMSOL strain values. All strain predictions are from the unchanged
analytical kernel driven by the Stage 6 global FV field. No parameter fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.model.stage6_coupled_forward import Stage6CoupledForward
from avant_model.model.stage6_coordinates import source_corners_global


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_or_none(value):
    v=float(value)
    return v if np.isfinite(v) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--hydraulic-config', required=True)
    parser.add_argument('--stations-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--splits-x', type=int, default=6)
    parser.add_argument('--splits-y', type=int, default=6)
    parser.add_argument('--quadrature-order', type=int, default=4)
    parser.add_argument('--times-h', type=float, nargs='+', default=[0, 9.5, 19, 24, 96])
    parser.add_argument('--support-factors', type=float, nargs='+', default=[1, 1.5, 2])
    args=parser.parse_args(argv)
    out=Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Output directory already exists: {out}')
    paths=[Path(args.contract),Path(args.hydraulic_config),Path(args.stations_csv)]
    contract=Stage6Contract.from_json(paths[0])
    config=json.loads(paths[1].read_text(encoding='utf-8'))
    stations=pd.read_csv(paths[2])
    model=Stage6CoupledForward(contract,config,stations,
                               splits=(args.splits_x,args.splits_y),
                               quadrature_order=args.quadrature_order)
    state=contract.reference()
    forward=model.predict(state,args.times_h)
    comparison=model.compare_support(state,args.times_h,tuple(args.support_factors))
    if not np.array_equal(forward.strain_ns, comparison.strain_ns[0]):
        raise RuntimeError('Baseline predict() must equal factor=1 support comparison')
    if not np.isfinite(forward.strain_ns).all():
        raise RuntimeError('Non-finite strain predicted')
    if np.max(np.abs(forward.strain_ns[forward.times_h==0])) > 1e-12:
        raise RuntimeError('Zero initial pressure must generate zero strain')
    report={
        'scope': 'Stage 6B-3 same-model one-way forward assembly and finite-support sensitivity; not inverse or independent COMSOL validation',
        'preserved_mechanics': 'pressure_superposition.build_pressure_kernel -> unchanged multi_station_strain/rotation',
        'full_global_fv_same_for_all_supports': True,
        'pressure_units': 'Pa', 'strain_units':'nanostrain',
        'times_h':list(map(float, forward.times_h)),
        'reference_state':state.as_dict(),
        'n_stations':len(stations), 'n_channels':len(forward.channels),
        'channels':list(forward.channels),
        'base_splits':list(forward.tile_splits),
        'quadrature_order':args.quadrature_order,
        'support_factors':list(comparison.factors),
        'support_splits':[list(a) for a in comparison.splits],
        'relative_strain_l2_difference_from_baseline':[
            finite_or_none(v) for v in comparison.relative_l2_difference_from_base],
        'relative_strain_difference_by_time':[[finite_or_none(v) for v in row]
                                             for row in comparison.relative_l2_difference_by_time],
        'max_mass_balance_residual_m3_s':forward.max_mass_balance_residual_m3_s,
        'max_abs_strain_ns_baseline':float(np.max(np.abs(forward.strain_ns))),
        'hydraulic_domain_note':'Provisional fixed-global zero-pressure boundaries; not COMSOL matched',
        'interpretation':('Pressure-support changes reveal sensitivity to omitted pressure loading, but expanded support is not independent truth. '
                          'A large change means fitted mechanical source geometry may be confounded with support truncation.'),
        'sha256':{label:sha256(path) for label,path in zip(('contract','hydraulic_config','stations_csv'),paths)},
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/'stage6b3_coupled_forward_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'stage6b3_forward_results.npz',times_h=forward.times_h,
                        strain_ns_base=forward.strain_ns,
                        pressure_tiles_pa=forward.tile_pressure_pa,
                        pressure_fv_pa=forward.fv_pressure_change_pa,
                        support_factors=np.array(comparison.factors),
                        strain_ns_support=np.stack(comparison.strain_ns,axis=0),
                        channels=np.array(forward.channels),
                        support_relative_l2=comparison.relative_l2_difference_by_time)
    # Figure 1: depth-independent FV pressure and mechanical source outline.
    tidx=int(np.argmin(np.abs(forward.times_h-19)))
    p=forward.fv_pressure_change_pa[tidx]/1000
    ext=[model.hydraulic.domain.xmin_prime_m,model.hydraulic.domain.xmax_prime_m,
         model.hydraulic.domain.ymin_prime_m,model.hydraulic.domain.ymax_prime_m]
    fig,ax=plt.subplots(figsize=(7,5.5),constrained_layout=True)
    im=ax.imshow(p,origin='lower',extent=ext,cmap='viridis',aspect='equal')
    corners=source_corners_global(state.source)
    closed=np.concatenate((corners,corners[:1]),axis=0)
    ax.plot(closed[:,0],closed[:,1],color='white',lw=2.0,label='Baseline finite mechanical support')
    ax.scatter([contract.domain.well_x_prime_m],[contract.domain.well_y_prime_m],
               color='red',marker='*',s=95,label='Fixed well')
    fig.colorbar(im,ax=ax,label='FV pressure change (kPa)')
    ax.set(xlabel="Global x' (m)",ylabel="Global y' (m)",
           title=f'Pressure at {forward.times_h[tidx]:g} h; mechanical support')
    ax.legend(loc='upper right',fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(out/f'stage6b3_pressure_and_source.{suffix}',dpi=230)
    plt.close(fig)
    # Figure 2: baseline and expanded-support predictions in representative channels.
    fig,axs=plt.subplots(2,2,figsize=(9,6.5),constrained_layout=True)
    indices=[i for i,v in enumerate(forward.channels) if v.endswith(('_S01','_S04'))][:4]
    if len(indices)<4:indices=list(range(min(4,len(forward.channels))))
    for ax,idx in zip(axs.ravel(),indices):
        for fact,curves in zip(comparison.factors,comparison.strain_ns):
            ax.plot(forward.times_h,curves[:,idx],marker='o',ms=3,label=f'{fact:g}x support')
        ax.set(title=forward.channels[idx],xlabel='Time (h)',ylabel='Strain (ns)')
        ax.grid(alpha=.25)
    axs.ravel()[0].legend(fontsize=8)
    fig.suptitle('Sensitivity to pressure-loading support; same FV field')
    for suffix in ('png','pdf'):fig.savefig(out/f'stage6b3_support_strain_comparison.{suffix}',dpi=230)
    plt.close(fig)
    print('Stage 6B-3 forward assembly saved:',out)
    print('Full suite must still be run in the AVANT repository.')
    print('Support relative strain changes:',report['relative_strain_l2_difference_from_baseline'])
    print('These are NOT independently validated strain errors or parameter-inversion results.')
    return out


if __name__ == '__main__':
    main()
