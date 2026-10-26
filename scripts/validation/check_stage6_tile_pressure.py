"""No-fit Stage 6B-2 diagnostic: global FV field -> moving rotated tile pressure averages.

Reads Stage 6B-1 saved numerical fields. Does not evaluate mechanical strain or
infer parameters. The support diagnostic measures absolute *pressure*, not strain.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.model.stage6_coordinates import local_to_global
from avant_model.model.stage6_tile_pressure import RotatedTilePressureOperator


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tile_polygons(op):
    """Global coordinate polygons ordered exactly as B2 tile pressure vectors."""
    s=op.source
    sx=2*s.a_m/op.nx;sy=2*s.b_m/op.ny
    polygons=[]
    for ix in range(op.nx):
        for iy in range(op.ny):
            xlo=-s.a_m+ix*sx; ylo=-s.b_m+iy*sy
            xx=xlo+np.array([0,sx,sx,0]);yy=ylo+np.array([0,0,sy,sy])
            x,y=local_to_global(xx,yy,s)
            polygons.append(np.column_stack((x,y)))
    return polygons


def main():
    parser=argparse.ArgumentParser(description='Stage 6B-2 moving-rotated-tile pressure quadrature validation (no mechanics)')
    parser.add_argument('--contract',required=True)
    parser.add_argument('--fields',required=True,help='NPZ from Stage 6B-1')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--splits-x',type=int,default=6)
    parser.add_argument('--splits-y',type=int,default=6)
    parser.add_argument('--quadrature-order',type=int,default=4)
    parser.add_argument('--reference-order',type=int,default=8)
    args=parser.parse_args()
    contract_path=Path(args.contract).resolve();fields_path=Path(args.fields).resolve()
    contract=Stage6Contract.from_json(contract_path)
    reference=contract.reference()
    if args.reference_order <= args.quadrature_order:
        raise ValueError('reference-order must exceed quadrature-order for refinement comparison')
    with np.load(fields_path,allow_pickle=False) as dataset:
        required={'times_h','x_prime_m','y_prime_m','pressure_change_pa'}
        if not required.issubset(dataset.files):
            raise ValueError(f'B1 NPZ missing fields: {sorted(required-set(dataset.files))}')
        times=np.array(dataset['times_h'],dtype=float)
        x=np.array(dataset['x_prime_m'],dtype=float)
        y=np.array(dataset['y_prime_m'],dtype=float)
        pressure=np.array(dataset['pressure_change_pa'],dtype=float)
    if (times.ndim!=1 or times.size==0 or not np.isfinite(times).all()
            or not np.all(np.diff(times)>0) or pressure.shape!=(len(times),len(y),len(x))):
        raise ValueError('Invalid B1 time or pressure arrays')
    dx=float(x[1]-x[0]);dy=float(y[1]-y[0])
    bounds=contract.domain.extent()
    inferred=(x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2)
    if not np.allclose(inferred,bounds,rtol=0,atol=1e-7):
        raise ValueError('Pressure grid physical domain does not match Stage 6A contract')
    source=reference.source
    op=RotatedTilePressureOperator(x,y,source,args.splits_x,args.splits_y,args.quadrature_order)
    op_ref=RotatedTilePressureOperator(x,y,source,args.splits_x,args.splits_y,args.reference_order)
    tile_means=op.mean_pressure_pa(pressure)
    tile_ref=op_ref.mean_pressure_pa(pressure)
    diff=tile_means-tile_ref
    rms=float(np.sqrt(np.mean(diff**2)))
    peak=float(np.max(np.abs(diff)))
    rel_rms=rms/max(float(np.sqrt(np.mean(tile_ref**2))),np.finfo(float).eps)
    support=op.pressure_support_summary(pressure)
    # The density of the FV grid changes both the global denominator and
    # local bilinear interpolation, so this support ratio is approximate.
    ratios=[None if not np.isfinite(v) else float(v) for v in np.atleast_1d(support.ratio_inside_over_global)]
    out=Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Output directory exists: {out}')
    out.mkdir(parents=True)
    np.savez_compressed(out/'stage6b2_tile_pressure_fields.npz',
        times_h=times,tile_pressure_pa=tile_means,
        high_order_tile_pressure_pa=tile_ref,
        tile_centers_global_prime_m=op.tile_centers_prime_m,
        source_corners_global_prime_m=op.parent_corners_prime_m,
        inside_abs_pressure_integral_pa_m2=support.inside_abs_pressure_pa_m2,
        global_abs_pressure_integral_pa_m2=support.global_abs_pressure_pa_m2,
        pressure_support_fraction_proxy=support.ratio_inside_over_global)
    report={
        'scope':'Stage 6B-2 GLOBAL-FV-to-rotated-tile pressure averaging only; no strain, no inverse model',
        'contract_sha256':sha256(contract_path),
        'fields_sha256':sha256(fields_path),
        'times_h':times.tolist(),
        'grid_shape_yx':[len(y),len(x)],
        'n_tile_x':op.nx,'n_tile_y':op.ny,'n_tiles':op.n_tiles,
        'quadrature_order':op.order,'reference_order':op_ref.order,
        'max_abs_quadrature_difference_pa':peak,
        'rms_quadrature_difference_pa':rms,
        'relative_rms_quadrature_difference':rel_rms,
        'source_area_m2':float(op.n_tiles*op.tile_area_m2),
        'pressure_support_fraction_proxy_by_time':ratios,
        'warning':'The pressure-support proxy is NOT the fraction of induced strain. Pressure outside source may matter substantially.',
        'limitations':[
            'Global FV grid is the Stage 6B-1 one-way homogeneous solution with provisional Dirichlet boundaries.',
            'Tile means are Gauss-Legendre quadrature of bilinear cell-center interpolation, not an exact FV cell-intersection integral.',
            'The reference-order comparison tests quadrature convergence for the SAME interpolated FV field, not mesh convergence.',
            'The inside/global |pressure| integral ratio uses different FV-midpoint and tile-Gauss rules and can exceed one slightly numerically.',
            'At zero pressure the support ratio is null (undefined), never treated as 100%.',
            'Pressure outside the mechanical source is omitted in the current finite-support mechanical interpretation.',
            'No output is an inferred spatial K field or independent COMSOL validation.',
        ]}
    (out/'stage6b2_tile_pressure_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

    # Show the global field and the mechanically sampled tile means side by side.
    selection=int(np.argmax(np.max(np.abs(pressure),axis=(1,2))))
    polygons=tile_polygons(op)
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.3),constrained_layout=True)
    vmax=float(np.max(np.abs(pressure[selection]))/1e3)
    vmin=float(np.min(pressure[selection])/1e3)
    im=axs[0].pcolormesh(x,y,pressure[selection]/1000,shading='nearest',cmap='viridis',vmin=vmin,vmax=vmax)
    pts=np.vstack([op.parent_corners_prime_m,op.parent_corners_prime_m[0]])
    axs[0].plot(pts[:,0],pts[:,1],color='white',linewidth=2.2,label='Mechanical support')
    axs[0].plot(contract.domain.well_x_prime_m,contract.domain.well_y_prime_m,'r*',markersize=11,label='Well')
    axs[0].set_title(f'Global FV pressure at {times[selection]:g} h')
    axs[0].set_aspect('equal')
    axs[0].legend(fontsize=8,loc='upper right')
    fig.colorbar(im,ax=axs[0],label='Pressure change (kPa)',shrink=.8)
    coll=PolyCollection(polygons,array=tile_means[selection]/1000,cmap='viridis',edgecolors='#333333',linewidths=.5)
    coll.set_clim(vmin,vmax)
    axs[1].add_collection(coll)
    axs[1].autoscale_view()
    axs[1].set_aspect('equal')
    axs[1].plot(contract.domain.well_x_prime_m,contract.domain.well_y_prime_m,'r*',markersize=11)
    axs[1].set_title(f'{op.nx} × {op.ny} rotated tile means')
    fig.colorbar(coll,ax=axs[1],label='Tile mean pressure change (kPa)',shrink=.8)
    for ax in axs:
        ax.set_xlabel("Global x' (m)");ax.set_ylabel("Global y' (m)")
    fig.suptitle('Stage 6B-2: pressure mapping only — no mechanical strain',fontsize=13)
    fig.savefig(out/'stage6b2_pressure_tile_mapping.png',dpi=180)
    fig.savefig(out/'stage6b2_pressure_tile_mapping.pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.4),constrained_layout=True)
    plotted=np.array(ratios,dtype=float)
    ax.plot(times,plotted,'o-',label='Mechanical footprint / global |pressure| integral')
    ax.axhline(1,color='gray',linestyle='--',linewidth=1)
    ax.set(xlabel='Time (h)',ylabel='Approximate inside/global |pressure| integral ratio',
           title='Pressure support diagnostic — NOT strain truncation')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.savefig(out/'stage6b2_pressure_support.png',dpi=180)
    fig.savefig(out/'stage6b2_pressure_support.pdf')
    plt.close(fig)
    print('Stage 6B-2 pressure-to-tile diagnostic saved:',out)
    print('Quadrature difference (RMS Pa):',rms,'; relative:',rel_rms)
    print('Pressure support fraction proxy:',dict(zip(times.tolist(),ratios)))
    print('No mechanical strain or parameter inversion is evaluated in Stage 6B-2.')

if __name__=='__main__':
    main()
