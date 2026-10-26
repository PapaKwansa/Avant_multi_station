#!/usr/bin/env python3
"""ONE M1 forward pilot: hydraulic K contrast, independent full loading support.

Use from AVANT repository root; does not fit parameters or compare to COMSOL.
The output directory must not exist, to prevent accidental overwrites.
"""
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
from avant_model.model.stage6_coordinates import SourceRectangle, source_corners_global
from avant_model.model.stage6_m1_forward import Stage6M1Forward, M1State


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args(argv)
    path = Path(args.config)
    conf = json.loads(path.read_text(encoding='utf-8'))
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing output directory: '+str(out))
    contract_path = Path(conf['stage6a_contract'])
    hc_path = Path(conf['hydraulic_config'])
    st_path = Path(conf['stations_csv'])
    contract = Stage6Contract.from_json(contract_path)
    hc = json.loads(hc_path.read_text(encoding='utf-8'))
    stations = pd.read_csv(st_path).to_dict(orient='list')
    support = SourceRectangle(**conf['mechanical_support'])
    base = contract.reference()
    geology = base.source
    model = Stage6M1Forward(hc, contract.domain, contract.mechanics, stations,
                           support, splits=tuple(conf['mechanical_tile_splits']),
                           quadrature_order=int(conf['quadrature_order']),
                           gauges_prime_m=conf['gauges_global_prime_m'])
    times = np.asarray(conf['output_times_h'], dtype=float)
    m1 = M1State(conf['K_inside_m_s'], conf['K_outside_m_s'], conf['Ss_per_m'],
                 conf['gamma_rel'], geology)
    homog = M1State(conf['K_inside_m_s'], conf['K_inside_m_s'], conf['Ss_per_m'],
                    conf['gamma_rel'], geology)
    res = model.predict(m1,times)
    ref = model.predict(homog,times)
    if res.max_mass_balance_residual_m3_s > 1e-10:
        raise AssertionError('FV mass balance exceeded pilot tolerance')
    if not (np.all(np.isfinite(res.strain_ns)) and
            np.all(np.isfinite(res.fv_pressure_pa))):
        raise AssertionError('Nonfinite M1 output')
    final = int(np.argmin(abs(times-19.)))
    dx,dy=model.hydraulic.dx,model.hydraulic.dy
    fraction_integral = float(np.sum(res.zone_fraction)*dx*dy)
    geometric_area=4*geology.a_m*geology.b_m
    if abs(fraction_integral-geometric_area)/geometric_area>1e-11:
        raise AssertionError('Geological polygon area mapping mismatch')
    # M1 geometry must change only hydraulics, not the fixed mechanical matrix.
    report = {
        'scope': 'M1 same-model, two-zone hydraulic forward pilot; NO inversion or independent COMSOL validation',
        'geological_zone': {'a_m':geology.a_m,'b_m':geology.b_m,
                            'theta_deg':geology.theta_deg,'center_x_prime_m':geology.x0_prime_m,
                            'center_y_prime_m':geology.y0_prime_m},
        'K_inside_m_s':m1.K_inside_m_s,'K_outside_m_s':m1.K_outside_m_s,
        'Ss_per_m':m1.Ss_per_m,'gamma_rel':m1.gamma_rel,
        'fixed_mechanical_loading_support':conf['mechanical_support'],
        'geological_area_m2':geometric_area,
        'cell_fraction_area_m2':fraction_integral,
        'time_h':times.tolist(),
        'max_mass_balance_residual_m3_s':res.max_mass_balance_residual_m3_s,
        'n_strain_channels':len(res.channels),'n_pressure_gauges':res.pressure_gauges_pa.shape[1],
        'm1_vs_homogeneous_strain_rms_ns':float(np.sqrt(np.mean((res.strain_ns-ref.strain_ns)**2))),
        'm1_vs_homogeneous_gauge_pressure_rms_kpa':float(np.sqrt(np.mean((res.pressure_gauges_pa-ref.pressure_gauges_pa)**2))/1000),
        'input_sha256':{str(p):checksum(p) for p in (path,contract_path,hc_path,st_path)},
        'limitations':[
            'All mechanical elastic properties are homogeneous across the geological interface.',
            'One-way 2D confined pressure diffusion with one effective specific-storage coefficient.',
            'Cell area fractions are geometrically precise, but scalar mixed-cell K and face harmonic conductance are approximate at oblique contacts.',
            'The mechanical support is finite and fixed; its extent, tile size and grid boundaries still require observable-level convergence.',
            'The reference hydraulic boundaries are provisional zero-head-change Dirichlet conditions.',
            'No barrier resistance, slip, fracture compliance, fault mechanics, vertical flow, or elastic-property contrast.',
            'The hydraulic boundary geometry is not identifiable if K_inside == K_outside.',
            'This is a forward pilot, not a completed multi-parameter inversion or independent physical validation.'
        ]
    }
    out.mkdir(parents=True)
    (out/'m1_forward_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'m1_forward_arrays.npz', times_h=times,
                        channels=np.asarray(res.channels),
                        m1_pressure_pa=res.fv_pressure_pa,
                        homogeneous_pressure_pa=ref.fv_pressure_pa,
                        m1_strain_ns=res.strain_ns,
                        homogeneous_strain_ns=ref.strain_ns,
                        m1_gauge_pressure_pa=res.pressure_gauges_pa,
                        homogeneous_gauge_pressure_pa=ref.pressure_gauges_pa,
                        K_field_m_s=res.conductivity_field_m_s,
                        zone_cell_fraction=res.zone_fraction,
                        x_prime_m=model.hydraulic.x_m,y_prime_m=model.hydraulic.y_m)
    extent = [model.hydraulic.x_m[0]-dx/2,model.hydraulic.x_m[-1]+dx/2,
              model.hydraulic.y_m[0]-dy/2,model.hydraulic.y_m[-1]+dy/2]
    corners = source_corners_global(geology)
    corners=np.vstack((corners,corners[0]))
    fig,axs=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    im=axs[0].imshow(np.log10(res.conductivity_field_m_s),extent=extent,origin='lower',aspect='equal',cmap='viridis')
    axs[0].plot(corners[:,0],corners[:,1],color='white',linewidth=1.5)
    axs[0].set_title('Two-zone hydraulic conductivity (log10 m/s)')
    fig.colorbar(im,ax=axs[0],fraction=.046)
    im=axs[1].imshow(res.fv_pressure_pa[final]/1e3,extent=extent,origin='lower',aspect='equal',cmap='viridis')
    axs[1].plot(corners[:,0],corners[:,1],color='white',linewidth=1.5)
    axs[1].set_title(f'M1 pressure at {times[final]:g} h (kPa)')
    fig.colorbar(im,ax=axs[1],fraction=.046)
    for ax in axs:ax.set(xlabel="Global x' (m)",ylabel="Global y' (m)")
    fig.savefig(out/'m1_k_and_pressure.png',dpi=220)
    fig.savefig(out/'m1_k_and_pressure.pdf')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i, idx in enumerate((0,min(3,len(res.channels)-1))):
        axs[i].plot(times,res.strain_ns[:,idx],'-o',label='M1 conductivity contrast')
        axs[i].plot(times,ref.strain_ns[:,idx],'--o',label='Homogeneous K')
        axs[i].set(title=res.channels[idx],xlabel='Time (h)',ylabel='Strain (nanostrain)')
        axs[i].grid(alpha=.25)
    axs[0].legend(fontsize=8)
    fig.savefig(out/'m1_strain_comparison.png',dpi=220)
    fig.savefig(out/'m1_strain_comparison.pdf')
    plt.close(fig)
    print('M1 forward pilot complete:',out)
    print('max mass balance residual m3/s:',res.max_mass_balance_residual_m3_s)
    print('M1 vs homogeneous strain RMSE ns:',report['m1_vs_homogeneous_strain_rms_ns'])
    print('This is NOT multi-parameter inversion or independent COMSOL validation.')


if __name__=='__main__':
    main()
