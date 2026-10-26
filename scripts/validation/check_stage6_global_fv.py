"""Stage 6B-1 validation: fixed-global homogeneous FV with physical well.

Read the Stage 6A fixed global boundary/well contract; reuse known Stage 4
loading and FV discretization parameters. Geometry and mechanics are NOT fit.
Output: manifest JSON, NPZ and figure in a new directory, never overwrite.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.model.stage6_global_fv import GlobalConfinedFVSolver


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description='Stage 6B-1 fixed-global FV benchmark; no moving geometry yet')
    ap.add_argument('--contract',required=True)
    ap.add_argument('--hydraulic-config',required=True)
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    contract_path=Path(args.contract).resolve()
    flow_path=Path(args.hydraulic_config).resolve()
    contract=Stage6Contract.from_json(contract_path)
    flow=json.loads(flow_path.read_text(encoding='utf-8'))
    solver=GlobalConfinedFVSolver(flow,contract.domain)
    ref=contract.reference()
    ts=np.array([0.,min(solver.tinj/2,solver.tinj),solver.tinj,
                 min(solver.tinj+5.,solver.tend),solver.tend],float)
    ts=np.unique(np.rint(ts/solver.dt_h)*solver.dt_h)
    result=solver.solve(ref.K_m_s,ref.Ss_per_m,ts)
    if result.max_mass_balance_residual_m3_s>1e-11:
        raise RuntimeError('Global FV mass balance residual exceeds 1e-11 m3/s')
    center=solver.well_operator.sample(result.pressure_change_pa)
    out=Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Output directory already exists: {out}')
    out.mkdir(parents=True)
    np.savez_compressed(out/'stage6b1_pressure_fields.npz',times_h=result.times_h,
                        x_prime_m=result.x_m,y_prime_m=result.y_m,
                        head_change_m=result.head_change_m,
                        pressure_change_pa=result.pressure_change_pa,
                        well_pressure_change_pa=center)
    report={
        'scope':'Stage 6B-1 fixed-global homogeneous pressure solver only; no general coupled strain/inversion',
        'source_config_sha256':sha(contract_path),
        'hydraulic_config_sha256':sha(flow_path),
        'global_domain_prime_m':list(contract.domain.extent()),
        'grid_size':[solver.nx,solver.ny],
        'cell_spacing_m':[solver.dx,solver.dy],
        'well_global_prime_m':[solver.domain.well_x_prime_m,solver.domain.well_y_prime_m],
        'well_effective_location_m':[solver.well_discretization.effective_x_prime_m,solver.well_discretization.effective_y_prime_m],
        'well_cell_indices':list(solver.well_discretization.cell_indices_flat),
        'well_weights':list(solver.well_discretization.weights),
        'reference_K_m_s':ref.K_m_s,'reference_Ss_per_m':ref.Ss_per_m,
        'output_times_h':ts.tolist(),
        'max_mass_balance_residual_m3_s':result.max_mass_balance_residual_m3_s,
        'injected_volume_m3':result.injected_volume_m3,
        'well_pressure_change_kpa':(center/1000).tolist(),
        'warnings':[
            'Global Dirichlet boundary locations are PROVISIONAL and must pass sensitivity testing.',
            'Source is a bilinearly weighted approximation of a point well, not a radius-resolved well.',
            'Hydraulic geometry stays fixed during source-geometry proposals.',
            'Pressure support outside the mechanical source is not yet evaluated.',
            'No analytical cuboid mechanics or inverse model is called by this benchmark.',
        ],
    }
    (out/'stage6b1_global_fv_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8.4,6.2),constrained_layout=True)
    p=result.pressure_change_pa[np.where(np.isclose(result.times_h,solver.tinj))[0][0]]/1000.
    mesh=ax.pcolormesh(solver.x_m,solver.y_m,p,shading='nearest',cmap='viridis')
    ax.plot(solver.domain.well_x_prime_m,solver.domain.well_y_prime_m,'r*',markersize=13,label='Physical well')
    ax.set(xlabel="Global x' (m)",ylabel="Global y' (m)",
           title=f'Stage 6B-1: fixed-global FV pressure at {solver.tinj:g} h (kPa)')
    ax.set_aspect('equal',adjustable='box')
    ax.legend(loc='upper right')
    fig.colorbar(mesh,ax=ax,label='Pressure change (kPa)')
    fig.savefig(out/'stage6b1_global_pressure_map.png',dpi=180)
    fig.savefig(out/'stage6b1_global_pressure_map.pdf')
    plt.close(fig)
    print('Stage 6B-1 fixed-global hydraulic benchmark complete:',out)
    print('Well deposition weights:',report['well_weights'])
    print('Mass-balance residual (m3/s):',result.max_mass_balance_residual_m3_s)
    print('This does not evaluate mechanical strain or parameter inversion.')

if __name__=='__main__':
    main()
