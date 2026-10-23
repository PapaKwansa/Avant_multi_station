"""Compare in-memory FV against existing saved AVANT standalone hydraulic fields.

Run from repository root. Writes a numerical report and two plots.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from avant_model.model.confined_fv import ConfinedFVSolver


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--reference',required=True,help='Existing hydraulic_fields.npz from run_injection_fd.py')
    ap.add_argument('--output-dir',required=True)
    a=ap.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    with np.load(a.reference) as ref:
        times=np.asarray(ref['times_h'],float)
        expected=np.asarray(ref['pressure_change_pa'],float)
        x=np.asarray(ref['x_m'],float); y=np.asarray(ref['y_m'],float)
    result=ConfinedFVSolver(cfg).solve(cfg['K_m_s'],cfg['Ss_per_m'],times)
    np.testing.assert_allclose(result.x_m,x,rtol=0,atol=1e-10)
    np.testing.assert_allclose(result.y_m,y,rtol=0,atol=1e-10)
    diff=result.pressure_change_pa-expected
    max_abs=float(np.max(np.abs(diff)))
    rms=float(np.sqrt(np.mean(diff**2)))
    scale=max(1.,float(np.max(np.abs(expected))))
    report=dict(scope='Numerical equivalence against saved standalone FV output',
                max_abs_difference_pa=max_abs,rmse_difference_pa=rms,
                relative_max_difference=max_abs/scale,
                max_mass_balance_residual_m3_s=result.max_mass_balance_residual_m3_s,
                times_h=times.tolist())
    (out/'equivalence_summary.json').write_text(json.dumps(report,indent=2))
    # Strict tolerance accommodates floating point summation/order changes.
    if max_abs/scale > 1e-9:
        raise AssertionError(f'FV regression mismatch: {report}')
    idx=int(np.argmin(abs(times-19.)))
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    extent=[x[0]-(x[1]-x[0])/2,x[-1]+(x[1]-x[0])/2,y[0]-(y[1]-y[0])/2,y[-1]+(y[1]-y[0])/2]
    for ax,p,title in zip(axes,[expected[idx],result.pressure_change_pa[idx],diff[idx]],
                          ['Original FV (Pa)','Reusable FV (Pa)','Difference (Pa)']):
        im=ax.imshow(p,origin='lower',extent=extent,aspect='equal',cmap='viridis')
        ax.set_title(f'{title}, t={times[idx]:g} h');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.savefig(out/'fv_equivalence_19h.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(times,np.max(np.abs(diff),axis=(1,2)),marker='o')
    ax.set(xlabel='Time (h)',ylabel='Maximum absolute difference (Pa)',title='Reusable versus original FV')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'fv_equivalence_history.png',dpi=220);plt.close(fig)
    print(json.dumps(report,indent=2))
    print('Saved:',out)

if __name__=='__main__': main()
