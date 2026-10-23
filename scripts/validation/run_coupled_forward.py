"""Run a fixed-geometry coupled forward benchmark and save strain histories."""
import argparse
import json
from os import times
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from avant_model.model.pressure_superposition import Cuboid, ElasticProperties
from avant_model.model.coupled_hydromechanics import CoupledHydroMechanicalForward




def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hydraulic-config', required=True)
    p.add_argument('--mechanical-config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--fine-n', type=int, default=18)
    p.add_argument('--K', type=float, default=1e-6)
    p.add_argument('--Ss', type=float, default=1e-6)
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    hc = json.loads(Path(a.hydraulic_config).read_text())
    mc = json.loads(Path(a.mechanical_config).read_text())
    stations = pd.read_csv(mc['station_file'])
    model = CoupledHydroMechanicalForward(hc, Cuboid(**mc['geometry']),
               ElasticProperties(**mc['elastic']), stations,
               mechanical_splits=(a.fine_n,a.fine_n,1),components=tuple(mc['components']))
    times = np.arange(
        0.0,
        hc["total_hours"] + 0.5 * hc["dt_hours"],
        hc["dt_hours"],
    )
    result = model.predict(a.K,a.Ss,times)
    frame = pd.DataFrame(result.strain_ns, columns=result.channels)
    frame.insert(0,'time_h',result.times_h)
    frame.to_csv(out/'coupled_strain_history.csv',index=False)
    np.savez_compressed(out/'coupled_forward.npz',times_h=result.times_h,
        strain_ns=result.strain_ns, pressure_coefficients=result.pressure_coefficients,
        channels=np.asarray(result.channels))
    fig, axs = plt.subplots(2,2,figsize=(12,8),sharex=True,constrained_layout=True)
    for ax, comp in zip(axs.flat,mc['components']):
        for name in stations.station.astype(str):
            channel = f'{comp}_{name}'
            if channel in frame:
                ax.plot(frame.time_h,frame[channel],marker='.',label=name)
        ax.axvline(hc['injection_hours'],ls='--',color='black',alpha=.5)
        ax.set_title(comp)
        ax.set_ylabel('Strain (nanostrain)')
        ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2,fontsize=8)
    axs[1,0].set_xlabel('Time (hours)')
    axs[1,1].set_xlabel('Time (hours)')
    fig.savefig(out/'coupled_strain_histories.png',dpi=220)
    plt.close(fig)
    summary = dict(scope='Synthetic coupled forward calculation; not field validation',
      K_m_s=a.K,Ss_per_m=a.Ss,fine_n=a.fine_n,H_shape=list(model.H.shape),
      max_mass_balance_residual_m3_s=result.max_mass_balance_residual_m3_s,
      peak_abs_strain_ns=float(np.max(np.abs(result.strain_ns))))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
