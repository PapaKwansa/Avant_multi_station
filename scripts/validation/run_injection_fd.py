"""Standalone 2-D confined aquifer finite-volume injection benchmark.
Run from repository root: python scripts/validation/run_injection_fd.py --config configs/pressure_superposition/experiments/injection_fd_30gpm.json --output-dir results/pressure_superposition/injection_fd_001
Uses cell-centered finite volumes, backward Euler, zero-head-change outer boundaries.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import splu

GPM_TO_M3S = 0.003785411784 / 60.0

def run(cfg, out):
    nx, ny = int(cfg['nx']), int(cfg['ny'])
    assert nx % 3 == 0 and ny % 3 == 0 and nx % 2 == 1 and ny % 2 == 1
    lx, ly, thick = (float(cfg[k]) for k in ('length_x_m','length_y_m','thickness_m'))
    K, Ss = float(cfg['K_m_s']), float(cfg['Ss_per_m'])
    rho, gravity = float(cfg['rho_kg_m3']), float(cfg['gravity_m_s2'])
    dt_h = float(cfg['dt_hours']); injection_h = float(cfg['injection_hours']); total_h = float(cfg['total_hours'])
    assert all(v>0 for v in (K,Ss,thick,dt_h)) and 0<injection_h<total_h
    assert abs(injection_h/dt_h-round(injection_h/dt_h))<1e-9
    assert abs(total_h/dt_h-round(total_h/dt_h))<1e-9
    dx,dy=lx/nx,ly/ny
    area=dx*dy; cell_volume=area*thick
    storage=Ss*cell_volume
    tx=K*thick*dy/dx; ty=K*thick*dx/dy
    # Cell-centered boundary flux uses half-cell distance to a fixed-head boundary.
    A=lil_matrix((nx*ny,nx*ny),dtype=float)
    def idx(i,j): return j*nx+i
    for j in range(ny):
        for i in range(nx):
            k=idx(i,j)
            for di,dj,t in ((1,0,tx),(-1,0,tx),(0,1,ty),(0,-1,ty)):
                ii,jj=i+di,j+dj
                if 0<=ii<nx and 0<=jj<ny:
                    A[k,k]+=t; A[k,idx(ii,jj)]-=t
                else: A[k,k]+=2*t
    A=A.tocsc()
    solver=splu((A+eye(nx*ny,format='csc')*(storage/(dt_h*3600))).tocsc())
    head=np.zeros(nx*ny)
    source=np.zeros(nx*ny)
    well_idx=idx(nx//2,ny//2)
    q=float(cfg['injection_rate_gpm'])*GPM_TO_M3S
    source[well_idx]=q
    nsteps=round(total_h/dt_h)
    times=np.arange(nsteps+1)*dt_h
    snapshot_hours=cfg['snapshot_hours']
    snapshots={0.0:head.copy()}
    rows=[]; history=[]
    for step in range(1,nsteps+1):
        time_h=step*dt_h
        qstep=q if time_h<=injection_h+1e-10 else 0.0
        src=source if qstep>0 else np.zeros_like(source)
        old=head.copy()
        head=solver.solve((storage/(dt_h*3600))*old+src)
        storage_rate=storage*np.sum(head-old)/(dt_h*3600)
        boundary_rate=float(np.sum(A@head))
        imbalance=qstep-storage_rate-boundary_rate
        rows.append(dict(time_h=time_h,injection_m3_s=qstep,storage_rate_m3_s=storage_rate,boundary_outflow_m3_s=boundary_rate,balance_residual_m3_s=imbalance))
        pressure=head*rho*gravity
        zones=pressure.reshape(ny,nx).reshape(3,ny//3,3,nx//3).mean(axis=(1,3))
        rec=dict(time_h=time_h,well_cell_mpa=pressure[well_idx]/1e6,mean_mpa=np.mean(pressure)/1e6,maximum_mpa=np.max(pressure)/1e6)
        for j in range(3):
            for i in range(3): rec[f'zone_row{j}_col{i}_mpa']=zones[j,i]/1e6
        history.append(rec)
        if any(abs(time_h-t)<1e-9 for t in snapshot_hours): snapshots[time_h]=head.copy()
    balance=pd.DataFrame(rows)
    balance.to_csv(out/'mass_balance.csv',index=False)
    pd.DataFrame(history).to_csv(out/'pressure_history.csv',index=False)
    x=(np.arange(nx)+.5)*dx-lx/2; y=(np.arange(ny)+.5)*dy-ly/2
    np.savez_compressed(out/'hydraulic_fields.npz',x_m=x,y_m=y,times_h=np.array(sorted(snapshots)),head_change_m=np.array([snapshots[t].reshape(ny,nx) for t in sorted(snapshots)]),pressure_change_pa=np.array([snapshots[t].reshape(ny,nx)*rho*gravity for t in sorted(snapshots)]))
    vmax=max(float(np.max(h))*rho*gravity/1e6 for h in snapshots.values())
    for t,h in sorted(snapshots.items()):
        p=h.reshape(ny,nx)*rho*gravity/1e6
        z=p.reshape(3,ny//3,3,nx//3).mean(axis=(1,3))
        fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        im=axs[0].imshow(p,origin='lower',extent=[-lx/2,lx/2,-ly/2,ly/2],vmin=0,vmax=vmax,cmap='viridis',aspect='auto')
        axs[0].scatter([0],[0],marker='+',c='red',s=100,label='Injection cell'); axs[0].legend(loc='upper right')
        for v in (-lx/6,lx/6): axs[0].axvline(v,color='white',alpha=.65,lw=.8)
        for v in (-ly/6,ly/6): axs[0].axhline(v,color='white',alpha=.65,lw=.8)
        axs[0].set_title(f'Fine-grid pressure change, {t:g} h'); axs[0].set_xlabel('Local x (m)'); axs[0].set_ylabel('Local y (m)')
        fig.colorbar(im,ax=axs[0],label='Pressure change (MPa)')
        im2=axs[1].imshow(z,origin='lower',extent=[-lx/2,lx/2,-ly/2,ly/2],vmin=0,vmax=vmax,cmap='viridis',aspect='auto')
        for j in range(3):
            for i in range(3): axs[1].text(-lx/2+(i+.5)*lx/3,-ly/2+(j+.5)*ly/3,f'{z[j,i]:.3f}',ha='center',va='center',color='white',fontsize=9)
        axs[1].set_title('Nine zone-averaged pressures (MPa)'); axs[1].set_xlabel('Local x (m)'); axs[1].set_ylabel('Local y (m)')
        fig.colorbar(im2,ax=axs[1],label='Pressure change (MPa)')
        fig.savefig(out/f'pressure_maps_{t:g}h.png',dpi=200)
        plt.close(fig)
    hist=pd.DataFrame(history)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(hist.time_h,hist.well_cell_mpa,label='Well cell')
    ax.plot(hist.time_h,hist.mean_mpa,label='Domain average')
    ax.plot(hist.time_h,hist.zone_row1_col1_mpa,label='Central zone average')
    ax.axvline(injection_h,color='black',ls='--',label='Injection stops')
    ax.set(xlabel='Time (hours)',ylabel='Pressure change (MPa)',title='Injection and shut-in pressure response')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'pressure_history.png',dpi=200);plt.close(fig)
    injected=q*injection_h*3600
    max_balance=float(np.max(np.abs(balance.balance_residual_m3_s)))
    # Note: these are pressure-change head values, not absolute pressure.
    report=dict(scope='Illustrative confined 2-D hydraulic benchmark, not field-calibrated',injected_volume_m3=injected,well_cell_max_mpa=float(hist.well_cell_mpa.max()),max_mass_balance_residual_m3_s=max_balance,final_time_h=total_h,grid=[nx,ny],note='3x3 array indexing uses bottom row first in the local hydraulic coordinates; verify mapping to analytical basis before inversion')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Saved maps and histories to:',out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    run(json.loads(Path(args.config).read_text()),out)
if __name__=='__main__':main()
