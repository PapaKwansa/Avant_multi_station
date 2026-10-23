"""Compare piecewise-constant and piecewise-linear 3x3 representations of saved FV pressure.
Run from AVANT repository root. This is a PRESSURE representation test, not strain validation.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_regions(p, x, y, nzone=3):
    ny, nx = p.shape
    if nx % nzone or ny % nzone:
        raise ValueError('Grid dimensions must be divisible by number of zones')
    X, Y = np.meshgrid(x, y)
    constant = np.empty_like(p)
    linear = np.empty_like(p)
    rows = []
    for iy in range(nzone):
        for ix in range(nzone):
            sy = slice(iy*ny//nzone, (iy+1)*ny//nzone)
            sx = slice(ix*nx//nzone, (ix+1)*nx//nzone)
            xx, yy, pp = X[sy,sx], Y[sy,sx], p[sy,sx]
            xc, yc = xx.mean(), yy.mean()
            A = np.column_stack([np.ones(pp.size), (xx-xc).ravel(), (yy-yc).ravel()])
            coeff = np.linalg.lstsq(A, pp.ravel(), rcond=None)[0]
            constant[sy,sx] = pp.mean()
            linear[sy,sx] = (A @ coeff).reshape(pp.shape)
            rows.append(dict(zone_row=iy,zone_col=ix,mean_kpa=float(coeff[0]),
                             gradient_x_kpa_m=float(coeff[1]),gradient_y_kpa_m=float(coeff[2]),
                             local_pressure_range_kpa=float(np.ptp(pp))))
    return constant, linear, rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fields',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--times',nargs='+',type=float,default=[5,19,24,48,96])
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with np.load(args.fields) as f:
        times=np.asarray(f['times_h'],float)
        pressure=np.asarray(f['pressure_change_pa'],float)/1000
        x=np.asarray(f['x_m'],float); y=np.asarray(f['y_m'],float)
    if pressure.shape != (len(times),len(y),len(x)):
        raise ValueError('Unexpected hydraulic_fields.npz array shapes')
    if not (np.all(np.diff(x)>0) and np.all(np.diff(y)>0)):
        raise ValueError('Expected increasing cell-center coordinates')
    metrics=[]; coeffs=[]
    for t in args.times:
        indices=np.where(np.isclose(times,t,atol=1e-8))[0]
        if len(indices)!=1:
            print(f'Skipping {t:g} h: no unique saved snapshot');continue
        p=pressure[indices[0]]
        pc,pl,rows=fit_regions(p,x,y)
        for row in rows: coeffs.append(dict(time_h=t,**row))
        rc=float(np.sqrt(np.mean((p-pc)**2)))
        rl=float(np.sqrt(np.mean((p-pl)**2)))
        peak=float(np.max(np.abs(p)))
        metrics.append(dict(time_h=t,peak_abs_pressure_kpa=peak,uniform_rmse_kpa=rc,
                            gradient_rmse_kpa=rl,improvement_pct=100*(1-rl/rc) if rc>0 else np.nan,
                            uniform_max_abs_error_kpa=float(np.max(np.abs(p-pc))),
                            gradient_max_abs_error_kpa=float(np.max(np.abs(p-pl)))))
        extent=[x[0]-(x[1]-x[0])/2,x[-1]+(x[1]-x[0])/2,
                y[0]-(y[1]-y[0])/2,y[-1]+(y[1]-y[0])/2]
        fig,axes=plt.subplots(2,3,figsize=(13,9),constrained_layout=True)
        vmax=max(float(np.max(p)),1e-12)
        errmax=max(float(np.max(np.abs(p-pc))),float(np.max(np.abs(p-pl))),1e-12)
        for ax,data,title in zip(axes[0],[p,pc,pl],['Fine-grid FV pressure','Nine uniform means','Nine mean + gradient']):
            im=ax.imshow(data,origin='lower',extent=extent,aspect='equal',cmap='viridis',vmin=0,vmax=vmax)
            ax.set_title(title)
        for ax,data,title in zip(axes[1],[p-pc,p-pl,pl-pc],['FV - uniform','FV - gradient','Gradient - uniform']):
            ax.imshow(data,origin='lower',extent=extent,aspect='equal',cmap='RdBu_r',vmin=-errmax,vmax=errmax)
            ax.set_title(title)
        for ax in axes.flat:
            for v in np.linspace(extent[0],extent[1],4)[1:-1]:ax.axvline(v,color='white',alpha=.5,lw=.6)
            for v in np.linspace(extent[2],extent[3],4)[1:-1]:ax.axhline(v,color='white',alpha=.5,lw=.6)
            ax.set_xlabel('Local x (m)');ax.set_ylabel('Local y (m)')
        fig.colorbar(axes[0,0].images[0],ax=list(axes[0]),label='Pressure change (kPa)',shrink=.75)
        fig.colorbar(axes[1,0].images[0],ax=list(axes[1]),label='Pressure difference (kPa)',shrink=.75)
        fig.suptitle(f'Pressure representation comparison at {t:g} h')
        fig.savefig(out/f'pressure_gradient_comparison_{t:g}h.png',dpi=200)
        plt.close(fig)
    pd.DataFrame(metrics).to_csv(out/'pressure_representation_metrics.csv',index=False)
    pd.DataFrame(coeffs).to_csv(out/'regional_pressure_gradients.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print('Saved to:',out)

if __name__=='__main__':main()
