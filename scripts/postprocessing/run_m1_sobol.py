#!/usr/bin/env python3
"""Optional independent-input global Sobol sensitivity for M1.

This is a NEW expensive forward experiment, NOT a posterior postprocessing
operation and NOT based on correlated DREAM samples. A & B are independent
scrambled Sobol designs in the chosen nine-dimensional illustrative-prior or
bounded-uniform input measure; AB_i replaces column i of A with B.

Runtime: (2 + 9) * base_n M1 forward solves. Start at N=32 or N=64 for
an exploratory pilot. N=256/512 plus interval and doubling checks are more
appropriate for a manuscript, but can still be insufficient.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.postprocessing.m1_publication_core import (
    PARAM_LABELS, sobol_jansen_saltelli, sobol_bootstrap,
    transformed_independent_priors,
)
from scripts.postprocessing.m1_publication_core import verify_bayesian_manifest
from scripts.postprocessing.postprocess_m1_publication import style, save
from avant_model.inversion.m1_bayesian import (
    M1GammaCachedForward, build_forward, load_saved_m1, read_prior_config,
)
from avant_model.inversion.m1_map_recovery import PARAMETER_NAMES

METRICS=('eXX_S07_at_19h_ns','heldout_pressure_at_19h_kPa',
         'training_network_strain_RMS_at_19h_ns',
         'training_pressure_gauge_mean_at_19h_kPa')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--base-n',type=int,default=32,help='power of two; N*(9+2) forward evaluations')
    p.add_argument('--measure',choices=['illustrative-priors','uniform-bounds'],
                   default='illustrative-priors')
    p.add_argument('--bootstrap',type=int,default=300)
    p.add_argument('--seed',type=int,default=311)
    p.add_argument('--dpi',type=int,default=300,choices=[300,600])
    p.add_argument('--dry-run',action='store_true',help='Validate manifest/design only; no forward solves')
    return p


def target_outputs(theta,predict,saved,at_19):
    strain,pressure=predict(theta)
    if strain.shape!=(len(saved.times_h),len(saved.channel_names)) or pressure.shape!=(len(saved.times_h),3):
        raise ValueError('Unexpected M1 response dimensions')
    e07=strain[at_19,saved.channel_names.index('eXX_S07')]
    train_mask=saved.layout.channel_train_mask
    return np.array([e07,pressure[at_19,2]/1000.,
                     np.sqrt(np.mean(strain[at_19,train_mask]**2)),
                     np.mean(pressure[at_19,:2])/1000.],float)


def evaluate_design(uu,priors,measure,predict,saved,at_19,base_n):
    d=len(PARAMETER_NAMES)
    aa=transformed_independent_priors(uu[:,:d],priors,PARAMETER_NAMES,measure=measure)
    bb=transformed_independent_priors(uu[:,d:],priors,PARAMETER_NAMES,measure=measure)
    ya=np.empty((base_n,len(METRICS)));yb=np.empty_like(ya)
    yab=np.empty((base_n,d,len(METRICS)))
    total=(2+d)*base_n;count=0
    for k in range(base_n):
        ya[k]=target_outputs(aa[k],predict,saved,at_19)
        yb[k]=target_outputs(bb[k],predict,saved,at_19)
        count+=2
        for j in range(d):
            mixed=aa[k].copy();mixed[j]=bb[k,j]
            yab[k,j]=target_outputs(mixed,predict,saved,at_19)
            count+=1
        if ((k+1)%max(1,base_n//8))==0:
            print(f'M1 independent Sobol design: {count}/{total} forward solves',flush=True)
    return ya,yb,yab


def save_indices_csv(path,s1,st,ci1,cit):
    with path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh)
        w.writerow(('response','parameter','first_order_raw','first_95_low','first_95_high',
                    'total_order_raw','total_95_low','total_95_high'))
        for metric in range(len(METRICS)):
            for j,name in enumerate(PARAMETER_NAMES):
                w.writerow((METRICS[metric],name,float(s1[j,metric]),
                            float(ci1[0,j,metric]),float(ci1[1,j,metric]),
                            float(st[j,metric]),float(cit[0,j,metric]),float(cit[1,j,metric])))


def plot_indices(s1,st,ci1,cit,n,measure,unstable,out,dpi):
    fig,axes=plt.subplots(2,2,figsize=(15,10.3),layout='constrained')
    for k,ax in enumerate(axes.flat):
        xx=np.arange(9);bw=.39
        aa=s1[:,k];bb=st[:,k]
        def error_bounds(vals,ci):
            return np.maximum(vals-ci[0,:,k],0),np.maximum(ci[1,:,k]-vals,0)
        ax.bar(xx-bw/2,aa,bw,color='#447EA8',label='First-order S1')
        ax.bar(xx+bw/2,bb,bw,color='#C88640',label='Total-order ST')
        ax.errorbar(xx-bw/2,aa,yerr=error_bounds(aa,ci1),fmt='none',ecolor='#172F47',lw=.8,capsize=2)
        ax.errorbar(xx+bw/2,bb,yerr=error_bounds(bb,cit),fmt='none',ecolor='#66431E',lw=.8,capsize=2)
        ax.axhline(0,color='black',lw=.6);ax.axhline(1,color='#888888',ls=':',lw=.7)
        ax.set_xticks(xx,PARAM_LABELS,rotation=45,ha='right')
        ax.set(ylabel='Variance-based Sobol index (raw estimate)',title=METRICS[k].replace('_',' '))
        ax.grid(axis='y')
    h,l=axes.flat[0].get_legend_handles_labels()
    fig.legend(h,l,ncol=2,loc='outside lower center',frameon=False)
    warning='EXPLORATORY / NUMERICALLY UNSTABLE' if unstable else 'SAMPLE-SIZE SCREEN PASSED (NOT FIELD VALIDATION)'
    fig.suptitle(f'M1 independent-input global sensitivity | N={n}; {measure} | {warning}',fontsize=13)
    save(fig,out,'m1_sobol_first_total_indices',dpi)


def plot_n_comparison(s1half,sthalf,s1,st,base_n,out,dpi):
    fig,axes=plt.subplots(2,2,figsize=(15,9.5),layout='constrained')
    for k,ax in enumerate(axes.flat):
        xx=np.arange(9)
        ax.plot(xx,s1half[:,k],marker='o',lw=1,markersize=3,label=f'S1, N={base_n//2}')
        ax.plot(xx,s1[:,k],marker='s',lw=1,markersize=3,label=f'S1, N={base_n}')
        ax.plot(xx,sthalf[:,k],marker='^',lw=1,markersize=3,label=f'ST, N={base_n//2}')
        ax.plot(xx,st[:,k],marker='d',lw=1,markersize=3,label=f'ST, N={base_n}')
        ax.set_xticks(xx,PARAM_LABELS,rotation=45,ha='right')
        ax.set(title=METRICS[k].replace('_',' '),ylabel='Raw index estimate')
        ax.grid()
    h,l=axes.flat[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Sobol pilot sample-size doubling — not a formal convergence guarantee',fontsize=13)
    save(fig,out,'m1_sobol_n_doubling',dpi)


def main(argv=None):
    a=parser().parse_args(argv)
    if a.base_n<4 or a.base_n&(a.base_n-1) or a.bootstrap<10:
        raise ValueError('base-n must be a power of two >=4; bootstrap >=10')
    run=Path(a.run_dir).resolve();out=Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Sobol output already exists; select a new directory: '+str(out))
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_bayesian_manifest(manifest)
    priors=read_prior_config(manifest['prior_config'])
    saved=load_saved_m1(manifest['map_dir'],manifest['config'])
    at=np.where(np.isclose(saved.times_h,19.,atol=1e-10,rtol=0))[0]
    if len(at)!=1:raise ValueError('Exactly one 19-hour observation time required')
    count=a.base_n*(len(PARAMETER_NAMES)+2)
    print(f'Sobol independent-input design, N={a.base_n}, {count} M1 forward solves; measure={a.measure}')
    if a.dry_run:
        print('Manifest and input distributions verified. No flow equations solved.')
        return
    start=time.perf_counter()
    model=build_forward(manifest['config'],saved)
    predictor=M1GammaCachedForward(model,saved.times_h)
    unit=qmc.Sobol(d=2*len(PARAMETER_NAMES),scramble=True,seed=a.seed).random_base2(int(math.log2(a.base_n)))
    ya,yb,yab=evaluate_design(unit,priors,a.measure,predictor,saved,int(at[0]),a.base_n)
    s1,st=sobol_jansen_saltelli(ya,yb,yab)
    ci1,cit=sobol_bootstrap(ya,yb,yab,draws=a.bootstrap,seed=a.seed+1)
    sh,th=sobol_jansen_saltelli(ya[:a.base_n//2],yb[:a.base_n//2],yab[:a.base_n//2])
    change=float(max(np.max(np.abs(sh-s1)),np.max(np.abs(th-st))))
    halfwidth=float(max(np.max(ci1[1]-ci1[0]),np.max(cit[1]-cit[0]))/2.)
    unstable=(a.base_n<256 or change>.075 or halfwidth>.12)
    out.mkdir(parents=True,exist_ok=True)
    style()
    save_indices_csv(out/'m1_sobol_indices.csv',s1,st,ci1,cit)
    plot_indices(s1,st,ci1,cit,a.base_n,a.measure,unstable,out,a.dpi)
    plot_n_comparison(sh,th,s1,st,a.base_n,out,a.dpi)
    report={'scope':'M1 independent-input forward Sobol experiment, NOT posterior sensitivity',
            'inputs':list(PARAMETER_NAMES),'outputs':list(METRICS),
            'measure':a.measure,'input_distributions_provenance':'Illustrative priors or their support bounds; not field measured',
            'base_sample_n':a.base_n,'forward_solves':count,
            'method':'scrambled Sobol base A/B matrices; Saltelli first order; Jansen total order; bootstrap paired rows',
            'bootstrap_replicates':a.bootstrap,'max_abs_index_change_half_to_full_n':change,
            'max_bootstrap_95_halfwidth':halfwidth,
            'exploratory_or_unstable_screen':unstable,
            'first_order':s1.tolist(),'total_order':st.tolist(),
            'first_95_CI':ci1.tolist(),'total_95_CI':cit.tolist(),
            'elapsed_seconds':time.perf_counter()-start,
            'limitations':[
                'Sobol measures variance under an EXPLICIT INDEPENDENT illustrative input measure, not the correlated posterior',
                'Negative S1 or >1 indices can occur from Monte Carlo estimation error and are not clipped',
                'Finite-N bootstrap intervals may be optimistic with quasi-random input designs',
                'Sample-size doubling and bootstrap uncertainty screen are necessary but not proof of convergence',
                'Output ranking is specific to 19-hour sensor/functional, geometry and assumed input support',
                'Do not run large global sensitivity at the same time as memory-intensive PyDREAM',
                'Same-model synthetic benchmark; no independent geologic field measurements',
            ]}
    (out/'m1_sobol_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved independent-input Sobol report and figures:',out)
    if unstable:print('EXPLORATORY OR NUMERICALLY UNSTABLE: do not publish index ranking as converged')


if __name__=='__main__':main()
