#!/usr/bin/env python3
"""Independent, add-only AVANT M1 publication-figure and posterior-predictive suite.

Consumes an immutable saved nine-parameter M1 DREAM run; does not change its
sampler, priors, forward model, saved chains or original postprocessor.
A run that fails the Rhat/ESS gate is labeled EXPLORATORY on EVERY plot.
Sobol indices require a separate independent-design forward experiment:
see run_m1_sobol.py, not these correlated MCMC samples.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.postprocessing.m1_publication_core import (
    PARAM_LABELS, COMPONENT_LABELS, membership_probability,
    posterior_predictive_bands, summarize,
)
from scripts.postprocessing.m1_publication_core import verify_bayesian_manifest
from avant_model.inversion.m1_bayesian import (
    M1GammaCachedForward, build_forward, load_saved_m1, read_prior_config, _pdf_for,
)
from avant_model.inversion.m1_map_recovery import PARAMETER_NAMES
from avant_model.inversion.m1_posterior_diagnostics import diagnose
from avant_model.model.stage6_coordinates import SourceRectangle, local_to_global


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--posterior-draws', type=int, default=60,
                   help='Number of costly M1 forward solves for predictive plots (>=8)')
    p.add_argument('--boundary-draws', type=int, default=0,
                   help='Number of cheap geometry-only draws; 0 uses ALL retained draws')
    p.add_argument('--noise-replicates', type=int, default=4)
    p.add_argument('--seed', type=int, default=1041)
    p.add_argument('--dpi', type=int, default=300, help='PNG resolution (300 or 600)')
    p.add_argument('--rhat-max', type=float, default=1.01)
    p.add_argument('--min-ess', type=int, default=400)
    p.add_argument('--local-report', default=None,
                   help='Optional M1 identifiability report for LOCAL Jacobian plot only')
    return p.parse_args(argv)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.labelsize':11,'axes.titlesize':11.5,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'xtick.labelsize':9,'ytick.labelsize':9,
                         'figure.facecolor':'white','savefig.facecolor':'white',
                         'pdf.fonttype':42, 'ps.fonttype':42,
                         'axes.linewidth':.8,'grid.alpha':.17})


def save(fig, out, stem, dpi):
    fig.savefig(out/(stem+'.pdf'), bbox_inches='tight', pad_inches=.08)
    fig.savefig(out/(stem+'.png'), dpi=dpi, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def title_with_status(title, converged):
    return title if converged else title + '  |  EXPLORATORY: CHAINS NOT CONVERGED'


def prior_pdf(prior_item, x):
    lo, hi = prior_item['bounds']
    if prior_item['distribution']=='uniform':
        return np.where((x >= lo) & (x <= hi), 1. / (hi-lo), 0.)
    return _pdf_for(prior_item).pdf(x)


def plot_marginals(draws, priors, truth, diagpass, out, dpi):
    stats = summarize(draws)
    fig, axs=plt.subplots(3,3,figsize=(14.5,10),layout='constrained')
    for j,ax in enumerate(axs.flat):
        name = PARAMETER_NAMES[j]; spec=priors['parameters'][name]
        lo,hi = spec['bounds'];grid=np.linspace(lo,hi,300)
        ax.hist(draws[:,j],bins=32,density=True,color='#497EAD',alpha=.68,
                label='DREAM sample' if j==0 else None)
        ax.plot(grid,prior_pdf(spec,grid),color='#B97240',lw=1.7,ls='--',
                label='Illustrative prior' if j==0 else None)
        ax.axvspan(stats['q025'][j],stats['q975'][j],color='#487EA8',alpha=.13,
                   label='95% sample interval' if j==0 else None)
        ax.axvline(stats['median'][j],color='#222222',lw=1.5,
                   label='Sample median' if j==0 else None)
        ax.axvline(truth[j],color='#B33842',lw=1.4,ls=':',
                   label='Synthetic truth' if j==0 else None)
        ax.set(xlabel=PARAM_LABELS[j],ylabel='Density')
        ax.grid(axis='y')
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle(title_with_status('M1 parameter uncertainty: prior, posterior sample and synthetic truth',diagpass),fontsize=15)
    save(fig,out,'m1_pub_marginals',dpi)


def plot_correlations(draws, diagpass, out,dpi):
    r=np.corrcoef(draws,rowvar=False)
    fig,ax=plt.subplots(figsize=(9.5,8.5),layout='constrained')
    im=ax.imshow(r,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_xticks(range(9), PARAM_LABELS,rotation=55,ha='right')
    ax.set_yticks(range(9), PARAM_LABELS)
    for i in range(9):
        for j in range(9):
            ax.text(j,i,f'{r[i,j]:.2f}',ha='center',va='center',fontsize=7.8,
                    color='white' if abs(r[i,j])>.58 else '#222222')
    fig.colorbar(im,ax=ax,shrink=.75,label='Posterior-sample correlation (NOT sensitivity index)')
    ax.set_title(title_with_status('M1 parameter dependence',diagpass),fontsize=13)
    save(fig,out,'m1_pub_correlations',dpi)


def plot_trace(chains,burn, diagpass,out,dpi):
    fig,axs=plt.subplots(3,3,figsize=(14.5,9.5),layout='constrained',sharex=True)
    for j,ax in enumerate(axs.flat):
        for ic,chain in enumerate(chains):
            ax.plot(chain[:,j],lw=.55,alpha=.75,label=f'Chain {ic+1}' if j==0 else None)
        ax.axvline(burn,color='#333333',ls='--',lw=1)
        ax.set(ylabel=PARAM_LABELS[j],xlabel='Iteration')
        ax.grid(alpha=.2)
    fig.legend(loc='outside lower center',ncol=len(chains),frameon=False)
    fig.suptitle(title_with_status('M1 DREAM mixing diagnostics; dashed = burn-in',diagpass),fontsize=15)
    save(fig,out,'m1_pub_traces',dpi)


def corners(t):
    z=SourceRectangle(*map(float,t[4:]))
    corners_local=np.array([[-z.a_m,-z.b_m],[z.a_m,-z.b_m],
                            [z.a_m,z.b_m],[-z.a_m,z.b_m],[-z.a_m,-z.b_m]])
    x,y=local_to_global(corners_local[:,0],corners_local[:,1],z)
    return x,y


def plot_zone(draws,truth,model,converged,out,dpi):
    x_axis=model.hydraulic.x_m; y_axis=model.hydraulic.y_m
    prob=membership_probability(draws,x_axis,y_axis)
    fig,ax=plt.subplots(figsize=(8.5,7),layout='constrained')
    dx=float(np.median(np.diff(x_axis)));dy=float(np.median(np.diff(y_axis)))
    ext=(x_axis[0]-dx/2,x_axis[-1]+dx/2,y_axis[0]-dy/2,y_axis[-1]+dy/2)
    im=ax.imshow(prob,origin='lower',extent=ext,vmin=0,vmax=1,aspect='equal',cmap='viridis')
    xx,yy=np.meshgrid(x_axis,y_axis)
    if 0<prob.max() and np.any((prob>0)&(prob<1)):
        ax.contour(xx,yy,prob,levels=[.1,.5,.9],colors=['#EEEEEE','#FFFFFF','#E0BA44'],
                   linewidths=[.7,1.7,.9])
    xt,yt=corners(truth)
    ax.plot(xt,yt,color='#E8E8E8',ls='--',lw=1.5,label='Synthetic generating boundary')
    ax.plot(truth[7],truth[8],marker='x',markersize=6,color='white',label='Synthetic generating center')
    fig.colorbar(im,ax=ax,shrink=.85,label='P(point inside higher-K geological compartment)')
    ax.set(xlabel="Global $x'$ (m)",ylabel="Global $y'$ (m)")
    ax.legend(fontsize=8,loc='upper right',framealpha=.9)
    ax.set_title(title_with_status('Geological membership uncertainty (all selected geometry draws)',converged))
    save(fig,out,'m1_pub_zone_probability',dpi)
    return prob


def station_figure(station, channels, times, observation, bands, converged,out,dpi):
    comps=('eXX','eYY','eZZ','eXY')
    fig,axs=plt.subplots(2,2,figsize=(12.6,8.1),layout='constrained',sharex=True)
    for comp,ax in zip(comps,axs.flat):
        name=f'{comp}_{station}';k=channels.index(name)
        latent=bands['strain']['latent_q'][:,:,k]
        noisy=bands['strain']['noisy_q'][:,:,k]
        ax.fill_between(times,noisy[0],noisy[2],color='#B6B6B6',alpha=.5,label='95% noisy predictive band')
        ax.fill_between(times,latent[0],latent[2],color='#377AA9',alpha=.32,label='95% latent model band')
        ax.plot(times,latent[1],color='#1E4059',lw=1.7,label='Posterior median')
        ax.scatter(times,observation[:,k],facecolor='white',edgecolor='#B25734',s=19,
                   lw=.95,zorder=5,label='Synthetic observations')
        ax.axvline(19,color='#555555',ls=':',lw=1)
        ax.grid()
        ax.set(title=COMPONENT_LABELS[comp],ylabel='Strain (nanostrain)',xlabel='Time (h)')
    handles, labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    split='HELD-OUT' if station in ('S07','S08') else 'TRAINING'
    fig.suptitle(title_with_status(f'{station} ({split}) — joint pressure–strain M1 posterior predictive',converged),fontsize=13)
    save(fig,out,f'm1_pub_station_{station}',dpi)


def gauges_figure(times,saved,bands,converged,out,dpi):
    fig,axs=plt.subplots(1,3,figsize=(15,4.4),layout='constrained',sharey=True)
    for k,ax in enumerate(axs):
        lat=bands['pressure']['latent_q'][:,:,k]/1000
        noise=bands['pressure']['noisy_q'][:,:,k]/1000
        ax.fill_between(times,noise[0],noise[2],color='#B6B6B6',alpha=.55,
                        label='95% noisy predictive')
        ax.fill_between(times,lat[0],lat[2],color='#377AA9',alpha=.32,label='95% latent model')
        ax.plot(times,lat[1],color='#193D59',lw=1.8,label='Posterior median')
        ax.scatter(times,saved.observed_pressure_pa[:,k]/1000,color='#BB5B2E',s=19,label='Synthetic observations')
        ax.axvline(19,color='#555555',ls=':',lw=1)
        ax.set(xlabel='Time (h)',ylabel='Pressure change (kPa)',
               title=('HELD-OUT' if k==2 else 'TRAINING')+
                f" gauge {k+1} at ({saved.gauges_prime_m[k,0]:g}, {saved.gauges_prime_m[k,1]:g}) m")
        ax.grid()
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle(title_with_status('Off-well pressure predictive checks — M1',converged),fontsize=13)
    save(fig,out,'m1_pub_pressure_gauges',dpi)


def coverage_metrics(saved,bands):
    rows=[]
    for source,obs,mask,noise in (
        ('strain_ns',saved.observed_strain_ns,saved.layout.channel_holdout_mask,False),
        ('pressure_kPa',saved.observed_pressure_pa/1000,np.array([False,False,True]),True),
    ):
        b=bands['pressure' if noise else 'strain']
        for subset,idx in [('all',np.ones(obs.shape[1],bool)),('heldout',mask)]:
            observed=obs[:,idx]
            latent=b['latent_q'][:, :, idx]
            total=b['noisy_q'][:, :, idx]
            if noise:
                latent=latent/1000;total=total/1000
            rows.append({'observable':source,'subset':subset,'n':int(observed.size),
                         'latent_95_coverage':float(np.mean((observed>=latent[0])&(observed<=latent[2]))),
                         'noisy_predictive_95_coverage':float(np.mean((observed>=total[0])&(observed<=total[2]))),
                         'median_rmse':float(np.sqrt(np.mean((observed-latent[1])**2)))})
    return rows


def plot_local_sensitivity(report_path,out,dpi):
    doc=json.loads(Path(report_path).read_text(encoding='utf-8'))
    norms=np.asarray(doc['scaled_column_norms'],float)
    if norms.shape!=(9,) or not np.isfinite(norms).all():
        raise ValueError('Incompatible local identifiability report')
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    ax.bar(range(9),norms/norms.max(),color='#4278A3')
    ax.set_xticks(range(9),PARAM_LABELS,rotation=30,ha='right')
    ax.set(ylabel='Normalized noise-weighted local Jacobian column norm',
           title='Local sensitivity at the synthetic benchmark (NOT Sobol indices)')
    ax.grid(axis='y')
    save(fig,out,'m1_pub_local_sensitivity',dpi)


def main(argv=None):
    a=arguments(argv)
    if a.posterior_draws<8 or a.boundary_draws<0 or a.noise_replicates<1 or a.dpi not in (300,600):
        raise ValueError('Need >=8 forward draws, >=0 boundary draws, >=1 noise replicate, DPI 300 or 600')
    run=Path(a.run_dir).resolve();out=(Path(a.output_dir).resolve() if a.output_dir else run/'publication')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Publication output is not empty; choose a fresh directory: {out}')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_bayesian_manifest(manifest)
    saved=load_saved_m1(manifest['map_dir'],manifest['config'])
    prior=read_prior_config(manifest['prior_config'])
    chains=np.load(run/'chains_physical_parameters.npy',allow_pickle=False)
    lp=np.load(run/'sampler_log_posterior.npy',allow_pickle=False)
    if chains.shape!=(manifest['nchains'],manifest['niterations'],9) or lp.shape!=chains.shape[:2] or not np.isfinite(chains).all() or not np.isfinite(lp).all():
        raise ValueError('Missing, mismatched or nonfinite sampler arrays')
    burn=int(manifest['burn_in']); diag=diagnose(chains,burn,rhat_max=a.rhat_max,min_ess=a.min_ess)
    posterior=chains[:,burn:,:].reshape(-1,9)
    if len(posterior)<8:raise ValueError('Too few retained draws')
    out.mkdir(parents=True,exist_ok=False) if not out.exists() else None
    style();rng=np.random.default_rng(a.seed)
    plot_trace(chains,burn,diag['pass'],out,a.dpi)
    plot_marginals(posterior,prior,saved.truth,diag['pass'],out,a.dpi)
    plot_correlations(posterior,diag['pass'],out,a.dpi)
    model=build_forward(manifest['config'],saved)
    geometry=posterior if a.boundary_draws==0 or a.boundary_draws>=len(posterior) else posterior[rng.choice(len(posterior),a.boundary_draws,replace=False)]
    plot_zone(geometry,saved.truth,model,diag['pass'],out,a.dpi)
    nd=min(a.posterior_draws,len(posterior)); chosen=posterior[rng.choice(len(posterior),nd,replace=False)]
    predict=M1GammaCachedForward(model,saved.times_h)
    strain,pres=[],[]
    for k,theta in enumerate(chosen):
        s,p=predict(theta)
        strain.append(s);pres.append(p)
        if (k+1)%10==0:print(f'Publication predictive forward draws: {k+1}/{nd}',flush=True)
    bands=posterior_predictive_bands(np.asarray(strain),np.asarray(pres),
                                    saved.layout.strain_sigma_ns,saved.layout.pressure_sigma_pa,
                                    seed=a.seed,replicas=a.noise_replicates)
    for station in [f'S{n:02d}' for n in range(1,9)]:
        station_figure(station,saved.channel_names,saved.times_h,saved.observed_strain_ns,bands,
                       diag['pass'],out,a.dpi)
    gauges_figure(saved.times_h,saved,bands,diag['pass'],out,a.dpi)
    rows=coverage_metrics(saved,bands)
    with (out/'m1_pub_predictive_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    if a.local_report:plot_local_sensitivity(a.local_report,out,a.dpi)
    stats=summarize(posterior)
    report={'scope':'Same-model synthetic M1 DREAM: publication-oriented postprocessing',
            'convergence_screen_pass':bool(diag['pass']),
            'not_converged_diagnostic_only':not diag['pass'],
            'number_draws_for_geometry':int(len(geometry)),
            'number_forward_predictive_draws':int(nd),
            'number_noise_replicates_per_draw':int(a.noise_replicates),
            'theta_names':list(PARAMETER_NAMES),
            'posterior_medians':stats['median'].tolist(),
            'posterior_95_percentile_endpoints':[stats['q025'].tolist(),stats['q975'].tolist()],
            'effective_alpha_over_E_median_per_Pa':float((0.7/8e9)*10**stats['median'][3]),
            'diagnostics':diag,'heldout_and_all_predictive_checks':rows,
            'sensitivity':('A separate local Jacobian report was plotted' if a.local_report else 'Not computed'),
            'sobol':'NOT computed from posterior samples; use separate run_m1_sobol.py experiment',
            'warnings':['Synthetic generating truth is shown for benchmark transparency, not field knowledge',
                        'Geology priors are illustrative and are not survey-derived',
                        '95% predictive coverage for one synthetic realization is not frequentist calibration',
                        'Independent Gaussian noise assumption; flow and elastic assumptions remain approximate',
                        'Converged MCMC does not establish physical model validation or unique global geometry']}
    (out/'m1_pub_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Publication M1 suite:',out, '\nCONVERGED SCREEN:',bool(diag['pass']))
    if not diag['pass']:print('DIAGNOSTIC-ONLY FIGURES: do not publish credible intervals as calibrated')


if __name__=='__main__':main()
