#!/usr/bin/env python3
"""M1 DREAM convergence diagnostics, uncertainty and held-out synthetic predictions.

Will NOT label intervals calibrated/credible when rank-folded Rhat or ESS fails.
Sources, prior, and MAP saved observations are hash-verified before plotting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from avant_model.inversion.m1_bayesian import (
    build_forward, load_saved_m1, read_prior_config, sha256, M1GammaCachedForward,
)
from avant_model.inversion.m1_map_recovery import PARAMETER_NAMES
from avant_model.inversion.m1_posterior_diagnostics import (
    diagnose, predictive_bands, heldout_score,
)
from avant_model.model.stage6_coordinates import SourceRectangle, global_to_local


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--posterior-draws',type=int,default=40)
    p.add_argument('--noise-replicates',type=int,default=16)
    p.add_argument('--seed',type=int,default=116)
    p.add_argument('--rhat-max',type=float,default=1.01)
    p.add_argument('--min-ess',type=float,default=400)
    return p


def verify_manifest(manifest):
    for key,hash_key,filename in (
        ('map_dir','map_report','m1_map_recovery_report.json'),
        ('map_dir','saved_observations','m1_synthetic_observations.npz'),
        ('config','config',None),
        ('prior_config','prior_config',None)):
        p=Path(manifest[key])
        if filename:p=p/filename
        if not p.is_file() or sha256(p)!=manifest['hashes'][hash_key]:
            raise ValueError('Bayesian input changed since sampling: '+str(p))
    if not manifest['sampling_completed'] or not manifest['truth_not_used_in_priors_likelihood_or_fixed_params']:
        raise ValueError('Run has not completed or truth separation not confirmed')


def posterior_stats(samples):
    x=np.asarray(samples,float)
    if x.ndim!=2 or x.shape[1]!=9 or len(x)<10 or not np.isfinite(x).all():
        raise ValueError('Expected finite nine-parameter posterior samples')
    q=np.quantile(x,[.025,.5,.975],axis=0)
    results={}
    for j,name in enumerate(PARAMETER_NAMES):
        results[name]={'median':float(q[1,j]), 'q025':float(q[0,j]),
                       'q975':float(q[2,j]),'mean':float(np.mean(x[:,j])),
                       'std':float(np.std(x[:,j],ddof=1))}
    return results


def save_trace(states,burn,out):
    fig,axes=plt.subplots(3,3,figsize=(14,9),constrained_layout=True)
    for j,ax in enumerate(axes.flat):
        for k in range(len(states)):
            ax.plot(states[k,:,j],lw=.5,alpha=.65,label=f'chain {k+1}')
        ax.axvline(burn,lw=1,ls='--',color='black')
        ax.set_title(PARAMETER_NAMES[j],fontsize=10)
        ax.set_xlabel('Iteration')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6,ncol=2)
    fig.suptitle('M1 nine-parameter DREAM traces; dashed line = burn-in')
    for ext in ('png','pdf'):fig.savefig(out/f'm1_dream_traces.{ext}',dpi=185)
    plt.close(fig)


def save_marginals(draws,priors,truth,converged,out):
    from avant_model.inversion.m1_bayesian import _pdf_for
    fig,axes=plt.subplots(3,3,figsize=(14,9.5),constrained_layout=True)
    for j,ax in enumerate(axes.flat):
        name=PARAMETER_NAMES[j];spec=priors['parameters'][name]
        lo,hi=spec['bounds'];v=np.linspace(lo,hi,201)
        if spec['distribution']=='uniform':pdf=np.full_like(v,1/(hi-lo))
        else:pdf=_pdf_for(spec).pdf(v)
        ax.plot(v,pdf,ls='--',lw=1.3,label='Illustrative prior')
        ax.hist(draws[:,j],bins=35,density=True,alpha=.5,label='DREAM samples')
        ax.axvline(truth[j],ls=':',lw=1.5,label='Synthetic generating value')
        ax.set_title(name,fontsize=10);ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle('M1 posterior marginals' if converged else
                 'PRELIMINARY M1 marginal samples — CONVERGENCE NOT VERIFIED')
    for ext in ('png','pdf'):fig.savefig(out/f'm1_dream_marginals.{ext}',dpi=200)
    plt.close(fig)


def save_correlations(draws,converged,out):
    corr=np.corrcoef(draws,rowvar=False)
    fig,ax=plt.subplots(figsize=(9.5,8.5),constrained_layout=True)
    im=ax.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(range(9),PARAMETER_NAMES,rotation=90,fontsize=8)
    ax.set_yticks(range(9),PARAMETER_NAMES,fontsize=8)
    fig.colorbar(im,ax=ax,label='Sample parameter correlation')
    ax.set_title('Sample correlations'+('' if converged else ' (NOT CONVERGED)'))
    for ext in ('png','pdf'):fig.savefig(out/f'm1_dream_correlations.{ext}',dpi=200)
    plt.close(fig)


def save_geometry_probability(draws,model,converged,out):
    xx,yy=np.meshgrid(model.hydraulic.x_m,model.hydraulic.y_m)
    prob=np.zeros_like(xx,float)
    for t in draws:
        zone=SourceRectangle(*t[4:])
        u,v=global_to_local(xx,yy,zone)
        prob+=(np.abs(u)<=zone.a_m)&(np.abs(v)<=zone.b_m)
    prob/=len(draws)
    fig,ax=plt.subplots(figsize=(8.5,6),constrained_layout=True)
    img=ax.imshow(prob,origin='lower',extent=model.hydraulic.domain.extent(),
                  vmin=0,vmax=1,aspect='auto')
    fig.colorbar(img,ax=ax,label='Posterior probability point is inside high-K zone')
    ax.set(xlabel="Global x' (m)",ylabel="Global y' (m)")
    ax.set_title('M1 geological zone membership'+('' if converged else ' — NONCONVERGED'))
    for ext in ('png','pdf'):fig.savefig(out/f'm1_zone_probability.{ext}',dpi=200)
    plt.close(fig)


def save_heldout(times,saved,strain_bands,pressure_bands,converged,out):
    channel='eXX_S07'
    k=saved.channel_names.index(channel)
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    for ax,bands,idx,obs,scale,title,unit in (
        (axes[0],strain_bands,k,saved.observed_strain_ns,1.,channel,'nanostrain'),
        (axes[1],pressure_bands,2,saved.observed_pressure_pa,1e3,
         'Invented held-out pressure gauge (0,-350) m','kPa')):
        ax.fill_between(times,bands['noisy_q05'][:,idx]/scale,bands['noisy_q95'][:,idx]/scale,
                        alpha=.25,color='gray',label='90% noisy predictive band')
        ax.fill_between(times,bands['latent_q05'][:,idx]/scale,bands['latent_q95'][:,idx]/scale,
                        alpha=.35,color='steelblue',label='90% latent model band')
        ax.plot(times,bands['mean'][:,idx]/scale,color='black',label='Posterior mean')
        ax.scatter(times,obs[:,idx]/scale,color='darkorange',s=20,label='Held-out synthetic data')
        ax.axvline(19,ls=':',color='black',alpha=.7)
        ax.set(xlabel='Time (h)',ylabel=unit,title=title)
    axes[0].legend(fontsize=7)
    fig.suptitle('M1 held-out posterior predictions' if converged else
                 'PRELIMINARY held-out predictions — CONVERGENCE NOT VERIFIED')
    for ext in ('png','pdf'):fig.savefig(out/f'm1_heldout_predictive.{ext}',dpi=200)
    plt.close(fig)


def main(argv=None):
    a=parser().parse_args(argv)
    if a.posterior_draws<2 or a.noise_replicates<1:
        raise ValueError('Need >=2 predictive draws and >=1 noise replicate')
    out=Path(a.run_dir).resolve()
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    verify_manifest(manifest)
    saved=load_saved_m1(manifest['map_dir'],manifest['config'])
    priors=read_prior_config(manifest['prior_config'])
    chains=np.load(out/'chains_physical_parameters.npy',allow_pickle=False)
    lp=np.load(out/'sampler_log_posterior.npy',allow_pickle=False)
    if chains.shape!=(manifest['nchains'],manifest['niterations'],9) or lp.shape!=chains.shape[:2]:
        raise ValueError('Saved DREAM chain dimensions inconsistent with manifest')
    diag=diagnose(chains,int(manifest['burn_in']),rhat_max=a.rhat_max,min_ess=a.min_ess)
    post=chains[:,manifest['burn_in']:,:].reshape(-1,9)
    stats=posterior_stats(post)
    print('DREAM convergence gate:',diag['pass'])
    for name,d in diag['metrics'].items():
        print(f'  {name:20s} Rhat={d["rank_folded_split_rhat"]:.4g} '
              f'bulkESS={d["bulk_ess"]:.1f} tailESS={d["tail_ess"]:.1f}')
    save_trace(chains,manifest['burn_in'],out)
    save_marginals(post,priors,saved.truth,diag['pass'],out)
    save_correlations(post,diag['pass'],out)
    rng=np.random.default_rng(a.seed)
    picked=post[rng.choice(len(post),min(a.posterior_draws,len(post)),replace=False)]
    model=build_forward(manifest['config'],saved)
    save_geometry_probability(picked,model,diag['pass'],out)
    predict=M1GammaCachedForward(model,saved.times_h)
    strain=[]; pressure=[]
    for j,t in enumerate(picked):
        s,p=predict(t)
        strain.append(s);pressure.append(p)
        if (j+1)%10==0:print('Posterior predictive draws:',j+1,flush=True)
    sband=predictive_bands(np.stack(strain),saved.layout.strain_sigma_ns,rng,a.noise_replicates)
    pband=predictive_bands(np.stack(pressure),saved.layout.pressure_sigma_pa,rng,a.noise_replicates)
    save_heldout(saved.times_h,saved,sband,pband,diag['pass'],out)
    metrics={'strain':heldout_score(sband,saved.observed_strain_ns,saved.layout.channel_holdout_mask),
             'pressure':heldout_score(pband,saved.observed_pressure_pa,np.array([False,False,True]))}
    metrics['pressure']['heldout_RMSE_kPa']=metrics['pressure'].pop('heldout_RMSE')/1000.
    summary={'scope':manifest['scope'],'diagnostic_pass':diag['pass'],
             'diagnostics':diag,'posterior_parameter_summary_exploratory_if_not_converged':stats,
             'heldout_predictive':metrics,
             'number_posterior_draws_used_for_predictive':len(picked),
             'effective_alpha_over_E_ref_per_Pa':float(0.7/8e9),
             'gamma_rel_median':float(10**stats['log10_gamma_rel']['median']),
             'alpha_over_E_median_per_Pa':float((0.7/8e9)*10**stats['log10_gamma_rel']['median']),
             'limitations':[
                 'Same-model synthetic observations, invented gauge sites, not COMSOL/field.',
                 'Illustrative soft geology priors; no independently measured geologic constraint.',
                 'Convergence diagnostics cannot establish global exploration of separated modes.',
                 'Nonconverged samples must not be interpreted as calibrated credible intervals.',
                 'One-way 2D flow, fixed global physical boundaries, fixed mechanical support.',
                 'Two-zone FV cell mixture and homogeneous elastic half-space approximate physical geology.',
                 'Independent Gaussian errors; correlations may create undercoverage as Stage 5 found.',
                 'E and alpha cannot be separated: only alpha/E is inferred.',
                 'True field boundary inference must be independently validated and checked for resolution effects.',
             ]}
    (out/'m1_posterior_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved posterior diagnostic report:',out/'m1_posterior_summary.json')
    print('Status:', 'CONVERGENCE SCREEN PASSED' if diag['pass'] else
          'NOT CONVERGED — no validated credible intervals or geological boundary probabilities')

if __name__=='__main__':main()
