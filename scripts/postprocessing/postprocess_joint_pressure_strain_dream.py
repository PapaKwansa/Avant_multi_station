#!/usr/bin/env python3
"""Stage 4B: joint DREAM convergence, comparison and held-out predictions.

Does NOT resample DREAM. Posterior intervals must only be interpreted after
rank-folded split R-hat, bulk/tail ESS and chain traces are satisfactory.
The held-out S07/S08 and P03 observations are NEVER used in calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.coupled_dream import (
    converged, rank_diagnostics, sha256_file)
from avant_model.inversion.coupled_parameters import HydraulicBounds, PARAMETER_NAMES
from avant_model.inversion.joint_dream import (
    build_joint_likelihood, joint_mle_record, load_stage4a)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--strain-only-run', default=None,
                   help='Converged Stage 3 run (e.g. hydraulic_dream_007) for posterior comparison')
    p.add_argument('--posterior-draws', type=int, default=200)
    p.add_argument('--noise-replicates', type=int, default=24,
                   help='Noisy pseudo-observations per parameter posterior draw')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--rhat-max', type=float, default=1.01)
    p.add_argument('--ess-min', type=float, default=400.)
    return p


def verify_manifest(manifest):
    checks = [
        ('stage4a_dir','joint_summary.json','stage4a_report_sha256'),
        ('stage4a_dir','synthetic_pressure_observations.npz','stage4a_pressure_observations_sha256'),
        ('stage1_dir','summary.json','stage1_summary_sha256'),
        ('stage1_dir','synthetic_observations.npz','stage1_strain_observations_sha256'),
        ('stage2_map_dir','summary.json','stage2_map_summary_sha256'),
    ]
    for folder, filename, hash_name in checks:
        target = Path(manifest[folder]) / filename
        if not target.is_file() or sha256_file(target) != manifest[hash_name]:
            raise ValueError(f'Input provenance changed since DREAM: {target}')
    for filename, hash_name in (
        ('hydraulic_config','hydraulic_config_sha256'),
        ('mechanical_config','mechanical_config_sha256'),
        ('gauges_csv','gauges_csv_sha256'),
        ('station_file','station_file_sha256'),
    ):
        target = Path(manifest[filename])
        if not target.is_file() or sha256_file(target) != manifest[hash_name]:
            raise ValueError(f'Input config changed since DREAM: {target}')
    if not manifest.get('sampling_completed'):
        raise ValueError('Sampling is not marked completed')


def posterior_statistics(samples):
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 16 or not np.isfinite(x).all():
        raise ValueError('Invalid posterior sample array')
    q = np.quantile(x, [0.025,0.5,0.975], axis=0)
    results = {}
    for j,name in enumerate(PARAMETER_NAMES):
        results[name] = {
            'mean':float(np.mean(x[:,j])), 'sd':float(np.std(x[:,j],ddof=1)),
            'q025':float(q[0,j]), 'median':float(q[1,j]), 'q975':float(q[2,j]),
            'physical_median':float(10**q[1,j]),
            'physical_95pct_interval':[float(10**q[0,j]),float(10**q[2,j])],
            'width_95pct_log10':float(q[2,j]-q[0,j]),
        }
    return results


def load_strain_only_comparison(stage3_run, manifest, joint_post):
    """Read Stage 3 posterior only if data, priors and diagnostics match."""
    if stage3_run is None:
        return None, None
    folder = Path(stage3_run).resolve()
    old_manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    old_summary = json.loads((folder/'posterior_summary.json').read_text(encoding='utf-8'))
    if old_manifest['stage1_observations_sha256'] != manifest['stage1_strain_observations_sha256']:
        raise ValueError('Stage 3 and Stage 4B must use EXACTLY the same strain dataset')
    if old_manifest['hydraulic_config_sha256'] != manifest['hydraulic_config_sha256'] or \
            old_manifest['mechanical_config_sha256'] != manifest['mechanical_config_sha256'] or \
            old_manifest['station_file_sha256'] != manifest['station_file_sha256']:
        raise ValueError('Stage 3 and Stage 4B forward-model configurations differ')
    if not np.allclose(np.asarray(old_manifest['bounds_log10'],float),
                       np.asarray(manifest['bounds_log10'],float),rtol=0,atol=0):
        raise ValueError('Stage 3 and Stage 4B prior bounds differ')
    if not bool(old_summary.get('diagnostic_pass',False)):
        raise ValueError('Strain-only Stage 3 posterior must have passed diagnostics')
    prior_chains = np.load(folder/'chains_log10.npy',allow_pickle=False)
    if prior_chains.ndim != 3 or prior_chains.shape[2] != 2:
        raise ValueError('Stage 3 chain format changed')
    prior_burn = int(old_manifest['burn_in_per_chain'])
    old_samples = prior_chains[:,prior_burn:,:].reshape(-1,2)
    old_post = posterior_statistics(old_samples)
    ratios = {}
    for name in PARAMETER_NAMES:
        ratios[name] = {
            'strain_only_95pct_width_log10':old_post[name]['width_95pct_log10'],
            'joint_95pct_width_log10':joint_post[name]['width_95pct_log10'],
            'joint_over_strain_only_95pct_width_ratio':float(
                joint_post[name]['width_95pct_log10']/old_post[name]['width_95pct_log10']),
            'strain_only_physical_95pct_interval':old_post[name]['physical_95pct_interval'],
        }
    return {'source_run_dir':str(folder),'interval_comparison':ratios}, old_samples


def save_traces(chains,burn,out):
    fig,axes=plt.subplots(2,1,figsize=(11.5,6.6),sharex=True,constrained_layout=True)
    for j,ax in enumerate(axes):
        for i,ch in enumerate(chains):
            ax.plot(ch[:,j],alpha=.65,lw=.6,label=f'Chain {i+1}')
        ax.axvline(burn,ls='--',lw=1,color='black',label='Burn-in')
        ax.set_ylabel(PARAMETER_NAMES[j]);ax.grid(alpha=.2)
    axes[0].legend(ncol=5,fontsize=8)
    axes[-1].set_xlabel('DREAM iteration per chain')
    fig.suptitle('Stage 4B joint pressure+strain DREAM traces')
    for suffix in ('png','pdf'):
        fig.savefig(out/f'joint_dream_trace.{suffix}',dpi=200)
    plt.close(fig)


def save_posteriors(samples,old_samples,theta_mle,truth,out):
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),constrained_layout=True)
    for j,ax in enumerate(axes):
        if old_samples is not None:
            ax.hist(old_samples[:,j],bins=50,density=True,alpha=.33,
                    color='slategray',label='Stage 3 strain-only')
        ax.hist(samples[:,j],bins=50,density=True,alpha=.66,
                color='steelblue',label='Stage 4B joint')
        ax.axvline(theta_mle[j],ls='--',color='black',lw=1.5,label='Stage 4A joint MLE')
        if truth is not None:
            ax.axvline(truth[j],color='firebrick',lw=1.5,label='Synthetic truth')
        ax.set_xlabel(PARAMETER_NAMES[j]);ax.set_ylabel('Posterior density');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Joint pressure+strain posterior vs strain-only posterior')
    for suffix in ('png','pdf'):
        fig.savefig(out/f'joint_posterior_marginals.{suffix}',dpi=210)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,5.7),constrained_layout=True)
    sel=np.random.default_rng(11).choice(len(samples),size=min(12000,len(samples)),replace=False)
    ax.scatter(samples[sel,0],samples[sel,1],alpha=.15,s=3,rasterized=True)
    ax.plot(theta_mle[0],theta_mle[1],marker='x',ms=10,color='black',mew=2,label='Stage 4A joint MLE')
    if truth is not None:
        ax.plot(truth[0],truth[1],marker='*',ms=12,color='firebrick',label='Synthetic truth')
    ax.set(xlabel=PARAMETER_NAMES[0],ylabel=PARAMETER_NAMES[1],title='Stage 4B joint posterior samples')
    ax.grid(alpha=.15);ax.legend()
    for suffix in ('png','pdf'):
        fig.savefig(out/f'joint_posterior_scatter.{suffix}',dpi=220)
    plt.close(fig)


def predictive_bands(latent,noise_sigma,rng,noise_replicates):
    """90% latent and replicated-observation pointwise bands."""
    latent=np.asarray(latent,float)
    if latent.ndim!=3 or latent.shape[0]<1 or noise_sigma<=0 or noise_replicates<1:
        raise ValueError('Invalid prediction samples/noise arguments')
    replicates=latent[:,None,:,:]+rng.normal(
        0.,noise_sigma,size=(latent.shape[0],noise_replicates,*latent.shape[1:]))
    replicated=replicates.reshape(-1,*latent.shape[1:])
    return {'mean':latent.mean(axis=0),
            'latent_p05':np.quantile(latent,.05,axis=0),
            'latent_p95':np.quantile(latent,.95,axis=0),
            'obs_p05':np.quantile(replicated,.05,axis=0),
            'obs_p95':np.quantile(replicated,.95,axis=0)}


def heldout_assessment(bands, observed,mask):
    observed=np.asarray(observed,float)[:,mask]
    mean=bands['mean'][:,mask]
    low=bands['obs_p05'][:,mask]
    high=bands['obs_p95'][:,mask]
    return float(np.sqrt(np.mean((observed-mean)**2))), float(np.mean((observed>=low)&(observed<=high)))


def save_predictive_table(obs,bands,labels,holdout_mask,output_csv,unit):
    rows=[]
    for it,t in enumerate(obs.times_h):
        for k,label in enumerate(labels):
            rows.append({'time_h':float(t),'channel_or_gauge':label,
                         'partition':'holdout' if holdout_mask[k] else 'training',
                         f'observed_{unit}':float(obs.observed_ns[it,k] if unit=='ns' else obs.observed_pa[it,k]),
                         f'posterior_mean_{unit}':float(bands['mean'][it,k]),
                         f'latent_p05_{unit}':float(bands['latent_p05'][it,k]),
                         f'latent_p95_{unit}':float(bands['latent_p95'][it,k]),
                         f'noisy_p05_{unit}':float(bands['obs_p05'][it,k]),
                         f'noisy_p95_{unit}':float(bands['obs_p95'][it,k])})
    pd.DataFrame(rows).to_csv(output_csv,index=False)


def save_heldout_strain(obs,bands,inj_h,out):
    channels=obs.channels
    stations=sorted({c.rsplit('_',1)[1] for c,h in zip(channels,obs.holdout_mask) if h})
    components=('eXX','eYY','eXY','eZZ')
    fig,axes=plt.subplots(len(components),len(stations),figsize=(5.25*len(stations),10.3),
                          sharex=True,squeeze=False,constrained_layout=True)
    for j,station in enumerate(stations):
        for i,component in enumerate(components):
            ax=axes[i,j];label=f'{component}_{station}'
            if label not in channels:
                ax.set_visible(False);continue
            k=channels.index(label)
            ax.fill_between(obs.times_h,bands['obs_p05'][:,k],bands['obs_p95'][:,k],
                            color='gray',alpha=.3,label='90% noisy observation band')
            ax.fill_between(obs.times_h,bands['latent_p05'][:,k],bands['latent_p95'][:,k],
                            color='steelblue',alpha=.4,label='90% latent-strain band')
            ax.plot(obs.times_h,bands['mean'][:,k],color='navy',lw=1.8,label='Joint posterior mean')
            ax.scatter(obs.times_h,obs.observed_ns[:,k],s=18,color='darkorange',
                       label='Held-out synthetic strain',zorder=4)
            ax.axvline(inj_h,ls='--',lw=.9,color='black')
            ax.set(title=label,ylabel='Strain (ns)');ax.grid(alpha=.15)
            if i==0 and j==0:ax.legend(fontsize=7)
    for ax in axes[-1]:ax.set_xlabel('Hours from injection start')
    fig.suptitle('Joint DREAM posterior predictive: S07/S08 held-out strain')
    for suffix in ('png','pdf'):
        fig.savefig(out/f'joint_heldout_strain_predictive.{suffix}',dpi=205)
    plt.close(fig)


def save_heldout_pressure(obs,bands,inj_h,out):
    names=obs.gauge_names
    indices=np.flatnonzero(obs.holdout_mask)
    fig,axes=plt.subplots(len(indices),1,figsize=(9.2,3.8*len(indices)),
                          squeeze=False,sharex=True,constrained_layout=True)
    for i,k in enumerate(indices):
        ax=axes[i,0]
        scale=1e6
        ax.fill_between(obs.times_h,bands['obs_p05'][:,k]/scale,bands['obs_p95'][:,k]/scale,
                        color='gray',alpha=.3,label='90% noisy pressure band')
        ax.fill_between(obs.times_h,bands['latent_p05'][:,k]/scale,bands['latent_p95'][:,k]/scale,
                        color='steelblue',alpha=.4,label='90% latent pressure band')
        ax.plot(obs.times_h,bands['mean'][:,k]/scale,color='navy',lw=1.8,label='Joint posterior mean')
        ax.scatter(obs.times_h,obs.observed_pa[:,k]/scale,s=19,color='darkorange',
                   label='Held-out synthetic pressure',zorder=4)
        ax.axvline(inj_h,ls='--',lw=.9,color='black')
        ax.set(title=f'{names[k]} HELD OUT at ({obs.gauge_x_m[k]:g},{obs.gauge_y_m[k]:g}) m local',
               ylabel='Pressure change (MPa)');ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    axes[-1,0].set_xlabel('Hours from injection start')
    fig.suptitle('Joint DREAM posterior predictive: held-out pressure gauge')
    for suffix in ('png','pdf'):
        fig.savefig(out/f'joint_heldout_pressure_predictive.{suffix}',dpi=220)
    plt.close(fig)


def main(argv=None):
    a=parser().parse_args(argv)
    run=Path(a.run_dir).resolve()
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_manifest(manifest)
    if a.posterior_draws<20 or a.noise_replicates<5 or a.rhat_max<=1 or a.ess_min<1:
        raise ValueError('Invalid postprocessing diagnostic/prediction settings')
    bounds=HydraulicBounds(*[tuple(v) for v in manifest['bounds_log10']])
    chains=np.load(run/'chains_log10.npy',allow_pickle=False)
    logps=np.load(run/'log_posterior.npy',allow_pickle=False)
    nchains,niterations=int(manifest['nchains']),int(manifest['niterations_per_chain'])
    if chains.shape!=(nchains,niterations,2) or logps.shape!=(nchains,niterations):
        raise ValueError('Unexpected DREAM chain/log probability shapes')
    if not np.isfinite(chains).all() or not np.isfinite(logps).all():
        raise ValueError('Nonfinite DREAM chain states or log-probabilities')
    for theta in chains.reshape(-1,2):
        if not bounds.contains(theta):
            raise ValueError('DREAM sample falls outside declared prior bounds')
    burn=int(manifest['burn_in_per_chain'])
    if not 0<=burn<niterations:
        raise ValueError('Invalid saved burn-in')
    posterior_chains=chains[:,burn:,:]
    samples=posterior_chains.reshape(-1,2)
    diag=rank_diagnostics(chains,burn)
    passed=converged(diag,threshold=a.rhat_max,min_ess=a.ess_min,
                     retained_per_chain=posterior_chains.shape[1])
    posterior=posterior_statistics(samples)
    comparison,old_samples=load_strain_only_comparison(a.strain_only_run,manifest,posterior)
    stage4a,stage1,strain,pressure=load_stage4a(
        manifest['stage4a_dir'],manifest['hydraulic_config'],
        manifest['mechanical_config'],manifest['gauges_csv'])
    joint_mle=joint_mle_record(stage4a)
    mle=np.asarray([joint_mle['log10_K'],joint_mle['log10_Ss']],float)
    truth=np.asarray(stage1['truth_log10_K_log10_Ss'],float)
    save_traces(chains,burn,run)
    save_posteriors(samples,old_samples,mle,truth,run)
    print('Building fixed forward operator once for posterior predictions...',flush=True)
    likelihood,hc=build_joint_likelihood(manifest['hydraulic_config'],
        manifest['mechanical_config'],stage1,strain,pressure,bounds)
    rng=np.random.default_rng(a.seed)
    idx=rng.choice(len(samples),size=min(a.posterior_draws,len(samples)),replace=False)
    latent_strain=[];latent_pressure=[]
    for i,k in enumerate(idx):
        eps,p=likelihood.predict(samples[k]);latent_strain.append(eps);latent_pressure.append(p)
        if (i+1)%50==0:
            print(f'Posterior predictive draw {i+1}/{len(idx)}',flush=True)
    s_bands=predictive_bands(np.asarray(latent_strain),strain.sigma_ns,rng,a.noise_replicates)
    p_bands=predictive_bands(np.asarray(latent_pressure),pressure.sigma_pa,rng,a.noise_replicates)
    s_rmse,s_cover=heldout_assessment(s_bands,strain.observed_ns,strain.holdout_mask)
    p_rmse,p_cover=heldout_assessment(p_bands,pressure.observed_pa,pressure.holdout_mask)
    save_predictive_table(strain,s_bands,strain.channels,strain.holdout_mask,
                          run/'joint_strain_posterior_predictive.csv','ns')
    save_predictive_table(pressure,p_bands,pressure.gauge_names,pressure.holdout_mask,
                          run/'joint_pressure_posterior_predictive.csv','pa')
    injection=float(hc['injection_hours'])
    save_heldout_strain(strain,s_bands,injection,run)
    save_heldout_pressure(pressure,p_bands,injection,run)
    csvrows=[]
    for ichain in range(nchains):
        for istep in range(burn,niterations):
            state=chains[ichain,istep]
            csvrows.append({'chain':ichain+1,'iteration':istep+1,
                PARAMETER_NAMES[0]:float(state[0]),PARAMETER_NAMES[1]:float(state[1]),
                'K_m_s':float(10**state[0]),'Ss_per_m':float(10**state[1]),
                'log_posterior_including_prior':float(logps[ichain,istep])})
    pd.DataFrame(csvrows).to_csv(run/'joint_posterior_samples.csv',index=False)
    report={
        'scope':'Stage 4B same-model synthetic joint pressure+strain DREAM; NOT field or COMSOL validation',
        'nchains':nchains,'iterations_per_chain':niterations,
        'burn_in_per_chain':burn,'retained_samples':int(len(samples)),
        'prior_bounds_log10':manifest['bounds_log10'],
        'posterior_log10':posterior,
        'posterior_log10_correlation':float(np.corrcoef(samples.T)[0,1]),
        'stage4a_joint_mle_log10':mle.tolist(),
        'synthetic_truth_log10':truth.tolist(),
        'diagnostics':diag,'rhat_threshold':a.rhat_max,'minimum_bulk_tail_ess':a.ess_min,
        'diagnostic_pass':bool(passed),
        'diagnostic_message':('PASS: rank-folded split Rhat and bulk/tail ESS thresholds satisfied'
          if passed else 'NOT CONVERGED: do not interpret joint posterior credible intervals as reliable'),
        'strain_only_posterior_comparison':comparison,
        'posterior_predictive_draws':int(len(idx)),
        'noise_replicates_per_posterior_draw':int(a.noise_replicates),
        'heldout_strain_rmse_ns':s_rmse,
        'heldout_strain_noisy_90pct_coverage':s_cover,
        'heldout_pressure_rmse_pa':p_rmse,
        'heldout_pressure_noisy_90pct_coverage':p_cover,
        'notes':[
            'Each parameter draw uses one FV solve to predict both pressure and strain.',
            'Pressure gauges sample full FV fields, strain uses fixed H = G @ A.',
            'Training uses ONLY S01-S06 and P01-P02; S07-S08 and P03 are held out.',
            'Latent bands show hydraulic parameter uncertainty; noisy bands add fixed independent sensor error.',
            'Full posterior remains conditional on homogeneous K/Ss, fixed geometry/mechanics/boundaries.',
            'This is same-model synthetic calibration, NOT independent numerical or field validation.',
            'Warm-started MCMC can miss distant modes; independent prior-wide exploration remains useful.',
        ],
    }
    (run/'joint_posterior_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('Saved Stage 4B diagnostics and predictive figures:',run,flush=True)
    return report


if __name__=='__main__':
    main()
