"""Generic rank-folded convergence diagnostics and held-out prediction bands for 9D M1.

No posterior is declared converged solely because a sampler returned states.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata, norm
from avant_model.inversion.m1_map_recovery import PARAMETER_NAMES


def normalize_rank(x):
    a = np.asarray(x,float)
    ranks = rankdata(a.ravel(),method='average')
    u = (ranks-.375)/(a.size+.25)
    return norm.ppf(u).reshape(a.shape)


def split_chains(chains):
    x = np.asarray(chains,float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 16 or not np.isfinite(x).all():
        raise ValueError('Need >=3 finite chains with >=16 retained samples')
    n = x.shape[1]//2
    return np.concatenate((x[:,:n],x[:,-n:]),axis=0)


def rhat(chains):
    x = np.asarray(chains,float)
    n = x.shape[1]
    W = float(np.mean(np.var(x,axis=1,ddof=1)))
    B = float(n*np.var(np.mean(x,axis=1),ddof=1))
    if W <= 0:
        return float('inf')
    return float(np.sqrt(((n-1)*W/n+B/n)/W))


def ess(chains):
    x = np.asarray(chains,float)
    m,n = x.shape
    centered = x - x.mean(axis=1,keepdims=True)
    fft_size = 1 << (2*n-1).bit_length()
    f = np.fft.rfft(centered,n=fft_size,axis=1)
    acov = np.fft.irfft(f * np.conj(f),n=fft_size,axis=1)[:,:n]
    acov /= np.arange(n,0,-1)[None,:]
    W = np.mean(acov[:,0])*n/(n-1)
    B = n*np.var(x.mean(axis=1),ddof=1)
    var_plus = (n-1)*W/n+B/n
    if not np.isfinite(var_plus) or var_plus <= 0:
        return float('nan')
    rho = np.ones(n)
    rho[1:] = 1-(W-np.mean(acov[:,1:],axis=0))/var_plus
    paired=[]
    for k in range(0,n-1,2):
        v = float(rho[k]+rho[k+1])
        if not np.isfinite(v) or v <= 0:
            break
        paired.append(min(v,paired[-1]) if paired else v)
    tau = max(1., -1.+2*sum(paired))
    return float(min(m*n,m*n/tau))


def diagnose(chains,burn_in,*,rhat_max=1.01,min_ess=400):
    x = np.asarray(chains,float)
    if (x.ndim != 3 or x.shape[0] < 4 or x.shape[2] != 9 or
        not 0 <= burn_in < x.shape[1] or not np.isfinite(x).all()):
        raise ValueError('Expected finite chain tensor (>=4,n_iterations,9)')
    post=x[:,burn_in:,:]
    if post.shape[1] < 20:
        raise ValueError('Insufficient post-burn draws')
    metrics={}
    for j,name in enumerate(PARAMETER_NAMES):
        split=split_chains(post[:,:,j])
        bulk=normalize_rank(split)
        folded=normalize_rank(np.abs(split-np.median(split)))
        rb=max(rhat(bulk),rhat(folded))
        bulk_ess=ess(bulk)
        tail_ess=min(ess((split<=np.quantile(split,q)).astype(float)) for q in (.05,.95))
        metrics[name]={'rank_folded_split_rhat':float(rb),
                       'bulk_ess':float(bulk_ess),
                       'tail_ess':float(tail_ess)}
    passed=(post.shape[1]>=100 and all(
        np.isfinite(d['rank_folded_split_rhat']) and d['rank_folded_split_rhat'] < rhat_max
        and np.isfinite(d['bulk_ess']) and d['bulk_ess'] >= min_ess
        and np.isfinite(d['tail_ess']) and d['tail_ess'] >= min_ess
        for d in metrics.values()))
    return {'pass':bool(passed), 'metrics':metrics,'rhat_max_threshold':float(rhat_max),
            'ess_min_threshold':float(min_ess),
            'retained_per_chain':int(post.shape[1]),
            'interpretation':('Convergence screen passed; not proof of global mode exploration'
                              if passed else 'DO NOT interpret sample quantiles as converged credible intervals')}


def predictive_bands(latent, sigma, rng, noise_replicates=12):
    x=np.asarray(latent,float)
    if x.ndim != 3 or len(x)<2 or not np.isfinite(x).all() or sigma<=0 or noise_replicates<1:
        raise ValueError('Invalid latent predictions or observation-noise scale')
    noisy=x[:,None,:,:]+rng.normal(0,sigma,size=(len(x),noise_replicates,*x.shape[1:]))
    noisy=noisy.reshape(-1,*x.shape[1:])
    return {'mean':x.mean(axis=0), 'latent_q05':np.quantile(x,.05,axis=0),
            'latent_q95':np.quantile(x,.95,axis=0),
            'noisy_q05':np.quantile(noisy,.05,axis=0),
            'noisy_q95':np.quantile(noisy,.95,axis=0)}


def heldout_score(bands,observed,mask):
    obs=np.asarray(observed,float)[:,mask]
    if obs.shape!=bands['mean'][:,mask].shape:
        raise ValueError('Invalid held-out prediction shapes')
    mean=bands['mean'][:,mask]
    inside=(obs>=bands['noisy_q05'][:,mask])&(obs<=bands['noisy_q95'][:,mask])
    return {'heldout_RMSE':float(np.sqrt(np.mean((obs-mean)**2))),
            'heldout_noisy_90pct_pointwise_coverage':float(np.mean(inside)),
            'heldout_count':int(obs.size)}
