"""Pure numerical utilities for M1 posterior figures and independent-input Sobol.

No simulation is executed on import. Neither MCMC samples nor their correlations
are treated as independent Sobol input draws.
"""
from __future__ import annotations

import numpy as np

PARAM_LABELS = (
    r'$\log_{10} K_{\mathrm{in}}$', r'$\log_{10} K_{\mathrm{out}}$',
    r'$\log_{10} S_s$', r'$\log_{10}\gamma_{\mathrm{rel}}$',
    r'$a$ (m)', r'$b$ (m)', r'$\theta$ ($^\circ$)',
    r"$x'_0$ (m)", r"$y'_0$ (m)",
)
COMPONENT_LABELS = {'eXX': r'$\varepsilon_{xx}$', 'eYY': r'$\varepsilon_{yy}$',
                    'eZZ': r'$\varepsilon_{zz}$', 'eXY': r'$\varepsilon_{xy}$'}


def summarize(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != 9 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError('Need a finite (n>=2,9) parameter sample array')
    return {'mean': np.mean(x, axis=0), 'median': np.median(x, axis=0),
            'q025': np.quantile(x, 0.025, axis=0),
            'q975': np.quantile(x, 0.975, axis=0)}


def membership_probability(draws, x_axis, y_axis, *, block_size=64):
    """Geological-zone inclusion probability. Does NOT evaluate a flow model.

    Rectangle x/y rotation must agree with the production M1 coordinate map:
    u = dx*cos(theta) - dy*sin(theta), v = dx*sin(theta) + dy*cos(theta).
    """
    x = np.asarray(draws, dtype=float)
    xx, yy = np.meshgrid(np.asarray(x_axis, float), np.asarray(y_axis, float))
    if x.ndim != 2 or x.shape[1] != 9 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite nine-parameter geometry draws')
    if np.any(x[:, 4] <= 0) or np.any(x[:, 5] <= x[:, 4]):
        raise ValueError('Invalid geometry: need b>a>0')
    prob = np.zeros_like(xx, dtype=float)
    for i in range(0, len(x), block_size):
        part = x[i:i+block_size]
        theta = np.deg2rad(part[:, 6])[:, None, None]
        dx = xx[None] - part[:, 7, None, None]
        dy = yy[None] - part[:, 8, None, None]
        u = dx*np.cos(theta) - dy*np.sin(theta)
        v = dx*np.sin(theta) + dy*np.cos(theta)
        inside = (np.abs(u) <= part[:, 4, None, None]) & (np.abs(v) <= part[:, 5, None, None])
        prob += inside.sum(axis=0)
    return prob / len(x)


def posterior_predictive_bands(pred_strain, pred_pressure, sigma_strain_ns,
                               sigma_pressure_pa, *, seed=1234, replicas=4):
    """Return latent and noisy [2.5%, 50%, 97.5%] intervals for both observables.

    Gaussian measurement-noise replicates are drawn INDEPENDENTLY of parameter draws.
    Posterior latent model uncertainty and total predictive uncertainty differ.
    """
    if replicas < 1 or sigma_strain_ns <= 0 or sigma_pressure_pa <= 0:
        raise ValueError('Positive noise SD and >=1 replication required')
    s, p = np.asarray(pred_strain, float), np.asarray(pred_pressure, float)
    if s.ndim != 3 or p.ndim != 3 or s.shape[:2] != p.shape[:2] or len(s) < 2:
        raise ValueError('Expected (draw,time,channel) predictions with common draw/time')
    if not np.isfinite(s).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite posterior predictions')
    rng = np.random.default_rng(seed)
    ret = {}
    for name, signal, sigma in (('strain', s, sigma_strain_ns),
                                ('pressure', p, sigma_pressure_pa)):
        noisy = signal[:, None, :, :] + rng.normal(0, sigma,
                                       (len(signal), replicas, *signal.shape[1:]))
        ret[name] = {
            'latent_q': np.quantile(signal, [.025, .5, .975], axis=0),
            'noisy_q': np.quantile(noisy.reshape(-1, *signal.shape[1:]),
                                   [.025, .5, .975], axis=0),
        }
    return ret


def sobol_jansen_saltelli(ya, yb, yab):
    """First order Saltelli (1993/2010) and total order Jansen (1999) estimates.

    AB_i replaces the i-th A input with B's i-th input.
    For independent inputs f=x_i, Y_AB_i - Y_A tracks the i-th effect.
    Shape: A/B (N,m), AB (N,d,m). Raw finite-N estimates are not clipped.
    """
    a, b, ab = np.asarray(ya, float), np.asarray(yb, float), np.asarray(yab, float)
    if a.ndim != 2 or b.shape != a.shape or ab.ndim != 3 or ab.shape[0] != a.shape[0] or ab.shape[2] != a.shape[1]:
        raise ValueError('Expected A/B (N,outputs) and AB (N,parameters,outputs)')
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(ab).all()):
        raise ValueError('Nonfinite Sobol model evaluations')
    variance = np.var(np.concatenate([a, b], axis=0), axis=0, ddof=1)
    if np.any(variance <= 1e-15):
        raise ValueError('Zero or negligible output variance; Sobol indices undefined')
    centered_b = b - np.mean(np.concatenate([a, b], axis=0), axis=0)
    s1 = np.mean(centered_b[:, None, :] * (ab - a[:, None, :]), axis=0) / variance[None, :]
    st = np.mean((a[:, None, :] - ab)**2, axis=0)/(2*variance[None, :])
    return s1, st


def sobol_bootstrap(ya, yb, yab, *, draws=300, seed=1729):
    if draws < 10:
        raise ValueError('Need at least ten bootstrap replicates')
    a = np.asarray(ya, float); b = np.asarray(yb, float); ab = np.asarray(yab, float)
    rng = np.random.default_rng(seed)
    s1_b, st_b = [], []
    for _ in range(draws):
        ix = rng.integers(0, len(a), len(a))
        try:
            s1, st = sobol_jansen_saltelli(a[ix], b[ix], ab[ix])
        except ValueError:
            continue
        s1_b.append(s1); st_b.append(st)
    if len(s1_b) < max(10, draws // 2):
        raise ValueError('Too many degenerate bootstrap replicates')
    return (np.quantile(s1_b, [.025, .975], axis=0),
            np.quantile(st_b, [.025, .975], axis=0))


def transformed_independent_priors(u, priors, names, *, measure='illustrative-priors'):
    from scipy.stats import truncnorm
    z = np.asarray(u, float)
    if z.ndim != 2 or z.shape[1] != len(names) or not np.isfinite(z).all() or np.any(z<0) or np.any(z>1):
        raise ValueError('Expected unit-hypercube independent input draws')
    if measure not in ('illustrative-priors', 'uniform-bounds'):
        raise ValueError('Unknown input measure')
    out = np.empty_like(z)
    for j,name in enumerate(names):
        spec = priors['parameters'][name]
        lo, hi = map(float, spec['bounds'])
        if measure == 'uniform-bounds' or spec['distribution'] == 'uniform':
            out[:, j] = lo + (hi-lo)*z[:,j]
        else:
            mean, sd = float(spec['mean']), float(spec['sd'])
            distribution = truncnorm((lo-mean)/sd, (hi-mean)/sd, loc=mean, scale=sd)
            out[:, j] = distribution.ppf(np.clip(z[:,j],1e-12,1-1e-12))
    return out


def verify_bayesian_manifest(manifest):
    """Fail closed if any saved MAP/prior/config inputs changed after sampling."""
    from pathlib import Path
    from avant_model.inversion.m1_bayesian import sha256
    for key, hash_key, filename in (
        ('map_dir','map_report','m1_map_recovery_report.json'),
        ('map_dir','saved_observations','m1_synthetic_observations.npz'),
        ('config','config',None),
        ('prior_config','prior_config',None)):
        p=Path(manifest[key])
        if filename:p=p/filename
        if not p.is_file() or sha256(p)!=manifest['hashes'][hash_key]:
            raise ValueError('Bayesian input was modified or moved after MCMC: '+str(p))
    if not (manifest.get('sampling_completed') and
            manifest.get('truth_not_used_in_priors_likelihood_or_fixed_params')):
        raise ValueError('Sampler completion/truth-separation requirements not satisfied')
