"""Stage 5D-2: repeated synthetic recovery under correct and wrong time-noise assumptions.

This module deliberately has no dependency on the AVANT forward model. Synthetic
pressure/strain predictions are supplied by the caller; both observations and
their temporal correlations are generated outside the optimizer. No DREAM.

Noise is independent across sensor channels and across observation modalities.
Within each channel it follows an exponential Gaussian time covariance.
The confidence intervals are *local Wald approximations*, not Bayesian credible
intervals and not a guarantee of coverage for nonlinear inverse problems.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.stats import norm

from avant_model.inversion.error_model_sensitivity import temporal_correlation_matrix


@dataclass(frozen=True)
class ErrorSpec:
    """Standard deviations and within-channel correlation times in hours."""

    strain_sigma_ns: float
    pressure_sigma_pa: float
    strain_tau_h: float = 0.0
    pressure_tau_h: float = 0.0

    def __post_init__(self):
        for name in ("strain_sigma_ns", "pressure_sigma_pa"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("strain_tau_h", "pressure_tau_h"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >=0")


def gaussian_temporal_noise(times_h, n_channels, sigma, tau_h, rng):
    """Draw (times, channels) Gaussian errors using actual irregular timestamps."""
    times = np.asarray(times_h, float)
    channels = int(n_channels)
    if isinstance(n_channels, (bool, np.bool_)) or channels != n_channels or channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    corr = temporal_correlation_matrix(times, tau_h)
    return float(sigma) * (np.linalg.cholesky(corr) @ rng.normal(size=(len(times), channels)))


def synthetic_observations(strain_truth, pressure_truth, times_h, errors, rng):
    """Return independent strain and pressure noise draws at ALL channels."""
    s = np.asarray(strain_truth, float)
    p = np.asarray(pressure_truth, float)
    if s.ndim != 2 or p.ndim != 2 or s.shape[0] != p.shape[0] or not s.size or not p.size:
        raise ValueError("Strain/pressure truth must be nonempty 2D time-major arrays")
    if s.shape[0] != len(times_h) or not np.isfinite(s).all() or not np.isfinite(p).all():
        raise ValueError("Truth shapes/timestamps or values are invalid")
    return (
        s + gaussian_temporal_noise(times_h, s.shape[1], errors.strain_sigma_ns, errors.strain_tau_h, rng),
        p + gaussian_temporal_noise(times_h, p.shape[1], errors.pressure_sigma_pa, errors.pressure_tau_h, rng),
    )


def whiten_time_major(residuals, times_h, sigma, tau_h):
    """Return time-major residual whitened under independent channels and exp time covariance."""
    e = np.asarray(residuals, float)
    if e.ndim != 2 or e.shape[0] != len(times_h) or e.shape[1] == 0 or not np.isfinite(e).all():
        raise ValueError("Residuals must be finite nonempty (time, channel) array")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if tau_h == 0:
        # Still validate times to catch invalid inputs across the two paths.
        temporal_correlation_matrix(times_h, 0)
        return e / float(sigma)
    lower = np.linalg.cholesky(temporal_correlation_matrix(times_h, tau_h))
    return solve_triangular(lower, e, lower=True, check_finite=True) / float(sigma)


def wald_covariance(jac, *, rank_rtol=1e-10):
    """Known-error, local Gaussian information covariance at a fitted state."""
    j = np.asarray(jac, float)
    if j.ndim != 2 or j.shape[0] < j.shape[1] or not np.isfinite(j).all():
        raise ValueError("Jacobian must be finite and have at least as many rows as columns")
    _u, sing, vt = np.linalg.svd(j, full_matrices=False)
    if not sing.size or sing[0] <= 0 or sing[-1] <= rank_rtol*sing[0]:
        return None
    cov = (vt.T * (1.0/sing**2)) @ vt
    return (cov + cov.T)/2


def fit_replicate(predict, times_h, strain_observed, pressure_observed,
                  strain_train_mask, pressure_train_mask, assumed_errors,
                  start, lower, upper, *, mode="joint", max_nfev=60):
    """Refit 2 log parameters to one synthetic replicate; produce conditional Wald SDs.

    predict(theta) -> (strain_ns, pressure_pa) over ALL channels/times.
    Same data may be refitted with different error assumptions for paired tests.
    """
    if mode not in ("strain_only", "pressure_only", "joint"):
        raise ValueError("Unknown fit mode")
    obs_s, obs_p = np.asarray(strain_observed, float), np.asarray(pressure_observed, float)
    sm, pm = np.asarray(strain_train_mask, bool), np.asarray(pressure_train_mask, bool)
    t = np.asarray(times_h, float)
    if (obs_s.ndim != 2 or obs_p.ndim != 2 or obs_s.shape[0] != t.size or
            obs_p.shape[0] != t.size or sm.shape != (obs_s.shape[1],) or
            pm.shape != (obs_p.shape[1],) or not sm.any() or not pm.any() or
            not np.isfinite(obs_s).all() or not np.isfinite(obs_p).all()):
        raise ValueError("Invalid observations, times or training masks")
    init, lo, hi = (np.asarray(x, float) for x in (start, lower, upper))
    if (init.shape != (2,) or lo.shape != (2,) or hi.shape != (2,) or
            not all(np.isfinite(x).all() for x in (init,lo,hi)) or
            np.any(lo >= hi) or np.any(init <= lo) or np.any(init >= hi)):
        raise ValueError("Expected interior 2-parameter start and valid bounds")

    def residual(theta):
        pred_s, pred_p = (np.asarray(x,float) for x in predict(theta))
        if pred_s.shape != obs_s.shape or pred_p.shape != obs_p.shape:
            raise ValueError("Forward model returned incompatible prediction shapes")
        blocks = []
        if mode in ("strain_only", "joint"):
            blocks.append(whiten_time_major(pred_s[:,sm]-obs_s[:,sm],t,
                                            assumed_errors.strain_sigma_ns,
                                            assumed_errors.strain_tau_h).ravel())
        if mode in ("pressure_only", "joint"):
            blocks.append(whiten_time_major(pred_p[:,pm]-obs_p[:,pm],t,
                                            assumed_errors.pressure_sigma_pa,
                                            assumed_errors.pressure_tau_h).ravel())
        return np.concatenate(blocks)

    result=least_squares(residual,init,bounds=(lo,hi),method="trf",jac="3-point",
                         max_nfev=int(max_nfev),xtol=1e-9,ftol=1e-9,gtol=1e-9)
    if not result.success:
        raise RuntimeError(f"Optimization did not converge: {result.message}")
    cov=wald_covariance(result.jac)
    if cov is None:
        raise RuntimeError("Fitted two-parameter Jacobian is rank-deficient")
    sd=np.sqrt(np.diag(cov))
    estimate=np.asarray(result.x,float)
    halfwidth=float(norm.ppf(.975))*sd
    pred_s,pred_p=predict(estimate)
    if np.any(sm == False):
        held_s=float(np.sqrt(np.mean((pred_s[:,~sm]-obs_s[:,~sm])**2)))
    else:
        held_s=None
    if np.any(pm == False):
        held_p=float(np.sqrt(np.mean((pred_p[:,~pm]-obs_p[:,~pm])**2)))
    else:
        held_p=None
    return {
        "estimate_log10":estimate,
        "local_sd_log10":sd,
        "wald_95pct_low_log10":estimate-halfwidth,
        "wald_95pct_high_log10":estimate+halfwidth,
        "weighted_sum_sq":float(2*result.cost),
        "heldout_strain_rmse_ns":held_s,
        "heldout_pressure_rmse_pa":held_p,
        "function_evaluations":int(result.nfev),
    }


def wilson_interval(successes, trials, *, alpha=.05):
    """Wilson binomial interval for empirical coverage uncertainty."""
    n=int(trials)
    k=int(successes)
    if n < 1 or k < 0 or k > n:
        raise ValueError("Invalid success count")
    z=float(norm.ppf(1-alpha/2))
    p=k/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    margin=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return float(max(0,center-margin) if k else 0.0),float(min(1,center+margin) if k<n else 1.0)
