"""Stage 5D: conditional Fisher-information sensitivity to observation-error models.

Inputs are EXISTING, independently whitened training Jacobians from Stage 5.
Here an assumed covariance is changed; data are not regenerated, parameters
are not refitted, and this is NOT empirical coverage or a Bayesian posterior.

Time correlation is independent between channels and applies ONLY along time:
    Corr[e_i(t), e_i(u)] = exp(-abs(t-u)/tau_hours).
The irregular actual times are used, not an assumed fixed sample index spacing.
The pressure and strain likelihood blocks remain independent of one another.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from avant_model.inversion.local_identifiability import analyze_jacobian


def temporal_correlation_matrix(times_h, tau_h=0.0):
    """PD exponential correlation for strictly increasing observation times."""
    times = np.asarray(times_h, dtype=float)
    tau = float(tau_h)
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("times_h must be a finite vector with >=2 timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times_h must be strictly increasing and unique")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau_h must be finite and nonnegative")
    if tau == 0:
        return np.eye(times.size)
    return np.exp(-np.abs(times[:, None] - times[None, :]) / tau)


def whiten_saved_jacobian(jacobian, times_h, n_channels, noise_multiplier=1.0,
                           correlation_tau_h=0.0):
    """Reweight an ALREADY independent-sigma-whitened time-major Jacobian.

    Original J0[t,channel,param] = d prediction / d param / original_sigma.
    The new covariance is (original_sigma * multiplier)^2 R, within each
    channel independently, with R an exponential time-correlation matrix.
    Thus J_new = (L^-1 J0) / multiplier, where R = L L^T.
    """
    t = np.asarray(times_h, float)
    mat = np.asarray(jacobian, float)
    n_ch = int(n_channels)
    mult = float(noise_multiplier)
    if mat.ndim != 2 or mat.shape[1] < 1 or not np.isfinite(mat).all():
        raise ValueError("jacobian must be nonempty finite 2D")
    if n_ch < 1 or isinstance(n_channels, (bool, np.bool_)) or n_channels != n_ch:
        raise ValueError("n_channels must be a positive integer")
    if mat.shape[0] != t.size * n_ch:
        raise ValueError("time-major Jacobian rows do not match time/channel dimensions")
    if not np.isfinite(mult) or mult <= 0:
        raise ValueError("noise_multiplier must be positive and finite")
    corr = temporal_correlation_matrix(t, correlation_tau_h)
    if correlation_tau_h == 0:
        return mat / mult
    lower = np.linalg.cholesky(corr)
    block = mat.reshape(t.size, n_ch * mat.shape[1])
    return (solve_triangular(lower, block, lower=True, check_finite=True) / mult).reshape(mat.shape)


def evaluate_error_case(jac_strain, jac_pressure, times_h, n_strain_channels,
                        n_pressure_channels, *, strain_factor=1.0,
                        pressure_factor=1.0, strain_tau_h=0.0,
                        pressure_tau_h=0.0):
    """Return strain/pressure/joint local information under specified covariance."""
    js = whiten_saved_jacobian(jac_strain, times_h, n_strain_channels,
                               strain_factor, strain_tau_h)
    jp = whiten_saved_jacobian(jac_pressure, times_h, n_pressure_channels,
                               pressure_factor, pressure_tau_h)
    reports = {
        "Strain only": analyze_jacobian(js),
        "Pressure only": analyze_jacobian(jp),
        "Joint": analyze_jacobian(np.vstack((js, jp))),
    }
    fisher_sum = (np.asarray(reports["Strain only"]["fisher_information"])
                  + np.asarray(reports["Pressure only"]["fisher_information"]))
    np.testing.assert_allclose(reports["Joint"]["fisher_information"],
                               fisher_sum, rtol=1e-12, atol=1e-7)
    ratio = {}
    for j, name in enumerate(("log10_K", "log10_Ss")):
        s_sd, j_sd = (reports[key]["local_sd_log10"]
                      for key in ("Strain only", "Joint"))
        ratio[name] = float(j_sd[j] / s_sd[j]) if s_sd is not None and j_sd is not None else None
    return {
        "assumed_strain_sigma_multiplier": float(strain_factor),
        "assumed_pressure_sigma_multiplier": float(pressure_factor),
        "assumed_strain_time_correlation_hours": float(strain_tau_h),
        "assumed_pressure_time_correlation_hours": float(pressure_tau_h),
        "information": reports,
        "joint_over_strain_local_sd_ratio": ratio,
    }
