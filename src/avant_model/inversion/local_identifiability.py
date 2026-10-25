"""Local, noise-whitened identifiability diagnostics for AVANT.

Numerical Jacobians are derivatives of model predictions with respect to the
log10 parameterization (not observations or likelihood values). Information
matrices are local, conditional on the specified independent Gaussian noise
model, fixed geometry/elastic properties, and the chosen model state.

No conclusion about global uniqueness follows from a full local Jacobian rank.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def central_difference_jacobian(
    predict: Callable[[np.ndarray], np.ndarray],
    theta,
    steps,
) -> np.ndarray:
    """Return d(predict)/d(theta) as an (n_observations, n_parameters) array.

    Prediction may be any nonempty finite array; it is flattened in C order.
    A symmetric central difference requires the caller to check prior bounds.
    """
    center = np.asarray(theta, dtype=float)
    h = np.asarray(steps, dtype=float)
    if center.ndim != 1 or center.size == 0 or h.shape != center.shape:
        raise ValueError("theta and steps must be matching nonempty 1D arrays")
    if not np.isfinite(center).all() or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("theta must be finite; steps must be finite and positive")
    columns = []
    output_shape = None
    for index, step in enumerate(h):
        hi = center.copy()
        lo = center.copy()
        hi[index] += step
        lo[index] -= step
        upper = np.asarray(predict(hi), dtype=float)
        lower = np.asarray(predict(lo), dtype=float)
        if upper.shape != lower.shape or upper.size == 0:
            raise ValueError("prediction shape changed at finite difference offsets")
        if output_shape is None:
            output_shape = upper.shape
        elif output_shape != upper.shape:
            raise ValueError("prediction output shapes differ by parameter")
        if not np.isfinite(upper).all() or not np.isfinite(lower).all():
            raise ValueError("nonfinite forward predictions in finite difference")
        columns.append((upper.ravel() - lower.ravel()) / (2.0 * step))
    return np.column_stack(columns)


def analyze_jacobian(jacobian, *, rank_rtol=1e-10) -> dict:
    """Analyze an already noise-whitened Jacobian (n_observations, n_parameters).

    ``rank_rtol`` is a numerical rank threshold relative to max singular value.
    This is a *local*, linearized Gaussian/Fisher diagnostic, not a proof of
    global identifiability or posterior validity.
    """
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or not jac.shape[0] or not jac.shape[1]:
        raise ValueError("jacobian must be a nonempty 2D array")
    if not np.isfinite(jac).all() or not 0.0 < rank_rtol < 1.0:
        raise ValueError("jacobian must be finite; rank_rtol must be in (0,1)")
    n_obs, n_param = jac.shape
    _u, singular, vt = np.linalg.svd(jac, full_matrices=True)
    singular_full = np.zeros(n_param, dtype=float)
    singular_full[: singular.size] = singular
    smax = float(singular_full.max())
    rank = int(np.count_nonzero(singular_full > rank_rtol * smax)) if smax > 0 else 0
    rank_full = rank == n_param
    fisher = jac.T @ jac
    norms = np.linalg.norm(jac, axis=0)
    cosines = np.divide(fisher, np.outer(norms, norms), out=np.full_like(fisher, np.nan),
                        where=np.outer(norms, norms) > 0)
    # Vt is (n_param,n_param), including the null space in underdetermined systems.
    weakest = vt[-1].copy()
    if weakest[np.argmax(np.abs(weakest))] < 0:
        weakest = -weakest
    result = {
        "n_observations": int(n_obs),
        "n_parameters": int(n_param),
        "numerical_rank": rank,
        "rank_full": bool(rank_full),
        "rank_threshold_relative": float(rank_rtol),
        "singular_values": singular_full.tolist(),
        "condition_number": float(singular_full[0] / singular_full[-1]) if rank_full else None,
        "fisher_information": fisher.tolist(),
        "sensitivity_column_norms": norms.tolist(),
        "sensitivity_column_cosines": cosines.tolist(),
        "weakest_direction_unit": weakest.tolist(),
        "local_covariance": None,
        "local_sd_log10": None,
        "local_correlation": None,
        "local_95pct_halfwidth_log10": None,
    }
    if rank_full:
        # Invert from singular vectors without squaring condition in a generic inverse.
        covariance = (vt.T * (1.0 / singular_full**2)) @ vt
        covariance = (covariance + covariance.T) * 0.5
        sd = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(sd, sd)
        result.update({
            "local_covariance": covariance.tolist(),
            "local_sd_log10": sd.tolist(),
            "local_correlation": correlation.tolist(),
            "local_95pct_halfwidth_log10": (1.95996398454 * sd).tolist(),
        })
    return result


def compare_jacobian_steps(jac_big, jac_half) -> float:
    """Report a relative difference between two finite-difference Jacobians.

    The denominator is max(absolute max Jacobian entry, machine tiny);
    this is a numerical step-stability diagnostic, not a model-error bound.
    """
    big, half = np.asarray(jac_big, float), np.asarray(jac_half, float)
    if big.shape != half.shape or big.ndim != 2 or not np.isfinite(big).all() or not np.isfinite(half).all():
        raise ValueError("jacobian arrays must have the same finite 2D shape")
    denom = max(float(np.max(np.abs(half))), float(np.finfo(float).tiny))
    return float(np.max(np.abs(big - half)) / denom)
