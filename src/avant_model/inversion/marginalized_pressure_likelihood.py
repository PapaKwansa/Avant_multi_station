"""Gaussian pressure-marginalized likelihood for a fixed proposed mechanical geometry.

Given d = H(gamma) q + error, q ~ N(m0,C0), error ~ N(0,R),
returns log p(d | gamma) and conditional p(q | d,gamma).

Units: d nanostrain, H nanostrain/Pa, q Pa, R nanostrain^2,
C0 Pa^2. This module does not sample geometry, assume hydraulic flow,
or silently introduce jitter. Rebuild H for each proposed geometry.

For full reference-differenced time series, supply the correctly stacked
covariance R (including shared-reference cross-time terms). Do NOT use an
IID approximation if fitting all times jointly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MarginalPressureResult:
    log_likelihood: float
    conditional_mean_pa: np.ndarray
    conditional_covariance_pa2: np.ndarray
    predicted_mean_ns: np.ndarray
    residual_ns: np.ndarray


def _cholesky(matrix, n, label):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite ({n}, {n}) matrix")
    if not np.allclose(matrix, matrix.T, atol=0, rtol=1e-10):
        raise ValueError(f"{label} must be symmetric")
    matrix = (matrix + matrix.T) / 2
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label} must be positive definite") from exc


def marginalized_pressure_likelihood(design_ns_per_pa, observed_ns, *,
                                     prior_mean_pa, prior_covariance_pa2,
                                     noise_covariance_ns2):
    """Exact Gaussian marginal log likelihood and conditional pressure posterior.

    Uses a small pressure-space Cholesky system (Woodbury + determinant lemma),
    avoiding construction of the dense M-by-M marginal covariance. Inputs are
    not mutated. Caller must supply a proper positive-definite prior and noise.
    """
    H = np.asarray(design_ns_per_pa, dtype=float)
    d = np.asarray(observed_ns, dtype=float)
    mu = np.asarray(prior_mean_pa, dtype=float)
    if H.ndim != 2 or min(H.shape) == 0 or not np.isfinite(H).all():
        raise ValueError("design must be a finite nonempty 2D matrix")
    m, n = H.shape
    if d.shape != (m,) or mu.shape != (n,) or not np.isfinite(d).all() or not np.isfinite(mu).all():
        raise ValueError("observation or prior mean shape/values invalid")
    L0 = _cholesky(prior_covariance_pa2, n, "prior covariance")
    LR = _cholesky(noise_covariance_ns2, m, "noise covariance")
    A = np.linalg.solve(LR, H @ L0)
    z = np.linalg.solve(LR, d - H @ mu)
    K = np.eye(n) + A.T @ A
    LK = np.linalg.cholesky(K)
    w = np.linalg.solve(LK.T, np.linalg.solve(LK, A.T @ z))
    q_mean = mu + L0 @ w
    K_inv = np.linalg.solve(LK.T, np.linalg.solve(LK, np.eye(n)))
    q_cov = L0 @ K_inv @ L0.T
    q_cov = (q_cov + q_cov.T) / 2
    quad = float(z @ z - (A.T @ z) @ w)
    logdet = 2 * (np.log(np.diag(LR)).sum() + np.log(np.diag(LK)).sum())
    ll = -0.5 * (m * np.log(2 * np.pi) + logdet + quad)
    pred = H @ q_mean
    return MarginalPressureResult(float(ll), q_mean, q_cov, pred, d - pred)


def reference_difference_covariance(base_covariance_ns2, n_nonreference_times):
    """Covariance of [d_t - d_0] for independent errors at each raw time.

    Assumes the SAME per-time SPD covariance R0 and independence between raw
    times. Output ordering is time-major. Off-diagonal blocks equal R0.
    This is NOT a model-discrepancy covariance and may be too optimistic.
    """
    R0 = np.asarray(base_covariance_ns2, dtype=float)
    if R0.ndim != 2 or R0.shape[0] == 0 or R0.shape[0] != R0.shape[1]:
        raise ValueError("base covariance must be a square matrix")
    _cholesky(R0, R0.shape[0], "base covariance")
    t = int(n_nonreference_times)
    if t < 1 or t != n_nonreference_times:
        raise ValueError("number of nonreference times must be a positive integer")
    return np.kron(np.ones((t, t)) + np.eye(t), R0)
