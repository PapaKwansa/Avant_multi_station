"""Exact finite-dimensional linear/Gaussian baseline; no DREAM or neural network.

Unknown coefficients have pressure units (Pa); use G @ B as the design matrix
for a reduced pressure basis. Uncertainty is conditional on fixed mechanics,
geometry, basis, prior, and observation covariance. Not field-total uncertainty.
Signed pressure changes are allowed. No permeability or flow model is included.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GaussianPressurePosterior:
    mean_pa: np.ndarray
    covariance_pa2: np.ndarray
    predicted_mean_ns: np.ndarray
    predicted_signal_covariance_ns2: np.ndarray
    predicted_observation_covariance_ns2: np.ndarray
    residual_ns: np.ndarray
    prior_whitened_singular_values: np.ndarray
    numerical_rank: int
    likelihood_nullity: int

    @property
    def std_pa(self):
        return np.sqrt(np.maximum(np.diag(self.covariance_pa2), 0.0))

    @property
    def informative_modes(self):
        """Prior-whitened s > 1; depends on BOTH noise and prior scales."""
        return int(np.sum(self.prior_whitened_singular_values > 1.0))


def _spd(matrix, size, name):
    C = np.asarray(matrix, dtype=float)
    if C.shape != (size, size) or not np.isfinite(C).all():
        raise ValueError(f"{name} must be a finite {size} x {size} matrix.")
    scale = max(float(np.max(np.abs(C))), np.finfo(float).tiny)
    if np.max(np.abs(C - C.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric.")
    C = (C + C.T) / 2
    try:
        return C, np.linalg.cholesky(C)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite.") from exc


def linear_gaussian_posterior(design_ns_per_pa, data_ns, *, prior_mean_pa,
                              prior_covariance_pa2, noise_std_ns=None,
                              noise_covariance_ns2=None):
    """Closed-form update via prior/noise whitening and SVD.

    Supply exactly one of noise_std_ns (scalar or M-vector) and a full SPD noise
    covariance. A proper SPD Gaussian prior keeps null-space uncertainty finite
    without falsely declaring it measured. No hidden jitter/regularization.
    """
    H = np.asarray(design_ns_per_pa, dtype=float)
    d = np.asarray(data_ns, dtype=float)
    m0 = np.asarray(prior_mean_pa, dtype=float)
    if H.ndim != 2 or min(H.shape) == 0 or not np.isfinite(H).all():
        raise ValueError("Design matrix must be finite, nonempty and two-dimensional.")
    m, n = H.shape
    if d.shape != (m,) or m0.shape != (n,):
        raise ValueError("Data and prior mean shapes must match the design matrix.")
    if not np.isfinite(d).all() or not np.isfinite(m0).all():
        raise ValueError("Data and prior mean must be finite.")
    C0, L0 = _spd(prior_covariance_pa2, n, "prior covariance")
    if (noise_std_ns is None) == (noise_covariance_ns2 is None):
        raise ValueError("Supply exactly one noise model: standard deviation or covariance.")
    if noise_std_ns is not None:
        sd = np.asarray(noise_std_ns, dtype=float)
        if sd.ndim == 0:
            sd = np.full(m, sd.item())
        if sd.shape != (m,) or not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError("Noise standard deviations must be positive and match channels.")
        R = np.diag(sd ** 2)
        A = (H @ L0) / sd[:, None]
        residual0 = (d - H @ m0) / sd
    else:
        R, LR = _spd(noise_covariance_ns2, m, "noise covariance")
        A = np.linalg.solve(LR, H @ L0)
        residual0 = np.linalg.solve(LR, d - H @ m0)
    # Full right singular basis retains all unobserved coefficient directions.
    U, s, Vt = np.linalg.svd(A, full_matrices=(m < n))
    r = s.size
    weights = np.ones(n)
    weights[:r] = 1.0 / (1.0 + s ** 2)
    zmean = Vt[:r].T @ ((s / (1.0 + s ** 2)) * (U[:, :r].T @ residual0))
    posterior_factor = L0 @ (Vt.T * np.sqrt(weights))
    covariance = posterior_factor @ posterior_factor.T
    mean = m0 + L0 @ zmean
    signal_factor = H @ posterior_factor
    signal_cov = signal_factor @ signal_factor.T
    tolerance = np.finfo(float).eps * max(A.shape) * (s[0] if s.size else 0.0)
    rank = int(np.sum(s > tolerance))
    prediction = H @ mean
    return GaussianPressurePosterior(mean, covariance, prediction, signal_cov,
                                     signal_cov + R, d - prediction, s, rank, n - rank)
