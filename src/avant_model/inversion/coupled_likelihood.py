"""Reusable, normalized Gaussian strain likelihood for homogeneous K and Ss.

The same callable can later be supplied to PyDREAM. A pressure likelihood
requires a separate, physically located pressure-gauge operator.
"""
import numpy as np
from avant_model.inversion.coupled_parameters import HydraulicBounds


def gaussian_log_likelihood(observed, predicted, sigma_ns):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sigma = float(sigma_ns)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("Observed/predicted shape mismatch or empty residual")
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(observed).all():
        raise ValueError("Invalid observations or noise scale")
    if not np.isfinite(predicted).all():
        return -np.inf
    residual = (observed - predicted) / sigma
    return float(-0.5 * np.sum(residual**2)
                 - observed.size * np.log(sigma)
                 - 0.5 * observed.size * np.log(2.0 * np.pi))


class CoupledStrainLikelihood:
    def __init__(self, model, observations, bounds=None):
        self.model = model
        self.observations = observations
        self.bounds = bounds if bounds is not None else HydraulicBounds()
        if tuple(model.channels) != tuple(observations.channels):
            raise ValueError("Forward model and observations have different channel ordering")

    def predict(self, theta):
        K, Ss = self.bounds.to_physical(theta)
        result = self.model.predict(K, Ss, self.observations.times_h)
        if tuple(result.channels) != self.observations.channels:
            raise ValueError("Predicted strain channels changed order")
        if not np.allclose(result.times_h, self.observations.times_h, rtol=0, atol=1e-10):
            raise ValueError("Predicted and observed times do not agree")
        if np.asarray(result.strain_ns).shape != self.observations.observed_ns.shape:
            raise ValueError("Predicted and observed strain arrays have different shapes")
        return np.asarray(result.strain_ns, dtype=float)

    def __call__(self, theta):
        if not self.bounds.contains(theta):
            return -np.inf
        try:
            pred = self.predict(theta)
            mask = self.observations.training_mask
            return gaussian_log_likelihood(self.observations.observed_ns[:, mask],
                                           pred[:, mask], self.observations.sigma_ns)
        except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
            return -np.inf
