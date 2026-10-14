"""Geometry-dependent, pressure-marginalized AVANT likelihood (single snapshot).

The caller supplies VERIFIED positive-downward receiver depths, strain changes in
nanostrain in the exact channel order, and explicit pressure/noise priors.
No COMSOL coordinate conversion, geometry prior, or DREAM sampler is hidden here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from avant_model.model.pressure_superposition import (
    Cuboid, ElasticProperties, build_pressure_kernel, coarse_block_basis,
)
from avant_model.inversion.marginalized_pressure_likelihood import (
    marginalized_pressure_likelihood,
)


@dataclass(frozen=True)
class GeometryPressureEvaluation:
    geometry: Cuboid
    channels: tuple[str, ...]
    design_ns_per_pa: np.ndarray
    log_likelihood: float
    conditional_mean_pa: np.ndarray
    conditional_covariance_pa2: np.ndarray
    predicted_mean_ns: np.ndarray
    residual_ns: np.ndarray


def evaluate_geometry_pressure(
    geometry: Cuboid,
    elastic: ElasticProperties,
    *, x_prime, y_prime, z, station_names,
    observed_ns, observed_channels,
    fine_splits=(4, 4, 1), pressure_splits=(2, 2, 1),
    components=("eXX", "eYY", "eXY", "eZZ"),
    prior_mean_pa, prior_covariance_pa2, noise_covariance_ns2,
) -> GeometryPressureEvaluation:
    """Rebuild G for proposed geometry and evaluate exact Gaussian marginal likelihood.

    The supplied observed_channels MUST match the kernel order exactly. To use
    only training stations, pass only their coordinates/names and matching data.
    This function deliberately does not reorder or silently drop channels.
    """
    kernel = build_pressure_kernel(
        geometry, elastic, x_prime=x_prime, y_prime=y_prime, z=z,
        station_names=station_names, splits=fine_splits, components=components,
    )
    channels = tuple(observed_channels)
    if channels != kernel.channels:
        raise ValueError("Observed channel order differs from computed kernel channels")
    basis = np.asarray(coarse_block_basis(fine_splits, pressure_splits), dtype=float)
    if basis.ndim != 2 or basis.shape[0] != kernel.G.shape[1]:
        raise ValueError("Pressure basis dimensions are incompatible with kernel")
    H = kernel.G @ basis
    result = marginalized_pressure_likelihood(
        H, observed_ns, prior_mean_pa=prior_mean_pa,
        prior_covariance_pa2=prior_covariance_pa2,
        noise_covariance_ns2=noise_covariance_ns2,
    )
    return GeometryPressureEvaluation(
        geometry=geometry, channels=kernel.channels, design_ns_per_pa=H,
        log_likelihood=result.log_likelihood,
        conditional_mean_pa=result.conditional_mean_pa,
        conditional_covariance_pa2=result.conditional_covariance_pa2,
        predicted_mean_ns=result.predicted_mean_ns,
        residual_ns=result.residual_ns,
    )
