#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fisher-information and local identifiability utilities for the AVANT
analytical-model optimal experimental design (OED) workflow.

This module is intentionally independent of repository paths, plotting,
command-line parsing, and PyDREAM.  It provides reusable numerical
building blocks for:

    - finite-difference Jacobians
    - parameter scaling
    - observation weighting
    - Fisher information matrices
    - regularized information matrices
    - local covariance / standard-error proxies
    - rank and condition-number diagnostics
    - station-subset design evaluation

Canonical physical parameter vector
-----------------------------------
    [a, b, h, theta_deg, x0_prime, y0_prime]

The Bayesian residual/noise parameter log10_sigma_strain is intentionally
not included in the physical OED parameter vector.

Conventions
-----------
Forward models should return an array shaped:

    (n_observations,)

for a specified observation design and parameter vector.

The Jacobian is defined as:

    J[i, j] = d y_i / d theta_j

The Fisher information matrix for independent Gaussian observations is:

    F = J^T W J

where W is a diagonal weight matrix or an equivalent 1-D weight vector.

Parameter and observation scaling are supported so that design metrics are
not dominated by units.  The caller should use scientifically justified
scales/noise levels rather than letting this module choose arbitrary
physical values.

No plotting or file I/O occurs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]
ForwardFunction = Callable[[np.ndarray], np.ndarray]


# ============================================================================
# Data containers
# ============================================================================


@dataclass(frozen=True)
class JacobianResult:
    """Finite-difference Jacobian and diagnostics."""

    parameters: np.ndarray
    baseline_output: np.ndarray
    jacobian: np.ndarray
    parameter_steps: np.ndarray


@dataclass(frozen=True)
class FisherResult:
    """Fisher-information and identifiability diagnostics."""

    fisher: np.ndarray
    regularized_fisher: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    log_determinant: float
    trace: float
    min_eigenvalue: float
    max_eigenvalue: float


@dataclass(frozen=True)
class DesignMetrics:
    """Scalar OED criteria for one candidate design."""

    d_optimality: float
    a_optimality: float
    e_optimality: float
    condition_number: float
    rank: int
    log_determinant: float
    trace_information: float
    min_eigenvalue: float
    max_eigenvalue: float


# ============================================================================
# Validation helpers
# ============================================================================


def _as_1d_float(
    values: ArrayLike,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            f"{name} must contain at least one value."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def _validate_positive_vector(
    values: ArrayLike,
    name: str,
) -> np.ndarray:
    array = _as_1d_float(
        values,
        name,
    )

    if np.any(array <= 0.0):
        raise ValueError(
            f"{name} must contain strictly positive values."
        )

    return array


def validate_parameter_bounds(
    parameter_names: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
) -> None:
    missing = [
        name
        for name in parameter_names
        if name not in bounds
    ]

    if missing:
        raise ValueError(
            "Missing bounds for parameters: "
            + ", ".join(missing)
        )

    for name in parameter_names:
        values = _as_1d_float(
            bounds[name],
            f"bounds[{name}]",
        )

        if values.size != 2:
            raise ValueError(
                f"bounds[{name}] must contain exactly two values."
            )

        if not values[1] > values[0]:
            raise ValueError(
                f"Upper bound must exceed lower bound for {name}."
            )


# ============================================================================
# Parameter scaling
# ============================================================================


def make_parameter_scales(
    parameter_names: Sequence[str],
    scales: Optional[Mapping[str, float]] = None,
    bounds: Optional[Mapping[str, Sequence[float]]] = None,
) -> np.ndarray:
    """
    Build positive parameter scaling values.

    Preferred behavior:
        1. Use explicitly supplied scales.
        2. Otherwise use prior/bounds widths if supplied.

    The bounds-width fallback is deterministic and unit-aware, but explicit
    scientific scales are preferred for publication OED studies.
    """

    names = list(
        parameter_names
    )

    if scales is not None:
        values = []

        for name in names:
            if name not in scales:
                raise ValueError(
                    f"Missing explicit parameter scale for {name}."
                )

            value = float(
                scales[name]
            )

            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Parameter scale for {name} must be positive."
                )

            values.append(value)

        return np.asarray(
            values,
            dtype=float,
        )

    if bounds is not None:
        validate_parameter_bounds(
            names,
            bounds,
        )

        return np.asarray(
            [
                float(bounds[name][1] - bounds[name][0])
                for name in names
            ],
            dtype=float,
        )

    return np.ones(
        len(names),
        dtype=float,
    )


def scale_jacobian_parameters(
    jacobian: np.ndarray,
    parameter_scales: ArrayLike,
) -> np.ndarray:
    """
    Convert a Jacobian into dimensionless parameter coordinates.

    If p_scaled = p / scale, then:

        d y / d p_scaled = J * scale
    """

    J = np.asarray(
        jacobian,
        dtype=float,
    )

    if J.ndim != 2:
        raise ValueError(
            "jacobian must be a 2-D array."
        )

    scales = _validate_positive_vector(
        parameter_scales,
        "parameter_scales",
    )

    if J.shape[1] != scales.size:
        raise ValueError(
            "Number of parameter scales does not match Jacobian columns."
        )

    return J * scales[np.newaxis, :]


# ============================================================================
# Observation weighting
# ============================================================================


def make_observation_weights(
    n_observations: int,
    sigma: Optional[Union[float, ArrayLike]] = None,
) -> np.ndarray:
    """
    Return diagonal Gaussian weights 1/sigma^2.

    sigma may be:
        - scalar: common uncertainty for all observations
        - vector: observation-specific uncertainty
        - None: unit variance weights
    """

    if n_observations <= 0:
        raise ValueError(
            "n_observations must be positive."
        )

    if sigma is None:
        sigma_vector = np.ones(
            n_observations,
            dtype=float,
        )

    elif np.isscalar(sigma):
        value = float(
            sigma
        )

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "Scalar sigma must be finite and positive."
            )

        sigma_vector = np.full(
            n_observations,
            value,
            dtype=float,
        )

    else:
        sigma_vector = _validate_positive_vector(
            sigma,
            "sigma",
        )

        if sigma_vector.size != n_observations:
            raise ValueError(
                "sigma vector length does not match the number "
                "of observations."
            )

    return 1.0 / np.square(
        sigma_vector
    )


def apply_observation_weights(
    jacobian: np.ndarray,
    weights: ArrayLike,
) -> np.ndarray:
    """Return W^(1/2) J for diagonal observation weights."""

    J = np.asarray(
        jacobian,
        dtype=float,
    )

    weight_vector = _validate_positive_vector(
        weights,
        "weights",
    )

    if J.ndim != 2:
        raise ValueError(
            "jacobian must be 2-D."
        )

    if J.shape[0] != weight_vector.size:
        raise ValueError(
            "Observation-weight length does not match Jacobian rows."
        )

    return np.sqrt(
        weight_vector
    )[:, np.newaxis] * J


# ============================================================================
# Finite-difference Jacobians
# ============================================================================


def finite_difference_jacobian(
    forward_function: ForwardFunction,
    parameters: ArrayLike,
    *,
    step_scales: Optional[ArrayLike] = None,
    relative_step: float = 1.0e-5,
    method: str = "central",
    bounds: Optional[Sequence[Sequence[float]]] = None,
) -> JacobianResult:
    """
    Compute a finite-difference Jacobian.

    Parameters
    ----------
    forward_function
        Callable receiving a 1-D parameter vector and returning a 1-D
        observation vector.

    parameters
        Baseline parameter vector.

    step_scales
        Optional absolute scales for each parameter. When omitted the
        perturbation is relative to max(abs(parameter), 1).

    relative_step
        Dimensionless finite-difference step.

    method
        "central" or "forward".

    bounds
        Optional parameter bounds. If supplied, perturbations are reflected
        into the feasible interval to avoid evaluating outside the design
        domain.
    """

    p = _as_1d_float(
        parameters,
        "parameters",
    )

    if relative_step <= 0.0:
        raise ValueError(
            "relative_step must be positive."
        )

    method = str(
        method
    ).lower().strip()

    if method not in {
        "central",
        "forward",
    }:
        raise ValueError(
            "method must be 'central' or 'forward'."
        )

    if step_scales is None:
        scales = np.maximum(
            np.abs(p),
            1.0,
        )
    else:
        scales = _validate_positive_vector(
            step_scales,
            "step_scales",
        )

        if scales.size != p.size:
            raise ValueError(
                "step_scales length does not match parameter vector."
            )

    steps = (
        relative_step
        * scales
    )

    if bounds is not None:
        bound_array = np.asarray(
            bounds,
            dtype=float,
        )

        if bound_array.shape != (
            p.size,
            2,
        ):
            raise ValueError(
                "bounds must have shape (n_parameters, 2)."
            )

        if not np.all(
            bound_array[:, 1]
            > bound_array[:, 0]
        ):
            raise ValueError(
                "Each parameter upper bound must exceed its lower bound."
            )

    baseline_output = _as_1d_float(
        forward_function(p),
        "forward output",
    )

    n_observations = (
        baseline_output.size
    )

    J = np.zeros(
        (
            n_observations,
            p.size,
        ),
        dtype=float,
    )

    for j in range(
        p.size
    ):
        step = float(
            steps[j]
        )

        if bounds is not None:
            lower = bound_array[j, 0]
            upper = bound_array[j, 1]

            plus = min(
                p[j] + step,
                upper,
            )

            minus = max(
                p[j] - step,
                lower,
            )

            if plus <= p[j] and minus >= p[j]:
                raise ValueError(
                    f"Parameter {j} has no feasible finite-difference "
                    "perturbation at the supplied baseline."
                )

            if method == "central":
                if plus > p[j] and minus < p[j]:
                    h_plus = plus - p[j]
                    h_minus = p[j] - minus

                    plus_vector = p.copy()
                    minus_vector = p.copy()

                    plus_vector[j] = plus
                    minus_vector[j] = minus

                    y_plus = _as_1d_float(
                        forward_function(plus_vector),
                        f"forward output (+ parameter {j})",
                    )

                    y_minus = _as_1d_float(
                        forward_function(minus_vector),
                        f"forward output (- parameter {j})",
                    )

                    if (
                        y_plus.size
                        != n_observations
                        or y_minus.size
                        != n_observations
                    ):
                        raise ValueError(
                            "Forward-model output size changed between "
                            "finite-difference evaluations."
                        )

                    if np.isclose(
                        h_plus + h_minus,
                        0.0,
                    ):
                        raise ValueError(
                            f"Degenerate finite-difference step for parameter {j}."
                        )

                    J[:, j] = (
                        y_plus - y_minus
                    ) / (
                        h_plus + h_minus
                    )

                    steps[j] = max(
                        h_plus,
                        h_minus,
                    )

                    continue

                # One-sided fallback at a bound.
                if plus > p[j]:
                    plus_vector = p.copy()
                    plus_vector[j] = plus

                    y_plus = _as_1d_float(
                        forward_function(plus_vector),
                        f"forward output (+ parameter {j})",
                    )

                    J[:, j] = (
                        y_plus - baseline_output
                    ) / (
                        plus - p[j]
                    )

                    steps[j] = (
                        plus - p[j]
                    )

                    continue

                minus_vector = p.copy()
                minus_vector[j] = minus

                y_minus = _as_1d_float(
                    forward_function(minus_vector),
                    f"forward output (- parameter {j})",
                )

                J[:, j] = (
                    baseline_output - y_minus
                ) / (
                    p[j] - minus
                )

                steps[j] = (
                    p[j] - minus
                )

                continue

            # Forward difference with bounds.
            if plus > p[j]:
                plus_vector = p.copy()
                plus_vector[j] = plus

                y_plus = _as_1d_float(
                    forward_function(plus_vector),
                    f"forward output (+ parameter {j})",
                )

                J[:, j] = (
                    y_plus - baseline_output
                ) / (
                    plus - p[j]
                )

                steps[j] = (
                    plus - p[j]
                )

            else:
                raise ValueError(
                    f"No feasible forward difference for parameter {j}."
                )

            continue

        # No bounds supplied.
        if method == "central":
            plus_vector = p.copy()
            minus_vector = p.copy()

            plus_vector[j] += step
            minus_vector[j] -= step

            y_plus = _as_1d_float(
                forward_function(plus_vector),
                f"forward output (+ parameter {j})",
            )

            y_minus = _as_1d_float(
                forward_function(minus_vector),
                f"forward output (- parameter {j})",
            )

            if (
                y_plus.size != n_observations
                or y_minus.size != n_observations
            ):
                raise ValueError(
                    "Forward-model output size changed between "
                    "finite-difference evaluations."
                )

            J[:, j] = (
                y_plus - y_minus
            ) / (
                2.0 * step
            )

        else:
            plus_vector = p.copy()
            plus_vector[j] += step

            y_plus = _as_1d_float(
                forward_function(plus_vector),
                f"forward output (+ parameter {j})",
            )

            if y_plus.size != n_observations:
                raise ValueError(
                    "Forward-model output size changed between "
                    "finite-difference evaluations."
                )

            J[:, j] = (
                y_plus - baseline_output
            ) / step

    if not np.all(
        np.isfinite(J)
    ):
        raise RuntimeError(
            "Finite-difference Jacobian contains non-finite values."
        )

    return JacobianResult(
        parameters=p.copy(),
        baseline_output=baseline_output,
        jacobian=J,
        parameter_steps=np.asarray(
            steps,
            dtype=float,
        ),
    )


# ============================================================================
# Fisher information
# ============================================================================


def fisher_information_matrix(
    jacobian: np.ndarray,
    *,
    observation_weights: Optional[ArrayLike] = None,
    parameter_scales: ArrayLike | None = None,
) -> np.ndarray:
    """
    Compute F = J^T W J.

    If parameter_scales are supplied, the Jacobian is transformed into
    dimensionless parameter coordinates before forming F.
    """

    J = np.asarray(
        jacobian,
        dtype=float,
    )

    if J.ndim != 2:
        raise ValueError(
            "jacobian must be 2-D."
        )

    if not np.all(
        np.isfinite(J)
    ):
        raise ValueError(
            "jacobian contains non-finite values."
        )

    if parameter_scales is not None:
        J = scale_jacobian_parameters(
            J,
            parameter_scales,
        )

    if observation_weights is None:
        weights = np.ones(
            J.shape[0],
            dtype=float,
        )
    else:
        weights = _validate_positive_vector(
            observation_weights,
            "observation_weights",
        )

    if weights.size != J.shape[0]:
        raise ValueError(
            "observation_weights length does not match Jacobian rows."
        )

    weighted_J = apply_observation_weights(
        J,
        weights,
    )

    F = (
        weighted_J.T
        @ weighted_J
    )

    F = 0.5 * (
        F + F.T
    )

    if not np.all(
        np.isfinite(F)
    ):
        raise RuntimeError(
            "Fisher information matrix contains non-finite values."
        )

    return F


def regularize_information_matrix(
    fisher: np.ndarray,
    regularization: float = 1.0e-10,
) -> np.ndarray:
    """
    Add diagonal Tikhonov regularization.

    regularization is applied to the diagonal of a dimensionless Fisher
    matrix. Use the smallest scientifically defensible value necessary for
    numerical stability.
    """

    F = np.asarray(
        fisher,
        dtype=float,
    )

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(
            "fisher must be a square matrix."
        )

    if regularization < 0.0:
        raise ValueError(
            "regularization must be non-negative."
        )

    regularized = (
        F
        + regularization
        * np.eye(
            F.shape[0]
        )
    )

    return 0.5 * (
        regularized
        + regularized.T
    )


def fisher_diagnostics(
    fisher: np.ndarray,
    *,
    regularization: float = 1.0e-10,
    rank_tolerance: float = 1.0e-10,
) -> FisherResult:
    """
    Calculate eigenvalues, rank, condition number, covariance proxy,
    standard-error proxy, and log determinant.
    """

    F = np.asarray(
        fisher,
        dtype=float,
    )

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(
            "fisher must be square."
        )

    regularized = regularize_information_matrix(
        F,
        regularization,
    )

    eigenvalues = np.linalg.eigvalsh(
        regularized
    )

    max_eigenvalue = float(
        np.max(eigenvalues)
    )

    min_eigenvalue = float(
        np.min(eigenvalues)
    )

    if max_eigenvalue <= 0.0:
        rank = 0
    else:
        cutoff = (
            rank_tolerance
            * max_eigenvalue
        )

        rank = int(
            np.sum(
                eigenvalues
                > cutoff
            )
        )

    if min_eigenvalue > 0.0:
        condition_number = float(
            max_eigenvalue
            / min_eigenvalue
        )
    else:
        condition_number = float(
            np.inf
        )

    sign, logdet = np.linalg.slogdet(
        regularized
    )

    if sign <= 0.0:
        log_determinant = float(
            -np.inf
        )
    else:
        log_determinant = float(
            logdet
        )

    try:
        covariance = np.linalg.pinv(
            regularized
        )
    except np.linalg.LinAlgError:
        covariance = np.full_like(
            regularized,
            np.nan,
        )

    diagonal = np.diag(
        covariance
    )

    standard_errors = np.sqrt(
        np.maximum(
            diagonal,
            0.0,
        )
    )

    return FisherResult(
        fisher=F.copy(),
        regularized_fisher=regularized,
        covariance=covariance,
        standard_errors=standard_errors,
        eigenvalues=eigenvalues,
        rank=rank,
        condition_number=condition_number,
        log_determinant=log_determinant,
        trace=float(
            np.trace(
                regularized
            )
        ),
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
    )


# ============================================================================
# Design metrics
# ============================================================================


def calculate_design_metrics(
    fisher_result: FisherResult,
) -> DesignMetrics:
    """
    Return standard local OED criteria.

    D-optimality:
        log det(F)

    A-optimality:
        trace(F^-1)

    E-optimality:
        minimum eigenvalue

    These are computed on the regularized/scaled information matrix.
    """

    F = fisher_result.regularized_fisher
    covariance = fisher_result.covariance

    return DesignMetrics(
        d_optimality=float(
            fisher_result.log_determinant
        ),
        a_optimality=float(
            np.trace(
                covariance
            )
        ),
        e_optimality=float(
            fisher_result.min_eigenvalue
        ),
        condition_number=float(
            fisher_result.condition_number
        ),
        rank=int(
            fisher_result.rank
        ),
        log_determinant=float(
            fisher_result.log_determinant
        ),
        trace_information=float(
            fisher_result.trace
        ),
        min_eigenvalue=float(
            fisher_result.min_eigenvalue
        ),
        max_eigenvalue=float(
            fisher_result.max_eigenvalue
        ),
    )


# ============================================================================
# Station-subset utilities
# ============================================================================


def canonical_channel_indices(
    station_count: int,
    station_indices: Sequence[int],
    n_components: int = 4,
) -> np.ndarray:
    """
    Return row indices for component-major channel ordering:

        eXX_S01...S08,
        eYY_S01...S08,
        eZZ_S01...S08,
        eXY_S01...S08

    Each station selection therefore retains all four strain components.
    """

    if station_count <= 0:
        raise ValueError(
            "station_count must be positive."
        )

    indices = [
        int(index)
        for index in station_indices
    ]

    if not indices:
        raise ValueError(
            "At least one station must be selected."
        )

    if any(
        index < 0
        or index >= station_count
        for index in indices
    ):
        raise IndexError(
            "station_indices contains an invalid station index."
        )

    rows = []

    for component_index in range(
        n_components
    ):
        offset = (
            component_index
            * station_count
        )

        rows.extend(
            offset + index
            for index in indices
        )

    return np.asarray(
        rows,
        dtype=int,
    )


def subset_jacobian_by_stations(
    jacobian: np.ndarray,
    station_count: int,
    station_indices: Sequence[int],
    n_components: int = 4,
) -> np.ndarray:
    """Restrict a full Jacobian to a candidate station subset."""

    J = np.asarray(
        jacobian,
        dtype=float,
    )

    indices = canonical_channel_indices(
        station_count,
        station_indices,
        n_components,
    )

    if J.shape[0] <= np.max(indices):
        raise ValueError(
            "Jacobian has fewer observation rows than required by the "
            "requested station subset."
        )

    return J[
        indices,
        :,
    ]


def evaluate_station_subset(
    full_jacobian: np.ndarray,
    *,
    station_count: int,
    station_indices: Sequence[int],
    observation_sigma: Optional[Union[float, ArrayLike]] = None,
    parameter_scales: ArrayLike | None = None,
    regularization: float = 1.0e-10,
    rank_tolerance: float = 1.0e-10,
    n_components: int = 4,
) -> Tuple[np.ndarray, FisherResult, DesignMetrics]:
    """
    Evaluate one station subset using a precomputed full Jacobian.
    """

    J_subset = subset_jacobian_by_stations(
        full_jacobian,
        station_count,
        station_indices,
        n_components,
    )

    weights = make_observation_weights(
        J_subset.shape[0],
        observation_sigma,
    )

    F = fisher_information_matrix(
        J_subset,
        observation_weights=weights,
        parameter_scales=parameter_scales,
    )

    result = fisher_diagnostics(
        F,
        regularization=regularization,
        rank_tolerance=rank_tolerance,
    )

    metrics = calculate_design_metrics(
        result
    )

    return (
        J_subset,
        result,
        metrics,
    )


def enumerate_station_subsets(
    station_count: int,
    subset_size: Optional[int] = None,
) -> Iterable[tuple[int, ...]]:
    """
    Enumerate candidate station subsets.

    If subset_size is None, every non-empty subset is returned.
    """

    if station_count <= 0:
        raise ValueError(
            "station_count must be positive."
        )

    if subset_size is None:
        sizes = range(
            1,
            station_count + 1,
        )
    else:
        if not (
            1 <= subset_size <= station_count
        ):
            raise ValueError(
                "subset_size must satisfy "
                "1 <= subset_size <= station_count."
            )

        sizes = [subset_size]

    for size in sizes:
        yield from combinations(
            range(
                station_count
            ),
            size,
        )


# ============================================================================
# Convenience wrapper
# ============================================================================


def compute_local_oed(
    forward_function: ForwardFunction,
    parameters: ArrayLike,
    *,
    parameter_scales: ArrayLike | None = None,
    observation_sigma: Optional[Union[float, ArrayLike]] = None,
    finite_difference_step: float = 1.0e-5,
    finite_difference_method: str = "central",
    parameter_bounds: Optional[Sequence[Sequence[float]]] = None,
    regularization: float = 1.0e-10,
    rank_tolerance: float = 1.0e-10,
) -> Tuple[JacobianResult, FisherResult, DesignMetrics]:
    """
    Compute a complete local OED diagnostic for one parameter point.
    """

    jacobian_result = finite_difference_jacobian(
        forward_function,
        parameters,
        relative_step=finite_difference_step,
        method=finite_difference_method,
        bounds=parameter_bounds,
    )

    if parameter_scales is None:
        scales = np.maximum(
            np.abs(
                jacobian_result.parameters
            ),
            1.0,
        )
    else:
        scales = _validate_positive_vector(
            parameter_scales,
            "parameter_scales",
        )

    weights = make_observation_weights(
        jacobian_result.jacobian.shape[0],
        observation_sigma,
    )

    F = fisher_information_matrix(
        jacobian_result.jacobian,
        observation_weights=weights,
        parameter_scales=scales,
    )

    fisher_result = fisher_diagnostics(
        F,
        regularization=regularization,
        rank_tolerance=rank_tolerance,
    )

    metrics = calculate_design_metrics(
        fisher_result
    )

    return (
        jacobian_result,
        fisher_result,
        metrics,
    )
