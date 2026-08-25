#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian optimal experimental design utilities for the AVANT analytical model.

This module provides the Bayesian layer above the existing OED components:

    geometry_uncertainty.py
    fisher_information.py
    design_metrics.py
    robust_design.py
    sensor_placement.py

Two Bayesian design levels are kept scientifically distinct.

1. Posterior-informed robust OED
--------------------------------
For a candidate design d and posterior geometry samples theta ~ p(theta|y),

    U(d) = E_theta[u(d, theta)]

where u may be D-, A-, or E-optimality. Conservative posterior quantiles,
worst-case utility, and utility variability are also retained.

This is a posterior-weighted robust design calculation. It is useful for
adaptive sensor placement after existing observations have constrained the
body geometry.

2. Expected information gain (EIG)
----------------------------------
For a future design d,

    EIG(d) = E_{theta,y_d}[
        log p(y_d | theta,d)
        - log p(y_d | d)
    ]

which is equivalent to the expected Kullback-Leibler information gain from
prior/current-posterior uncertainty to the updated distribution after a
future observation.

This module implements a Monte Carlo EIG estimator for Gaussian observation
noise. The estimator is intentionally modular and computationally expensive;
large production calculations belong on Palmetto.

Canonical physical state
------------------------
    [a, b, h, theta_deg, x0_prime, y0_prime]

The nuisance parameter log10_sigma_strain is not a physical geometry
parameter. It may, however, be used to define the predictive observation
noise in Bayesian EIG calculations when scientifically appropriate.

Python compatibility
--------------------
Python 3.9 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from avant_model.oed.design_metrics import (
    DEFAULT_PRIMARY_METRIC,
    metric_direction,
)
from avant_model.oed.geometry_uncertainty import (
    GEOMETRY_PARAMETER_NAMES,
    load_posterior_scenarios_from_run,
    posterior_geometry_scenarios,
)
from avant_model.oed.robust_design import (
    robust_ranking_from_scenario_table,
    weighted_quantile,
    weighted_robust_summary,
)


# ============================================================================
# Type aliases / data containers
# ============================================================================

BayesianForwardFunction = Callable[
    [Mapping[str, float]],
    np.ndarray,
]


@dataclass(frozen=True)
class BayesianDesignSummary:
    """Posterior-informed robust utility for one design."""

    design_id: str
    design_size: int
    metric: str
    higher_is_better: bool
    expected_utility: float
    median_utility: float
    conservative_utility: float
    worst_case_utility: float
    utility_std: float
    utility_cv: float
    effective_scenario_count: float
    n_scenarios: int


@dataclass(frozen=True)
class ExpectedInformationGainResult:
    """Monte Carlo expected-information-gain estimate."""

    design_id: str
    design_size: int
    eig_nats: float
    eig_bits: float
    standard_error_nats: float
    n_outer_samples: int
    n_inner_samples: int
    observation_dimension: int
    noise_model: str


# ============================================================================
# Numerical helpers
# ============================================================================


def _logsumexp(
    values: np.ndarray,
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Stable log(sum(exp(values))) without requiring scipy.special.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    maximum = np.max(
        array,
        axis=axis,
        keepdims=True,
    )

    shifted = array - maximum

    result = maximum + np.log(
        np.sum(
            np.exp(
                shifted
            ),
            axis=axis,
            keepdims=True,
        )
    )

    if axis is not None:
        result = np.squeeze(
            result,
            axis=axis,
        )

    return result


def _normalize_weights(
    weights: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            "At least one weight is required."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "Weights contain non-finite values."
        )

    if np.any(
        array < 0.0
    ):
        raise ValueError(
            "Weights must be non-negative."
        )

    total = float(
        np.sum(
            array
        )
    )

    if total <= 0.0:
        raise ValueError(
            "Weights must have positive total mass."
        )

    return array / total


def _geometry_matrix_from_dataframe(
    scenarios: pd.DataFrame,
) -> np.ndarray:
    missing = [
        name
        for name in GEOMETRY_PARAMETER_NAMES
        if name not in scenarios.columns
    ]

    if missing:
        raise ValueError(
            "Scenario table is missing geometry parameters: "
            + ", ".join(missing)
        )

    matrix = scenarios[
        list(
            GEOMETRY_PARAMETER_NAMES
        )
    ].to_numpy(
        dtype=float
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Geometry scenario matrix must be 2-D."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "Geometry scenario matrix contains non-finite values."
        )

    return matrix


def geometry_mapping(
    parameter_vector: Sequence[float],
) -> Dict[str, float]:
    values = np.asarray(
        parameter_vector,
        dtype=float,
    ).reshape(-1)

    if values.size != len(
        GEOMETRY_PARAMETER_NAMES
    ):
        raise ValueError(
            "Geometry vector must contain {0} parameters.".format(
                len(
                    GEOMETRY_PARAMETER_NAMES
                )
            )
        )

    return {
        name: float(
            value
        )
        for name, value in zip(
            GEOMETRY_PARAMETER_NAMES,
            values,
        )
    }


# ============================================================================
# Posterior scenario loading
# ============================================================================


def posterior_scenarios_from_samples(
    posterior_samples: np.ndarray,
    parameter_names: Sequence[str],
    *,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Convert retained Bayesian samples into equally weighted geometry scenarios.

    Extra posterior columns, including log10_sigma_strain, are ignored.
    """

    return posterior_geometry_scenarios(
        posterior_samples=posterior_samples,
        parameter_names=parameter_names,
        n_samples=n_samples,
        seed=seed,
        source="bayesian_posterior",
    )


def posterior_scenarios_from_run(
    run_dir,
    *,
    n_samples: Optional[int] = None,
    burn_in_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load geometry scenarios from the production Bayesian run directory.
    """

    return load_posterior_scenarios_from_run(
        run_dir=run_dir,
        n_samples=n_samples,
        burn_in_fraction=burn_in_fraction,
        seed=seed,
    )


# ============================================================================
# Posterior-informed robust design
# ============================================================================


def summarize_posterior_design_utility(
    scenario_table: pd.DataFrame,
    *,
    metric: str = DEFAULT_PRIMARY_METRIC,
    conservative_quantile: float = 0.10,
) -> pd.DataFrame:
    """
    Rank candidate designs using posterior-weighted OED utility.

    Expected input is a long table with at least:

        design_id
        design_size
        scenario_weight
        <metric>

    Such a table is produced naturally by robust_design.py after evaluating
    candidate designs across posterior geometry scenarios.
    """

    return robust_ranking_from_scenario_table(
        scenario_table,
        metric=metric,
        conservative_quantile=conservative_quantile,
        ranking="conservative",
    )


def bayesian_design_summary(
    design_id: str,
    design_size: int,
    utilities: Sequence[float],
    scenario_weights: Sequence[float],
    *,
    metric: str = DEFAULT_PRIMARY_METRIC,
    conservative_quantile: float = 0.10,
) -> BayesianDesignSummary:
    """
    Summarize posterior-weighted utility for one candidate design.
    """

    values = np.asarray(
        utilities,
        dtype=float,
    ).reshape(-1)

    weights = _normalize_weights(
        scenario_weights
    )

    if values.size != weights.size:
        raise ValueError(
            "utilities and scenario_weights must have equal lengths."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "utilities contain non-finite values."
        )

    if not (
        0.0
        < conservative_quantile
        < 0.5
    ):
        raise ValueError(
            "conservative_quantile must satisfy 0 < q < 0.5."
        )

    summary = weighted_robust_summary(
        values,
        weights,
        metric=metric,
    )

    higher = metric_direction(
        metric
    )

    if higher:
        conservative = weighted_quantile(
            values,
            weights,
            conservative_quantile,
        )

        worst = float(
            np.min(
                values
            )
        )

    else:
        conservative = weighted_quantile(
            values,
            weights,
            1.0
            - conservative_quantile,
        )

        worst = float(
            np.max(
                values
            )
        )

    return BayesianDesignSummary(
        design_id=str(
            design_id
        ),
        design_size=int(
            design_size
        ),
        metric=str(
            metric
        ),
        higher_is_better=bool(
            higher
        ),
        expected_utility=float(
            summary[
                "weighted_mean"
            ]
        ),
        median_utility=float(
            summary[
                "weighted_median"
            ]
        ),
        conservative_utility=float(
            conservative
        ),
        worst_case_utility=float(
            worst
        ),
        utility_std=float(
            summary[
                "weighted_std"
            ]
        ),
        utility_cv=float(
            summary[
                "coefficient_of_variation"
            ]
        ),
        effective_scenario_count=float(
            summary[
                "effective_scenario_count"
            ]
        ),
        n_scenarios=int(
            summary[
                "n_scenarios"
            ]
        ),
    )


# ============================================================================
# Gaussian observation model
# ============================================================================


def observation_sigma_vector(
    sigma: Sequence[float],
    observation_dimension: int,
) -> np.ndarray:
    """
    Validate/expand scalar or vector observation uncertainty.
    """

    if np.isscalar(
        sigma
    ):
        value = float(
            sigma
        )

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "Observation sigma must be finite and positive."
            )

        return np.full(
            observation_dimension,
            value,
            dtype=float,
        )

    vector = np.asarray(
        sigma,
        dtype=float,
    ).reshape(-1)

    if vector.size != observation_dimension:
        raise ValueError(
            "Observation sigma vector length does not match "
            "the design observation dimension."
        )

    if not np.all(
        np.isfinite(vector)
    ) or np.any(
        vector <= 0.0
    ):
        raise ValueError(
            "Observation sigma values must be finite and positive."
        )

    return vector


def gaussian_log_likelihood(
    observation: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Independent Gaussian log likelihood.
    """

    observation = np.asarray(
        observation,
        dtype=float,
    ).reshape(-1)

    prediction = np.asarray(
        prediction,
        dtype=float,
    ).reshape(-1)

    sigma = np.asarray(
        sigma,
        dtype=float,
    ).reshape(-1)

    if not (
        observation.size
        == prediction.size
        == sigma.size
    ):
        raise ValueError(
            "Observation, prediction, and sigma dimensions must agree."
        )

    residual = (
        observation
        - prediction
    )

    return float(
        -0.5
        * np.sum(
            np.square(
                residual
                / sigma
            )
            + np.log(
                2.0
                * np.pi
                * np.square(
                    sigma
                )
            )
        )
    )


# ============================================================================
# Forward prediction cache
# ============================================================================


def evaluate_design_predictions(
    scenarios: pd.DataFrame,
    forward_function: BayesianForwardFunction,
) -> np.ndarray:
    """
    Evaluate one candidate design for every geometry scenario.

    The supplied forward function already represents the candidate design.
    It receives a geometry mapping and returns the selected future
    observations as a 1-D vector.
    """

    geometry_matrix = (
        _geometry_matrix_from_dataframe(
            scenarios
        )
    )

    predictions = []

    expected_dimension = None

    for row in geometry_matrix:
        prediction = np.asarray(
            forward_function(
                geometry_mapping(
                    row
                )
            ),
            dtype=float,
        ).reshape(-1)

        if prediction.size == 0:
            raise ValueError(
                "Forward design prediction cannot be empty."
            )

        if not np.all(
            np.isfinite(
                prediction
            )
        ):
            raise ValueError(
                "Forward design prediction contains non-finite values."
            )

        if expected_dimension is None:
            expected_dimension = int(
                prediction.size
            )

        elif prediction.size != (
            expected_dimension
        ):
            raise ValueError(
                "Forward design prediction dimension changed between "
                "geometry scenarios."
            )

        predictions.append(
            prediction
        )

    return np.vstack(
        predictions
    )


# ============================================================================
# Expected information gain
# ============================================================================


def estimate_expected_information_gain(
    design_id: str,
    design_size: int,
    scenarios: pd.DataFrame,
    forward_function: BayesianForwardFunction,
    *,
    observation_sigma,
    n_outer_samples: Optional[int] = None,
    n_inner_samples: Optional[int] = None,
    seed: int = 42,
) -> ExpectedInformationGainResult:
    """
    Estimate Bayesian expected information gain by nested Monte Carlo.

    Current uncertainty represented by ``scenarios`` is treated as the
    design prior. This may be:
        - the broad pre-data geometry prior, or
        - the current Bayesian posterior for adaptive design.

    For each outer sample theta_i:
        1. Generate a synthetic future observation
               y_i ~ p(y | theta_i, d)
        2. Evaluate
               log p(y_i | theta_i,d)
        3. Approximate the evidence
               p(y_i | d)
           using inner uncertainty samples.
        4. Accumulate
               log p(y_i | theta_i,d) - log p(y_i | d)

    The returned EIG is in both nats and bits.

    Notes
    -----
    This is computationally expensive. The prediction matrix is cached once
    for the supplied design, but likelihood comparisons still scale roughly
    as O(N_outer * N_inner * n_observations).

    For final high-resolution spatial OED, run this stage on Palmetto.
    """

    required = {
        "weight",
        *GEOMETRY_PARAMETER_NAMES,
    }

    missing = (
        required
        - set(
            scenarios.columns
        )
    )

    if missing:
        raise ValueError(
            "Bayesian scenario table is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if scenarios.empty:
        raise ValueError(
            "Bayesian scenario table is empty."
        )

    weights = _normalize_weights(
        scenarios[
            "weight"
        ].to_numpy(
            dtype=float
        )
    )

    predictions = evaluate_design_predictions(
        scenarios,
        forward_function,
    )

    n_available = predictions.shape[
        0
    ]

    observation_dimension = predictions.shape[
        1
    ]

    sigma = observation_sigma_vector(
        observation_sigma,
        observation_dimension,
    )

    if n_outer_samples is None:
        n_outer = n_available
    else:
        n_outer = int(
            n_outer_samples
        )

        if n_outer <= 0:
            raise ValueError(
                "n_outer_samples must be positive."
            )

    if n_inner_samples is None:
        n_inner = n_available
    else:
        n_inner = int(
            n_inner_samples
        )

        if n_inner <= 0:
            raise ValueError(
                "n_inner_samples must be positive."
            )

    rng = np.random.default_rng(
        seed
    )

    outer_indices = rng.choice(
        n_available,
        size=n_outer,
        replace=True,
        p=weights,
    )

    inner_indices = rng.choice(
        n_available,
        size=n_inner,
        replace=True,
        p=weights,
    )

    inner_predictions = predictions[
        inner_indices
    ]

    # Because inner samples are drawn according to the scenario weights,
    # the Monte Carlo evidence estimator is the simple arithmetic average.
    log_information_gains = np.zeros(
        n_outer,
        dtype=float,
    )

    log_n_inner = float(
        np.log(
            n_inner
        )
    )

    log_normalization = (
        -0.5
        * np.sum(
            np.log(
                2.0
                * np.pi
                * np.square(
                    sigma
                )
            )
        )
    )

    for outer_position, scenario_index in enumerate(
        outer_indices
    ):
        true_prediction = predictions[
            scenario_index
        ]

        synthetic_observation = (
            true_prediction
            + rng.normal(
                loc=0.0,
                scale=sigma,
                size=observation_dimension,
            )
        )

        true_residual = (
            synthetic_observation
            - true_prediction
        )

        log_likelihood_true = float(
            log_normalization
            - 0.5
            * np.sum(
                np.square(
                    true_residual
                    / sigma
                )
            )
        )

        residuals = (
            synthetic_observation[
                np.newaxis,
                :
            ]
            - inner_predictions
        )

        inner_log_likelihoods = (
            log_normalization
            - 0.5
            * np.sum(
                np.square(
                    residuals
                    / sigma[
                        np.newaxis,
                        :
                    ]
                ),
                axis=1,
            )
        )

        log_evidence = float(
            _logsumexp(
                inner_log_likelihoods,
                axis=0,
            )
            - log_n_inner
        )

        log_information_gains[
            outer_position
        ] = (
            log_likelihood_true
            - log_evidence
        )

    eig_nats = float(
        np.mean(
            log_information_gains
        )
    )

    if n_outer > 1:
        standard_error = float(
            np.std(
                log_information_gains,
                ddof=1,
            )
            / np.sqrt(
                n_outer
            )
        )
    else:
        standard_error = float(
            np.nan
        )

    eig_bits = float(
        eig_nats
        / np.log(
            2.0
        )
    )

    return ExpectedInformationGainResult(
        design_id=str(
            design_id
        ),
        design_size=int(
            design_size
        ),
        eig_nats=eig_nats,
        eig_bits=eig_bits,
        standard_error_nats=standard_error,
        n_outer_samples=int(
            n_outer
        ),
        n_inner_samples=int(
            n_inner
        ),
        observation_dimension=int(
            observation_dimension
        ),
        noise_model="independent_gaussian",
    )


# ============================================================================
# EIG ranking across candidate designs
# ============================================================================


def rank_expected_information_gain(
    results: Sequence[
        ExpectedInformationGainResult
    ],
) -> pd.DataFrame:
    """
    Rank candidate designs by expected information gain.
    """

    if not results:
        raise ValueError(
            "At least one EIG result is required."
        )

    records = []

    for result in results:
        records.append(
            {
                "design_id": str(
                    result.design_id
                ),
                "design_size": int(
                    result.design_size
                ),
                "eig_nats": float(
                    result.eig_nats
                ),
                "eig_bits": float(
                    result.eig_bits
                ),
                "standard_error_nats": float(
                    result.standard_error_nats
                ),
                "n_outer_samples": int(
                    result.n_outer_samples
                ),
                "n_inner_samples": int(
                    result.n_inner_samples
                ),
                "observation_dimension": int(
                    result.observation_dimension
                ),
                "noise_model": str(
                    result.noise_model
                ),
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    if dataframe[
        "design_id"
    ].duplicated().any():
        raise ValueError(
            "EIG design IDs must be unique."
        )

    dataframe = dataframe.sort_values(
        by=[
            "eig_nats",
            "design_size",
            "design_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    dataframe[
        "eig_rank"
    ] = np.arange(
        1,
        len(
            dataframe
        )
        + 1,
        dtype=int,
    )

    return dataframe


# ============================================================================
# Adaptive-design comparison
# ============================================================================


def compare_prior_and_posterior_designs(
    prior_design_table: pd.DataFrame,
    posterior_design_table: pd.DataFrame,
    *,
    score_column: str = "primary_conservative",
) -> pd.DataFrame:
    """
    Compare pre-data robust OED with posterior-informed adaptive OED.

    This helps quantify whether learning from the existing strain data changes
    which sensor network is preferred.
    """

    required = {
        "design_id",
        "design_size",
        score_column,
    }

    for name, table in (
        (
            "prior_design_table",
            prior_design_table,
        ),
        (
            "posterior_design_table",
            posterior_design_table,
        ),
    ):
        missing = (
            required
            - set(
                table.columns
            )
        )

        if missing:
            raise ValueError(
                "{0} is missing columns: {1}".format(
                    name,
                    ", ".join(
                        sorted(
                            missing
                        )
                    ),
                )
            )

    prior = prior_design_table[
        [
            "design_id",
            "design_size",
            score_column,
        ]
    ].copy()

    posterior = posterior_design_table[
        [
            "design_id",
            "design_size",
            score_column,
        ]
    ].copy()

    prior = prior.rename(
        columns={
            score_column: (
                "prior_{0}".format(
                    score_column
                )
            )
        }
    )

    posterior = posterior.rename(
        columns={
            score_column: (
                "posterior_{0}".format(
                    score_column
                )
            )
        }
    )

    merged = prior.merge(
        posterior,
        on=[
            "design_id",
            "design_size",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged[
        "adaptive_score_change"
    ] = (
        merged[
            "posterior_{0}".format(
                score_column
            )
        ]
        - merged[
            "prior_{0}".format(
                score_column
            )
        ]
    )

    return merged


# ============================================================================
# Posterior sigma extraction
# ============================================================================


def posterior_sigma_samples(
    posterior_samples: np.ndarray,
    parameter_names: Sequence[str],
) -> np.ndarray:
    """
    Convert posterior log10_sigma_strain samples to physical nstrain sigma.

    This utility does not make sigma a geometry parameter. It only provides
    the observation-noise scale for predictive/EIG calculations.
    """

    samples = np.asarray(
        posterior_samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "posterior_samples must be 2-D."
        )

    names = [
        str(name)
        for name in parameter_names
    ]

    if len(
        names
    ) != samples.shape[1]:
        raise ValueError(
            "parameter_names length does not match posterior columns."
        )

    noise_name = (
        "log10_sigma_strain"
    )

    if noise_name not in names:
        raise ValueError(
            "Posterior does not contain log10_sigma_strain."
        )

    index = names.index(
        noise_name
    )

    sigma = (
        10.0
        ** samples[
            :,
            index
        ]
    )

    if not np.all(
        np.isfinite(
            sigma
        )
    ) or np.any(
        sigma <= 0.0
    ):
        raise ValueError(
            "Converted posterior sigma contains invalid values."
        )

    return sigma


def representative_posterior_sigma(
    posterior_samples: np.ndarray,
    parameter_names: Sequence[str],
    statistic: str = "median",
) -> float:
    """
    Return a representative posterior residual/noise scale for Bayesian OED.

    For production EIG analyses, using the full posterior noise distribution
    can be added later. The median is a stable default for initial design
    comparisons.
    """

    sigma = posterior_sigma_samples(
        posterior_samples,
        parameter_names,
    )

    statistic = str(
        statistic
    ).strip().lower()

    if statistic == "median":
        return float(
            np.median(
                sigma
            )
        )

    if statistic == "mean":
        return float(
            np.mean(
                sigma
            )
        )

    if statistic == "q05":
        return float(
            np.percentile(
                sigma,
                5.0,
            )
        )

    if statistic == "q95":
        return float(
            np.percentile(
                sigma,
                95.0,
            )
        )

    raise ValueError(
        "Unknown statistic '{0}'. Choose median, mean, q05, or q95.".format(
            statistic
        )
    )
