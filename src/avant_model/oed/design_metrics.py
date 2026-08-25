#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Design metrics and ranking utilities for the AVANT analytical-model OED
workflow.

This module sits above ``fisher_information.py`` and provides the scalar
criteria and aggregation logic needed to compare experimental designs.

It is deliberately independent of:
    - repository paths
    - plotting
    - command-line parsing
    - the analytical forward model
    - PyDREAM

That separation allows the same design metrics to be used for:

1. Local OED
   Evaluate one candidate design at one nominal parameter vector.

2. Robust OED
   Evaluate a design across an ensemble representing uncertain body size,
   shape, depth, orientation, center, or boundary scenarios.

3. Posterior-informed / Bayesian robust OED
   Evaluate design utility across parameter draws from a Bayesian posterior.

4. Bayesian expected-information-gain workflows
   Use the generic utility aggregation/ranking tools here after utilities
   have been computed by the later ``bayesian_design.py`` module.

Canonical physical parameter vector
-----------------------------------
    [a, b, h, theta_deg, x0_prime, y0_prime]

The statistical nuisance parameter ``log10_sigma_strain`` is not part of
the physical OED state.

Metric directions
-----------------
Higher is better:
    D-optimality        = log det(F)
    E-optimality        = minimum eigenvalue of F
    trace information   = trace(F)
    rank
    D-efficiency
    E-efficiency

Lower is better:
    A-optimality        = trace(F^-1)
    condition number
    A-efficiency denominator / uncertainty proxy

Robust aggregation
------------------
For a design utility u(d, theta), this module supports:
    mean
    median
    standard deviation
    lower quantiles
    upper quantiles
    worst case
    coefficient of variation

The robust design layer can therefore compare expected performance and
guard against designs that fail for plausible unknown geometries.

Python compatibility
--------------------
This module is compatible with Python 3.9.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from avant_model.oed.fisher_information import (
    DesignMetrics,
    FisherResult,
    calculate_design_metrics,
)


# ============================================================================
# Metric metadata
# ============================================================================

HIGHER_IS_BETTER = {
    "d_optimality": True,
    "a_optimality": False,
    "e_optimality": True,
    "condition_number": False,
    "rank": True,
    "log_determinant": True,
    "trace_information": True,
    "min_eigenvalue": True,
    "max_eigenvalue": True,
    "d_efficiency": True,
    "a_efficiency": True,
    "e_efficiency": True,
}

DEFAULT_PRIMARY_METRIC = "d_optimality"


# ============================================================================
# Data containers
# ============================================================================


@dataclass(frozen=True)
class DesignScore:
    """
    Scalar metrics associated with one candidate design.
    """

    design_id: str
    design_size: int
    d_optimality: float
    a_optimality: float
    e_optimality: float
    condition_number: float
    rank: int
    log_determinant: float
    trace_information: float
    min_eigenvalue: float
    max_eigenvalue: float
    d_efficiency: float = np.nan
    a_efficiency: float = np.nan
    e_efficiency: float = np.nan


@dataclass(frozen=True)
class RobustMetricSummary:
    """
    Distribution summary for one design metric across uncertain scenarios.
    """

    metric: str
    n_scenarios: int
    mean: float
    median: float
    std: float
    q05: float
    q10: float
    q25: float
    q75: float
    q90: float
    q95: float
    minimum: float
    maximum: float
    worst_case: float
    coefficient_of_variation: float


@dataclass(frozen=True)
class RobustDesignScore:
    """
    Robust utility summary for one design.

    ``primary_expected`` describes mean utility.
    ``primary_conservative`` describes lower-tail or upper-tail performance
    after respecting the direction of the metric.
    ``primary_worst_case`` is the worst plausible scenario.
    """

    design_id: str
    design_size: int
    primary_metric: str
    higher_is_better: bool
    primary_expected: float
    primary_median: float
    primary_conservative: float
    primary_worst_case: float
    primary_std: float
    primary_cv: float
    n_scenarios: int


# ============================================================================
# Validation
# ============================================================================


def metric_direction(metric: str) -> bool:
    """
    Return True when larger metric values are preferable.
    """

    if metric not in HIGHER_IS_BETTER:
        raise ValueError(
            "Unknown design metric: {0}. Available metrics: {1}".format(
                metric,
                sorted(HIGHER_IS_BETTER),
            )
        )

    return bool(
        HIGHER_IS_BETTER[metric]
    )


def _finite_array(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            "{0} must contain at least one value.".format(
                name
            )
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "{0} contains non-finite values.".format(
                name
            )
        )

    return array


def validate_design_score_table(
    table: pd.DataFrame,
) -> None:
    required = {
        "design_id",
        "design_size",
    }

    missing = (
        required
        - set(table.columns)
    )

    if missing:
        raise ValueError(
            "Design score table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if table.empty:
        raise ValueError(
            "Design score table is empty."
        )

    if table["design_id"].duplicated().any():
        raise ValueError(
            "design_id values must be unique."
        )


# ============================================================================
# Conversion helpers
# ============================================================================


def metrics_to_dict(
    metrics: DesignMetrics,
) -> Dict[str, float]:
    """
    Convert DesignMetrics to a plain dictionary.
    """

    result = asdict(
        metrics
    )

    result["rank"] = int(
        result["rank"]
    )

    return result


def score_from_fisher_result(
    design_id: str,
    design_size: int,
    fisher_result: FisherResult,
) -> DesignScore:
    """
    Construct a DesignScore directly from a FisherResult.
    """

    metrics = calculate_design_metrics(
        fisher_result
    )

    return DesignScore(
        design_id=str(design_id),
        design_size=int(design_size),
        d_optimality=float(
            metrics.d_optimality
        ),
        a_optimality=float(
            metrics.a_optimality
        ),
        e_optimality=float(
            metrics.e_optimality
        ),
        condition_number=float(
            metrics.condition_number
        ),
        rank=int(
            metrics.rank
        ),
        log_determinant=float(
            metrics.log_determinant
        ),
        trace_information=float(
            metrics.trace_information
        ),
        min_eigenvalue=float(
            metrics.min_eigenvalue
        ),
        max_eigenvalue=float(
            metrics.max_eigenvalue
        ),
    )


def scores_to_dataframe(
    scores: Sequence[DesignScore],
) -> pd.DataFrame:
    """
    Convert design-score objects into a stable public table.
    """

    if not scores:
        raise ValueError(
            "At least one DesignScore is required."
        )

    dataframe = pd.DataFrame(
        [
            asdict(score)
            for score in scores
        ]
    )

    validate_design_score_table(
        dataframe
    )

    return dataframe


# ============================================================================
# Relative efficiencies
# ============================================================================


def _safe_exp(
    value: float,
) -> float:
    """
    Exponentiate while avoiding numerical overflow warnings.
    """

    return float(
        np.exp(
            np.clip(
                value,
                -700.0,
                700.0,
            )
        )
    )


def add_relative_efficiencies(
    table: pd.DataFrame,
    n_parameters: int,
) -> pd.DataFrame:
    """
    Add D-, A-, and E-efficiency relative to the best design in the table.

    D-efficiency
    ------------
    For p inferred parameters,

        eff_D = exp((logdet(F_d) - logdet(F_best)) / p)

    A-efficiency
    ------------
        eff_A = A_best / A_d

    because smaller trace(F^-1) is preferable.

    E-efficiency
    ------------
        eff_E = lambda_min(F_d) / lambda_min(F_best)

    All efficiencies are ideally in [0, 1], subject to floating-point
    tolerance and regularization.
    """

    if n_parameters <= 0:
        raise ValueError(
            "n_parameters must be positive."
        )

    dataframe = table.copy()

    validate_design_score_table(
        dataframe
    )

    required = {
        "d_optimality",
        "a_optimality",
        "e_optimality",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Efficiency calculation requires columns: "
            + ", ".join(sorted(missing))
        )

    d_values = dataframe[
        "d_optimality"
    ].to_numpy(
        dtype=float
    )

    a_values = dataframe[
        "a_optimality"
    ].to_numpy(
        dtype=float
    )

    e_values = dataframe[
        "e_optimality"
    ].to_numpy(
        dtype=float
    )

    finite_d = np.isfinite(
        d_values
    )

    if np.any(finite_d):
        best_d = float(
            np.max(
                d_values[
                    finite_d
                ]
            )
        )

        d_efficiency = np.zeros(
            len(dataframe),
            dtype=float,
        )

        for index, value in enumerate(
            d_values
        ):
            if np.isfinite(value):
                d_efficiency[index] = _safe_exp(
                    (
                        value
                        - best_d
                    )
                    / float(
                        n_parameters
                    )
                )
            else:
                d_efficiency[index] = 0.0
    else:
        d_efficiency = np.full(
            len(dataframe),
            np.nan,
        )

    valid_a = (
        np.isfinite(a_values)
        & (a_values > 0.0)
    )

    if np.any(valid_a):
        best_a = float(
            np.min(
                a_values[
                    valid_a
                ]
            )
        )

        a_efficiency = np.zeros(
            len(dataframe),
            dtype=float,
        )

        for index, value in enumerate(
            a_values
        ):
            if np.isfinite(value) and value > 0.0:
                a_efficiency[index] = (
                    best_a
                    / value
                )
            else:
                a_efficiency[index] = 0.0
    else:
        a_efficiency = np.full(
            len(dataframe),
            np.nan,
        )

    valid_e = (
        np.isfinite(e_values)
        & (e_values > 0.0)
    )

    if np.any(valid_e):
        best_e = float(
            np.max(
                e_values[
                    valid_e
                ]
            )
        )

        e_efficiency = np.zeros(
            len(dataframe),
            dtype=float,
        )

        for index, value in enumerate(
            e_values
        ):
            if np.isfinite(value) and best_e > 0.0:
                e_efficiency[index] = (
                    value
                    / best_e
                )
            else:
                e_efficiency[index] = 0.0
    else:
        e_efficiency = np.full(
            len(dataframe),
            np.nan,
        )

    dataframe[
        "d_efficiency"
    ] = np.clip(
        d_efficiency,
        0.0,
        1.0,
    )

    dataframe[
        "a_efficiency"
    ] = np.clip(
        a_efficiency,
        0.0,
        1.0,
    )

    dataframe[
        "e_efficiency"
    ] = np.clip(
        e_efficiency,
        0.0,
        1.0,
    )

    return dataframe


# ============================================================================
# Local design ranking
# ============================================================================


def rank_designs(
    table: pd.DataFrame,
    metric: str = DEFAULT_PRIMARY_METRIC,
    *,
    require_full_rank: bool = True,
    full_rank: Optional[int] = None,
    rank_column: str = "rank",
) -> pd.DataFrame:
    """
    Rank candidate designs by one scalar OED metric.

    Full-rank designs are preferred when ``require_full_rank`` is True.
    """

    dataframe = table.copy()

    validate_design_score_table(
        dataframe
    )

    if metric not in dataframe.columns:
        raise ValueError(
            "Metric '{0}' is not present in the design table.".format(
                metric
            )
        )

    higher = metric_direction(
        metric
    )

    if require_full_rank:
        if rank_column not in dataframe.columns:
            raise ValueError(
                "Full-rank filtering requires column '{0}'.".format(
                    rank_column
                )
            )

        if full_rank is None:
            full_rank = int(
                dataframe[
                    rank_column
                ].max()
            )

        dataframe[
            "full_rank"
        ] = (
            dataframe[
                rank_column
            ].astype(int)
            >= int(full_rank)
        )
    else:
        dataframe[
            "full_rank"
        ] = True

    # Rank full-rank designs first, then the selected metric.
    dataframe = dataframe.sort_values(
        by=[
            "full_rank",
            metric,
            "design_size",
            "design_id",
        ],
        ascending=[
            False,
            not higher,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    dataframe[
        "rank_{0}".format(
            metric
        )
    ] = np.arange(
        1,
        len(dataframe) + 1,
        dtype=int,
    )

    return dataframe


def best_design(
    table: pd.DataFrame,
    metric: str = DEFAULT_PRIMARY_METRIC,
    *,
    require_full_rank: bool = True,
    full_rank: Optional[int] = None,
) -> pd.Series:
    """
    Return the best candidate design under one metric.
    """

    ranked = rank_designs(
        table,
        metric=metric,
        require_full_rank=require_full_rank,
        full_rank=full_rank,
    )

    return ranked.iloc[
        0
    ].copy()


# ============================================================================
# Pareto analysis
# ============================================================================


def pareto_front(
    table: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """
    Return non-dominated designs for multiple OED criteria.

    Metric directions are inferred from HIGHER_IS_BETTER.

    A design A dominates B if A is no worse in every selected metric and
    strictly better in at least one.
    """

    dataframe = table.copy()

    validate_design_score_table(
        dataframe
    )

    if not metrics:
        raise ValueError(
            "At least one metric is required for Pareto analysis."
        )

    for metric in metrics:
        if metric not in dataframe.columns:
            raise ValueError(
                "Metric '{0}' is missing from the table.".format(
                    metric
                )
            )

        metric_direction(
            metric
        )

    values = dataframe[
        list(metrics)
    ].to_numpy(
        dtype=float
    )

    # Convert all objectives to maximization.
    transformed = values.copy()

    for column, metric in enumerate(
        metrics
    ):
        if not metric_direction(
            metric
        ):
            transformed[
                :,
                column
            ] *= -1.0

    n_designs = len(
        dataframe
    )

    dominated = np.zeros(
        n_designs,
        dtype=bool,
    )

    for i in range(
        n_designs
    ):
        if dominated[i]:
            continue

        for j in range(
            n_designs
        ):
            if i == j:
                continue

            no_worse = np.all(
                transformed[j]
                >= transformed[i]
            )

            strictly_better = np.any(
                transformed[j]
                > transformed[i]
            )

            if (
                no_worse
                and strictly_better
            ):
                dominated[i] = True
                break

    result = dataframe.loc[
        ~dominated
    ].copy()

    result[
        "pareto_optimal"
    ] = True

    return result.reset_index(
        drop=True
    )


# ============================================================================
# Robust metric summaries
# ============================================================================


def summarize_metric_distribution(
    values: Sequence[float],
    metric: str,
) -> RobustMetricSummary:
    """
    Summarize one metric across uncertain geometry/parameter scenarios.
    """

    array = _finite_array(
        values,
        metric,
    )

    higher = metric_direction(
        metric
    )

    mean = float(
        np.mean(array)
    )

    std = float(
        np.std(
            array,
            ddof=1,
        )
    ) if array.size > 1 else 0.0

    if np.isclose(
        mean,
        0.0,
    ):
        cv = float(
            np.nan
        )
    else:
        cv = float(
            abs(
                std / mean
            )
        )

    if higher:
        worst_case = float(
            np.min(array)
        )
    else:
        worst_case = float(
            np.max(array)
        )

    return RobustMetricSummary(
        metric=str(metric),
        n_scenarios=int(
            array.size
        ),
        mean=mean,
        median=float(
            np.median(array)
        ),
        std=std,
        q05=float(
            np.percentile(
                array,
                5.0,
            )
        ),
        q10=float(
            np.percentile(
                array,
                10.0,
            )
        ),
        q25=float(
            np.percentile(
                array,
                25.0,
            )
        ),
        q75=float(
            np.percentile(
                array,
                75.0,
            )
        ),
        q90=float(
            np.percentile(
                array,
                90.0,
            )
        ),
        q95=float(
            np.percentile(
                array,
                95.0,
            )
        ),
        minimum=float(
            np.min(array)
        ),
        maximum=float(
            np.max(array)
        ),
        worst_case=worst_case,
        coefficient_of_variation=cv,
    )


def robust_design_score(
    design_id: str,
    design_size: int,
    values: Sequence[float],
    metric: str = DEFAULT_PRIMARY_METRIC,
    *,
    conservative_quantile: float = 0.10,
) -> RobustDesignScore:
    """
    Build a robust score from utility values across uncertain scenarios.

    For higher-is-better metrics, the conservative utility is the lower
    quantile (default 10th percentile).

    For lower-is-better metrics, the conservative utility is the upper
    quantile (default 90th percentile).
    """

    if not (
        0.0
        < conservative_quantile
        < 0.5
    ):
        raise ValueError(
            "conservative_quantile must satisfy 0 < q < 0.5."
        )

    array = _finite_array(
        values,
        metric,
    )

    higher = metric_direction(
        metric
    )

    summary = summarize_metric_distribution(
        array,
        metric,
    )

    if higher:
        conservative = float(
            np.quantile(
                array,
                conservative_quantile,
            )
        )
    else:
        conservative = float(
            np.quantile(
                array,
                1.0
                - conservative_quantile,
            )
        )

    return RobustDesignScore(
        design_id=str(
            design_id
        ),
        design_size=int(
            design_size
        ),
        primary_metric=str(
            metric
        ),
        higher_is_better=bool(
            higher
        ),
        primary_expected=float(
            summary.mean
        ),
        primary_median=float(
            summary.median
        ),
        primary_conservative=conservative,
        primary_worst_case=float(
            summary.worst_case
        ),
        primary_std=float(
            summary.std
        ),
        primary_cv=float(
            summary.coefficient_of_variation
        ),
        n_scenarios=int(
            summary.n_scenarios
        ),
    )


def robust_scores_to_dataframe(
    scores: Sequence[RobustDesignScore],
) -> pd.DataFrame:
    if not scores:
        raise ValueError(
            "At least one RobustDesignScore is required."
        )

    dataframe = pd.DataFrame(
        [
            asdict(score)
            for score in scores
        ]
    )

    validate_design_score_table(
        dataframe
    )

    return dataframe


# ============================================================================
# Robust ranking
# ============================================================================


def rank_robust_designs(
    table: pd.DataFrame,
    *,
    ranking: str = "conservative",
) -> pd.DataFrame:
    """
    Rank robust candidate designs.

    ranking options
    ---------------
    expected
        Rank by mean utility across scenarios.

    median
        Rank by median utility.

    conservative
        Rank by the conservative tail utility.

    worst_case
        Rank by the worst plausible scenario.

    stability
        Rank by coefficient of variation (lower is better).

    balanced
        Rank by a normalized combination of expected utility,
        conservative utility, worst-case utility, and stability.
    """

    dataframe = table.copy()

    validate_design_score_table(
        dataframe
    )

    required = {
        "higher_is_better",
        "primary_expected",
        "primary_median",
        "primary_conservative",
        "primary_worst_case",
        "primary_cv",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Robust design table is missing columns: "
            + ", ".join(sorted(missing))
        )

    unique_directions = dataframe[
        "higher_is_better"
    ].unique()

    if len(
        unique_directions
    ) != 1:
        raise ValueError(
            "A robust ranking table must use one common primary metric."
        )

    higher = bool(
        unique_directions[0]
    )

    ranking = str(
        ranking
    ).strip().lower()

    column_map = {
        "expected": "primary_expected",
        "median": "primary_median",
        "conservative": "primary_conservative",
        "worst_case": "primary_worst_case",
        "stability": "primary_cv",
    }

    if ranking == "balanced":
        dataframe = add_balanced_robust_score(
            dataframe
        )

        sort_column = (
            "balanced_robust_score"
        )

        ascending = False

    elif ranking in column_map:
        sort_column = column_map[
            ranking
        ]

        if ranking == "stability":
            ascending = True
        else:
            ascending = (
                not higher
            )

    else:
        raise ValueError(
            "Unknown robust ranking mode '{0}'. "
            "Choose expected, median, conservative, worst_case, "
            "stability, or balanced.".format(
                ranking
            )
        )

    dataframe = dataframe.sort_values(
        by=[
            sort_column,
            "design_size",
            "design_id",
        ],
        ascending=[
            ascending,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    dataframe[
        "robust_rank"
    ] = np.arange(
        1,
        len(dataframe) + 1,
        dtype=int,
    )

    dataframe[
        "robust_ranking_mode"
    ] = ranking

    return dataframe


# ============================================================================
# Normalized multi-criterion score
# ============================================================================


def _normalize_benefit(
    values: np.ndarray,
) -> np.ndarray:
    """
    Normalize a higher-is-better quantity to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    span = (
        maximum
        - minimum
    )

    if np.isclose(
        span,
        0.0,
    ):
        return np.ones_like(
            values,
            dtype=float,
        )

    return (
        values
        - minimum
    ) / span


def _normalize_cost(
    values: np.ndarray,
) -> np.ndarray:
    """
    Normalize a lower-is-better quantity to [0, 1], with 1 best.
    """

    return 1.0 - _normalize_benefit(
        values
    )


def add_balanced_robust_score(
    table: pd.DataFrame,
    *,
    expected_weight: float = 0.35,
    conservative_weight: float = 0.35,
    worst_case_weight: float = 0.20,
    stability_weight: float = 0.10,
) -> pd.DataFrame:
    """
    Add a transparent normalized robust-design score.

    The default weighting emphasizes expected and conservative performance
    while still rewarding worst-case behavior and stability.

    This is a convenience ranking score, not a replacement for reporting the
    underlying D/A/E criteria.
    """

    weights = np.asarray(
        [
            expected_weight,
            conservative_weight,
            worst_case_weight,
            stability_weight,
        ],
        dtype=float,
    )

    if np.any(
        weights < 0.0
    ):
        raise ValueError(
            "Balanced robust-score weights must be non-negative."
        )

    if not np.isclose(
        np.sum(weights),
        1.0,
    ):
        raise ValueError(
            "Balanced robust-score weights must sum to 1."
        )

    dataframe = table.copy()

    required = {
        "higher_is_better",
        "primary_expected",
        "primary_conservative",
        "primary_worst_case",
        "primary_cv",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Balanced robust score requires columns: "
            + ", ".join(sorted(missing))
        )

    directions = dataframe[
        "higher_is_better"
    ].unique()

    if len(
        directions
    ) != 1:
        raise ValueError(
            "Balanced robust scoring requires one common primary metric."
        )

    higher = bool(
        directions[0]
    )

    expected = dataframe[
        "primary_expected"
    ].to_numpy(
        dtype=float
    )

    conservative = dataframe[
        "primary_conservative"
    ].to_numpy(
        dtype=float
    )

    worst_case = dataframe[
        "primary_worst_case"
    ].to_numpy(
        dtype=float
    )

    stability = dataframe[
        "primary_cv"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(
            expected
        )
    ):
        raise ValueError(
            "primary_expected contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            conservative
        )
    ):
        raise ValueError(
            "primary_conservative contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            worst_case
        )
    ):
        raise ValueError(
            "primary_worst_case contains non-finite values."
        )

    # A zero mean can make CV undefined. For balanced ranking, replace
    # non-finite CV by the largest finite CV so such a design is not rewarded.
    finite_stability = np.isfinite(
        stability
    )

    if np.any(
        finite_stability
    ):
        maximum_finite_cv = float(
            np.max(
                stability[
                    finite_stability
                ]
            )
        )

        stability = np.where(
            finite_stability,
            stability,
            maximum_finite_cv,
        )
    else:
        stability = np.zeros_like(
            stability
        )

    if higher:
        expected_score = _normalize_benefit(
            expected
        )

        conservative_score = _normalize_benefit(
            conservative
        )

        worst_score = _normalize_benefit(
            worst_case
        )

    else:
        expected_score = _normalize_cost(
            expected
        )

        conservative_score = _normalize_cost(
            conservative
        )

        worst_score = _normalize_cost(
            worst_case
        )

    stability_score = _normalize_cost(
        stability
    )

    dataframe[
        "normalized_expected_score"
    ] = expected_score

    dataframe[
        "normalized_conservative_score"
    ] = conservative_score

    dataframe[
        "normalized_worst_case_score"
    ] = worst_score

    dataframe[
        "normalized_stability_score"
    ] = stability_score

    dataframe[
        "balanced_robust_score"
    ] = (
        expected_weight
        * expected_score
        + conservative_weight
        * conservative_score
        + worst_case_weight
        * worst_score
        + stability_weight
        * stability_score
    )

    return dataframe


# ============================================================================
# Scenario-table aggregation
# ============================================================================


def aggregate_scenario_table(
    scenario_table: pd.DataFrame,
    *,
    design_column: str = "design_id",
    design_size_column: str = "design_size",
    metric: str = DEFAULT_PRIMARY_METRIC,
    conservative_quantile: float = 0.10,
) -> pd.DataFrame:
    """
    Convert a long scenario table into one robust score per design.

    Expected input
    --------------
    Each row represents one design evaluated for one uncertain
    geometry/parameter realization.

    Required columns:
        design_id
        design_size
        <metric>

    Optional columns such as scenario_id, a, b, h, theta_deg, x0_prime,
    y0_prime may remain in the input and are ignored by this aggregation.
    """

    required = {
        design_column,
        design_size_column,
        metric,
    }

    missing = (
        required
        - set(scenario_table.columns)
    )

    if missing:
        raise ValueError(
            "Scenario table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if scenario_table.empty:
        raise ValueError(
            "Scenario table is empty."
        )

    robust_scores = []

    grouped = scenario_table.groupby(
        design_column,
        sort=False,
    )

    for design_id, group in grouped:
        sizes = group[
            design_size_column
        ].unique()

        if len(
            sizes
        ) != 1:
            raise ValueError(
                "Design '{0}' has inconsistent design_size values.".format(
                    design_id
                )
            )

        score = robust_design_score(
            design_id=str(
                design_id
            ),
            design_size=int(
                sizes[0]
            ),
            values=group[
                metric
            ].to_numpy(
                dtype=float
            ),
            metric=metric,
            conservative_quantile=conservative_quantile,
        )

        robust_scores.append(
            score
        )

    return robust_scores_to_dataframe(
        robust_scores
    )


# ============================================================================
# Design-size efficiency
# ============================================================================


def best_by_design_size(
    table: pd.DataFrame,
    metric: str = DEFAULT_PRIMARY_METRIC,
) -> pd.DataFrame:
    """
    Return the best design for each sensor/station count.

    Useful for questions such as:
        "How much information do we gain by going from 4 to 5 sensors?"
    """

    dataframe = table.copy()

    validate_design_score_table(
        dataframe
    )

    if metric not in dataframe.columns:
        raise ValueError(
            "Metric '{0}' is missing from the design table.".format(
                metric
            )
        )

    higher = metric_direction(
        metric
    )

    rows = []

    for design_size, group in dataframe.groupby(
        "design_size",
        sort=True,
    ):
        if higher:
            index = group[
                metric
            ].idxmax()
        else:
            index = group[
                metric
            ].idxmin()

        row = dataframe.loc[
            index
        ].copy()

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    ).reset_index(
        drop=True
    )

    return result


def incremental_information_gain(
    best_size_table: pd.DataFrame,
    metric: str = DEFAULT_PRIMARY_METRIC,
) -> pd.DataFrame:
    """
    Compute information gain as the optimal design size increases.

    For higher-is-better metrics:
        delta = metric(k) - metric(k-1)

    For lower-is-better metrics:
        delta = metric(k-1) - metric(k)

    Positive values therefore always indicate improvement.
    """

    if "design_size" not in best_size_table.columns:
        raise ValueError(
            "best_size_table must contain design_size."
        )

    if metric not in best_size_table.columns:
        raise ValueError(
            "Metric '{0}' is missing.".format(
                metric
            )
        )

    dataframe = (
        best_size_table
        .sort_values(
            "design_size"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    values = dataframe[
        metric
    ].to_numpy(
        dtype=float
    )

    higher = metric_direction(
        metric
    )

    gain = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    if len(
        dataframe
    ) > 1:
        if higher:
            gain[1:] = (
                values[1:]
                - values[:-1]
            )
        else:
            gain[1:] = (
                values[:-1]
                - values[1:]
            )

    dataframe[
        "incremental_{0}_gain".format(
            metric
        )
    ] = gain

    return dataframe


# ============================================================================
# Convenience summaries
# ============================================================================


def local_design_report(
    table: pd.DataFrame,
    *,
    n_parameters: int,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    require_full_rank: bool = True,
) -> Dict[str, object]:
    """
    Produce a compact machine-readable summary of a local OED table.
    """

    with_efficiency = add_relative_efficiencies(
        table,
        n_parameters=n_parameters,
    )

    ranked = rank_designs(
        with_efficiency,
        metric=primary_metric,
        require_full_rank=require_full_rank,
        full_rank=n_parameters,
    )

    best = ranked.iloc[
        0
    ]

    size_best = best_by_design_size(
        with_efficiency,
        metric=primary_metric,
    )

    size_gain = incremental_information_gain(
        size_best,
        metric=primary_metric,
    )

    return {
        "primary_metric": primary_metric,
        "higher_is_better": metric_direction(
            primary_metric
        ),
        "n_parameters": int(
            n_parameters
        ),
        "n_designs": int(
            len(table)
        ),
        "best_design": {
            key: (
                value.item()
                if hasattr(
                    value,
                    "item",
                )
                else value
            )
            for key, value in best.to_dict().items()
        },
        "best_by_design_size": size_gain.to_dict(
            orient="records"
        ),
    }
