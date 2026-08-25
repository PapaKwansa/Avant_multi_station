#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust optimal experimental design utilities for the AVANT analytical model.

This module integrates the lower-level OED components:

    geometry_uncertainty
        ↓
    forward-model Jacobian
        ↓
    Fisher information
        ↓
    design metrics
        ↓
    robust design ranking

The primary scientific use case is sensor-network design when the body
geometry is uncertain.

Canonical physical parameter vector
-----------------------------------
    [a, b, h, theta_deg, x0_prime, y0_prime]

The statistical nuisance parameter log10_sigma_strain is not included in
the physical OED state.

Design interpretation
---------------------
For a candidate sensor design d and uncertain geometry theta^(k),

    F_d(theta^(k)) = J_d^T W J_d

is evaluated and reduced to a design utility, for example

    u_D = log det(F_d)

The resulting utilities are aggregated over geometry scenarios.

Supported robust summaries include:

    expected / mean utility
    median utility
    conservative quantile
    worst-case utility
    utility variability

The module also supports weighted scenario ensembles, so Bayesian posterior
draws can be supplied directly as geometry scenarios.

No plotting, command-line parsing, or repository-specific file I/O occurs
here. Those belong to the user-facing script layer.

Python compatibility
--------------------
Python 3.9 compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from avant_model.oed.design_metrics import (
    DEFAULT_PRIMARY_METRIC,
    DesignScore,
    RobustDesignScore,
    add_relative_efficiencies,
    aggregate_scenario_table,
    best_by_design_size,
    calculate_design_metrics,
    metric_direction,
    rank_designs,
    rank_robust_designs,
    robust_design_score,
    scores_to_dataframe,
)
from avant_model.oed.fisher_information import (
    FisherResult,
    finite_difference_jacobian,
    fisher_diagnostics,
    fisher_information_matrix,
    make_observation_weights,
    scale_jacobian_parameters,
    subset_jacobian_by_stations,
)
from avant_model.oed.geometry_uncertainty import (
    GEOMETRY_PARAMETER_NAMES,
    GeometryScenario,
    dataframe_to_scenarios,
)


# ============================================================================
# Data containers
# ============================================================================


@dataclass(frozen=True)
class ScenarioJacobian:
    """
    Jacobian information associated with one uncertain geometry scenario.
    """

    scenario_id: str
    source: str
    weight: float
    parameters: np.ndarray
    baseline_output: np.ndarray
    jacobian: np.ndarray
    parameter_steps: np.ndarray


@dataclass(frozen=True)
class RobustDesignEvaluation:
    """
    Evaluation of one candidate design over multiple uncertainty scenarios.
    """

    design_id: str
    design_size: int
    scenario_metrics: pd.DataFrame
    robust_score: RobustDesignScore


# ============================================================================
# Validation
# ============================================================================


def _as_positive_vector(
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

    if np.any(
        array <= 0.0
    ):
        raise ValueError(
            "{0} must contain strictly positive values.".format(
                name
            )
        )

    return array


def _validate_scenario_table(
    scenarios: pd.DataFrame,
) -> None:
    required = {
        "scenario_id",
        "source",
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
            "Scenario table is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if scenarios.empty:
        raise ValueError(
            "Scenario table is empty."
        )

    if scenarios[
        "scenario_id"
    ].duplicated().any():
        raise ValueError(
            "Scenario IDs must be unique."
        )

    weights = _as_positive_vector(
        scenarios[
            "weight"
        ].to_numpy(
            dtype=float
        ),
        "scenario weights",
    )

    # Robust OED allows zero-weight scenario records, but negative weights
    # are never permissible. Re-check without the positive-vector helper.
    raw_weights = scenarios[
        "weight"
    ].to_numpy(
        dtype=float
    )

    if np.any(
        raw_weights < 0.0
    ):
        raise ValueError(
            "Scenario weights must be non-negative."
        )

    if np.sum(
        raw_weights
    ) <= 0.0:
        raise ValueError(
            "Scenario weights must contain positive total mass."
        )

    for name in GEOMETRY_PARAMETER_NAMES:
        values = scenarios[
            name
        ].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Scenario parameter {0} contains non-finite values.".format(
                    name
                )
            )


# ============================================================================
# Scenario conversion
# ============================================================================


def geometry_dataframe_to_mappings(
    scenarios: pd.DataFrame,
) -> List[Tuple[str, str, float, Dict[str, float]]]:
    """
    Convert a geometry-scenario DataFrame to an explicit list of mappings.
    """

    _validate_scenario_table(
        scenarios
    )

    rows = []

    for _, row in scenarios.iterrows():
        geometry = {
            name: float(
                row[name]
            )
            for name in GEOMETRY_PARAMETER_NAMES
        }

        rows.append(
            (
                str(
                    row["scenario_id"]
                ),
                str(
                    row["source"]
                ),
                float(
                    row["weight"]
                ),
                geometry,
            )
        )

    return rows


def geometry_scenario_objects(
    scenarios: pd.DataFrame,
) -> List[GeometryScenario]:
    """
    Convert a scenario DataFrame to GeometryScenario objects.
    """

    _validate_scenario_table(
        scenarios
    )

    return list(
        dataframe_to_scenarios(
            scenarios
        )
    )


# ============================================================================
# Jacobian evaluation
# ============================================================================


def make_forward_function(
    parameter_names: Sequence[str],
    base_forward_function: Callable[[Mapping[str, float]], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Adapt a mapping-based forward model to the vector-based finite-difference
    API in fisher_information.py.
    """

    names = tuple(
        str(name)
        for name in parameter_names
    )

    if names != GEOMETRY_PARAMETER_NAMES:
        raise ValueError(
            "The robust-design geometry state must use the canonical "
            "parameter ordering: {0}".format(
                GEOMETRY_PARAMETER_NAMES
            )
        )

    def forward_vector(
        parameter_vector: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            parameter_vector,
            dtype=float,
        ).reshape(-1)

        if values.size != len(
            names
        ):
            raise ValueError(
                "Geometry parameter vector has incorrect length."
            )

        geometry = {
            name: float(
                value
            )
            for name, value in zip(
                names,
                values,
            )
        }

        output = np.asarray(
            base_forward_function(
                geometry
            ),
            dtype=float,
        ).reshape(-1)

        return output

    return forward_vector


def evaluate_scenario_jacobians(
    scenarios: pd.DataFrame,
    forward_function_factory: Callable[
        [Mapping[str, float]],
        np.ndarray,
    ],
    *,
    finite_difference_step: float = 1.0e-5,
    finite_difference_method: str = "central",
    parameter_scales: Optional[
        Sequence[float]
    ] = None,
    parameter_bounds: Optional[
        Sequence[Sequence[float]]
    ] = None,
) -> List[ScenarioJacobian]:
    """
    Compute one finite-difference Jacobian for every geometry scenario.

    ``forward_function_factory`` receives a geometry mapping and must return
    a callable suitable for finite_difference_jacobian, or a forward output
    directly through the adapted path below.

    The preferred contract is:

        forward_function_factory(geometry) -> callable(parameter_vector)

    where the callable is evaluated around the supplied geometry.

    This intentionally allows the caller to encapsulate the analytical
    forward-model setup without hard-coding repository-specific data paths.
    """

    _validate_scenario_table(
        scenarios
    )

    results = []

    for (
        scenario_id,
        source,
        weight,
        geometry,
    ) in geometry_dataframe_to_mappings(
        scenarios
    ):

        forward_callable = (
            forward_function_factory(
                geometry
            )
        )

        if not callable(
            forward_callable
        ):
            raise TypeError(
                "forward_function_factory must return a callable."
            )

        baseline_vector = np.asarray(
            [
                geometry[name]
                for name in GEOMETRY_PARAMETER_NAMES
            ],
            dtype=float,
        )

        jacobian_result = (
            finite_difference_jacobian(
                forward_callable,
                baseline_vector,
                relative_step=finite_difference_step,
                method=finite_difference_method,
                step_scales=parameter_scales,
                bounds=parameter_bounds,
            )
        )

        results.append(
            ScenarioJacobian(
                scenario_id=scenario_id,
                source=source,
                weight=weight,
                parameters=(
                    jacobian_result.parameters.copy()
                ),
                baseline_output=(
                    jacobian_result.baseline_output.copy()
                ),
                jacobian=(
                    jacobian_result.jacobian.copy()
                ),
                parameter_steps=(
                    jacobian_result.parameter_steps.copy()
                ),
            )
        )

    return results


# ============================================================================
# Fisher calculations for one scenario/design
# ============================================================================


def evaluate_design_at_scenario(
    jacobian: np.ndarray,
    *,
    design_id: str,
    design_size: int,
    station_indices: Optional[
        Sequence[int]
    ] = None,
    station_count: Optional[int] = None,
    observation_sigma: Optional[
        float | Sequence[float]
    ] = None,
    parameter_scales: Optional[
        Sequence[float]
    ] = None,
    regularization: float = 1.0e-10,
    rank_tolerance: float = 1.0e-10,
    n_components: int = 4,
) -> DesignScore:
    """
    Evaluate one candidate design for one scenario.

    If station_indices is None, the complete observation design is used.
    Otherwise the Jacobian is restricted to the specified station subset.
    """

    J = np.asarray(
        jacobian,
        dtype=float,
    )

    if J.ndim != 2:
        raise ValueError(
            "jacobian must be a 2-D array."
        )

    if station_indices is not None:
        if station_count is None:
            raise ValueError(
                "station_count is required when station_indices are supplied."
            )

        J = subset_jacobian_by_stations(
            J,
            station_count=station_count,
            station_indices=station_indices,
            n_components=n_components,
        )

    weights = make_observation_weights(
        J.shape[0],
        observation_sigma,
    )

    fisher = fisher_information_matrix(
        J,
        observation_weights=weights,
        parameter_scales=parameter_scales,
    )

    diagnostics = fisher_diagnostics(
        fisher,
        regularization=regularization,
        rank_tolerance=rank_tolerance,
    )

    metrics = calculate_design_metrics(
        diagnostics
    )

    return DesignScore(
        design_id=str(
            design_id
        ),
        design_size=int(
            design_size
        ),
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


# ============================================================================
# Full robust design evaluation
# ============================================================================


def evaluate_design_across_scenarios(
    scenario_jacobians: Sequence[ScenarioJacobian],
    *,
    design_id: str,
    design_size: int,
    station_indices: Optional[
        Sequence[int]
    ] = None,
    station_count: Optional[int] = None,
    observation_sigma: Optional[
        float | Sequence[float]
    ] = None,
    parameter_scales: Optional[
        Sequence[float]
    ] = None,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    conservative_quantile: float = 0.10,
    regularization: float = 1.0e-10,
    rank_tolerance: float = 1.0e-10,
    n_components: int = 4,
) -> RobustDesignEvaluation:
    """
    Evaluate one candidate sensor network over an uncertainty ensemble.
    """

    if not scenario_jacobians:
        raise ValueError(
            "At least one scenario Jacobian is required."
        )

    records = []

    for scenario in scenario_jacobians:

        score = evaluate_design_at_scenario(
            scenario.jacobian,
            design_id=design_id,
            design_size=design_size,
            station_indices=station_indices,
            station_count=station_count,
            observation_sigma=observation_sigma,
            parameter_scales=parameter_scales,
            regularization=regularization,
            rank_tolerance=rank_tolerance,
            n_components=n_components,
        )

        record = {
            "design_id": str(
                design_id
            ),
            "design_size": int(
                design_size
            ),
            "scenario_id": str(
                scenario.scenario_id
            ),
            "source": str(
                scenario.source
            ),
            "scenario_weight": float(
                scenario.weight
            ),
            **{
                key: value
                for key, value in asdict(
                    score
                ).items()
                if key
                not in {
                    "design_id",
                    "design_size",
                }
            },
        }

        # Preserve all design metrics with stable names.
        records.append(
            record
        )

    scenario_metrics = pd.DataFrame(
        records
    )

    weights = scenario_metrics[
        "scenario_weight"
    ].to_numpy(
        dtype=float
    )

    weights = (
        weights
        / np.sum(
            weights
        )
    )

    scenario_metrics[
        "scenario_weight"
    ] = weights

    # Weighted robust summary for the primary metric.
    primary_values = scenario_metrics[
        primary_metric
    ].to_numpy(
        dtype=float
    )

    robust_score = (
        robust_design_score(
            design_id=design_id,
            design_size=design_size,
            values=primary_values,
            metric=primary_metric,
            conservative_quantile=conservative_quantile,
        )
    )

    return RobustDesignEvaluation(
        design_id=str(
            design_id
        ),
        design_size=int(
            design_size
        ),
        scenario_metrics=scenario_metrics,
        robust_score=robust_score,
    )


# ============================================================================
# Weighted robust statistics
# ============================================================================


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    """
    Compute a weighted empirical quantile.

    Parameters
    ----------
    values
        Utility values.

    weights
        Non-negative scenario weights.

    quantile
        Between 0 and 1.
    """

    values_array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    weights_array = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if values_array.size != (
        weights_array.size
    ):
        raise ValueError(
            "values and weights must have equal lengths."
        )

    if values_array.size == 0:
        raise ValueError(
            "At least one value is required."
        )

    if not (
        0.0
        <= quantile
        <= 1.0
    ):
        raise ValueError(
            "quantile must be between 0 and 1."
        )

    if not np.all(
        np.isfinite(
            values_array
        )
    ):
        raise ValueError(
            "values contain non-finite values."
        )

    if not np.all(
        np.isfinite(
            weights_array
        )
    ):
        raise ValueError(
            "weights contain non-finite values."
        )

    if np.any(
        weights_array < 0.0
    ):
        raise ValueError(
            "weights must be non-negative."
        )

    total_weight = float(
        np.sum(
            weights_array
        )
    )

    if total_weight <= 0.0:
        raise ValueError(
            "weights must have positive total weight."
        )

    order = np.argsort(
        values_array
    )

    sorted_values = (
        values_array[
            order
        ]
    )

    sorted_weights = (
        weights_array[
            order
        ]
    )

    cumulative = np.cumsum(
        sorted_weights
        / total_weight
    )

    index = int(
        np.searchsorted(
            cumulative,
            quantile,
            side="left",
        )
    )

    index = min(
        max(index, 0),
        sorted_values.size - 1,
    )

    return float(
        sorted_values[
            index
        ]
    )


def weighted_robust_summary(
    values: Sequence[float],
    weights: Sequence[float],
    metric: str = DEFAULT_PRIMARY_METRIC,
) -> Dict[str, float]:
    """
    Weighted distribution summary for robust/Bayesian design utilities.
    """

    values_array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    weights_array = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if values_array.size != (
        weights_array.size
    ):
        raise ValueError(
            "values and weights must have equal lengths."
        )

    if values_array.size == 0:
        raise ValueError(
            "At least one utility value is required."
        )

    if not np.all(
        np.isfinite(
            values_array
        )
    ):
        raise ValueError(
            "utility values contain non-finite values."
        )

    if not np.all(
        np.isfinite(
            weights_array
        )
    ):
        raise ValueError(
            "scenario weights contain non-finite values."
        )

    if np.any(
        weights_array < 0.0
    ):
        raise ValueError(
            "scenario weights must be non-negative."
        )

    total = float(
        np.sum(
            weights_array
        )
    )

    if total <= 0.0:
        raise ValueError(
            "scenario weights must have positive total mass."
        )

    normalized = (
        weights_array
        / total
    )

    mean = float(
        np.sum(
            normalized
            * values_array
        )
    )

    variance = float(
        np.sum(
            normalized
            * np.square(
                values_array
                - mean
            )
        )
    )

    std = float(
        np.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )

    higher = metric_direction(
        metric
    )

    if higher:
        worst = float(
            np.min(
                values_array
            )
        )

        conservative = weighted_quantile(
            values_array,
            normalized,
            0.10,
        )
    else:
        worst = float(
            np.max(
                values_array
            )
        )

        conservative = weighted_quantile(
            values_array,
            normalized,
            0.90,
        )

    median = weighted_quantile(
        values_array,
        normalized,
        0.50,
    )

    q05 = weighted_quantile(
        values_array,
        normalized,
        0.05,
    )

    q95 = weighted_quantile(
        values_array,
        normalized,
        0.95,
    )

    cv = (
        abs(
            std / mean
        )
        if not np.isclose(
            mean,
            0.0,
        )
        else float(
            np.nan
        )
    )

    effective_n = float(
        1.0
        / np.sum(
            normalized**2
        )
    )

    return {
        "metric": str(
            metric
        ),
        "n_scenarios": int(
            values_array.size
        ),
        "effective_scenario_count": effective_n,
        "weighted_mean": mean,
        "weighted_median": median,
        "weighted_std": std,
        "weighted_q05": q05,
        "weighted_q95": q95,
        "conservative_utility": float(
            conservative
        ),
        "worst_case_utility": worst,
        "coefficient_of_variation": cv,
    }


# ============================================================================
# Robust ranking from long scenario data
# ============================================================================


def robust_ranking_from_scenario_table(
    scenario_table: pd.DataFrame,
    *,
    metric: str = DEFAULT_PRIMARY_METRIC,
    conservative_quantile: float = 0.10,
    ranking: str = "conservative",
) -> pd.DataFrame:
    """
    Rank designs directly from a long design/scenario table.

    The ranking uses scenario weights rather than treating every scenario
    as equally likely when explicit weights are supplied.
    """

    required = {
        "design_id",
        "design_size",
        "scenario_weight",
        metric,
    }

    missing = (
        required
        - set(
            scenario_table.columns
        )
    )

    if missing:
        raise ValueError(
            "Scenario table is missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    robust_rows = []

    for (
        design_id,
        group,
    ) in scenario_table.groupby(
        "design_id",
        sort=False,
    ):

        size_values = group[
            "design_size"
        ].unique()

        if len(
            size_values
        ) != 1:
            raise ValueError(
                "Design {0} has inconsistent design_size values.".format(
                    design_id
                )
            )

        values = group[
            metric
        ].to_numpy(
            dtype=float
        )

        weights = group[
            "scenario_weight"
        ].to_numpy(
            dtype=float
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
            primary_conservative = (
                weighted_quantile(
                    values,
                    weights,
                    conservative_quantile,
                )
            )
        else:
            primary_conservative = (
                weighted_quantile(
                    values,
                    weights,
                    1.0
                    - conservative_quantile,
                )
            )

        robust_rows.append(
            {
                "design_id": str(
                    design_id
                ),
                "design_size": int(
                    size_values[0]
                ),
                "primary_metric": str(
                    metric
                ),
                "higher_is_better": bool(
                    higher
                ),
                "primary_expected": float(
                    summary[
                        "weighted_mean"
                    ]
                ),
                "primary_median": float(
                    summary[
                        "weighted_median"
                    ]
                ),
                "primary_conservative": float(
                    primary_conservative
                ),
                "primary_worst_case": float(
                    summary[
                        "worst_case_utility"
                    ]
                ),
                "primary_std": float(
                    summary[
                        "weighted_std"
                    ]
                ),
                "primary_cv": float(
                    summary[
                        "coefficient_of_variation"
                    ]
                ),
                "n_scenarios": int(
                    summary[
                        "n_scenarios"
                    ]
                ),
                "effective_scenario_count": float(
                    summary[
                        "effective_scenario_count"
                    ]
                ),
            }
        )

    robust_table = pd.DataFrame(
        robust_rows
    )

    return rank_robust_designs(
        robust_table,
        ranking=ranking,
    )


# ============================================================================
# Station-network utilities
# ============================================================================


def evaluate_station_networks(
    scenario_jacobians: Sequence[ScenarioJacobian],
    station_count: int,
    candidate_designs: Sequence[
        Sequence[int]
    ],
    *,
    observation_sigma: Optional[
        float | Sequence[float]
    ] = None,
    parameter_scales: Optional[
        Sequence[float]
    ] = None,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    conservative_quantile: float = 0.10,
    regularization: float = 1.0e-10,
    rank_tolerance: float = 1.0e-10,
    n_components: int = 4,
) -> pd.DataFrame:
    """
    Evaluate a collection of candidate station networks.

    Candidate designs are zero-based station-index tuples/lists.
    """

    if station_count <= 0:
        raise ValueError(
            "station_count must be positive."
        )

    if not candidate_designs:
        raise ValueError(
            "At least one candidate design is required."
        )

    all_scenario_records = []

    for design_number, station_indices in enumerate(
        candidate_designs
    ):
        design_id = (
            "design_{0:04d}".format(
                design_number + 1
            )
        )

        clean_indices = tuple(
            sorted(
                set(
                    int(index)
                    for index in station_indices
                )
            )
        )

        if not clean_indices:
            raise ValueError(
                "Candidate station designs cannot be empty."
            )

        if any(
            index < 0
            or index >= station_count
            for index in clean_indices
        ):
            raise IndexError(
                "A candidate design contains an invalid station index."
            )

        evaluation = (
            evaluate_design_across_scenarios(
                scenario_jacobians,
                design_id=design_id,
                design_size=len(
                    clean_indices
                ),
                station_indices=clean_indices,
                station_count=station_count,
                observation_sigma=observation_sigma,
                parameter_scales=parameter_scales,
                primary_metric=primary_metric,
                conservative_quantile=conservative_quantile,
                regularization=regularization,
                rank_tolerance=rank_tolerance,
                n_components=n_components,
            )
        )

        all_scenario_records.append(
            evaluation.scenario_metrics
        )

    return pd.concat(
        all_scenario_records,
        axis=0,
        ignore_index=True,
    )


# ============================================================================
# Posterior-informed robust utility
# ============================================================================


def posterior_weighted_design_summary(
    scenario_table: pd.DataFrame,
    *,
    metric: str = DEFAULT_PRIMARY_METRIC,
    conservative_quantile: float = 0.10,
) -> pd.DataFrame:
    """
    Convert posterior-weighted scenario utilities into one summary per design.

    The scenario table is expected to contain:
        design_id
        design_size
        scenario_weight
        metric

    This function is deliberately agnostic to how the scenarios were
    generated. They may come from an LHS prior, posterior MCMC draws, a
    mixed prior/posterior ensemble, or explicit geometry stress cases.
    """

    ranked = robust_ranking_from_scenario_table(
        scenario_table,
        metric=metric,
        conservative_quantile=conservative_quantile,
        ranking="conservative",
    )

    return ranked


# ============================================================================
# Utility comparison across multiple OED criteria
# ============================================================================


def summarize_multi_metric_design(
    local_table: pd.DataFrame,
    n_parameters: int,
) -> Dict[str, pd.DataFrame]:
    """
    Return ranked tables for D, A, and E criteria.

    This is intended for reporting the same candidate network under multiple
    standard OED criteria rather than collapsing everything immediately into
    one arbitrary score.
    """

    with_efficiency = add_relative_efficiencies(
        local_table,
        n_parameters=n_parameters,
    )

    reports = {}

    for metric in (
        "d_optimality",
        "a_optimality",
        "e_optimality",
    ):
        reports[
            metric
        ] = rank_designs(
            with_efficiency,
            metric=metric,
            require_full_rank=True,
            full_rank=n_parameters,
        )

    return reports


# ============================================================================
# Validation of robust design results
# ============================================================================


def validate_robust_result_table(
    table: pd.DataFrame,
) -> None:
    required = {
        "design_id",
        "design_size",
        "primary_metric",
        "higher_is_better",
        "primary_expected",
        "primary_conservative",
        "primary_worst_case",
        "primary_std",
        "primary_cv",
        "n_scenarios",
    }

    missing = (
        required
        - set(
            table.columns
        )
    )

    if missing:
        raise ValueError(
            "Robust design result table is missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if table.empty:
        raise ValueError(
            "Robust design result table is empty."
        )

    for column in (
        "primary_expected",
        "primary_conservative",
        "primary_worst_case",
        "primary_std",
        "n_scenarios",
    ):
        values = table[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Robust design result column {0} contains non-finite values.".format(
                    column
                )
            )


# ============================================================================
# Compact report
# ============================================================================


def compact_robust_report(
    robust_table: pd.DataFrame,
    *,
    top_n: int = 10,
    ranking: str = "conservative",
) -> pd.DataFrame:
    """
    Return the top robust designs in a compact reporting table.
    """

    if top_n <= 0:
        raise ValueError(
            "top_n must be positive."
        )

    validate_robust_result_table(
        robust_table
    )

    ranked = rank_robust_designs(
        robust_table,
        ranking=ranking,
    )

    columns = [
        "robust_rank",
        "design_id",
        "design_size",
        "primary_metric",
        "primary_expected",
        "primary_median",
        "primary_conservative",
        "primary_worst_case",
        "primary_std",
        "primary_cv",
        "n_scenarios",
    ]

    existing = [
        column
        for column in columns
        if column in ranked.columns
    ]

    return ranked[
        existing
    ].head(
        top_n
    ).copy()
