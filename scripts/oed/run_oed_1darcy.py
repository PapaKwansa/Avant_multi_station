#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production OED runner for the AVANT 1-Darcy analytical transient model.

This is the public command-line orchestration layer above:

    avant_model.oed.fisher_information
    avant_model.oed.design_metrics
    avant_model.oed.geometry_uncertainty
    avant_model.oed.robust_design
    avant_model.oed.sensor_placement

It performs classical/local and prior-robust OED. Bayesian posterior-informed
and expected-information-gain analyses are intentionally delegated to the
separate run_bayesian_oed_1darcy.py runner.

Physical OED state
------------------
    [a, b, h, theta_deg, x0_prime, y0_prime]

Fixed current analytical inputs
-------------------------------
    c, E, nu, pmax, tpeak, d, alpha

Analysis modes
--------------
local
    Full-network Fisher/Jacobian diagnostics at the reference geometry.

existing
    Exhaustive station-subset analysis for the canonical S01--S08 network.

robust
    Prior-domain robust station-subset OED over uncertain body geometry.

boundary
    Stress-test station subsets at deterministic geometry-domain extremes.

placement
    Generate candidate x-y strainmeter locations and screen candidate
    networks under geometry uncertainty.

all
    Run all of the above.

Outputs
-------
results/oed/1darcy/
    run_manifest.json
    local/
    existing_network/
    robust/
    boundary/
    placement/
    tables/
    figures/

The runner writes machine-readable CSV/JSON results and publication-quality
PNG/PDF figures. Legends are placed below figures when a legend is needed.

Python compatibility
--------------------
Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time as _time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Repository imports
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from avant_model.data import multi_stations_input as input_data
from avant_model.model.forward_model_multi_station import (
    forward_model_multi_station,
)

from avant_model.oed.design_metrics import (
    add_relative_efficiencies,
    best_by_design_size,
    incremental_information_gain,
    rank_designs,
)

from avant_model.oed.fisher_information import (
    finite_difference_jacobian,
    fisher_diagnostics,
    fisher_information_matrix,
    make_observation_weights,
    subset_jacobian_by_stations,
)

from avant_model.oed.geometry_uncertainty import (
    GEOMETRY_PARAMETER_NAMES,
    boundary_scenarios,
    reference_scenario,
    sample_prior_lhs,
    scenario_summary,
)

from avant_model.oed.robust_design import (
    ScenarioJacobian,
    robust_ranking_from_scenario_table,
)

from avant_model.oed.sensor_placement import (
    SensorLocation,
    SensorNetwork,
    generate_rectangular_grid,
    greedy_network_by_distance,
    random_candidate_networks,
)


# ============================================================================
# Canonical definitions
# ============================================================================

PARAMETER_NAMES = tuple(
    GEOMETRY_PARAMETER_NAMES
)

PARAMETER_LABELS = {
    "a": r"$a$",
    "b": r"$b$",
    "h": r"$h$",
    "theta_deg": r"$\theta$",
    "x0_prime": r"$x_0'$",
    "y0_prime": r"$y_0'$",
}

COMPONENTS = (
    "eXX",
    "eYY",
    "eZZ",
    "eXY",
)

DEFAULT_REFERENCE = {
    "a": 190.0,
    "b": 325.0,
    "h": 520.0,
    "theta_deg": 15.0,
    "x0_prime": 45.0,
    "y0_prime": 170.0,
}

DEFAULT_BOUNDS = {
    "a": [40.0, 900.0],
    "b": [20.0, 1000.0],
    "h": [50.0, 900.0],
    "theta_deg": [-90.0, 90.0],
    "x0_prime": [-900.0, 900.0],
    "y0_prime": [-900.0, 900.0],
}


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run production local, station-subset, robust, boundary, and "
            "spatial-placement OED for the AVANT 1-Darcy analytical model."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "local",
            "existing",
            "robust",
            "boundary",
            "placement",
            "all",
        ],
        default="all",
        help="OED analysis mode.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/oed/1darcy",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help=(
            "Assumed common strain uncertainty in nstrain for Fisher "
            "weighting. Since common scalar sigma rescales F uniformly, "
            "station rankings are often unchanged; report the chosen value."
        ),
    )

    parser.add_argument(
        "--fd-step",
        type=float,
        default=1.0e-4,
        help="Relative central finite-difference step.",
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-10,
        help="Dimensionless Fisher diagonal regularization.",
    )

    parser.add_argument(
        "--robust-scenarios",
        type=int,
        default=32,
        help=(
            "Number of prior-LHS geometry scenarios for robust OED. "
            "Use a small value for local smoke tests and larger values "
            "for Palmetto production runs."
        ),
    )

    parser.add_argument(
        "--robust-subset-sizes",
        type=str,
        default="4,8",
        help=(
            "Comma-separated station counts evaluated in robust/boundary "
            "modes. Default: 4,8. Use 1,2,3,4,5,6,7,8 for exhaustive "
            "production analysis."
        ),
    )

    parser.add_argument(
        "--placement-grid-nx",
        type=int,
        default=9,
        help="Candidate placement grid count in x.",
    )

    parser.add_argument(
        "--placement-grid-ny",
        type=int,
        default=9,
        help="Candidate placement grid count in y.",
    )

    parser.add_argument(
        "--placement-xmin",
        type=float,
        default=-900.0,
    )

    parser.add_argument(
        "--placement-xmax",
        type=float,
        default=900.0,
    )

    parser.add_argument(
        "--placement-ymin",
        type=float,
        default=-900.0,
    )

    parser.add_argument(
        "--placement-ymax",
        type=float,
        default=900.0,
    )

    parser.add_argument(
        "--placement-size",
        type=int,
        default=4,
        help="Number of strainmeters in each proposed placement network.",
    )

    parser.add_argument(
        "--placement-networks",
        type=int,
        default=100,
        help="Number of random candidate placement networks to screen.",
    )

    parser.add_argument(
        "--placement-min-spacing",
        type=float,
        default=100.0,
        help="Minimum x-y spacing between proposed strainmeters in meters.",
    )

    parser.add_argument(
        "--placement-scenarios",
        type=int,
        default=16,
        help="Prior-LHS scenarios used for free-placement screening.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================================
# Basic utilities
# ============================================================================

def save_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=_json_default,
        )


def _json_default(value):
    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    raise TypeError(
        "Object of type {0} is not JSON serializable.".format(
            type(
                value
            ).__name__
        )
    )


def parse_subset_sizes(
    text: str,
    station_count: int,
) -> List[int]:
    values = sorted(
        set(
            int(
                item.strip()
            )
            for item in text.split(",")
            if item.strip()
        )
    )

    if not values:
        raise ValueError(
            "At least one robust subset size is required."
        )

    if any(
        value < 1
        or value > station_count
        for value in values
    ):
        raise ValueError(
            "Robust subset sizes must lie between 1 and {0}.".format(
                station_count
            )
        )

    return values


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.20,
        }
    )


def save_figure(
    fig,
    output_base: Path,
    *,
    legend_handles=None,
    legend_labels=None,
    legend_ncol: int = 2,
) -> None:
    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        legend_handles is not None
        and legend_labels is not None
        and len(
            legend_handles
        ) > 0
    ):
        fig.subplots_adjust(
            bottom=0.18
        )

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(
                0.5,
                0.02,
            ),
            ncol=legend_ncol,
            frameon=False,
        )

    fig.savefig(
        output_base.with_suffix(
            ".png"
        )
    )

    fig.savefig(
        output_base.with_suffix(
            ".pdf"
        )
    )

    plt.close(
        fig
    )


# ============================================================================
# Canonical data/model configuration
# ============================================================================

def load_station_metadata() -> pd.DataFrame:
    path = (
        REPO_ROOT
        / "datasets"
        / "comsol"
        / "1darcy"
        / "metadata"
        / "stations.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Canonical station metadata not found:\n    {0}".format(
                path
            )
        )

    stations = pd.read_csv(
        path
    )

    required = {
        "station",
        "x",
        "y",
        "z",
    }

    missing = (
        required
        - set(
            stations.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Station metadata is missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    stations = stations[
        [
            "station",
            "x",
            "y",
            "z",
        ]
    ].copy()

    stations[
        "station"
    ] = (
        stations[
            "station"
        ]
        .astype(str)
        .str.strip()
    )

    for column in (
        "x",
        "y",
        "z",
    ):
        stations[
            column
        ] = pd.to_numeric(
            stations[
                column
            ],
            errors="raise",
        )

    return stations


def load_observed_data(
    stations: pd.DataFrame,
) -> pd.DataFrame:
    path = (
        REPO_ROOT
        / "datasets"
        / "comsol"
        / "1darcy"
        / "processed"
        / "strain.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Canonical 1-Darcy strain data not found:\n    {0}".format(
                path
            )
        )

    observed = pd.read_csv(
        path
    )

    station_names = (
        stations[
            "station"
        ].tolist()
    )

    channels = [
        "{0}_{1}".format(
            component,
            station,
        )
        for component in COMPONENTS
        for station in station_names
    ]

    missing = [
        channel
        for channel in channels
        if channel not in observed.columns
    ]

    if "time_s" not in observed.columns:
        missing = [
            "time_s",
            *missing,
        ]

    if missing:
        raise RuntimeError(
            "Observed dataset is missing columns:\n"
            + "\n".join(
                "    {0}".format(
                    column
                )
                for column in missing
            )
        )

    observed = observed[
        [
            "time_s",
            *channels,
        ]
    ].copy()

    for column in observed.columns:
        observed[
            column
        ] = pd.to_numeric(
            observed[
                column
            ],
            errors="raise",
        )

    return observed


def fixed_model_inputs() -> Dict[str, float]:
    params = input_data.read_input()

    return {
        "c": float(
            params.get(
                "c_fixed",
                3.125,
            )
        ),
        "E": float(
            params.get(
                "E",
                8.0e9,
            )
        ),
        "nu": float(
            params.get(
                "nu",
                0.35,
            )
        ),
        "pmax": float(
            params.get(
                "pmax",
                9.99310e5,
            )
        ),
        "tpeak": float(
            params.get(
                "tpeak",
                3.5e5,
            )
        ),
        "d": float(
            params.get(
                "d",
                0.4,
            )
        ),
        "alpha": float(
            params.get(
                "alpha",
                0.8,
            )
        ),
    }


def parameter_scales_from_bounds(
    bounds: Mapping[
        str,
        Sequence[float],
    ],
) -> np.ndarray:
    return np.asarray(
        [
            float(
                bounds[
                    name
                ][1]
                - bounds[
                    name
                ][0]
            )
            for name in PARAMETER_NAMES
        ],
        dtype=float,
    )


def parameter_bounds_array(
    bounds: Mapping[
        str,
        Sequence[float],
    ],
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(
                    bounds[
                        name
                    ][0]
                ),
                float(
                    bounds[
                        name
                    ][1]
                ),
            ]
            for name in PARAMETER_NAMES
        ],
        dtype=float,
    )


# ============================================================================
# Forward-model factories
# ============================================================================

def full_network_forward_factory(
    stations: pd.DataFrame,
    observed: pd.DataFrame,
    fixed: Mapping[str, float],
):
    station_names = (
        stations[
            "station"
        ].to_numpy(
            str
        )
    )

    x_prime = stations[
        "x"
    ].to_numpy(
        float
    )

    y_prime = stations[
        "y"
    ].to_numpy(
        float
    )

    z = stations[
        "z"
    ].to_numpy(
        float
    )

    time = observed[
        "time_s"
    ].to_numpy(
        float
    )

    channels = [
        "{0}_{1}".format(
            component,
            station,
        )
        for component in COMPONENTS
        for station in station_names
    ]

    def forward(
        parameter_vector: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            parameter_vector,
            dtype=float,
        ).reshape(-1)

        if values.size != len(
            PARAMETER_NAMES
        ):
            raise ValueError(
                "OED geometry vector has incorrect length."
            )

        geometry = {
            name: float(
                value
            )
            for name, value in zip(
                PARAMETER_NAMES,
                values,
            )
        }

        prediction = (
            forward_model_multi_station(
                pmax=fixed["pmax"],
                tpeak=fixed["tpeak"],
                d=fixed["d"],
                time=time,
                x_prime=x_prime,
                y_prime=y_prime,
                x0_prime=geometry[
                    "x0_prime"
                ],
                y0_prime=geometry[
                    "y0_prime"
                ],
                z=z,
                a=geometry[
                    "a"
                ],
                b=geometry[
                    "b"
                ],
                c=fixed[
                    "c"
                ],
                nu=fixed[
                    "nu"
                ],
                h=geometry[
                    "h"
                ],
                E=fixed[
                    "E"
                ],
                theta_deg=geometry[
                    "theta_deg"
                ],
                alpha=fixed[
                    "alpha"
                ],
                station_names=station_names,
                debug=False,
            )
        )

        # Critical ordering:
        # time-major rows, component-major/station-minor channels.
        # This allows station-subset row selection after flattening.
        return prediction[
            channels
        ].to_numpy(
            float
        ).reshape(-1)

    return forward


def placement_forward_factory(
    locations: Sequence[SensorLocation],
    time: np.ndarray,
    fixed: Mapping[str, float],
):
    station_names = np.asarray(
        [
            location.sensor_id
            for location in locations
        ],
        dtype=str,
    )

    x_prime = np.asarray(
        [
            location.x
            for location in locations
        ],
        dtype=float,
    )

    y_prime = np.asarray(
        [
            location.y
            for location in locations
        ],
        dtype=float,
    )

    z = np.asarray(
        [
            location.z
            for location in locations
        ],
        dtype=float,
    )

    channels = [
        "{0}_{1}".format(
            component,
            station,
        )
        for component in COMPONENTS
        for station in station_names
    ]

    def forward(
        parameter_vector: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            parameter_vector,
            dtype=float,
        ).reshape(-1)

        geometry = {
            name: float(
                value
            )
            for name, value in zip(
                PARAMETER_NAMES,
                values,
            )
        }

        prediction = (
            forward_model_multi_station(
                pmax=fixed["pmax"],
                tpeak=fixed["tpeak"],
                d=fixed["d"],
                time=time,
                x_prime=x_prime,
                y_prime=y_prime,
                x0_prime=geometry[
                    "x0_prime"
                ],
                y0_prime=geometry[
                    "y0_prime"
                ],
                z=z,
                a=geometry[
                    "a"
                ],
                b=geometry[
                    "b"
                ],
                c=fixed[
                    "c"
                ],
                nu=fixed[
                    "nu"
                ],
                h=geometry[
                    "h"
                ],
                E=fixed[
                    "E"
                ],
                theta_deg=geometry[
                    "theta_deg"
                ],
                alpha=fixed[
                    "alpha"
                ],
                station_names=station_names,
                debug=False,
            )
        )

        return prediction[
            channels
        ].to_numpy(
            float
        ).reshape(-1)

    return forward


# ============================================================================
# Correct time-series station row selection
# ============================================================================

def station_rows_time_series(
    n_times: int,
    station_count: int,
    station_indices: Sequence[int],
    n_components: int = 4,
) -> np.ndarray:
    """
    Return flattened row indices for selected stations.

    Forward outputs are flattened from shape:

        (n_times, n_components * station_count)

    in C order. Therefore station selection must be repeated at every time.
    """

    selected = tuple(
        sorted(
            set(
                int(index)
                for index in station_indices
            )
        )
    )

    if not selected:
        raise ValueError(
            "At least one station is required."
        )

    if any(
        index < 0
        or index >= station_count
        for index in selected
    ):
        raise IndexError(
            "Invalid station index."
        )

    channel_indices = []

    for component_index in range(
        n_components
    ):
        offset = (
            component_index
            * station_count
        )

        channel_indices.extend(
            offset + station_index
            for station_index in selected
        )

    channel_indices = np.asarray(
        channel_indices,
        dtype=int,
    )

    rows = []

    channels_per_time = (
        n_components
        * station_count
    )

    for time_index in range(
        n_times
    ):
        rows.extend(
            time_index
            * channels_per_time
            + channel_indices
        )

    return np.asarray(
        rows,
        dtype=int,
    )


# ============================================================================
# Jacobian calculations
# ============================================================================

def compute_jacobian(
    forward_function,
    geometry: Mapping[str, float],
    bounds: Mapping[
        str,
        Sequence[float],
    ],
    fd_step: float,
):
    parameter_vector = np.asarray(
        [
            float(
                geometry[
                    name
                ]
            )
            for name in PARAMETER_NAMES
        ],
        dtype=float,
    )

    scales = parameter_scales_from_bounds(
        bounds
    )

    bounds_array = parameter_bounds_array(
        bounds
    )

    return finite_difference_jacobian(
        forward_function,
        parameter_vector,
        step_scales=scales,
        relative_step=fd_step,
        method="central",
        bounds=bounds_array,
    )


def evaluate_subset_from_full_jacobian(
    full_jacobian: np.ndarray,
    *,
    n_times: int,
    station_count: int,
    station_indices: Sequence[int],
    sigma: float,
    parameter_scales: np.ndarray,
    regularization: float,
) -> Dict[str, object]:
    rows = station_rows_time_series(
        n_times=n_times,
        station_count=station_count,
        station_indices=station_indices,
        n_components=len(
            COMPONENTS
        ),
    )

    J = full_jacobian[
        rows,
        :,
    ]

    weights = make_observation_weights(
        J.shape[0],
        sigma,
    )

    F = fisher_information_matrix(
        J,
        observation_weights=weights,
        parameter_scales=parameter_scales,
    )

    diagnostics = fisher_diagnostics(
        F,
        regularization=regularization,
    )

    covariance = diagnostics.covariance

    std = diagnostics.standard_errors

    correlation = np.zeros_like(
        covariance
    )

    denominator = np.outer(
        std,
        std,
    )

    valid = denominator > 0.0

    correlation[
        valid
    ] = (
        covariance[
            valid
        ]
        / denominator[
            valid
        ]
    )

    return {
        "jacobian": J,
        "fisher_result": diagnostics,
        "correlation": correlation,
    }


# ============================================================================
# Local OED
# ============================================================================

def run_local(
    output_dir: Path,
    stations: pd.DataFrame,
    observed: pd.DataFrame,
    fixed: Mapping[str, float],
    reference: Mapping[str, float],
    bounds: Mapping[
        str,
        Sequence[float],
    ],
    sigma: float,
    fd_step: float,
    regularization: float,
) -> None:
    print(
        "\n[OED] Local full-network identifiability"
    )

    local_dir = (
        output_dir
        / "local"
    )

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    forward = full_network_forward_factory(
        stations,
        observed,
        fixed,
    )

    jacobian_result = compute_jacobian(
        forward,
        reference,
        bounds,
        fd_step,
    )

    scales = parameter_scales_from_bounds(
        bounds
    )

    result = evaluate_subset_from_full_jacobian(
        jacobian_result.jacobian,
        n_times=len(
            observed
        ),
        station_count=len(
            stations
        ),
        station_indices=tuple(
            range(
                len(
                    stations
                )
            )
        ),
        sigma=sigma,
        parameter_scales=scales,
        regularization=regularization,
    )

    fisher = result[
        "fisher_result"
    ]

    np.save(
        local_dir
        / "jacobian.npy",
        jacobian_result.jacobian,
    )

    np.save(
        local_dir
        / "fisher_information.npy",
        fisher.fisher,
    )

    np.save(
        local_dir
        / "covariance_proxy.npy",
        fisher.covariance,
    )

    pd.DataFrame(
        fisher.fisher,
        index=PARAMETER_NAMES,
        columns=PARAMETER_NAMES,
    ).to_csv(
        local_dir
        / "fisher_information.csv"
    )

    pd.DataFrame(
        result[
            "correlation"
        ],
        index=PARAMETER_NAMES,
        columns=PARAMETER_NAMES,
    ).to_csv(
        local_dir
        / "parameter_correlation.csv"
    )

    parameter_table = pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "reference_value": [
                reference[
                    name
                ]
                for name in PARAMETER_NAMES
            ],
            "parameter_scale": scales,
            "standard_error_scaled": (
                fisher.standard_errors
            ),
        }
    )

    parameter_table.to_csv(
        local_dir
        / "parameter_identifiability.csv",
        index=False,
    )

    save_json(
        local_dir
        / "local_oed_summary.json",
        {
            "rank": fisher.rank,
            "condition_number": (
                fisher.condition_number
            ),
            "log_determinant": (
                fisher.log_determinant
            ),
            "trace_information": (
                fisher.trace
            ),
            "min_eigenvalue": (
                fisher.min_eigenvalue
            ),
            "max_eigenvalue": (
                fisher.max_eigenvalue
            ),
            "finite_difference_steps": (
                jacobian_result.parameter_steps
            ),
        },
    )

    # Fisher heatmap.
    fig, ax = plt.subplots(
        figsize=(
            8.5,
            7.0,
        )
    )

    image = ax.imshow(
        np.log10(
            np.maximum(
                np.abs(
                    fisher.fisher
                ),
                1.0e-300,
            )
        ),
        cmap="viridis",
        aspect="auto",
    )

    labels = [
        PARAMETER_LABELS[
            name
        ]
        for name in PARAMETER_NAMES
    ]

    ax.set_xticks(
        np.arange(
            len(
                labels
            )
        )
    )

    ax.set_yticks(
        np.arange(
            len(
                labels
            )
        )
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_yticklabels(
        labels
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        r"$\log_{10}|F_{ij}|$"
    )

    ax.set_title(
        "Full-network Fisher information"
    )

    save_figure(
        fig,
        local_dir
        / "fisher_information_heatmap",
    )

    # Parameter correlation.
    fig, ax = plt.subplots(
        figsize=(
            8.5,
            7.0,
        )
    )

    image = ax.imshow(
        result[
            "correlation"
        ],
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(
                labels
            )
        )
    )

    ax.set_yticks(
        np.arange(
            len(
                labels
            )
        )
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_yticklabels(
        labels
    )

    for row in range(
        len(
            labels
        )
    ):
        for column in range(
            len(
                labels
            )
        ):
            ax.text(
                column,
                row,
                "{0:.2f}".format(
                    result[
                        "correlation"
                    ][
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Local parameter correlation"
    )

    ax.set_title(
        "Local parameter tradeoffs"
    )

    save_figure(
        fig,
        local_dir
        / "parameter_correlation_heatmap",
    )

    print(
        "  rank = {0}/{1}".format(
            fisher.rank,
            len(
                PARAMETER_NAMES
            ),
        )
    )

    print(
        "  condition number = {0:.6g}".format(
            fisher.condition_number
        )
    )

    print(
        "  log det(F) = {0:.6g}".format(
            fisher.log_determinant
        )
    )


# ============================================================================
# Existing station subsets
# ============================================================================

def station_subset_records(
    full_jacobian: np.ndarray,
    stations: pd.DataFrame,
    observed: pd.DataFrame,
    subset_sizes: Sequence[int],
    sigma: float,
    parameter_scales: np.ndarray,
    regularization: float,
) -> pd.DataFrame:
    station_names = (
        stations[
            "station"
        ].tolist()
    )

    records = []

    for subset_size in subset_sizes:
        for subset in combinations(
            range(
                len(
                    stations
                )
            ),
            subset_size,
        ):
            result = (
                evaluate_subset_from_full_jacobian(
                    full_jacobian,
                    n_times=len(
                        observed
                    ),
                    station_count=len(
                        stations
                    ),
                    station_indices=subset,
                    sigma=sigma,
                    parameter_scales=parameter_scales,
                    regularization=regularization,
                )
            )

            fisher = result[
                "fisher_result"
            ]

            sensor_ids = [
                station_names[
                    index
                ]
                for index in subset
            ]

            records.append(
                {
                    "design_id": (
                        "+".join(
                            sensor_ids
                        )
                    ),
                    "design_size": int(
                        subset_size
                    ),
                    "station_indices": ",".join(
                        str(
                            index
                        )
                        for index in subset
                    ),
                    "sensor_ids": ",".join(
                        sensor_ids
                    ),
                    "d_optimality": float(
                        fisher.log_determinant
                    ),
                    "a_optimality": float(
                        np.trace(
                            fisher.covariance
                        )
                    ),
                    "e_optimality": float(
                        fisher.min_eigenvalue
                    ),
                    "condition_number": float(
                        fisher.condition_number
                    ),
                    "rank": int(
                        fisher.rank
                    ),
                    "log_determinant": float(
                        fisher.log_determinant
                    ),
                    "trace_information": float(
                        fisher.trace
                    ),
                    "min_eigenvalue": float(
                        fisher.min_eigenvalue
                    ),
                    "max_eigenvalue": float(
                        fisher.max_eigenvalue
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def plot_existing_results(
    table: pd.DataFrame,
    stations: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures = (
        output_dir
        / "figures"
    )

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_size = best_by_design_size(
        table,
        metric="d_optimality",
    )

    gain = incremental_information_gain(
        best_size,
        metric="d_optimality",
    )

    gain.to_csv(
        output_dir
        / "tables"
        / "best_existing_network_by_size.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            6.2,
        )
    )

    ax.plot(
        gain[
            "design_size"
        ],
        gain[
            "d_optimality"
        ],
        marker="o",
        linewidth=2.2,
        label="Best D-optimal network",
    )

    ax.set_xlabel(
        "Number of strainmeters"
    )

    ax.set_ylabel(
        r"$\log\det(F)$"
    )

    ax.set_title(
        "Information versus existing-network size"
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    save_figure(
        fig,
        figures
        / "existing_information_vs_sensor_count",
        legend_handles=handles,
        legend_labels=labels,
        legend_ncol=1,
    )

    # Existing station map.
    fig, ax = plt.subplots(
        figsize=(
            8.5,
            8.0,
        )
    )

    ax.scatter(
        stations[
            "x"
        ],
        stations[
            "y"
        ],
        s=85,
        label="Existing strainmeters",
    )

    for _, row in stations.iterrows():
        ax.annotate(
            row[
                "station"
            ],
            (
                row[
                    "x"
                ],
                row[
                    "y"
                ],
            ),
            xytext=(
                6,
                6,
            ),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "$x'$ (m)"
    )

    ax.set_ylabel(
        "$y'$ (m)"
    )

    ax.set_title(
        "Canonical existing strainmeter network"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    save_figure(
        fig,
        figures
        / "existing_station_network",
        legend_handles=handles,
        legend_labels=labels,
        legend_ncol=1,
    )


def run_existing(
    output_dir: Path,
    stations: pd.DataFrame,
    observed: pd.DataFrame,
    fixed: Mapping[str, float],
    reference: Mapping[str, float],
    bounds: Mapping[
        str,
        Sequence[float],
    ],
    sigma: float,
    fd_step: float,
    regularization: float,
) -> pd.DataFrame:
    print(
        "\n[OED] Existing S01-S08 station-subset analysis"
    )

    forward = full_network_forward_factory(
        stations,
        observed,
        fixed,
    )

    jacobian_result = compute_jacobian(
        forward,
        reference,
        bounds,
        fd_step,
    )

    scales = parameter_scales_from_bounds(
        bounds
    )

    table = station_subset_records(
        jacobian_result.jacobian,
        stations,
        observed,
        subset_sizes=list(
            range(
                1,
                len(
                    stations
                )
                + 1,
            )
        ),
        sigma=sigma,
        parameter_scales=scales,
        regularization=regularization,
    )

    table = add_relative_efficiencies(
        table,
        n_parameters=len(
            PARAMETER_NAMES
        ),
    )

    table = rank_designs(
        table,
        metric="d_optimality",
        require_full_rank=True,
        full_rank=len(
            PARAMETER_NAMES
        ),
    )

    table_dir = (
        output_dir
        / "tables"
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        table_dir
        / "existing_station_subset_oed.csv",
        index=False,
    )

    plot_existing_results(
        table,
        stations,
        output_dir,
    )

    print(
        "  evaluated {0} station subsets".format(
            len(
                table
            )
        )
    )

    print(
        "  best D-optimal network: {0}".format(
            table.iloc[
                0
            ][
                "sensor_ids"
            ]
        )
    )

    return table


# ============================================================================
# Robust existing-network OED
# ============================================================================

def scenario_jacobians_existing(
    scenarios: pd.DataFrame,
    stations: pd.DataFrame,
    observed: pd.DataFrame,
    fixed: Mapping[str, float],
    bounds: Mapping[
        str,
        Sequence[float],
    ],
    fd_step: float,
) -> List[ScenarioJacobian]:
    forward = full_network_forward_factory(
        stations,
        observed,
        fixed,
    )

    results = []

    for _, row in scenarios.iterrows():
        geometry = {
            name: float(
                row[
                    name
                ]
            )
            for name in PARAMETER_NAMES
        }

        jacobian_result = compute_jacobian(
            forward,
            geometry,
            bounds,
            fd_step,
        )

        results.append(
            ScenarioJacobian(
                scenario_id=str(
                    row[
                        "scenario_id"
                    ]
                ),
                source=str(
                    row[
                        "source"
                    ]
                ),
                weight=float(
                    row[
                        "weight"
                    ]
                ),
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


def robust_existing_long_table(
    scenario_jacobians: Sequence[ScenarioJacobian],
    stations: pd.DataFrame,
    observed: pd.DataFrame,
    subset_sizes: Sequence[int],
    sigma: float,
    parameter_scales: np.ndarray,
    regularization: float,
) -> pd.DataFrame:
    station_names = (
        stations[
            "station"
        ].tolist()
    )

    records = []

    for subset_size in subset_sizes:
        for subset in combinations(
            range(
                len(
                    stations
                )
            ),
            subset_size,
        ):
            sensor_ids = [
                station_names[
                    index
                ]
                for index in subset
            ]

            design_id = "+".join(
                sensor_ids
            )

            for scenario in scenario_jacobians:
                result = (
                    evaluate_subset_from_full_jacobian(
                        scenario.jacobian,
                        n_times=len(
                            observed
                        ),
                        station_count=len(
                            stations
                        ),
                        station_indices=subset,
                        sigma=sigma,
                        parameter_scales=parameter_scales,
                        regularization=regularization,
                    )
                )

                fisher = result[
                    "fisher_result"
                ]

                records.append(
                    {
                        "design_id": design_id,
                        "design_size": int(
                            subset_size
                        ),
                        "sensor_ids": ",".join(
                            sensor_ids
                        ),
                        "scenario_id": (
                            scenario.scenario_id
                        ),
                        "source": (
                            scenario.source
                        ),
                        "scenario_weight": float(
                            scenario.weight
                        ),
                        "d_optimality": float(
                            fisher.log_determinant
                        ),
                        "a_optimality": float(
                            np.trace(
                                fisher.covariance
                            )
                        ),
                        "e_optimality": float(
                            fisher.min_eigenvalue
                        ),
                        "condition_number": float(
                            fisher.condition_number
                        ),
                        "rank": int(
                            fisher.rank
                        ),
                    }
                )

    return pd.DataFrame(
        records
    )


def plot_robust_rankings(
    ranking: pd.DataFrame,
    output_base: Path,
    title: str,
    top_n: int = 15,
) -> None:
    top = ranking.head(
        min(
            top_n,
            len(
                ranking
            ),
        )
    ).copy()

    top = top.iloc[
        ::-1
    ]

    fig, ax = plt.subplots(
        figsize=(
            11.0,
            max(
                6.5,
                0.45
                * len(
                    top
                ),
            ),
        )
    )

    ax.barh(
        top[
            "design_id"
        ],
        top[
            "primary_conservative"
        ],
        label="Conservative robust utility",
    )

    ax.scatter(
        top[
            "primary_expected"
        ],
        top[
            "design_id"
        ],
        marker="o",
        label="Expected utility",
        zorder=3,
    )

    ax.set_xlabel(
        r"Robust D-optimal utility, $\log\det(F)$"
    )

    ax.set_ylabel(
        "Sensor network"
    )

    ax.set_title(
        title
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    save_figure(
        fig,
        output_base,
        legend_handles=handles,
        legend_labels=labels,
        legend_ncol=2,
    )


def run_robust_existing(
    output_dir: Path,
    stations: pd.DataFrame,
    observed: pd.DataFrame,
    fixed: Mapping[str, float],
    bounds: Mapping[
        str,
        Sequence[float],
    ],
    subset_sizes: Sequence[int],
    sigma: float,
    fd_step: float,
    regularization: float,
    n_scenarios: int,
    seed: int,
    *,
    boundary_mode: bool = False,
) -> pd.DataFrame:
    label = (
        "Boundary stress-test OED"
        if boundary_mode
        else "Prior-robust OED"
    )

    print(
        "\n[OED] {0}".format(
            label
        )
    )

    if boundary_mode:
        scenarios = boundary_scenarios(
            bounds,
            reference=DEFAULT_REFERENCE,
            include_reference=True,
        )

        subdirectory = (
            output_dir
            / "boundary"
        )
    else:
        scenarios = sample_prior_lhs(
            bounds,
            n_samples=n_scenarios,
            seed=seed,
            source="prior_lhs",
        )

        subdirectory = (
            output_dir
            / "robust"
        )

    subdirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios.to_csv(
        subdirectory
        / "geometry_scenarios.csv",
        index=False,
    )

    scenario_summary(
        scenarios
    ).to_csv(
        subdirectory
        / "geometry_scenario_summary.csv",
        index=False,
    )

    print(
        "  computing {0} scenario Jacobians...".format(
            len(
                scenarios
            )
        )
    )

    scenario_jacobians = (
        scenario_jacobians_existing(
            scenarios,
            stations,
            observed,
            fixed,
            bounds,
            fd_step,
        )
    )

    scales = parameter_scales_from_bounds(
        bounds
    )

    long_table = robust_existing_long_table(
        scenario_jacobians,
        stations,
        observed,
        subset_sizes=subset_sizes,
        sigma=sigma,
        parameter_scales=scales,
        regularization=regularization,
    )

    long_table.to_csv(
        subdirectory
        / "scenario_design_metrics.csv",
        index=False,
    )

    ranking = robust_ranking_from_scenario_table(
        long_table,
        metric="d_optimality",
        conservative_quantile=0.10,
        ranking="conservative",
    )

    # Restore sensor IDs for reporting.
    design_sensor_map = (
        long_table[
            [
                "design_id",
                "sensor_ids",
            ]
        ]
        .drop_duplicates(
            "design_id"
        )
    )

    ranking = ranking.merge(
        design_sensor_map,
        on="design_id",
        how="left",
        validate="one_to_one",
    )

    ranking.to_csv(
        subdirectory
        / "robust_station_ranking.csv",
        index=False,
    )

    plot_robust_rankings(
        ranking,
        subdirectory
        / "robust_station_ranking",
        title=label,
    )

    print(
        "  best conservative network: {0}".format(
            ranking.iloc[
                0
            ][
                "sensor_ids"
            ]
        )
    )

    return ranking


# ============================================================================
# Free spatial placement
# ============================================================================

def network_jacobian_for_scenario(
    network: SensorNetwork,
    geometry: Mapping[str, float],
    time: np.ndarray,
    fixed: Mapping[str, float],
    bounds: Mapping[
        str,
        Sequence[float],
    ],
    fd_step: float,
):
    forward = placement_forward_factory(
        network.sensor_locations,
        time,
        fixed,
    )

    return compute_jacobian(
        forward,
        geometry,
        bounds,
        fd_step,
    )


def run_placement(
    output_dir: Path,
    observed: pd.DataFrame,
    fixed: Mapping[str, float],
    bounds: Mapping[
        str,
        Sequence[float],
    ],
    args: argparse.Namespace,
) -> pd.DataFrame:
    print(
        "\n[OED] Free spatial strainmeter placement"
    )

    placement_dir = (
        output_dir
        / "placement"
    )

    placement_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    deployment_bounds = [
        args.placement_xmin,
        args.placement_xmax,
        args.placement_ymin,
        args.placement_ymax,
    ]

    candidates = generate_rectangular_grid(
        deployment_bounds,
        nx=args.placement_grid_nx,
        ny=args.placement_grid_ny,
        z=float(
            -40.0
        ),
        prefix="P",
        include_boundary=True,
    )

    # Geometry-only dispersed network retained as a baseline.
    baseline_network = (
        greedy_network_by_distance(
            candidates,
            design_size=args.placement_size,
            min_spacing=(
                args.placement_min_spacing
            ),
        )
    )

    random_networks = (
        random_candidate_networks(
            candidates,
            design_size=args.placement_size,
            n_networks=args.placement_networks,
            seed=args.seed,
            min_spacing=(
                args.placement_min_spacing
            ),
        )
    )

    networks = [
        SensorNetwork(
            design_id="spatial_baseline",
            sensor_ids=(
                baseline_network.sensor_ids
            ),
            sensor_locations=(
                baseline_network.sensor_locations
            ),
            design_size=(
                baseline_network.design_size
            ),
        ),
        *random_networks,
    ]

    # Candidate-location table.
    pd.DataFrame(
        [
            {
                "sensor_id": location.sensor_id,
                "x": location.x,
                "y": location.y,
                "z": location.z,
            }
            for location in candidates
        ]
    ).to_csv(
        placement_dir
        / "candidate_locations.csv",
        index=False,
    )

    scenarios = sample_prior_lhs(
        bounds,
        n_samples=args.placement_scenarios,
        seed=args.seed,
        source="placement_prior_lhs",
    )

    scenarios.to_csv(
        placement_dir
        / "placement_geometry_scenarios.csv",
        index=False,
    )

    time = observed[
        "time_s"
    ].to_numpy(
        float
    )

    scales = parameter_scales_from_bounds(
        bounds
    )

    records = []

    total = (
        len(
            networks
        )
        * len(
            scenarios
        )
    )

    counter = 0

    for network in networks:
        for _, scenario_row in scenarios.iterrows():
            counter += 1

            if (
                counter == 1
                or counter % 50 == 0
                or counter == total
            ):
                print(
                    "  placement Jacobian {0}/{1}".format(
                        counter,
                        total,
                    )
                )

            geometry = {
                name: float(
                    scenario_row[
                        name
                    ]
                )
                for name in PARAMETER_NAMES
            }

            jacobian_result = (
                network_jacobian_for_scenario(
                    network,
                    geometry,
                    time,
                    fixed,
                    bounds,
                    args.fd_step,
                )
            )

            J = (
                jacobian_result.jacobian
            )

            weights = make_observation_weights(
                J.shape[0],
                args.sigma,
            )

            fisher_matrix = (
                fisher_information_matrix(
                    J,
                    observation_weights=weights,
                    parameter_scales=scales,
                )
            )

            fisher = fisher_diagnostics(
                fisher_matrix,
                regularization=(
                    args.regularization
                ),
            )

            records.append(
                {
                    "design_id": (
                        network.design_id
                    ),
                    "design_size": (
                        network.design_size
                    ),
                    "sensor_ids": ",".join(
                        network.sensor_ids
                    ),
                    "scenario_id": str(
                        scenario_row[
                            "scenario_id"
                        ]
                    ),
                    "source": str(
                        scenario_row[
                            "source"
                        ]
                    ),
                    "scenario_weight": float(
                        scenario_row[
                            "weight"
                        ]
                    ),
                    "d_optimality": float(
                        fisher.log_determinant
                    ),
                    "a_optimality": float(
                        np.trace(
                            fisher.covariance
                        )
                    ),
                    "e_optimality": float(
                        fisher.min_eigenvalue
                    ),
                    "condition_number": float(
                        fisher.condition_number
                    ),
                    "rank": int(
                        fisher.rank
                    ),
                }
            )

    long_table = pd.DataFrame(
        records
    )

    long_table.to_csv(
        placement_dir
        / "placement_scenario_metrics.csv",
        index=False,
    )

    ranking = robust_ranking_from_scenario_table(
        long_table,
        metric="d_optimality",
        conservative_quantile=0.10,
        ranking="conservative",
    )

    sensor_map = (
        long_table[
            [
                "design_id",
                "sensor_ids",
            ]
        ]
        .drop_duplicates(
            "design_id"
        )
    )

    ranking = ranking.merge(
        sensor_map,
        on="design_id",
        how="left",
        validate="one_to_one",
    )

    ranking.to_csv(
        placement_dir
        / "placement_robust_ranking.csv",
        index=False,
    )

    plot_robust_rankings(
        ranking,
        placement_dir
        / "placement_robust_ranking",
        title=(
            "Robust strainmeter placement under unknown geometry"
        ),
    )

    # Best-network spatial map.
    best_design_id = str(
        ranking.iloc[
            0
        ][
            "design_id"
        ]
    )

    network_map = {
        network.design_id: network
        for network in networks
    }

    best_network = network_map[
        best_design_id
    ]

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            8.5,
        )
    )

    ax.scatter(
        [
            location.x
            for location in candidates
        ],
        [
            location.y
            for location in candidates
        ],
        s=20,
        alpha=0.35,
        label="Candidate locations",
    )

    ax.scatter(
        [
            location.x
            for location
            in best_network.sensor_locations
        ],
        [
            location.y
            for location
            in best_network.sensor_locations
        ],
        s=120,
        marker="*",
        label="Best robust network",
    )

    for location in (
        best_network.sensor_locations
    ):
        ax.annotate(
            location.sensor_id,
            (
                location.x,
                location.y,
            ),
            xytext=(
                6,
                6,
            ),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "$x'$ (m)"
    )

    ax.set_ylabel(
        "$y'$ (m)"
    )

    ax.set_title(
        "Robust strainmeter placement with unknown body boundaries"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    save_figure(
        fig,
        placement_dir
        / "best_robust_placement_map",
        legend_handles=handles,
        legend_labels=labels,
        legend_ncol=2,
    )

    print(
        "  best robust placement: {0}".format(
            ranking.iloc[
                0
            ][
                "sensor_ids"
            ]
        )
    )

    return ranking


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    if args.sigma <= 0.0:
        raise ValueError(
            "--sigma must be positive."
        )

    if args.fd_step <= 0.0:
        raise ValueError(
            "--fd-step must be positive."
        )

    if args.regularization < 0.0:
        raise ValueError(
            "--regularization cannot be negative."
        )

    if args.robust_scenarios <= 0:
        raise ValueError(
            "--robust-scenarios must be positive."
        )

    configure_plot_style()

    np.random.seed(
        args.seed
    )

    if args.output_dir is None:
        output_dir = (
            REPO_ROOT
            / "results"
            / "oed"
            / "1darcy"
        )
    else:
        output_dir = (
            args.output_dir.resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "tables"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "figures"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    stations = load_station_metadata()

    observed = load_observed_data(
        stations
    )

    fixed = fixed_model_inputs()

    reference = dict(
        DEFAULT_REFERENCE
    )

    bounds = {
        key: list(
            value
        )
        for key, value
        in DEFAULT_BOUNDS.items()
    }

    robust_subset_sizes = (
        parse_subset_sizes(
            args.robust_subset_sizes,
            len(
                stations
            ),
        )
    )

    print(
        "=" * 78
    )

    print(
        "AVANT 1-DARCY PRODUCTION OPTIMAL EXPERIMENTAL DESIGN"
    )

    print(
        "=" * 78
    )

    print(
        "\nMode: {0}".format(
            args.mode
        )
    )

    print(
        "\nPhysical OED parameters:"
    )

    for name in PARAMETER_NAMES:
        print(
            "  {0:16s} reference={1:.8g} bounds={2}".format(
                name,
                reference[
                    name
                ],
                bounds[
                    name
                ],
            )
        )

    print(
        "\nFixed analytical inputs:"
    )

    for key, value in fixed.items():
        print(
            "  {0:8s} = {1}".format(
                key,
                value,
            )
        )

    print(
        "\nData:"
    )

    print(
        "  stations       = {0}".format(
            len(
                stations
            )
        )
    )

    print(
        "  time steps     = {0}".format(
            len(
                observed
            )
        )
    )

    print(
        "  strain channels= {0}".format(
            len(
                stations
            )
            * len(
                COMPONENTS
            )
        )
    )

    print(
        "  observations   = {0}".format(
            len(
                observed
            )
            * len(
                stations
            )
            * len(
                COMPONENTS
            )
        )
    )

    save_json(
        output_dir
        / "run_manifest.json",
        {
            "mode": args.mode,
            "physical_parameter_names": (
                PARAMETER_NAMES
            ),
            "reference_geometry": reference,
            "parameter_bounds": bounds,
            "parameter_scales": (
                parameter_scales_from_bounds(
                    bounds
                )
            ),
            "fixed_model_inputs": fixed,
            "observation_sigma_nstrain": (
                args.sigma
            ),
            "finite_difference_step": (
                args.fd_step
            ),
            "regularization": (
                args.regularization
            ),
            "robust_scenarios": (
                args.robust_scenarios
            ),
            "robust_subset_sizes": (
                robust_subset_sizes
            ),
            "placement": {
                "grid_nx": (
                    args.placement_grid_nx
                ),
                "grid_ny": (
                    args.placement_grid_ny
                ),
                "bounds": [
                    args.placement_xmin,
                    args.placement_xmax,
                    args.placement_ymin,
                    args.placement_ymax,
                ],
                "design_size": (
                    args.placement_size
                ),
                "candidate_networks": (
                    args.placement_networks
                ),
                "minimum_spacing_m": (
                    args.placement_min_spacing
                ),
                "geometry_scenarios": (
                    args.placement_scenarios
                ),
            },
            "seed": args.seed,
        },
    )

    start_time = _time.time()

    if args.mode in (
        "local",
        "all",
    ):
        run_local(
            output_dir,
            stations,
            observed,
            fixed,
            reference,
            bounds,
            args.sigma,
            args.fd_step,
            args.regularization,
        )

    if args.mode in (
        "existing",
        "all",
    ):
        run_existing(
            output_dir,
            stations,
            observed,
            fixed,
            reference,
            bounds,
            args.sigma,
            args.fd_step,
            args.regularization,
        )

    if args.mode in (
        "robust",
        "all",
    ):
        run_robust_existing(
            output_dir,
            stations,
            observed,
            fixed,
            bounds,
            robust_subset_sizes,
            args.sigma,
            args.fd_step,
            args.regularization,
            args.robust_scenarios,
            args.seed,
            boundary_mode=False,
        )

    if args.mode in (
        "boundary",
        "all",
    ):
        run_robust_existing(
            output_dir,
            stations,
            observed,
            fixed,
            bounds,
            robust_subset_sizes,
            args.sigma,
            args.fd_step,
            args.regularization,
            args.robust_scenarios,
            args.seed,
            boundary_mode=True,
        )

    if args.mode in (
        "placement",
        "all",
    ):
        run_placement(
            output_dir,
            observed,
            fixed,
            bounds,
            args,
        )

    elapsed = (
        _time.time()
        - start_time
    )

    save_json(
        output_dir
        / "completion_summary.json",
        {
            "completed": True,
            "mode": args.mode,
            "elapsed_seconds": elapsed,
            "output_directory": (
                str(
                    output_dir
                )
            ),
        },
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "OED ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "Elapsed time: {0:.1f} s".format(
            elapsed
        )
    )

    print(
        "Output directory:\n    {0}".format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
