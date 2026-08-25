#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production global sensitivity analysis for the AVANT 1-Darcy analytical
transient forward model.

This script adapts the validated sensitivity-analysis workflow to the
current production repository layout and Bayesian parameterization.

Primary physical sensitivity parameters
---------------------------------------
    a
    b
    h
    theta_deg
    x0_prime
    y0_prime

The Bayesian nuisance parameter

    log10_sigma_strain

is intentionally excluded from forward-model Sobol analysis because it is
a statistical residual/noise parameter rather than a physical input to the
analytical solution.

Canonical transient data
------------------------
    datasets/comsol/1darcy/metadata/stations.csv
    datasets/comsol/1darcy/processed/strain.csv

Fixed model parameters are read from
avant_model.data.multi_stations_input.read_input().

If a completed Bayesian run directory is supplied with --run-dir, its
manifest is used to establish the exact fixed-model values and prior bounds.
This is recommended for final production sensitivity analyses because it
guarantees that the sensitivity domain matches the Bayesian parameter domain.

Analyses
--------
1. One-at-a-time (OAT) parameter sweeps.
2. Selected 2-D RMSE response surfaces.
3. Random global parameter search.
4. Sobol first-order (S1) and total-order (ST) indices.
5. Sobol uncertainty estimates.
6. Global, component-wise, and station-wise Sobol summaries.
7. Parameter-vs-RMSE correlation diagnostics.
8. Publication-quality figures with legends below the axes.

Outputs
-------
The output directory contains CSV, JSON, PNG, and PDF results suitable for
later manuscript/poster use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
except ImportError as exc:
    raise ImportError(
        "SALib is required for the production Sobol analysis. "
        "Install it with: pip install SALib"
    ) from exc

# ============================================================================
# Repository imports
# ============================================================================

REPO_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avant_model.data import (
    multi_stations_input as input_data,
)

from avant_model.model.forward_model_multi_station import (
    forward_model_multi_station,
)


# ============================================================================
# Canonical definitions
# ============================================================================

PHYSICAL_PARAMETERS = (
    "a",
    "b",
    "h",
    "theta_deg",
    "x0_prime",
    "y0_prime",
)

COMPONENTS = (
    "eXX",
    "eYY",
    "eZZ",
    "eXY",
)

COMPONENT_LABELS = {
    "eXX": r"$\varepsilon_{xx}$",
    "eYY": r"$\varepsilon_{yy}$",
    "eZZ": r"$\varepsilon_{zz}$",
    "eXY": r"$\varepsilon_{xy}$",
}

PARAMETER_LABELS = {
    "a": r"$a$ (m)",
    "b": r"$b$ (m)",
    "h": r"$h$ (m)",
    "theta_deg": r"$\theta$ (deg)",
    "x0_prime": r"$x_0'$ (m)",
    "y0_prime": r"$y_0'$ (m)",
}

PARAMETER_COLORS = {
    "a": "#1f77b4",
    "b": "#ff7f0e",
    "h": "#2ca02c",
    "theta_deg": "#d62728",
    "x0_prime": "#9467bd",
    "y0_prime": "#8c564b",
}

STATION_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)

DEFAULT_SOBOL_BASE = 256
DEFAULT_OAT_POINTS = 17
DEFAULT_HEATMAP_POINTS = 31
DEFAULT_GLOBAL_SAMPLES = 1000
DEFAULT_SEED = 42


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production OAT, 2-D, global-search, and Sobol "
            "sensitivity analysis for the AVANT 1-Darcy analytical model."
        )
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional completed Bayesian run directory. If supplied, the "
            "run manifest fixes the exact Bayesian prior domain and fixed "
            "analytical-model parameters."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Sensitivity output directory. Default: "
            "results/sensitivity/1darcy."
        ),
    )

    parser.add_argument(
        "--sobol-base",
        type=int,
        default=DEFAULT_SOBOL_BASE,
        help=(
            "Base Sobol sample size. For six parameters and first/total "
            "order indices this gives N*(2D+2) model evaluations."
        ),
    )

    parser.add_argument(
        "--oat-points",
        type=int,
        default=DEFAULT_OAT_POINTS,
        help="Number of points in each one-at-a-time sweep.",
    )

    parser.add_argument(
        "--heatmap-points",
        type=int,
        default=DEFAULT_HEATMAP_POINTS,
        help="Number of points per axis for each 2-D response surface.",
    )

    parser.add_argument(
        "--global-samples",
        type=int,
        default=DEFAULT_GLOBAL_SAMPLES,
        help="Number of samples in the random global search.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--skip-oat",
        action="store_true",
        help="Skip one-at-a-time sweeps.",
    )

    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip 2-D response-surface plots.",
    )

    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Skip random global search.",
    )

    parser.add_argument(
        "--skip-sobol",
        action="store_true",
        help="Skip Sobol analysis.",
    )

    return parser.parse_args()


# ============================================================================
# Data loading
# ============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n    {path}"
        )


def save_json(path: Path, payload) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


def load_manifest(run_dir: Path | None) -> dict | None:
    if run_dir is None:
        return None

    manifest_path = (
        run_dir.resolve()
        / "run_manifest.json"
    )

    require_file(
        manifest_path,
        "Bayesian run manifest",
    )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_station_metadata(
    manifest: dict | None,
) -> pd.DataFrame:
    if manifest is not None:
        station_file = Path(
            manifest["data_contract"]["station_file"]
        )
    else:
        station_file = (
            REPO_ROOT
            / "datasets"
            / "comsol"
            / "1darcy"
            / "metadata"
            / "stations.csv"
        )

    require_file(
        station_file,
        "1-Darcy station metadata",
    )

    stations = pd.read_csv(
        station_file
    )

    stations.columns = [
        str(column).strip()
        for column in stations.columns
    ]

    required = {
        "station",
        "x",
        "y",
        "z",
    }

    missing = (
        required
        - set(stations.columns)
    )

    if missing:
        raise RuntimeError(
            "Station metadata is missing required columns: "
            + ", ".join(sorted(missing))
        )

    stations = stations.loc[
        :,
        [
            "station",
            "x",
            "y",
            "z",
        ],
    ].copy()

    stations["station"] = (
        stations["station"]
        .astype(str)
        .str.strip()
    )

    for column in (
        "x",
        "y",
        "z",
    ):
        stations[column] = pd.to_numeric(
            stations[column],
            errors="raise",
        )

    if stations["station"].duplicated().any():
        raise RuntimeError(
            "Station names must be unique."
        )

    if not np.all(
        np.isfinite(
            stations[
                ["x", "y", "z"]
            ].to_numpy(float)
        )
    ):
        raise RuntimeError(
            "Station coordinates contain non-finite values."
        )

    if manifest is not None:
        manifest_names = tuple(
            manifest["data_contract"].get(
                "station_names",
                [],
            )
        )

        actual_names = tuple(
            stations["station"].tolist()
        )

        if manifest_names and (
            actual_names != manifest_names
        ):
            raise RuntimeError(
                "Station metadata order does not match the Bayesian "
                "run manifest."
            )

    return stations


def load_observed_dataset(
    manifest: dict | None,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    if manifest is not None:
        observed_file = Path(
            manifest["data_contract"]["observed_file"]
        )
    else:
        observed_file = (
            REPO_ROOT
            / "datasets"
            / "comsol"
            / "1darcy"
            / "processed"
            / "strain.csv"
        )

    require_file(
        observed_file,
        "1-Darcy transient strain dataset",
    )

    observed = pd.read_csv(
        observed_file
    )

    observed.columns = [
        str(column).strip()
        for column in observed.columns
    ]

    if "time_s" not in observed.columns:
        raise RuntimeError(
            "Observed transient dataset must contain 'time_s'."
        )

    station_names = (
        stations["station"].tolist()
    )

    expected_channels = [
        f"{component}_{station}"
        for component in COMPONENTS
        for station in station_names
    ]

    missing = [
        channel
        for channel in expected_channels
        if channel not in observed.columns
    ]

    if missing:
        raise RuntimeError(
            "Observed transient dataset is missing channels:\n"
            + "\n".join(
                f"    {channel}"
                for channel in missing
            )
        )

    unexpected = [
        column
        for column in observed.columns
        if column != "time_s"
        and column not in expected_channels
    ]

    if unexpected:
        raise RuntimeError(
            "Observed transient dataset contains unexpected columns:\n"
            + "\n".join(
                f"    {column}"
                for column in unexpected
            )
        )

    observed = observed.loc[
        :,
        [
            "time_s",
            *expected_channels,
        ],
    ].copy()

    for column in observed.columns:
        observed[column] = pd.to_numeric(
            observed[column],
            errors="raise",
        )

    if observed.empty:
        raise RuntimeError(
            "Observed transient dataset is empty."
        )

    if not np.all(
        np.isfinite(
            observed.to_numpy(float)
        )
    ):
        raise RuntimeError(
            "Observed transient dataset contains non-finite values."
        )

    time = observed[
        "time_s"
    ].to_numpy(float)

    if not np.all(
        np.diff(time) >= 0.0
    ):
        raise RuntimeError(
            "time_s must be monotonically non-decreasing."
        )

    return observed


# ============================================================================
# Configuration
# ============================================================================

def safe_float(
    value,
    fallback: float,
) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def build_configuration(
    manifest: dict | None,
) -> dict:
    """
    Build the baseline analytical-model configuration and the six physical
    sensitivity bounds.

    If --run-dir is supplied, the fixed model inputs and prior bounds are
    taken from the Bayesian manifest so the sensitivity study exactly matches
    the Bayesian parameter domain.
    """

    params = input_data.read_input()

    # Default fixed values from the current analytical input module.
    fixed = {
        "c": safe_float(
            params.get("c_fixed", 3.125),
            3.125,
        ),
        "E": safe_float(
            params.get("E", 8.0e9),
            8.0e9,
        ),
        "nu": safe_float(
            params.get("nu", 0.35),
            0.35,
        ),
        "pmax": safe_float(
            params.get("pmax", 9.99310e5),
            9.99310e5,
        ),
        "tpeak": safe_float(
            params.get("tpeak", 3.5e5),
            3.5e5,
        ),
        "d": safe_float(
            params.get("d", 0.4),
            0.4,
        ),
        "alpha": safe_float(
            params.get("alpha", 0.8),
            0.8,
        ),
    }

    # Current Bayesian baseline.
    baseline = {
        "a": 190.0,
        "b": 325.0,
        "h": 520.0,
        "theta_deg": 15.0,
        "x0_prime": 45.0,
        "y0_prime": 170.0,
    }

    # If a Bayesian run manifest exists, use its fixed analytical values and
    # the MAP point from posterior_run_summary.json when available.
    prior_bounds = None
    map_values = {}

    if manifest is not None:

        fixed_manifest = (
            manifest.get(
                "fixed_model_inputs",
                {},
            )
        )

        for key in (
            "c",
            "E",
            "nu",
            "pmax",
            "tpeak",
            "d",
            "alpha",
        ):
            if key in fixed_manifest:
                fixed[key] = float(
                    fixed_manifest[key]
                )

        prior_bounds = (
            manifest.get(
                "priors",
                {},
            )
        )

        posterior_summary_path = (
            Path(
                manifest.get(
                    "output_directory",
                    "",
                )
            )
            / "posterior_run_summary.json"
        )

        if posterior_summary_path.exists():

            with open(
                posterior_summary_path,
                "r",
                encoding="utf-8",
            ) as handle:
                posterior_summary = json.load(
                    handle
                )

            map_values = (
                posterior_summary.get(
                    "map",
                    {},
                )
            )

            for key in PHYSICAL_PARAMETERS:
                if key in map_values:
                    baseline[key] = float(
                        map_values[key]
                    )

    # If there is no Bayesian manifest, use scientifically meaningful broad
    # production ranges based on the current transient analytical workflow.
    if prior_bounds is None:

        prior_bounds = {
            "a": [
                40.0,
                900.0,
            ],
            "b": [
                20.0,
                1000.0,
            ],
            "h": [
                50.0,
                900.0,
            ],
            "theta_deg": [
                -90.0,
                90.0,
            ],
            "x0_prime": [
                -900.0,
                900.0,
            ],
            "y0_prime": [
                -900.0,
                900.0,
            ],
        }

    # Retain only physical parameters.
    prior_bounds = {
        key: [
            float(value[0]),
            float(value[1]),
        ]
        for key, value in prior_bounds.items()
        if key in PHYSICAL_PARAMETERS
    }

    return {
        "baseline": baseline,
        "fixed": fixed,
        "prior_bounds": prior_bounds,
    }


# ============================================================================
# Forward model
# ============================================================================

def build_channel_order(
    stations: pd.DataFrame,
) -> List[str]:
    station_names = (
        stations["station"].tolist()
    )

    return [
        f"{component}_{station}"
        for component in COMPONENTS
        for station in station_names
    ]


def evaluate_model(
    sample: Dict[str, float],
    fixed: Dict[str, float],
    observed: pd.DataFrame,
    stations: pd.DataFrame,
) -> dict:
    station_names = (
        stations["station"]
        .to_numpy(str)
    )

    time = observed[
        "time_s"
    ].to_numpy(float)

    x_prime = stations[
        "x"
    ].to_numpy(float)

    y_prime = stations[
        "y"
    ].to_numpy(float)

    z = stations[
        "z"
    ].to_numpy(float)

    df_pred = (
        forward_model_multi_station(
            pmax=fixed["pmax"],
            tpeak=fixed["tpeak"],
            d=fixed["d"],
            time=time,
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=sample["x0_prime"],
            y0_prime=sample["y0_prime"],
            z=z,
            a=sample["a"],
            b=sample["b"],
            c=fixed["c"],
            nu=fixed["nu"],
            h=sample["h"],
            E=fixed["E"],
            theta_deg=sample["theta_deg"],
            alpha=fixed["alpha"],
            station_names=station_names,
            debug=False,
        )
    )

    channels = build_channel_order(
        stations
    )

    prediction = (
        df_pred[
            channels
        ].to_numpy(float)
    )

    observation = (
        observed[
            channels
        ].to_numpy(float)
    )

    residual = (
        prediction
        - observation
    )

    global_rmse = float(
        np.sqrt(
            np.mean(
                residual**2
            )
        )
    )

    station_names_list = (
        stations["station"].tolist()
    )

    station_rmse = {}

    for station_index, station in enumerate(
        station_names_list
    ):
        channel_indices = [
            component_index
            * len(station_names_list)
            + station_index
            for component_index in range(
                len(COMPONENTS)
            )
        ]

        station_residual = residual[
            :,
            channel_indices,
        ]

        station_rmse[
            station
        ] = float(
            np.sqrt(
                np.mean(
                    station_residual**2
                )
            )
        )

    component_rmse = {}

    for component_index, component in enumerate(
        COMPONENTS
    ):
        start = (
            component_index
            * len(station_names_list)
        )

        stop = (
            start
            + len(station_names_list)
        )

        component_residual = residual[
            :,
            start:stop,
        ]

        component_rmse[
            component
        ] = float(
            np.sqrt(
                np.mean(
                    component_residual**2
                )
            )
        )

    return {
        "prediction": prediction,
        "global_rmse": global_rmse,
        "station_rmse": station_rmse,
        "component_rmse": component_rmse,
    }


# ============================================================================
# Publication plotting helpers
# ============================================================================

def configure_publication_style() -> None:
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
            "grid.alpha": 0.22,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
        }
    )


def save_figure_with_legend_below(
    fig,
    handles,
    labels,
    path: Path,
    ncol: int = 2,
) -> None:
    if handles:
        fig.subplots_adjust(
            bottom=0.20
        )

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(
                0.5,
                0.025,
            ),
            ncol=ncol,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.8,
        )

    fig.savefig(
        path
    )

    pdf_path = path.with_suffix(
        ".pdf"
    )

    fig.savefig(
        pdf_path
    )

    plt.close(fig)


# ============================================================================
# OAT analysis
# ============================================================================

def build_oat_grid(
    bounds: Sequence[float],
    points: int,
) -> np.ndarray:
    low, high = bounds

    if points < 3:
        raise ValueError(
            "OAT points must be at least 3."
        )

    return np.linspace(
        low,
        high,
        points,
    )


def run_oat(
    baseline: Dict[str, float],
    bounds: Dict[str, Sequence[float]],
    fixed: Dict[str, float],
    observed: pd.DataFrame,
    stations: pd.DataFrame,
    output_dir: Path,
    n_points: int,
) -> pd.DataFrame:

    records = []
    results = {}

    for parameter in PHYSICAL_PARAMETERS:

        values = build_oat_grid(
            bounds[parameter],
            n_points,
        )

        rmse_values = []

        for value in values:

            sample = baseline.copy()

            sample[
                parameter
            ] = float(value)

            result = evaluate_model(
                sample,
                fixed,
                observed,
                stations,
            )

            rmse_values.append(
                result["global_rmse"]
            )

            record = {
                "parameter": parameter,
                "parameter_label": PARAMETER_LABELS[
                    parameter
                ],
                "value": float(value),
                "global_rmse": float(
                    result[
                        "global_rmse"
                    ]
                ),
                "baseline_value": float(
                    baseline[
                        parameter
                    ]
                ),
            }

            for component in COMPONENTS:
                record[
                    f"rmse_{component}"
                ] = float(
                    result[
                        "component_rmse"
                    ][
                        component
                    ]
                )

            for station in stations[
                "station"
            ]:
                record[
                    f"rmse_{station}"
                ] = float(
                    result[
                        "station_rmse"
                    ][
                        station
                    ]
                )

            records.append(
                record
            )

        results[
            parameter
        ] = (
            values,
            np.asarray(
                rmse_values
            )
        )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        output_dir
        / "sensitivity_oat_results.csv",
        index=False,
    )

    # Summary.
    summary_rows = []

    baseline_result = evaluate_model(
        baseline,
        fixed,
        observed,
        stations,
    )

    for parameter in PHYSICAL_PARAMETERS:

        parameter_df = dataframe[
            dataframe[
                "parameter"
            ]
            == parameter
        ]

        best_index = (
            parameter_df[
                "global_rmse"
            ].idxmin()
        )

        best_row = (
            parameter_df.loc[
                best_index
            ]
        )

        baseline_rmse = (
            baseline_result[
                "global_rmse"
            ]
        )

        best_rmse = float(
            best_row[
                "global_rmse"
            ]
        )

        summary_rows.append(
            {
                "parameter": parameter,
                "baseline_value": float(
                    baseline[
                        parameter
                    ]
                ),
                "best_oat_value": float(
                    best_row[
                        "value"
                    ]
                ),
                "baseline_rmse": float(
                    baseline_rmse
                ),
                "best_oat_rmse": best_rmse,
                "improvement_percent": (
                    100.0
                    * (
                        baseline_rmse
                        - best_rmse
                    )
                    / max(
                        baseline_rmse,
                        1.0e-30,
                    )
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        output_dir
        / "sensitivity_oat_summary.csv",
        index=False,
    )

    # High-quality multi-panel figure.
    n_parameters = len(
        PHYSICAL_PARAMETERS
    )

    n_columns = 3

    n_rows = int(
        math.ceil(
            n_parameters
            / n_columns
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            17,
            5.0 * n_rows,
        ),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="navy",
            marker="o",
            linewidth=2.2,
            label="Global strain RMSE",
        ),
        Line2D(
            [0],
            [0],
            color="crimson",
            linestyle="--",
            linewidth=2.0,
            label="Baseline value",
        ),
    ]

    for index, parameter in enumerate(
        PHYSICAL_PARAMETERS
    ):

        ax = axes_flat[
            index
        ]

        parameter_df = dataframe[
            dataframe[
                "parameter"
            ]
            == parameter
        ]

        ax.plot(
            parameter_df[
                "value"
            ],
            parameter_df[
                "global_rmse"
            ],
            color="navy",
            marker="o",
            markersize=4.5,
            linewidth=2.0,
        )

        ax.axvline(
            baseline[
                parameter
            ],
            color="crimson",
            linestyle="--",
            linewidth=1.8,
        )

        ax.set_xlabel(
            PARAMETER_LABELS[
                parameter
            ]
        )

        ax.set_ylabel(
            "Strain RMSE (nε)"
        )

        ax.set_title(
            PARAMETER_LABELS[
                parameter
            ]
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.20,
        )

    for index in range(
        n_parameters,
        len(axes_flat),
    ):
        axes_flat[
            index
        ].axis("off")

    fig.suptitle(
        "One-at-a-time global sensitivity of the AVANT analytical model",
        y=0.995,
    )

    save_figure_with_legend_below(
        fig,
        legend_handles,
        [
            "Global strain RMSE",
            "Baseline value",
        ],
        output_dir
        / "sensitivity_oat_global_rmse.png",
        ncol=2,
    )

    # Heatmap-style station sensitivity summaries.
    sensitivity_matrix = []

    for parameter in PHYSICAL_PARAMETERS:

        parameter_df = dataframe[
            dataframe[
                "parameter"
            ]
            == parameter
        ]

        # Use normalized RMSE range as a simple OAT sensitivity score.
        scores = []

        for station in stations[
            "station"
        ]:

            column = (
                f"rmse_{station}"
            )

            values = parameter_df[
                column
            ].to_numpy()

            denominator = max(
                np.mean(values),
                1.0e-30,
            )

            score = (
                np.max(values)
                - np.min(values)
            ) / denominator

            scores.append(
                score
            )

        sensitivity_matrix.append(
            scores
        )

    sensitivity_matrix = np.asarray(
        sensitivity_matrix
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    image = ax.imshow(
        sensitivity_matrix,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xticks(
        np.arange(
            len(stations)
        )
    )

    ax.set_xticklabels(
        stations[
            "station"
        ].tolist()
    )

    ax.set_yticks(
        np.arange(
            len(PHYSICAL_PARAMETERS)
        )
    )

    ax.set_yticklabels(
        [
            PARAMETER_LABELS[
                parameter
            ]
            for parameter
            in PHYSICAL_PARAMETERS
        ]
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Normalized OAT sensitivity"
    )

    ax.set_title(
        "Station-level one-at-a-time sensitivity"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "sensitivity_oat_station_heatmap.png"
    )

    fig.savefig(
        output_dir
        / "sensitivity_oat_station_heatmap.pdf"
    )

    plt.close(fig)

    return dataframe


# ============================================================================
# 2-D response surfaces
# ============================================================================

def run_2d_surface(
    parameter_x: str,
    parameter_y: str,
    baseline: Dict[str, float],
    bounds: Dict[str, Sequence[float]],
    fixed: Dict[str, float],
    observed: pd.DataFrame,
    stations: pd.DataFrame,
    output_dir: Path,
    points: int,
) -> None:

    x_values = np.linspace(
        bounds[
            parameter_x
        ][0],
        bounds[
            parameter_x
        ][1],
        points,
    )

    y_values = np.linspace(
        bounds[
            parameter_y
        ][0],
        bounds[
            parameter_y
        ][1],
        points,
    )

    surface = np.zeros(
        (
            len(y_values),
            len(x_values),
        )
    )

    for iy, y_value in enumerate(
        y_values
    ):

        for ix, x_value in enumerate(
            x_values
        ):

            sample = baseline.copy()

            sample[
                parameter_x
            ] = float(x_value)

            sample[
                parameter_y
            ] = float(y_value)

            result = evaluate_model(
                sample,
                fixed,
                observed,
                stations,
            )

            surface[
                iy,
                ix
            ] = result[
                "global_rmse"
            ]

    X, Y = np.meshgrid(
        x_values,
        y_values,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            7.8,
        )
    )

    image = ax.pcolormesh(
        X,
        Y,
        surface,
        shading="auto",
        cmap="viridis",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Global strain RMSE (nε)"
    )

    ax.scatter(
        [
            baseline[
                parameter_x
            ]
        ],
        [
            baseline[
                parameter_y
            ]
        ],
        color="red",
        marker="x",
        s=140,
        linewidths=3.0,
    )

    # Global minimum within the surface.
    minimum_index = np.unravel_index(
        np.argmin(surface),
        surface.shape,
    )

    ax.scatter(
        [
            x_values[
                minimum_index[1]
            ]
        ],
        [
            y_values[
                minimum_index[0]
            ]
        ],
        color="white",
        marker="*",
        s=180,
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_xlabel(
        PARAMETER_LABELS[
            parameter_x
        ]
    )

    ax.set_ylabel(
        PARAMETER_LABELS[
            parameter_y
        ]
    )

    ax.set_title(
        "2-D global sensitivity response surface"
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="red",
            linewidth=0,
            markersize=10,
            label="Baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="white",
            markeredgecolor="black",
            linewidth=0,
            markersize=11,
            label="Surface minimum",
        ),
    ]

    save_figure_with_legend_below(
        fig,
        handles,
        [
            "Baseline",
            "Surface minimum",
        ],
        output_dir
        / (
            f"sensitivity_2d_"
            f"{parameter_x}_"
            f"{parameter_y}.png"
        ),
        ncol=2,
    )

    surface_table = pd.DataFrame(
        surface,
        index=y_values,
        columns=x_values,
    )

    surface_table.index.name = (
        parameter_y
    )

    surface_table.to_csv(
        output_dir
        / (
            f"sensitivity_2d_"
            f"{parameter_x}_"
            f"{parameter_y}.csv"
        )
    )


# ============================================================================
# Random global search
# ============================================================================

def run_global_search(
    baseline: Dict[str, float],
    bounds: Dict[str, Sequence[float]],
    fixed: Dict[str, float],
    observed: pd.DataFrame,
    stations: pd.DataFrame,
    output_dir: Path,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:

    rng = np.random.default_rng(
        seed
    )

    samples = {}

    for parameter in PHYSICAL_PARAMETERS:

        low, high = (
            bounds[
                parameter
            ]
        )

        samples[
            parameter
        ] = rng.uniform(
            low,
            high,
            size=n_samples,
        )

    records = []

    best_rmse = np.inf
    best_sample = None

    for index in range(
        n_samples
    ):

        sample = {
            parameter: float(
                samples[
                    parameter
                ][index]
            )
            for parameter
            in PHYSICAL_PARAMETERS
        }

        result = evaluate_model(
            sample,
            fixed,
            observed,
            stations,
        )

        record = {
            **sample,
            "global_rmse": float(
                result[
                    "global_rmse"
                ]
            ),
        }

        for component in COMPONENTS:
            record[
                f"rmse_{component}"
            ] = float(
                result[
                    "component_rmse"
                ][
                    component
                ]
            )

        records.append(
            record
        )

        if (
            result[
                "global_rmse"
            ]
            < best_rmse
        ):

            best_rmse = float(
                result[
                    "global_rmse"
                ]
            )

            best_sample = (
                sample.copy()
            )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        output_dir
        / "global_search_results.csv",
        index=False,
    )

    with open(
        output_dir
        / "global_search_best.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            {
                "global_rmse": best_rmse,
                "parameters": best_sample,
            },
            handle,
            indent=2,
        )

    # Parameter vs RMSE plots.
    for parameter in PHYSICAL_PARAMETERS:

        fig, ax = plt.subplots(
            figsize=(
                8.5,
                6.5,
            )
        )

        ax.scatter(
            dataframe[
                parameter
            ],
            dataframe[
                "global_rmse"
            ],
            s=16,
            alpha=0.35,
            rasterized=True,
            color=PARAMETER_COLORS[
                parameter
            ],
        )

        ax.axvline(
            baseline[
                parameter
            ],
            color="crimson",
            linestyle="--",
            linewidth=1.8,
        )

        ax.set_xlabel(
            PARAMETER_LABELS[
                parameter
            ]
        )

        ax.set_ylabel(
            "Global strain RMSE (nε)"
        )

        ax.set_title(
            f"Global search: RMSE vs "
            f"{PARAMETER_LABELS[parameter]}"
        )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=PARAMETER_COLORS[
                    parameter
                ],
                linewidth=0,
                markersize=6,
                label="Random samples",
            ),
            Line2D(
                [0],
                [0],
                color="crimson",
                linestyle="--",
                linewidth=1.8,
                label="Baseline",
            ),
        ]

        save_figure_with_legend_below(
            fig,
            handles,
            [
                "Random samples",
                "Baseline",
            ],
            output_dir
            / (
                f"global_search_rmse_vs_"
                f"{parameter}.png"
            ),
            ncol=2,
        )

    correlation_columns = [
        *PHYSICAL_PARAMETERS,
        "global_rmse",
    ]

    correlation = dataframe[
        correlation_columns
    ].corr()

    correlation.to_csv(
        output_dir
        / "global_search_correlation_matrix.csv"
    )

    fig, ax = plt.subplots(
        figsize=(
            9.5,
            7.5,
        )
    )

    matrix = correlation.to_numpy()

    image = ax.imshow(
        matrix,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    labels = [
        PARAMETER_LABELS[
            parameter
        ]
        for parameter
        in PHYSICAL_PARAMETERS
    ] + [
        "Global RMSE"
    ]

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        labels
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Pearson correlation"
    )

    ax.set_title(
        "Global-search parameter/RMSE correlation"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "global_search_correlation_matrix.png"
    )

    fig.savefig(
        output_dir
        / "global_search_correlation_matrix.pdf"
    )

    plt.close(fig)

    return dataframe


# ============================================================================
# Sobol analysis
# ============================================================================

def run_sobol_analysis(
    baseline: Dict[str, float],
    bounds: Dict[str, Sequence[float]],
    fixed: Dict[str, float],
    observed: pd.DataFrame,
    stations: pd.DataFrame,
    output_dir: Path,
    base_sample_size: int,
    seed: int,
) -> None:

    problem = {
        "num_vars": len(
            PHYSICAL_PARAMETERS
        ),
        "names": list(
            PHYSICAL_PARAMETERS
        ),
        "bounds": [
            list(
                bounds[
                    parameter
                ]
            )
            for parameter
            in PHYSICAL_PARAMETERS
        ],
    }

    # Sobol/SALib generates a deterministic sequence when given a seed.
    sample_matrix = (
        sobol_sample.sample(
            problem,
            base_sample_size,
            calc_second_order=False,
            seed=seed,
        )
    )

    print(
        f"[INFO] Sobol model evaluations: "
        f"{sample_matrix.shape[0]}"
    )

    n_samples = (
        sample_matrix.shape[0]
    )

    global_output = np.zeros(
        n_samples,
        dtype=float,
    )

    component_output = {
        component: np.zeros(
            n_samples,
            dtype=float,
        )
        for component in COMPONENTS
    }

    station_output = {
        station: np.zeros(
            n_samples,
            dtype=float,
        )
        for station in stations[
            "station"
        ]
    }

    for index in range(
        n_samples
    ):

        sample = {
            parameter: float(
                sample_matrix[
                    index,
                    parameter_index,
                ]
            )
            for parameter_index, parameter
            in enumerate(
                PHYSICAL_PARAMETERS
            )
        }

        result = evaluate_model(
            sample,
            fixed,
            observed,
            stations,
        )

        global_output[
            index
        ] = result[
            "global_rmse"
        ]

        for component in COMPONENTS:
            component_output[
                component
            ][
                index
            ] = result[
                "component_rmse"
            ][
                component
            ]

        for station in stations[
            "station"
        ]:
            station_output[
                station
            ][
                index
            ] = result[
                "station_rmse"
            ][
                station
            ]

    # Save samples/outputs for complete reproducibility.
    sample_dataframe = pd.DataFrame(
        sample_matrix,
        columns=PHYSICAL_PARAMETERS,
    )

    sample_dataframe[
        "global_rmse"
    ] = global_output

    for component in COMPONENTS:
        sample_dataframe[
            f"rmse_{component}"
        ] = component_output[
            component
        ]

    for station in stations[
        "station"
    ]:
        sample_dataframe[
            f"rmse_{station}"
        ] = station_output[
            station
        ]

    sample_dataframe.to_csv(
        output_dir
        / "sobol_sample_evaluations.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Analyze one output metric at a time.
    # ------------------------------------------------------------------

    all_outputs = {
        "global_rmse": global_output,
    }

    all_outputs.update(
        {
            f"component_{component}_rmse":
                component_output[
                    component
                ]
            for component in COMPONENTS
        }
    )

    all_outputs.update(
        {
            f"station_{station}_rmse":
                station_output[
                    station
                ]
            for station in stations[
                "station"
            ]
        }
    )

    sobol_records = []

    for output_name, output_values in (
        all_outputs.items()
    ):

        analysis = sobol_analyze.analyze(
            problem,
            output_values,
            calc_second_order=False,
            print_to_console=False,
            seed=seed,
        )

        S1 = np.asarray(
            analysis["S1"],
            dtype=float,
        )

        ST = np.asarray(
            analysis["ST"],
            dtype=float,
        )

        S1_conf = np.asarray(
            analysis["S1_conf"],
            dtype=float,
        )

        ST_conf = np.asarray(
            analysis["ST_conf"],
            dtype=float,
        )

        for index, parameter in enumerate(
            PHYSICAL_PARAMETERS
        ):

            sobol_records.append(
                {
                    "output": output_name,
                    "parameter": parameter,
                    "S1": float(
                        S1[index]
                    ),
                    "S1_conf": float(
                        S1_conf[index]
                    ),
                    "ST": float(
                        ST[index]
                    ),
                    "ST_conf": float(
                        ST_conf[index]
                    ),
                }
            )

    sobol_dataframe = pd.DataFrame(
        sobol_records
    )

    sobol_dataframe.to_csv(
        output_dir
        / "sobol_indices_all_outputs.csv",
        index=False,
    )

    # Global-only table for convenient manuscript use.
    global_sobol = (
        sobol_dataframe[
            sobol_dataframe[
                "output"
            ]
            == "global_rmse"
        ]
        .copy()
        .sort_values(
            "ST",
            ascending=False,
        )
    )

    global_sobol.to_csv(
        output_dir
        / "sobol_global_rmse_ranking.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Global Sobol plot.
    # ------------------------------------------------------------------

    ordered = (
        global_sobol.sort_values(
            "ST",
            ascending=True,
        )
    )

    y = np.arange(
        len(ordered)
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            6.8,
        )
    )

    ax.barh(
        y - 0.18,
        ordered["S1"],
        height=0.34,
        color="steelblue",
        alpha=0.92,
        xerr=ordered["S1_conf"],
        capsize=4,
        label="First-order $S_1$",
    )

    ax.barh(
        y + 0.18,
        ordered["ST"],
        height=0.34,
        color="darkorange",
        alpha=0.92,
        xerr=ordered["ST_conf"],
        capsize=4,
        label="Total-order $S_T$",
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        [
            PARAMETER_LABELS[
                parameter
            ]
            for parameter
            in ordered[
                "parameter"
            ]
        ]
    )

    ax.set_xlabel(
        "Sobol index"
    )

    ax.set_title(
        "Global Sobol sensitivity of 1-Darcy strain RMSE"
    )

    ax.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.20,
    )

    save_figure_with_legend_below(
        fig,
        *(
            ax.get_legend_handles_labels()
        ),
        output_dir
        / "sobol_global_rmse_indices.png",
        ncol=2,
    )

    # ------------------------------------------------------------------
    # Component heatmap.
    # ------------------------------------------------------------------

    component_names = [
        f"component_{component}_rmse"
        for component in COMPONENTS
    ]

    component_matrix = np.zeros(
        (
            len(COMPONENTS),
            len(PHYSICAL_PARAMETERS),
        )
    )

    for row, output_name in enumerate(
        component_names
    ):

        subset = sobol_dataframe[
            sobol_dataframe[
                "output"
            ]
            == output_name
        ]

        for column, parameter in enumerate(
            PHYSICAL_PARAMETERS
        ):

            value = subset.loc[
                subset[
                    "parameter"
                ]
                == parameter,
                "ST",
            ]

            component_matrix[
                row,
                column,
            ] = float(
                value.iloc[0]
            )

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            6.5,
        )
    )

    image = ax.imshow(
        component_matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
    )

    ax.set_xticks(
        np.arange(
            len(PHYSICAL_PARAMETERS)
        )
    )

    ax.set_xticklabels(
        [
            PARAMETER_LABELS[
                parameter
            ]
            for parameter
            in PHYSICAL_PARAMETERS
        ],
        rotation=30,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(COMPONENTS)
        )
    )

    ax.set_yticklabels(
        [
            COMPONENT_LABELS[
                component
            ]
            for component
            in COMPONENTS
        ]
    )

    for row in range(
        component_matrix.shape[0]
    ):
        for column in range(
            component_matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                f"{component_matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Total-order $S_T$"
    )

    ax.set_title(
        "Component-wise Sobol total-order sensitivity"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "sobol_component_total_order_heatmap.png"
    )

    fig.savefig(
        output_dir
        / "sobol_component_total_order_heatmap.pdf"
    )

    plt.close(fig)

    # ------------------------------------------------------------------
    # Station heatmap.
    # ------------------------------------------------------------------

    station_names = (
        stations[
            "station"
        ].tolist()
    )

    station_matrix = np.zeros(
        (
            len(station_names),
            len(PHYSICAL_PARAMETERS),
        )
    )

    for row, station in enumerate(
        station_names
    ):

        output_name = (
            f"station_{station}_rmse"
        )

        subset = sobol_dataframe[
            sobol_dataframe[
                "output"
            ]
            == output_name
        ]

        for column, parameter in enumerate(
            PHYSICAL_PARAMETERS
        ):

            value = subset.loc[
                subset[
                    "parameter"
                ]
                == parameter,
                "ST",
            ]

            station_matrix[
                row,
                column,
            ] = float(
                value.iloc[0]
            )

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            7.0,
        )
    )

    image = ax.imshow(
        station_matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
    )

    ax.set_xticks(
        np.arange(
            len(PHYSICAL_PARAMETERS)
        )
    )

    ax.set_xticklabels(
        [
            PARAMETER_LABELS[
                parameter
            ]
            for parameter
            in PHYSICAL_PARAMETERS
        ],
        rotation=30,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(station_names)
        )
    )

    ax.set_yticklabels(
        station_names
    )

    for row in range(
        station_matrix.shape[0]
    ):
        for column in range(
            station_matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                f"{station_matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Total-order $S_T$"
    )

    ax.set_title(
        "Station-wise Sobol total-order sensitivity"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "sobol_station_total_order_heatmap.png"
    )

    fig.savefig(
        output_dir
        / "sobol_station_total_order_heatmap.pdf"
    )

    plt.close(fig)

    # ------------------------------------------------------------------
    # JSON provenance
    # ------------------------------------------------------------------

    save_json(
        output_dir
        / "sobol_run_manifest.json",
        {
            "parameter_names": list(
                PHYSICAL_PARAMETERS
            ),
            "bounds": {
                parameter: list(
                    bounds[
                        parameter
                    ]
                )
                for parameter
                in PHYSICAL_PARAMETERS
            },
            "base_sample_size": int(
                base_sample_size
            ),
            "model_evaluations": int(
                sample_matrix.shape[0]
            ),
            "seed": int(seed),
            "second_order": False,
            "outputs": list(
                all_outputs.keys()
            ),
        },
    )


# ============================================================================
# Representative time-series sensitivity figure
# ============================================================================

def representative_time_series(
    baseline: Dict[str, float],
    bounds: Dict[str, Sequence[float]],
    fixed: Dict[str, float],
    observed: pd.DataFrame,
    stations: pd.DataFrame,
    output_dir: Path,
) -> None:

    station = (
        stations[
            "station"
        ].iloc[-1]
    )

    component = "eZZ"

    channel = (
        f"{component}_{station}"
    )

    station_names = (
        stations[
            "station"
        ].tolist()
    )

    channel_order = (
        build_channel_order(
            stations
        )
    )

    channel_index = (
        channel_order.index(
            channel
        )
    )

    baseline_prediction = (
        evaluate_model(
            baseline,
            fixed,
            observed,
            stations,
        )["prediction"][
            :,
            channel_index,
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            6.8,
        )
    )

    time_hours = (
        observed[
            "time_s"
        ].to_numpy(float)
        / 3600.0
    )

    ax.plot(
        time_hours,
        observed[
            channel
        ].to_numpy(float),
        color="black",
        linewidth=1.8,
        label="Observed",
    )

    ax.plot(
        time_hours,
        baseline_prediction,
        color="royalblue",
        linewidth=2.2,
        linestyle="--",
        label="Baseline analytical model",
    )

    representative_parameters = (
        [
            "a",
            "b",
            "h",
            "theta_deg",
            "x0_prime",
            "y0_prime",
        ]
    )

    for parameter in representative_parameters:

        sample = baseline.copy()

        low, high = (
            bounds[
                parameter
            ]
        )

        sample[
            parameter
        ] = float(
            low
        )

        low_prediction = (
            evaluate_model(
                sample,
                fixed,
                observed,
                stations,
            )[
                "prediction"
            ][
                :,
                channel_index,
            ]
        )

        sample[
            parameter
        ] = float(
            high
        )

        high_prediction = (
            evaluate_model(
                sample,
                fixed,
                observed,
                stations,
            )[
                "prediction"
            ][
                :,
                channel_index,
            ]
        )

        ax.plot(
            time_hours,
            low_prediction,
            color=PARAMETER_COLORS[
                parameter
            ],
            linewidth=1.3,
            alpha=0.65,
        )

        ax.plot(
            time_hours,
            high_prediction,
            color=PARAMETER_COLORS[
                parameter
            ],
            linewidth=1.3,
            alpha=0.65,
            linestyle=":",
        )

    parameter_handles = [
        Line2D(
            [0],
            [0],
            color=PARAMETER_COLORS[
                parameter
            ],
            linewidth=2.0,
            label=(
                PARAMETER_LABELS[
                    parameter
                ]
                + " low/high"
            ),
        )
        for parameter
        in representative_parameters
    ]

    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            label="Observed",
        ),
        Line2D(
            [0],
            [0],
            color="royalblue",
            linestyle="--",
            linewidth=2.2,
            label="Baseline analytical model",
        ),
    ] + parameter_handles

    ax.set_xlabel(
        "Time (h)"
    )

    ax.set_ylabel(
        "Strain (nε)"
    )

    ax.set_title(
        f"Representative sensitivity response: "
        f"{channel}"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.20,
    )

    save_figure_with_legend_below(
        fig,
        handles,
        [
            handle.get_label()
            for handle in handles
        ],
        output_dir
        / "sensitivity_representative_time_series.png",
        ncol=3,
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    if args.sobol_base < 16:
        raise ValueError(
            "--sobol-base must be at least 16."
        )

    if args.oat_points < 3:
        raise ValueError(
            "--oat-points must be at least 3."
        )

    if args.heatmap_points < 3:
        raise ValueError(
            "--heatmap-points must be at least 3."
        )

    if args.global_samples < 10:
        raise ValueError(
            "--global-samples must be at least 10."
        )

    np.random.seed(
        args.seed
    )

    configure_publication_style()

    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else None
    )

    manifest = load_manifest(
        run_dir
    )

    if args.output_dir is None:
        output_dir = (
            REPO_ROOT
            / "results"
            / "sensitivity"
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

    stations = load_station_metadata(
        manifest
    )

    observed = load_observed_dataset(
        manifest,
        stations,
    )

    configuration = build_configuration(
        manifest
    )

    baseline = configuration[
        "baseline"
    ]

    fixed = configuration[
        "fixed"
    ]

    bounds = configuration[
        "prior_bounds"
    ]

    print(
        "=" * 78
    )

    print(
        "AVANT 1-DARCY GLOBAL SENSITIVITY ANALYSIS"
    )

    print(
        "=" * 78
    )

    print(
        "\nPhysical sensitivity parameters:"
    )

    for parameter in PHYSICAL_PARAMETERS:
        print(
            f"  {parameter:16s} "
            f"baseline={baseline[parameter]:.8g} "
            f"bounds={bounds[parameter]}"
        )

    print(
        "\nFixed analytical parameters:"
    )

    for key in (
        "c",
        "E",
        "nu",
        "pmax",
        "tpeak",
        "d",
        "alpha",
    ):
        print(
            f"  {key:8s} = {fixed[key]}"
        )

    print(
        "\nData:"
    )

    print(
        f"  stations = "
        f"{len(stations)}"
    )

    print(
        f"  time steps = "
        f"{len(observed)}"
    )

    print(
        f"  strain channels = "
        f"{4 * len(stations)}"
    )

    # ------------------------------------------------------------------
    # Save configuration
    # ------------------------------------------------------------------

    save_json(
        output_dir
        / "sensitivity_run_manifest.json",
        {
            "run_dir": (
                None
                if run_dir is None
                else str(run_dir)
            ),
            "station_file": str(
                (
                    Path(
                        manifest["data_contract"]["station_file"]
                    )
                    if manifest is not None
                    else (
                        REPO_ROOT
                        / "datasets"
                        / "comsol"
                        / "1darcy"
                        / "metadata"
                        / "stations.csv"
                    )
                )
            ),
            "observed_file": str(
                (
                    Path(
                        manifest["data_contract"]["observed_file"]
                    )
                    if manifest is not None
                    else (
                        REPO_ROOT
                        / "datasets"
                        / "comsol"
                        / "1darcy"
                        / "processed"
                        / "strain.csv"
                    )
                )
            ),
            "physical_parameters": list(
                PHYSICAL_PARAMETERS
            ),
            "baseline": baseline,
            "bounds": bounds,
            "fixed_model_inputs": fixed,
            "seed": int(args.seed),
            "sobol_base": int(
                args.sobol_base
            ),
            "oat_points": int(
                args.oat_points
            ),
            "heatmap_points": int(
                args.heatmap_points
            ),
            "global_samples": int(
                args.global_samples
            ),
        },
    )

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    baseline_result = evaluate_model(
        baseline,
        fixed,
        observed,
        stations,
    )

    baseline_summary = {
        "parameters": baseline,
        "fixed_model_inputs": fixed,
        "global_rmse": baseline_result[
            "global_rmse"
        ],
        "component_rmse": baseline_result[
            "component_rmse"
        ],
        "station_rmse": baseline_result[
            "station_rmse"
        ],
    }

    save_json(
        output_dir
        / "baseline_sensitivity_diagnostics.json",
        baseline_summary,
    )

    print(
        "\nBaseline global strain RMSE = "
        f"{baseline_result['global_rmse']:.6g} nε"
    )

    # ------------------------------------------------------------------
    # OAT
    # ------------------------------------------------------------------

    if not args.skip_oat:

        print(
            "\n[INFO] Running one-at-a-time sensitivity analysis."
        )

        run_oat(
            baseline,
            bounds,
            fixed,
            observed,
            stations,
            output_dir,
            args.oat_points,
        )

    # ------------------------------------------------------------------
    # 2-D surfaces
    # ------------------------------------------------------------------

    if not args.skip_heatmaps:

        print(
            "\n[INFO] Running 2-D response surfaces."
        )

        # These pairs are selected because they are particularly relevant
        # to geometry/elastic tradeoffs and the identifiability problem.
        heatmap_pairs = (
            ("a", "b"),
            ("h", "theta_deg"),
            ("x0_prime", "y0_prime"),
        )

        for parameter_x, parameter_y in (
            heatmap_pairs
        ):

            print(
                f"  surface: "
                f"{parameter_x} vs {parameter_y}"
            )

            run_2d_surface(
                parameter_x,
                parameter_y,
                baseline,
                bounds,
                fixed,
                observed,
                stations,
                output_dir,
                args.heatmap_points,
            )

    # ------------------------------------------------------------------
    # Global random search
    # ------------------------------------------------------------------

    if not args.skip_global:

        print(
            "\n[INFO] Running random global parameter search."
        )

        global_dataframe = (
            run_global_search(
                baseline,
                bounds,
                fixed,
                observed,
                stations,
                output_dir,
                args.global_samples,
                args.seed,
            )
        )

        print(
            "  best global-search RMSE = "
            f"{global_dataframe['global_rmse'].min():.6g} nε"
        )

    # ------------------------------------------------------------------
    # Sobol
    # ------------------------------------------------------------------

    if not args.skip_sobol:

        print(
            "\n[INFO] Running Sobol global sensitivity analysis."
        )

        run_sobol_analysis(
            baseline,
            bounds,
            fixed,
            observed,
            stations,
            output_dir,
            args.sobol_base,
            args.seed,
        )

    # ------------------------------------------------------------------
    # Representative time-series plot
    # ------------------------------------------------------------------

    representative_time_series(
        baseline,
        bounds,
        fixed,
        observed,
        stations,
        output_dir,
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    enabled = [
        name
        for name, disabled
        in (
            ("OAT", args.skip_oat),
            ("2-D response surfaces", args.skip_heatmaps),
            ("Global random search", args.skip_global),
            ("Sobol", args.skip_sobol),
        )
        if not disabled
    ]

    print(
        "\n"
        + "=" * 78
    )

    print(
        "1-DARCY SENSITIVITY ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nAnalyses completed:"
    )

    for name in enabled:
        print(
            f"  {name}"
        )

    print(
        f"\nOutput directory:\n"
        f"    {output_dir}"
    )


if __name__ == "__main__":
    main()
