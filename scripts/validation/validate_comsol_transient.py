#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate the AVANT analytical transient strain response against a
processed COMSOL transient dataset.

Purpose
-------
This script performs a NO-FIT transient verification using:

    1. Larry's processed COMSOL strain time series;
    2. Larry's processed COMSOL pressure-monitor time series;
    3. the fixed physical/geometric benchmark configuration already
       established by the steady-state COMSOL validation workflow;
    4. the production AVANT analytical forward-model physics.

The script does NOT:

    - optimize geometry;
    - optimize material parameters;
    - perform Bayesian inversion;
    - modify the production forward model;
    - modify the observational datasets.

Instead, it answers the first scientific question:

    "Using the verified benchmark geometry/material configuration and
     the observed COMSOL pressure history as the transient forcing,
     how closely does the analytical model reproduce the COMSOL
     transient strain observations?"

Important modeling convention
-----------------------------
The production forward model internally creates a piecewise pressure
history from pmax, tpeak, and d.

For this verification workflow, the actual COMSOL pressure-monitor
history is supplied externally.

Because the implemented analytical strain response is linear with
pressure, the script evaluates the production forward model once at
a reference pressure of 1 MPa and scales that spatial response by

    P_COMSOL(t) / 1e6

at each observation time.

Thus:

    analytical_strain(t)
        =
    analytical_strain_at_1MPa
        * P_COMSOL(t) / 1e6

This avoids changing the production forward-model API while allowing
the new Larry pressure record to drive the transient verification.

Coordinate convention
---------------------
The supplied COMSOL x/y station coordinates remain in the global
input frame and are passed directly to the production forward model.
The production model performs the established horizontal coordinate
transformation and strain-tensor rotation.

The COMSOL vertical coordinate is converted before calling the forward
model:

    analytical_depth = COMSOL_SURFACE_Z - COMSOL_GLOBAL_Z

For Larry's current transient dataset:

    COMSOL surface z       = +520 m
    COMSOL strainmeter z   = +490 m
    analytical depth       = 30 m

Fixed benchmark
---------------
    E       = 8 GPa
    alpha   = 0.7
    nu      = 0.25

    a       = 215 m
    b       = 400 m
    c       = 2.5 m

    theta   = 15 deg
    x0      = 53 m
    y0      = 141 m

    h       = 522.5 m

Units
-----
COMSOL strain values are stored as nanostrain in the processed
observation CSV. The production forward model also returns nanostrain.

For metrics and machine-readable comparison tables, both are converted
to dimensionless strain by multiplying by 1e-9.

Pressure is retained in Pa.

Inputs
------
Default inputs correspond to the processed files created from Larry's
transient COMSOL export.

Outputs
-------
<output-dir>/
    transient_predictions.csv
    transient_residuals.csv
    component_metrics.csv
    station_metrics.csv
    overall_metrics.json
    verification_summary.json

    plots/
        pressure_forcing.png
        stations/
            S01.png
            ...
            S08.png
        components/
            eXX.png
            eYY.png
            eZZ.png
            eXY.png
        overview/
            all_stations.png
            all_components.png
        parity/
            parity_all_observations.png

Definitions
-----------
Residual:

    COMSOL - analytical

RMSE:

    sqrt(mean(residual^2))

MAE:

    mean(abs(residual))

Bias:

    mean(residual)

R2:

    1 - sum(residual^2) / sum((observed-mean(observed))^2)

The script reports R2 as NaN when the observed series has zero
variance.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Repository import
# ============================================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from avant_model.model.forward_model_multi_station import (
    forward_model_multi_station,
)


# ============================================================================
# Fixed Larry COMSOL verification benchmark
# ============================================================================

E = 8.0e9
ALPHA = 0.7
NU = 0.25

A = 215.0
B = 400.0
C = 2.5

THETA_DEG = 15.0

REFERENCE_X0 = 53.0
REFERENCE_Y0 = 141.0

REFERENCE_PRESSURE_PA = 1.0e6

COMSOL_SURFACE_Z_M = 520.0

COMSOL_INCLUSION_TOP_Z_M = 0.0
COMSOL_INCLUSION_CENTER_Z_M = -C
COMSOL_INCLUSION_BOTTOM_Z_M = -2.0 * C

INCLUSION_TOP_DEPTH_M = (
    COMSOL_SURFACE_Z_M
    - COMSOL_INCLUSION_TOP_Z_M
)

H = (
    COMSOL_SURFACE_Z_M
    - COMSOL_INCLUSION_CENTER_Z_M
)

INCLUSION_BOTTOM_DEPTH_M = (
    COMSOL_SURFACE_Z_M
    - COMSOL_INCLUSION_BOTTOM_Z_M
)


# ============================================================================
# Observation definitions
# ============================================================================

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

COMPONENT_COLORS = {
    "eXX": "#1f77b4",
    "eYY": "#ff7f0e",
    "eZZ": "#2ca02c",
    "eXY": "#d62728",
}

STRAIN_PATTERN = re.compile(
    r"^(eXX|eYY|eZZ|eXY)_(.+)$"
)

PRESSURE_PATTERN = re.compile(
    r"^pressure_(P\d+)_(.+)$",
    flags=re.IGNORECASE,
)

COMSOL_EXPORT_TO_DIMENSIONLESS = 1.0e-9
FORWARD_OUTPUT_TO_DIMENSIONLESS = 1.0e-9


# ============================================================================
# Default paths
# ============================================================================

DEFAULT_STRAIN_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "processed"
    / "verification_transient"
    / "strain_unique_time.csv"
)

DEFAULT_PRESSURE_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "processed"
    / "verification_transient"
    / "pressure_unique_time.csv"
)

DEFAULT_STATIONS_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "metadata"
    / "verification_transient"
    / "strain_stations.csv"
)

DEFAULT_PRESSURE_MONITORS_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "metadata"
    / "verification_transient"
    / "pressure_monitors.csv"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "validation"
    / "comsol_transient_1darcy"
)


# ============================================================================
# Plot style
# ============================================================================

def configure_plot_style() -> None:
    """
    Configure consistent publication-quality plotting.
    """

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 500,
        }
    )


# ============================================================================
# Validation helpers
# ============================================================================

def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Require a file to exist.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n"
            f"    {path}"
        )


def validate_time_vector(
    time: np.ndarray,
    description: str,
) -> None:
    """
    Validate a time vector.

    Duplicate timestamps are allowed here because this script expects
    analysis-ready unique-time input by default but should fail only if
    a user accidentally supplies a malformed time vector.
    """

    if time.ndim != 1:
        raise ValueError(
            f"{description} must be one-dimensional."
        )

    if time.size == 0:
        raise ValueError(
            f"{description} is empty."
        )

    if not np.isfinite(
        time
    ).all():

        raise ValueError(
            f"{description} contains NaN or infinite values."
        )

    if np.any(
        np.diff(time) < 0.0
    ):
        raise ValueError(
            f"{description} must be monotonically non-decreasing."
        )


def validate_unique_time(
    time: np.ndarray,
    description: str,
) -> None:
    """
    Require strictly unique timestamps for the transient verification.
    """

    if np.unique(
        time
    ).size != time.size:

        duplicate_count = (
            time.size
            - np.unique(time).size
        )

        raise ValueError(
            f"{description} contains "
            f"{duplicate_count} duplicate timestamp(s).\n"
            "Use the 107-row unique-time processed datasets for "
            "transient model verification."
        )


def validate_numeric_frame(
    dataframe: pd.DataFrame,
    columns: list[str],
    description: str,
) -> None:
    """
    Validate that selected columns are finite and numeric.
    """

    values = (
        dataframe[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            f"{description} contains NaN or infinite values."
        )


# ============================================================================
# Data loading
# ============================================================================

def load_strain_dataset(
    path: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[str],
    list[str],
]:
    """
    Load and validate the processed strain dataset.
    """

    require_file(
        path,
        "Processed strain dataset",
    )

    dataframe = pd.read_csv(
        path
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if "time_s" not in dataframe.columns:
        raise ValueError(
            "Strain dataset must contain 'time_s'."
        )

    time = (
        dataframe[
            "time_s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    validate_time_vector(
        time,
        "Strain time",
    )

    validate_unique_time(
        time,
        "Strain time",
    )

    station_names: list[str] = []
    available_components: set[str] = set()

    for column in dataframe.columns:

        if column == "time_s":
            continue

        match = STRAIN_PATTERN.match(
            column
        )

        if match is None:
            raise ValueError(
                "Unexpected strain column: "
                f"{column!r}"
            )

        component = match.group(1)
        station = match.group(2).strip()

        available_components.add(
            component
        )

        if station not in station_names:
            station_names.append(
                station
            )

    components = [
        component
        for component in COMPONENTS
        if component in available_components
    ]

    if not station_names:
        raise ValueError(
            "No strain stations were discovered."
        )

    if not components:
        raise ValueError(
            "No supported strain components were discovered."
        )

    for station in station_names:

        for component in components:

            column = (
                f"{component}_{station}"
            )

            if column not in dataframe.columns:
                raise ValueError(
                    "Missing strain channel: "
                    f"{column}"
                )

    strain_columns = [
        f"{component}_{station}"
        for component in components
        for station in station_names
    ]

    validate_numeric_frame(
        dataframe,
        strain_columns,
        "Strain dataset",
    )

    return (
        dataframe,
        time,
        station_names,
        components,
    )


def load_pressure_dataset(
    path: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[dict[str, str]],
]:
    """
    Load and validate the processed pressure dataset.
    """

    require_file(
        path,
        "Processed pressure dataset",
    )

    dataframe = pd.read_csv(
        path
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if "time_s" not in dataframe.columns:
        raise ValueError(
            "Pressure dataset must contain 'time_s'."
        )

    time = (
        dataframe[
            "time_s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    validate_time_vector(
        time,
        "Pressure time",
    )

    validate_unique_time(
        time,
        "Pressure time",
    )

    channels: list[
        dict[str, str]
    ] = []

    for column in dataframe.columns:

        if column == "time_s":
            continue

        match = PRESSURE_PATTERN.match(
            column
        )

        if match is None:
            raise ValueError(
                "Unexpected pressure column: "
                f"{column!r}"
            )

        channels.append(
            {
                "column": column,
                "monitor": (
                    match.group(1).upper()
                ),
                "units": (
                    match.group(2)
                ),
            }
        )

    if not channels:
        raise ValueError(
            "No pressure-monitor channels were found."
        )

    pressure_columns = [
        channel["column"]
        for channel in channels
    ]

    validate_numeric_frame(
        dataframe,
        pressure_columns,
        "Pressure dataset",
    )

    return (
        dataframe,
        time,
        channels,
    )


def load_station_metadata(
    path: Path,
    station_names: list[str],
) -> pd.DataFrame:
    """
    Load and validate strain-station metadata.
    """

    require_file(
        path,
        "Strain-station metadata",
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
        - set(stations.columns)
    )

    if missing:
        raise ValueError(
            "Station metadata are missing columns: "
            + ", ".join(
                sorted(missing)
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

    for coordinate in (
        "x",
        "y",
        "z",
    ):

        stations[
            coordinate
        ] = pd.to_numeric(
            stations[
                coordinate
            ],
            errors="raise",
        )

    actual_names = (
        stations[
            "station"
        ].tolist()
    )

    if actual_names != station_names:

        raise ValueError(
            "Station metadata order does not match "
            "the strain dataset.\n"
            f"Data:     {station_names}\n"
            f"Metadata: {actual_names}"
        )

    coordinates = (
        stations[
            [
                "x",
                "y",
                "z",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        coordinates
    ).all():

        raise ValueError(
            "Station metadata contain non-finite coordinates."
        )

    return stations


def load_pressure_monitor_metadata(
    path: Path,
    channels: list[dict[str, str]],
) -> pd.DataFrame:
    """
    Load and validate pressure-monitor metadata.
    """

    require_file(
        path,
        "Pressure-monitor metadata",
    )

    monitors = pd.read_csv(
        path
    )

    required = {
        "monitor",
        "x",
        "y",
        "z",
        "units",
    }

    missing = (
        required
        - set(monitors.columns)
    )

    if missing:
        raise ValueError(
            "Pressure-monitor metadata are missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    monitors = monitors[
        [
            "monitor",
            "x",
            "y",
            "z",
            "units",
        ]
    ].copy()

    monitors[
        "monitor"
    ] = (
        monitors[
            "monitor"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    monitors[
        "units"
    ] = (
        monitors[
            "units"
        ]
        .astype(str)
        .str.strip()
    )

    for coordinate in (
        "x",
        "y",
        "z",
    ):

        monitors[
            coordinate
        ] = pd.to_numeric(
            monitors[
                coordinate
            ],
            errors="raise",
        )

    expected_monitors = [
        channel[
            "monitor"
        ]
        for channel in channels
    ]

    actual_monitors = (
        monitors[
            "monitor"
        ].tolist()
    )

    if actual_monitors != expected_monitors:

        raise ValueError(
            "Pressure-monitor metadata order does not match "
            "the pressure dataset.\n"
            f"Data:     {expected_monitors}\n"
            f"Metadata: {actual_monitors}"
        )

    return monitors


# ============================================================================
# Time alignment
# ============================================================================

def validate_time_alignment(
    strain_time: np.ndarray,
    pressure_time: np.ndarray,
) -> None:
    """
    Require exact strain/pressure time alignment.
    """

    if not np.array_equal(
        strain_time,
        pressure_time,
    ):

        raise ValueError(
            "Strain and pressure time vectors do not match exactly."
        )


# ============================================================================
# Coordinate handling
# ============================================================================

def convert_comsol_z_to_analytical_depth(
    comsol_z: np.ndarray,
) -> np.ndarray:
    """
    Convert COMSOL global z coordinates to analytical positive-downward
    depths.
    """

    return (
        COMSOL_SURFACE_Z_M
        - np.asarray(
            comsol_z,
            dtype=float,
        )
    )


def validate_vertical_mapping(
    stations: pd.DataFrame,
) -> np.ndarray:
    """
    Validate the established COMSOL vertical convention.
    """

    comsol_z = (
        stations[
            "z"
        ]
        .to_numpy(
            dtype=float
        )
    )

    analytical_depth = (
        convert_comsol_z_to_analytical_depth(
            comsol_z
        )
    )

    if not np.allclose(
        analytical_depth,
        analytical_depth[0],
    ):

        raise ValueError(
            "Current transient strain stations do not lie on a "
            "single horizontal observation plane."
        )

    if not np.isclose(
        analytical_depth[0],
        30.0,
    ):

        raise ValueError(
            "Expected COMSOL z=490 m to map to "
            "30 m analytical depth."
        )

    return analytical_depth


# ============================================================================
# Analytical 1-MPa reference response
# ============================================================================

def run_reference_forward_model(
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate the production forward model at 1 MPa and t=1 s.

    The pressure history is deliberately chosen as:

        pmax  = 1e6 Pa
        tpeak = 1 s
        d     = 0
        time  = [1 s]

    which gives an exact 1-MPa pressure value at the single evaluation
    time and therefore provides the spatial response per 1 MPa.
    """

    x = (
        stations[
            "x"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        stations[
            "y"
        ]
        .to_numpy(
            dtype=float
        )
    )

    comsol_z = (
        stations[
            "z"
        ]
        .to_numpy(
            dtype=float
        )
    )

    z = (
        convert_comsol_z_to_analytical_depth(
            comsol_z
        )
    )

    names = (
        stations[
            "station"
        ]
        .astype(str)
        .to_numpy()
    )

    prediction = (
        forward_model_multi_station(
            pmax=REFERENCE_PRESSURE_PA,
            tpeak=1.0,
            d=0.0,
            time=np.array(
                [1.0],
                dtype=float,
            ),
            x_prime=x,
            y_prime=y,
            x0_prime=REFERENCE_X0,
            y0_prime=REFERENCE_Y0,
            z=z,
            a=A,
            b=B,
            c=C,
            nu=NU,
            h=H,
            E=E,
            theta_deg=THETA_DEG,
            alpha=ALPHA,
            station_names=names,
        )
    )

    return prediction


# ============================================================================
# Pressure-driven transient scaling
# ============================================================================

def build_transient_prediction(
    reference_prediction: pd.DataFrame,
    pressure: pd.DataFrame,
    station_names: list[str],
) -> pd.DataFrame:
    """
    Scale the 1-MPa analytical spatial response using the observed
    COMSOL pressure history.

    The reference forward model returns nanostrain.
    The pressure ratio is dimensionless.
    """

    time = (
        pressure[
            "time_s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pressure_columns = [
        column
        for column in pressure.columns
        if column != "time_s"
    ]

    if len(
        pressure_columns
    ) != 1:

        raise ValueError(
            "Transient verification currently requires exactly "
            "one pressure-monitor channel."
        )

    pressure_values = (
        pressure[
            pressure_columns[0]
        ]
        .to_numpy(
            dtype=float
        )
    )

    pressure_ratio = (
        pressure_values
        / REFERENCE_PRESSURE_PA
    )

    output: dict[
        str,
        np.ndarray,
    ] = {
        "time_s": time
    }

    for component in COMPONENTS:

        for station in station_names:

            column = (
                f"{component}_{station}"
            )

            reference_value = float(
                reference_prediction[
                    column
                ].iloc[0]
            )

            output[
                column
            ] = (
                reference_value
                * pressure_ratio
            )

    return pd.DataFrame(
        output
    )


# ============================================================================
# Unit conversion
# ============================================================================

def nanostrain_to_dimensionless(
    dataframe: pd.DataFrame,
    station_names: list[str],
) -> pd.DataFrame:
    """
    Convert strain columns from nanostrain to dimensionless strain.
    """

    output = dataframe.copy()

    columns = [
        f"{component}_{station}"
        for component in COMPONENTS
        for station in station_names
    ]

    output[
        columns
    ] = (
        output[
            columns
        ]
        * COMSOL_EXPORT_TO_DIMENSIONLESS
    )

    return output


# ============================================================================
# Comparison tables
# ============================================================================

def build_comparison_table(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    station_names: list[str],
) -> pd.DataFrame:
    """
    Build a long-form time/component/station comparison table.
    """

    rows: list[
        dict[str, float | str]
    ] = []

    time = (
        observed[
            "time_s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    for component in COMPONENTS:

        for station in station_names:

            column = (
                f"{component}_{station}"
            )

            observed_values = (
                observed[
                    column
                ]
                .to_numpy(
                    dtype=float
                )
                * COMSOL_EXPORT_TO_DIMENSIONLESS
            )

            predicted_values = (
                predicted[
                    column
                ]
                .to_numpy(
                    dtype=float
                )
                * FORWARD_OUTPUT_TO_DIMENSIONLESS
            )

            residual = (
                observed_values
                - predicted_values
            )

            for index in range(
                time.size
            ):

                rows.append(
                    {
                        "time_s": time[index],
                        "station": station,
                        "component": component,
                        "observed_strain": (
                            observed_values[
                                index
                            ]
                        ),
                        "analytical_strain": (
                            predicted_values[
                                index
                            ]
                        ),
                        "residual_comsol_minus_analytical": (
                            residual[
                                index
                            ]
                        ),
                        "absolute_residual": (
                            abs(
                                residual[
                                    index
                                ]
                            )
                        ),
                    }
                )

    return pd.DataFrame.from_records(
        rows
    )


# ============================================================================
# Metrics
# ============================================================================

def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """
    Calculate standard comparison metrics in dimensionless strain units.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    residual = (
        observed
        - predicted
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual**2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(residual)
        )
    )

    bias = float(
        np.mean(
            residual
        )
    )

    max_abs = float(
        np.max(
            np.abs(residual)
        )
    )

    ss_res = float(
        np.sum(
            residual**2
        )
    )

    centered_observed = (
        observed
        - np.mean(observed)
    )

    ss_tot = float(
        np.sum(
            centered_observed**2
        )
    )

    if ss_tot > 0.0:

        r2 = float(
            1.0
            - ss_res
            / ss_tot
        )

    else:

        r2 = float(
            "nan"
        )

    return {
        "rmse_strain": rmse,
        "mae_strain": mae,
        "bias_observed_minus_analytical_strain": bias,
        "max_abs_residual_strain": max_abs,
        "r2": r2,
    }


def build_component_metrics(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    station_names: list[str],
) -> pd.DataFrame:
    """
    Build metrics aggregated across stations for each component.
    """

    rows: list[
        dict[str, float | str]
    ] = []

    for component in COMPONENTS:

        observed_values: list[
            float
        ] = []

        predicted_values: list[
            float
        ] = []

        for station in station_names:

            column = (
                f"{component}_{station}"
            )

            observed_values.extend(
                (
                    observed[
                        column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    * COMSOL_EXPORT_TO_DIMENSIONLESS
                ).tolist()
            )

            predicted_values.extend(
                (
                    predicted[
                        column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    * FORWARD_OUTPUT_TO_DIMENSIONLESS
                ).tolist()
            )

        metrics = calculate_metrics(
            np.asarray(
                observed_values
            ),
            np.asarray(
                predicted_values
            ),
        )

        rows.append(
            {
                "component": component,
                **metrics,
            }
        )

    return pd.DataFrame.from_records(
        rows
    )


def build_station_metrics(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    station_names: list[str],
) -> pd.DataFrame:
    """
    Build metrics aggregated across components for each station.
    """

    rows: list[
        dict[str, float | str]
    ] = []

    for station in station_names:

        observed_values: list[
            float
        ] = []

        predicted_values: list[
            float
        ] = []

        for component in COMPONENTS:

            column = (
                f"{component}_{station}"
            )

            observed_values.extend(
                (
                    observed[
                        column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    * COMSOL_EXPORT_TO_DIMENSIONLESS
                ).tolist()
            )

            predicted_values.extend(
                (
                    predicted[
                        column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    * FORWARD_OUTPUT_TO_DIMENSIONLESS
                ).tolist()
            )

        metrics = calculate_metrics(
            np.asarray(
                observed_values
            ),
            np.asarray(
                predicted_values
            ),
        )

        rows.append(
            {
                "station": station,
                **metrics,
            }
        )

    return pd.DataFrame.from_records(
        rows
    )


def build_overall_metrics(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    station_names: list[str],
) -> dict[str, float]:
    """
    Build metrics across all stations, components, and times.
    """

    observed_values: list[
        float
    ] = []

    predicted_values: list[
        float
    ] = []

    for component in COMPONENTS:

        for station in station_names:

            column = (
                f"{component}_{station}"
            )

            observed_values.extend(
                (
                    observed[
                        column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    * COMSOL_EXPORT_TO_DIMENSIONLESS
                ).tolist()
            )

            predicted_values.extend(
                (
                    predicted[
                        column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    * FORWARD_OUTPUT_TO_DIMENSIONLESS
                ).tolist()
            )

    return calculate_metrics(
        np.asarray(
            observed_values
        ),
        np.asarray(
            predicted_values
        ),
    )


# ============================================================================
# Figure saving
# ============================================================================

def save_figure(
    fig: plt.Figure,
    path: Path,
    show: bool,
) -> None:
    """
    Save a figure as PNG and PDF.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=500,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
        facecolor="white",
    )

    if show:
        plt.show()

    plt.close(
        fig
    )

    print(
        f"[INFO] Saved {path}"
    )


# ============================================================================
# Pressure plot
# ============================================================================

def plot_pressure_forcing(
    pressure: pd.DataFrame,
    channels: list[dict[str, str]],
    output_dir: Path,
    show: bool,
) -> None:
    """
    Plot the COMSOL pressure-monitor forcing history.
    """

    time = (
        pressure[
            "time_s"
        ].to_numpy(
            dtype=float
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            12.0,
            6.8,
        )
    )

    for channel in channels:

        values = (
            pressure[
                channel[
                    "column"
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        ax.plot(
            time,
            values,
            linewidth=2.3,
            label=channel[
                "monitor"
            ],
        )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Pressure (Pa)"
    )

    ax.set_title(
        "Larry COMSOL Pressure-Monitor History",
        pad=12,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    if len(channels) > 1:
        ax.legend(
            frameon=True
        )

    fig.tight_layout(
        pad=1.6
    )

    save_figure(
        fig,
        output_dir
        / "pressure_forcing.png",
        show,
    )


# ============================================================================
# Station comparison plots
# ============================================================================

def plot_station_comparison(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    time: np.ndarray,
    station: str,
    output_dir: Path,
    show: bool,
) -> None:
    """
    Plot observed and analytical transient strain for one station.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            13.5,
            9.0,
        ),
        sharex=True,
    )

    axes = axes.ravel()

    for index, component in enumerate(
        COMPONENTS
    ):

        ax = axes[
            index
        ]

        column = (
            f"{component}_{station}"
        )

        observed_values = (
            observed[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        predicted_values = (
            predicted[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        ax.plot(
            time,
            observed_values,
            color="black",
            linewidth=2.1,
            label="COMSOL",
        )

        ax.plot(
            time,
            predicted_values,
            color=COMPONENT_COLORS[
                component
            ],
            linewidth=2.0,
            linestyle="--",
            label="Analytical",
        )

        ax.axhline(
            0.0,
            color="black",
            linestyle=":",
            linewidth=0.9,
            alpha=0.5,
        )

        ax.set_title(
            COMPONENT_LABELS[
                component
            ],
            pad=7,
        )

        ax.set_ylabel(
            r"Strain (n$\varepsilon$)"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.22,
        )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[3].set_xlabel(
        "Time (s)"
    )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(
            0.5,
            0.015,
        ),
    )

    fig.suptitle(
        (
            "COMSOL vs Analytical Transient Strain — "
            f"{station}"
        ),
        y=0.985,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.07,
            1.0,
            0.95,
        ),
        pad=1.4,
    )

    save_figure(
        fig,
        (
            output_dir
            / "stations"
            / f"{station}.png"
        ),
        show,
    )


# ============================================================================
# Component comparison plots
# ============================================================================

def plot_component_comparison(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    time: np.ndarray,
    component: str,
    station_names: list[str],
    output_dir: Path,
    show: bool,
) -> None:
    """
    Plot one strain component across all stations.
    """

    fig, ax = plt.subplots(
        figsize=(
            12.5,
            7.0,
        )
    )

    for station in station_names:

        column = (
            f"{component}_{station}"
        )

        ax.plot(
            time,
            observed[
                column
            ].to_numpy(
                dtype=float
            ),
            linewidth=1.4,
            alpha=0.55,
            color=COMPONENT_COLORS[
                component
            ],
        )

        ax.plot(
            time,
            predicted[
                column
            ].to_numpy(
                dtype=float
            ),
            linewidth=1.6,
            linestyle="--",
            color="black",
            alpha=0.65,
        )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        r"Strain (n$\varepsilon$)"
    )

    ax.set_title(
        (
            f"{COMPONENT_LABELS[component]} — "
            "COMSOL and Analytical"
        ),
        pad=12,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    fig.tight_layout(
        pad=1.6
    )

    save_figure(
        fig,
        (
            output_dir
            / "components"
            / f"{component}.png"
        ),
        show,
    )


# ============================================================================
# Overview plots
# ============================================================================

def plot_all_stations_overview(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    time: np.ndarray,
    station_names: list[str],
    output_dir: Path,
    show: bool,
) -> None:
    """
    Create a multi-station overview using eXX observations and predictions.
    """

    n_stations = len(
        station_names
    )

    n_columns = 2

    n_rows = int(
        np.ceil(
            n_stations
            / n_columns
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            14.0,
            3.5 * n_rows,
        ),
        sharex=True,
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    for index, station in enumerate(
        station_names
    ):

        ax = axes[
            index
        ]

        column = (
            f"eXX_{station}"
        )

        ax.plot(
            time,
            observed[
                column
            ].to_numpy(
                dtype=float
            ),
            color=COMPONENT_COLORS[
                "eXX"
            ],
            linewidth=1.8,
            label="COMSOL",
        )

        ax.plot(
            time,
            predicted[
                column
            ].to_numpy(
                dtype=float
            ),
            color="black",
            linewidth=1.6,
            linestyle="--",
            label="Analytical",
        )

        ax.axhline(
            0.0,
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )

        ax.set_title(
            station,
            pad=7,
        )

        ax.set_ylabel(
            r"$\varepsilon_{xx}$ (n$\varepsilon$)"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.20,
        )

    for ax in axes:

        if ax.get_visible():
            ax.set_xlabel(
                "Time (s)"
            )

    for index in range(
        n_stations,
        len(axes),
    ):
        axes[
            index
        ].set_visible(
            False
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
    )

    fig.suptitle(
        "eXX Transient Verification — All Stations",
        y=0.992,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.06,
            1.0,
            0.965,
        ),
        pad=1.2,
    )

    save_figure(
        fig,
        (
            output_dir
            / "overview"
            / "all_stations.png"
        ),
        show,
    )


def plot_all_components_overview(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    time: np.ndarray,
    station_names: list[str],
    output_dir: Path,
    show: bool,
) -> None:
    """
    Create a compact overview with all four strain components aggregated
    across the station set.

    For each component, the station-average strain is plotted.
    """

    fig, ax = plt.subplots(
        figsize=(
            12.5,
            7.0,
        )
    )

    for component in COMPONENTS:

        observed_matrix = np.column_stack(
            [
                observed[
                    f"{component}_{station}"
                ].to_numpy(
                    dtype=float
                )
                for station in station_names
            ]
        )

        predicted_matrix = np.column_stack(
            [
                predicted[
                    f"{component}_{station}"
                ].to_numpy(
                    dtype=float
                )
                for station in station_names
            ]
        )

        observed_mean = (
            np.mean(
                observed_matrix,
                axis=1,
            )
        )

        predicted_mean = (
            np.mean(
                predicted_matrix,
                axis=1,
            )
        )

        ax.plot(
            time,
            observed_mean,
            color=COMPONENT_COLORS[
                component
            ],
            linewidth=2.0,
            label=(
                f"COMSOL "
                f"{COMPONENT_LABELS[component]}"
            ),
        )

        ax.plot(
            time,
            predicted_mean,
            color=COMPONENT_COLORS[
                component
            ],
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
            label=(
                f"Analytical "
                f"{COMPONENT_LABELS[component]}"
            ),
        )

    ax.axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=0.9,
        alpha=0.5,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        r"Station-mean strain (n$\varepsilon$)"
    )

    ax.set_title(
        "Transient Strain Verification — Station Means",
        pad=12,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(
        loc="best",
        frameon=True,
        ncol=2,
    )

    fig.tight_layout(
        pad=1.6
    )

    save_figure(
        fig,
        (
            output_dir
            / "overview"
            / "all_components.png"
        ),
        show,
    )


# ============================================================================
# Parity plot
# ============================================================================

def plot_parity(
    comparison: pd.DataFrame,
    output_dir: Path,
    show: bool,
) -> None:
    """
    Plot observed versus analytical strain parity.
    """

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            8.0,
        )
    )

    for component in COMPONENTS:

        subset = comparison.loc[
            comparison[
                "component"
            ] == component
        ]

        ax.scatter(
            subset[
                "observed_strain"
            ],
            subset[
                "analytical_strain"
            ],
            s=18,
            alpha=0.45,
            color=COMPONENT_COLORS[
                component
            ],
            label=COMPONENT_LABELS[
                component
            ],
        )

    all_observed = (
        comparison[
            "observed_strain"
        ]
        .to_numpy(
            dtype=float
        )
    )

    all_predicted = (
        comparison[
            "analytical_strain"
        ]
        .to_numpy(
            dtype=float
        )
    )

    limit = float(
        max(
            np.max(
                np.abs(
                    all_observed
                )
            ),
            np.max(
                np.abs(
                    all_predicted
                )
            ),
            np.finfo(float).eps,
        )
    )

    limit *= 1.08

    ax.plot(
        [
            -limit,
            limit,
        ],
        [
            -limit,
            limit,
        ],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="1:1",
    )

    ax.set_xlim(
        -limit,
        limit,
    )

    ax.set_ylim(
        -limit,
        limit,
    )

    ax.set_xlabel(
        "COMSOL strain (dimensionless)"
    )

    ax.set_ylabel(
        "Analytical strain (dimensionless)"
    )

    ax.set_title(
        "COMSOL vs Analytical Transient Strain",
        pad=12,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.20,
    )

    ax.legend(
        frameon=True
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout(
        pad=1.5
    )

    save_figure(
        fig,
        (
            output_dir
            / "parity"
            / "parity_all_observations.png"
        ),
        show,
    )


# ============================================================================
# Report
# ============================================================================

def build_verification_summary(
    strain_file: Path,
    pressure_file: Path,
    stations_file: Path,
    pressure_monitors_file: Path,
    strain: pd.DataFrame,
    pressure: pd.DataFrame,
    station_names: list[str],
    components: list[str],
    pressure_channels: list[dict[str, str]],
    station_metrics: pd.DataFrame,
    component_metrics: pd.DataFrame,
    overall_metrics: dict[str, float],
) -> dict:
    """
    Build a machine-readable verification summary.
    """

    pressure_column = (
        pressure_channels[0][
            "column"
        ]
    )

    pressure_values = (
        pressure[
            pressure_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    pressure_time = (
        pressure[
            "time_s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pressure_peak_index = int(
        np.argmax(
            np.abs(
                pressure_values
            )
        )
    )

    station_metadata = []

    for station in station_names:

        station_metadata.append(
            {
                "station": station,
            }
        )

    return {
        "workflow": {
            "name": (
                "AVANT transient COMSOL "
                "pressure-driven verification"
            ),
            "fit_performed": False,
            "inversion_performed": False,
            "production_forward_model_modified": False,
        },
        "inputs": {
            "strain_file": str(
                strain_file
            ),
            "pressure_file": str(
                pressure_file
            ),
            "stations_file": str(
                stations_file
            ),
            "pressure_monitors_file": str(
                pressure_monitors_file
            ),
        },
        "dataset": {
            "n_time_samples": int(
                len(strain)
            ),
            "n_strain_stations": int(
                len(station_names)
            ),
            "n_components": int(
                len(components)
            ),
            "strain_channels": int(
                len(station_names)
                * len(components)
            ),
            "pressure_channels": int(
                len(pressure_channels)
            ),
            "time_start_s": float(
                strain[
                    "time_s"
                ].min()
            ),
            "time_end_s": float(
                strain[
                    "time_s"
                ].max()
            ),
        },
        "benchmark": {
            "E_Pa": E,
            "alpha": ALPHA,
            "nu": NU,
            "a_m": A,
            "b_m": B,
            "c_m": C,
            "h_m": H,
            "theta_deg": THETA_DEG,
            "x0_prime_m": REFERENCE_X0,
            "y0_prime_m": REFERENCE_Y0,
        },
        "vertical_coordinate_mapping": {
            "comsol_surface_z_m": (
                COMSOL_SURFACE_Z_M
            ),
            "analytical_top_depth_m": (
                INCLUSION_TOP_DEPTH_M
            ),
            "analytical_center_depth_m": H,
            "analytical_bottom_depth_m": (
                INCLUSION_BOTTOM_DEPTH_M
            ),
            "strainmeter_depth_m": 30.0,
            "formula": (
                "analytical_depth = "
                "COMSOL_surface_z - COMSOL_global_z"
            ),
        },
        "pressure_forcing": {
            "monitor": pressure_channels[0][
                "monitor"
            ],
            "units": pressure_channels[0][
                "units"
            ],
            "peak_absolute_pressure_Pa": float(
                np.max(
                    np.abs(
                        pressure_values
                    )
                )
            ),
            "peak_absolute_pressure_time_s": float(
                pressure_time[
                    pressure_peak_index
                ]
            ),
            "minimum_pressure_Pa": float(
                np.min(
                    pressure_values
                )
            ),
            "maximum_pressure_Pa": float(
                np.max(
                    pressure_values
                )
            ),
        },
        "station_metrics": (
            station_metrics
            .to_dict(
                orient="records"
            )
        ),
        "component_metrics": (
            component_metrics
            .to_dict(
                orient="records"
            )
        ),
        "overall_metrics": overall_metrics,
        "station_names": station_names,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Validate the production AVANT analytical transient "
            "strain response against Larry's COMSOL transient "
            "pressure-driven dataset."
        )
    )

    parser.add_argument(
        "--strain-file",
        type=Path,
        default=DEFAULT_STRAIN_FILE,
        help=(
            "Unique-time processed COMSOL strain CSV."
        ),
    )

    parser.add_argument(
        "--pressure-file",
        type=Path,
        default=DEFAULT_PRESSURE_FILE,
        help=(
            "Unique-time processed COMSOL pressure CSV."
        ),
    )

    parser.add_argument(
        "--stations-file",
        type=Path,
        default=DEFAULT_STATIONS_FILE,
        help=(
            "COMSOL strain-station metadata CSV."
        ),
    )

    parser.add_argument(
        "--pressure-monitors-file",
        type=Path,
        default=DEFAULT_PRESSURE_MONITORS_FILE,
        help=(
            "COMSOL pressure-monitor metadata CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for verification outputs."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display figures interactively."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """
    Execute transient verification.
    """

    args = parse_args()

    configure_plot_style()

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 78
    )

    print(
        "AVANT / COMSOL TRANSIENT VERIFICATION"
    )

    print(
        "=" * 78
    )

    print(
        "\nFixed benchmark parameters"
    )

    print(
        f"  E                 = {E:.6g} Pa"
    )

    print(
        f"  alpha             = {ALPHA:g}"
    )

    print(
        f"  nu                = {NU:g}"
    )

    print(
        f"  a                 = {A:g} m"
    )

    print(
        f"  b                 = {B:g} m"
    )

    print(
        f"  c                 = {C:g} m"
    )

    print(
        f"  h                 = {H:g} m"
    )

    print(
        f"  theta             = {THETA_DEG:g} deg"
    )

    print(
        f"  x0_prime          = {REFERENCE_X0:g} m"
    )

    print(
        f"  y0_prime          = {REFERENCE_Y0:g} m"
    )

    print(
        "\nVertical-coordinate convention"
    )

    print(
        f"  COMSOL surface z  = "
        f"{COMSOL_SURFACE_Z_M:g} m"
    )

    print(
        "  formula            = "
        "analytical_depth = 520 - COMSOL_z"
    )

    print(
        f"  strainmeter depth = "
        f"{30.0:g} m"
    )

    # ------------------------------------------------------------------------
    # Load observations
    # ------------------------------------------------------------------------

    print(
        "\nLoading strain dataset..."
    )

    (
        strain,
        strain_time,
        station_names,
        components,
    ) = load_strain_dataset(
        args.strain_file
    )

    print(
        f"  rows       = {len(strain)}"
    )

    print(
        f"  stations   = {station_names}"
    )

    print(
        f"  components = {components}"
    )

    print(
        f"  channels   = "
        f"{len(station_names) * len(components)}"
    )

    print(
        f"  time range = "
        f"{strain_time.min():g} to "
        f"{strain_time.max():g} s"
    )

    print(
        "\nLoading pressure dataset..."
    )

    (
        pressure,
        pressure_time,
        pressure_channels,
    ) = load_pressure_dataset(
        args.pressure_file
    )

    print(
        f"  rows       = {len(pressure)}"
    )

    print(
        f"  channels   = {pressure_channels}"
    )

    print(
        f"  time range = "
        f"{pressure_time.min():g} to "
        f"{pressure_time.max():g} s"
    )

    validate_time_alignment(
        strain_time,
        pressure_time,
    )

    stations = load_station_metadata(
        args.stations_file,
        station_names,
    )

    pressure_monitors = (
        load_pressure_monitor_metadata(
            args.pressure_monitors_file,
            pressure_channels,
        )
    )

    analytical_depths = (
        validate_vertical_mapping(
            stations
        )
    )

    print(
        "\nValidated station coordinates"
    )

    for index, station in enumerate(
        station_names
    ):

        row = stations.iloc[
            index
        ]

        print(
            f"  {station}: "
            f"COMSOL=({row['x']:g}, "
            f"{row['y']:g}, "
            f"{row['z']:g}) m -> "
            f"analytical depth="
            f"{analytical_depths[index]:g} m"
        )

    # ------------------------------------------------------------------------
    # Validate pressure monitor
    # ------------------------------------------------------------------------

    if len(
        pressure_channels
    ) != 1:

        raise ValueError(
            "The current transient verification workflow expects "
            "exactly one pressure monitor."
        )

    monitor = pressure_channels[0][
        "monitor"
    ]

    monitor_row = pressure_monitors.iloc[
        0
    ]

    print(
        "\nPressure monitor"
    )

    print(
        f"  monitor = {monitor}"
    )

    print(
        f"  x       = {monitor_row['x']:g} m"
    )

    print(
        f"  y       = {monitor_row['y']:g} m"
    )

    print(
        f"  z       = {monitor_row['z']:g} m"
    )

    print(
        f"  units   = {monitor_row['units']}"
    )

    # ------------------------------------------------------------------------
    # Plot pressure forcing
    # ------------------------------------------------------------------------

    plot_pressure_forcing(
        pressure,
        pressure_channels,
        output_dir / "plots",
        args.show,
    )

    # ------------------------------------------------------------------------
    # Run reference analytical forward model
    # ------------------------------------------------------------------------

    print(
        "\nEvaluating production analytical model "
        "at 1 MPa reference pressure..."
    )

    reference_prediction = (
        run_reference_forward_model(
            stations
        )
    )

    print(
        "[INFO] Reference forward-model evaluation complete."
    )

    # ------------------------------------------------------------------------
    # Build pressure-driven transient prediction
    # ------------------------------------------------------------------------

    print(
        "\nBuilding pressure-driven transient prediction..."
    )

    predicted = (
        build_transient_prediction(
            reference_prediction,
            pressure,
            station_names,
        )
    )

    if not np.array_equal(
        predicted[
            "time_s"
        ].to_numpy(
            dtype=float
        ),
        strain_time,
    ):

        raise RuntimeError(
            "Predicted and observed time vectors do not match."
        )

    print(
        "[INFO] Transient prediction generated."
    )

    # ------------------------------------------------------------------------
    # Build comparison tables
    # ------------------------------------------------------------------------

    comparison = (
        build_comparison_table(
            observed=strain,
            predicted=predicted,
            station_names=station_names,
        )
    )

    comparison.to_csv(
        output_dir
        / "transient_residuals.csv",
        index=False,
    )

    predicted.to_csv(
        output_dir
        / "transient_predictions.csv",
        index=False,
    )

    print(
        f"[INFO] Saved "
        f"{output_dir / 'transient_predictions.csv'}"
    )

    print(
        f"[INFO] Saved "
        f"{output_dir / 'transient_residuals.csv'}"
    )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    component_metrics = (
        build_component_metrics(
            observed=strain,
            predicted=predicted,
            station_names=station_names,
        )
    )

    station_metrics = (
        build_station_metrics(
            observed=strain,
            predicted=predicted,
            station_names=station_names,
        )
    )

    overall_metrics = (
        build_overall_metrics(
            observed=strain,
            predicted=predicted,
            station_names=station_names,
        )
    )

    component_metrics.to_csv(
        output_dir
        / "component_metrics.csv",
        index=False,
    )

    station_metrics.to_csv(
        output_dir
        / "station_metrics.csv",
        index=False,
    )

    with open(
        output_dir
        / "overall_metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            overall_metrics,
            handle,
            indent=2,
        )

    print(
        f"[INFO] Saved "
        f"{output_dir / 'component_metrics.csv'}"
    )

    print(
        f"[INFO] Saved "
        f"{output_dir / 'station_metrics.csv'}"
    )

    print(
        f"[INFO] Saved "
        f"{output_dir / 'overall_metrics.json'}"
    )

    # ------------------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------------------

    print(
        "\nComponent metrics"
    )

    for _, row in (
        component_metrics.iterrows()
    ):

        r2 = row[
            "r2"
        ]

        print(
            f"  {row['component']}: "
            f"RMSE={row['rmse_strain']:.4e}, "
            f"MAE={row['mae_strain']:.4e}, "
            f"bias={row['bias_observed_minus_analytical_strain']:.4e}, "
            f"R2={r2:.6f}"
        )

    print(
        "\nStation metrics"
    )

    for _, row in (
        station_metrics.iterrows()
    ):

        r2 = row[
            "r2"
        ]

        print(
            f"  {row['station']}: "
            f"RMSE={row['rmse_strain']:.4e}, "
            f"MAE={row['mae_strain']:.4e}, "
            f"bias={row['bias_observed_minus_analytical_strain']:.4e}, "
            f"R2={r2:.6f}"
        )

    print(
        "\nOverall transient comparison"
    )

    print(
        f"  RMSE = "
        f"{overall_metrics['rmse_strain']:.6e} strain"
    )

    print(
        f"  MAE  = "
        f"{overall_metrics['mae_strain']:.6e} strain"
    )

    print(
        f"  bias = "
        f"{overall_metrics['bias_observed_minus_analytical_strain']:.6e} strain"
    )

    print(
        f"  max  = "
        f"{overall_metrics['max_abs_residual_strain']:.6e} strain"
    )

    print(
        f"  R2   = "
        f"{overall_metrics['r2']:.6f}"
    )

    # ------------------------------------------------------------------------
    # Comparison figures
    # ------------------------------------------------------------------------

    for station in station_names:

        plot_station_comparison(
            observed=strain,
            predicted=predicted,
            time=strain_time,
            station=station,
            output_dir=output_dir / "plots",
            show=args.show,
        )

    for component in COMPONENTS:

        plot_component_comparison(
            observed=strain,
            predicted=predicted,
            time=strain_time,
            component=component,
            station_names=station_names,
            output_dir=output_dir / "plots",
            show=args.show,
        )

    plot_all_stations_overview(
        observed=strain,
        predicted=predicted,
        time=strain_time,
        station_names=station_names,
        output_dir=output_dir / "plots",
        show=args.show,
    )

    plot_all_components_overview(
        observed=strain,
        predicted=predicted,
        time=strain_time,
        station_names=station_names,
        output_dir=output_dir / "plots",
        show=args.show,
    )

    plot_parity(
        comparison,
        output_dir / "plots",
        args.show,
    )

    # ------------------------------------------------------------------------
    # Machine-readable summary
    # ------------------------------------------------------------------------

    summary = build_verification_summary(
        strain_file=args.strain_file,
        pressure_file=args.pressure_file,
        stations_file=args.stations_file,
        pressure_monitors_file=(
            args.pressure_monitors_file
        ),
        strain=strain,
        pressure=pressure,
        station_names=station_names,
        components=components,
        pressure_channels=pressure_channels,
        station_metrics=station_metrics,
        component_metrics=component_metrics,
        overall_metrics=overall_metrics,
    )

    with open(
        output_dir
        / "verification_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        f"[INFO] Saved "
        f"{output_dir / 'verification_summary.json'}"
    )

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        "TRANSIENT VERIFICATION COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"\nOutput directory:\n"
        f"  {output_dir}"
    )


if __name__ == "__main__":
    main()