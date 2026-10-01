#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot cleaned transient strain and optional pressure observations.

This script provides the observational-data quality-control plotting layer
for the AVANT multi-station workflow.

It reads processed observational datasets only. It does NOT:

    - run the analytical forward model;
    - perform parameter fitting;
    - perform sensitivity analysis;
    - perform Bayesian inversion;
    - modify, smooth, interpolate, resample, or deduplicate observations.

Required strain-data contract
-----------------------------
The strain CSV must contain:

    time_s
    eXX_<station>
    eYY_<station>
    eZZ_<station>
    eXY_<station>

For example:

    time_s
    eXX_S01 ... eXX_S08
    eYY_S01 ... eYY_S08
    eZZ_S01 ... eZZ_S08
    eXY_S01 ... eXY_S08

Optional strain-station metadata
--------------------------------
If supplied, the station metadata CSV must contain:

    station,x,y,z

Optional pressure-data contract
-------------------------------
If supplied, the pressure CSV must contain:

    time_s
    pressure_<monitor>_<units>

For example:

    time_s
    pressure_P01_Pa

Optional pressure-monitor metadata
----------------------------------
If supplied, the pressure-monitor metadata CSV must contain:

    monitor,x,y,z,units

Outputs
-------
<output-dir>/
    stations/
        strain_S01.png
        strain_S02.png
        ...
    panels/
        strain_S01_panels.png
        strain_S02_panels.png
        ...
    overview/
        strain_all_stations.png
    pressure/
        pressure_P01.png
        pressure_all_monitors.png

PDF versions are optionally produced with --pdf.

Duplicate timestamps
--------------------
Duplicate timestamps are preserved exactly as supplied. They are reported
to the terminal because this script is intended to visualize the processed
observations without silently altering their data contract.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


# ============================================================================
# Plot style
# ============================================================================

def configure_plot_style() -> None:
    """
    Configure a consistent publication-quality Matplotlib style.
    """

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "legend.fontsize": 10,
            "figure.titlesize": 17,
            "savefig.dpi": 400,
        }
    )


# ============================================================================
# Command-line interface
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot cleaned AVANT transient strain observations "
            "and optional pressure-monitor observations."
        )
    )

    parser.add_argument(
        "--strain-file",
        type=Path,
        required=True,
        help=(
            "Processed strain CSV containing time_s and "
            "component/station columns."
        ),
    )

    parser.add_argument(
        "--pressure-file",
        type=Path,
        default=None,
        help=(
            "Optional processed pressure CSV containing "
            "time_s and pressure-monitor columns."
        ),
    )

    parser.add_argument(
        "--stations-file",
        type=Path,
        default=None,
        help=(
            "Optional strain-station metadata CSV containing "
            "station,x,y,z."
        ),
    )

    parser.add_argument(
        "--pressure-monitors-file",
        type=Path,
        default=None,
        help=(
            "Optional pressure-monitor metadata CSV containing "
            "monitor,x,y,z,units."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated figures.",
    )

    parser.add_argument(
        "--title-prefix",
        type=str,
        default="COMSOL Transient Observations",
        help=(
            "Main descriptive text used in figure titles. "
            "Default: 'COMSOL Transient Observations'."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG resolution. Default: 400 dpi.",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save vector PDF versions.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display generated figures interactively.",
    )

    return parser


# ============================================================================
# General validation
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


def validate_time_column(
    dataframe: pd.DataFrame,
    description: str,
) -> np.ndarray:
    """
    Validate and return the time vector.

    Duplicate timestamps are allowed because observational QA must preserve
    the processed dataset exactly.
    """

    if "time_s" not in dataframe.columns:
        raise RuntimeError(
            f"{description} must contain a 'time_s' column."
        )

    time = pd.to_numeric(
        dataframe["time_s"],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    if time.size == 0:
        raise RuntimeError(
            f"{description} contains no observations."
        )

    if not np.all(
        np.isfinite(time)
    ):
        raise RuntimeError(
            f"{description} contains non-finite time values."
        )

    if np.any(
        np.diff(time) < 0.0
    ):
        raise RuntimeError(
            f"{description} time values are not "
            "monotonically non-decreasing."
        )

    duplicate_count = int(
        pd.Series(
            time
        ).duplicated().sum()
    )

    if duplicate_count:

        print(
            f"[WARNING] {description}: "
            f"{duplicate_count} duplicate timestamp(s) found."
        )

        print(
            "[WARNING] Duplicate timestamps are preserved "
            "for observational QA."
        )

    return time


def validate_numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    description: str,
) -> None:
    """
    Require selected columns to contain finite numeric values.
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

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            f"{description} contains NaN or infinite values."
        )


# ============================================================================
# Strain-data discovery
# ============================================================================

def discover_strain_structure(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Discover station IDs and strain components from canonical column names.

    Station ordering follows first appearance in the dataset.
    Component ordering follows COMPONENTS.
    """

    station_order: list[str] = []

    available_components: set[str] = set()

    for column in dataframe.columns:

        if column == "time_s":
            continue

        match = STRAIN_PATTERN.match(
            str(column).strip()
        )

        if match is None:
            raise RuntimeError(
                "Unexpected column in strain dataset: "
                f"{column!r}"
            )

        component = (
            match.group(1)
        )

        station = (
            match.group(2).strip()
        )

        available_components.add(
            component
        )

        if station not in station_order:
            station_order.append(
                station
            )

    if not station_order:
        raise RuntimeError(
            "No strain stations were discovered."
        )

    components = [
        component
        for component in COMPONENTS
        if component in available_components
    ]

    if not components:
        raise RuntimeError(
            "No supported strain components were discovered."
        )

    for station in station_order:

        for component in components:

            column = (
                f"{component}_{station}"
            )

            if column not in dataframe.columns:
                raise RuntimeError(
                    "Incomplete strain-data contract. "
                    f"Missing channel: {column}"
                )

    return (
        station_order,
        components,
    )


# ============================================================================
# Strain-station metadata
# ============================================================================

def load_station_metadata(
    path: Path | None,
    station_names: list[str],
) -> pd.DataFrame | None:
    """
    Load and validate optional strain-station metadata.
    """

    if path is None:
        return None

    require_file(
        path,
        "Station metadata",
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
        raise RuntimeError(
            "Station metadata are missing required columns: "
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

    stations["station"] = (
        stations["station"]
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

    if stations[
        "station"
    ].duplicated().any():

        raise RuntimeError(
            "Station metadata contain duplicate station IDs."
        )

    metadata_names = (
        stations[
            "station"
        ].tolist()
    )

    if metadata_names != station_names:

        raise RuntimeError(
            "Station ordering in metadata does not match "
            "the strain dataset.\n"
            f"Data:     {station_names}\n"
            f"Metadata: {metadata_names}"
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

    if not np.all(
        np.isfinite(coordinates)
    ):
        raise RuntimeError(
            "Station metadata contain non-finite coordinates."
        )

    return stations


def station_subtitle(
    station: str,
    stations: pd.DataFrame | None,
) -> str:
    """
    Return a concise coordinate annotation for a strain station.
    """

    if stations is None:
        return ""

    row = stations.loc[
        stations[
            "station"
        ] == station
    ]

    if row.empty:
        return ""

    record = (
        row.iloc[0]
    )

    return (
        f"x = {record['x']:g} m, "
        f"y = {record['y']:g} m, "
        f"z = {record['z']:g} m"
    )


# ============================================================================
# Pressure-data discovery
# ============================================================================

def discover_pressure_channels(
    dataframe: pd.DataFrame,
) -> list[dict[str, str]]:
    """
    Discover canonical pressure-monitor channels.
    """

    channels: list[
        dict[str, str]
    ] = []

    for column in dataframe.columns:

        if column == "time_s":
            continue

        match = PRESSURE_PATTERN.match(
            str(column).strip()
        )

        if match is None:
            raise RuntimeError(
                "Unexpected column in pressure dataset: "
                f"{column!r}"
            )

        channels.append(
            {
                "column": str(column),
                "monitor": (
                    match.group(1).upper()
                ),
                "units": (
                    match.group(2)
                ),
            }
        )

    if not channels:
        raise RuntimeError(
            "Pressure file contains no pressure-monitor channels."
        )

    return channels


# ============================================================================
# Pressure-monitor metadata
# ============================================================================

def load_pressure_monitor_metadata(
    path: Path | None,
    pressure_channels: list[dict[str, str]],
) -> pd.DataFrame | None:
    """
    Load and validate optional pressure-monitor metadata.
    """

    if path is None:
        return None

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
        raise RuntimeError(
            "Pressure-monitor metadata are missing "
            "required columns: "
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

    if monitors[
        "monitor"
    ].duplicated().any():

        raise RuntimeError(
            "Pressure-monitor metadata contain duplicate monitor IDs."
        )

    coordinates = (
        monitors[
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

    if not np.all(
        np.isfinite(coordinates)
    ):
        raise RuntimeError(
            "Pressure-monitor metadata contain "
            "non-finite coordinates."
        )

    expected_monitors = [
        channel[
            "monitor"
        ]
        for channel in pressure_channels
    ]

    actual_monitors = (
        monitors[
            "monitor"
        ].tolist()
    )

    if actual_monitors != expected_monitors:

        raise RuntimeError(
            "Pressure-monitor metadata order does not match "
            "the pressure dataset.\n"
            f"Data:     {expected_monitors}\n"
            f"Metadata: {actual_monitors}"
        )

    return monitors


def pressure_monitor_subtitle(
    monitor: str,
    monitors: pd.DataFrame | None,
) -> str:
    """
    Return a concise coordinate annotation for a pressure monitor.
    """

    if monitors is None:
        return ""

    row = monitors.loc[
        monitors[
            "monitor"
        ] == monitor
    ]

    if row.empty:
        return ""

    record = (
        row.iloc[0]
    )

    return (
        f"x = {record['x']:g} m, "
        f"y = {record['y']:g} m, "
        f"z = {record['z']:g} m"
    )


# ============================================================================
# Figure-title helpers
# ============================================================================

def build_title(
    main_title: str,
    subtitle: str = "",
) -> str:
    """
    Build a consistent one- or two-line figure title.
    """

    if subtitle:

        return (
            f"{main_title}\n"
            f"{subtitle}"
        )

    return main_title


# ============================================================================
# Figure saving
# ============================================================================

def save_figure(
    fig: plt.Figure,
    output_base: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """
    Save a figure as PNG and optionally PDF.
    """

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_base.with_suffix(
            ".png"
        )
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    print(
        f"[INFO] Saved {png_path}"
    )

    if save_pdf:

        pdf_path = (
            output_base.with_suffix(
                ".pdf"
            )
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        print(
            f"[INFO] Saved {pdf_path}"
        )


# ============================================================================
# Combined strain figure
# ============================================================================

def plot_station_combined(
    dataframe: pd.DataFrame,
    time: np.ndarray,
    station: str,
    components: list[str],
    stations: pd.DataFrame | None,
    title_prefix: str,
) -> plt.Figure:
    """
    Plot all strain components for one station on a common axis.
    """

    fig, ax = plt.subplots(
        figsize=(
            12.0,
            7.2,
        )
    )

    for component in components:

        column = (
            f"{component}_{station}"
        )

        values = (
            dataframe[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        ax.plot(
            time,
            values,
            linewidth=2.3,
            label=COMPONENT_LABELS[
                component
            ],
            color=COMPONENT_COLORS[
                component
            ],
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        r"Strain (n$\varepsilon$)"
    )

    subtitle = (
        station_subtitle(
            station,
            stations,
        )
    )

    title = build_title(
        (
            f"{title_prefix} — "
            f"{station}"
        ),
        subtitle,
    )

    ax.set_title(
        title,
        pad=14,
        linespacing=1.5,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    fig.tight_layout(
        pad=1.8
    )

    return fig


# ============================================================================
# Separate-component station figure
# ============================================================================

def plot_station_panels(
    dataframe: pd.DataFrame,
    time: np.ndarray,
    station: str,
    components: list[str],
    stations: pd.DataFrame | None,
    title_prefix: str,
) -> plt.Figure:
    """
    Plot strain components for one station in separate panels.

    Separate axes are useful when components differ substantially in
    magnitude.
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

    axes = (
        axes.ravel()
    )

    for index, ax in enumerate(
        axes
    ):

        if index >= len(
            components
        ):

            ax.set_visible(
                False
            )

            continue

        component = (
            components[
                index
            ]
        )

        column = (
            f"{component}_{station}"
        )

        values = (
            dataframe[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        ax.plot(
            time,
            values,
            linewidth=2.2,
            color=COMPONENT_COLORS[
                component
            ],
        )

        ax.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
        )

        ax.set_title(
            COMPONENT_LABELS[
                component
            ],
            pad=8,
        )

        ax.set_ylabel(
            r"Strain (n$\varepsilon$)"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.25,
        )

    for ax in axes[
        -2:
    ]:

        if ax.get_visible():

            ax.set_xlabel(
                "Time (s)"
            )

    subtitle = (
        station_subtitle(
            station,
            stations,
        )
    )

    title = build_title(
        (
            f"{title_prefix} — "
            f"{station}"
        ),
        subtitle,
    )

    fig.suptitle(
        title,
        y=0.985,
        linespacing=1.5,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.90,
        ),
        pad=1.4,
    )

    return fig


# ============================================================================
# Strain overview
# ============================================================================

def plot_strain_overview(
    dataframe: pd.DataFrame,
    time: np.ndarray,
    stations: list[str],
    components: list[str],
    title_prefix: str,
) -> plt.Figure:
    """
    Plot all strain stations in one multi-panel overview figure.
    """

    n_stations = (
        len(stations)
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
            3.6 * n_rows,
        ),
        sharex=True,
    )

    axes_array = (
        np.atleast_1d(
            axes
        ).ravel()
    )

    for index, ax in enumerate(
        axes_array
    ):

        if index >= n_stations:

            ax.set_visible(
                False
            )

            continue

        station = (
            stations[
                index
            ]
        )

        for component in components:

            column = (
                f"{component}_{station}"
            )

            values = (
                dataframe[
                    column
                ]
                .to_numpy(
                    dtype=float
                )
            )

            ax.plot(
                time,
                values,
                linewidth=1.6,
                color=COMPONENT_COLORS[
                    component
                ],
                label=COMPONENT_LABELS[
                    component
                ],
            )

        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
            alpha=0.4,
        )

        ax.set_title(
            station,
            pad=7,
        )

        ax.set_ylabel(
            r"Strain (n$\varepsilon$)"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.20,
        )

    for ax in axes_array:

        if ax.get_visible():

            ax.set_xlabel(
                "Time (s)"
            )

    visible_axes = [
        ax
        for ax in axes_array
        if ax.get_visible()
    ]

    if visible_axes:

        handles, labels = (
            visible_axes[
                0
            ].get_legend_handles_labels()
        )

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(
                components
            ),
            frameon=True,
            bbox_to_anchor=(
                0.5,
                0.012,
            ),
        )

    fig.suptitle(
        (
            f"{title_prefix} — "
            "All Strain Stations"
        ),
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

    return fig


# ============================================================================
# Pressure display scaling
# ============================================================================

def choose_pressure_scale(
    values: np.ndarray,
    units: str,
) -> tuple[float, str]:
    """
    Choose readable pressure display units.

    The underlying pressure observations are not modified.
    """

    normalized_units = (
        units
        .strip()
        .lower()
    )

    if normalized_units != "pa":

        return (
            1.0,
            units,
        )

    maximum = float(
        np.max(
            np.abs(
                values
            )
        )
    )

    if maximum >= 1.0e6:

        return (
            1.0e-6,
            "MPa",
        )

    if maximum >= 1.0e3:

        return (
            1.0e-3,
            "kPa",
        )

    return (
        1.0,
        "Pa",
    )


# ============================================================================
# Individual pressure-monitor figure
# ============================================================================

def plot_pressure_monitor(
    dataframe: pd.DataFrame,
    time: np.ndarray,
    channel: dict[str, str],
    monitors: pd.DataFrame | None,
    title_prefix: str,
) -> plt.Figure:
    """
    Plot one pressure-monitor time series.
    """

    column = (
        channel[
            "column"
        ]
    )

    monitor = (
        channel[
            "monitor"
        ]
    )

    units = (
        channel[
            "units"
        ]
    )

    values = (
        dataframe[
            column
        ]
        .to_numpy(
            dtype=float
        )
    )

    (
        scale,
        display_units,
    ) = choose_pressure_scale(
        values,
        units,
    )

    fig, ax = plt.subplots(
        figsize=(
            12.0,
            6.8,
        )
    )

    ax.plot(
        time,
        values * scale,
        linewidth=2.4,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        f"Pressure ({display_units})"
    )

    subtitle = (
        pressure_monitor_subtitle(
            monitor,
            monitors,
        )
    )

    title = build_title(
        (
            f"{title_prefix} — "
            f"Pressure {monitor}"
        ),
        subtitle,
    )

    ax.set_title(
        title,
        pad=14,
        linespacing=1.5,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    fig.tight_layout(
        pad=1.8
    )

    return fig


# ============================================================================
# Pressure overview
# ============================================================================

def plot_pressure_overview(
    dataframe: pd.DataFrame,
    time: np.ndarray,
    channels: list[dict[str, str]],
    title_prefix: str,
) -> plt.Figure:
    """
    Plot all pressure monitors on one common figure.
    """

    fig, ax = plt.subplots(
        figsize=(
            12.0,
            6.8,
        )
    )

    display_units_used: set[
        str
    ] = set()

    for channel in channels:

        values = (
            dataframe[
                channel[
                    "column"
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        (
            scale,
            display_units,
        ) = choose_pressure_scale(
            values,
            channel[
                "units"
            ],
        )

        display_units_used.add(
            display_units
        )

        ax.plot(
            time,
            values * scale,
            linewidth=2.2,
            label=channel[
                "monitor"
            ],
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    if len(
        display_units_used
    ) == 1:

        display_units = next(
            iter(
                display_units_used
            )
        )

        ax.set_ylabel(
            f"Pressure ({display_units})"
        )

    else:

        ax.set_ylabel(
            "Pressure (display-scaled units)"
        )

    ax.set_title(
        (
            f"{title_prefix} — "
            "Pressure Monitors"
        ),
        pad=14,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    fig.tight_layout(
        pad=1.8
    )

    return fig


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """
    Command-line entry point.
    """

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    configure_plot_style()

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )

    # ------------------------------------------------------------------------
    # Load strain observations
    # ------------------------------------------------------------------------

    require_file(
        args.strain_file,
        "Processed strain dataset",
    )

    strain = pd.read_csv(
        args.strain_file
    )

    strain.columns = [
        str(
            column
        ).strip()
        for column in strain.columns
    ]

    strain_time = (
        validate_time_column(
            strain,
            "Strain dataset",
        )
    )

    (
        station_names,
        components,
    ) = discover_strain_structure(
        strain
    )

    strain_columns = [
        f"{component}_{station}"
        for component in components
        for station in station_names
    ]

    validate_numeric_columns(
        strain,
        strain_columns,
        "Strain dataset",
    )

    stations = (
        load_station_metadata(
            args.stations_file,
            station_names,
        )
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print(
        "=" * 72
    )

    print(
        "AVANT OBSERVED TRANSIENT DATA QA"
    )

    print(
        "=" * 72
    )

    print(
        "\nStrain dataset"
    )

    print(
        f"  file       = "
        f"{args.strain_file}"
    )

    print(
        f"  rows       = "
        f"{len(strain)}"
    )

    print(
        f"  stations   = "
        f"{len(station_names)}"
    )

    print(
        f"  names      = "
        f"{station_names}"
    )

    print(
        f"  components = "
        f"{components}"
    )

    print(
        f"  channels   = "
        f"{len(strain_columns)}"
    )

    print(
        f"  time range = "
        f"{strain_time.min():g} to "
        f"{strain_time.max():g} s"
    )

    # ------------------------------------------------------------------------
    # Individual strain-station figures
    # ------------------------------------------------------------------------

    for station in station_names:

        combined = (
            plot_station_combined(
                dataframe=strain,
                time=strain_time,
                station=station,
                components=components,
                stations=stations,
                title_prefix=(
                    args.title_prefix
                ),
            )
        )

        save_figure(
            combined,
            (
                args.output_dir
                / "stations"
                / f"strain_{station}"
            ),
            dpi=args.dpi,
            save_pdf=args.pdf,
        )

        if args.show:
            plt.show()

        plt.close(
            combined
        )

        panels = (
            plot_station_panels(
                dataframe=strain,
                time=strain_time,
                station=station,
                components=components,
                stations=stations,
                title_prefix=(
                    args.title_prefix
                ),
            )
        )

        save_figure(
            panels,
            (
                args.output_dir
                / "panels"
                / f"strain_{station}_panels"
            ),
            dpi=args.dpi,
            save_pdf=args.pdf,
        )

        if args.show:
            plt.show()

        plt.close(
            panels
        )

    # ------------------------------------------------------------------------
    # Strain overview
    # ------------------------------------------------------------------------

    overview = (
        plot_strain_overview(
            dataframe=strain,
            time=strain_time,
            stations=station_names,
            components=components,
            title_prefix=(
                args.title_prefix
            ),
        )
    )

    save_figure(
        overview,
        (
            args.output_dir
            / "overview"
            / "strain_all_stations"
        ),
        dpi=args.dpi,
        save_pdf=args.pdf,
    )

    if args.show:
        plt.show()

    plt.close(
        overview
    )

    # ------------------------------------------------------------------------
    # Optional pressure observations
    # ------------------------------------------------------------------------

    if args.pressure_file is None:

        print()

        print(
            "[INFO] No pressure file supplied."
        )

        print(
            "[INFO] Strain plotting complete."
        )

    else:

        require_file(
            args.pressure_file,
            "Processed pressure dataset",
        )

        pressure = pd.read_csv(
            args.pressure_file
        )

        pressure.columns = [
            str(
                column
            ).strip()
            for column in pressure.columns
        ]

        pressure_time = (
            validate_time_column(
                pressure,
                "Pressure dataset",
            )
        )

        pressure_channels = (
            discover_pressure_channels(
                pressure
            )
        )

        pressure_columns = [
            channel[
                "column"
            ]
            for channel in pressure_channels
        ]

        validate_numeric_columns(
            pressure,
            pressure_columns,
            "Pressure dataset",
        )

        if (
            pressure_time.shape
            != strain_time.shape
            or not np.array_equal(
                pressure_time,
                strain_time,
            )
        ):

            raise RuntimeError(
                "Pressure and strain time vectors "
                "do not match exactly."
            )

        monitors = (
            load_pressure_monitor_metadata(
                args.pressure_monitors_file,
                pressure_channels,
            )
        )

        print(
            "\nPressure dataset"
        )

        print(
            f"  file       = "
            f"{args.pressure_file}"
        )

        print(
            f"  rows       = "
            f"{len(pressure)}"
        )

        print(
            f"  monitors   = "
            f"{len(pressure_channels)}"
        )

        pressure_monitor_names = [
            channel["monitor"]
            for channel in pressure_channels
        ]

        print(
            f"  names      = "
            f"{pressure_monitor_names}"
        )
        # --------------------------------------------------------------------
        # Individual pressure-monitor figures
        # --------------------------------------------------------------------

        for channel in pressure_channels:

            figure = (
                plot_pressure_monitor(
                    dataframe=pressure,
                    time=pressure_time,
                    channel=channel,
                    monitors=monitors,
                    title_prefix=(
                        args.title_prefix
                    ),
                )
            )

            save_figure(
                figure,
                (
                    args.output_dir
                    / "pressure"
                    / (
                        "pressure_"
                        f"{channel['monitor']}"
                    )
                ),
                dpi=args.dpi,
                save_pdf=args.pdf,
            )

            if args.show:
                plt.show()

            plt.close(
                figure
            )

        # --------------------------------------------------------------------
        # Pressure overview
        # --------------------------------------------------------------------

        pressure_overview = (
            plot_pressure_overview(
                dataframe=pressure,
                time=pressure_time,
                channels=pressure_channels,
                title_prefix=(
                    args.title_prefix
                ),
            )
        )

        save_figure(
            pressure_overview,
            (
                args.output_dir
                / "pressure"
                / "pressure_all_monitors"
            ),
            dpi=args.dpi,
            save_pdf=args.pdf,
        )

        if args.show:
            plt.show()

        plt.close(
            pressure_overview
        )

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "OBSERVED TRANSIENT DATA QA COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Output directory:\n"
        f"  {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()