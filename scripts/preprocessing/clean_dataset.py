#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess COMSOL observation exports for the AVANT multi-station workflow.

The script supports:

1. Strain observations:
       eXX
       eYY
       eZZ
       eXY

2. Optional pressure-monitor observations:
       Pressure (Pa), Point: (x, y, z)

Canonical strain output
-----------------------
    time_s
    eXX_S01 ... eXX_S08
    eYY_S01 ... eYY_S08
    eZZ_S01 ... eZZ_S08
    eXY_S01 ... eXY_S08

Canonical strain-station metadata
---------------------------------
    station
    x
    y
    z

Canonical pressure output
-------------------------
    time_s
    pressure_P01_Pa
    pressure_P02_Pa
    ...

Canonical pressure-monitor metadata
-----------------------------------
    monitor
    x
    y
    z
    units

The raw COMSOL export is never modified.

Station coordinates, pressure-monitor coordinates, strain-component
labels, and pressure units are extracted directly from the COMSOL
header.

Canonical station labels S01, S02, ... are assigned according to the
order in which unique strain-station coordinates first appear.

Pressure-monitor labels P01, P02, ... are assigned according to the
order in which pressure channels appear.

Backward compatibility
----------------------
The existing strain-only workflow remains valid:

python scripts/preprocessing/clean_dataset.py ^
    datasets/comsol/1darcy/raw/k1D.txt ^
    --output datasets/comsol/1darcy/processed/strain.csv ^
    --stations datasets/comsol/1darcy/metadata/stations.csv

Transient strain + pressure example
-----------------------------------
python scripts/preprocessing/clean_dataset.py ^
    datasets/comsol/1darcy/raw/avant_transient_k1darcy_strain_pressure.txt ^
    --output datasets/comsol/1darcy/processed/verification_transient/strain.csv ^
    --stations datasets/comsol/1darcy/metadata/verification_transient/strain_stations.csv ^
    --pressure-output datasets/comsol/1darcy/processed/verification_transient/pressure.csv ^
    --pressure-monitors datasets/comsol/1darcy/metadata/verification_transient/pressure_monitors.csv

Duplicate timestamps
--------------------
Duplicate timestamps are preserved by default and reported.

If duplicate timestamps are known to represent duplicate COMSOL output
records and should be collapsed, use --deduplicate-time.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# COMSOL format definitions
# ============================================================================

SUPPORTED_COMPONENTS = (
    "eXX",
    "eYY",
    "eZZ",
    "eXY",
)


# Example:
#
# solid2.eXX*1E9 (1), Point: (250, 433, -40)
#
STRAIN_CHANNEL_PATTERN = re.compile(
    r"solid2\.(eXX|eYY|eZZ|eXY)\*1E9\s*\([^)]*\),\s*"
    r"Point:\s*\(([^)]+)\)"
)


# Example:
#
# Pressure (Pa), Point: (-185, 25, -3)
#
PRESSURE_CHANNEL_PATTERN = re.compile(
    r"Pressure\s*\(([^)]+)\),\s*"
    r"Point:\s*\(([^)]+)\)"
)


# ============================================================================
# Coordinate parsing
# ============================================================================

def parse_coordinates(
    coordinate_text: str,
    observation_type: str,
) -> tuple[float, float, float]:
    """
    Parse a COMSOL coordinate string.

    Parameters
    ----------
    coordinate_text
        Text such as ``250, 433, -40``.

    observation_type
        Human-readable observation type used in error messages.

    Returns
    -------
    tuple
        (x, y, z)
    """

    coordinate_tokens = [
        value.strip()
        for value in coordinate_text.split(",")
    ]

    if len(coordinate_tokens) != 3:
        raise ValueError(
            f"Could not parse {observation_type} coordinates "
            f"from COMSOL header: {coordinate_text!r}"
        )

    try:
        x, y, z = (
            float(value)
            for value in coordinate_tokens
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid {observation_type} coordinates in "
            f"COMSOL header: {coordinate_text!r}"
        ) from exc

    return x, y, z


# ============================================================================
# Raw COMSOL reader
# ============================================================================

def read_comsol_file(
    path: Path,
) -> tuple[list[str], list[list[float]]]:
    """
    Read a raw COMSOL text export.

    COMSOL metadata/header lines begin with ``%``. Remaining non-empty
    lines are expected to contain numeric data.

    Returns
    -------
    header_lines
        COMSOL metadata/header lines.

    numeric_rows
        Numeric rows containing time followed by observation values.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"COMSOL input file not found: {path}"
        )

    header_lines: list[str] = []
    numeric_rows: list[list[float]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("%"):
                header_lines.append(line)
                continue

            tokens = line.split()

            try:
                float(tokens[0])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Unexpected non-comment line "
                    f"{line_number} in {path}:\n"
                    f"{raw_line.rstrip()}"
                ) from exc

            try:
                row = [
                    float(token)
                    for token in tokens
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric data on "
                    f"line {line_number} in {path}."
                ) from exc

            numeric_rows.append(row)

    if not header_lines:
        raise ValueError(
            f"No COMSOL header lines were found in {path}."
        )

    if not numeric_rows:
        raise ValueError(
            f"No numeric data rows were found in {path}."
        )

    return header_lines, numeric_rows


# ============================================================================
# Header parsing
# ============================================================================

def parse_strain_channels(
    header_lines: list[str],
) -> list[
    tuple[
        str,
        tuple[float, float, float],
    ]
]:
    """
    Extract strain-component and station-coordinate information.

    Channel ordering is preserved exactly as it appears in the COMSOL
    header.
    """

    channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ] = []

    for line in header_lines:

        for match in STRAIN_CHANNEL_PATTERN.finditer(
            line
        ):

            component = match.group(1)

            coordinates = parse_coordinates(
                match.group(2),
                "strain-station",
            )

            channels.append(
                (
                    component,
                    coordinates,
                )
            )

    if not channels:
        raise ValueError(
            "No supported strain channels were found in "
            "the COMSOL header.\n"
            "Expected entries similar to:\n"
            "'solid2.eXX*1E9 (1), Point: (x, y, z)'"
        )

    return channels


def parse_pressure_channels(
    header_lines: list[str],
) -> list[
    tuple[
        str,
        tuple[float, float, float],
    ]
]:
    """
    Extract optional pressure-monitor channels.

    Returns
    -------
    pressure_channels
        Ordered list containing

            (units, (x, y, z))

        for each pressure channel.

    Notes
    -----
    Pressure is optional. A strain-only COMSOL export therefore
    returns an empty list.
    """

    channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ] = []

    for line in header_lines:

        for match in PRESSURE_CHANNEL_PATTERN.finditer(
            line
        ):

            units = match.group(1).strip()

            coordinates = parse_coordinates(
                match.group(2),
                "pressure-monitor",
            )

            channels.append(
                (
                    units,
                    coordinates,
                )
            )

    return channels


# Backward-compatible alias for existing imports/users.
parse_channel_header = parse_strain_channels


# ============================================================================
# Raw numeric-data validation
# ============================================================================

def validate_raw_numeric_rows(
    numeric_rows: list[list[float]],
    strain_channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ],
    pressure_channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ],
) -> None:
    """
    Validate raw numeric row lengths against parsed observations.
    """

    if not numeric_rows:
        raise ValueError(
            "No numeric COMSOL rows were supplied."
        )

    n_data_columns = len(
        numeric_rows[0]
    )

    for row_number, row in enumerate(
        numeric_rows,
        start=1,
    ):

        if len(row) != n_data_columns:

            raise ValueError(
                "Inconsistent number of columns in "
                "numeric COMSOL data. "
                f"Row {row_number} has {len(row)} "
                f"columns; expected {n_data_columns}."
            )

    expected_columns = (
        1
        + len(strain_channels)
        + len(pressure_channels)
    )

    if n_data_columns != expected_columns:

        raise ValueError(
            "Mismatch between COMSOL header and numeric data.\n"
            f"Header describes "
            f"{len(strain_channels)} strain channels, "
            f"{len(pressure_channels)} pressure channels, "
            f"plus time "
            f"({expected_columns} total columns), "
            f"but numeric rows contain "
            f"{n_data_columns} columns."
        )


# ============================================================================
# Strain-station metadata
# ============================================================================

def build_station_table(
    channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ],
) -> tuple[
    pd.DataFrame,
    dict[
        tuple[float, float, float],
        str,
    ],
]:
    """
    Construct canonical strain-station metadata.

    Station IDs are assigned according to first appearance of each
    unique coordinate in the COMSOL strain-channel header.
    """

    coordinate_to_station: dict[
        tuple[float, float, float],
        str,
    ] = {}

    records: list[
        dict[str, float | str]
    ] = []

    for _, coordinates in channels:

        if coordinates in coordinate_to_station:
            continue

        station_id = (
            f"S{len(coordinate_to_station) + 1:02d}"
        )

        coordinate_to_station[
            coordinates
        ] = station_id

        x, y, z = coordinates

        records.append(
            {
                "station": station_id,
                "x": x,
                "y": y,
                "z": z,
            }
        )

    stations = pd.DataFrame(
        records,
        columns=[
            "station",
            "x",
            "y",
            "z",
        ],
    )

    return (
        stations,
        coordinate_to_station,
    )


# ============================================================================
# Pressure-monitor metadata
# ============================================================================

def build_pressure_monitor_table(
    pressure_channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ],
) -> pd.DataFrame:
    """
    Construct canonical pressure-monitor metadata.
    """

    records: list[
        dict[str, float | str]
    ] = []

    for index, (
        units,
        coordinates,
    ) in enumerate(
        pressure_channels,
        start=1,
    ):

        x, y, z = coordinates

        records.append(
            {
                "monitor": f"P{index:02d}",
                "x": x,
                "y": y,
                "z": z,
                "units": units,
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "monitor",
            "x",
            "y",
            "z",
            "units",
        ],
    )


# ============================================================================
# Canonical strain dataset
# ============================================================================

def build_clean_strain_dataframe(
    numeric_rows: list[list[float]],
    channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ],
    coordinate_to_station: dict[
        tuple[float, float, float],
        str,
    ],
) -> pd.DataFrame:
    """
    Convert raw COMSOL strain observations into canonical format.

    Output ordering is component-major, then station-major:

        time_s

        eXX_S01 ... eXX_S08
        eYY_S01 ... eYY_S08
        eZZ_S01 ... eZZ_S08
        eXY_S01 ... eXY_S08
    """

    data = np.asarray(
        numeric_rows,
        dtype=float,
    )

    raw = pd.DataFrame(data)

    time = raw.iloc[
        :,
        0,
    ].to_numpy(
        dtype=float
    )

    channel_map: dict[
        tuple[str, str],
        int,
    ] = {}

    station_ids: list[str] = []

    for raw_index, (
        component,
        coordinates,
    ) in enumerate(
        channels,
        start=1,
    ):

        station = (
            coordinate_to_station[
                coordinates
            ]
        )

        station_ids.append(
            station
        )

        key = (
            component,
            station,
        )

        if key in channel_map:
            raise ValueError(
                "Duplicate COMSOL strain channel "
                f"detected: {component} at {station}."
            )

        channel_map[
            key
        ] = raw_index

    unique_station_ids = list(
        dict.fromkeys(
            station_ids
        )
    )

    ordered_columns: dict[
        str,
        np.ndarray,
    ] = {
        "time_s": time
    }

    for component in SUPPORTED_COMPONENTS:

        for station in unique_station_ids:

            key = (
                component,
                station,
            )

            if key not in channel_map:
                raise ValueError(
                    f"Missing {component} channel "
                    f"for station {station}."
                )

            raw_index = (
                channel_map[
                    key
                ]
            )

            ordered_columns[
                f"{component}_{station}"
            ] = raw.iloc[
                :,
                raw_index,
            ].to_numpy(
                dtype=float
            )

    return pd.DataFrame(
        ordered_columns
    )


# ============================================================================
# Canonical pressure dataset
# ============================================================================

def sanitize_units(
    units: str,
) -> str:
    """
    Convert a COMSOL unit string into a safe column-name token.
    """

    safe_units = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        units,
    ).strip("_")

    if not safe_units:
        return "unknown"

    return safe_units


def build_clean_pressure_dataframe(
    numeric_rows: list[list[float]],
    strain_channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ],
    pressure_channels: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ],
) -> pd.DataFrame:
    """
    Convert optional COMSOL pressure observations to canonical format.

    The current COMSOL export contract places pressure channels after
    the strain channels. The complete raw column count is validated
    before this function is called.
    """

    data = np.asarray(
        numeric_rows,
        dtype=float,
    )

    output: dict[
        str,
        np.ndarray,
    ] = {
        "time_s": data[:, 0]
    }

    first_pressure_column = (
        1
        + len(strain_channels)
    )

    for offset, (
        units,
        _,
    ) in enumerate(
        pressure_channels
    ):

        monitor = (
            f"P{offset + 1:02d}"
        )

        safe_units = sanitize_units(
            units
        )

        column_name = (
            f"pressure_{monitor}_{safe_units}"
        )

        raw_column = (
            first_pressure_column
            + offset
        )

        output[
            column_name
        ] = data[
            :,
            raw_column,
        ]

    return pd.DataFrame(
        output
    )


# ============================================================================
# Strain validation
# ============================================================================

def validate_clean_dataset(
    df: pd.DataFrame,
    stations: pd.DataFrame,
) -> None:
    """
    Validate processed strain observations and station metadata.
    """

    if "time_s" not in df.columns:
        raise ValueError(
            "Cleaned strain dataset must "
            "contain 'time_s'."
        )

    time = df[
        "time_s"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        time
    ).all():
        raise ValueError(
            "Time column contains NaN "
            "or infinite values."
        )

    duplicate_count = int(
        df[
            "time_s"
        ].duplicated().sum()
    )

    if duplicate_count:

        print(
            f"[WARNING] Found {duplicate_count} "
            "duplicate time rows."
        )

        print(
            "[WARNING] Duplicate rows are preserved "
            "unless --deduplicate-time is specified."
        )

    if len(stations) == 0:
        raise ValueError(
            "No station metadata were generated."
        )

    required_station_columns = {
        "station",
        "x",
        "y",
        "z",
    }

    missing_station_columns = (
        required_station_columns
        - set(
            stations.columns
        )
    )

    if missing_station_columns:
        raise ValueError(
            "Station metadata are missing "
            "required columns: "
            f"{sorted(missing_station_columns)}"
        )

    if stations[
        "station"
    ].duplicated().any():
        raise ValueError(
            "Station IDs are not unique."
        )

    if stations[
        [
            "x",
            "y",
            "z",
        ]
    ].isna().any().any():

        raise ValueError(
            "Station metadata contain "
            "missing coordinates."
        )

    strain_columns = [
        column
        for column in df.columns
        if column != "time_s"
    ]

    if not strain_columns:
        raise ValueError(
            "No strain columns were generated."
        )

    strain_values = df[
        strain_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        strain_values
    ).all():

        raise ValueError(
            "Strain data contain NaN "
            "or infinite values."
        )

    expected_strain_columns = (
        len(SUPPORTED_COMPONENTS)
        * len(stations)
    )

    if (
        len(strain_columns)
        != expected_strain_columns
    ):

        raise ValueError(
            f"Expected {expected_strain_columns} "
            f"strain columns "
            f"({len(SUPPORTED_COMPONENTS)} "
            f"components x "
            f"{len(stations)} stations), "
            f"found {len(strain_columns)}."
        )


# ============================================================================
# Pressure validation
# ============================================================================

def validate_pressure_dataset(
    pressure: pd.DataFrame,
    monitors: pd.DataFrame,
) -> None:
    """
    Validate processed pressure observations and monitor metadata.
    """

    if "time_s" not in pressure.columns:
        raise ValueError(
            "Pressure dataset must "
            "contain 'time_s'."
        )

    if len(monitors) == 0:
        raise ValueError(
            "Pressure data were requested but "
            "no pressure monitors were parsed."
        )

    required_monitor_columns = {
        "monitor",
        "x",
        "y",
        "z",
        "units",
    }

    missing_columns = (
        required_monitor_columns
        - set(
            monitors.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Pressure-monitor metadata are "
            "missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if monitors[
        "monitor"
    ].duplicated().any():

        raise ValueError(
            "Pressure-monitor IDs "
            "are not unique."
        )

    if monitors[
        [
            "x",
            "y",
            "z",
        ]
    ].isna().any().any():

        raise ValueError(
            "Pressure-monitor metadata "
            "contain missing coordinates."
        )

    pressure_columns = [
        column
        for column in pressure.columns
        if column != "time_s"
    ]

    if (
        len(pressure_columns)
        != len(monitors)
    ):

        raise ValueError(
            "Number of pressure-data columns "
            "does not match number of "
            "pressure monitors."
        )

    values = pressure.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "Pressure dataset contains "
            "NaN or infinite values."
        )


# ============================================================================
# Duplicate-time handling
# ============================================================================

def deduplicate_time_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse duplicate timestamps by averaging numeric observations.

    Duplicate timestamps are intentionally preserved by default.
    """

    if not df[
        "time_s"
    ].duplicated().any():

        return df

    return (
        df.groupby(
            "time_s",
            as_index=False,
            sort=True,
        )
        .mean(
            numeric_only=True
        )
    )


# ============================================================================
# Main preprocessing workflow
# ============================================================================

def process_comsol_dataset(
    input_file: Path,
    output_file: Path,
    stations_file: Path,
    deduplicate_time: bool = False,
    pressure_output_file: Path | None = None,
    pressure_monitors_file: Path | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """
    Process a raw COMSOL observation export.

    Returns
    -------
    strain
        Canonical strain DataFrame.

    stations
        Strain-station metadata.

    pressure
        Canonical pressure DataFrame if pressure channels are present;
        otherwise None.

    pressure_monitors
        Pressure-monitor metadata if pressure channels are present;
        otherwise None.
    """

    print(
        f"[INFO] Reading raw COMSOL file: "
        f"{input_file}"
    )

    header_lines, numeric_rows = (
        read_comsol_file(
            input_file
        )
    )

    print(
        f"[INFO] Parsed "
        f"{len(header_lines)} "
        "header/comment lines."
    )

    print(
        f"[INFO] Parsed "
        f"{len(numeric_rows)} "
        "numeric rows."
    )

    # ----------------------------------------------------------------
    # Parse observations
    # ----------------------------------------------------------------

    strain_channels = (
        parse_strain_channels(
            header_lines
        )
    )

    pressure_channels = (
        parse_pressure_channels(
            header_lines
        )
    )

    print(
        f"[INFO] Identified "
        f"{len(strain_channels)} "
        "strain channels."
    )

    print(
        f"[INFO] Identified "
        f"{len(pressure_channels)} "
        "pressure channels."
    )

    # ----------------------------------------------------------------
    # Validate raw observation count
    # ----------------------------------------------------------------

    validate_raw_numeric_rows(
        numeric_rows=numeric_rows,
        strain_channels=strain_channels,
        pressure_channels=pressure_channels,
    )

    # ----------------------------------------------------------------
    # Build strain metadata
    # ----------------------------------------------------------------

    (
        stations,
        coordinate_to_station,
    ) = build_station_table(
        strain_channels
    )

    print(
        f"[INFO] Identified "
        f"{len(stations)} "
        "unique strain stations."
    )

    # ----------------------------------------------------------------
    # Build strain data
    # ----------------------------------------------------------------

    strain = (
        build_clean_strain_dataframe(
            numeric_rows=numeric_rows,
            channels=strain_channels,
            coordinate_to_station=(
                coordinate_to_station
            ),
        )
    )

    validate_clean_dataset(
        strain,
        stations,
    )

    # ----------------------------------------------------------------
    # Optional pressure data
    # ----------------------------------------------------------------

    pressure: pd.DataFrame | None = None

    pressure_monitors: (
        pd.DataFrame | None
    ) = None

    if pressure_channels:

        pressure_monitors = (
            build_pressure_monitor_table(
                pressure_channels
            )
        )

        pressure = (
            build_clean_pressure_dataframe(
                numeric_rows=numeric_rows,
                strain_channels=(
                    strain_channels
                ),
                pressure_channels=(
                    pressure_channels
                ),
            )
        )

        validate_pressure_dataset(
            pressure,
            pressure_monitors,
        )

    # ----------------------------------------------------------------
    # Optional duplicate-time processing
    # ----------------------------------------------------------------

    if deduplicate_time:

        before = len(strain)

        strain = (
            deduplicate_time_rows(
                strain
            )
        )

        after = len(strain)

        print(
            f"[INFO] Deduplicated "
            f"strain time rows: "
            f"{before} -> {after}."
        )

        validate_clean_dataset(
            strain,
            stations,
        )

        if pressure is not None:

            pressure_before = (
                len(pressure)
            )

            pressure = (
                deduplicate_time_rows(
                    pressure
                )
            )

            pressure_after = (
                len(pressure)
            )

            print(
                f"[INFO] Deduplicated "
                f"pressure time rows: "
                f"{pressure_before} -> "
                f"{pressure_after}."
            )

            validate_pressure_dataset(
                pressure,
                pressure_monitors,
            )

    # ----------------------------------------------------------------
    # Validate pressure-output configuration
    # ----------------------------------------------------------------

    if (
        pressure_output_file is None
    ) != (
        pressure_monitors_file is None
    ):
        raise ValueError(
            "--pressure-output and "
            "--pressure-monitors must "
            "be supplied together."
        )

    if (
        pressure_output_file is not None
        and pressure is None
    ):
        raise ValueError(
            "Pressure output paths were supplied, "
            "but no pressure channels were found "
            "in the COMSOL input."
        )

    if (
        pressure is not None
        and pressure_output_file is None
    ):
        print(
            "[WARNING] Pressure observations were "
            "found but no pressure output paths "
            "were supplied."
        )

        print(
            "[WARNING] Strain outputs will be "
            "written; pressure observations will "
            "not be written."
        )

    # ----------------------------------------------------------------
    # Create output directories
    # ----------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stations_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------
    # Save strain outputs
    # ----------------------------------------------------------------

    strain.to_csv(
        output_file,
        index=False,
    )

    stations.to_csv(
        stations_file,
        index=False,
    )

    # ----------------------------------------------------------------
    # Save optional pressure outputs
    # ----------------------------------------------------------------

    if (
        pressure is not None
        and pressure_output_file is not None
        and pressure_monitors_file is not None
    ):

        pressure_output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pressure_monitors_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pressure.to_csv(
            pressure_output_file,
            index=False,
        )

        pressure_monitors.to_csv(
            pressure_monitors_file,
            index=False,
        )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    print()
    print(
        "[INFO] COMSOL preprocessing complete."
    )

    print(
        f"[INFO] Clean strain dataset: "
        f"{output_file}"
    )

    print(
        f"[INFO] Strain-station metadata: "
        f"{stations_file}"
    )

    print(
        f"[INFO] Dataset rows: "
        f"{len(strain)}"
    )

    print(
        f"[INFO] Strain channels: "
        f"{len(strain.columns) - 1}"
    )

    print(
        f"[INFO] Number of strain stations: "
        f"{len(stations)}"
    )

    if pressure is not None:

        print(
            f"[INFO] Pressure channels: "
            f"{len(pressure.columns) - 1}"
        )

        print(
            f"[INFO] Number of pressure monitors: "
            f"{len(pressure_monitors)}"
        )

        if pressure_output_file is not None:

            print(
                f"[INFO] Clean pressure dataset: "
                f"{pressure_output_file}"
            )

            print(
                f"[INFO] Pressure-monitor metadata: "
                f"{pressure_monitors_file}"
            )

    return (
        strain,
        stations,
        pressure,
        pressure_monitors,
    )


# ============================================================================
# Command-line interface
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw COMSOL observation "
            "text export into canonical AVANT "
            "strain data and optional pressure data."
        )
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help=(
            "Path to the raw COMSOL .txt export."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Path for the cleaned strain CSV."
        ),
    )

    parser.add_argument(
        "--stations",
        type=Path,
        required=True,
        help=(
            "Path for the strain-station "
            "metadata CSV."
        ),
    )

    parser.add_argument(
        "--pressure-output",
        type=Path,
        default=None,
        help=(
            "Optional path for the cleaned "
            "pressure CSV."
        ),
    )

    parser.add_argument(
        "--pressure-monitors",
        type=Path,
        default=None,
        help=(
            "Optional path for pressure-monitor "
            "metadata CSV."
        ),
    )

    parser.add_argument(
        "--deduplicate-time",
        action="store_true",
        help=(
            "Average duplicate timestamps. "
            "By default duplicate rows are "
            "preserved and only reported."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    process_comsol_dataset(
        input_file=args.input_file,
        output_file=args.output,
        stations_file=args.stations,
        deduplicate_time=(
            args.deduplicate_time
        ),
        pressure_output_file=(
            args.pressure_output
        ),
        pressure_monitors_file=(
            args.pressure_monitors
        ),
    )


if __name__ == "__main__":
    main()