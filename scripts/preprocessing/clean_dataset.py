"""
Preprocess COMSOL strain exports for the Avant multi-station workflow.

This script converts a raw COMSOL text export into two canonical files:

1. A cleaned strain CSV with columns such as:
       time_s
       eXX_S01 ... eXX_S08
       eYY_S01 ... eYY_S08
       eZZ_S01 ... eZZ_S08
       eXY_S01 ... eXY_S08

2. A station metadata CSV with:
       station
       x
       y
       z

The raw COMSOL export is never modified.

Station coordinates and strain-component labels are extracted directly
from the COMSOL column headers. Canonical station labels S01, S02, ...
are assigned according to the order in which unique station coordinates
first appear in the COMSOL header.

Example
-------
python scripts/preprocessing/clean_dataset.py ^
    datasets/comsol/1darcy/raw/k1D.txt ^
    --output datasets/comsol/1darcy/processed/strain.csv ^
    --stations datasets/comsol/1darcy/metadata/stations.csv

Duplicate timestamps
--------------------
Duplicate timestamps are preserved by default and reported to the user.

If duplicate timestamps are known to represent duplicate COMSOL output
records and should be collapsed, use:

python scripts/preprocessing/clean_dataset.py ^
    datasets/comsol/1darcy/raw/k1D.txt ^
    --output datasets/comsol/1darcy/processed/strain.csv ^
    --stations datasets/comsol/1darcy/metadata/stations.csv ^
    --deduplicate-time
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# COMSOL format definition
# ---------------------------------------------------------------------

SUPPORTED_COMPONENTS = ("eXX", "eYY", "eZZ", "eXY")

# Example COMSOL header entry:
#
# solid2.eXX*1E9 (1), Point: (250, 433, -40)
#
# Captures:
#   1. strain component: eXX
#   2. coordinate string: 250, 433, -40
#
CHANNEL_PATTERN = re.compile(
    r"solid2\.(eXX|eYY|eZZ|eXY)\*1E9\s*\([^)]*\),\s*"
    r"Point:\s*\(([^)]+)\)"
)


# ---------------------------------------------------------------------
# Raw COMSOL reader
# ---------------------------------------------------------------------

def read_comsol_file(
    path: Path,
) -> tuple[list[str], list[list[float]]]:
    """
    Read a raw COMSOL text export.

    COMSOL metadata/header lines begin with '%'. Numeric rows contain
    time followed by strain values.

    Parameters
    ----------
    path
        Path to the raw COMSOL text export.

    Returns
    -------
    header_lines
        COMSOL metadata/header lines.

    numeric_rows
        Numeric rows containing time and strain values.
    """

    if not path.exists():
        raise FileNotFoundError(f"COMSOL input file not found: {path}")

    header_lines: list[str] = []
    numeric_rows: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:

        for line_number, raw_line in enumerate(handle, start=1):

            line = raw_line.strip()

            if not line:
                continue

            # COMSOL metadata/header
            if line.startswith("%"):
                header_lines.append(line)
                continue

            tokens = line.split()

            # Every remaining line should be numeric.
            try:
                float(tokens[0])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Unexpected non-comment line {line_number} in "
                    f"{path}:\n{raw_line.rstrip()}"
                ) from exc

            try:
                row = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric data on line "
                    f"{line_number} in {path}."
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


# ---------------------------------------------------------------------
# COMSOL header parser
# ---------------------------------------------------------------------

def parse_channel_header(
    header_lines: list[str],
) -> list[tuple[str, tuple[float, float, float]]]:
    """
    Extract strain-component and station-coordinate information from
    the COMSOL header.

    Returns
    -------
    channels
        Ordered list containing:

            (component, (x, y, z))

        for every strain channel in the COMSOL export.

    Notes
    -----
    Channel ordering is preserved exactly as it appears in the raw
    COMSOL file.
    """

    channels: list[
        tuple[str, tuple[float, float, float]]
    ] = []

    for line in header_lines:

        for match in CHANNEL_PATTERN.finditer(line):

            component = match.group(1)
            coordinate_text = match.group(2)

            coordinate_tokens = [
                value.strip()
                for value in coordinate_text.split(",")
            ]

            if len(coordinate_tokens) != 3:
                raise ValueError(
                    "Could not parse station coordinates from "
                    f"COMSOL header: {coordinate_text!r}"
                )

            try:
                x, y, z = (
                    float(value)
                    for value in coordinate_tokens
                )
            except ValueError as exc:
                raise ValueError(
                    "Invalid station coordinates in COMSOL "
                    f"header: {coordinate_text!r}"
                ) from exc

            channels.append(
                (component, (x, y, z))
            )

    if not channels:
        raise ValueError(
            "No supported strain channels were found in the "
            "COMSOL header.\n"
            "Expected entries similar to:\n"
            "'solid2.eXX*1E9 (1), Point: (x, y, z)'"
        )

    return channels


# ---------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------

def build_station_table(
    channels: list[
        tuple[str, tuple[float, float, float]]
    ],
) -> tuple[
    pd.DataFrame,
    dict[tuple[float, float, float], str],
]:
    """
    Construct canonical station metadata.

    Station IDs are assigned according to the first appearance of each
    unique coordinate in the COMSOL header.

    For example:

        station,x,y,z
        S01,250,433,-40
        S02,250,-433,-40

    Only coordinates explicitly contained in the COMSOL export are
    stored. Derived quantities such as radial distance are intentionally
    not added to the canonical metadata.

    Returns
    -------
    stations
        DataFrame containing station, x, y, and z.

    coordinate_to_station
        Mapping between COMSOL coordinates and canonical station IDs.
    """

    coordinate_to_station: dict[
        tuple[float, float, float], str
    ] = {}

    records: list[dict[str, float | str]] = []

    for _, coordinates in channels:

        if coordinates in coordinate_to_station:
            continue

        station_id = (
            f"S{len(coordinate_to_station) + 1:02d}"
        )

        coordinate_to_station[coordinates] = station_id

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
        columns=["station", "x", "y", "z"],
    )

    return stations, coordinate_to_station


# ---------------------------------------------------------------------
# Canonical strain dataset
# ---------------------------------------------------------------------

def build_clean_strain_dataframe(
    numeric_rows: list[list[float]],
    channels: list[
        tuple[str, tuple[float, float, float]]
    ],
    coordinate_to_station: dict[
        tuple[float, float, float], str
    ],
) -> pd.DataFrame:
    """
    Convert raw COMSOL numeric data into the canonical strain format.

    Output ordering is component-major, then station-major:

        time_s

        eXX_S01 ... eXX_S08
        eYY_S01 ... eYY_S08
        eZZ_S01 ... eZZ_S08
        eXY_S01 ... eXY_S08

    This ordering provides a stable interface between COMSOL data and
    the analytical forward/inversion workflow.
    """

    # -------------------------------------------------------------
    # Validate numeric row lengths
    # -------------------------------------------------------------

    n_data_columns = len(numeric_rows[0])

    for row_number, row in enumerate(
        numeric_rows,
        start=1,
    ):

        if len(row) != n_data_columns:

            raise ValueError(
                "Inconsistent number of columns in numeric data. "
                f"Row {row_number} has {len(row)} columns; "
                f"expected {n_data_columns}."
            )

    expected_columns = 1 + len(channels)

    if n_data_columns != expected_columns:

        raise ValueError(
            "Mismatch between COMSOL header and numeric data.\n"
            f"Header describes {len(channels)} strain channels "
            f"plus time ({expected_columns} total columns), "
            f"but numeric rows contain {n_data_columns} columns."
        )

    # -------------------------------------------------------------
    # Convert to numeric table
    # -------------------------------------------------------------

    data = np.asarray(
        numeric_rows,
        dtype=float,
    )

    raw = pd.DataFrame(data)

    time = raw.iloc[:, 0].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------
    # Map raw COMSOL columns to component/station pairs
    # -------------------------------------------------------------

    channel_map: dict[
        tuple[str, str], int
    ] = {}

    station_ids: list[str] = []

    for raw_index, (
        component,
        coordinates,
    ) in enumerate(
        channels,
        start=1,
    ):

        station = coordinate_to_station[coordinates]

        station_ids.append(station)

        key = (component, station)

        if key in channel_map:

            raise ValueError(
                "Duplicate COMSOL strain channel detected: "
                f"{component} at {station}."
            )

        channel_map[key] = raw_index

    # Preserve station order determined from the raw header.
    unique_station_ids = list(
        dict.fromkeys(station_ids)
    )

    # -------------------------------------------------------------
    # Build canonical output
    # -------------------------------------------------------------

    ordered_columns: dict[
        str, np.ndarray
    ] = {
        "time_s": time
    }

    for component in SUPPORTED_COMPONENTS:

        for station in unique_station_ids:

            key = (component, station)

            if key not in channel_map:

                raise ValueError(
                    f"Missing {component} channel "
                    f"for station {station}."
                )

            raw_index = channel_map[key]

            ordered_columns[
                f"{component}_{station}"
            ] = raw.iloc[
                :, raw_index
            ].to_numpy(dtype=float)

    return pd.DataFrame(
        ordered_columns
    )


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------

def validate_clean_dataset(
    df: pd.DataFrame,
    stations: pd.DataFrame,
) -> None:
    """
    Validate the processed strain dataset and station metadata.

    Validation includes:

    - presence of time_s
    - finite time values
    - finite strain values
    - expected number of strain channels
    - unique station IDs
    - complete x/y/z station coordinates
    - duplicate timestamp reporting
    """

    # -------------------------------------------------------------
    # Time
    # -------------------------------------------------------------

    if "time_s" not in df.columns:
        raise ValueError(
            "Cleaned dataset must contain 'time_s'."
        )

    time = df["time_s"].to_numpy(
        dtype=float
    )

    if not np.isfinite(time).all():
        raise ValueError(
            "Time column contains NaN or infinite values."
        )

    # -------------------------------------------------------------
    # Duplicate timestamps
    # -------------------------------------------------------------

    duplicate_count = int(
        df["time_s"].duplicated().sum()
    )

    if duplicate_count:

        print(
            f"[WARNING] Found {duplicate_count} duplicate "
            "time rows."
        )

        print(
            "[WARNING] Duplicate rows are preserved unless "
            "--deduplicate-time is specified."
        )

    # -------------------------------------------------------------
    # Station metadata
    # -------------------------------------------------------------

    if len(stations) == 0:
        raise ValueError(
            "No station metadata were generated."
        )

    if stations["station"].duplicated().any():
        raise ValueError(
            "Station IDs are not unique."
        )

    required_station_columns = {
        "station",
        "x",
        "y",
        "z",
    }

    missing_station_columns = (
        required_station_columns
        - set(stations.columns)
    )

    if missing_station_columns:

        raise ValueError(
            "Station metadata are missing required columns: "
            f"{sorted(missing_station_columns)}"
        )

    if stations[
        ["x", "y", "z"]
    ].isna().any().any():

        raise ValueError(
            "Station metadata contain missing coordinates."
        )

    # -------------------------------------------------------------
    # Strain data
    # -------------------------------------------------------------

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
    ].to_numpy(dtype=float)

    if not np.isfinite(
        strain_values
    ).all():

        raise ValueError(
            "Strain data contain NaN or infinite values."
        )

    expected_strain_columns = (
        len(SUPPORTED_COMPONENTS)
        * len(stations)
    )

    if len(strain_columns) != expected_strain_columns:

        raise ValueError(
            f"Expected {expected_strain_columns} strain "
            f"columns "
            f"({len(SUPPORTED_COMPONENTS)} components x "
            f"{len(stations)} stations), "
            f"found {len(strain_columns)}."
        )


# ---------------------------------------------------------------------
# Duplicate-time handling
# ---------------------------------------------------------------------

def deduplicate_time_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse duplicate timestamps by averaging strain values.

    This operation is optional. Duplicate timestamps are preserved by
    default because preprocessing should not silently discard or alter
    observations from the raw COMSOL export.
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


# ---------------------------------------------------------------------
# Main preprocessing workflow
# ---------------------------------------------------------------------

def process_comsol_dataset(
    input_file: Path,
    output_file: Path,
    stations_file: Path,
    deduplicate_time: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a raw COMSOL strain export.

    Parameters
    ----------
    input_file
        Raw COMSOL text export.

    output_file
        Destination for the canonical strain CSV.

    stations_file
        Destination for the station metadata CSV.

    deduplicate_time
        If True, duplicate timestamps are collapsed by averaging.

    Returns
    -------
    cleaned
        Canonical strain DataFrame.

    stations
        Station metadata DataFrame.
    """

    print(
        f"[INFO] Reading raw COMSOL file: "
        f"{input_file}"
    )

    # -------------------------------------------------------------
    # Read raw data
    # -------------------------------------------------------------

    header_lines, numeric_rows = (
        read_comsol_file(input_file)
    )

    print(
        f"[INFO] Parsed "
        f"{len(header_lines)} header/comment lines."
    )

    print(
        f"[INFO] Parsed "
        f"{len(numeric_rows)} numeric rows."
    )

    # -------------------------------------------------------------
    # Parse COMSOL channel metadata
    # -------------------------------------------------------------

    channels = parse_channel_header(
        header_lines
    )

    print(
        f"[INFO] Identified "
        f"{len(channels)} strain channels."
    )

    # -------------------------------------------------------------
    # Build station metadata
    # -------------------------------------------------------------

    stations, coordinate_to_station = (
        build_station_table(channels)
    )

    print(
        f"[INFO] Identified "
        f"{len(stations)} unique stations."
    )

    # -------------------------------------------------------------
    # Construct canonical dataset
    # -------------------------------------------------------------

    cleaned = build_clean_strain_dataframe(
        numeric_rows=numeric_rows,
        channels=channels,
        coordinate_to_station=coordinate_to_station,
    )

    # -------------------------------------------------------------
    # Validate before writing
    # -------------------------------------------------------------

    validate_clean_dataset(
        cleaned,
        stations,
    )

    # -------------------------------------------------------------
    # Optional duplicate-time processing
    # -------------------------------------------------------------

    if deduplicate_time:

        before = len(cleaned)

        cleaned = deduplicate_time_rows(
            cleaned
        )

        after = len(cleaned)

        print(
            f"[INFO] Deduplicated time rows: "
            f"{before} -> {after}."
        )

        # Validate the final output again.
        validate_clean_dataset(
            cleaned,
            stations,
        )

    # -------------------------------------------------------------
    # Create output directories
    # -------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stations_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save canonical outputs
    # -------------------------------------------------------------

    cleaned.to_csv(
        output_file,
        index=False,
    )

    stations.to_csv(
        stations_file,
        index=False,
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print()
    print("[INFO] COMSOL preprocessing complete.")
    print(
        f"[INFO] Clean strain dataset: "
        f"{output_file}"
    )
    print(
        f"[INFO] Station metadata:      "
        f"{stations_file}"
    )
    print(
        f"[INFO] Dataset rows:          "
        f"{len(cleaned)}"
    )
    print(
        f"[INFO] Strain channels:       "
        f"{len(cleaned.columns) - 1}"
    )
    print(
        f"[INFO] Number of stations:    "
        f"{len(stations)}"
    )

    return cleaned, stations


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw COMSOL strain text export into "
            "a canonical strain CSV and station metadata CSV."
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
            "Path for the station metadata CSV."
        ),
    )

    parser.add_argument(
        "--deduplicate-time",
        action="store_true",
        help=(
            "Average duplicate timestamps. "
            "By default duplicate rows are preserved "
            "and only reported."
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
        deduplicate_time=args.deduplicate_time,
    )


if __name__ == "__main__":
    main()