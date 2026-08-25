"""
Shared input configuration for the Avant 1-Darcy COMSOL experiment.

This module provides the fixed/baseline experiment inputs used by the
analytical forward model and by downstream fitting, plotting, sensitivity,
and inversion workflows.

Observed station coordinates and strain data are read from the canonical
1-Darcy dataset produced by:

    scripts/preprocessing/clean_dataset.py

Dataset structure
-----------------
datasets/
    comsol/
        1darcy/
            raw/
                k1D.txt
            processed/
                strain.csv
            metadata/
                stations.csv

Important
---------
This module provides experiment inputs only.

It does not define inversion parameter distributions, likelihoods,
optimization strategies, or posterior analysis. Those belong in the
corresponding fitting/inversion scripts.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# REPOSITORY / DATASET PATHS
# =====================================================================

# Current file:
#   repo/src/avant_model/data/multi_stations_input.py
#
# parents[0] -> data
# parents[1] -> avant_model
# parents[2] -> src
# parents[3] -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
)

STATION_FILE = (
    DATASET_DIR
    / "metadata"
    / "stations.csv"
)

OBSERVED_FILE = (
    DATASET_DIR
    / "processed"
    / "strain.csv"
)


# =====================================================================
# MATERIAL PARAMETERS
# =====================================================================

# Poisson's ratio.
# Confirmed for the new 1-Darcy COMSOL model.
nu = 0.35

# Biot coefficient.
#
# Retained from the established analytical-model configuration.
# Verify against the final COMSOL setup before treating it as an
# independently calibrated quantity.
alpha = 0.8


# =====================================================================
# FIXED / BASELINE INCLUSION GEOMETRY
# =====================================================================

# Best-fit analytical inclusion geometry [m].
#
# These values are restored to the current best-fit solution obtained
# from the unconstrained 1-Darcy analytical fit.
a_fixed = 261.744313
b_fixed = 452.090306
c_fixed = 3.125


# =====================================================================
# ELASTIC PARAMETERS
# =====================================================================

# Young's modulus [Pa].
# Confirmed for the new 1-Darcy COMSOL dataset.
E = 8.0e9

# Best-fit analytical orientation [degrees].
theta_deg = 14.448960


# =====================================================================
# INCLUSION LOCATION
# =====================================================================

# Best-fit analytical horizontal inclusion center [m].
x0_prime = 42.580217
y0_prime = 120.093461

# Best-fit analytical depth parameter [m].
h = 121.677004


# =====================================================================
# STATION METADATA
# =====================================================================

if not STATION_FILE.exists():
    raise FileNotFoundError(
        "Station metadata file was not found:\n"
        f"    {STATION_FILE}\n\n"
        "Run scripts/preprocessing/clean_dataset.py first."
    )

stations_df = pd.read_csv(STATION_FILE)

required_station_columns = {
    "station",
    "x",
    "y",
    "z",
}

missing_station_columns = (
    required_station_columns
    - set(stations_df.columns)
)

if missing_station_columns:
    raise RuntimeError(
        "Station metadata are missing required columns: "
        f"{sorted(missing_station_columns)}"
    )

if stations_df.empty:
    raise RuntimeError(
        f"No station records were found in {STATION_FILE}."
    )

if stations_df["station"].duplicated().any():
    raise RuntimeError(
        f"Duplicate station names found in {STATION_FILE}."
    )

if stations_df[["x", "y", "z"]].isna().any().any():
    raise RuntimeError(
        f"Station coordinates contain missing values in {STATION_FILE}."
    )

# Canonical station identifiers.
station_names = (
    stations_df["station"]
    .astype(str)
    .str.strip()
    .values
)

# Preserve the established forward-model interface:
# x_prime, y_prime, and z correspond to the coordinates supplied by
# the station metadata.
x_prime = (
    stations_df["x"]
    .to_numpy(dtype=float)
)

y_prime = (
    stations_df["y"]
    .to_numpy(dtype=float)
)

z = (
    stations_df["z"]
    .to_numpy(dtype=float)
)


# =====================================================================
# OBSERVED 1-DARCY STRAIN DATA
# =====================================================================

if not OBSERVED_FILE.exists():
    raise FileNotFoundError(
        "Processed strain dataset was not found:\n"
        f"    {OBSERVED_FILE}\n\n"
        "Run scripts/preprocessing/clean_dataset.py first."
    )

obs_df = pd.read_csv(OBSERVED_FILE)

if "time_s" not in obs_df.columns:
    raise RuntimeError(
        "Processed strain dataset must contain a 'time_s' column:\n"
        f"    {OBSERVED_FILE}"
    )

# Canonical observed time vector [s].
time_vals = (
    obs_df["time_s"]
    .to_numpy(dtype=float)
)

if time_vals.size == 0:
    raise RuntimeError(
        f"No time samples were found in {OBSERVED_FILE}."
    )

if not np.isfinite(time_vals).all():
    raise RuntimeError(
        f"Observed time values contain NaN or infinite values in "
        f"{OBSERVED_FILE}."
    )

# The preprocessing stage has already removed exact duplicate timestamps.
# Keep this as a defensive check so downstream scripts never silently
# receive a duplicated time vector.
if pd.Series(time_vals).duplicated().any():
    raise RuntimeError(
        "Processed strain dataset contains duplicate timestamps:\n"
        f"    {OBSERVED_FILE}"
    )


# =====================================================================
# OBSERVED STRAIN CHANNEL VALIDATION
# =====================================================================

COMPONENTS = (
    "eXX",
    "eYY",
    "eZZ",
    "eXY",
)

expected_observed_columns = [
    f"{component}_{station}"
    for component in COMPONENTS
    for station in station_names
]

missing_observed_columns = [
    column
    for column in expected_observed_columns
    if column not in obs_df.columns
]

if missing_observed_columns:
    raise RuntimeError(
        "Processed strain dataset is missing expected "
        "station/component columns:\n"
        + "\n".join(
            f"    {column}"
            for column in missing_observed_columns
        )
    )

# Confirm that the expected number of channels is present.
expected_channel_count = (
    len(COMPONENTS) * len(station_names)
)

actual_channel_count = len(obs_df.columns) - 1

if actual_channel_count != expected_channel_count:
    raise RuntimeError(
        "Unexpected number of strain channels.\n"
        f"Expected: {expected_channel_count}\n"
        f"Found:    {actual_channel_count}\n"
        f"Dataset:  {OBSERVED_FILE}"
    )

# Confirm that strain values are finite.
observed_strain = obs_df[
    expected_observed_columns
].to_numpy(dtype=float)

if not np.isfinite(observed_strain).all():
    raise RuntimeError(
        f"Observed strain data contain NaN or infinite values in "
        f"{OBSERVED_FILE}."
    )


# =====================================================================
# PRESSURE HISTORY
# =====================================================================

# Baseline pressure-history parameters for the current best-fit
# analytical configuration.
#
# The separately introduced Larry WW29 pressure file remains available
# for the field-constrained diagnostics; this shared input file is restored
# to the established analytical baseline used by the fitting/OED workflow.

pmax = 0.99931e6      # Peak pressure [Pa]
tpeak = 350000.0    # Time of peak pressure [s]
d = 0.4             # Post-peak decay parameter


# =====================================================================
# NOISE / ERROR SCALE
# =====================================================================

# Retained for compatibility with downstream analyses that expect a
# sigma_noise input. The final observation/model error treatment will
# be established separately in the fitting/inversion workflow.
sigma_noise = 2.0


# =====================================================================
# SHARED INPUT INTERFACE
# =====================================================================

def read_input() -> dict:
    """
    Return the shared 1-Darcy experiment configuration.

    This function provides model/data inputs only. Parameter priors,
    likelihood functions, optimization settings, and posterior
    distributions are defined by the downstream fitting/inversion
    workflows.
    """

    return {
        # -------------------------------------------------------------
        # Dataset paths
        # -------------------------------------------------------------
        "dataset_dir": DATASET_DIR,
        "station_file": STATION_FILE,
        "observed_file": OBSERVED_FILE,

        # -------------------------------------------------------------
        # Material
        # -------------------------------------------------------------
        "nu": nu,
        "alpha": alpha,

        # -------------------------------------------------------------
        # Geometry
        # -------------------------------------------------------------
        "a_fixed": a_fixed,
        "b_fixed": b_fixed,
        "c_fixed": c_fixed,

        # -------------------------------------------------------------
        # Inclusion center / depth
        # -------------------------------------------------------------
        "x0_prime": x0_prime,
        "y0_prime": y0_prime,
        "h": h,

        # -------------------------------------------------------------
        # Station coordinates
        # -------------------------------------------------------------
        "x_prime": x_prime,
        "y_prime": y_prime,
        "z": z,

        # -------------------------------------------------------------
        # Pressure history
        # -------------------------------------------------------------
        "pmax": pmax,
        "tpeak": tpeak,
        "d": d,
        "time": time_vals,

        # -------------------------------------------------------------
        # Elastic / orientation parameters
        # -------------------------------------------------------------
        "E": E,
        "theta_deg": theta_deg,

        # -------------------------------------------------------------
        # Error scale
        # -------------------------------------------------------------
        "sigma_noise": sigma_noise,

        # -------------------------------------------------------------
        # Station information
        # -------------------------------------------------------------
        "station_names": station_names,
    }


# Compatibility with existing code that imports get_params.
get_params = read_input()