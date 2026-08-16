"""
avant_sensitivity_config.py

Shared configuration and utilities for AVANT sensitivity analyses.
This file is the SINGLE SOURCE OF TRUTH for:

- Parameter bounds
- Parameter names
- Strain component names
- Station classification (near / mid / far)
- Loading of fixed model inputs

Used by:
- avant_sensitivity_lhs.py
- avant_sensitivity_sobol.py
"""

import numpy as np
import pandas as pd

# Your forward model
from forward_model_multi_station import forward_model_multi_station
import multi_stations_input

# ==============================================================
# 1) PARAMETER SPACE (single source of truth)
# ==============================================================

PARAM_BOUNDS = {
    "a": (50.0, 300.0),
    "b": (10.0, 100.0),
    "c": (50.0, 300.0),
    "pmax": (1e6, 2e7),
    "alpha": (0.2, 1.0),
    "E": (1e9, 1e11),
    "nu": (0.2, 0.35),
    "h": (500.0, 5000.0),
    "theta_deg": (-30.0, 30.0),
    "x0_prime": (-500.0, 500.0),
    "y0_prime": (-500.0, 500.0),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

# ==============================================================
# 2) STRAIN COMPONENT NAMES (for parsing DataFrames)
# ==============================================================

COMP_BASES = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

# ==============================================================
# 3) LOAD FIXED MODEL INPUTS (AVANT)
# ==============================================================

def load_fixed_args():
    """
    Read AVANT inputs from multi_stations_input and return
    a dictionary of fixed arguments for the forward model.
    """
    params = multi_stations_input.read_input()

    fixed_args = {
        "pmax": params["pmax"],
        "tpeak": params["tpeak"],
        "d": params["d"],
        "time": np.asarray(params["time"]),
        "x_prime": np.asarray(params["x_prime"]),
        "y_prime": np.asarray(params["y_prime"]),
        "x0_prime": params["x0_prime"],
        "y0_prime": params["y0_prime"],
        "z": np.asarray(params["z"]),
        "a": params["a"],
        "b": params["b"],
        "c": params["c"],
        "nu": params["nu"],
        "h": params["h"],
        "E": params["E"],
        "theta_deg": params["theta_deg"],
        "alpha": params.get("alpha", None),
        "station_names": params.get("station_names", None),
    }

    return fixed_args

# ==============================================================
# 4) STATION CLASSIFICATION (NEAR / MID / FAR)
# ==============================================================

def classify_stations(x_prime, y_prime, x0_prime, y0_prime):
    """
    Classify stations into near / mid / far based on distance
    from the source (x0_prime, y0_prime).

    Returns:
        near_idx, mid_idx, far_idx  (arrays of station indices)
    """
    x = np.asarray(x_prime)
    y = np.asarray(y_prime)

    dist = np.sqrt((x - x0_prime)**2 + (y - y0_prime)**2)

    p33, p66 = np.percentile(dist, [33, 66])

    near = np.where(dist < p33)[0]
    mid = np.where((dist >= p33) & (dist < p66))[0]
    far = np.where(dist >= p66)[0]

    return near, mid, far

# ==============================================================
# 5) UTILITIES FOR EXTRACTING STRAIN FROM MODEL OUTPUT
# ==============================================================

def extract_component_matrix(df, comp_base):
    """
    Given model output DataFrame, return (Nt x Ns) array for a strain component.
    """
    comp_cols = [c for c in df.columns if c.startswith(comp_base)]
    return df[comp_cols].values

def peak_abs_max(arr):
    """Maximum absolute value over time and stations."""
    return float(np.max(np.abs(arr)))

def rms_avg(arr):
    """Root-mean-square averaged over time and stations."""
    return float(np.sqrt(np.mean(arr**2)))

# ==============================================================
# 6) WRAPPER THAT CALLS YOUR EXISTING FORWARD MODEL
# ==============================================================

def run_forward_model_from_params(param_row, fixed_args):
    """
    Call YOUR existing forward_model_multi_station using:
      - sampled parameters (param_row)
      - fixed_args from AVANT input

    param_row: dict or pandas Series with keys matching PARAM_NAMES
    """

    df = forward_model_multi_station(
        pmax=float(param_row.get("pmax", fixed_args["pmax"])),
        tpeak=fixed_args["tpeak"],
        d=fixed_args["d"],
        time=fixed_args["time"],
        x_prime=fixed_args["x_prime"],
        y_prime=fixed_args["y_prime"],
        x0_prime=float(param_row.get("x0_prime", fixed_args["x0_prime"])),
        y0_prime=float(param_row.get("y0_prime", fixed_args["y0_prime"])),
        z=fixed_args["z"],
        a=float(param_row.get("a", fixed_args["a"])),
        b=float(param_row.get("b", fixed_args["b"])),
        c=float(param_row.get("c", fixed_args["c"])),
        nu=float(param_row.get("nu", fixed_args["nu"])),
        h=float(param_row.get("h", fixed_args["h"])),
        E=float(param_row.get("E", fixed_args["E"])),
        theta_deg=float(param_row.get("theta_deg", fixed_args["theta_deg"])),
        alpha=float(param_row.get("alpha", fixed_args["alpha"])),
        station_names=fixed_args.get("station_names", None),
    )

    return df
