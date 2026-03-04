"""
Clean input-parameter module for multi-station strain forward modeling.
All units assumed consistent with the rest of the model (e.g., meters, Pascals).
"""

import numpy as np
import pandas as pd
import os

# ============================================================
# ------------------- MATERIAL / INCLUSION PROPERTIES --------
# ============================================================

# Material / poroelastic parameters
p       = 10e6       # Pressure amplitude [Pa]
nu      = 0.25       # Poisson ratio [-]
E       = 1.0e10     # Young's modulus [Pa]
alpha   = 0.8        # Biot coefficient [-]
h       = 2500       # Depth to top of inclusion [m]

# ============================================================
# ---------------------- INCLUSION GEOMETRY ------------------
# ============================================================

x0_prime = -67.04      # Inclusion center x' [m]
y0_prime = -129.72     # Inclusion center y' [m]

a = 175               # Inclusion semi-axis a [m]
b = 25                # Inclusion semi-axis b [m]
c = 125               # Inclusion semi-axis c [m]

theta_deg = -17.2     # Rotation angle [deg]

# ============================================================
# ------------------- OBSERVATION STATIONS --------------------
# ============================================================

# Read AVANT station file
stations_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'AVANT_stations.csv'))

# --- CLEAN STATION NAMES ---
station_names = stations_df['station'].astype(str).str.strip().values
x_prime = stations_df['x_prime'].values
y_prime = stations_df['y_prime'].values
z = stations_df['depth'].values

# Verification (will raise if lengths mismatch)
if len(x_prime) != len(y_prime):
    raise ValueError("x_prime and y_prime must have same length (number of stations).")

# ============================================================
# ---------------------- PRESSURE TIME SERIES -----------------
# ============================================================

pmax      = 10e6
tpeak     = 8
d         = 2
time_vals = np.linspace(0, 12, 20)   #  time steps

# ============================================================
# ----------------------- PARAMETER EXPORT --------------------
# ============================================================

def read_input():
    """Return all parameters in a clean dict for the forward model."""
    return {
        # Material properties
        "p": p,
        "nu": nu,
        "E": E,
        "alpha": alpha,
        "h": h,
        "z": z,
        "station_names": station_names,

        # Inclusion geometry
        "x0_prime": x0_prime,
        "y0_prime": y0_prime,
        "a": a,
        "b": b,
        "c": c,
        "theta_deg": theta_deg,

        # Multi-station coordinates
        "x_prime": x_prime,
        "y_prime": y_prime,

        # Pressure time series
        "pmax": pmax,
        "tpeak": tpeak,
        "d": d,
        "time": time_vals,
    }

# Alias required by the current pipeline
get_params = read_input
