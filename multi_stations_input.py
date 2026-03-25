import numpy as np
import pandas as pd
import os
from scipy.stats import uniform

# ---------------------- MATERIAL ----------------------
nu = 0.25         # Poisson ratio [-]
alpha = 0.8       # Biot coefficient [-]

# ---------------------- FIXED GEOMETRY FROM LATEST SEARCH ----------------------
# Provisional best geometry from the shape inversion
a_fixed = 300
b_fixed = 5
c_fixed = 580
E = 0.2e10
theta_deg = 0.0
# ---------------------- INCLUSION (fixed center) ----------------------
x0_prime = -208.33333333333337
y0_prime = 83.33333333333326
h = 518.29  # From latest geometry inversion; should be close to the true depth of the inclusion center             

# ---------------------- STATIONS ----------------------
stations_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "AVANT_stations.csv"))
station_names = stations_df["station"].astype(str).str.strip().values
x_prime = stations_df["x_prime"].values
y_prime = stations_df["y_prime"].values
z = stations_df["depth"].values

# ---------------------- PRESSURE ----------------------
pmax = 9.75e6
tpeak = 393333.0
d = 0.4

obs_df = pd.read_csv("avant_cleaned_strain.csv")
time_vals = obs_df["time_s"].values

# ---------------------- NOISE SCALE ----------------------
# Fixed likelihood scale for the inversion; not sampled as a parameter.
# You can tune this later if needed.
sigma_noise = 2

# ---------------------- PRIORS ----------------------
# Main inversion parameters:
#   - E
#   - theta_deg
#
# Geometry is fixed here and should not be re-inverted in this stage.
priors = {
    "E": uniform(loc=0.5e10, scale=2.5e10),        # 0.5e10 to 3.0e10
    "theta_deg": uniform(loc=-90.0, scale=180.0),  # -90 to +90 degrees
}

def read_input():
    return {
        "nu": nu,
        "alpha": alpha,
        "a_fixed": a_fixed,
        "b_fixed": b_fixed,
        "c_fixed": c_fixed,
        "h": h,
        "x0_prime": x0_prime,
        "y0_prime": y0_prime,
        "x_prime": x_prime,
        "y_prime": y_prime,
        "z": z,
        "pmax": pmax,
        "tpeak": tpeak,
        "d": d,
        "time": time_vals,
        "sigma_noise": sigma_noise,
        "station_names": station_names,
        "priors": priors,
    }

get_params = read_input()