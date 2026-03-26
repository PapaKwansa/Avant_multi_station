import numpy as np
import pandas as pd
import os
from scipy.stats import uniform

# ---------------------- MATERIAL ----------------------
nu = 0.25
alpha = 0.8

# ---------------------- FIXED GEOMETRY (semi-axes) ----------------------
a_fixed = 150.0   # short axis / 2
b_fixed = 250.0   # thickness / 2
c_fixed = 150.0   # long axis / 2

# ---------------------- ELASTIC PARAMETERS ----------------------
# Initial guess (inversion will explore range)
E = 0.2e10
theta_deg = -15   # COMSOL uses no rotation

# ---------------------- INCLUSION CENTER ----------------------
x0_prime = 0.0
y0_prime = 0.0
h = 518.29   # depth of inclusion top

# ---------------------- STATION COORDINATES ----------------------
stations_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "AVANT_stations.csv"))
station_names = stations_df["station"].astype(str).str.strip().values
x_prime = stations_df["x_prime"].values.astype(float)
y_prime = stations_df["y_prime"].values.astype(float)
z = stations_df["depth"].values.astype(float)

# ---------------------- PRESSURE HISTORY ----------------------
pmax = 0.4e6
tpeak = 393333.0
d = 0.4

# Time array from observed data
obs_df = pd.read_csv("avant_cleaned_strain.csv")
time_vals = obs_df["time_s"].values.astype(float)

# ---------------------- NOISE ----------------------
sigma_noise = 2

# ---------------------- PRIORS ----------------------
priors = {
    "E": uniform(loc=0.5e10, scale=2.5e10),
    "theta_deg": uniform(loc=-90.0, scale=180.0),
}

# ---------------------- READ INPUT ----------------------
def read_input():
    return {
        # Material
        "nu": nu,
        "alpha": alpha,

        # Geometry
        "a_fixed": a_fixed,
        "b_fixed": b_fixed,
        "c_fixed": c_fixed,

        # Inclusion center
        "x0_prime": x0_prime,
        "y0_prime": y0_prime,
        "h": h,

        # Station coordinates
        "x_prime": x_prime,
        "y_prime": y_prime,
        "z": z,

        # Pressure history
        "pmax": pmax,
        "tpeak": tpeak,
        "d": d,
        "time": time_vals,

        # Elastic parameters
        "E": E,
        "theta_deg": theta_deg,

        # Noise
        "sigma_noise": sigma_noise,

        # Names + priors
        "station_names": station_names,
        "priors": priors,
    }

get_params = read_input()
