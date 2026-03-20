import numpy as np
import pandas as pd
import os
from scipy.stats import uniform

# ---------------------- MATERIAL ----------------------
nu = 0.25         # Poisson ratio [-]
alpha = 0.8       # Biot coefficient [-]

# ---------------------- INCLUSION (fixed center) ----------------------
x0_prime = -67.04
y0_prime = -129.72
h = 2500  # can fix or later invert

# ---------------------- STATIONS ----------------------
stations_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'AVANT_stations.csv'))
station_names = stations_df['station'].astype(str).str.strip().values
x_prime = stations_df['x_prime'].values
y_prime = stations_df['y_prime'].values
z = stations_df['depth'].values

# ---------------------- PRESSURE ----------------------
pmax = 10e6
tpeak = 8
d = 2
obs_df = pd.read_csv("avant_cleaned_strain.csv")
time_vals = obs_df["time_s"].values

# ---------------------- GEOMETRY RATIOS ----------------------
a0 = 1.0          # base scale
b0 = 25.0 / 175.0
c0 = 125.0 / 175.0
theta_deg_start = -17.2

# ---------------------- PRIORS ----------------------
priors = {
    "s": uniform(loc=50.0, scale=150.0),
    "E": uniform(loc=0.8e10, scale=0.4e10),
    "theta_deg": uniform(loc=theta_deg_start - 15.0, scale=30.0),
    "sigma": uniform(loc=1e-9, scale=1e-8),
}

def read_input():
    return {
        "nu": nu,
        "alpha": alpha,
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
        "a0": a0,
        "b0": b0,
        "c0": c0,
        "theta_deg_start": theta_deg_start,
        "station_names": station_names,
        "priors": priors
    }

get_params = read_input