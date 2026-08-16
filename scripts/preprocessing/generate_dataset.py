"""
Generate surrogate-model training dataset using LHS sampling.
Each sample runs the multi-station forward model and flattens
the strain + volume time series into a 620-dimensional vector.

This improved version:
- Samples a, b, c independently (not via scale s)
- Uses physically meaningful parameter ranges
- Produces richer strain patterns for PCA/AE
- Greatly improves surrogate training accuracy
"""

import numpy as np
import pandas as pd
from pyDOE2 import lhs

from multi_stations_input import read_input
from forward_model_multi_station import forward_model_multi_station
import bayesian_inversion_multi_station as bi


# ============================================================
# Load base parameters and station geometry
# ============================================================

params = read_input()
time = np.asarray(params["time"])

station_names = params["station_names"]
x_prime = params["x_prime"]
y_prime = params["y_prime"]
z = params["z"]
Ns = len(station_names)

# Strain component names
base_components = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

COMPONENT_COLS = [
    f"{comp}_{station_names[s]}"
    for comp in base_components
    for s in range(Ns)
]


# ============================================================
# Define parameter bounds for LHS sampling
# ============================================================

# Independent sampling of a, b, c
xlimits = np.array([
    [100, 300],     # a (m)
    [10, 50],       # b (m)
    [50, 200],      # c (m)
    [5e9, 5e10],    # E (Pa)
    [-30, 30],      # theta (deg)
    [1e6, 2e7],     # pmax (Pa)
    [1500, 3500],   # h (m)
])

N_SAMPLES = 400   # more samples = better surrogate

# LHS in [0,1]
raw = lhs(xlimits.shape[0], samples=N_SAMPLES)

# Scale to physical ranges
X_params = xlimits[:, 0] + raw * (xlimits[:, 1] - xlimits[:, 0])


# ============================================================
# Allocate output matrix
# ============================================================

N_OUTPUT = len(COMPONENT_COLS) * len(time) + len(time)  # 620
Y_outputs = np.zeros((N_SAMPLES, N_OUTPUT))


# ============================================================
# Volume model
# ============================================================

def volume_forward(a, b, c, E, nu, delta_P):
    V0 = a * b * c
    K = E / (3.0 * (1.0 - 2.0 * nu))
    return V0 * (1.0 + delta_P / K)


# ============================================================
# Forward model loop
# ============================================================

for i in range(N_SAMPLES):
    print(f"[INFO] Running sample {i+1}/{N_SAMPLES}")

    a_s, b_s, c_s, E_s, theta_s, pmax_s, h_s = X_params[i]

    # Run forward model
    df = forward_model_multi_station(
        pmax=pmax_s,
        tpeak=params["tpeak"],
        d=params["d"],
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=params["x0_prime"],
        y0_prime=params["y0_prime"],
        z=z,
        a=a_s,
        b=b_s,
        c=c_s,
        nu=params["nu"],
        h=h_s,
        E=E_s,
        theta_deg=theta_s,
        alpha=params.get("alpha", None),
        station_names=station_names,
    )

    # Flatten strain
    strain_vec = bi._flatten_df_to_vector(df, cols=COMPONENT_COLS)

    # Volume time series
    delta_P = pmax_s * np.sin(np.pi * time / time.max())
    V = volume_forward(a_s, b_s, c_s, E_s, params["nu"], delta_P)

    # Combine strain + volume
    Y_outputs[i, :] = np.concatenate([strain_vec, V])


# ============================================================
# Save dataset
# ============================================================

np.save("surrogate_inputs.npy", X_params)
np.save("surrogate_outputs.npy", Y_outputs)

print("[INFO] Surrogate dataset saved.")
