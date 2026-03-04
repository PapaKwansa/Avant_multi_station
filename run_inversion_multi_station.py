"""
Run PyDREAM for multi-station strain + volume inversion.
Geometry re-parameterized using a single scale parameter s.
AVANT stations used consistently (NO FS fallback).
"""

import os
import numpy as np
import multiprocessing
import time as _time
import pandas as pd

from scipy.stats import uniform, norm
from pydream.core import run_dream
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin

import multi_stations_input
from forward_model_multi_station import forward_model_multi_station
import bayesian_inversion_multi_station as bi

# ============================================================
# Environment
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

np.random.seed(42)

# ============================================================
# Read input + AVANT stations (SINGLE SOURCE OF TRUTH)
# ============================================================

params = multi_stations_input.read_input()
time = np.asarray(params["time"])

RESULTS_DIR = os.path.join(os.path.expanduser("~"), "Avant_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")

stations_df = pd.read_csv(STATION_FILE)

# Required columns check
required_cols = ["station", "x_prime", "y_prime", "depth"]
missing = set(required_cols) - set(stations_df.columns)
if missing:
    raise RuntimeError(f"Missing columns in AVANT_stations.csv: {missing}")

# Clean station names to avoid hidden spaces
station_names = stations_df["station"].astype(str).str.strip().values
x_prime = stations_df["x_prime"].values
y_prime = stations_df["y_prime"].values
z = stations_df["depth"].values

Ns = len(station_names)
print(f"[INFO] Using {Ns} AVANT stations: {station_names}")

# ============================================================
# Geometry shape ratios (FIXED)
# ============================================================

a0 = 1.0
b0 = 25.0 / 175.0
c0 = 125.0 / 175.0

# ============================================================
# Base strain components
# ============================================================

base_components = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

# ============================================================
# Create synthetic strain dataset (AVANT-aware)
# ============================================================

df_clean = forward_model_multi_station(
    pmax=params["pmax"],
    tpeak=params["tpeak"],
    d=params["d"],
    time=time,
    x_prime=x_prime,
    y_prime=y_prime,
    x0_prime=params["x0_prime"],
    y0_prime=params["y0_prime"],
    z=z,
    a=params["a"],
    b=params["b"],
    c=params["c"],
    nu=params["nu"],
    h=params["h"],
    E=params["E"],
    theta_deg=params["theta_deg"],
    alpha=params.get("alpha", None),
    station_names=station_names,
)

print(df_clean.columns.tolist())

# ============================================================
# Component columns (AVANT, NOT FS)
# ============================================================

COMPONENT_COLS = [
    f"{comp}_{station_names[s].strip()}"
    for comp in base_components
    for s in range(Ns)
]

# Safety check: must come AFTER df_clean is created
missing_cols = [c for c in COMPONENT_COLS if c not in df_clean.columns]
if missing_cols:
    print("[DEBUG] df_clean columns:", df_clean.columns.tolist())
    print("[DEBUG] COMPONENT_COLS expected:", COMPONENT_COLS)
    raise RuntimeError(
        "Missing columns in forward output:\n" + "\n".join(missing_cols)
    )

print("[INFO] Component columns verified (AVANT)")

# ============================================================
# Flatten synthetic strain data
# ============================================================

clean_vector = bi._flatten_df_to_vector(df_clean, cols=COMPONENT_COLS)
sigma_noise = 0.05 * np.std(clean_vector)

observed_vector = clean_vector + np.random.normal(
    0.0, sigma_noise, size=clean_vector.size
)

observed_df = pd.DataFrame(
    observed_vector.reshape(len(COMPONENT_COLS), len(time)).T,
    columns=COMPONENT_COLS,
)

# ============================================================
# Volume forward model
# ============================================================

def bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))

def volume_forward(a, b, c, E, nu, delta_P):
    V0 = a * b * c
    K = bulk_modulus(E, nu)
    return V0 * (1.0 + delta_P / K)

# ============================================================
# Synthetic volume data
# ============================================================

delta_P = params["pmax"] * np.sin(np.pi * time / time.max())
V_true = volume_forward(params["a"], params["b"], params["c"],
                        params["E"], params["nu"], delta_P)

sigma_volume = 0.05 * np.std(V_true)
V_obs = V_true + np.random.normal(0.0, sigma_volume, size=V_true.size)
observed_df["Volume"] = V_obs

out_csv = os.path.join(RESULTS_DIR, "strain_volume_dataset.csv")
observed_df.to_csv(out_csv, index=False)
print(f"[INFO] Synthetic strain + volume dataset saved to {out_csv}")

# ============================================================
# Likelihood wrapper
# ============================================================

def likelihood_wrapper(params_in):
    s, E_s, theta_deg_s, sigma_s = np.ravel(params_in)
    a_s = s * a0
    b_s = s * b0
    c_s = s * c0

    return bi.likelihood(
        params=[a_s, b_s, c_s, E_s, theta_deg_s, sigma_s],
        observed_vector=observed_vector,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=params["x0_prime"],
        y0_prime=params["y0_prime"],
        z=z,
        nu=params["nu"],
        pmax=params["pmax"],
        tpeak=params["tpeak"],
        d=params["d"],
        h=params["h"],
        alpha=params.get("alpha", None),
        component_cols=COMPONENT_COLS,
        volume_obs=V_obs,
        delta_P=delta_P,
        sigma_volume=sigma_volume,
        volume_weight=1.0,
        station_names=station_names,   # <-- key addition
    )

# ============================================================
# Priors
# ============================================================

param_priors = [
    SampledParam(uniform, loc=100.0, scale=150.0),  # s
    SampledParam(uniform, loc=0.8 * params["E"], scale=0.4 * params["E"]),
    SampledParam(uniform, loc=params["theta_deg"] - 15.0, scale=30.0),
    SampledParam(norm, loc=max(1e-8, sigma_noise),
                 scale=max(1e-8, 0.5 * sigma_noise)),
]

# ============================================================
# Run control
# ============================================================

MAX_ITER = 50000
BATCH_SIZE = 20000
NCHAINS = 4
R_HAT_THRESH = 1.1
MODEL_NAME = "pydream_multi_station_reparam_volume_AVANT"

MP_CTX = multiprocessing.get_context("spawn")

# ============================================================
# Normalize chains
# ============================================================

def normalize_sampled_params(sampled_params):
    arr = np.asarray(sampled_params)
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]
    raise RuntimeError(f"Unexpected chain shape {arr.shape}")

# ============================================================
# Main loop
# ============================================================

def main():
    chains_list = None
    logps_list = []
    total_iters = 0
    batch_no = 0

    while total_iters < MAX_ITER:

        batch_no += 1
        this_batch = min(BATCH_SIZE, MAX_ITER - total_iters)

        print(f"\n[INFO] Batch {batch_no}: {this_batch} iterations")
        t0 = _time.time()

        sampled_params, logps = run_dream(
            parameters=param_priors,
            likelihood=likelihood_wrapper,
            niterations=this_batch,
            nchains=NCHAINS,
            mp_context=MP_CTX,
            snooker=True,
            adapt_gamma=True,
            save_history=True,
            model_name=MODEL_NAME,
        )

        batch_chains = normalize_sampled_params(sampled_params)
        logps_list.append(np.asarray(logps).ravel())

        if chains_list is None:
            chains_list = [ch.copy() for ch in batch_chains]
        else:
            for i in range(len(chains_list)):
                chains_list[i] = np.vstack([chains_list[i], batch_chains[i]])

        total_iters += this_batch

        print(f"[INFO] Batch finished in {_time.time() - t0:.1f} s")

        try:
            Rhat = Gelman_Rubin([np.asarray(ch) for ch in chains_list])
            print("[INFO] R-hat:", np.round(Rhat, 3))
            if np.all(Rhat < R_HAT_THRESH):
                print("[INFO] Converged — stopping early.")
                break
        except Exception as e:
            print("[WARN] R-hat failed:", e)

    np.save("multi_station_sampled_params.npy", np.stack(chains_list))
    np.save("multi_station_logps.npy", np.concatenate(logps_list))

    print("[INFO] Posterior samples saved.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
