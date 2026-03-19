import os
import numpy as np
import multiprocessing
import time as _time
import pandas as pd

from scipy.stats import uniform, norm
from pydream.core import run_dream
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as input_data

# ============================================================
# Setup
# ============================================================

params = input_data.read_input()
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
np.random.seed(42)

# ============================================================
# Read AVANT stations & observed strain dataset
# ============================================================

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

stations_df = pd.read_csv(STATION_FILE)
required_cols = ["station", "x_prime", "y_prime", "depth"]
missing = set(required_cols) - set(stations_df.columns)
if missing:
    raise RuntimeError(f"Missing columns in AVANT_stations.csv: {missing}")

station_names = stations_df["station"].astype(str).str.strip().values
x_prime = stations_df["x_prime"].values
y_prime = stations_df["y_prime"].values
z = stations_df["depth"].values
Ns = len(station_names)
print(f"[INFO] Using {Ns} AVANT stations: {station_names}")

observed_df = pd.read_csv(OBSERVED_FILE)
time = observed_df["time_s"].values
COMPONENT_COLS = [col for col in observed_df.columns if col != "time_s"]
observed_vector = observed_df[COMPONENT_COLS].values.flatten()
sigma_noise = 0.05 * np.std(observed_vector)

# ============================================================
# Geometry ratios (fixed)
# ============================================================

a0 = 1.0
b0 = 25.0 / 175.0
c0 = 125.0 / 175.0

# ============================================================
# Likelihood wrapper (robust, strain-only)
# ============================================================

def likelihood_wrapper(params_in):
    """
    params_in = [s, E_s, theta_deg_s, sigma_s]
    """
    s, E_s, theta_deg_s, sigma_s = np.ravel(params_in)
    a_s = s * a0
    b_s = s * b0
    c_s = s * c0

    try:
        # Run forward model
        df_pred = forward_model_multi_station(
            pmax=params["pmax"],
            tpeak=params["tpeak"],
            d=params["d"],
            time=time,
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=params["x0_prime"],
            y0_prime=params["y0_prime"],
            z=z,
            a=a_s, b=b_s, c=c_s,
            nu=params["nu"],
            h=params["h"],
            E=E_s,
            theta_deg=theta_deg_s,
            alpha=params.get("alpha", None),
            station_names=station_names,
        )

        # Make sure prediction has the same columns as observed
        pred_vector = []
        for col in COMPONENT_COLS:
            if col not in df_pred.columns:
                raise ValueError(f"[ERROR] Forward model missing column: {col}")
            pred_vector.append(df_pred[col].values)
        pred_vector = np.array(pred_vector).flatten()

        # Gaussian likelihood
        residual = observed_vector - pred_vector
        log_likelihood = -0.5 * np.sum((residual / sigma_s) ** 2 + np.log(2 * np.pi * sigma_s ** 2))
        return log_likelihood

    except Exception as e:
        print(f"[ERROR] Likelihood failed for params {params_in}: {e}")
        return -np.inf

# ============================================================
# Priors
# ============================================================

param_priors = [
    SampledParam(uniform, loc=50.0, scale=150.0),  # s
    SampledParam(uniform, loc=0.8e10, scale=0.4e10),  # E
    SampledParam(uniform, loc=params["theta_deg_start"] - 15.0, scale=30.0),  # theta
    SampledParam(norm, loc=max(1e-8, sigma_noise), scale=max(1e-8, 0.5 * sigma_noise)),  # sigma
]

# ============================================================
# Run control
# ============================================================

MAX_ITER = 50000
BATCH_SIZE = 20000
NCHAINS = 4
R_HAT_THRESH = 1.1
MODEL_NAME = "pydream_multi_station_strain_only"
MP_CTX = multiprocessing.get_context("spawn")

def normalize_sampled_params(sampled_params):
    arr = np.asarray(sampled_params)
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]
    raise RuntimeError(f"Unexpected chain shape {arr.shape}")

# ============================================================
# Main PyDREAM loop
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

    # Save posterior
    np.save("multi_station_sampled_params.npy", np.stack(chains_list))
    np.save("multi_station_logps.npy", np.concatenate(logps_list))
    print("[INFO] Posterior samples saved.")

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()