import os
import numpy as np
import multiprocessing
import time as _time
import pandas as pd

from scipy.stats import uniform
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
# Read AVANT stations and observed strain dataset
# ============================================================

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

stations_df = pd.read_csv(STATION_FILE)
required_cols = ["station", "x_prime", "y_prime", "depth"]
missing = set(required_cols) - set(stations_df.columns)
if missing:
    raise RuntimeError(f"Missing columns in AVANT_stations.csv: {missing}")

stations_df["station"] = stations_df["station"].astype(str).str.strip()
station_names = stations_df["station"].values
x_prime = stations_df["x_prime"].values.astype(float)
y_prime = stations_df["y_prime"].values.astype(float)
z = stations_df["depth"].values.astype(float)

Ns = len(station_names)
print(f"[INFO] Using {Ns} AVANT stations: {station_names}")

observed_df = pd.read_csv(OBSERVED_FILE)

if "time_s" not in observed_df.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

time = observed_df["time_s"].values.astype(float)

# Expected output order from the new forward model:
#   eXX_S1, eXX_S2, eXX_S3, eXX_S4,
#   eYY_S1, eYY_S2, eYY_S3, eYY_S4,
#   eXY_S1, eXY_S2, eXY_S3, eXY_S4,
#   eZZ_S1, eZZ_S2, eZZ_S3, eZZ_S4
EXPECTED_COMPONENT_COLS = [
    f"{comp}_{sname}"
    for comp in ["eXX", "eYY", "eXY", "eZZ"]
    for sname in station_names
]

# If the file already has explicit names, use them.
# Otherwise assume the non-time columns are in the COMSOL order.
non_time_cols = [c for c in observed_df.columns if c != "time_s"]

if all(col in observed_df.columns for col in EXPECTED_COMPONENT_COLS):
    observed_matrix = observed_df[EXPECTED_COMPONENT_COLS].values.astype(float)
elif len(non_time_cols) == len(EXPECTED_COMPONENT_COLS):
    observed_matrix = observed_df[non_time_cols].values.astype(float)
else:
    raise RuntimeError(
        "Observed file columns do not match the expected 16-channel layout. "
        f"Found {len(non_time_cols)} non-time columns, expected {len(EXPECTED_COMPONENT_COLS)}."
    )

observed_vector = observed_matrix.flatten()

# A simple noise scale estimate in the same units as the strains (nstrain)
sigma_noise = np.std(observed_vector)
sigma_noise = max(sigma_noise, 1.0)

# ============================================================
# Geometry ratios (fixed)
# ============================================================

a0 = 1.0
b0 = 25.0 / 175.0
c0 = 125.0 / 175.0

# ============================================================
# Fix the inclusion center for now
# ============================================================

# A reasonable starting point is the station centroid in primed coordinates.
# You can replace these later if you want to invert for the inclusion center.
x0_prime = float(np.mean(x_prime))
y0_prime = float(np.mean(y_prime))

print(f"[INFO] Using fixed inclusion center x0_prime={x0_prime:.3f}, y0_prime={y0_prime:.3f}")

# ============================================================
# Likelihood wrapper
# ============================================================

def likelihood_wrapper(params_in):
    """
    params_in = [s, E_s, theta_deg_s, sigma_s]
    """
    s, E_s, theta_deg_s, sigma_s = np.ravel(params_in)

    # Enforce positivity of sigma in a very simple way
    if sigma_s <= 0:
        return -np.inf

    a_s = s * a0
    b_s = s * b0
    c_s = s * c0

    try:
        df_pred = forward_model_multi_station(
            pmax=params["pmax"],
            tpeak=params["tpeak"],
            d=params["d"],
            time=time,
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=x0_prime,
            y0_prime=y0_prime,
            z=z,
            a=a_s,
            b=b_s,
            c=c_s,
            nu=params["nu"],
            h=params["h"],
            E=E_s,
            theta_deg=theta_deg_s,
            alpha=params.get("alpha", None),
            station_names=station_names,
        )

        # Ensure the prediction uses the same 16-channel layout
        if not all(col in df_pred.columns for col in EXPECTED_COMPONENT_COLS):
            missing_pred = [c for c in EXPECTED_COMPONENT_COLS if c not in df_pred.columns]
            raise ValueError(f"Forward model missing columns: {missing_pred}")

        pred_vector = df_pred[EXPECTED_COMPONENT_COLS].values.astype(float).flatten()

        residual = observed_vector - pred_vector

        # Gaussian log-likelihood with scalar sigma
        log_likelihood = -0.5 * np.sum(
            (residual / sigma_s) ** 2 + np.log(2.0 * np.pi * sigma_s ** 2)
        )

        return float(log_likelihood)

    except Exception as e:
        print(f"[ERROR] Likelihood failed for params {params_in}: {e}")
        return -np.inf

# ============================================================
# Priors
# ============================================================

param_priors = [
    SampledParam(uniform, loc=50.0, scale=150.0),                # s
    SampledParam(uniform, loc=0.8e10, scale=0.4e10),             # E
    SampledParam(uniform, loc=params["theta_deg_start"] - 15.0, scale=30.0),  # theta
    SampledParam(uniform, loc=1.0, scale=200.0),                 # sigma (positive)
]

# ============================================================
# Run control
# ============================================================

MAX_ITER = 50000
BATCH_SIZE = 20000
NCHAINS = 4
R_HAT_THRESH = 1.1
MODEL_NAME = "pydream_multi_station_strain_comsol_order"
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

    np.save("multi_station_sampled_params.npy", np.stack(chains_list))
    np.save("multi_station_logps.npy", np.concatenate(logps_list))
    print("[INFO] Posterior samples saved.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()