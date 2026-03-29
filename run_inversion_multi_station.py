# main_inversion_reparam_volume.py
import os
import json
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

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")
VOLUME_FILE = os.path.join(BASE_DIR, "observed_volume.csv")  # optional

# ============================================================
# Read stations and observed strain data
# ============================================================

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

observed_df = pd.read_csv(OBSERVED_FILE).drop_duplicates().reset_index(drop=True)
if "time_s" not in observed_df.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

time = observed_df["time_s"].values.astype(float)

EXPECTED_COMPONENT_COLS = [
    f"{comp}_{sname}"
    for comp in ["eXX", "eYY", "eXY", "eZZ"]
    for sname in station_names
]

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
print(f"[INFO] Observed matrix shape: {observed_matrix.shape}")
print(f"[INFO] Observed min/max: {np.min(observed_matrix):.6g} / {np.max(observed_matrix):.6g}")
print(f"[INFO] Observed mean abs: {np.mean(np.abs(observed_matrix)):.6g}")
print(f"[INFO] Observed std: {np.std(observed_matrix):.6g}")

# Optional volume observations:
# 1) if observed_df has a Volume column, use it;
# 2) else if observed_volume.csv exists, load it and interpolate to strain time grid;
# 3) else disable the volume term.
use_volume_term = False
volume_obs = None

if "Volume" in observed_df.columns:
    volume_obs = observed_df["Volume"].values.astype(float)
    if len(volume_obs) != len(time):
        raise RuntimeError("Volume column in observed strain file must have same length as time_s.")
    use_volume_term = True
    print("[INFO] Using Volume column from observed strain file.")
elif os.path.exists(VOLUME_FILE):
    vol_df = pd.read_csv(VOLUME_FILE).drop_duplicates().reset_index(drop=True)
    if not {"time_s", "Volume"}.issubset(vol_df.columns):
        raise RuntimeError("observed_volume.csv must contain columns 'time_s' and 'Volume'.")
    vol_time = vol_df["time_s"].values.astype(float)
    vol_vals = vol_df["Volume"].values.astype(float)
    # interpolate onto strain time grid if needed
    volume_obs = np.interp(time, vol_time, vol_vals)
    use_volume_term = True
    print(f"[INFO] Loaded volume observations from {VOLUME_FILE} and interpolated to strain time grid.")
else:
    print("[WARN] No observed volume data found. Volume term will be disabled.")

# ============================================================
# Fixed physical values and reparameterization
# ============================================================

nu = float(params.get("nu", 0.25))
alpha = float(params.get("alpha", 0.8))

pmax_fixed = float(params.get("pmax", 2.0e6))
tpeak_fixed = float(params.get("tpeak", 393333.0))
d_fixed = float(params.get("d", 0.4))

# Fixed center from the current input script
x0_prime = float(params.get("x0_prime", -125.0))
y0_prime = float(params.get("y0_prime", 4.0))

# Fixed thickness semi-axis (c) and reference in-plane geometry
a_ref = float(params.get("a_fixed", 150.0))
b_ref = float(params.get("b_fixed", 290.0))
c_fixed = float(params.get("c_fixed", 2.5))

# Reparameterization:
#   a = s * a_ref
#   b = s * b_ref
#   c = c_fixed
#
# This keeps thickness fixed while letting the in-plane scale vary.
theta_start = float(params.get("theta_deg_start", 75.0))

print("[INFO] Fixed values:")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       tpeak = {tpeak_fixed:.6g}")
print(f"       d = {d_fixed:.6g}")
print(f"       x0_prime = {x0_prime:.6g}")
print(f"       y0_prime = {y0_prime:.6g}")
print(f"       a_ref = {a_ref:.6g}")
print(f"       b_ref = {b_ref:.6g}")
print(f"       c_fixed = {c_fixed:.6g}")
print(f"       theta_start = {theta_start:.6g}")
print(f"       use_volume_term = {use_volume_term}")

# ============================================================
# Volume forward model
# ============================================================

def bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))

def volume_forward(a, b, c, E, nu, delta_P):
    V0 = a * b * c
    K = bulk_modulus(E, nu)
    return V0 * (1.0 + delta_P / K)

delta_P = pmax_fixed * np.sin(np.pi * time / np.max(time))

if use_volume_term:
    # 5% of the observed volume standard deviation, with a small floor
    sigma_volume = max(0.05 * np.std(volume_obs), 1e-6)
    print(f"[INFO] Volume sigma (fixed) = {sigma_volume:.6g}")
else:
    sigma_volume = None

# ============================================================
# Likelihood wrapper
# ============================================================

def likelihood_wrapper(params_in):
    """
    params_in = [s, E, theta_deg, sigma_strain]
    Geometry is reconstructed as:
      a = s * a_ref
      b = s * b_ref
      c = c_fixed
    """
    s, E_s, theta_deg_s, sigma_strain_s = np.ravel(params_in)

    if sigma_strain_s <= 0:
        return -np.inf

    a_s = s * a_ref
    b_s = s * b_ref
    c_s = c_fixed

    try:
        df_pred = forward_model_multi_station(
            pmax=pmax_fixed,
            tpeak=tpeak_fixed,
            d=d_fixed,
            time=time,
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=x0_prime,
            y0_prime=y0_prime,
            z=z,
            a=a_s,
            b=b_s,
            c=c_s,
            nu=nu,
            h=float(params.get("h", 518.29)),
            E=E_s,
            theta_deg=theta_deg_s,
            alpha=alpha,
            station_names=station_names,
            debug=False,
        )

        if not all(col in df_pred.columns for col in EXPECTED_COMPONENT_COLS):
            missing_pred = [c for c in EXPECTED_COMPONENT_COLS if c not in df_pred.columns]
            raise ValueError(f"Forward model missing columns: {missing_pred}")

        pred_vector = df_pred[EXPECTED_COMPONENT_COLS].values.astype(float).flatten()
        residual_strain = observed_vector - pred_vector

        log_like_strain = -0.5 * np.sum(
            (residual_strain / sigma_strain_s) ** 2
            + np.log(2.0 * np.pi * sigma_strain_s ** 2)
        )

        log_like_volume = 0.0
        if use_volume_term:
            volume_pred = volume_forward(a_s, b_s, c_s, E_s, nu, delta_P)
            residual_volume = volume_obs - volume_pred
            log_like_volume = -0.5 * np.sum(
                (residual_volume / sigma_volume) ** 2
                + np.log(2.0 * np.pi * sigma_volume ** 2)
            )

        return float(log_like_strain + log_like_volume)

    except Exception as e:
        print(f"[ERROR] Likelihood failed for params {params_in}: {e}")
        return -np.inf

# ============================================================
# Priors
# ============================================================

param_priors = [
    # s: broad scale around the reference in-plane geometry
    SampledParam(uniform, loc=0.1, scale=9.9),              # s in [0.1, 10.0]

    # Young's modulus
    SampledParam(uniform, loc=0.5e9, scale=3.0e10),         # E in [0.5 GPa, 30.5 GPa]

    # Orientation
    SampledParam(uniform, loc=-90.0, scale=180.0),          # theta in [-90, 90]

    # Strain noise scale
    SampledParam(uniform, loc=1.0, scale=250.0),            # sigma_strain in [1, 251]
]

# ============================================================
# Run control
# ============================================================

MAX_ITER = 50000
BATCH_SIZE = 20000
NCHAINS = 4
R_HAT_THRESH = 1.1
MODEL_NAME = "pydream_multi_station_reparam_strain_volume"
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
    # Simple post-processing
    # ============================================================

    samples = np.load("multi_station_sampled_params.npy")
    logps = np.load("multi_station_logps.npy").flatten()
    flat_samples = samples.reshape(-1, samples.shape[-1])

    map_idx = int(np.argmax(logps))
    s_map, E_map, theta_map, sigma_strain_map = flat_samples[map_idx]

    a_map = s_map * a_ref
    b_map = s_map * b_ref
    c_map = c_fixed

    df_map = forward_model_multi_station(
        pmax=pmax_fixed,
        tpeak=tpeak_fixed,
        d=d_fixed,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_prime,
        y0_prime=y0_prime,
        z=z,
        a=a_map,
        b=b_map,
        c=c_map,
        nu=nu,
        h=float(params.get("h", 518.29)),
        E=E_map,
        theta_deg=theta_map,
        alpha=alpha,
        station_names=station_names,
        debug=False,
    )

    pred_vector = df_map[EXPECTED_COMPONENT_COLS].values.astype(float).flatten()
    ss_res = np.sum((observed_vector - pred_vector) ** 2)
    ss_tot = np.sum((observed_vector - np.mean(observed_vector)) ** 2)
    r2_strain = float(1.0 - ss_res / ss_tot)

    r2_volume = None
    if use_volume_term:
        volume_map = volume_forward(a_map, b_map, c_map, E_map, nu, delta_P)
        ss_res_v = np.sum((volume_obs - volume_map) ** 2)
        ss_tot_v = np.sum((volume_obs - np.mean(volume_obs)) ** 2)
        r2_volume = float(1.0 - ss_res_v / ss_tot_v)

    burn_in = int(0.5 * len(flat_samples))
    posterior_burn_in = flat_samples[burn_in:]

    summary = {
        "s": {"MAP": float(s_map), "STD": float(np.std(posterior_burn_in[:, 0]))},
        "E": {"MAP": float(E_map), "STD": float(np.std(posterior_burn_in[:, 1]))},
        "theta": {"MAP": float(theta_map), "STD": float(np.std(posterior_burn_in[:, 2]))},
        "sigma_strain": {"MAP": float(sigma_strain_map), "STD": float(np.std(posterior_burn_in[:, 3]))},
        "r2_strain": r2_strain,
        "r2_volume": r2_volume,
        "volume_term_used": bool(use_volume_term),
        "fixed_values": {
            "pmax": pmax_fixed,
            "tpeak": tpeak_fixed,
            "d": d_fixed,
            "x0_prime": x0_prime,
            "y0_prime": y0_prime,
            "a_ref": a_ref,
            "b_ref": b_ref,
            "c_fixed": c_fixed,
            "h": float(params.get("h", 518.29)),
            "nu": nu,
            "alpha": alpha,
        },
    }

    with open("posterior_summary_reparam_volume.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("[INFO] Saved posterior_summary_reparam_volume.json")
    print(f"[INFO] Strain R2 = {r2_strain:.6g}")
    if r2_volume is not None:
        print(f"[INFO] Volume R2 = {r2_volume:.6g}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()