"""
PyDREAM inversion for multi-station strain + volume.

Reparameterization:
    b = b_free
    a = b_free * (a_fixed / b_fixed)
    c = b_free * (c_fixed / b_fixed)

This means b is the inferred length scale, and a/c follow the fixed
shape ratios from the input file.

Noise scale is kept fixed at sigma_noise from the input file.

IMPORTANT:
The observed strain vector is built in the same component-major order
as the forward-model prediction to avoid misaligned likelihood terms.
"""

import os
import json
import time as _time
import multiprocessing

import numpy as np
import pandas as pd
from scipy.stats import uniform

from pydream.core import run_dream
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin

from forward_model_multi_station import forward_model_multi_station
import bayesian_inversion_multi_station as bi
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
# Files
# ============================================================

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")
VOLUME_FILE = os.path.join(BASE_DIR, "avant_cleaned_volume.csv")

# ============================================================
# Station metadata
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

# ============================================================
# Geometry / forward-model naming detection
# ============================================================

a_fixed = float(params["a_fixed"])
b_fixed = float(params["b_fixed"])
c_fixed = float(params["c_fixed"])

a_over_b = a_fixed / b_fixed
c_over_b = c_fixed / b_fixed

def reparameterize_geometry(b_free):
    """
    Reconstruct geometry using b as the inferred scale.
    """
    b_s = float(b_free)
    a_s = b_s * a_over_b
    c_s = b_s * c_over_b
    return a_s, b_s, c_s

def detect_component_columns():
    """
    Detect whether the forward model returns new-style names:
        eXX_FS01, eYY_FS01, ...
    or legacy names:
        Epsilon_XX_nanostrain_FS01, ...

    The detected ordering becomes the canonical ordering for the inversion.
    """
    new_cols = [f"{comp}_{sname}" for comp in ["eXX", "eYY", "eXY", "eZZ"] for sname in station_names]
    legacy_cols = [
        f"{comp}_{sname}"
        for comp in [
            "Epsilon_XX_nanostrain",
            "Epsilon_YY_nanostrain",
            "Epsilon_XY_nanostrain",
            "Epsilon_ZZ_nanostrain",
        ]
        for sname in station_names
    ]

    # Probe the forward model once with nominal values from the input file.
    try:
        test_df = forward_model_multi_station(
            pmax=params["pmax"],
            tpeak=params["tpeak"],
            d=params["d"],
            time=np.asarray(params["time"], dtype=float),
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=float(params["x0_prime"]),
            y0_prime=float(params["y0_prime"]),
            z=z,
            a=a_fixed,
            b=b_fixed,
            c=c_fixed,
            nu=params["nu"],
            h=float(params["h"]),
            E=float(params["E"]),
            theta_deg=float(params["theta_deg"]),
            alpha=params.get("alpha", None),
            station_names=station_names,
        )
    except Exception as e:
        print(f"[WARN] Could not probe forward-model column names: {e}")
        # Default to new-style names if probing fails.
        return new_cols

    if all(c in test_df.columns for c in new_cols):
        return new_cols
    if all(c in test_df.columns for c in legacy_cols):
        return legacy_cols

    raise RuntimeError(
        "Could not detect a valid forward-model strain column convention.\n"
        f"New-style example columns: {new_cols[:4]}\n"
        f"Legacy-style example columns: {legacy_cols[:4]}\n"
        f"Available forward-model columns: {list(test_df.columns)}"
    )

COMPONENT_COLS = detect_component_columns()
print(f"[INFO] Using strain column convention: {COMPONENT_COLS[0].split('_')[0]} ...")

# ============================================================
# Observed strain data
# ============================================================

observed_df = pd.read_csv(OBSERVED_FILE)
observed_df.columns = [str(c).strip() for c in observed_df.columns]

if "time_s" not in observed_df.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

time = observed_df["time_s"].values.astype(float)

def standardize_observed_strain(df, component_cols):
    """
    Return a dataframe whose strain columns are in the exact order used by
    the forward model / likelihood.

    Supports:
      - exact component_cols
      - opposite naming convention (new vs legacy)
      - generic 16-channel files (columns in file order)
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    strain_cols = [c for c in df.columns if c not in ["time_s", "Volume"]]

    # Case 1: already in the exact detected convention
    if all(c in df.columns for c in component_cols):
        return df, component_cols

    # Case 2: rename from the alternate convention
    rename_map = {}
    if component_cols[0].startswith("e"):
        # Detected new-style columns; try legacy-to-new renaming
        legacy_bases = {
            "eXX": "Epsilon_XX_nanostrain",
            "eYY": "Epsilon_YY_nanostrain",
            "eXY": "Epsilon_XY_nanostrain",
            "eZZ": "Epsilon_ZZ_nanostrain",
        }
        for comp in ["eXX", "eYY", "eXY", "eZZ"]:
            for sname in station_names:
                old = f"{legacy_bases[comp]}_{sname}"
                new = f"{comp}_{sname}"
                if old in df.columns and new not in df.columns:
                    rename_map[old] = new
    else:
        # Detected legacy-style columns; try new-to-legacy renaming
        new_bases = {
            "Epsilon_XX_nanostrain": "eXX",
            "Epsilon_YY_nanostrain": "eYY",
            "Epsilon_XY_nanostrain": "eXY",
            "Epsilon_ZZ_nanostrain": "eZZ",
        }
        for legacy_base, comp in new_bases.items():
            for sname in station_names:
                old = f"{comp}_{sname}"
                new = f"{legacy_base}_{sname}"
                if old in df.columns and new not in df.columns:
                    rename_map[old] = new

    if rename_map:
        df = df.rename(columns=rename_map)
        if all(c in df.columns for c in component_cols):
            return df, component_cols

    # Case 3: generic 16-channel file in file order
    if len(strain_cols) == len(component_cols):
        rename_map = {old: new for old, new in zip(strain_cols, component_cols)}
        df = df.rename(columns=rename_map)
        return df, component_cols

    raise RuntimeError(
        "Observed file columns do not match the expected strain layout.\n"
        f"Expected {len(component_cols)} strain columns.\n"
        f"Available columns: {list(df.columns)}"
    )

observed_df, COMPONENT_COLS = standardize_observed_strain(observed_df, COMPONENT_COLS)

def flatten_columns(df, cols):
    return np.concatenate([df[c].values.astype(float) for c in cols])

observed_vector = flatten_columns(observed_df, COMPONENT_COLS)

# Fixed strain noise scale from input file
sigma_noise = 0.2 * np.std(observed_vector)
print(f"[INFO] Observed vector shape: {observed_vector.shape}")
print(f"[INFO] Estimated noise scale: {sigma_noise:.3f}")

# ============================================================
# Volume observations
# ============================================================

VOLUME_COLUMN = "Volume"
VOLUME_WEIGHT = 1.0

volume_obs = None
sigma_volume = None

if VOLUME_COLUMN in observed_df.columns:
    volume_obs = observed_df[VOLUME_COLUMN].values.astype(float)
elif os.path.exists(VOLUME_FILE):
    volume_df = pd.read_csv(VOLUME_FILE)
    volume_df.columns = [str(c).strip() for c in volume_df.columns]

    if VOLUME_COLUMN not in volume_df.columns:
        raise RuntimeError(f"'{VOLUME_FILE}' exists but does not contain a '{VOLUME_COLUMN}' column.")

    volume_obs = volume_df[VOLUME_COLUMN].values.astype(float)

    if "time_s" in volume_df.columns:
        vol_time = volume_df["time_s"].values.astype(float)
        if len(vol_time) != len(time) or not np.allclose(vol_time, time):
            raise RuntimeError("Volume time array does not match strain time array.")

if volume_obs is not None:
    sigma_volume = max(0.05 * np.std(volume_obs), 1.0)

delta_P = params["pmax"] * np.sin(np.pi * time / np.max(time))

# ============================================================
# Inclusion center
# ============================================================

x0_prime = float(params["x0_prime"])
y0_prime = float(params["y0_prime"])

print(f"[INFO] Using inclusion center x0_prime={x0_prime:.3f}, y0_prime={y0_prime:.3f}")

# ============================================================
# Likelihood wrapper
# ============================================================

def likelihood_wrapper(params_in):
    """
    params_in = [b_free, E_s, theta_deg_s]
    """
    b_free, E_s, theta_deg_s = np.ravel(params_in)

    if b_free <= 0 or E_s <= 0:
        return -np.inf

    a_s, b_s, c_s = reparameterize_geometry(b_free)

    try:
        return bi.likelihood(
            params=[a_s, b_s, c_s, E_s, theta_deg_s, sigma_noise],
            observed_vector=observed_vector,
            time=time,
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=x0_prime,
            y0_prime=y0_prime,
            z=z,
            nu=params["nu"],
            pmax=params["pmax"],
            tpeak=params["tpeak"],
            d=params["d"],
            h=params["h"],
            alpha=params.get("alpha", None),
            component_cols=COMPONENT_COLS,
            volume_obs=volume_obs,
            delta_P=delta_P,
            sigma_volume=sigma_volume,
            volume_weight=VOLUME_WEIGHT,
            station_names=station_names,
        )
    except Exception as e:
        print(f"[ERROR] Likelihood failed for params {params_in}: {e}")
        return -np.inf

# ============================================================
# Priors
# ============================================================

param_priors = [
    # b: 100 to 600
    SampledParam(uniform, loc=100.0, scale=500.0),

    # E: 1 GPa to 30 GPa
    SampledParam(uniform, loc=0.5e9, scale=2.9e10),

    # theta: 30 to 100 degrees
    SampledParam(uniform, loc=30.0, scale=70.0),
]

# ============================================================
# Run control
# ============================================================

MAX_ITER = 50000
BATCH_SIZE = 5000
NCHAINS = 4
R_HAT_THRESH = 1.1
MODEL_NAME = "pydream_multi_station_strain_infer_b_only"
MP_CTX = multiprocessing.get_context("spawn")

def normalize_sampled_params(sampled_params):
    arr = np.asarray(sampled_params)
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]
    raise RuntimeError(f"Unexpected chain shape {arr.shape}")

def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot

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

    # ========================================================
    # MAP summary
    # ========================================================

    samples = np.load("multi_station_sampled_params.npy")
    logps = np.load("multi_station_logps.npy")

    flat_samples = samples.reshape(-1, samples.shape[-1])
    map_idx = int(np.argmax(logps))
    map_params = flat_samples[map_idx]

    b_scale_map, E_map, theta_map = map_params
    a_map, b_map, c_map = reparameterize_geometry(b_scale_map)

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
        a=a_map,
        b=b_map,
        c=c_map,
        nu=params["nu"],
        h=params["h"],
        E=E_map,
        theta_deg=theta_map,
        alpha=params.get("alpha", None),
        station_names=station_names,
    )

    if not all(col in df_pred.columns for col in COMPONENT_COLS):
        missing_pred = [c for c in COMPONENT_COLS if c not in df_pred.columns]
        raise RuntimeError(f"Forward model missing columns: {missing_pred}")

    pred_vector = flatten_columns(df_pred, COMPONENT_COLS)
    strain_r2 = compute_r2(observed_vector, pred_vector)

    volume_r2 = None
    if volume_obs is not None:
        V_pred_map = bi.volume_forward(a_map, b_map, c_map, E_map, params["nu"], delta_P)
        volume_r2 = compute_r2(volume_obs, V_pred_map)

    burn_frac = 0.5
    burn_in = int(len(flat_samples) * burn_frac)
    posterior_burn_in = flat_samples[burn_in:]

    b_std = float(np.std(posterior_burn_in[:, 0]))
    E_std = float(np.std(posterior_burn_in[:, 1]))
    theta_std = float(np.std(posterior_burn_in[:, 2]))

    summary = {
        "geometry": {
            "a_fixed": float(a_fixed),
            "b_fixed": float(b_fixed),
            "c_fixed": float(c_fixed),
            "MAP": {
                "b_scale": float(b_scale_map),
                "a": float(a_map),
                "b": float(b_map),
                "c": float(c_map),
            },
            "STD": {
                "b_scale": b_std,
            },
        },
        "E": {"MAP": float(E_map), "STD": E_std},
        "theta_deg": {"MAP": float(theta_map), "STD": theta_std},
        "sigma_noise_fixed": float(sigma_noise),
        "strain_R2": float(strain_r2),
        "volume_R2": None if volume_r2 is None else float(volume_r2),
        "volume_weight": float(VOLUME_WEIGHT),
        "priors": {
            "b_scale": [100.0, 600.0],
            "E": [1.0e9, 3.0e10],
            "theta_deg": [30.0, 100.0],
        },
    }

    with open("posterior_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("[INFO] Saved posterior_summary.json")
    print(f"[INFO] MAP geometry: a={a_map:.3f}, b={b_map:.3f}, c={c_map:.3f}")
    print(f"[INFO] MAP E: {E_map:.6e}")
    print(f"[INFO] MAP theta_deg: {theta_map:.3f}")
    print(f"[INFO] Strain R^2: {strain_r2:.4f}")
    if volume_r2 is not None:
        print(f"[INFO] Volume R^2: {volume_r2:.4f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()