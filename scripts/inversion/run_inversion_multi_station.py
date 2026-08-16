#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyDREAM inversion for multi-station strain.

Inferred parameters:
    a, b, theta_deg, x0_prime, y0_prime, log10_sigma_strain

Fixed to true input values:
    pmax, E, c, and all other forward-model settings not listed above.

Notes:
- sigma_strain is interpreted as an effective aleatoric noise scale that absorbs
  mismatch between the layered COMSOL field model and the simplified analytical
  half-space inclusion model.
- This script uses only strain data. The volume term is intentionally omitted.
- The forward model and observed data are aligned in component-major order:
      eXX for all stations,
      eYY for all stations,
      eXY for all stations,
      eZZ for all stations.
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
# Fixed true values from input script
# ============================================================

pmax = float(params["pmax"])
E_fixed = float(params["E"])
c_fixed = float(params["c_fixed"])
nu = float(params["nu"])
alpha = params.get("alpha", None)
time = np.asarray(params["time"], dtype=float)
tpeak = float(params["tpeak"])
d = float(params["d"])
h = float(params["h"])

print(f"[INFO] Fixed inputs: pmax={pmax:.6g}, E={E_fixed:.6g}, c={c_fixed:.6g}")

# ============================================================
# Forward-model column detection
# ============================================================

def detect_component_columns():
    """Detect whether the forward model returns new-style or legacy strain names."""
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

    try:
        test_df = forward_model_multi_station(
            pmax=pmax,
            tpeak=tpeak,
            d=d,
            time=time,
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=float(params["x0_prime"]),
            y0_prime=float(params["y0_prime"]),
            z=z,
            a=float(params["a_fixed"]),
            b=float(params["b_fixed"]),
            c=c_fixed,
            nu=nu,
            h=h,
            E=E_fixed,
            theta_deg=float(params["theta_deg"]),
            alpha=alpha,
            station_names=station_names,
        )
    except Exception as e:
        print(f"[WARN] Could not probe forward-model column names: {e}")
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

def standardize_observed_strain(df, component_cols):
    """Return a dataframe whose strain columns match the forward-model ordering."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    strain_cols = [c for c in df.columns if c != "time_s"]

    if all(c in df.columns for c in component_cols):
        return df, component_cols

    rename_map = {}
    if component_cols[0].startswith("e"):
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
print(f"[INFO] Observed vector shape: {observed_vector.shape}")

# ============================================================
# Inclusion center (fixed true values used for the initial probe only)
# ============================================================

x0_fixed = float(params["x0_prime"])
y0_fixed = float(params["y0_prime"])

print(f"[INFO] Baseline inclusion center: x0={x0_fixed:.3f}, y0={y0_fixed:.3f}")

# ============================================================
# Likelihood wrapper
# ============================================================

# parameters = [a, b, theta_deg, x0_prime, y0_prime, log10_sigma_strain]

def likelihood_wrapper(params_in):
    a_s, b_s, theta_deg_s, x0_s, y0_s, log10_sigma_s = np.ravel(params_in)

    if a_s <= 0 or b_s <= 0:
        return -np.inf

    sigma_strain = 10.0 ** log10_sigma_s
    if sigma_strain <= 0:
        return -np.inf

    try:
        return _gaussian_strain_likelihood(
            a_s=a_s,
            b_s=b_s,
            theta_deg_s=theta_deg_s,
            x0_s=x0_s,
            y0_s=y0_s,
            sigma_strain=sigma_strain,
        )
    except Exception as e:
        print(f"[ERROR] Likelihood failed for params {params_in}: {e}")
        return -np.inf


def _gaussian_strain_likelihood(a_s, b_s, theta_deg_s, x0_s, y0_s, sigma_strain):
    """Compute strain-only Gaussian log-likelihood for the current parameter set."""
    df_pred = forward_model_multi_station(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_s,
        y0_prime=y0_s,
        z=z,
        a=a_s,
        b=b_s,
        c=c_fixed,
        nu=nu,
        h=h,
        E=E_fixed,
        theta_deg=theta_deg_s,
        alpha=alpha,
        station_names=station_names,
    )

    if not all(col in df_pred.columns for col in COMPONENT_COLS):
        missing_pred = [c for c in COMPONENT_COLS if c not in df_pred.columns]
        raise RuntimeError(f"Forward model missing columns: {missing_pred}")

    pred_vector = flatten_columns(df_pred, COMPONENT_COLS)
    resid = observed_vector - pred_vector

    # Simple IID Gaussian noise model on every strain datum.
    # This is an effective noise term absorbing data/model mismatch.
    n = resid.size
    log_like = -0.5 * np.sum((resid / sigma_strain) ** 2) - n * np.log(sigma_strain) - 0.5 * n * np.log(2.0 * np.pi)
    return log_like

# ============================================================
# Priors
# ============================================================

# Broad uniform priors. You can tighten them later if needed.
# a, b are inferred shape parameters in meters.
# theta_deg is in degrees.
# x0_prime and y0_prime are in the same coordinate system as the station layout.
# log10_sigma_strain spans a broad range in nanostrain units.

a_min, a_max = 40.0, 500.0
b_min, b_max = 20.0, 400.0
theta_min, theta_max = -90.0, 90.0
x0_min, x0_max = -400.0, 600.0
y0_min, y0_max = -500.0, 500.0
log10_sigma_min, log10_sigma_max = 0.0, np.log10(2000.0)

param_priors = [
    SampledParam(uniform, loc=a_min, scale=a_max - a_min),
    SampledParam(uniform, loc=b_min, scale=b_max - b_min),
    SampledParam(uniform, loc=theta_min, scale=theta_max - theta_min),
    SampledParam(uniform, loc=x0_min, scale=x0_max - x0_min),
    SampledParam(uniform, loc=y0_min, scale=y0_max - y0_min),
    SampledParam(uniform, loc=log10_sigma_min, scale=log10_sigma_max - log10_sigma_min),
]

PARAM_NAMES = ["a", "b", "theta_deg", "x0_prime", "y0_prime", "log10_sigma_strain"]

# ============================================================
# Run control
# ============================================================

MAX_ITER = 50000
BATCH_SIZE = 5000
NCHAINS = 4
R_HAT_THRESH = 1.1
MODEL_NAME = "pydream_multi_station_strain_ab_theta_center_sigma"
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

    a_map, b_map, theta_map, x0_map, y0_map, log10_sigma_map = map_params
    sigma_map = 10.0 ** log10_sigma_map

    df_pred = forward_model_multi_station(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_map,
        y0_prime=y0_map,
        z=z,
        a=a_map,
        b=b_map,
        c=c_fixed,
        nu=nu,
        h=h,
        E=E_fixed,
        theta_deg=theta_map,
        alpha=alpha,
        station_names=station_names,
    )

    if not all(col in df_pred.columns for col in COMPONENT_COLS):
        missing_pred = [c for c in COMPONENT_COLS if c not in df_pred.columns]
        raise RuntimeError(f"Forward model missing columns: {missing_pred}")

    pred_vector = flatten_columns(df_pred, COMPONENT_COLS)
    strain_r2 = compute_r2(observed_vector, pred_vector)

    burn_frac = 0.5
    burn_in = int(len(flat_samples) * burn_frac)
    posterior_burn_in = flat_samples[burn_in:]

    a_std = float(np.std(posterior_burn_in[:, 0]))
    b_std = float(np.std(posterior_burn_in[:, 1]))
    theta_std = float(np.std(posterior_burn_in[:, 2]))
    x0_std = float(np.std(posterior_burn_in[:, 3]))
    y0_std = float(np.std(posterior_burn_in[:, 4]))
    sigma_std = float(np.std(10.0 ** posterior_burn_in[:, 5]))

    summary = {
        "MAP": {
            "a": float(a_map),
            "b": float(b_map),
            "theta_deg": float(theta_map),
            "x0_prime": float(x0_map),
            "y0_prime": float(y0_map),
            "sigma_strain": float(sigma_map),
        },
        "STD": {
            "a": a_std,
            "b": b_std,
            "theta_deg": theta_std,
            "x0_prime": x0_std,
            "y0_prime": y0_std,
            "sigma_strain": sigma_std,
        },
        "fixed_inputs": {
            "pmax": pmax,
            "E": E_fixed,
            "c": c_fixed,
            "nu": nu,
            "h": h,
            "d": d,
        },
        "fit": {
            "strain_R2_MAP": float(strain_r2),
        },
        "priors": {
            "a": [a_min, a_max],
            "b": [b_min, b_max],
            "theta_deg": [theta_min, theta_max],
            "x0_prime": [x0_min, x0_max],
            "y0_prime": [y0_min, y0_max],
            "log10_sigma_strain": [log10_sigma_min, log10_sigma_max],
        },
        "parameter_names": PARAM_NAMES,
    }

    with open("posterior_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("[INFO] Saved posterior_summary.json")
    print(f"[INFO] MAP a: {a_map:.3f}, b: {b_map:.3f}")
    print(f"[INFO] MAP theta_deg: {theta_map:.3f}")
    print(f"[INFO] MAP x0_prime: {x0_map:.3f}, y0_prime: {y0_map:.3f}")
    print(f"[INFO] MAP sigma_strain: {sigma_map:.3f}")
    print(f"[INFO] Strain R^2: {strain_r2:.4f}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
