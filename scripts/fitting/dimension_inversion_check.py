#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyDREAM inversion for multi-station strain with explicit geometry naming.

This script enforces a clear convention:
    L_long_half  = long half-length  (m)
    L_short_half = short half-length (m)

For the current forward model, these are still passed as the ellipsoidal
semi-axis inputs `a` and `b`. The goal here is to prevent axis swapping
in the inference by using explicit names and an ordering constraint.

Important:
- This does NOT turn the forward model into a true rectangular inclusion solver.
- It only makes the parameter naming explicit and draws the plan-view footprint
  as a rectangle for visualization.
- If you need true rectangular physics, `forward_model_multi_station` must be
  replaced with a rectangular-inclusion solution.
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
# Geometry convention
# ============================================================
# Explicit names prevent axis swapping.
# Use these values if you want to hold geometry fixed:
L_LONG_HALF_FIXED = 295.0   # long half-length (m)
L_SHORT_HALF_FIXED = 150.0  # short half-length (m)
C_HALF_FIXED = 2.5          # thickness half-length (m)

# Set to True if you want to infer geometry; False keeps geometry fixed.
FIT_GEOMETRY = True

# Rectangle orientation in the plot uses the same theta_deg parameter.
# Note: the forward model still interprets this as the inclusion rotation.

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
# Fixed values from input script
# ============================================================

pmax = float(params["pmax"])
E_fixed = float(params["E"])
nu = float(params["nu"])
alpha = params.get("alpha", None)
time = np.asarray(params["time"], dtype=float)
tpeak = float(params["tpeak"])
d = float(params["d"])
h = float(params["h"])

# Use the explicit geometry naming from above.
# If you need a different convention, change the two assignments below only.
L_LONG_HALF_INPUT = float(params.get("b_fixed", L_LONG_HALF_FIXED))
L_SHORT_HALF_INPUT = float(params.get("a_fixed", L_SHORT_HALF_FIXED))
C_FIXED = float(params.get("c_fixed", C_HALF_FIXED))

# Geometry center from input script
x0_true = float(params["x0_prime"])
y0_true = float(params["y0_prime"])

theta_true = float(params["theta_deg"])

print(f"[INFO] Fixed inputs: pmax={pmax:.6g}, E={E_fixed:.6g}, c={C_FIXED:.6g}")
print(f"[INFO] Input geometry: L_long_half={L_LONG_HALF_INPUT:.6g}, L_short_half={L_SHORT_HALF_INPUT:.6g}")
print(f"[INFO] Input center: x0={x0_true:.6g}, y0={y0_true:.6g}, theta={theta_true:.6g}")

# ============================================================
# Component columns
# ============================================================

def detect_component_columns():
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
            x0_prime=x0_true,
            y0_prime=y0_true,
            z=z,
            a=L_LONG_HALF_INPUT,
            b=L_SHORT_HALF_INPUT,
            c=C_FIXED,
            nu=nu,
            h=h,
            E=E_fixed,
            theta_deg=theta_true,
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
# Geometry helper
# ============================================================


def geometry_to_forward_params(L_long_half, L_short_half):
    """Map explicit long/short half-lengths to forward-model a/b."""
    return float(L_long_half), float(L_short_half)


# ============================================================
# Likelihood
# ============================================================
# Parameters:
#   If FIT_GEOMETRY=True:
#       [L_short_half, delta_half, theta_deg, x0_prime, y0_prime, log10_sigma_strain]
#       where L_long_half = L_short_half + delta_half
#       This enforces L_long_half >= L_short_half.
#   If FIT_GEOMETRY=False:
#       only theta_deg, x0_prime, y0_prime, log10_sigma_strain are inferred
#       and geometry is fixed to the explicit values above.


def likelihood_wrapper(params_in):
    arr = np.ravel(params_in)

    if FIT_GEOMETRY:
        L_short_half_s, delta_half_s, theta_deg_s, x0_s, y0_s, log10_sigma_s = arr
        if L_short_half_s <= 0 or delta_half_s < 0:
            return -np.inf
        L_long_half_s = L_short_half_s + delta_half_s
    else:
        theta_deg_s, x0_s, y0_s, log10_sigma_s = arr
        L_short_half_s = L_SHORT_HALF_INPUT
        L_long_half_s = L_LONG_HALF_INPUT

    sigma_strain = 10.0 ** log10_sigma_s
    if sigma_strain <= 0:
        return -np.inf

    try:
        return _gaussian_strain_likelihood(
            L_long_half_s=L_long_half_s,
            L_short_half_s=L_short_half_s,
            theta_deg_s=theta_deg_s,
            x0_s=x0_s,
            y0_s=y0_s,
            sigma_strain=sigma_strain,
        )
    except Exception as e:
        print(f"[ERROR] Likelihood failed for params {params_in}: {e}")
        return -np.inf



def _gaussian_strain_likelihood(L_long_half_s, L_short_half_s, theta_deg_s, x0_s, y0_s, sigma_strain):
    # Map explicit geometry names to the current forward model's a/b inputs.
    a_s, b_s = geometry_to_forward_params(L_long_half_s, L_short_half_s)

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
        c=C_FIXED,
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

    n = resid.size
    log_like = (
        -0.5 * np.sum((resid / sigma_strain) ** 2)
        - n * np.log(sigma_strain)
        - 0.5 * n * np.log(2.0 * np.pi)
    )
    return log_like

# ============================================================
# Priors
# ============================================================

if FIT_GEOMETRY:
    # Explicit geometry with enforced ordering:
    #   L_long_half = L_short_half + delta_half
    L_short_min, L_short_max = 50.0, 250.0
    delta_min, delta_max = 0.0, 300.0
    theta_min, theta_max = -90.0, 90.0
    x0_min, x0_max = -400.0, 600.0
    y0_min, y0_max = -500.0, 500.0
    log10_sigma_min, log10_sigma_max = 0.0, np.log10(2000.0)

    param_priors = [
        SampledParam(uniform, loc=L_short_min, scale=L_short_max - L_short_min),
        SampledParam(uniform, loc=delta_min, scale=delta_max - delta_min),
        SampledParam(uniform, loc=theta_min, scale=theta_max - theta_min),
        SampledParam(uniform, loc=x0_min, scale=x0_max - x0_min),
        SampledParam(uniform, loc=y0_min, scale=y0_max - y0_min),
        SampledParam(uniform, loc=log10_sigma_min, scale=log10_sigma_max - log10_sigma_min),
    ]
    PARAM_NAMES = [
        "L_short_half",
        "delta_half",
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "log10_sigma_strain",
    ]
else:
    theta_min, theta_max = -90.0, 90.0
    x0_min, x0_max = -400.0, 600.0
    y0_min, y0_max = -500.0, 500.0
    log10_sigma_min, log10_sigma_max = 0.0, np.log10(2000.0)

    param_priors = [
        SampledParam(uniform, loc=theta_min, scale=theta_max - theta_min),
        SampledParam(uniform, loc=x0_min, scale=x0_max - x0_min),
        SampledParam(uniform, loc=y0_min, scale=y0_max - y0_min),
        SampledParam(uniform, loc=log10_sigma_min, scale=log10_sigma_max - log10_sigma_min),
    ]
    PARAM_NAMES = ["theta_deg", "x0_prime", "y0_prime", "log10_sigma_strain"]

# ============================================================
# Run control
# ============================================================

MAX_ITER = 50000
BATCH_SIZE = 5000
NCHAINS = 4
R_HAT_THRESH = 1.1
MODEL_NAME = "pydream_multi_station_explicit_geometry"
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

    np.save("explicit_geometry_sampled_params.npy", np.stack(chains_list))
    np.save("explicit_geometry_logps.npy", np.concatenate(logps_list))
    print("[INFO] Posterior samples saved.")

    # ========================================================
    # MAP summary
    # ========================================================

    samples = np.load("explicit_geometry_sampled_params.npy")
    logps = np.load("explicit_geometry_logps.npy")

    flat_samples = samples.reshape(-1, samples.shape[-1])
    map_idx = int(np.argmax(logps))
    map_params = flat_samples[map_idx]

    if FIT_GEOMETRY:
        L_short_map, delta_map, theta_map, x0_map, y0_map, log10_sigma_map = map_params
        L_long_map = L_short_map + delta_map
    else:
        theta_map, x0_map, y0_map, log10_sigma_map = map_params
        L_long_map = L_LONG_HALF_INPUT
        L_short_map = L_SHORT_HALF_INPUT

    sigma_map = 10.0 ** log10_sigma_map

    a_map, b_map = geometry_to_forward_params(L_long_map, L_short_map)

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
        c=C_FIXED,
        nu=nu,
        h=h,
        E=E_fixed,
        theta_deg=theta_map,
        alpha=alpha,
        station_names=station_names,
    )

    pred_vector = flatten_columns(df_pred, COMPONENT_COLS)
    strain_r2 = compute_r2(observed_vector, pred_vector)

    summary = {
        "MAP": {
            "L_long_half": float(L_long_map),
            "L_short_half": float(L_short_map),
            "theta_deg": float(theta_map),
            "x0_prime": float(x0_map),
            "y0_prime": float(y0_map),
            "sigma_strain": float(sigma_map),
        },
        "fixed_inputs": {
            "pmax": pmax,
            "E": E_fixed,
            "c": C_FIXED,
            "nu": nu,
            "h": h,
            "d": d,
            "x0_prime_input": x0_true,
            "y0_prime_input": y0_true,
            "theta_deg_input": theta_true,
            "L_long_half_input": L_LONG_HALF_INPUT,
            "L_short_half_input": L_SHORT_HALF_INPUT,
        },
        "fit": {
            "strain_R2_MAP": float(strain_r2),
        },
        "parameter_names": PARAM_NAMES,
        "FIT_GEOMETRY": bool(FIT_GEOMETRY),
        "geometry_convention": {
            "L_long_half": "mapped to forward-model a",
            "L_short_half": "mapped to forward-model b",
        },
    }

    with open("posterior_summary_explicit_geometry.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("[INFO] Saved posterior_summary_explicit_geometry.json")
    print(f"[INFO] MAP L_long_half: {L_long_map:.3f}")
    print(f"[INFO] MAP L_short_half: {L_short_map:.3f}")
    print(f"[INFO] MAP theta_deg: {theta_map:.3f}")
    print(f"[INFO] MAP x0_prime: {x0_map:.3f}, y0_prime: {y0_map:.3f}")
    print(f"[INFO] MAP sigma_strain: {sigma_map:.3f}")
    print(f"[INFO] Strain R^2: {strain_r2:.4f}")

    # ========================================================
    # Plan-view rectangle plot
    # ========================================================

    import matplotlib.pyplot as plt

    theta_rad = np.radians(theta_map)
    half_w = L_short_map
    half_h = L_long_map

    corners_local = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
        [-half_w, -half_h],
    ])

    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)],
    ])

    corners_rot = (R @ corners_local.T).T
    x_plot = corners_rot[:, 0] + x0_map
    y_plot = corners_rot[:, 1] + y0_map

    plt.figure(figsize=(7, 6))
    plt.plot(x_plot, y_plot, "r-", linewidth=2.5, label="Lens boundary")
    plt.scatter(x_prime, y_prime, c="k", s=60, label="Stations")
    for s, xs, ys in zip(station_names, x_prime, y_prime):
        plt.text(xs + 10, ys + 10, s, fontsize=10)
    plt.scatter([x0_map], [y0_map], marker="x", s=120, c="blue", label="Lens center")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("x' (m)")
    plt.ylabel("y' (m)")
    plt.title("Plan-view explicit rectangle geometry and station layout")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("explicit_geometry_rectangle.png", dpi=200)
    plt.close()
    print("[INFO] Saved explicit_geometry_rectangle.png")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
