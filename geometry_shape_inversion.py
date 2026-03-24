# geometry_shape_inversion_fixed_thickness.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as input_data

# ============================================================
# Setup
# ============================================================

np.random.seed(42)

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

params = input_data.read_input()

# ============================================================
# Load stations
# ============================================================

stations_df = pd.read_csv(STATION_FILE)
stations_df["station"] = stations_df["station"].astype(str).str.strip()

required_cols = ["station", "x_prime", "y_prime", "depth"]
missing = set(required_cols) - set(stations_df.columns)
if missing:
    raise RuntimeError(f"Missing columns in AVANT_stations.csv: {missing}")

station_names = stations_df["station"].values
x_prime = stations_df["x_prime"].values.astype(float)
y_prime = stations_df["y_prime"].values.astype(float)
z = stations_df["depth"].values.astype(float)

# ============================================================
# Load observed data
# ============================================================

obs_df = pd.read_csv(OBSERVED_FILE).drop_duplicates().reset_index(drop=True)

if "time_s" not in obs_df.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

time_vals = obs_df["time_s"].values.astype(float)

model_cols = [
    f"{comp}_{sname}"
    for comp in ["eXX", "eYY", "eXY", "eZZ"]
    for sname in station_names
]

non_time_cols = [c for c in obs_df.columns if c != "time_s"]

if all(c in obs_df.columns for c in model_cols):
    obs_cols = model_cols
elif len(non_time_cols) == len(model_cols):
    obs_cols = non_time_cols
else:
    raise RuntimeError(
        f"Observed layout mismatch. Found {len(non_time_cols)} non-time columns, "
        f"expected {len(model_cols)}."
    )

obs_matrix = obs_df[obs_cols].values.astype(float)

print(f"[INFO] Observed matrix shape: {obs_matrix.shape}")
print(f"[INFO] Using observed columns: {obs_cols[:4]} ... {obs_cols[-4:]}")
print(f"[INFO] Observed min/max: {np.min(obs_matrix):.6g} / {np.max(obs_matrix):.6g}")
print(f"[INFO] Observed mean abs: {np.mean(np.abs(obs_matrix)):.6g}")
print(f"[INFO] Observed std: {np.std(obs_matrix):.6g}")

# ============================================================
# Fixed physical values
# ============================================================

nu = float(params["nu"])
alpha = float(params["alpha"])

# COMSOL-based values
pmax_fixed = 1.0e6
tpeak_fixed = float(params.get("tpeak", 393333.0))
d_fixed = float(params.get("d", 0.4))
E_fixed = float(params.get("E", 1.0e10))
h_fixed = 518.29
x0_fixed = float(params.get("x0_prime", 0.0))
y0_fixed = float(params.get("y0_prime", 0.0))
theta_fixed = -15.0

# FIX thickness:
# current forward kernel uses z ± c, so c is semi-thickness
lens_thickness_full = 5.0
c_fixed = lens_thickness_full / 2.0
# If your convention is that c is the full thickness, use:
# c_fixed = 5.0

flip_exy = False

print("[INFO] Fixed values:")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       tpeak = {tpeak_fixed:.6g}")
print(f"       d = {d_fixed:.6g}")
print(f"       E = {E_fixed:.6g}")
print(f"       h = {h_fixed:.6g}")
print(f"       x0_prime = {x0_fixed:.6g}")
print(f"       y0_prime = {y0_fixed:.6g}")
print(f"       theta_deg = {theta_fixed:.6g}")
print(f"       lens_thickness_full = {lens_thickness_full:.6g}")
print(f"       c_fixed = {c_fixed:.6g}")
print(f"       flip_exy = {flip_exy}")

# ============================================================
# Helpers
# ============================================================

def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))

def best_scalar_multiplier(pred, obs):
    p = pred.reshape(-1)
    o = obs.reshape(-1)
    denom = np.dot(p, p)
    if denom <= 0:
        return np.nan
    return float(np.dot(p, o) / denom)

def make_predicted_matrix(a, b):
    df_pred = forward_model_multi_station(
        pmax=pmax_fixed,
        tpeak=tpeak_fixed,
        d=d_fixed,
        time=time_vals,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_fixed,
        y0_prime=y0_fixed,
        z=z,
        a=a,
        b=b,
        c=c_fixed,
        nu=nu,
        h=h_fixed,
        E=E_fixed,
        theta_deg=theta_fixed,
        alpha=alpha,
        station_names=station_names,
        debug=False,
    )

    missing_pred = [c for c in model_cols if c not in df_pred.columns]
    if missing_pred:
        raise RuntimeError(f"Forward model missing columns: {missing_pred}")

    return df_pred[model_cols].values.astype(float)

# ============================================================
# Baseline diagnostic
# ============================================================

a_guess = 40.0
b_guess = 8.0

baseline_pred = make_predicted_matrix(a_guess, b_guess)

print("\n[INFO] Baseline diagnostic")
print(f"[INFO] baseline a = {a_guess:.6g}")
print(f"[INFO] baseline b = {b_guess:.6g}")
print(f"[INFO] baseline pred min/max: {np.min(baseline_pred):.6g} / {np.max(baseline_pred):.6g}")
print(f"[INFO] baseline pred mean abs: {np.mean(np.abs(baseline_pred)):.6g}")
print(f"[INFO] baseline RMSE: {rmse(baseline_pred, obs_matrix):.6g}")
print(f"[INFO] baseline best scalar a: {best_scalar_multiplier(baseline_pred, obs_matrix):.6g}")

# ============================================================
# Optimize a and b
# ============================================================

def objective(x):
    a_val, b_val = x
    pred = make_predicted_matrix(a_val, b_val)
    return rmse(pred, obs_matrix)

bounds = [
    (1.0, 20000.0),   # a
    (1.0, 80000.0),    # b
]

print("\n[INFO] Starting differential evolution over a and b...")
print(f"[INFO] bounds = {bounds}")

result = differential_evolution(
    objective,
    bounds=bounds,
    seed=42,
    maxiter=120,
    popsize=12,
    polish=True,
    updating="deferred",
    workers=1,
)

a_best, b_best = result.x
best_rmse = float(result.fun)
best_pred = make_predicted_matrix(a_best, b_best)
best_scalar = best_scalar_multiplier(best_pred, obs_matrix)

print("\n[RESULT]")
print(f"a = {a_best:.6g}")
print(f"b = {b_best:.6g}")
print(f"c = {c_fixed:.6g}")
print(f"b/a = {b_best / a_best:.6g}")
print(f"RMSE = {best_rmse:.6g}")
print(f"best scalar multiplier = {best_scalar:.6g}")

# ============================================================
# Save results
# ============================================================

results = {
    "a_best": float(a_best),
    "b_best": float(b_best),
    "c_fixed": float(c_fixed),
    "b_over_a": float(b_best / a_best),
    "rmse": float(best_rmse),
    "best_scalar": float(best_scalar),
    "baseline": {
        "a": float(a_guess),
        "b": float(b_guess),
    },
    "bounds": [
        [float(bounds[0][0]), float(bounds[0][1])],
        [float(bounds[1][0]), float(bounds[1][1])],
    ],
    "fixed_values": {
        "pmax": pmax_fixed,
        "tpeak": tpeak_fixed,
        "d": d_fixed,
        "E": E_fixed,
        "h": h_fixed,
        "x0_prime": x0_fixed,
        "y0_prime": y0_fixed,
        "theta_deg": theta_fixed,
        "nu": nu,
        "alpha": alpha,
        "lens_thickness_full": lens_thickness_full,
    },
}

with open("geometry_shape_results_fixed_thickness.json", "w") as f:
    json.dump(results, f, indent=2)

print("[INFO] Saved geometry_shape_results_fixed_thickness.json")

# ============================================================
# Plot best fit
# ============================================================

plt.figure(figsize=(16, 12))
for i in range(min(16, obs_matrix.shape[1])):
    ax = plt.subplot(4, 4, i + 1)
    ax.plot(time_vals, obs_matrix[:, i], "k.", markersize=2, label="obs")
    ax.plot(time_vals, best_pred[:, i], "r-", linewidth=1.2, label="pred")
    ax.set_title(model_cols[i], fontsize=9)
    ax.grid(True, alpha=0.25)
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("geometry_shape_best_fit_fixed_thickness.png", dpi=200)
plt.close()
print("[INFO] Saved geometry_shape_best_fit_fixed_thickness.png")

# ============================================================
# Plot reconstructed dimensions
# ============================================================

plt.figure(figsize=(8, 4))
plt.bar(["a", "b", "c"], [a_best, b_best, c_fixed])
plt.ylabel("meters")
plt.title("Best reconstructed geometry with fixed thickness")
plt.tight_layout()
plt.savefig("geometry_shape_dimensions_fixed_thickness.png", dpi=200)
plt.close()
print("[INFO] Saved geometry_shape_dimensions_fixed_thickness.png")