# location_fit_xy.py
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

print(f"[INFO] Stations: {station_names}")
print(f"[INFO] Observed matrix shape: {obs_matrix.shape}")
print(f"[INFO] Using observed columns: {obs_cols[:4]} ... {obs_cols[-4:]}")
print(f"[INFO] Observed min/max: {np.min(obs_matrix):.6g} / {np.max(obs_matrix):.6g}")
print(f"[INFO] Observed mean abs: {np.mean(np.abs(obs_matrix)):.6g}")
print(f"[INFO] Observed std: {np.std(obs_matrix):.6g}")

# ============================================================
# Fixed physical values
# ============================================================

nu = float(params.get("nu", 0.25))
alpha = float(params.get("alpha", 0.8))

pmax_fixed = 1.0e6
tpeak_fixed = float(params.get("tpeak", 393333.0))
d_fixed = float(params.get("d", 0.4))

E_fixed = 2.0e9
h_fixed = 518.29
theta_fixed = -345.0

# Fixed lens geometry
a_fixed = 150.0
b_fixed = 2.5
c_fixed = 290.0

print("[INFO] Fixed values:")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       tpeak = {tpeak_fixed:.6g}")
print(f"       d = {d_fixed:.6g}")
print(f"       E = {E_fixed:.6g}")
print(f"       h = {h_fixed:.6g}")
print(f"       a = {a_fixed:.6g}")
print(f"       b = {b_fixed:.6g}")
print(f"       c = {c_fixed:.6g}")
print(f"       theta_deg = {theta_fixed:.6g}")

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

def make_predicted_matrix(x0_prime, y0_prime):
    df_pred = forward_model_multi_station(
        pmax=pmax_fixed,
        tpeak=tpeak_fixed,
        d=d_fixed,
        time=time_vals,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_prime,
        y0_prime=y0_prime,
        z=z,
        a=a_fixed,
        b=b_fixed,
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

x0_guess = float(params.get("x0_prime", np.mean(x_prime)))
y0_guess = float(params.get("y0_prime", np.mean(y_prime)))

baseline_pred = make_predicted_matrix(x0_guess, y0_guess)

print("\n[INFO] Baseline diagnostic")
print(f"[INFO] baseline x0_prime = {x0_guess:.6g}")
print(f"[INFO] baseline y0_prime = {y0_guess:.6g}")
print(f"[INFO] baseline pred min/max: {np.min(baseline_pred):.6g} / {np.max(baseline_pred):.6g}")
print(f"[INFO] baseline pred mean abs: {np.mean(np.abs(baseline_pred)):.6g}")
print(f"[INFO] baseline RMSE: {rmse(baseline_pred, obs_matrix):.6g}")
print(f"[INFO] baseline best scalar a: {best_scalar_multiplier(baseline_pred, obs_matrix):.6g}")

# ============================================================
# Optimize x0 and y0
# ============================================================

def objective(x):
    x0_prime, y0_prime = x
    pred = make_predicted_matrix(x0_prime, y0_prime)
    return rmse(pred, obs_matrix)

# Broad bounds for the inclusion center
x_bounds = (min(x_prime) - 2000.0, max(x_prime) + 2000.0)
y_bounds = (min(y_prime) - 2000.0, max(y_prime) + 2000.0)

bounds = [x_bounds, y_bounds]

print("\n[INFO] Starting differential evolution over x0_prime and y0_prime...")
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

x0_best, y0_best = result.x
best_rmse = float(result.fun)
best_pred = make_predicted_matrix(x0_best, y0_best)
best_scalar = best_scalar_multiplier(best_pred, obs_matrix)

print("\n[RESULT]")
print(f"x0_prime = {x0_best:.6g}")
print(f"y0_prime = {y0_best:.6g}")
print(f"RMSE = {best_rmse:.6g}")
print(f"best scalar multiplier = {best_scalar:.6g}")

# ============================================================
# Save results
# ============================================================

results = {
    "x0_prime_best": float(x0_best),
    "y0_prime_best": float(y0_best),
    "rmse": float(best_rmse),
    "best_scalar": float(best_scalar),
    "baseline": {
        "x0_prime": float(x0_guess),
        "y0_prime": float(y0_guess),
    },
    "fixed_values": {
        "pmax": pmax_fixed,
        "tpeak": tpeak_fixed,
        "d": d_fixed,
        "E": E_fixed,
        "h": h_fixed,
        "a": a_fixed,
        "b": b_fixed,
        "c": c_fixed,
        "theta_deg": theta_fixed,
        "nu": nu,
        "alpha": alpha,
    },
    "bounds": [
        [float(bounds[0][0]), float(bounds[0][1])],
        [float(bounds[1][0]), float(bounds[1][1])],
    ],
}

with open("location_fit_xy_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("[INFO] Saved location_fit_xy_results.json")

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
plt.savefig("location_fit_xy_best_fit.png", dpi=200)
plt.close()
print("[INFO] Saved location_fit_xy_best_fit.png")

# ============================================================
# Plot center result
# ============================================================

plt.figure(figsize=(6, 5))
plt.scatter(x_prime, y_prime, label="stations")
plt.scatter([x0_best], [y0_best], marker="x", s=120, label="best center")
plt.xlabel("x_prime")
plt.ylabel("y_prime")
plt.title("Best inclusion center")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("location_fit_xy_center.png", dpi=200)
plt.close()
print("[INFO] Saved location_fit_xy_center.png")