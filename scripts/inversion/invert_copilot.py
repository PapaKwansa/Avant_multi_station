# UPDATED geometry_shape_inversion.py - 5 PARAMETER INVERSION
# This version inverts for a, b, c, x0, y0 simultaneously

import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as input_data

print('[FULL GEOMETRY INVERSION] Inverting: a, b, c, x0, y0')

# Load data
params = input_data.read_input()
stations_df = pd.read_csv('AVANT_stations.csv')
stations_df["station"] = stations_df["station"].astype(str).str.strip()
station_names = stations_df["station"].values
x_prime = stations_df["x_prime"].values.astype(float)
y_prime = stations_df["y_prime"].values.astype(float)
z = stations_df["depth"].values.astype(float)

obs_df = pd.read_csv('Avant_cleaned_strain.csv').drop_duplicates().reset_index(drop=True)
time_vals = obs_df["time_s"].values.astype(float)

model_cols = [f"{comp}_{sname}" for comp in ["eXX", "eYY", "eXY", "eZZ"] for sname in station_names]
non_time_cols = [c for c in obs_df.columns if c != "time_s"]
obs_cols = model_cols if all(c in obs_df.columns for c in model_cols) else non_time_cols
obs_matrix = obs_df[obs_cols].values.astype(float)

# FIXED parameters
nu = float(params["nu"])
alpha = float(params["alpha"])
pmax = float(params.get("pmax", 9.75e6))
tpeak = float(params.get("tpeak", 393333.0))
d = float(params.get("d", 0.4))
E_fixed = float(params.get("E", 1.0e10))
h_fixed = float(params.get("h", 2400.0))
theta_fixed = float(params.get("theta_deg", -15.0))

print(f'Data: {obs_matrix.shape}, range [{np.min(obs_matrix):.2e}, {np.max(obs_matrix):.2e}]')
print(f'Thickness constraint: 5 meters (c ≤ 2.5 m)')

def rmse(pred, obs):
    return np.sqrt(np.mean((pred - obs) ** 2))

def make_prediction(a, b, c, x0, y0):
    df_pred = forward_model_multi_station(
        pmax=pmax, tpeak=tpeak, d=d, time=time_vals,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0, y0_prime=y0, z=z,
        a=a, b=b, c=c,
        nu=nu, h=h_fixed, E=E_fixed, theta_deg=theta_fixed, alpha=alpha,
        station_names=station_names, debug=False,
    )
    return df_pred[obs_cols].values.astype(float)

def objective(x):
    a_val, b_val, c_val, x0_val, y0_val = x
    try:
        pred = make_prediction(a_val, b_val, c_val, x0_val, y0_val)
        return rmse(pred, obs_matrix)
    except:
        return np.inf

# OPTIMIZATION BOUNDS
# Match thickness to 5m total, so c <= 2.5
bounds = [
    (5.0, 15000.0),      # a: semi-major axis (5-1000 m)
    (0.5, 80000.0),      # b: semi-minor axis (0.5-200 m)
    (0.5, 2.5),        # c: semi-thickness (0.5-2.5 m, so full 1-5 m)
    (-500.0, 500.0),   # x0: center x (-500 to 500 m)
    (-500.0, 500.0),   # y0: center y (-500 to 500 m)
]

print('\n[OPTIMIZATION] Running differential evolution for 5 parameters...')
print(f'Bounds: a={bounds[0]}, b={bounds[1]}, c={bounds[2]}, x0={bounds[3]}, y0={bounds[4]}')

result = differential_evolution(
    objective,
    bounds=bounds,
    seed=42,
    maxiter=500,  # Increased iterations
    popsize=40,   # Increased population
    atol=1e-8,
    tol=1e-8,
    polish=True,
    updating="deferred",
    workers=1,
)

a_best, b_best, c_best, x0_best, y0_best = result.x

print(f'\n[RESULT]')
print(f'a = {a_best:.6e} m')
print(f'b = {b_best:.6e} m')
print(f'c = {c_best:.6e} m  (full thickness = {2*c_best:.6e} m)')
print(f'x0 = {x0_best:.6e} m')
print(f'y0 = {y0_best:.6e} m')
print(f'RMSE = {result.fun:.6e}')
print(f'b/a ratio = {b_best/a_best:.6f}')

best_pred = make_prediction(a_best, b_best, c_best, x0_best, y0_best)
pred_flat = best_pred.reshape(-1)
obs_flat = obs_matrix.reshape(-1)
scalar = np.dot(pred_flat, obs_flat) / (np.dot(pred_flat, pred_flat) + 1e-10)
print(f'Best scalar multiplier = {scalar:.6f}')

# Save
results = {
    "a_best": float(a_best),
    "b_best": float(b_best),
    "c_best": float(c_best),
    "thickness_full": float(2*c_best),
    "x0_best": float(x0_best),
    "y0_best": float(y0_best),
    "rmse": float(result.fun),
    "b_over_a": float(b_best/a_best),
    "scalar": float(scalar),
}

with open("geometry_5param_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[SAVED] geometry_5param_results.json")

# Plot
fig, axes = plt.subplots(4, 4, figsize=(14, 10))
for i in range(min(16, obs_matrix.shape[1])):
    ax = axes[i // 4, i % 4]
    ax.plot(time_vals, obs_matrix[:, i], "k.", markersize=2, label="COMSOL", alpha=0.6)
    ax.plot(time_vals, best_pred[:, i], "r-", linewidth=1.5, label="Best")
    ax.set_title(f"Ch{i}", fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("geometry_5param_fit.png", dpi=150)
plt.close()
print("[SAVED] geometry_5param_fit.png")