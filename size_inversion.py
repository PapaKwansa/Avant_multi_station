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

# --- DEBUG / EXPERIMENT SWITCHES --------------------------------
EYY_FLIP = False
Y_FLIP = False
THETA_COMSOL = 15
# ---------------------------------------------------------------

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

if Y_FLIP:
    print("[DEBUG] Flipping sign of y_prime for all stations")
    y_prime = -y_prime

print("[INFO] Station coordinates (x_prime, y_prime, depth):")
for s, x, y, zz in zip(station_names, x_prime, y_prime, z):
    print(f"   {s}: ({x:.3f}, {y:.3f}, {zz:.3f})")

# --- USER-DEFINED CENTER ------------------------------------
X0_USER = -125.0
Y0_USER = 4.0
# -------------------------------------------------------------

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

pmax_fixed = 0.45e6
tpeak_fixed = float(params.get("tpeak", 393333.0))
d_fixed = float(params.get("d", 0.4))

E_fixed = 2.0e9
h_fixed = 518.29
theta_fixed = float(THETA_COMSOL)

print("[INFO] Fixed values:")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       tpeak = {tpeak_fixed:.6g}")
print(f"       d = {d_fixed:.6g}")
print(f"       E = {E_fixed:.6g}")
print(f"       h = {h_fixed:.6g}")
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

def per_component_rmse(pred, obs, station_names):
    n_comp = 4
    n_stat = len(station_names)
    out = {}
    for ic, comp in enumerate(["eXX", "eYY", "eXY", "eZZ"]):
        out[comp] = {}
        for js, sname in enumerate(station_names):
            col_idx = ic * n_stat + js
            r = rmse(pred[:, col_idx], obs[:, col_idx])
            out[comp][sname] = r
    return out

# ============================================================
# Geometry-only inversion: optimize a, b, c
# ============================================================

def objective_abc(params_abc):
    a_val, b_val, c_val = params_abc

    df_pred = forward_model_multi_station(
        pmax=pmax_fixed,
        tpeak=tpeak_fixed,
        d=d_fixed,
        time=time_vals,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=X0_USER,
        y0_prime=Y0_USER,
        z=z,
        a=a_val,
        b=b_val,
        c=c_val,
        nu=nu,
        h=h_fixed,
        E=E_fixed,
        theta_deg=theta_fixed,
        alpha=alpha,
        station_names=station_names,
        debug=False,
    )

    pred = df_pred[model_cols].values.astype(float)
    return rmse(pred, obs_matrix)

a_bounds = (20.0, 300.0)
b_bounds = (20.0, 250.0)
c_bounds = (0.5, 20.0)

bounds_abc = [a_bounds, b_bounds, c_bounds]

print("\n[INFO] Starting differential evolution for a, b, c...")
print(f"[INFO] bounds = {bounds_abc}")

result_abc = differential_evolution(
    objective_abc,
    bounds=bounds_abc,
    seed=42,
    maxiter=120,
    popsize=12,
    polish=True,
    updating="deferred",
    workers=1,
)

a_best, b_best, c_best = result_abc.x
best_rmse_abc = float(result_abc.fun)

print("\n[RESULT] Geometry-only inversion")
print(f"a_best = {a_best:.6g}")
print(f"b_best = {b_best:.6g}")
print(f"c_best = {c_best:.6g}")
print(f"RMSE = {best_rmse_abc:.6g}")

best_pred_abc = forward_model_multi_station(
    pmax=pmax_fixed,
    tpeak=tpeak_fixed,
    d=d_fixed,
    time=time_vals,
    x_prime=x_prime,
    y_prime=y_prime,
    x0_prime=X0_USER,
    y0_prime=Y0_USER,
    z=z,
    a=a_best,
    b=b_best,
    c=c_best,
    nu=nu,
    h=h_fixed,
    E=E_fixed,
    theta_deg=theta_fixed,
    alpha=alpha,
    station_names=station_names,
    debug=False,
)[model_cols].values.astype(float)

best_scalar_abc = best_scalar_multiplier(best_pred_abc, obs_matrix)

print(f"best scalar multiplier = {best_scalar_abc:.6g}")

pc_rmse_best = per_component_rmse(best_pred_abc, obs_matrix, station_names)

# ============================================================
# Save results
# ============================================================

results = {
    "a_best": float(a_best),
    "b_best": float(b_best),
    "c_best": float(c_best),
    "rmse": float(best_rmse_abc),
    "best_scalar": float(best_scalar_abc),
    "x0_prime": float(X0_USER),
    "y0_prime": float(Y0_USER),
    "fixed_values": {
        "pmax": pmax_fixed,
        "tpeak": tpeak_fixed,
        "d": d_fixed,
        "E": E_fixed,
        "h": h_fixed,
        "theta_deg": theta_fixed,
        "nu": nu,
        "alpha": alpha,
    },
    "bounds": bounds_abc,
    "per_component_rmse_best": pc_rmse_best,
}

with open("location_fit_xy_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("[INFO] Saved location_fit_xy_results.json")

# ============================================================
# Plot best fit
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.linewidth": 1.8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 200,
})

fig, axes = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)

for i in range(16):
    ax = axes.flat[i]

    ax.plot(time_vals, obs_matrix[:, i], "k.", markersize=3,
            label="Observed" if i == 0 else None)
    ax.plot(time_vals, best_pred_abc[:, i], color="tab:red", linewidth=2.2,
            label="Predicted" if i == 0 else None)

    ax.set_title(model_cols[i], fontsize=12, pad=6)
    ax.grid(True, alpha=0.25)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           fontsize=14, bbox_to_anchor=(0.5, -0.02))

plt.savefig("location_fit_xy_best_fit.png", dpi=350, bbox_inches="tight")
plt.close()
print("[INFO] Saved location_fit_xy_best_fit.png")

# ============================================================
# Plot optimized lens geometry
# ============================================================

theta_rad = np.radians(theta_fixed)

t = np.linspace(0, 2*np.pi, 400)
x_ell = a_best * np.cos(t)
y_ell = b_best * np.sin(t)

R = np.array([
    [np.cos(theta_rad), -np.sin(theta_rad)],
    [np.sin(theta_rad),  np.cos(theta_rad)]
])

xy_rot = R @ np.vstack([x_ell, y_ell])

x_plot = xy_rot[0, :] + X0_USER
y_plot = xy_rot[1, :] + Y0_USER

plt.figure(figsize=(7, 6))
plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Optimized lens boundary')
plt.scatter(x_prime, y_prime, c='k', s=60, label='Stations')

for s, xs, ys in zip(station_names, x_prime, y_prime):
    plt.text(xs + 10, ys + 10, s, fontsize=10)

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

plt.xlabel("x' (m)")
plt.ylabel("y' (m)")
plt.title("Optimized Lens Geometry and Station Layout")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lens_geometry_planview.png", dpi=200)
plt.close()

print("[INFO] Saved lens_geometry_planview.png")
