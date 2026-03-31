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
EYY_FLIP = False    # set True to test flipping sign of model eYY
Y_FLIP = False     # set True to test y_prime = -y_prime
THETA_COMSOL = 15  # COMSOL-based tilt angle in degrees
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
X0_USER = -125.0   # <-- you type your chosen x0 here
Y0_USER = 4.0    # <-- you type your chosen y0 here
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

pmax_fixed = 2.0e6
tpeak_fixed = float(params.get("tpeak", 393333.0))
d_fixed = float(params.get("d", 0.4))

E_fixed = 2.0e9
h_fixed = 518.29
theta_fixed = 0.0 if THETA_COMSOL is None else float(THETA_COMSOL)

# Fixed lens geometry
a_fixed = 60.0   # short axis / 2
b_fixed = 114.0   # long axis / 2
c_fixed = 2.5     # thickness / 2

print("[INFO] Fixed values:")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       tpeak = {tpeak_fixed:.6g}")
print(f"       d = {d_fixed:.6g}")
print(f"       E = {E_fixed:.6g}")
print(f"       h = {h_fixed:.6g}")
print(f"       a = {a_fixed:.6g}")
print(f"       b = {b_fixed:.6g}")
print(f"       c = {c_fixed:.6g}")
print(f"       theta_deg (initial) = {theta_fixed:.6g}")

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

def make_predicted_matrix(E_val):
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
        a=a_fixed,
        b=b_fixed,
        c=c_fixed,
        nu=nu,
        h=h_fixed,
        E=E_val,
        theta_deg=theta_fixed,
        alpha=alpha,
        station_names=station_names,
        debug=False,
    )
    # Optional: flip sign of eYY components to test convention
    if EYY_FLIP:
        eyy_cols = [c for c in df_pred.columns if c.startswith("eYY_")]
        df_pred[eyy_cols] = -df_pred[eyy_cols]

    missing_pred = [c for c in model_cols if c not in df_pred.columns]
    if missing_pred:
        raise RuntimeError(f"Forward model missing columns: {missing_pred}")

    return df_pred[model_cols].values.astype(float)

def per_component_rmse(pred, obs, station_names):
    """Compute RMSE per component (eXX/eYY/eXY/eZZ) and per station."""
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
# Baseline diagnostic
# ============================================================

baseline_pred = make_predicted_matrix(E_fixed)

a_best = best_scalar_multiplier(baseline_pred, obs_matrix)
rmse_rescaled = rmse(a_best * baseline_pred, obs_matrix)

print(f"[INFO] baseline best scalar a = {a_best:.6g}")
print(f"[INFO] baseline RMSE (after rescaling) = {rmse_rescaled:.6g}")

print("\n[INFO] Baseline diagnostic")
print(f"[INFO] center fixed at (0,0)")
print(f"[INFO] baseline pred min/max: {np.min(baseline_pred):.6g} / {np.max(baseline_pred):.6g}")
print(f"[INFO] baseline pred mean abs: {np.mean(np.abs(baseline_pred)):.6g}")
print(f"[INFO] baseline RMSE: {rmse(baseline_pred, obs_matrix):.6g}")
print(f"[INFO] baseline best scalar a: {best_scalar_multiplier(baseline_pred, obs_matrix):.6g}")

# --- Per-component / per-station RMSE ------------------------
pc_rmse = per_component_rmse(baseline_pred, obs_matrix, station_names)
print("\n[INFO] Per-component RMSE (baseline):")
for comp, dct in pc_rmse.items():
    line = ", ".join(f"{s}: {v:.2f}" for s, v in dct.items())
    print(f"   {comp}: {line}")

# --- Focus on eYY at S2 and S3 --------------------------------
try:
    idx_eYY_S2 = model_cols.index("eYY_S2")
    idx_eYY_S3 = model_cols.index("eYY_S3")
    print("\n[DEBUG] eYY_S2 (obs min/max): "
          f"{obs_matrix[:, idx_eYY_S2].min():.3f} / {obs_matrix[:, idx_eYY_S2].max():.3f}")
    print("[DEBUG] eYY_S2 (pred min/max): "
          f"{baseline_pred[:, idx_eYY_S2].min():.3f} / {baseline_pred[:, idx_eYY_S2].max():.3f}")
    print("[DEBUG] eYY_S3 (obs min/max): "
          f"{obs_matrix[:, idx_eYY_S3].min():.3f} / {obs_matrix[:, idx_eYY_S3].max():.3f}")
    print("[DEBUG] eYY_S3 (pred min/max): "
          f"{baseline_pred[:, idx_eYY_S3].min():.3f} / {baseline_pred[:, idx_eYY_S3].max():.3f}")
except ValueError:
    print("[WARN] Could not find eYY_S2 or eYY_S3 in model_cols")

# --- Focus on S4 all components --------------------------------
try:
    idx_eXX_S4 = model_cols.index("eXX_S4")
    idx_eYY_S4 = model_cols.index("eYY_S4")
    idx_eXY_S4 = model_cols.index("eXY_S4")
    idx_eZZ_S4 = model_cols.index("eZZ_S4")

    print("\n[DEBUG] S4 observed vs predicted ranges:")
    for name, idx in [("eXX_S4", idx_eXX_S4),
                      ("eYY_S4", idx_eYY_S4),
                      ("eXY_S4", idx_eXY_S4),
                      ("eZZ_S4", idx_eZZ_S4)]:
        print(f"   {name} obs min/max: "
              f"{obs_matrix[:, idx].min():.3f} / {obs_matrix[:, idx].max():.3f}")
        print(f"   {name} pred min/max: "
              f"{baseline_pred[:, idx].min():.3f} / {baseline_pred[:, idx].max():.3f}")
except ValueError:
    print("[WARN] Could not find S4 components in model_cols")


# ============================================================
# Optimize E and theta
# ============================================================

def objective(E_array):
    E_val = E_array[0]
    pred = make_predicted_matrix(E_val)
    return rmse(pred, obs_matrix)

E_bounds = (1e9, 3e10)        # Pa
if THETA_COMSOL is None:
    theta_bounds = (30.0, 120.0)  # free rotation
else:
    # narrow bounds around COMSOL tilt, e.g. ±15°
    theta_bounds = (THETA_COMSOL - 15.0, THETA_COMSOL + 15.0)

bounds = [E_bounds]

print("\n[INFO] Starting differential evolution over E and theta...")
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

E_best = result.x[0]
theta_best = theta_fixed   # because θ is no longer inverted
best_rmse = float(result.fun)
best_pred = make_predicted_matrix(E_best)
best_scalar = best_scalar_multiplier(best_pred, obs_matrix)

print("\n[RESULT]")
print(f"E = {E_best:.6g}")
print(f"theta_deg = {theta_best:.6g}")
print(f"RMSE = {best_rmse:.6g}")
print(f"best scalar multiplier = {best_scalar:.6g}")

pc_rmse_best = per_component_rmse(best_pred, obs_matrix, station_names)
print("\n[INFO] Per-component RMSE (best fit):")
for comp, dct in pc_rmse_best.items():
    line = ", ".join(f"{s}: {v:.2f}" for s, v in dct.items())
    print(f"   {comp}: {line}")

# ============================================================
# Save results
# ============================================================

results = {
    "E_best": float(E_best),
    "theta_deg_best": float(theta_best),
    "rmse": float(best_rmse),
    "best_scalar": float(best_scalar),
    # include user-defined center for reference
    "x0_prime": float(X0_USER),
    "y0_prime": float(Y0_USER),
    "fixed_center": [0.0, 0.0],
    "fixed_values": {
        "pmax": pmax_fixed,
        "tpeak": tpeak_fixed,
        "d": d_fixed,
        "E_initial": E_fixed,
        "h": h_fixed,
        "a": a_fixed,
        "b": b_fixed,
        "c": c_fixed,
        "theta_initial": theta_fixed,
        "nu": nu,
        "alpha": alpha,
    },
   "bounds": [
    [float(E_bounds[0]), float(E_bounds[1])]
    
    ],

    "Y_FLIP": bool(Y_FLIP),
    "THETA_COMSOL": None if THETA_COMSOL is None else float(THETA_COMSOL),
    "per_component_rmse_baseline": pc_rmse,
    "per_component_rmse_best": pc_rmse_best,
    }

with open("location_fit_xy_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("[INFO] Saved location_fit_xy_results.json")

# ============================================================
# Plot best fit
# ============================================================

# ============================================================
# High‑quality best‑fit plot (replaces old block)
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

    # Observed vs predicted
    ax.plot(
        time_vals,
        obs_matrix[:, i],
        "k.",
        markersize=3,
        label="Observed" if i == 0 else None,
    )
    ax.plot(
        time_vals,
        best_pred[:, i],
        color="tab:red",
        linewidth=2.2,
        label="Predicted" if i == 0 else None,
    )

    ax.set_title(model_cols[i], fontsize=12, pad=6)
    ax.grid(True, alpha=0.25)

# One shared legend
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=14,
    bbox_to_anchor=(0.5, -0.02),
)

plt.savefig("location_fit_xy_best_fit.png", dpi=350, bbox_inches="tight")
plt.close()
print("[INFO] Saved high‑quality location_fit_xy_best_fit.png")


# ============================================================
# Plot center (fixed)
# ============================================================

plt.figure(figsize=(6, 5))
plt.scatter(x_prime, y_prime, label="stations")
plt.scatter([0.0], [0.0], marker="x", s=120, label="fixed center (0,0)")
plt.xlabel("x_prime")
plt.ylabel("y_prime")
plt.title("Inclusion center (fixed at 0,0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("location_fit_xy_center.png")
plt.close()
print("[INFO] Saved location_fit_xy_center.png")


# ============================================================
# Plot lens geometry in plan view
# ============================================================

# Use best-fit center if you inverted for it; otherwise (0,0)
x0 = 0.0
y0 = 0.0

# If you have best-fit center from another script, insert here:
# x0 = x0_best
# y0 = y0_best

theta_rad = np.radians(theta_best)

# Parametric ellipse
t = np.linspace(0, 2*np.pi, 400)
x_ell = a_fixed * np.cos(t)
y_ell = b_fixed * np.sin(t)

# Rotation matrix
R = np.array([
    [np.cos(theta_rad), -np.sin(theta_rad)],
    [np.sin(theta_rad),  np.cos(theta_rad)]
])

xy_rot = R @ np.vstack([x_ell, y_ell])

x_plot = xy_rot[0, :] + x0
y_plot = xy_rot[1, :] + y0

plt.figure(figsize=(7, 6))
plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Lens boundary')

# Plot stations
plt.scatter(x_prime, y_prime, c='k', s=60, label='Stations')

# Label stations
for s, xs, ys in zip(station_names, x_prime, y_prime):
    plt.text(xs + 10, ys + 10, s, fontsize=10)

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

plt.xlabel("x' (m)")
plt.ylabel("y' (m)")
plt.title("Plan-view lens geometry and station layout")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lens_geometry_planview.png", dpi=200)
plt.close()

print("[INFO] Saved lens_geometry_planview.png")
