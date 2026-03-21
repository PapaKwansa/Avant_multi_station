# geometry_shape_inversion_logspace.py
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
# Fixed best values from prior sweeps
# ============================================================

nu = float(params["nu"])
alpha = float(params["alpha"])

pmax_fixed = 9.75e6
tpeak_fixed = 393333.0
d_fixed = 0.4
E_fixed = 1.0e10
h_fixed = 2400.0
x0_fixed = -208.33333333333337
y0_fixed = 83.33333333333326
theta_fixed = -60.0

print("[INFO] Fixed values:")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       tpeak = {tpeak_fixed:.6g}")
print(f"       d = {d_fixed:.6g}")
print(f"       E = {E_fixed:.6g}")
print(f"       h = {h_fixed:.6g}")
print(f"       x0_prime = {x0_fixed:.6g}")
print(f"       y0_prime = {y0_fixed:.6g}")
print(f"       theta_deg = {theta_fixed:.6g}")

# ============================================================
# Geometry parameterization
#   a = L
#   b = L * rb
#   c = L * rb * rc
#
# Use logs for ratios:
#   log_rb = log(rb)
#   log_rc = log(rc)
# ============================================================

def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def best_scalar_multiplier(pred, obs):
    """
    Least-squares scalar a minimizing ||a*pred - obs||^2.
    Useful for checking amplitude mismatch.
    """
    p = pred.reshape(-1)
    o = obs.reshape(-1)
    denom = np.dot(p, p)
    if denom <= 0:
        return np.nan
    return float(np.dot(p, o) / denom)


def make_predicted_matrix(L, log_rb, log_rc):
    rb = float(np.exp(log_rb))
    rc = float(np.exp(log_rc))

    a = L
    b = L * rb
    c = L * rb * rc

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
        c=c,
        nu=nu,
        h=h_fixed,
        E=E_fixed,
        theta_deg=theta_fixed,
        alpha=alpha,
        station_names=station_names,
        debug=False,
        flip_exy=True,   # forward model handles the sign correction once
    )

    missing_pred = [c for c in model_cols if c not in df_pred.columns]
    if missing_pred:
        raise RuntimeError(f"Forward model missing columns: {missing_pred}")

    return df_pred[model_cols].values.astype(float)


def objective(x):
    L, log_rb, log_rc = x
    pred = make_predicted_matrix(L, log_rb, log_rc)
    return rmse(pred, obs_matrix)


# ============================================================
# Search bounds
# ============================================================
# These are deliberately wider than the first run.
# If rc again hits the top bound, widen it further.
#
# L      : overall size
# rb     : b/a
# rc     : c/b
#
# In log-space:
# log_rb = log(rb)
# log_rc = log(rc)
# ============================================================

bounds = [
    (20.0, 150.0),                 # L
    (np.log(0.08), np.log(0.60)),  # log(rb)
    (np.log(10.0), np.log(300.0)), # log(rc)
]

print("\n[INFO] Starting differential evolution over:")
print("       L, log_rb, log_rc")
print(f"       bounds = {bounds}")

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

L_best, log_rb_best, log_rc_best = result.x
rb_best = float(np.exp(log_rb_best))
rc_best = float(np.exp(log_rc_best))
best_rmse = float(result.fun)

a_best = L_best
b_best = L_best * rb_best
c_best = L_best * rb_best * rc_best

print("\n[RESULT]")
print(f"L       = {L_best:.6g}")
print(f"rb      = {rb_best:.6g}")
print(f"rc      = {rc_best:.6g}")
print(f"a       = {a_best:.6g}")
print(f"b       = {b_best:.6g}")
print(f"c       = {c_best:.6g}")
print(f"RMSE    = {best_rmse:.6g}")

best_pred = make_predicted_matrix(L_best, log_rb_best, log_rc_best)
best_scalar = best_scalar_multiplier(best_pred, obs_matrix)

print(f"best scalar multiplier = {best_scalar:.6g}")

# ============================================================
# Save results
# ============================================================

results = {
    "L_best": float(L_best),
    "rb_best": float(rb_best),
    "rc_best": float(rc_best),
    "a_best": float(a_best),
    "b_best": float(b_best),
    "c_best": float(c_best),
    "rmse": best_rmse,
    "best_scalar": float(best_scalar),
    "bounds": [
        [float(bounds[0][0]), float(bounds[0][1])],
        [float(bounds[1][0]), float(bounds[1][1])],
        [float(bounds[2][0]), float(bounds[2][1])],
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
    },
}

with open("geometry_shape_results_logspace.json", "w") as f:
    json.dump(results, f, indent=2)

print("[INFO] Saved geometry_shape_results_logspace.json")

# ============================================================
# Plot RMSE vs observed/predicted
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
plt.savefig("geometry_shape_best_fit_logspace.png", dpi=200)
plt.close()
print("[INFO] Saved geometry_shape_best_fit_logspace.png")

# ============================================================
# Plot optional RMSE summary against reconstructed dimensions
# ============================================================

plt.figure(figsize=(8, 4))
plt.bar(["a", "b", "c"], [a_best, b_best, c_best])
plt.ylabel("meters")
plt.title("Best reconstructed geometry")
plt.tight_layout()
plt.savefig("geometry_shape_dimensions_logspace.png", dpi=200)
plt.close()
print("[INFO] Saved geometry_shape_dimensions_logspace.png")