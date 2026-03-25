import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as inp


# ============================================================
# Load inputs
# ============================================================

params = inp.read_input()

BASE_DIR = os.path.dirname(__file__)
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

obs_df = pd.read_csv(OBSERVED_FILE).drop_duplicates().reset_index(drop=True)

if "time_s" not in obs_df.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

t_obs = obs_df["time_s"].values.astype(float)

# ============================================================
# Pull fixed values from the input file with safe fallbacks
# ============================================================

pmax = float(params.get("pmax", 1.0e6))
tpeak = float(params.get("tpeak", 1.0))
d = float(params.get("d", 1.0))

nu = float(params.get("nu", 0.25))
alpha = float(params.get("alpha", 0.8))

x0_prime = float(params.get("x0_prime", 0.0))
y0_prime = float(params.get("y0_prime", 0.0))
h = float(params.get("h", 518.29))

# Geometry
# Prefer direct fixed geometry if available.
if all(k in params for k in ["a_fixed", "b_fixed", "c_fixed"]):
    a = float(params["a_fixed"])
    b = float(params["b_fixed"])
    c = float(params["c_fixed"])
elif all(k in params for k in ["a", "b", "c"]):
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
else:
    raise KeyError(
        "Could not find fixed geometry in input. Expected a_fixed, b_fixed, c_fixed "
        "or a, b, c."
    )

# Material / orientation
E = float(params.get("E", params.get("E_fixed", 1.0e10)))
theta_deg = float(params.get("theta_deg", params.get("theta_deg_start", -15.0)))

# Station metadata
stations = params.get("station_names", None)
if stations is None:
    raise KeyError("station_names not found in input script")

stations = list(stations)

x_prime = np.asarray(params.get("x_prime", None), dtype=float)
y_prime = np.asarray(params.get("y_prime", None), dtype=float)
z = np.asarray(params.get("z", None), dtype=float)

if x_prime is None or y_prime is None or z is None:
    raise KeyError("x_prime, y_prime, and z must be present in the input script")

print(f"[INFO] Stations: {stations}")
print(f"[INFO] Fixed geometry: a={a:.6g}, b={b:.6g}, c={c:.6g}")
print(f"[INFO] Fixed values: pmax={pmax:.6g}, tpeak={tpeak:.6g}, d={d:.6g}, h={h:.6g}")
print(f"[INFO] Material/orientation: E={E:.6g}, theta_deg={theta_deg:.6g}")

# ============================================================
# Observed data columns
# ============================================================

# Expected model columns: eXX_S1, eXX_S2, ... eZZ_S4
model_cols = [
    f"{comp}_{sname}"
    for comp in ["eXX", "eYY", "eXY", "eZZ"]
    for sname in stations
]

non_time_cols = [c for c in obs_df.columns if c != "time_s"]

if all(c in obs_df.columns for c in model_cols):
    obs_cols = model_cols
elif len(non_time_cols) == len(model_cols):
    # Fall back to the order in the file if it already has 16 strain columns
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
# Forward model prediction
# ============================================================

pred_df = forward_model_multi_station(
    pmax=pmax,
    tpeak=tpeak,
    d=d,
    time=t_obs,
    x_prime=x_prime,
    y_prime=y_prime,
    x0_prime=x0_prime,
    y0_prime=y0_prime,
    z=z,
    a=a,
    b=b,
    c=c,
    nu=nu,
    h=h,
    E=E,
    theta_deg=theta_deg,
    alpha=alpha,
    station_names=stations,
    debug=False,
)

missing_pred = [c for c in model_cols if c not in pred_df.columns]
if missing_pred:
    raise RuntimeError(f"Predicted dataframe missing columns: {missing_pred}")

pred_matrix = pred_df[model_cols].values.astype(float)

# ============================================================
# Metrics
# ============================================================

rmse = float(np.sqrt(np.mean((pred_matrix - obs_matrix) ** 2)))
mae_per_time = np.mean(np.abs(obs_matrix - pred_matrix), axis=1)
mae_overall = float(np.mean(np.abs(obs_matrix - pred_matrix)))

print(f"[INFO] Overall RMSE = {rmse:.6g}")
print(f"[INFO] Overall MAE  = {mae_overall:.6g}")

# Per-channel RMSE
channel_rmse = np.sqrt(np.mean((pred_matrix - obs_matrix) ** 2, axis=0))
channel_df = pd.DataFrame({
    "channel": model_cols,
    "rmse": channel_rmse,
})
channel_df.to_csv("observed_vs_predicted_channel_rmse.csv", index=False)
print("[INFO] Saved observed_vs_predicted_channel_rmse.csv")

# ============================================================
# Plot 16-channel time series
# ============================================================

fig, axs = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
axs = axs.flatten()

for ci, comp in enumerate(["eXX", "eYY", "eXY", "eZZ"]):
    for si, st in enumerate(stations):
        idx = ci * len(stations) + si
        obs_col = obs_cols[idx]
        pred_col = f"{comp}_{st}"

        ax = axs[idx]
        ax.plot(t_obs, obs_df[obs_col], label="observed", color="tab:blue", alpha=0.75)
        ax.plot(t_obs, pred_df[pred_col], label="predicted", color="tab:orange", linestyle="--", alpha=0.9)
        ax.set_title(f"{comp} @ {st}")
        ax.grid(True, linewidth=0.3)

        if idx % 4 == 0:
            ax.set_ylabel("Strain (nε)")

for ax in axs[-4:]:
    ax.set_xlabel("Time (s)")

axs[0].legend(fontsize=8)
plt.suptitle("Observed vs Predicted Strain Time Series (16 channels)")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("observed_vs_predicted_strain_all_channels.png", dpi=150)
plt.close()
print("[INFO] Saved observed_vs_predicted_strain_all_channels.png")

# ============================================================
# Plot MAE over time
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(t_obs, mae_per_time, color="tab:red")
plt.xlabel("Time (s)")
plt.ylabel("MAE (nε)")
plt.title("Observed vs Predicted Strain MAE over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("observed_vs_predicted_mae.png", dpi=150)
plt.close()
print("[INFO] Saved observed_vs_predicted_mae.png")
