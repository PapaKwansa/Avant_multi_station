# mini_inversion_sweep_pmax_refine.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as input_data


np.random.seed(42)

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

params = input_data.read_input()

stations_df = pd.read_csv(STATION_FILE)
stations_df["station"] = stations_df["station"].astype(str).str.strip()

station_names = stations_df["station"].values
x_prime = stations_df["x_prime"].values.astype(float)
y_prime = stations_df["y_prime"].values.astype(float)
z = stations_df["depth"].values.astype(float)

obs_df = pd.read_csv(OBSERVED_FILE)
obs_df = obs_df.drop_duplicates().reset_index(drop=True)

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
        f"Observed data layout mismatch. Found {len(non_time_cols)} non-time columns, expected {len(model_cols)}."
    )

obs_matrix = obs_df[obs_cols].values.astype(float)

print(f"[INFO] Stations: {station_names}")
print(f"[INFO] Observed matrix shape: {obs_matrix.shape}")
print(f"[INFO] Using observed columns: {obs_cols[:4]} ... {obs_cols[-4:]}")
print(f"[INFO] Observed min/max: {np.min(obs_matrix):.6g} / {np.max(obs_matrix):.6g}")
print(f"[INFO] Observed mean abs: {np.mean(np.abs(obs_matrix)):.6g}")
print(f"[INFO] Observed std: {np.std(obs_matrix):.6g}")

nu = params["nu"]
alpha = params["alpha"]

s_fixed = 244.0
h_fixed = 2450.0
x0_fixed = -208.33333333333337
y0_fixed = 83.33333333333326
theta_fixed = -60.0
tpeak_fixed = 393333.0
d_fixed = 0.6
E_fixed = 0.8e10

a0 = params["a0"]
b0 = params["b0"]
c0 = params["c0"]

baseline_pmax = float(params.get("pmax", 1.0e7))

print("[INFO] Fixed values for this sweep:")
print(f"       s = {s_fixed}")
print(f"       h = {h_fixed}")
print(f"       x0_prime = {x0_fixed}")
print(f"       y0_prime = {y0_fixed}")
print(f"       theta_deg = {theta_fixed}")
print(f"       tpeak = {tpeak_fixed}")
print(f"       d = {d_fixed}")
print(f"       E = {E_fixed:.6g}")
print(f"[INFO] Baseline pmax = {baseline_pmax:.6g}")


def make_predicted_matrix(pmax):
    a = s_fixed * a0
    b = s_fixed * b0
    c = s_fixed * c0

    df_pred = forward_model_multi_station(
        pmax=pmax,
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
    )

    missing_pred = [c for c in model_cols if c not in df_pred.columns]
    if missing_pred:
        raise RuntimeError(f"Forward model missing columns: {missing_pred}")

    return df_pred[model_cols].values.astype(float)


def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def best_scalar_multiplier(pred, obs):
    p = pred.reshape(-1)
    o = obs.reshape(-1)
    denom = np.dot(p, p)
    if denom <= 0:
        return np.nan
    return float(np.dot(p, o) / denom)


baseline_pred = make_predicted_matrix(baseline_pmax)

print("\n[INFO] Baseline diagnostic")
print(f"[INFO] baseline pred min/max: {np.min(baseline_pred):.6g} / {np.max(baseline_pred):.6g}")
print(f"[INFO] baseline pred mean abs: {np.mean(np.abs(baseline_pred)):.6g}")
print(f"[INFO] baseline RMSE: {rmse(baseline_pred, obs_matrix):.6g}")
print(f"[INFO] baseline best scalar a: {best_scalar_multiplier(baseline_pred, obs_matrix):.6g}")

pmax_vals = np.linspace(7.0e6, 1.3e7, 25)

rmse_vals = []
scalar_vals = []

print("\n[INFO] Sweeping pmax...")
for pmax in pmax_vals:
    pred = make_predicted_matrix(pmax)
    rmse_vals.append(rmse(pred, obs_matrix))
    scalar_vals.append(best_scalar_multiplier(pred, obs_matrix))

rmse_vals = np.asarray(rmse_vals)
scalar_vals = np.asarray(scalar_vals)

best_idx = int(np.argmin(rmse_vals))
best_pmax = float(pmax_vals[best_idx])
best_rmse = float(rmse_vals[best_idx])
best_scalar = float(scalar_vals[best_idx])

print(f"\n[RESULT] Best pmax = {best_pmax:.6g}")
print(f"[RESULT] Best RMSE = {best_rmse:.6g}")
print(f"[RESULT] Best scalar at best pmax = {best_scalar:.6g}")

results = {
    "best_pmax": best_pmax,
    "best_rmse": best_rmse,
    "best_scalar": best_scalar,
    "pmax_values": pmax_vals.tolist(),
    "rmse_values": rmse_vals.tolist(),
    "scalar_values": scalar_vals.tolist(),
    "baseline_pmax": baseline_pmax,
    "s_fixed": s_fixed,
    "h_fixed": h_fixed,
    "x0_fixed": x0_fixed,
    "y0_fixed": y0_fixed,
    "theta_fixed": theta_fixed,
    "tpeak_fixed": tpeak_fixed,
    "d_fixed": d_fixed,
    "E_fixed": E_fixed,
}
with open("pmax_refine_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[INFO] Saved pmax_refine_results.json")

plt.figure(figsize=(7, 4))
plt.plot(pmax_vals, rmse_vals, "o-", linewidth=2)
plt.xlabel("pmax")
plt.ylabel("RMSE")
plt.title("Refined sweep: pmax")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pmax_refine_rmse.png", dpi=200)
plt.close()
print("[INFO] Saved pmax_refine_rmse.png")

best_pred = make_predicted_matrix(best_pmax)

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
plt.savefig("pmax_refine_best_fit.png", dpi=200)
plt.close()
print("[INFO] Saved pmax_refine_best_fit.png")