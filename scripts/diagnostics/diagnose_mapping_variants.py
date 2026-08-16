# diagnose_mapping_variants.py
import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import multi_stations_input as input_data
import multi_station_strain as sf

try:
    import multi_station_rotation as rotmod
except Exception:
    rotmod = None

try:
    import multi_station_coord_transform as ctmod
except Exception:
    ctmod = None


# ============================================================
# Setup
# ============================================================

np.random.seed(42)

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

params = input_data.read_input()

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
Ns = len(station_names)

obs_df = pd.read_csv(OBSERVED_FILE)
obs_df = obs_df.drop_duplicates().reset_index(drop=True)

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
        f"Observed data layout mismatch. Found {len(non_time_cols)} non-time columns, "
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
# Current best values
# ============================================================

nu = float(params["nu"])
alpha = float(params["alpha"])

s_fixed = 244.0
h_fixed = 2400.0
x0_fixed = -208.33333333333337
y0_fixed = 83.33333333333326
theta_fixed = -60.0
tpeak_fixed = 393333.0
d_fixed = 0.6
pmax_fixed = 9.75e6
E_fixed = 1.0e10

a0 = float(params["a0"])
b0 = float(params["b0"])
c0 = float(params["c0"])

print("[INFO] Fixed values:")
print(f"       s = {s_fixed}")
print(f"       h = {h_fixed}")
print(f"       x0_prime = {x0_fixed}")
print(f"       y0_prime = {y0_fixed}")
print(f"       theta_deg = {theta_fixed}")
print(f"       tpeak = {tpeak_fixed}")
print(f"       d = {d_fixed}")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       E = {E_fixed:.6g}")


# ============================================================
# Helpers
# ============================================================

def pressure_time_series(pmax, tpeak, d, time):
    t = np.asarray(time, dtype=float)
    p = np.zeros_like(t, dtype=float)
    rising = t <= tpeak
    falling = t > tpeak
    p[rising] = (pmax / tpeak) * t[rising]
    p[falling] = pmax * np.exp(-d * (t[falling] / tpeak - 1.0))
    return p


def primed_to_unprimed_fallback(x_prime, y_prime, x0_prime, y0_prime, theta_deg):
    theta = np.radians(theta_deg)
    dx = np.asarray(x_prime, dtype=float) - x0_prime
    dy = np.asarray(y_prime, dtype=float) - y0_prime
    x = dx * np.cos(theta) - dy * np.sin(theta)
    y = dx * np.sin(theta) + dy * np.cos(theta)
    return x, y


def best_scalar_multiplier(pred, obs):
    p = pred.reshape(-1)
    o = obs.reshape(-1)
    denom = np.dot(p, p)
    if denom <= 0:
        return np.nan
    return float(np.dot(p, o) / denom)


def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def build_prediction(use_rotation=True, signs=(1, 1, 1, 1)):
    """
    Returns (Nt, 4*Ns) in the order:
        eXX for all stations,
        eYY for all stations,
        eXY for all stations,
        eZZ for all stations.
    """
    a = s_fixed * a0
    b = s_fixed * b0
    c = s_fixed * c0

    if ctmod is not None and hasattr(ctmod, "primed_to_unprimed"):
        x_arr, y_arr = ctmod.primed_to_unprimed(
            x_prime, y_prime, x0_fixed, y0_fixed, theta_fixed
        )
    else:
        x_arr, y_arr = primed_to_unprimed_fallback(
            x_prime, y_prime, x0_fixed, y0_fixed, theta_fixed
        )

    p_series = pressure_time_series(pmax_fixed, tpeak_fixed, d_fixed, time_vals)

    try:
        setattr(sf, "nu", nu)
        setattr(sf, "E", E_fixed)
        setattr(sf, "alpha", alpha)
    except Exception:
        pass

    if not (hasattr(sf, "linear_trans") and hasattr(sf, "charac_strain") and hasattr(sf, "strain")):
        raise RuntimeError("multi_station_strain must provide linear_trans, charac_strain, and strain")

    out = np.zeros((len(time_vals), 4 * Ns), dtype=float)

    for it, p_val in enumerate(p_series):
        lt = sf.linear_trans(alpha, nu, p_val, E_fixed)
        ec = sf.charac_strain(lt, nu)

        exx_list = []
        eyy_list = []
        exy_list = []
        ezz_list = []

        for js, (xx, yy) in enumerate(zip(x_arr, y_arr)):
            S = sf.strain(x=xx, y=yy, z=z[js], a=a, b=b, c=c, ec=ec, h=h_fixed, nu=nu)

            exx, eyy, ezz = S[0, 0], S[1, 1], S[2, 2]
            exy, exz, eyz = S[0, 1], S[0, 2], S[1, 2]

            if use_rotation:
                if hasattr(sf, "rotate_strain_tensor"):
                    exx, eyy, exy, exz, eyz, ezz = sf.rotate_strain_tensor(
                        exx, eyy, exy, exz, eyz, ezz, theta_fixed
                    )
                elif rotmod is not None and hasattr(rotmod, "rotate_strain_tensor"):
                    exx, eyy, exy, exz, eyz, ezz = rotmod.rotate_strain_tensor(
                        exx, eyy, exy, exz, eyz, ezz, theta_fixed
                    )

            exx_list.append(exx * signs[0] * 1e9)
            eyy_list.append(eyy * signs[1] * 1e9)
            exy_list.append(exy * signs[2] * 1e9)
            ezz_list.append(ezz * signs[3] * 1e9)

        out[it, :] = np.concatenate([exx_list, eyy_list, exy_list, ezz_list])

    return out


def sign_name(signs):
    labels = ["eXX", "eYY", "eXY", "eZZ"]
    parts = []
    for lab, sgn in zip(labels, signs):
        if sgn < 0:
            parts.append(f"-{lab}")
        else:
            parts.append(f"+{lab}")
    return ",".join(parts)


# ============================================================
# Variant search
# ============================================================

variants = []

rotation_options = [True, False]
sign_options = list(itertools.product([-1, 1], repeat=4))

print("\n[INFO] Evaluating mapping variants...")

for use_rotation in rotation_options:
    for signs in sign_options:
        pred = build_prediction(use_rotation=use_rotation, signs=signs)
        score = rmse(pred, obs_matrix)
        a_hat = best_scalar_multiplier(pred, obs_matrix)
        variants.append({
            "use_rotation": use_rotation,
            "signs": signs,
            "sign_name": sign_name(signs),
            "rmse": score,
            "scalar": a_hat,
            "pred": pred,
        })

variants.sort(key=lambda d: d["rmse"])

print("\n[RESULT] Top 10 variants:")
for i, v in enumerate(variants[:10], start=1):
    print(
        f"{i:2d}. rmse={v['rmse']:.6g}  scalar={v['scalar']:.6g}  "
        f"rotation={v['use_rotation']}  signs={v['sign_name']}"
    )

best = variants[0]
print("\n[BEST]")
print(f"RMSE   = {best['rmse']:.6g}")
print(f"scalar = {best['scalar']:.6g}")
print(f"rotation = {best['use_rotation']}")
print(f"signs = {best['sign_name']}")


# ============================================================
# Save results
# ============================================================

results = []
for v in variants:
    results.append({
        "use_rotation": v["use_rotation"],
        "signs": list(v["signs"]),
        "sign_name": v["sign_name"],
        "rmse": v["rmse"],
        "scalar": v["scalar"],
    })

with open("mapping_diagnostics.json", "w") as f:
    json.dump(results, f, indent=2)
print("[INFO] Saved mapping_diagnostics.json")

df_rank = pd.DataFrame(results)
df_rank.to_csv("mapping_diagnostics.csv", index=False)
print("[INFO] Saved mapping_diagnostics.csv")


# ============================================================
# Plot best fit
# ============================================================

best_pred = best["pred"]

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
plt.savefig("mapping_diagnostic_best_fit.png", dpi=200)
plt.close()
print("[INFO] Saved mapping_diagnostic_best_fit.png")