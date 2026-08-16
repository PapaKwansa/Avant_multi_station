"""
Main inversion post-processing script
-------------------------------------
For the current inversion posterior:
    params = [E, theta_deg, sigma]

Fixed inputs from the geometry stage:
    a_fixed, b_fixed, c_fixed, x0_prime, y0_prime, h, pmax, tpeak, d, nu, alpha

Outputs:
- trace plots
- posterior histograms with KDE
- posterior summary table
- correlation matrix
- pairwise scatter matrix
- posterior predictive strain bands
- residual plots by station/component
"""

import os
import json
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from pandas.plotting import scatter_matrix

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as input_data

# ============================================================
# Settings
# ============================================================

SAMPLE_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"
OUT_DIR = "."
TOP_PERCENT = 0.10
N_PRED_SAMPLES = 250  # posterior predictive ensemble size
RANDOM_SEED = 42

PARAM_NAMES = ["E", "theta_deg", "sigma"]
np.random.seed(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load inputs
# ============================================================

params = input_data.read_input()

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

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

obs_df = pd.read_csv(OBSERVED_FILE).drop_duplicates().reset_index(drop=True)
if "time_s" not in obs_df.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

time = obs_df["time_s"].values.astype(float)

EXPECTED_COMPONENT_COLS = [
    f"{comp}_{sname}"
    for comp in ["eXX", "eYY", "eXY", "eZZ"]
    for sname in station_names
]

non_time_cols = [c for c in obs_df.columns if c != "time_s"]

if all(col in obs_df.columns for col in EXPECTED_COMPONENT_COLS):
    observed_matrix = obs_df[EXPECTED_COMPONENT_COLS].values.astype(float)
elif len(non_time_cols) == len(EXPECTED_COMPONENT_COLS):
    observed_matrix = obs_df[non_time_cols].values.astype(float)
else:
    raise RuntimeError(
        "Observed file columns do not match the expected 16-channel layout. "
        f"Found {len(non_time_cols)} non-time columns, expected {len(EXPECTED_COMPONENT_COLS)}."
    )

observed_vector = observed_matrix.flatten()

print(f"[INFO] Stations: {station_names}")
print(f"[INFO] Observed matrix shape: {observed_matrix.shape}")
print(f"[INFO] Using observed columns: {EXPECTED_COMPONENT_COLS[:4]} ... {EXPECTED_COMPONENT_COLS[-4:]}")
print(f"[INFO] Observed min/max: {np.min(observed_matrix):.6g} / {np.max(observed_matrix):.6g}")
print(f"[INFO] Observed mean abs: {np.mean(np.abs(observed_matrix)):.6g}")
print(f"[INFO] Observed std: {np.std(observed_matrix):.6g}")

# ============================================================
# Fixed values
# ============================================================

nu = float(params["nu"])
alpha = float(params["alpha"])

# Prefer fixed geometry from the current geometry inversion.
# Fallback to old synthetic scaling if needed.
a_fixed = float(params.get("a_fixed", np.nan))
b_fixed = float(params.get("b_fixed", np.nan))
c_fixed = float(params.get("c_fixed", np.nan))

if np.any(np.isnan([a_fixed, b_fixed, c_fixed])):
    if all(k in params for k in ["s", "a0", "b0", "c0"]):
        a_fixed = float(params["s"] * params["a0"])
        b_fixed = float(params["s"] * params["b0"])
        c_fixed = float(params["s"] * params["c0"])
        print("[WARN] a_fixed/b_fixed/c_fixed not found; using legacy s/a0/b0/c0 fallback.")
    else:
        raise RuntimeError("No fixed geometry available in input script.")

x0_fixed = float(params["x0_prime"])
y0_fixed = float(params["y0_prime"])
h_fixed = float(params["h"])
pmax_fixed = float(params["pmax"])
tpeak_fixed = float(params["tpeak"])
d_fixed = float(params["d"])

print("[INFO] Fixed geometry:")
print(f"       a = {a_fixed:.6g}")
print(f"       b = {b_fixed:.6g}")
print(f"       c = {c_fixed:.6g}")
print("[INFO] Fixed non-inverted values:")
print(f"       x0_prime = {x0_fixed:.6g}")
print(f"       y0_prime = {y0_fixed:.6g}")
print(f"       h = {h_fixed:.6g}")
print(f"       pmax = {pmax_fixed:.6g}")
print(f"       tpeak = {tpeak_fixed:.6g}")
print(f"       d = {d_fixed:.6g}")
print(f"       nu = {nu:.6g}")
print(f"       alpha = {alpha:.6g}")

# ============================================================
# Load posterior samples
# ============================================================

posterior_samples = np.load(SAMPLE_FILE, allow_pickle=True)
logps = np.load(LOGP_FILE, allow_pickle=True).flatten()

if posterior_samples.ndim == 3:
    nchains, niterations, nparams = posterior_samples.shape
    posterior_flat = posterior_samples.reshape(-1, nparams)
elif posterior_samples.ndim == 2:
    nchains, niterations, nparams = 1, posterior_samples.shape[0], posterior_samples.shape[1]
    posterior_flat = posterior_samples
else:
    raise RuntimeError(f"Unexpected posterior sample shape: {posterior_samples.shape}")

if nparams != len(PARAM_NAMES):
    raise RuntimeError(
        f"Loaded {nparams} parameters, but PARAM_NAMES has {len(PARAM_NAMES)} entries."
    )

if len(logps) != posterior_flat.shape[0]:
    print(
        f"[WARN] logps length ({len(logps)}) does not match samples ({posterior_flat.shape[0]}). "
        "Some diagnostic plots will use sample order only."
    )

print(f"[INFO] Loaded posterior samples: shape={posterior_samples.shape}")
print(f"[INFO] Flattened posterior: {posterior_flat.shape}")

df_post = pd.DataFrame(posterior_flat, columns=PARAM_NAMES)

# ============================================================
# Helper functions
# ============================================================

def kde_mode(values, grid_points=2000):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float(values[0])
    kde = gaussian_kde(values)
    x_grid = np.linspace(values.min(), values.max(), grid_points)
    y_grid = kde(x_grid)
    return float(x_grid[np.argmax(y_grid)])


def savefig(path):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, path), dpi=300)
    plt.close()
    print(f"Saved {path}")


def predict_strain(E, theta_deg):
    df_pred = forward_model_multi_station(
        pmax=pmax_fixed,
        tpeak=tpeak_fixed,
        d=d_fixed,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_fixed,
        y0_prime=y0_fixed,
        z=z,
        a=a_fixed,
        b=b_fixed,
        c=c_fixed,
        nu=nu,
        h=h_fixed,
        E=E,
        theta_deg=theta_deg,
        alpha=alpha,
        station_names=station_names,
        debug=False,
        flip_exy=True,
    )
    return df_pred[EXPECTED_COMPONENT_COLS].values.astype(float)


def best_scalar_multiplier(pred, obs):
    p = pred.reshape(-1)
    o = obs.reshape(-1)
    denom = np.dot(p, p)
    if denom <= 0:
        return np.nan
    return float(np.dot(p, o) / denom)


# ============================================================
# Trace plots
# ============================================================

fig, axes = plt.subplots(len(PARAM_NAMES), 1, figsize=(12, 3 * len(PARAM_NAMES)), sharex=True)
if len(PARAM_NAMES) == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    for ch in range(nchains):
        if posterior_samples.ndim == 3:
            ax.plot(posterior_samples[ch, :, i], alpha=0.8, lw=1)
        else:
            ax.plot(posterior_samples[:, i], alpha=0.8, lw=1)
    ax.set_ylabel(PARAM_NAMES[i])
axes[-1].set_xlabel("Iteration")
savefig("trace_plot.png")

# ============================================================
# Posterior summary table
# ============================================================

top_n = max(1, int(TOP_PERCENT * posterior_flat.shape[0]))
if len(logps) == posterior_flat.shape[0]:
    top_idx = np.argsort(logps)[-top_n:]
else:
    top_idx = np.arange(posterior_flat.shape[0])[-top_n:]

top_draws = posterior_flat[top_idx]

summary = pd.DataFrame({
    "Parameter": PARAM_NAMES,
    "Mean": [np.mean(posterior_flat[:, i]) for i in range(len(PARAM_NAMES))],
    "Std": [np.std(posterior_flat[:, i]) for i in range(len(PARAM_NAMES))],
    "Median": [np.median(posterior_flat[:, i]) for i in range(len(PARAM_NAMES))],
    "MAP_marginal": [kde_mode(posterior_flat[:, i]) for i in range(len(PARAM_NAMES))],
    "2.5%": [np.percentile(posterior_flat[:, i], 2.5) for i in range(len(PARAM_NAMES))],
    "97.5%": [np.percentile(posterior_flat[:, i], 97.5) for i in range(len(PARAM_NAMES))],
})

summary.to_csv(os.path.join(OUT_DIR, "posterior_summary.csv"), index=False)
print("Saved posterior_summary.csv")

# Also save a small JSON with fixed settings
with open(os.path.join(OUT_DIR, "posterior_fixed_values.json"), "w") as f:
    json.dump(
        {
            "a_fixed": a_fixed,
            "b_fixed": b_fixed,
            "c_fixed": c_fixed,
            "x0_prime": x0_fixed,
            "y0_prime": y0_fixed,
            "h": h_fixed,
            "pmax": pmax_fixed,
            "tpeak": tpeak_fixed,
            "d": d_fixed,
            "nu": nu,
            "alpha": alpha,
        },
        f,
        indent=2,
    )
print("Saved posterior_fixed_values.json")

# ============================================================
# Posterior histograms + KDE
# ============================================================

fig, axes = plt.subplots(len(PARAM_NAMES), 1, figsize=(10, 3 * len(PARAM_NAMES)))
if len(PARAM_NAMES) == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    data = posterior_flat[:, i]
    top_data = top_draws[:, i]

    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 2000)
    kde_y = kde(x_grid)

    mean_val = np.mean(data)
    median_val = np.median(data)
    map_val = x_grid[np.argmax(kde_y)]
    ci_lower, ci_upper = np.percentile(data, [2.5, 97.5])

    ax.hist(data, bins=60, density=True, alpha=0.40, color="steelblue", label="Posterior")
    ax.hist(top_data, bins=40, density=True, alpha=0.55, color="orange", label=f"Top {int(TOP_PERCENT*100)}%")
    ax.plot(x_grid, kde_y, color="black", lw=2, label="KDE")
    ax.axvspan(ci_lower, ci_upper, color="gray", alpha=0.2)
    ax.axvline(mean_val, color="red", ls="--", lw=1.5, label="Mean")
    ax.axvline(median_val, color="green", ls="-.", lw=1.5, label="Median")
    ax.axvline(map_val, color="purple", ls=":", lw=2, label="KDE mode")
    ax.set_title(PARAM_NAMES[i])
    ax.legend(fontsize=8)

savefig("posterior_parameter_distributions.png")

# ============================================================
# Correlation matrix
# ============================================================

corr_matrix = df_post.corr().values

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(PARAM_NAMES)))
ax.set_yticks(range(len(PARAM_NAMES)))
ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
ax.set_yticklabels(PARAM_NAMES)
for i in range(len(PARAM_NAMES)):
    for j in range(len(PARAM_NAMES)):
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.85)
ax.set_title("Posterior Correlation Matrix")
savefig("posterior_correlation_matrix.png")

# ============================================================
# Pairwise scatter plot
# ============================================================

sample_df = df_post.copy()
if sample_df.shape[0] > 5000:
    sample_df = sample_df.sample(n=5000, random_state=RANDOM_SEED)

axes = scatter_matrix(
    sample_df,
    figsize=(9, 9),
    diagonal="kde",
    alpha=0.35,
    marker="o",
    hist_kwds={"bins": 30},
    range_padding=0.1,
)
plt.suptitle("Posterior Pairwise Scatter", y=1.02)
savefig("posterior_pairwise_scatter.png")

# ============================================================
# Posterior predictive ensemble
# ============================================================

# MAP draw by highest log posterior if available; otherwise use the last draw.
if len(logps) == posterior_flat.shape[0]:
    map_idx = int(np.argmax(logps))
else:
    map_idx = int(np.argmax(np.sum(posterior_flat, axis=1)))  # fallback only

map_draw = posterior_flat[map_idx]
E_map, theta_map, sigma_map = map_draw

print("[INFO] MAP-like draw:")
print(f"       E = {E_map:.6g}")
print(f"       theta_deg = {theta_map:.6g}")
print(f"       sigma = {sigma_map:.6g}")

pred_map = predict_strain(E_map, theta_map)

n_channels = len(EXPECTED_COMPONENT_COLS)
Nt = len(time)

ensemble_pred = np.zeros((N_PRED_SAMPLES, Nt, n_channels), dtype=float)

draw_indices = np.random.choice(posterior_flat.shape[0], size=N_PRED_SAMPLES, replace=True)
for k, idx in enumerate(draw_indices):
    E_k, theta_k, sigma_k = posterior_flat[idx]
    ensemble_pred[k] = predict_strain(E_k, theta_k)

pred_mean = ensemble_pred.mean(axis=0)
pred_lo = np.percentile(ensemble_pred, 2.5, axis=0)
pred_hi = np.percentile(ensemble_pred, 97.5, axis=0)

# Save predictive mean to CSV
pred_mean_df = pd.DataFrame({"time_s": time})
for i, col in enumerate(EXPECTED_COMPONENT_COLS):
    pred_mean_df[col] = pred_mean[:, i]
pred_mean_df.to_csv(os.path.join(OUT_DIR, "posterior_predictive_mean.csv"), index=False)
print("Saved posterior_predictive_mean.csv")

# ============================================================
# Posterior predictive plot (all 16 channels)
# ============================================================

fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
axes = np.asarray(axes)

for i, col in enumerate(EXPECTED_COMPONENT_COLS):
    r = i // 4
    c = i % 4
    ax = axes[r, c]
    ax.fill_between(time, pred_lo[:, i], pred_hi[:, i], alpha=0.25)
    ax.plot(time, pred_mean[:, i], lw=2, label="Posterior mean")
    ax.plot(time, observed_matrix[:, i], "k.", markersize=2, label="Observed")
    ax.plot(time, pred_map[:, i], "r--", lw=1.2, label="MAP")
    ax.set_title(col, fontsize=9)
    ax.grid(True, alpha=0.25)
    if i == 0:
        ax.legend(fontsize=8)

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("nstrain")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "posterior_predictive_strain.png"), dpi=300)
plt.close()
print("Saved posterior_predictive_strain.png")

# ============================================================
# Residual plots by station
# ============================================================

# residuals for posterior mean and MAP
res_mean = observed_matrix - pred_mean
res_map = observed_matrix - pred_map

component_order = ["eXX", "eYY", "eXY", "eZZ"]

for si, sname in enumerate(station_names):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ci, comp in enumerate(component_order):
        col_idx = ci * Ns + si
        ax = axes[ci]
        ax.plot(time, res_mean[:, col_idx], label="Residual (obs - posterior mean)")
        ax.plot(time, res_map[:, col_idx], label="Residual (obs - MAP)", alpha=0.75)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_ylabel("nstrain")
        ax.set_title(f"{sname} - {comp}")
        ax.grid(True, alpha=0.25)
        if ci == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"residuals_{sname}.png"), dpi=300)
    plt.close()
    print(f"Saved residuals_{sname}.png")

# ============================================================
# Quick posterior fit metrics
# ============================================================

rmse_map = np.sqrt(np.mean((pred_map - observed_matrix) ** 2))
rmse_mean = np.sqrt(np.mean((pred_mean - observed_matrix) ** 2))
scalar_map = best_scalar_multiplier(pred_map, observed_matrix)

metrics = {
    "rmse_map": float(rmse_map),
    "rmse_posterior_mean": float(rmse_mean),
    "best_scalar_map": float(scalar_map),
}

with open(os.path.join(OUT_DIR, "posterior_fit_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved posterior_fit_metrics.json")

print("[INFO] Post-processing complete.")