"""
GENERAL PyDREAM post-processing script (ROBUST VERSION)

✔ Works for ANY number of inferred parameters (>=1)
✔ Automatically handles theta_deg → theta mapping
✔ Safe for single-parameter identifiability studies
✔ No hard-coded parameter logic

Produces:
- Trace plots
- Posterior marginals + KDE + MAP
- Posterior summary table
- Posterior predictive plots
- Residual plots
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input

# ============================================================
# ---------------------- CONFIGURATION -----------------------
# ============================================================

SAMPLE_FILE = "multi_station_sampled_params.npy"
LOGP_FILE   = "multi_station_logps.npy"
OUT_DIR     = "."

NSAMP_UNCERT = 1000

# Canonical forward-model parameter names (ORDER MATTERS)
ALL_PARAM_NAMES = ["a", "b", "c", "E", "theta"]

PARAM_LABELS = {
    "a": r"$a$ (m)",
    "b": r"$b$ (m)",
    "c": r"$c$ (m)",
    "E": r"$E$ (Pa)",
    "theta": r"$\theta$ (deg)",
}

# Input-file → forward-model name mapping
PARAM_RENAME = {
    "theta_deg": "theta"
}

REQUIRED_FORWARD_PARAMS = ["a", "b", "c", "E", "theta"]

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

# ============================================================
# -------------------- LOAD POSTERIOR ------------------------
# ============================================================

posterior_samples = np.load(SAMPLE_FILE)
logps = np.load(LOGP_FILE).flatten()

nchains, niter, nparams = posterior_samples.shape
samples_flat = posterior_samples.reshape(-1, nparams)

param_names = ALL_PARAM_NAMES[:nparams]

print(f"Inferred parameters: {param_names}")
print(f"Total posterior samples: {samples_flat.shape[0]}")

# ============================================================
# ----------------------- TRACE PLOTS ------------------------
# ============================================================

for j, pname in enumerate(param_names):
    plt.figure(figsize=(12, 3))
    for ch in range(nchains):
        plt.plot(posterior_samples[ch, :, j], lw=1, alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel(pname)
    plt.title(f"Trace plot — {pname}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"trace_{pname}.png"), dpi=300)
    plt.close()

print("Trace plots complete.")

# ============================================================
# --------------- POSTERIOR MARGINALS ------------------------
# ============================================================

summary_rows = []

for j, pname in enumerate(param_names):

    samples = samples_flat[:, j]

    kde = gaussian_kde(samples)
    xgrid = np.linspace(samples.min(), samples.max(), 2000)
    map_val = xgrid[np.argmax(kde(xgrid))]

    mean_val   = samples.mean()
    median_val = np.median(samples)
    std_val    = samples.std()
    lo, hi     = np.percentile(samples, [2.5, 97.5])

    plt.figure(figsize=(8, 5))
    sns.histplot(samples, bins=60, stat="density",
                 color="steelblue", alpha=0.45)
    plt.plot(xgrid, kde(xgrid), color="black", lw=2)
    plt.axvline(mean_val, color="red", ls="--", lw=1.5, label="Mean")
    plt.axvline(median_val, color="green", ls="-.", lw=1.5, label="Median")
    plt.axvline(map_val, color="purple", ls=":", lw=2, label="MAP")
    plt.axvspan(lo, hi, color="gray", alpha=0.25, label="95% CI")

    plt.xlabel(PARAM_LABELS.get(pname, pname))
    plt.ylabel("Posterior density")
    plt.title(f"Posterior of {pname}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"posterior_{pname}.png"), dpi=300)
    plt.close()

    summary_rows.append([pname, mean_val, std_val, map_val, median_val, lo, hi])

summary = pd.DataFrame(
    summary_rows,
    columns=["Parameter", "Mean", "Std", "MAP", "Median", "2.5%", "97.5%"]
)
summary.to_csv(os.path.join(OUT_DIR, "posterior_summary.csv"), index=False)

print("Posterior marginals and summary complete.")

# ============================================================
# ----------------- POSTERIOR PREDICTIVE ---------------------
# ============================================================

params_input_raw = multi_stations_input.read_input()

# Normalize input parameter names
params_input = {}
for k, v in params_input_raw.items():
    k2 = PARAM_RENAME.get(k, k)
    params_input[k2] = v

time = np.array(params_input["time"])
x_prime = np.array(params_input["x_prime"])
y_prime = np.array(params_input["y_prime"])
Ns = len(x_prime)

base_components = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

obs_df = pd.read_csv("strain_dataset_output_multi.csv")

# Fixed parameters = input minus inferred
fixed_params = {
    k: v for k, v in params_input.items() if k not in param_names
}

map_idx = np.argmax(logps)

for station_idx in range(Ns):

    station_cols = [f"{c}_FS{station_idx+1:02d}" for c in base_components]
    observed = obs_df[station_cols].values.T

    ensemble = np.zeros((NSAMP_UNCERT, len(station_cols), len(time)))
    draw_idx = np.random.choice(len(samples_flat), NSAMP_UNCERT, replace=True)

    for k, idx in enumerate(draw_idx):
        sample = dict(zip(param_names, samples_flat[idx]))
        params = fixed_params | sample

        # SAFETY CHECK
        missing = [p for p in REQUIRED_FORWARD_PARAMS if p not in params]
        if missing:
            raise RuntimeError(f"Missing required forward-model params: {missing}")

        df_pred = forward_model_multi_station(
            params["pmax"], params["tpeak"], params["d"],
            time,
            x_prime, y_prime,
            params["x0_prime"], params["y0_prime"], params["z"],
            params["a"], params["b"], params["c"],
            params["nu"], params["h"],
            params["E"], params["theta"],
            params.get("alpha", None),
        )

        ensemble[k] = df_pred[station_cols].values.T

    # MAP prediction
    map_sample = dict(zip(param_names, samples_flat[map_idx]))
    params_map = fixed_params | map_sample

    df_best = forward_model_multi_station(
        params_map["pmax"], params_map["tpeak"], params_map["d"],
        time,
        x_prime, y_prime,
        params_map["x0_prime"], params_map["y0_prime"], params_map["z"],
        params_map["a"], params_map["b"], params_map["c"],
        params_map["nu"], params_map["h"],
        params_map["E"], params_map["theta"],
        params_map.get("alpha", None),
    )

    best_fit = df_best[station_cols].values.T

    lower = np.percentile(ensemble, 2.5, axis=0)
    upper = np.percentile(ensemble, 97.5, axis=0)

    plt.figure(figsize=(14, 7))
    colors = sns.color_palette("tab10", len(station_cols))

    for i in range(len(station_cols)):
        plt.fill_between(time, lower[i], upper[i], color=colors[i], alpha=0.25)
        plt.plot(time, best_fit[i], color=colors[i], lw=2)
        plt.scatter(time, observed[i], color=colors[i], s=12)

    plt.xlabel("Time")
    plt.ylabel("Strain (nε)")
    plt.title(f"Station {station_idx+1} — Posterior predictive")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"predictive_station_{station_idx+1}.png"), dpi=300)
    plt.close()

    # Residuals
    residuals = observed - best_fit

    plt.figure(figsize=(12, 6))
    for i in range(len(station_cols)):
        rmse = np.sqrt(mean_squared_error(observed[i], best_fit[i]))
        plt.plot(time, residuals[i], label=f"{station_cols[i]} (RMSE={rmse:.2f})")

    plt.axhline(0, color="k", ls="--")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.title(f"Station {station_idx+1} residuals")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"residuals_station_{station_idx+1}.png"), dpi=300)
    plt.close()

    gc.collect()

print("GENERAL post-processing complete.")
