import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from pydream.convergence import Gelman_Rubin

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as input_data

sns.set_style("whitegrid")

# ============================================================
# FILES
# ============================================================

SAMPLE_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"
OBSERVED_FILE = "avant_cleaned_strain.csv"

param_names = ['s', 'E', 'theta_deg', 'sigma']
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

samples = np.load(SAMPLE_FILE)
logps = np.load(LOGP_FILE).flatten()

nchains, niter, nparams = samples.shape
posterior = samples.reshape(-1, nparams)

print(f"Samples shape: {samples.shape}")

# Observed data
obs_df = pd.read_csv(OBSERVED_FILE)
time = obs_df["time_s"].values
strain_cols = [c for c in obs_df.columns if c != "time_s"]
observed = obs_df[strain_cols].values

# ============================================================
# SUMMARY STATS
# ============================================================

summary = []
for i, name in enumerate(param_names):
    vals = posterior[:, i]
    summary.append([
        name,
        np.mean(vals),
        np.std(vals),
        np.percentile(vals, 2.5),
        np.median(vals),
        np.percentile(vals, 97.5)
    ])

summary_df = pd.DataFrame(summary, columns=[
    "param", "mean", "std", "2.5%", "50%", "97.5%"
])

print(summary_df)
summary_df.to_csv("posterior_summary.csv", index=False)

# ============================================================
# TRACE PLOTS
# ============================================================

fig, axes = plt.subplots(nparams, 1, figsize=(10, 3*nparams), sharex=True)

for i in range(nparams):
    for ch in range(nchains):
        axes[i].plot(samples[ch,:,i], alpha=0.6)
    axes[i].set_ylabel(param_names[i])

axes[-1].set_xlabel("Iteration")
plt.tight_layout()
plt.savefig("trace_plot.png", dpi=300)
plt.close()

# ============================================================
# POSTERIOR HISTOGRAMS
# ============================================================

fig, axes = plt.subplots(nparams, 1, figsize=(10, 3*nparams))

for i, ax in enumerate(axes):
    data = posterior[:, i]
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)

    ax.hist(data, bins=50, density=True, alpha=0.5)
    ax.plot(x, kde(x), 'k-')
    ax.set_title(param_names[i])

plt.tight_layout()
plt.savefig("posterior_histograms.png", dpi=300)
plt.close()

# ============================================================
# BEST (MAP) PARAMETERS
# ============================================================

best_idx = np.argmax(logps)
best_params = posterior[best_idx]

print("Best params:", best_params)

# ============================================================
# FORWARD MODEL (MAP)
# ============================================================

params = input_data.read_input()

s, E_val, theta, sigma = best_params

a0, b0, c0 = 1.0, 25/175, 125/175
a, b, c = s*a0, s*b0, s*c0

df_best = forward_model_multi_station(
    pmax=params["pmax"],
    tpeak=params["tpeak"],
    d=params["d"],
    time=time,
    x_prime=params["x_prime"],
    y_prime=params["y_prime"],
    x0_prime=params["x0_prime"],
    y0_prime=params["y0_prime"],
    z=params["z"],
    a=a, b=b, c=c,
    nu=params["nu"],
    h=params["h"],
    E=E_val,
    theta_deg=theta,
    alpha=params.get("alpha", None),
    station_names=params.get("station_names", None)
)

pred_best = df_best[strain_cols].values

# ============================================================
# RESIDUALS (CORRECT)
# ============================================================

residual = observed - pred_best
rmse = np.sqrt(np.mean(residual**2))

print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(10,6))
plt.plot(time, residual)
plt.axhline(0, linestyle="--")
plt.title("Residuals (Observed - Model)")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.savefig("residuals.png", dpi=300)
plt.close()

# ============================================================
# POSTERIOR PREDICTIVE (UNCERTAINTY)
# ============================================================

Nsamp = 300
ensemble = np.zeros((Nsamp, len(time), len(strain_cols)))

for i in range(Nsamp):
    idx = np.random.randint(0, posterior.shape[0])
    s, E_val, theta, _ = posterior[idx]

    a, b, c = s*a0, s*b0, s*c0

    df_pred = forward_model_multi_station(
        pmax=params["pmax"],
        tpeak=params["tpeak"],
        d=params["d"],
        time=time,
        x_prime=params["x_prime"],
        y_prime=params["y_prime"],
        x0_prime=params["x0_prime"],
        y0_prime=params["y0_prime"],
        z=params["z"],
        a=a, b=b, c=c,
        nu=params["nu"],
        h=params["h"],
        E=E_val,
        theta_deg=theta,
        alpha=params.get("alpha", None),
        station_names=params.get("station_names", None)
    )

    ensemble[i,:,:] = df_pred[strain_cols].values

# Compute uncertainty
lower = np.percentile(ensemble, 2.5, axis=0)
upper = np.percentile(ensemble, 97.5, axis=0)
mean_pred = ensemble.mean(axis=0)

# Plot first few strain components (avoid clutter)
for i in range(min(6, len(strain_cols))):
    plt.figure(figsize=(8,5))
    plt.fill_between(time, lower[:,i], upper[:,i], alpha=0.3)
    plt.plot(time, mean_pred[:,i], label="Posterior mean")
    plt.plot(time, observed[:,i], 'k.', label="Observed")
    plt.title(f"strain_{i+1}")
    plt.legend()
    plt.savefig(f"predictive_strain_{i+1}.png", dpi=300)
    plt.close()

print("✅ Post-processing complete.")