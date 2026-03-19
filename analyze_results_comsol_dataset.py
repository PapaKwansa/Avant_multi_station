import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as input_data

sns.set_style("whitegrid")

# ============================================================
# FILES
# ============================================================

SAMPLE_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"
OBSERVED_FILE = "avant_cleaned_strain.csv"

param_names = ["s", "E", "theta", "sigma"]

# ============================================================
# LOAD DATA
# ============================================================

samples = np.load(SAMPLE_FILE)
logps = np.load(LOGP_FILE).flatten()

nchains, niter, nparams = samples.shape
samples_flat = samples.reshape(-1, nparams)

print("Samples shape:", samples.shape)

# Observed data
obs_df = pd.read_csv(OBSERVED_FILE)
time = obs_df["time_s"].values
COMPONENT_COLS = [c for c in obs_df.columns if c != "time_s"]
observed = obs_df[COMPONENT_COLS].values

# ============================================================
# TRACE PLOTS
# ============================================================

fig, axes = plt.subplots(nparams, 1, figsize=(10, 3*nparams))

for i in range(nparams):
    for ch in range(nchains):
        axes[i].plot(samples[ch,:,i], alpha=0.6)
    axes[i].set_ylabel(param_names[i])

plt.xlabel("Iteration")
plt.tight_layout()
plt.savefig("trace_plot.png", dpi=300)
plt.close()

# ============================================================
# POSTERIOR DISTRIBUTIONS
# ============================================================

for i, name in enumerate(param_names):
    plt.figure()
    sns.histplot(samples_flat[:, i], bins=50, kde=True)
    plt.title(f"Posterior: {name}")
    plt.savefig(f"posterior_{name}.png", dpi=300)
    plt.close()

# ============================================================
# SUMMARY STATS
# ============================================================

summary = []

for i, name in enumerate(param_names):
    data = samples_flat[:, i]
    summary.append({
        "param": name,
        "mean": np.mean(data),
        "std": np.std(data),
        "2.5%": np.percentile(data, 2.5),
        "50%": np.median(data),
        "97.5%": np.percentile(data, 97.5)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("posterior_summary.csv", index=False)
print(summary_df)

# ============================================================
# CORRELATION MATRIX
# ============================================================

df_post = pd.DataFrame(samples_flat, columns=param_names)

plt.figure(figsize=(6,5))
sns.heatmap(df_post.corr(), annot=True, cmap="coolwarm")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.close()

# ============================================================
# BEST (MAP) PARAMS
# ============================================================

best_idx = np.argmax(logps)
best_params = samples_flat[best_idx]

print("Best params:", best_params)

# ============================================================
# FORWARD MODEL SETUP
# ============================================================

params = input_data.read_input()

stations_df = pd.read_csv("AVANT_stations.csv")
station_names = stations_df["station"].astype(str).str.strip().values
x_prime = stations_df["x_prime"].values
y_prime = stations_df["y_prime"].values
z = stations_df["depth"].values

# Geometry ratios
a0 = 1.0
b0 = 25.0 / 175.0
c0 = 125.0 / 175.0

# ============================================================
# RUN BEST MODEL
# ============================================================

s, E, theta, sigma = best_params
a, b, c = s*a0, s*b0, s*c0

df_best_raw = forward_model_multi_station(
    pmax=params["pmax"],
    tpeak=params["tpeak"],
    d=params["d"],
    time=time,
    x_prime=x_prime,
    y_prime=y_prime,
    x0_prime=params["x0_prime"],
    y0_prime=params["y0_prime"],
    z=z,
    a=a, b=b, c=c,
    nu=params["nu"],
    h=params["h"],
    E=E,
    theta_deg=theta,
    alpha=params.get("alpha", None),
    station_names=station_names,
)

# ============================================================
# 🔥 IMPORTANT: MAP FORWARD MODEL → strain_1...strain_16
# ============================================================

# Convert forward model columns → match observed
def map_forward_to_strain(df):
    cols = []
    for st in station_names:
        cols.extend([
            f"Epsilon_XX_nanostrain_{st}",
            f"Epsilon_YY_nanostrain_{st}",
            f"Epsilon_XY_nanostrain_{st}",
            f"Epsilon_ZZ_nanostrain_{st}",
        ])
    df_out = pd.DataFrame()
    for i, col in enumerate(cols):
        df_out[f"strain_{i+1}"] = df[col]
    return df_out

df_best = map_forward_to_strain(df_best_raw)

# ============================================================
# FIT PLOTS
# ============================================================

for col in COMPONENT_COLS:
    plt.figure()
    plt.plot(time, obs_df[col], label="Observed")
    plt.plot(time, df_best[col], label="Model")
    plt.title(col)
    plt.legend()
    plt.savefig(f"fit_{col}.png", dpi=200)
    plt.close()

# ============================================================
# POSTERIOR PREDICTIVE (UNCERTAINTY)
# ============================================================

Nsamp = 100
pred_ensemble = np.zeros((Nsamp, len(time), len(COMPONENT_COLS)))

for i in range(Nsamp):
    idx = np.random.randint(len(samples_flat))
    s, E, theta, sigma = samples_flat[idx]

    a, b, c = s*a0, s*b0, s*c0

    df_pred_raw = forward_model_multi_station(
        pmax=params["pmax"],
        tpeak=params["tpeak"],
        d=params["d"],
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=params["x0_prime"],
        y0_prime=params["y0_prime"],
        z=z,
        a=a, b=b, c=c,
        nu=params["nu"],
        h=params["h"],
        E=E,
        theta_deg=theta,
        alpha=params.get("alpha", None),
        station_names=station_names,
    )

    df_pred = map_forward_to_strain(df_pred_raw)
    pred_ensemble[i,:,:] = df_pred.values

# ============================================================
# UNCERTAINTY PLOTS
# ============================================================

mean_pred = pred_ensemble.mean(axis=0)
low = np.percentile(pred_ensemble, 2.5, axis=0)
high = np.percentile(pred_ensemble, 97.5, axis=0)

for i, col in enumerate(COMPONENT_COLS):
    plt.figure()
    plt.fill_between(time, low[:,i], high[:,i], alpha=0.3, label="95% CI")
    plt.plot(time, mean_pred[:,i], label="Mean")
    plt.plot(time, obs_df[col], 'k.', label="Observed")
    plt.title(col)
    plt.legend()
    plt.savefig(f"uncertainty_{col}.png", dpi=200)
    plt.close()

print("✅ Post-processing complete!")