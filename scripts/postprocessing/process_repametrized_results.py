"""
Multi-station PyDREAM Post-Processing Script (Strain + Volume)
- Trace plots
- Posterior histograms with KDE (full posterior vs top 10%)
- Posterior summary table
- Posterior correlation matrix
- Predictive posterior plots with uncertainty bands per component per station
- Volume posterior predictive plot
- Residual plots per station and for volume
- Pairwise scatter plots of posterior
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

# -------------------------
# Parameters
# -------------------------
SAMPLE_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"
param_names = ['s', 'E', 'theta_deg', 'sigma']  # sampled parameters
top_percent = 0.10
Nplot = 200
NSAMP_UNCERT = 500  # posterior draws for predictive uncertainty
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Fixed geometry ratios
# -------------------------
params_input = multi_stations_input.read_input()
a0 = 1.0
b0 = 25.0 / 175.0
c0 = 125.0 / 175.0
nu = params_input['nu']
time = np.array(params_input['time'])
delta_P = params_input['pmax'] * np.sin(np.pi * time / max(time))

# Multi-station setup
station_names = np.array(params_input['station_names']) if 'station_names' in params_input else None
if station_names is None:
    Ns = len(params_input['x_prime'])
    station_names = np.array([f"ST{idx+1:02d}" for idx in range(Ns)])
else:
    Ns = len(station_names)

base_components = [
    "Epsilon_XX_nanostrain", "Epsilon_YY_nanostrain", "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain", "Epsilon_XZ_nanostrain", "Epsilon_YZ_nanostrain"
]

COMPONENT_COLS = [f"{comp}_{station_names[s]}" for comp in base_components for s in range(Ns)]

# -------------------------
# Load PyDREAM results
# -------------------------
posterior_samples = np.load(SAMPLE_FILE)
logps = np.load(LOGP_FILE).flatten()
nchains, niterations, nparams_loaded = posterior_samples.shape
posterior_flat = posterior_samples.reshape(-1, nparams_loaded)
print(f"Loaded posterior samples: chains={nchains}, iterations={niterations}, params={nparams_loaded}")

# -------------------------
# Trace plots
# -------------------------
fig, axes = plt.subplots(nparams_loaded, 1, figsize=(12, 3*nparams_loaded), sharex=True)
for i, ax in enumerate(axes):
    for ch in range(nchains):
        ax.plot(posterior_samples[ch,:,i], alpha=0.7, lw=1)
    ax.set_ylabel(param_names[i])
axes[-1].set_xlabel("Iteration")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trace_plot.png"), dpi=300)
plt.close()
print("Saved trace plot: trace_plot.png")

# -------------------------
# Posterior correlation matrix
# -------------------------
df_post = pd.DataFrame(posterior_flat, columns=param_names)
corr_matrix = df_post.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, fmt=".2f", linewidths=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "posterior_correlation_matrix.png"), dpi=300)
plt.close()
print("Saved posterior correlation matrix: posterior_correlation_matrix.png")

# -------------------------
# Posterior histograms
# -------------------------
n_top = int(top_percent * posterior_flat.shape[0])
top_indices = np.argsort(logps)[-n_top:]
top_thetas = posterior_flat[top_indices]

fig, axes = plt.subplots(nparams_loaded, 1, figsize=(10, 3*nparams_loaded))
for i, ax in enumerate(axes):
    data = posterior_flat[:, i]
    top_data = top_thetas[:, i]
    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 2000)
    map_val = x_grid[np.argmax(kde(x_grid))]
    mean_val = np.mean(data)
    median_val = np.median(data)
    ci_lower, ci_upper = np.percentile(data, [2.5, 97.5])

    sns.histplot(data, bins=60, stat="density", color="lightsteelblue", alpha=0.45, ax=ax, label="Posterior")
    sns.histplot(top_data, bins=40, stat="density", color="steelblue", alpha=0.65, ax=ax, label="Top 10%")
    ax.plot(x_grid, kde(x_grid), color='black', lw=2, label="KDE")
    ax.axvspan(ci_lower, ci_upper, color='gray', alpha=0.2)
    ax.axvline(mean_val, color='red', linestyle='--', lw=1.5)
    ax.axvline(median_val, color='green', linestyle='-.', lw=1.5)
    ax.axvline(map_val, color='purple', linestyle=':', lw=2)
    ax.set_title(param_names[i])
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "posterior_parameter_distributions.png"), dpi=300)
plt.close()
print("Saved posterior histograms: posterior_parameter_distributions.png")

# -------------------------
# Posterior summary table
# -------------------------
summary = pd.DataFrame({
    "Parameter": param_names,
    "Mean": [np.mean(posterior_flat[:, i]) for i in range(nparams_loaded)],
    "Std": [np.std(posterior_flat[:, i]) for i in range(nparams_loaded)],
    "MAP": [posterior_flat[:, i][np.argmax(np.histogram(posterior_flat[:, i], bins=40)[0])] for i in range(nparams_loaded)],
    "Median": [np.median(posterior_flat[:, i]) for i in range(nparams_loaded)],
    "2.5%": [np.percentile(posterior_flat[:, i], 2.5) for i in range(nparams_loaded)],
    "97.5%": [np.percentile(posterior_flat[:, i], 97.5) for i in range(nparams_loaded)]
})

# Compute mean a,b,c from s
s_mean = summary.loc[summary['Parameter']=='s', 'Mean'].values[0]
summary['Mean_a'] = summary['Mean_b'] = summary['Mean_c'] = np.nan
summary.loc[summary['Parameter']=='s', ['Mean_a','Mean_b','Mean_c']] = [s_mean*a0, s_mean*b0, s_mean*c0]
summary.to_csv(os.path.join(OUT_DIR,"posterior_summary.csv"), index=False)
print("Saved posterior summary table: posterior_summary.csv")

# -------------------------
# Volume forward function
# -------------------------
def bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))

def volume_forward(a, b, c, E, nu, delta_P):
    V0 = a * b * c
    K = bulk_modulus(E, nu)
    return V0 * (1.0 + delta_P / K)

V_obs = pd.read_csv(os.path.join(OUT_DIR, 'strain_volume_dataset.csv'))['Volume'].values

# -------------------------
# Posterior predictive: strain + volume
# -------------------------
def predictive_posterior(post, logps, param_names, n_samples=NSAMP_UNCERT):
    ensemble_volume = np.zeros((n_samples, len(time)))
    ensemble_strain = np.zeros((n_samples, len(base_components), len(time)))
    
    # MAP draw
    map_idx = np.argmax(logps)
    map_draw = post[map_idx]
    s_map = map_draw[param_names.index('s')]
    E_map = map_draw[param_names.index('E')]
    theta_map = map_draw[param_names.index('theta_deg')]
    a_map, b_map, c_map = s_map*a0, s_map*b0, s_map*c0

    df_best = forward_model_multi_station(
        params_input['pmax'], params_input['tpeak'], params_input['d'], time,
        np.array(params_input['x_prime']), np.array(params_input['y_prime']),
        params_input['x0_prime'], params_input['y0_prime'], params_input['z'],
        a_map, b_map, c_map, nu, params_input['h'], E_map, theta_map,
        params_input.get('alpha', None)
    )
    best_fit_strain = df_best[COMPONENT_COLS].values.T
    best_fit_volume = volume_forward(a_map, b_map, c_map, E_map, nu, delta_P)

    # Posterior predictive ensemble
    for sidx in range(n_samples):
        draw_idx = np.random.choice(post.shape[0])
        draw = post[draw_idx]
        s_val = draw[param_names.index('s')]
        E_val = draw[param_names.index('E')]
        theta_val = draw[param_names.index('theta_deg')]
        a_s, b_s, c_s = s_val*a0, s_val*b0, s_val*c0

        df_pred = forward_model_multi_station(
            params_input['pmax'], params_input['tpeak'], params_input['d'], time,
            np.array(params_input['x_prime']), np.array(params_input['y_prime']),
            params_input['x0_prime'], params_input['y0_prime'], params_input['z'],
            a_s, b_s, c_s, nu, params_input['h'], E_val, theta_val,
            params_input.get('alpha', None)
        )
        ensemble_strain[sidx, :, :] = df_pred[COMPONENT_COLS].values.T
        ensemble_volume[sidx, :] = volume_forward(a_s, b_s, c_s, E_val, nu, delta_P)

    return ensemble_strain, ensemble_volume, best_fit_strain, best_fit_volume

ensemble_strain, ensemble_volume, best_strain, best_volume = predictive_posterior(posterior_flat, logps, param_names)

# -------------------------
# Volume plot
# -------------------------
lower_vol = np.percentile(ensemble_volume, 2.5, axis=0)
upper_vol = np.percentile(ensemble_volume, 97.5, axis=0)

plt.figure(figsize=(12,6))
plt.fill_between(time, lower_vol, upper_vol, color='lightcoral', alpha=0.25)
plt.plot(time, ensemble_volume.mean(axis=0), color='red', lw=2, label='Posterior mean')
plt.plot(time, V_obs, 'k.', label='Observed')
plt.plot(time, best_volume, 'b--', lw=2, label='MAP')
plt.xlabel('Time (hours)')
plt.ylabel('Volume')
plt.title('Posterior Predictive — Volume')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'predictive_fit_volume.png'), dpi=300)
plt.close()
print("Saved predictive volume plot: predictive_fit_volume.png")

# -------------------------
# Residuals: Volume
# -------------------------
residual_volume = V_obs - ensemble_volume.mean(axis=0)
rmse_vol = np.sqrt(np.mean(residual_volume**2))
print(f"Volume RMSE = {rmse_vol:.6f}")

# -------------------------
# Residuals: Strain per station
# -------------------------
def plot_residuals_strain(ensemble_strain, best_strain, COMPONENT_COLS, time):
    Ns = ensemble_strain.shape[1] // 6
    for s in range(Ns):
        obs = ensemble_strain.mean(axis=0)[:, :]
        best = best_strain
        cols = COMPONENT_COLS[s*6:(s+1)*6]
        residuals = obs - best
        plt.figure(figsize=(12,6))
        for i in range(6):
            plt.plot(time, residuals[i], label=cols[i])
        plt.axhline(0, color='k', ls='--')
        plt.xlabel('Time')
        plt.ylabel('Residual (nε)')
        plt.title(f'Station {s+1} Strain Residuals')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,f'residuals_station_{s+1}.png'), dpi=300)
        plt.close()

plot_residuals_strain(ensemble_strain, best_strain, COMPONENT_COLS, time)

# -------------------------
# Posterior pairwise scatter
# -------------------------
def plot_pairwise_scatter(posterior_flat, param_names, out_png="posterior_pairwise.png", nmax=5000):
    import seaborn as sns
    df = pd.DataFrame(posterior_flat, columns=param_names)
    if df.shape[0] > nmax:
        df = df.sample(n=nmax, random_state=42)
    g = sns.pairplot(df, corner=False, diag_kind='kde', plot_kws={'s':10, 'alpha':0.5})
    g.figure.suptitle("Posterior pairwise scatter (sampled)", y=1.02)
    g.figure.set_size_inches(12,12)
    g.savefig(out_png, dpi=200)
    print("Saved pairwise plot:", out_png)

plot_pairwise_scatter(posterior_flat, param_names)

print("Post-processing complete!")
