"""
Multi-station PyDREAM Post-Processing Script
- Trace plots
- Posterior histograms with KDE (full posterior vs top 10%)
- Posterior summary table
- Predictive posterior plots with uncertainty bands per component per station
- Residual plots per station
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
param_names = ['a', 'b', 'c', 'E', 'theta_deg', 'sigma']
top_percent = 0.10
Nplot = 200           # For top-N plotting (posterior predictive)
NSAMP_UNCERT = 1000    # Number of posterior draws per component for uncertainty bands
OUT_DIR = "."


#-------------------------
# Parameters labels for plotting
# -------------------------
param_labels = {
    'a': r'$a$ — Source semi-axis (m)',
    'b': r'$b$ — Source semi-axis (m)',
    'c': r'$c$ — Vertical extent (m)',
    'E': r'$E$ — Young’s modulus (Pa)',
    'theta_deg': r'$\theta$ — Source orientation (deg)',
    'sigma': r'$\sigma$ — Noise std (n$\varepsilon$)'
}

# =========================================================
# Load PyDREAM results
# -------------------------
if not os.path.exists(SAMPLE_FILE):
    raise FileNotFoundError(f"{SAMPLE_FILE} not found. Run the inversion first!")

posterior_samples = np.load(SAMPLE_FILE)  # shape: (nchains, niterations, nparams)
logps = np.load(LOGP_FILE).flatten()      # shape: (total_samples,)
nchains, niterations, nparams = posterior_samples.shape
posterior_flat = posterior_samples.reshape(-1, nparams)

# -------------------------
# Correlation Matrix (Posterior)
# -------------------------

print("Computing posterior correlation matrix...")

df_post = pd.DataFrame(posterior_flat, columns=param_names)
corr_matrix = df_post.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.7}
)
#plt.title("Posterior Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "posterior_correlation_matrix.png"), dpi=300)
plt.close()

print("Saved posterior correlation matrix: posterior_correlation_matrix.png")


print(f"Loaded posterior samples: chains={nchains}, iterations={niterations}, params={nparams}")

# -------------------------
# Trace plots
# -------------------------
fig, axes = plt.subplots(nparams, 1, figsize=(12, 3*nparams), sharex=True)
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
# Select top 10% posterior samples
# -------------------------
n_top = int(top_percent * posterior_flat.shape[0])
top_indices = np.argsort(logps)[-n_top:]
top_thetas = posterior_flat[top_indices]
top_logps = logps[top_indices]

# Sort top by logp ascending
sort_idx = np.argsort(top_logps)
top_thetas_sorted = top_thetas[sort_idx]
top_logps_sorted = top_logps[sort_idx]

# Limit to Nplot
thetas_plot = top_thetas_sorted[-Nplot:]
thetas_plot_logp = top_logps_sorted[-Nplot:]
print(f"Selected top {top_percent*100:.0f}% ({n_top} samples) for predictive uncertainty. Using Nplot={Nplot}.")

# -------------------------
# Posterior histograms (full + top 10% with legend)
# -------------------------

cols = 3
rows = int(np.ceil(nparams / cols))
fig, axes = plt.subplots(rows, cols, figsize=(6.5*cols, 4.5*rows))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= nparams:
        ax.axis('off')
        continue

    data = posterior_flat[:, i]
    top_data = top_thetas[:, i]

    # --- KDE-based MAP estimate (robust) ---
    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 2000)
    map_val = x_grid[np.argmax(kde(x_grid))]

    # --- Summary statistics ---
    mean_val = np.mean(data)
    median_val = np.median(data)
    ci_lower, ci_upper = np.percentile(data, [2.5, 97.5])

    # --- Histograms ---
    sns.histplot(
        data,
        bins=60,
        stat="density",
        color="lightsteelblue",
        alpha=0.45,
        edgecolor=None,
        ax=ax,
        label="Posterior"
    )

    sns.histplot(
        top_data,
        bins=40,
        stat="density",
        color="steelblue",
        alpha=0.65,
        edgecolor=None,
        ax=ax,
        label="Top 10%"
    )

    # --- KDE curve ---
    ax.plot(x_grid, kde(x_grid), color='black', lw=2, label="KDE")

    # --- Credible interval ---
    ax.axvspan(ci_lower, ci_upper, color='gray', alpha=0.2, label="95% CI")

    # --- Central estimates ---
    ax.axvline(mean_val, color='red', linestyle='--', lw=1.6, label="Mean")
    ax.axvline(median_val, color='green', linestyle='-.', lw=1.6, label="Median")
    ax.axvline(map_val, color='purple', linestyle=':', lw=2.0, label="MAP")

    # --- Multimodality check (BIC-based GMM) ---
    data_reshape = data.reshape(-1, 1)
    bic_scores = [
        GaussianMixture(n_components=k, random_state=42)
        .fit(data_reshape)
        .bic(data_reshape)
        for k in range(1, 4)
    ]
    best_modes = np.argmin(bic_scores) + 1

    # --- Titles & labels ---
    pname = param_names[i]
    ax.set_title(
        f"{param_labels.get(pname, pname)}\n(Inferred posterior modes = {best_modes})",
        fontsize=13
    )
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Posterior density")

    # --- Legend (clean) ---
    ax.legend(fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "posterior_parameter_distributions.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved publication-ready posterior distributions: posterior_parameter_distributions.png")


# -------------------------
# Posterior summary table
# -------------------------
summary = pd.DataFrame({
    "Parameter": param_names,
    "Mean": [np.mean(posterior_flat[:, i]) for i in range(nparams)],
    "Std": [np.std(posterior_flat[:, i]) for i in range(nparams)],
    "MAP": [posterior_flat[:, i][np.argmax(np.histogram(posterior_flat[:, i], bins=40)[0])] for i in range(nparams)],
    "Median": [np.median(posterior_flat[:, i]) for i in range(nparams)],
    "2.5%": [np.percentile(posterior_flat[:, i], 2.5) for i in range(nparams)],
    "97.5%": [np.percentile(posterior_flat[:, i], 97.5) for i in range(nparams)]
})
summary.to_csv(os.path.join(OUT_DIR,"posterior_summary.csv"), index=False)
print("Saved posterior summary table: posterior_summary.csv")

# -------------------------
# Predictive plots per station with uncertainty bands (component-wise)
# ------------------------- 

def plot_predictive_multimodal(post, logps, param_names,
                               n_samples_per_comp=500,
                               max_modes=3,
                               plot_ensemble_lines=True):
    """
    Predictive plots per station with uncertainty bands that account for multimodal posterior distributions.
    Draws are stratified across modes identified by Gaussian Mixture Model (GMM) to capture full spread.

    Args:
    - post: ndarray, posterior samples (n_total_samples x n_params)
    - logps: ndarray, posterior log-probabilities
    - param_names: list of parameter names
    - n_samples_per_comp: number of posterior draws per component
    - max_modes: maximum number of modes for GMM
    - plot_ensemble_lines: bool, plot all individual draws faintly
    """

    import multi_stations_input
    params_input = multi_stations_input.read_input()
    time = np.array(params_input['time'])
    Ns = len(params_input['x_prime'])
    OUT_DIR = "."

    base_components = ["Epsilon_XX_nanostrain", "Epsilon_YY_nanostrain", "Epsilon_ZZ_nanostrain",
                       "Epsilon_XY_nanostrain", "Epsilon_XZ_nanostrain", "Epsilon_YZ_nanostrain"]

    obs_df = pd.read_csv(os.path.join(OUT_DIR, 'strain_dataset_output_multi.csv'))
    all_observed = []
    all_bestfit = []

    # Identify modes for each parameter using GMM
    n_params = post.shape[1]
    mode_indices_per_param = []

    for i in range(n_params):
        data = post[:, i].reshape(-1, 1)
        bic_scores = []
        gm_models = []
        for k in range(1, max_modes + 1):
            gm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gm.fit(data)
            gm_models.append(gm)
            bic_scores.append(gm.bic(data))
        best_k = np.argmin(bic_scores) + 1
        gm_best = gm_models[best_k-1]
        # assign each sample to a mode
        labels = gm_best.predict(data)
        mode_indices_per_param.append(labels)

    # For each station
    for station_idx in range(Ns):
        station_cols = [f"{c}_FS{station_idx+1:02d}" for c in base_components]
        n_comp = len(station_cols)
        short_labels = [c.replace('_nanostrain','') for c in base_components]

        observed = obs_df[station_cols].values.T
        all_observed.append(observed)

        ensemble = np.zeros((n_samples_per_comp, n_comp, len(time)))

        for comp_idx in range(n_comp):
            # Stratified sampling across modes
            param_idx = param_names.index(['a','b','c','E','theta_deg','sigma'][comp_idx % n_params])
            labels = mode_indices_per_param[param_idx]
            unique_modes = np.unique(labels)
            draws_per_mode = max(1, n_samples_per_comp // len(unique_modes))
            draw_indices = []
            for mode in unique_modes:
                mode_indices = np.where(labels == mode)[0]
                sampled = np.random.choice(mode_indices, draws_per_mode, replace=len(mode_indices)<draws_per_mode)
                draw_indices.extend(sampled)
            # fill remaining draws randomly if needed
            while len(draw_indices) < n_samples_per_comp:
                draw_indices.append(np.random.choice(post.shape[0]))

            draw_indices = np.array(draw_indices)[:n_samples_per_comp]

            for sidx, ps_idx in enumerate(draw_indices):
                draw = post[ps_idx]
                a = draw[param_names.index('a')]
                b = draw[param_names.index('b')]
                c = draw[param_names.index('c')]
                E = draw[param_names.index('E')]
                theta = draw[param_names.index('theta_deg')]

                df_pred = forward_model_multi_station(
                    params_input['pmax'], params_input['tpeak'], params_input['d'], time,
                    np.array(params_input['x_prime']), np.array(params_input['y_prime']),
                    params_input['x0_prime'], params_input['y0_prime'],
                    params_input['z'], a, b, c, params_input['nu'], params_input['h'], E,
                    theta,
                    params_input.get('alpha', None)
                )
                ensemble[sidx, comp_idx, :] = df_pred[station_cols[comp_idx]].values

        # Best-fit: use MAP (max logp)
        map_idx = np.argmax(logps)
        draw = post[map_idx]
        a_b = draw[param_names.index('a')]
        b_b = draw[param_names.index('b')]
        c_b = draw[param_names.index('c')]
        E_b = draw[param_names.index('E')]
        theta_b = draw[param_names.index('theta_deg')]

        df_best = forward_model_multi_station(
            params_input['pmax'], params_input['tpeak'], params_input['d'], time,
            np.array(params_input['x_prime']), np.array(params_input['y_prime']),
            params_input['x0_prime'], params_input['y0_prime'],
            params_input['z'], a_b, b_b, c_b, params_input['nu'], params_input['h'], E_b,
            theta_b,
            params_input.get('alpha', None)
        )
        best_fit = df_best[station_cols].values.T
        all_bestfit.append(best_fit)

        # 95% CI
        lower = np.percentile(ensemble, 2.5, axis=0)
        upper = np.percentile(ensemble, 97.5, axis=0)

        # Plot
        plt.figure(figsize=(14,7))
        colors = sns.color_palette("tab10", n_comp)
        for i in range(n_comp):
            if plot_ensemble_lines:
                for sidx in range(n_samples_per_comp):
                    plt.plot(time, ensemble[sidx, i, :], color=colors[i], alpha=0.5, lw=0.7, zorder=0)
            plt.fill_between(time, lower[i], upper[i], color=colors[i], alpha=0.25, zorder=1)
            plt.plot(time, best_fit[i], color=colors[i], lw=2.2, label=short_labels[i], zorder=3)
            plt.scatter(time, observed[i], color=colors[i], s=14, alpha=0.7, zorder=4)

        plt.xlabel("Time (hours)")
        plt.ylabel("Strain (nε)")
        plt.title(f"Station {station_idx+1} — Observed vs Posterior Predictive")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=9, ncol=2)
        out_png = os.path.join(OUT_DIR, f"predictive_fit_station_{station_idx+1}_componentwise.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved predictive plot: {out_png}")
        gc.collect()

    return all_observed, all_bestfit, time, [f"{c}_FS{s+1:02d}" for s in range(Ns) for c in base_components]


# -------------------------
# Residual plots
# -------------------------
def plot_residuals_combined(observed, best_fit, time, station_idx=1, COMPONENT_COLS=None):
    residuals = observed - best_fit
    n_comp = observed.shape[0]
    colors = sns.color_palette("tab10", n_comp)

    print(f"\nResiduals for station {station_idx} (RMSE):")
    for i in range(n_comp):
        rmse = np.sqrt(mean_squared_error(observed[i], best_fit[i]))
        print(f"  {COMPONENT_COLS[i]}: RMSE = {rmse:.6f} nε")

    plt.figure(figsize=(12,6))
    for i in range(n_comp):
        plt.plot(time, residuals[i], color=colors[i], lw=1.2, label=COMPONENT_COLS[i])
    plt.axhline(0, color='k', ls='--')
    plt.xlabel('Time (hours)')
    plt.ylabel('Residual (nε)')
    plt.title(f'Station {station_idx} Residuals (time series)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    out_ts = os.path.join(OUT_DIR, f'residuals_station_{station_idx}_timeseries.png')
    plt.tight_layout()
    plt.savefig(out_ts, dpi=300)
    plt.close()
    print(f"Saved residual time-series: {out_ts}")

    plt.figure(figsize=(12,6))
    for i in range(n_comp):
        plt.subplot(2,int(np.ceil(n_comp/2)),i+1)
        sns.histplot(residuals[i], bins=40, kde=True)
        plt.title(COMPONENT_COLS[i])
        plt.xlabel('Residual (nε)')
    plt.suptitle(f'Station {station_idx} Residual Distributions')
    out_hist = os.path.join(OUT_DIR, f'residuals_station_{station_idx}_histograms.png')
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(out_hist, dpi=300)
    plt.close()
    print(f"Saved residual histograms: {out_hist}")
    gc.collect()
    return residuals

# Pairwise scatter of posterior (if many samples, sample randomly to N)
def plot_pairwise_scatter(posterior_flat, param_names, out_png="posterior_pairwise.png", nmax=5000):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(posterior_flat, columns=param_names)
    if df.shape[0] > nmax:
        df = df.sample(n=nmax, random_state=42)
    # Use pairplot (diag = kde)
    g = sns.pairplot(df, corner=False, diag_kind='kde', plot_kws={'s':10, 'alpha':0.5})
    g.figure.suptitle("Posterior pairwise scatter (sampled)", y=1.02)
    g.figure.set_size_inches(12, 12)
    g.savefig(out_png, dpi=200)
    print("Saved pairwise plot:", out_png)
    


# -------------------------
# Main
# -------------------------
def main():
    post = posterior_flat

    # Pairwise scatter plot
    plot_pairwise_scatter(post, param_names, out_png=os.path.join(OUT_DIR, "posterior_pairwise.png"), nmax=5000)

    # Predictive fits for all stations
    all_observed, all_bestfit, time, COMPONENT_COLS = plot_predictive_multimodal(
        post, logps, param_names, n_samples_per_comp=NSAMP_UNCERT
    )

    Ns = len(all_observed)
    for s in range(Ns):
        station_cols = COMPONENT_COLS[s*6:(s+1)*6]
        plot_residuals_combined(all_observed[s], all_bestfit[s], time,
                                station_idx=s+1,
                                COMPONENT_COLS=[sc.replace('_nanostrain','') for sc in station_cols])

    print("Multi-station post-processing complete. Figures saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
