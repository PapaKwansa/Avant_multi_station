"""
Bayesian mode analysis with predictive uncertainty including sigma.

Requirements:
- multi_stations_input.py
- forward_model_multi_station.py
- multi_station_sampled_params.npy (from run_inversion_multi_station.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

import multi_stations_input
from forward_model_multi_station import forward_model_multi_station

# ------------------ CONFIG ------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD POSTERIOR ------------------
post_file = "multi_station_sampled_params.npy"  # from run_inversion_multi_station.py
post = np.load(post_file, allow_pickle=True)  # shape: (nchains, niters, nparams)
post = np.vstack(post)  # flatten chains
print(f"Loaded posterior. Flattened shape: {post.shape}")

param_names = ['a', 'b', 'c', 'E', 'sigma']

# ------------------ TRACE PLOT ------------------
def plot_trace(flat_params, param_names):
    plt.figure(figsize=(12, 8))
    for i, pname in enumerate(param_names):
        plt.subplot(len(param_names), 1, i + 1)
        plt.plot(flat_params[:, i], alpha=0.5, color='tab:blue')
        plt.ylabel(pname)
        plt.grid(True)
    plt.xlabel("Sample index")
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "pydream_traces.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved trace plot: {out_file}")

# ------------------ PARAMETER DISTRIBUTIONS (KDE) ------------------
def plot_kde(flat_params, param_names):
    plt.figure(figsize=(12, 8))
    for i, pname in enumerate(param_names):
        plt.subplot(len(param_names), 1, i + 1)
        sns.kdeplot(flat_params[:, i], fill=True, color='tab:green')
        plt.xlabel(pname)
        plt.grid(True)
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "pydream_parameters_kde.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved parameter distributions (KDE): {out_file}")

# ------------------ PREDICTIVE UNCERTAINTY ------------------
def plot_predictive_with_uncertainty(flat_params, param_names, n_samples=500):
    """Posterior predictive bands including sigma uncertainty and observed points."""
    # Subsample posterior
    idxs = np.random.choice(len(flat_params), size=min(n_samples, len(flat_params)), replace=False)
    sampled_post = flat_params[idxs]

    # Load input parameters
    params = multi_stations_input.read_input()
    x_prime = np.asarray(params['x_prime'])
    y_prime = np.asarray(params['y_prime'])
    x0_prime = params['x0_prime']
    y0_prime = params['y0_prime']
    z = params['z']
    nu = params['nu']
    h = params['h']
    theta_deg = params['theta_deg']
    alpha = params.get('alpha', None)
    time = np.array(params['time'])
    Ns = len(x_prime)

    base_components = [
        "Epsilon_XX_nanostrain",
        "Epsilon_YY_nanostrain",
        "Epsilon_ZZ_nanostrain",
        "Epsilon_XY_nanostrain",
        "Epsilon_XZ_nanostrain",
        "Epsilon_YZ_nanostrain",
    ]
    COMPONENT_COLS = []
    for comp in base_components:
        for s in range(Ns):
            COMPONENT_COLS.append(f"{comp}_FS{s+1:02d}")

    # Load observed data
    obs_csv = os.path.join(os.path.dirname(__file__), 'strain_dataset_output_multi.csv')
    df_obs = pd.read_csv(obs_csv)

    # Build ensemble dictionary
    ensemble_dict = {col: [] for col in COMPONENT_COLS}
    for theta in sampled_post:
        a, b, c, E, sigma = theta
        df_pred = forward_model_multi_station(
            pmax=params['pmax'], tpeak=params['tpeak'], d=params['d'], time=time,
            x_prime=x_prime, y_prime=y_prime,
            x0_prime=x0_prime, y0_prime=y0_prime,
            z=z, a=a, b=b, c=c,
            nu=nu, h=h, E=E, theta_deg=theta_deg,
            alpha=alpha
        )
        noisy_pred = df_pred[COMPONENT_COLS].values.T + np.random.normal(0, sigma, df_pred[COMPONENT_COLS].values.T.shape)
        for i, col in enumerate(COMPONENT_COLS):
            ensemble_dict[col].append(noisy_pred[i])

    # Plot per station
    for s in range(Ns):
        plt.figure(figsize=(12, 8))
        for comp in base_components:
            col_name = f"{comp}_FS{s+1:02d}"
            data = np.array(ensemble_dict[col_name])
            mean = data.mean(axis=0)
            lower = np.percentile(data, 5, axis=0)
            upper = np.percentile(data, 95, axis=0)
            plt.fill_between(time, lower, upper, alpha=0.3, label=f"{comp} 5-95% band")
            plt.plot(time, mean, '-', label=f"{comp} mean")
            plt.plot(time, df_obs[col_name].values, 'o', label=f"{comp} observed")
        plt.xlabel("Time")
        plt.ylabel("Nanostrain")
        plt.title(f"Station {s+1} posterior predictive bands")
        # plt.legend()
        plt.legend(
            loc='upper center',         # place legend above or below the plot
            bbox_to_anchor=(0.5, -0.15), # coordinates relative to axes (x=0.5 centered, y=-0.15 below)
            ncol=3,                     # number of columns in legend
            fontsize=10,
            frameon=False               # optional: remove box around legend
        )
        plt.tight_layout()  # ensures the plot adjusts for the legend

        plt.grid(True)
        out_file = os.path.join(OUTPUT_DIR, f"predictive_fit_station_{s+1}_combined.png")
        plt.savefig(out_file, dpi=200)
        plt.close()
        print(f"Saved predictive plot: {out_file}")

# ------------------ RESIDUAL PLOTS ------------------
def plot_residuals(flat_params, param_names, n_samples=500):
    """Compare posterior predictive to synthetic observed data."""
    params = multi_stations_input.read_input()
    x_prime = np.asarray(params['x_prime'])
    y_prime = np.asarray(params['y_prime'])
    x0_prime = params['x0_prime']
    y0_prime = params['y0_prime']
    z = params['z']
    nu = params['nu']
    h = params['h']
    theta_deg = params['theta_deg']
    alpha = params.get('alpha', None)
    time = np.array(params['time'])
    Ns = len(x_prime)

    base_components = [
        "Epsilon_XX_nanostrain",
        "Epsilon_YY_nanostrain",
        "Epsilon_ZZ_nanostrain",
        "Epsilon_XY_nanostrain",
        "Epsilon_XZ_nanostrain",
        "Epsilon_YZ_nanostrain",
    ]
    COMPONENT_COLS = []
    for comp in base_components:
        for s in range(Ns):
            COMPONENT_COLS.append(f"{comp}_FS{s+1:02d}")

    # Load synthetic observed data
    obs_csv = os.path.join(os.path.dirname(__file__), 'strain_dataset_output_multi.csv')
    df_obs = pd.read_csv(obs_csv)

    # Subsample posterior for residual computation
    idxs = np.random.choice(len(flat_params), size=min(n_samples, len(flat_params)), replace=False)
    sampled_post = flat_params[idxs]

    for s in range(Ns):
        plt.figure(figsize=(12, 8))
        for comp in base_components:
            col_name = f"{comp}_FS{s+1:02d}"
            obs = df_obs[col_name].values
            ensemble = []
            for theta in sampled_post:
                a, b, c, E, sigma = theta
                df_pred = forward_model_multi_station(
                    pmax=params['pmax'], tpeak=params['tpeak'], d=params['d'], time=time,
                    x_prime=x_prime, y_prime=y_prime,
                    x0_prime=x0_prime, y0_prime=y0_prime,
                    z=z, a=a, b=b, c=c,
                    nu=nu, h=h, E=E, theta_deg=theta_deg,
                    alpha=alpha
                )
                noisy_pred = df_pred[col_name].values + np.random.normal(0, sigma, len(time))
                ensemble.append(noisy_pred)
            ensemble = np.array(ensemble)
            mean_pred = ensemble.mean(axis=0)
            plt.plot(time, obs, 'o', label=f"{comp} observed")
            plt.plot(time, mean_pred, '-', label=f"{comp} predicted")
        plt.xlabel("Time")
        plt.ylabel("Nanostrain")
        plt.title(f"Residual comparison: Station {s+1}")
        plt.legend()
        plt.grid(True)
        out_file = os.path.join(OUTPUT_DIR, f"residuals_station_{s+1}.png")
        plt.savefig(out_file, dpi=200)
        plt.close()
        print(f"Saved residual plot: {out_file}")

# ------------------ MAIN ------------------
def main():
    plot_trace(post, param_names)
    plot_kde(post, param_names)
    plot_predictive_with_uncertainty(post, param_names, n_samples=500)
    plot_residuals(post, param_names, n_samples=500)
    print("All plots saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
