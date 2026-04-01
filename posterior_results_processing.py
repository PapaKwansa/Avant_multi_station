#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-processing for PyDREAM inversion:
    parameters = [a, b, c, E, theta_deg, log10_sigma_strain]

This script:
1) Loads posterior samples and log-probabilities.
2) Builds high-quality posterior histograms (a, b, c, E, theta, sigma_strain).
3) Plots a correlation matrix of posterior parameters.
4) Plots predictive fits with:
       - epistemic uncertainty (parameter variability),
       - aleatoric uncertainty (inferred strain noise sigma).
5) Saves a JSON summary.

All strain ordering is kept consistent with the forward model:
    eXX for all stations,
    eYY for all stations,
    eXY for all stations,
    eZZ for all stations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import multi_stations_input as input_data
import bayesian_inversion_multi_station as bi
from forward_model_multi_station import forward_model_multi_station

# ============================================================
# High-quality plot settings
# ============================================================

plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.2,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.dpi": 220,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ============================================================
# User settings / file names
# ============================================================

POSTERIOR_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"

BASE_DIR = os.path.dirname(__file__)
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

HIST_OUTFILE = "posterior_histograms_geom_E_theta_sigma.png"
CORR_OUTFILE = "posterior_correlation_matrix.png"
SUMMARY_JSON = "posterior_processing_summary_geom_E_theta_sigma.json"
PLOT_PREFIX = "predictive_fit_station_geom_E_theta_sigma"

# Posterior selection
TOP_PERCENT = 5.0          # for histograms and correlation
N_POSTERIOR_SAMPLES = 1500 # for predictive fits

LOW = 5
HIGH = 95

# ============================================================
# Load inversion input (same geometry / stations as inversion)
# ============================================================

params = input_data.read_input()

time = np.asarray(params["time"], dtype=float)
x_prime = np.asarray(params["x_prime"], dtype=float)
y_prime = np.asarray(params["y_prime"], dtype=float)
z = np.asarray(params["z"], dtype=float)
station_names = np.asarray(params["station_names"], dtype=str)

# These are *baseline* geometry values used in the inversion priors,
# but the inversion itself inferred a, b, c.
a_baseline = float(params["a_fixed"])
b_baseline = float(params["b_fixed"])
c_baseline = float(params["c_fixed"])

x0_prime = float(params["x0_prime"])
y0_prime = float(params["y0_prime"])

nu = float(params["nu"])
alpha = params.get("alpha", None)
pmax = float(params["pmax"])
tpeak = float(params["tpeak"])
d = float(params["d"])
h = float(params["h"])

# Simple pressure history for volume (if needed)
delta_P = pmax * np.sin(np.pi * time / np.max(time))

# ============================================================
# Helpers
# ============================================================

COMPONENTS = ["eXX", "eYY", "eXY", "eZZ"]
COMPONENT_LABELS = [
    r"$\varepsilon_{xx}$",
    r"$\varepsilon_{yy}$",
    r"$\varepsilon_{xy}$",
    r"$\varepsilon_{zz}$",
]

LEGACY_COMPONENT_MAP = {
    "eXX": "Epsilon_XX_nanostrain",
    "eYY": "Epsilon_YY_nanostrain",
    "eXY": "Epsilon_XY_nanostrain",
    "eZZ": "Epsilon_ZZ_nanostrain",
}

def expected_component_cols(station_names_local):
    return [f"{comp}_{sname}" for comp in COMPONENTS for sname in station_names_local]

def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot

def flatten_columns(df, cols):
    return np.concatenate([df[c].values.astype(float) for c in cols])

def standardize_observed_strain(df, station_names_local):
    """
    Make observed strain columns match the forward-model naming:
        eXX_S1, eYY_S1, ..., eZZ_Sn
    Handles legacy names like Epsilon_XX_nanostrain_S1, etc.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "time_s" not in df.columns:
        raise RuntimeError("Observed file must contain a 'time_s' column")

    exp_cols = expected_component_cols(station_names_local)

    # Case 1: already in new-style naming
    if all(c in df.columns for c in exp_cols):
        return df, exp_cols

    # Case 2: legacy naming -> rename to new-style
    rename_map = {}
    for comp, legacy_base in LEGACY_COMPONENT_MAP.items():
        for sname in station_names_local:
            legacy = f"{legacy_base}_{sname}"
            new = f"{comp}_{sname}"
            if legacy in df.columns and new not in df.columns:
                rename_map[legacy] = new

    if rename_map:
        df = df.rename(columns=rename_map)
        if all(c in df.columns for c in exp_cols):
            return df, exp_cols

    # Case 3: generic N-channel file in file order
    strain_cols = [c for c in df.columns if c not in ["time_s", "Volume"]]
    if len(strain_cols) == len(exp_cols):
        rename_map = {old: new for old, new in zip(strain_cols, exp_cols)}
        df = df.rename(columns=rename_map)
        return df, exp_cols

    raise RuntimeError(
        "Observed strain columns do not match expected layout.\n"
        f"Expected {len(exp_cols)} strain columns.\n"
        f"Available columns: {list(df.columns)}"
    )

# ============================================================
# Load observed strain
# ============================================================

obs_df_raw = pd.read_csv(OBSERVED_FILE)
obs_df_raw.columns = [str(c).strip() for c in obs_df_raw.columns]

obs_df, EXPECTED_COLS = standardize_observed_strain(obs_df_raw, station_names)
observed_vector = flatten_columns(obs_df, EXPECTED_COLS)

print(f"[INFO] Observed vector shape: {observed_vector.shape}")

# ============================================================
# Load posterior
# ============================================================

posterior_samples = np.load(POSTERIOR_FILE)
logps = np.load(LOGP_FILE).reshape(-1)

if posterior_samples.ndim != 3:
    raise RuntimeError(f"Expected posterior samples to be 3D, got {posterior_samples.shape}")

nchains, niter, nparams = posterior_samples.shape
if nparams != 6:
    raise RuntimeError("Expected 6 parameters [a, b, c, E, theta_deg, log10_sigma_strain]")

posterior_flat = posterior_samples.reshape(-1, 6)
posterior_flat = posterior_flat[:len(logps)]

# Names for convenience
PARAM_NAMES = ["a", "b", "c", "E", "theta_deg", "log10_sigma_strain"]

# ============================================================
# Posterior selection utilities
# ============================================================

def get_top_posterior(posterior_flat, logps, top_percent):
    sorted_idx = np.argsort(logps)[::-1]
    n_keep = max(1, int(len(sorted_idx) * (top_percent / 100.0)))
    return posterior_flat[sorted_idx[:n_keep]], logps[sorted_idx[:n_keep]]

# ============================================================
# Posterior histograms
# ============================================================

def plot_posterior_histograms(posterior_flat, logps):
    top_samples, top_logps = get_top_posterior(posterior_flat, logps, TOP_PERCENT)
    print(f"[INFO] Using top {TOP_PERCENT}% posterior samples ({len(top_samples)} draws) for histograms.")

    # Convert log10_sigma_strain to sigma_strain for plotting
    sigma_strain = 10.0 ** top_samples[:, 5]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.ravel()

    labels = [
        "a (m)",
        "b (m)",
        "c (m)",
        "Young's modulus E (Pa)",
        "Orientation θ (deg)",
        "Strain noise σ (nε)",
    ]

    data_list = [
        top_samples[:, 0],
        top_samples[:, 1],
        top_samples[:, 2],
        top_samples[:, 3],
        top_samples[:, 4],
        sigma_strain,
    ]

    # We will keep references to last handles for a clean legend below
    mean_line = median_line = map_line = None

    # MAP from full posterior
    map_idx = int(np.argmax(logps))
    map_params = posterior_flat[map_idx]
    sigma_map = 10.0 ** map_params[5]

    for i, ax in enumerate(axes):
        data = data_list[i]
        label = labels[i]

        mean_val = float(np.mean(data))
        median_val = float(np.median(data))
        map_val = float(map_params[i] if i < 5 else sigma_map)

        n, bins, patches = ax.hist(
            data,
            bins=40,
            density=True,
            color="steelblue",
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
            label="Posterior density",
        )

        mean_line = ax.axvline(mean_val, color="red", linestyle="--", linewidth=3, label="Mean")
        median_line = ax.axvline(median_val, color="green", linestyle="-.", linewidth=3, label="Median")
        map_line = ax.axvline(map_val, color="purple", linestyle=":", linewidth=3.5, label="MAP")

        ax.set_title(label)
        ax.set_ylabel("Posterior density")

    # Legend beneath the entire figure
    legend_handles = [mean_line, median_line, map_line]
    legend_labels = ["Mean", "Median", "MAP"]

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.suptitle("Posterior distributions: geometry, stiffness, orientation, and strain noise", y=0.995)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=15,
    )

    plt.savefig(HIST_OUTFILE)
    plt.close()
    print(f"[INFO] Posterior histograms saved to {HIST_OUTFILE}")

# ============================================================
# Correlation matrix of posterior parameters
# ============================================================

def plot_correlation_matrix(posterior_flat, logps):
    top_samples, _ = get_top_posterior(posterior_flat, logps, TOP_PERCENT)
    # Convert log10_sigma_strain to sigma_strain
    sigma_strain = 10.0 ** top_samples[:, 5]
    data = np.column_stack([top_samples[:, 0:5], sigma_strain])

    names = ["a", "b", "c", "E", "theta_deg", "sigma_strain"]

    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(names)):
            val = corr[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="black" if abs(val) < 0.75 else "white",
                fontsize=12,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontweight="bold")

    ax.set_title("Correlation matrix: posterior parameters", pad=16)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(CORR_OUTFILE)
    plt.close()
    print(f"[INFO] Posterior correlation matrix saved to {CORR_OUTFILE}")

# ============================================================
# Predictive fits with epistemic + aleatoric uncertainty
# ============================================================

def plot_predictive_fit(posterior_flat, logps):
    rng = np.random.default_rng(42)

    # MAP parameters
    map_idx = int(np.argmax(logps))
    a_map, b_map, c_map, E_map, theta_map, log10_sigma_map = posterior_flat[map_idx]
    sigma_map = 10.0 ** log10_sigma_map

    df_map = forward_model_multi_station(
        pmax=pmax, tpeak=tpeak, d=d, time=time,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0_prime, y0_prime=y0_prime,
        z=z, a=a_map, b=b_map, c=c_map,
        nu=nu, h=h, E=E_map, theta_deg=theta_map,
        alpha=alpha, station_names=station_names,
    )

    n_draws = min(N_POSTERIOR_SAMPLES, len(posterior_flat))
    draw_idx = rng.choice(len(posterior_flat), size=n_draws, replace=False)
    posterior_subset = posterior_flat[draw_idx]

    print(f"[INFO] Using {n_draws} posterior draws for predictive uncertainty.")
    print(f"[INFO] Using MAP sigma_strain={sigma_map:.3f} as aleatoric noise scale.")

    colors = plt.cm.tab10(np.linspace(0, 1, len(COMPONENTS)))

    for station_name in station_names:
        fig, ax = plt.subplots(figsize=(18, 10))

        for i, comp in enumerate(COMPONENTS):
            col = f"{comp}_{station_name}"

            param_curves = []
            total_curves = []

            for draw in posterior_subset:
                a_i, b_i, c_i, E_i, theta_i, log10_sigma_i = draw
                sigma_i = 10.0 ** log10_sigma_i

                df_i = forward_model_multi_station(
                    pmax=pmax, tpeak=tpeak, d=d, time=time,
                    x_prime=x_prime, y_prime=y_prime,
                    x0_prime=x0_prime, y0_prime=y0_prime,
                    z=z, a=a_i, b=b_i, c=c_i,
                    nu=nu, h=h, E=E_i, theta_deg=theta_i,
                    alpha=alpha, station_names=station_names,
                )

                curve = df_i[col].values.astype(float)
                param_curves.append(curve)

                # Aleatoric noise: use inferred sigma (here we use sigma_map
                # for a clean, single aleatoric scale; you could also use sigma_i)
                noise = rng.normal(0.0, sigma_map, size=len(curve))
                total_curves.append(curve + noise)

            param_curves = np.asarray(param_curves)
            total_curves = np.asarray(total_curves)

            lower_param = np.percentile(param_curves, LOW, axis=0)
            upper_param = np.percentile(param_curves, HIGH, axis=0)
            lower_total = np.percentile(total_curves, LOW, axis=0)
            upper_total = np.percentile(total_curves, HIGH, axis=0)

            # Total uncertainty band (epistemic + aleatoric)
            total_band = ax.fill_between(
                time, lower_total, upper_total,
                color=colors[i], alpha=0.14, zorder=1
            )

            # Epistemic band (parameter uncertainty only)
            epistemic_band = ax.fill_between(
                time, lower_param, upper_param,
                color=colors[i], alpha=0.55, zorder=2
            )

            # Observations
            ax.scatter(
                obs_df["time_s"].values.astype(float),
                obs_df[col].values.astype(float),
                facecolors="white",
                edgecolors="black",
                s=40,
                linewidths=1.0,
                zorder=5,
            )

            # MAP curve
            map_line = ax.plot(
                time,
                df_map[col].values.astype(float),
                linestyle="--",
                color=colors[i],
                linewidth=3.2,
                label=f"{COMPONENT_LABELS[i]} (MAP)",
                zorder=4,
            )[0]

        ax.set_title(f"Station {station_name} — Posterior (epistemic) and noise (aleatoric) uncertainty")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain (nε)")

        # Build legend: MAP components + epistemic + total
        handles, labels = ax.get_legend_handles_labels()

        epistemic_patch = Patch(facecolor="gray", alpha=0.55, label="Epistemic uncertainty")
        total_patch = Patch(facecolor="gray", alpha=0.14, label="Total uncertainty (epistemic + aleatoric)")

        handles.extend([epistemic_patch, total_patch])
        labels.extend(["Epistemic uncertainty", "Total uncertainty (epistemic + aleatoric)"])

        # Legend beneath the plot
        fig.subplots_adjust(bottom=0.22)
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=15,
        )

        out_name = f"{PLOT_PREFIX}_{station_name}_uncertainty.png"
        plt.savefig(out_name)
        plt.close()
        print(f"[INFO] Saved {out_name}")

# ============================================================
# Summary JSON
# ============================================================

def save_summary(posterior_flat, logps):
    map_idx = int(np.argmax(logps))
    a_map, b_map, c_map, E_map, theta_map, log10_sigma_map = posterior_flat[map_idx]
    sigma_map = 10.0 ** log10_sigma_map

    df_map = forward_model_multi_station(
        pmax=pmax, tpeak=tpeak, d=d, time=time,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0_prime, y0_prime=y0_prime,
        z=z, a=a_map, b=b_map, c=c_map,
        nu=nu, h=h, E=E_map, theta_deg=theta_map,
        alpha=alpha, station_names=station_names,
    )

    pred_vec = flatten_columns(df_map, EXPECTED_COLS)
    strain_r2 = compute_r2(observed_vector, pred_vec)

    # Simple posterior std estimates (after burn-in)
    burn_frac = 0.5
    burn_in = int(len(posterior_flat) * burn_frac)
    posterior_burn = posterior_flat[burn_in:]

    a_std = float(np.std(posterior_burn[:, 0]))
    b_std = float(np.std(posterior_burn[:, 1]))
    c_std = float(np.std(posterior_burn[:, 2]))
    E_std = float(np.std(posterior_burn[:, 3]))
    theta_std = float(np.std(posterior_burn[:, 4]))
    sigma_std = float(np.std(10.0 ** posterior_burn[:, 5]))

    summary = {
        "MAP": {
            "a": float(a_map),
            "b": float(b_map),
            "c": float(c_map),
            "E": float(E_map),
            "theta_deg": float(theta_map),
            "sigma_strain": float(sigma_map),
        },
        "STD": {
            "a": a_std,
            "b": b_std,
            "c": c_std,
            "E": E_std,
            "theta_deg": theta_std,
            "sigma_strain": sigma_std,
        },
        "geometry_baseline_for_priors": {
            "a_baseline": float(a_baseline),
            "b_baseline": float(b_baseline),
            "c_baseline": float(c_baseline),
        },
        "fit": {
            "strain_R2_MAP": float(strain_r2),
        },
        "n_posterior_draws_for_fit": int(min(N_POSTERIOR_SAMPLES, len(posterior_flat))),
        "top_percent_histogram_and_corr": float(TOP_PERCENT),
        "parameter_names": PARAM_NAMES,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Saved summary JSON to {SUMMARY_JSON}")
    print(f"[INFO] MAP strain R^2: {strain_r2:.4f}")

# ============================================================
# Main
# ============================================================

def main():
    plot_posterior_histograms(posterior_flat, logps)
    plot_correlation_matrix(posterior_flat, logps)
    plot_predictive_fit(posterior_flat, logps)
    save_summary(posterior_flat, logps)
    print("[INFO] Post-processing complete.")

if __name__ == "__main__":
    main()
