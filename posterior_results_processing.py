#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-processing for the multi-station PyDREAM inversion.

Inversion parameter vector:
    [a, b, theta_deg, x0_prime, y0_prime, log10_sigma_strain]

Fixed inputs from multi_stations_input:
    pmax, E, c, nu, h, d, tpeak, alpha, time, station geometry

This script:
1) Loads posterior samples and log-probabilities.
2) Keeps post-burn-in samples from each chain.
3) Builds posterior histograms using the full post-burn-in posterior.
4) Plots a correlation matrix of the posterior parameters.
5) Plots predictive fits with epistemic uncertainty and aleatoric noise.
6) Saves a JSON summary.

Notes:
- The histograms and correlations are based on the post-burn-in posterior,
  not the top-likelihood subset.
- Aleatoric noise is the inferred sigma_strain parameter.
- Strain ordering is component-major:
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
from forward_model_multi_station import forward_model_multi_station

# ============================================================
# Plot settings
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
# Files / settings
# ============================================================

POSTERIOR_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"

BASE_DIR = os.path.dirname(__file__)
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

HIST_OUTFILE = "posterior_histograms_ab_theta_center_sigma.png"
CORR_OUTFILE = "posterior_correlation_matrix_ab_theta_center_sigma.png"
SUMMARY_JSON = "posterior_processing_summary_ab_theta_center_sigma.json"
PLOT_PREFIX = "predictive_fit_station_ab_theta_center_sigma"

BURN_FRAC = 0.5
N_POSTERIOR_SAMPLES = 400
LOW = 5
HIGH = 95

# ============================================================
# Load inversion inputs
# ============================================================

params = input_data.read_input()

time = np.asarray(params["time"], dtype=float)
x_prime = np.asarray(params["x_prime"], dtype=float)
y_prime = np.asarray(params["y_prime"], dtype=float)
z = np.asarray(params["z"], dtype=float)
station_names = np.asarray(params["station_names"], dtype=str)

pmax = float(params["pmax"])
E_fixed = float(params["E"])
c_fixed = float(params["c_fixed"])
nu = float(params["nu"])
h = float(params["h"])
d = float(params["d"])
tpeak = float(params["tpeak"])
alpha = params.get("alpha", None)

x0_true = float(params["x0_prime"])
y0_true = float(params["y0_prime"])

print(f"[INFO] Fixed inputs: pmax={pmax:.6g}, E={E_fixed:.6g}, c={c_fixed:.6g}")
print(f"[INFO] True center from input: x0={x0_true:.3f}, y0={y0_true:.3f}")

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

PARAM_NAMES = ["a", "b", "theta_deg", "x0_prime", "y0_prime", "log10_sigma_strain"]


def expected_component_cols(station_names_local):
    return [f"{comp}_{sname}" for comp in COMPONENTS for sname in station_names_local]


def flatten_columns(df, cols):
    return np.concatenate([df[c].values.astype(float) for c in cols])


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot


def standardize_observed_strain(df, station_names_local):
    """Rename observed strain columns to match the forward-model convention."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "time_s" not in df.columns:
        raise RuntimeError("Observed file must contain a 'time_s' column")

    exp_cols = expected_component_cols(station_names_local)

    if all(c in df.columns for c in exp_cols):
        return df, exp_cols

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

    strain_cols = [c for c in df.columns if c not in ["time_s", "Volume"]]
    if len(strain_cols) == len(exp_cols):
        rename_map = {old: new for old, new in zip(strain_cols, exp_cols)}
        df = df.rename(columns=rename_map)
        return df, exp_cols

    raise RuntimeError(
        "Observed file columns do not match expected layout.\n"
        f"Expected {len(exp_cols)} strain columns.\n"
        f"Available columns: {list(df.columns)}"
    )


def chainwise_burn_in(samples_3d, logps_2d, burn_frac):
    """Apply burn-in per chain, then flatten."""
    if samples_3d.ndim != 3:
        raise RuntimeError(f"Expected posterior samples to be 3D, got {samples_3d.shape}")
    if logps_2d.ndim != 2:
        raise RuntimeError(f"Expected logps to be 2D, got {logps_2d.shape}")

    nchains, niter, nparams = samples_3d.shape
    burn_in = int(niter * burn_frac)

    if logps_2d.shape[0] != nchains or logps_2d.shape[1] != niter:
        raise RuntimeError(
            f"Sample/logp shape mismatch: samples={samples_3d.shape}, logps={logps_2d.shape}"
        )

    samples_burn = samples_3d[:, burn_in:, :].reshape(-1, nparams)
    logps_burn = logps_2d[:, burn_in:].reshape(-1)
    return samples_burn, logps_burn, burn_in


def get_map_params(samples_burn, logps_burn):
    idx = int(np.argmax(logps_burn))
    return samples_burn[idx], idx


# ============================================================
# Load observed strain
# ============================================================

obs_df_raw = pd.read_csv(OBSERVED_FILE)
obs_df_raw.columns = [str(c).strip() for c in obs_df_raw.columns]
obs_df, EXPECTED_COLS = standardize_observed_strain(obs_df_raw, station_names)
observed_vector = flatten_columns(obs_df, EXPECTED_COLS)
print(f"[INFO] Observed vector shape: {observed_vector.shape}")

# ============================================================
# Load posterior and apply burn-in
# ============================================================

posterior_samples = np.load(POSTERIOR_FILE)
logps_flat = np.load(LOGP_FILE).reshape(-1)

if posterior_samples.ndim != 3:
    raise RuntimeError(f"Expected posterior samples to be 3D, got {posterior_samples.shape}")

nchains, niter, nparams = posterior_samples.shape
if nparams != 6:
    raise RuntimeError(
        "Expected 6 parameters [a, b, theta_deg, x0_prime, y0_prime, log10_sigma_strain]"
    )

if logps_flat.size != nchains * niter:
    raise RuntimeError(
        f"logps size mismatch: got {logps_flat.size}, expected {nchains * niter}"
    )

logps_2d = logps_flat.reshape(nchains, niter)
posterior_burn, logps_burn, burn_in = chainwise_burn_in(posterior_samples, logps_2d, BURN_FRAC)
print(f"[INFO] Using burn-in of {burn_in} iterations per chain.")
print(f"[INFO] Post-burn-in samples: {posterior_burn.shape[0]}")

# ============================================================
# Posterior histograms
# ============================================================

def plot_posterior_histograms(samples, logps):
    sigma_strain = 10.0 ** samples[:, 5]
    map_params, map_idx = get_map_params(samples, logps)
    sigma_map = 10.0 ** map_params[5]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.ravel()

    labels = [
        "a (m)",
        "b (m)",
        "Orientation θ (deg)",
        "x0 (m)",
        "y0 (m)",
        "Strain noise σ (nε)",
    ]

    data_list = [
        samples[:, 0],
        samples[:, 1],
        samples[:, 2],
        samples[:, 3],
        samples[:, 4],
        sigma_strain,
    ]

    mean_line = median_line = map_line = None

    for i, ax in enumerate(axes):
        data = data_list[i]
        mean_val = float(np.mean(data))
        median_val = float(np.median(data))
        map_val = float(map_params[i] if i < 5 else sigma_map)

        ax.hist(
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
        ax.set_title(labels[i])
        ax.set_ylabel("Posterior density")

    fig.suptitle("Posterior distributions: geometry, orientation, center, and strain noise", y=0.995)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.legend(
        [mean_line, median_line, map_line],
        ["Mean", "Median", "MAP"],
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
# Correlation matrix
# ============================================================

def plot_correlation_matrix(samples):
    sigma_strain = 10.0 ** samples[:, 5]
    data = np.column_stack([samples[:, 0:5], sigma_strain])
    names = ["a", "b", "theta_deg", "x0_prime", "y0_prime", "sigma_strain"]
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
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if abs(val) < 0.75 else "white",
                fontsize=12,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontweight="bold")
    ax.set_title("Correlation matrix: post-burn-in posterior parameters", pad=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    plt.savefig(CORR_OUTFILE)
    plt.close()
    print(f"[INFO] Posterior correlation matrix saved to {CORR_OUTFILE}")


# ============================================================
# Predictive fits
# ============================================================

def plot_predictive_fit(samples, logps):
    rng = np.random.default_rng(42)

    map_params, map_idx = get_map_params(samples, logps)
    a_map, b_map, theta_map, x0_map, y0_map, log10_sigma_map = map_params
    sigma_map = 10.0 ** log10_sigma_map

    df_map = forward_model_multi_station(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_map,
        y0_prime=y0_map,
        z=z,
        a=a_map,
        b=b_map,
        c=c_fixed,
        nu=nu,
        h=h,
        E=E_fixed,
        theta_deg=theta_map,
        alpha=alpha,
        station_names=station_names,
    )

    n_draws = min(N_POSTERIOR_SAMPLES, len(samples))
    draw_idx = rng.choice(len(samples), size=n_draws, replace=False)
    posterior_subset = samples[draw_idx]

    print(f"[INFO] Using {n_draws} post-burn-in posterior draws for predictive uncertainty.")
    print(f"[INFO] Using MAP sigma_strain={sigma_map:.3f} as aleatoric noise scale.")

    colors = plt.cm.tab10(np.linspace(0, 1, len(COMPONENTS)))

    for station_name in station_names:
        fig, ax = plt.subplots(figsize=(18, 10))

        for i, comp in enumerate(COMPONENTS):
            col = f"{comp}_{station_name}"
            param_curves = []
            total_curves = []

            for draw in posterior_subset:
                a_i, b_i, theta_i, x0_i, y0_i, log10_sigma_i = draw
                sigma_i = 10.0 ** log10_sigma_i

                df_i = forward_model_multi_station(
                    pmax=pmax,
                    tpeak=tpeak,
                    d=d,
                    time=time,
                    x_prime=x_prime,
                    y_prime=y_prime,
                    x0_prime=x0_i,
                    y0_prime=y0_i,
                    z=z,
                    a=a_i,
                    b=b_i,
                    c=c_fixed,
                    nu=nu,
                    h=h,
                    E=E_fixed,
                    theta_deg=theta_i,
                    alpha=alpha,
                    station_names=station_names,
                )

                curve = df_i[col].values.astype(float)
                param_curves.append(curve)

                noise = rng.normal(0.0, sigma_i, size=len(curve))
                total_curves.append(curve + noise)

            param_curves = np.asarray(param_curves)
            total_curves = np.asarray(total_curves)

            lower_param = np.percentile(param_curves, LOW, axis=0)
            upper_param = np.percentile(param_curves, HIGH, axis=0)
            lower_total = np.percentile(total_curves, LOW, axis=0)
            upper_total = np.percentile(total_curves, HIGH, axis=0)

            ax.fill_between(time, lower_total, upper_total, color=colors[i], alpha=0.14, zorder=1)
            ax.fill_between(time, lower_param, upper_param, color=colors[i], alpha=0.55, zorder=2)

            ax.scatter(
                obs_df["time_s"].values.astype(float),
                obs_df[col].values.astype(float),
                facecolors="white",
                edgecolors="black",
                s=40,
                linewidths=1.0,
                zorder=5,
            )

            ax.plot(
                time,
                df_map[col].values.astype(float),
                linestyle="--",
                color=colors[i],
                linewidth=3.2,
                label=f"{COMPONENT_LABELS[i]} (MAP)",
                zorder=4,
            )

        ax.set_title(f"Station {station_name} — Posterior and noise uncertainty")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain (nε)")

        handles, labels = ax.get_legend_handles_labels()
        epistemic_patch = Patch(facecolor="gray", alpha=0.55, label="Epistemic uncertainty")
        total_patch = Patch(facecolor="gray", alpha=0.14, label="Total uncertainty (epistemic + aleatoric)")
        handles.extend([epistemic_patch, total_patch])
        labels.extend(["Epistemic uncertainty", "Total uncertainty (epistemic + aleatoric)"])

        fig.subplots_adjust(bottom=0.22)
        fig.legend(
            handles,
            labels,
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

def save_summary(samples, logps):
    map_params, map_idx = get_map_params(samples, logps)
    a_map, b_map, theta_map, x0_map, y0_map, log10_sigma_map = map_params
    sigma_map = 10.0 ** log10_sigma_map

    df_map = forward_model_multi_station(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_map,
        y0_prime=y0_map,
        z=z,
        a=a_map,
        b=b_map,
        c=c_fixed,
        nu=nu,
        h=h,
        E=E_fixed,
        theta_deg=theta_map,
        alpha=alpha,
        station_names=station_names,
    )

    pred_vec = flatten_columns(df_map, EXPECTED_COLS)
    strain_r2 = compute_r2(observed_vector, pred_vec)

    a_std = float(np.std(samples[:, 0]))
    b_std = float(np.std(samples[:, 1]))
    theta_std = float(np.std(samples[:, 2]))
    x0_std = float(np.std(samples[:, 3]))
    y0_std = float(np.std(samples[:, 4]))
    sigma_std = float(np.std(10.0 ** samples[:, 5]))

    summary = {
        "MAP": {
            "a": float(a_map),
            "b": float(b_map),
            "theta_deg": float(theta_map),
            "x0_prime": float(x0_map),
            "y0_prime": float(y0_map),
            "sigma_strain": float(sigma_map),
        },
        "STD": {
            "a": a_std,
            "b": b_std,
            "theta_deg": theta_std,
            "x0_prime": x0_std,
            "y0_prime": y0_std,
            "sigma_strain": sigma_std,
        },
        "fixed_inputs": {
            "pmax": pmax,
            "E": E_fixed,
            "c": c_fixed,
            "nu": nu,
            "h": h,
            "d": d,
            "tpeak": tpeak,
            "alpha": alpha,
        },
        "fit": {
            "strain_R2_MAP": float(strain_r2),
        },
        "burn_in_fraction": float(BURN_FRAC),
        "n_post_burn_samples": int(len(samples)),
        "n_draws_for_predictive_fit": int(min(N_POSTERIOR_SAMPLES, len(samples))),
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
    plot_posterior_histograms(posterior_burn, logps_burn)
    plot_correlation_matrix(posterior_burn)
    plot_predictive_fit(posterior_burn, logps_burn)
    save_summary(posterior_burn, logps_burn)
    print("[INFO] Post-processing complete.")


if __name__ == "__main__":
    main()
