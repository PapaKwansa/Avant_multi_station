"""
Post-processing for the corrected PyDREAM inversion (E, theta only).

Geometry is FIXED:
    a = a_fixed
    b = b_fixed
    c = c_fixed

Posterior parameters saved by the inversion:
    [E, theta]

This script:
1) plots posterior histograms,
2) plots predictive fits for each station,
3) saves a JSON summary.

It uses the same strain ordering as the forward model:
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
    "figure.dpi": 200,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

# ============================================================
# User settings
# ============================================================

POSTERIOR_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"

BASE_DIR = os.path.dirname(__file__)
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")

HIST_OUTFILE = "posterior_histograms_E_theta.png"
SUMMARY_JSON = "posterior_processing_summary.json"
PLOT_PREFIX = "predictive_fit_station"

# Top X% of posterior (by logp) used for histograms
TOP_PERCENT = 5.0
N_POSTERIOR_SAMPLES = 1500

LOW = 5
HIGH = 95

SHOW_TRUE = False  # only for synthetic tests

# ============================================================
# Load inversion input
# ============================================================

params = input_data.read_input()

time = np.asarray(params["time"], dtype=float)
x_prime = np.asarray(params["x_prime"], dtype=float)
y_prime = np.asarray(params["y_prime"], dtype=float)
z = np.asarray(params["z"], dtype=float)
station_names = np.asarray(params["station_names"], dtype=str)

a_fixed = float(params["a_fixed"])
b_fixed = float(params["b_fixed"])
c_fixed = float(params["c_fixed"])

x0_prime = float(params["x0_prime"])
y0_prime = float(params["y0_prime"])

nu = float(params["nu"])
alpha = params.get("alpha", None)
pmax = float(params["pmax"])
tpeak = float(params["tpeak"])
d = float(params["d"])
h = float(params["h"])

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

PRIOR_RANGES = {
    "E": (1.0e9, 3.0e10),
    "theta": (-90.0, 90.0),
}

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
        eXX_S1, eYY_S1, ..., eZZ_S4
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

    # Case 3: generic 16-channel file in file order
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

sigma_noise = 0.1 * np.std(observed_vector)
print(f"[INFO] Observed vector shape: {observed_vector.shape}")
print(f"[INFO] Estimated noise scale: {sigma_noise:.3f}")

# ============================================================
# Load posterior
# ============================================================

posterior_samples = np.load(POSTERIOR_FILE)
logps = np.load(LOGP_FILE).reshape(-1)

if posterior_samples.ndim != 3:
    raise RuntimeError(f"Expected posterior samples to be 3D, got {posterior_samples.shape}")

nchains, niter, nparams = posterior_samples.shape
if nparams != 2:
    raise RuntimeError("Expected 2 parameters [E, theta]")

posterior_flat = posterior_samples.reshape(-1, 2)
posterior_flat = posterior_flat[:len(logps)]

# ============================================================
# Posterior histograms
# ============================================================

def plot_posterior_histograms(posterior_flat, logps):
    map_idx = int(np.argmax(logps))
    map_params = posterior_flat[map_idx]

    sorted_idx = np.argsort(logps)[::-1]
    n_keep = max(1, int(len(sorted_idx) * (TOP_PERCENT / 100.0)))
    posterior_top = posterior_flat[sorted_idx[:n_keep]]

    print(f"[INFO] Using top {TOP_PERCENT}% posterior samples ({n_keep} draws) for histograms.")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    names = ["E", "theta"]
    titles = ["Young's Modulus (E)", "Orientation (Theta)"]

    for i, ax in enumerate(axes):
        name = names[i]
        title = titles[i]
        data = posterior_top[:, i]

        mean_val = float(np.mean(data))
        median_val = float(np.median(data))

        # Histogram
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

        # Prior range
        lo, hi = PRIOR_RANGES[name]
        prior_patch = ax.axvspan(lo, hi, color="gold", alpha=0.18, label="Prior range")

        # Mean, median, MAP
        mean_line = ax.axvline(mean_val, color="red", linestyle="--", linewidth=3, label="Mean")
        median_line = ax.axvline(median_val, color="green", linestyle="-.", linewidth=3, label="Median")
        map_line = ax.axvline(map_params[i], color="purple", linestyle=":", linewidth=3.5, label="MAP")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Build a clean legend under the figure
    legend_handles = [
        mean_line,
        median_line,
        map_line,
        prior_patch,
    ]
    legend_labels = ["Mean", "Median", "MAP", "Prior range"]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=15,
    )

    plt.savefig(HIST_OUTFILE)
    plt.close()
    print(f"[INFO] Posterior histograms saved to {HIST_OUTFILE}")

# ============================================================
# Predictive fits
# ============================================================

def plot_predictive_fit(posterior_flat, logps):
    rng = np.random.default_rng(42)

    map_idx = int(np.argmax(logps))
    E_map, theta_map = posterior_flat[map_idx]

    df_map = forward_model_multi_station(
        pmax=pmax, tpeak=tpeak, d=d, time=time,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0_prime, y0_prime=y0_prime,
        z=z, a=a_fixed, b=b_fixed, c=c_fixed,
        nu=nu, h=h, E=E_map, theta_deg=theta_map,
        alpha=alpha, station_names=station_names,
    )

    n_draws = min(N_POSTERIOR_SAMPLES, len(posterior_flat))
    draw_idx = rng.choice(len(posterior_flat), size=n_draws, replace=False)
    posterior_subset = posterior_flat[draw_idx]

    print(f"[INFO] Using {n_draws} posterior draws for predictive uncertainty.")

    colors = plt.cm.tab10(np.linspace(0, 1, len(COMPONENTS)))

    for station_name in station_names:
        fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

        for i, comp in enumerate(COMPONENTS):
            col = f"{comp}_{station_name}"

            param_curves = []
            total_curves = []

            for draw in posterior_subset:
                E_i, theta_i = draw

                df_i = forward_model_multi_station(
                    pmax=pmax, tpeak=tpeak, d=d, time=time,
                    x_prime=x_prime, y_prime=y_prime,
                    x0_prime=x0_prime, y0_prime=y0_prime,
                    z=z, a=a_fixed, b=b_fixed, c=c_fixed,
                    nu=nu, h=h, E=E_i, theta_deg=theta_i,
                    alpha=alpha, station_names=station_names,
                )

                curve = df_i[col].values.astype(float)
                param_curves.append(curve)
                total_curves.append(curve + rng.normal(0.0, sigma_noise, size=len(curve)))

            param_curves = np.asarray(param_curves)
            total_curves = np.asarray(total_curves)

            lower_param = np.percentile(param_curves, LOW, axis=0)
            upper_param = np.percentile(param_curves, HIGH, axis=0)
            lower_total = np.percentile(total_curves, LOW, axis=0)
            upper_total = np.percentile(total_curves, HIGH, axis=0)

            # Total uncertainty band (epistemic + noise)
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

        ax.set_title(f"Station {station_name} — Posterior and total uncertainty")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain (nε)")
        ax.grid(True, alpha=0.3)

        # Build legend: MAP components + epistemic + total
        handles, labels = ax.get_legend_handles_labels()

        epistemic_patch = Patch(facecolor="gray", alpha=0.55, label="Epistemic uncertainty")
        total_patch = Patch(facecolor="gray", alpha=0.14, label="Total uncertainty")

        handles.extend([epistemic_patch, total_patch])
        labels.extend(["Epistemic uncertainty", "Total uncertainty"])

        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
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
    E_map, theta_map = posterior_flat[map_idx]

    df_map = forward_model_multi_station(
        pmax=pmax, tpeak=tpeak, d=d, time=time,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0_prime, y0_prime=y0_prime,
        z=z, a=a_fixed, b=b_fixed, c=c_fixed,
        nu=nu, h=h, E=E_map, theta_deg=theta_map,
        alpha=alpha, station_names=station_names,
    )

    pred_vec = flatten_columns(df_map, EXPECTED_COLS)
    strain_r2 = compute_r2(observed_vector, pred_vec)

    summary = {
        "MAP": {
            "E": float(E_map),
            "theta_deg": float(theta_map),
        },
        "priors": {
            "E": [1.0e9, 3.0e10],
            "theta_deg": [-90.0, 90.0],
        },
        "geometry_fixed": {
            "a": float(a_fixed),
            "b": float(b_fixed),
            "c": float(c_fixed),
        },
        "fit": {
            "strain_R2_MAP": float(strain_r2),
        },
        "n_posterior_draws_for_fit": int(min(N_POSTERIOR_SAMPLES, len(posterior_flat))),
        "top_percent_histogram": float(TOP_PERCENT),
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
    plot_predictive_fit(posterior_flat, logps)
    save_summary(posterior_flat, logps)
    print("[INFO] Post-processing complete.")

if __name__ == "__main__":
    main()
