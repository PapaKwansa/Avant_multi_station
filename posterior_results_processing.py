"""
Post-processing for the corrected PyDREAM inversion.

Inversion parameterization:
    b = inferred scale parameter
    a = b * (a_fixed / b_fixed)
    c = b * (c_fixed / b_fixed)

Posterior parameters saved by the inversion:
    [b, E, theta]

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
    "font.size": 15,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.8,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# ============================================================
# User settings
# ============================================================

POSTERIOR_FILE = "multi_station_sampled_params.npy"
LOGP_FILE = "multi_station_logps.npy"

BASE_DIR = os.path.dirname(__file__)
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")
VOLUME_FILE = os.path.join(BASE_DIR, "avant_cleaned_volume.csv")

HIST_OUTFILE = "posterior_histograms_b_inversion.png"
SUMMARY_JSON = "posterior_processing_summary.json"
PLOT_PREFIX = "predictive_fit_station"

TOP_PERCENT = 1.0
N_POSTERIOR_SAMPLES = 1500

LOW = 5
HIGH = 95

SHOW_TRUE = False  # set True only for synthetic tests

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

# Geometry ratios implied by the inversion
a_over_b = a_fixed / b_fixed
c_over_b = c_fixed / b_fixed

def reconstruct_geometry_from_b(b_val):
    b_val = float(b_val)
    a_val = b_val * a_over_b
    c_val = b_val * c_over_b
    return a_val, b_val, c_val

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
    "b": (100.0, 600.0),
    "E": (1.0e9, 3.0e10),
    "theta": (30.0, 100.0),
}

LEGACY_COMPONENT_MAP = {
    "eXX": "Epsilon_XX_nanostrain",
    "eYY": "Epsilon_YY_nanostrain",
    "eXY": "Epsilon_XY_nanostrain",
    "eZZ": "Epsilon_ZZ_nanostrain",
}

def expected_component_cols(station_names_local):
    return [f"{comp}_{sname}" for comp in COMPONENTS for sname in station_names_local]

def align_time_dataframe(df, target_time):
    if "time_s" not in df.columns:
        return df

    df = df.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    target_time = np.asarray(target_time, dtype=float)

    obs_time = df["time_s"].to_numpy(dtype=float)

    if len(obs_time) != len(target_time):
        raise RuntimeError(
            f"time_s length mismatch: observed {len(obs_time)}, expected {len(target_time)}"
        )

    if np.allclose(obs_time, target_time, rtol=1e-10, atol=1e-10):
        return df

    idx = pd.Index(obs_time).get_indexer(target_time)
    if np.any(idx < 0):
        raise RuntimeError(
            "Observed time_s does not match the inversion time array, and exact reindexing failed."
        )

    return df.iloc[idx].reset_index(drop=True)

def standardize_observed_strain(df, target_time, station_names_local):
    """
    Return a dataframe whose strain columns are in the exact order used by
    the forward model / likelihood.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = align_time_dataframe(df, target_time)

    exp_cols = expected_component_cols(station_names_local)

    # Case 1: already exact
    if all(c in df.columns for c in exp_cols):
        return df, exp_cols

    # Case 2: legacy named columns
    rename_map = {}
    for comp, legacy_base in LEGACY_COMPONENT_MAP.items():
        for sname in station_names_local:
            legacy = f"{legacy_base}_{sname}"
            new = f"{comp}_{sname}"
            if legacy in df.columns:
                rename_map[legacy] = new

    if rename_map:
        df = df.rename(columns=rename_map)
        if all(c in df.columns for c in exp_cols):
            return df, exp_cols

    # Case 3: generic strain_1..strain_16
    generic_cols = [c for c in df.columns if c not in ["time_s", "Volume"]]
    if len(generic_cols) == len(exp_cols):
        rename_map = {old: new for old, new in zip(generic_cols, exp_cols)}
        df = df.rename(columns=rename_map)
        return df, exp_cols

    raise RuntimeError(
        "Observed strain columns do not match the expected forward-model layout.\n"
        f"Expected {len(exp_cols)} strain columns.\n"
        f"Available columns: {list(df.columns)}"
    )

def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot

def load_posterior():
    posterior_samples = np.load(POSTERIOR_FILE)
    logps = np.load(LOGP_FILE).reshape(-1)

    if posterior_samples.ndim != 3:
        raise RuntimeError(f"Expected posterior samples to be 3D, got shape {posterior_samples.shape}")

    nchains, niter, nparams = posterior_samples.shape
    if nparams != 3:
        raise RuntimeError(f"Expected 3 parameters [b, E, theta], found {nparams}")

    posterior_flat = posterior_samples.reshape(-1, nparams)

    n = min(len(posterior_flat), len(logps))
    posterior_flat = posterior_flat[:n]
    logps = logps[:n]

    return posterior_samples, posterior_flat, logps

def flatten_columns(df, cols):
    return np.concatenate([df[c].values.astype(float) for c in cols])

# ============================================================
# Load observed strain / volume
# ============================================================

obs_df_raw = pd.read_csv(OBSERVED_FILE)
obs_df_raw.columns = [str(c).strip() for c in obs_df_raw.columns]

if "time_s" not in obs_df_raw.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

obs_df, EXPECTED_COLS = standardize_observed_strain(obs_df_raw, time, station_names)
observed_matrix = obs_df[EXPECTED_COLS].values.astype(float)
observed_vector = bi._flatten_df_to_vector(obs_df, EXPECTED_COLS)

# Use a fixed, data-scaled noise estimate for the aleatoric bands
sigma_noise = 0.1 * np.std(observed_vector)
print(f"[INFO] Observed vector shape: {observed_vector.shape}")
print(f"[INFO] Estimated noise scale: {sigma_noise:.3f}")

volume_obs = None
if "Volume" in obs_df_raw.columns:
    volume_obs = obs_df_raw["Volume"].values.astype(float)
elif os.path.exists(VOLUME_FILE):
    vol_df = pd.read_csv(VOLUME_FILE)
    vol_df.columns = [str(c).strip() for c in vol_df.columns]
    if "Volume" in vol_df.columns:
        volume_obs = vol_df["Volume"].values.astype(float)
        if "time_s" in vol_df.columns:
            vol_time = vol_df["time_s"].values.astype(float)
            if len(vol_time) != len(time) or not np.allclose(vol_time, time):
                raise RuntimeError("Volume time array does not match the inversion time array.")

# ============================================================
# Posterior histograms
# ============================================================

def plot_posterior_histograms(posterior_flat, logps):
    map_idx = int(np.argmax(logps))
    map_params = posterior_flat[map_idx]

    sorted_idx = np.argsort(logps)[::-1]
    n_keep = max(1, int(len(sorted_idx) * (TOP_PERCENT / 100.0)))
    posterior_top = posterior_flat[sorted_idx[:n_keep]]

    print(f"[INFO] Using top {TOP_PERCENT}% posterior samples ({n_keep} draws)")

    param_names = ["b", "E", "theta"]
    display_names = [
        "Inferred length scale (b)",
        "Young's Modulus (E)",
        "Orientation (Theta)",
    ]

    true_vals = {
        "b": float(params["b_fixed"]),
        "E": float(params["E"]),
        "theta": float(params["theta_deg"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)

    for i, ax in enumerate(axes):
        name = param_names[i]
        title = display_names[i]
        data = posterior_top[:, i]

        mean_val = float(np.mean(data))
        median_val = float(np.median(data))

        ax.hist(
            data,
            bins=45,
            density=True,
            color="steelblue",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
            label=f"Top {TOP_PERCENT}% posterior",
        )

        lo, hi = PRIOR_RANGES[name]
        ax.axvspan(lo, hi, color="gold", alpha=0.18, label="Prior range")

        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2.8, label="Mean")
        ax.axvline(median_val, color="green", linestyle="-.", linewidth=2.8, label="Median")
        ax.axvline(map_params[i], color="purple", linestyle=":", linewidth=3.2, label="MAP")

        if SHOW_TRUE:
            ax.axvline(true_vals[name], color="black", linewidth=3.0, label="True")

        ax.set_title(title)
        ax.grid(True, alpha=0.28)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )

    plt.savefig(HIST_OUTFILE, dpi=300)
    plt.close()

    print(f"[INFO] Posterior histograms saved to {HIST_OUTFILE}")

    return map_params, posterior_top

# ============================================================
# Predictive fit plots
# ============================================================

def plot_predictive_fit(posterior_flat, logps):
    rng = np.random.default_rng(42)

    map_idx = int(np.argmax(logps))
    b_map_raw, E_map, theta_map = posterior_flat[map_idx]
    a_map, b_map, c_map = reconstruct_geometry_from_b(b_map_raw)

    if SHOW_TRUE:
        b_true = float(params["b_fixed"])
        a_true, b_true, c_true = reconstruct_geometry_from_b(b_true)
        E_true = float(params["E"])
        theta_true = float(params["theta_deg"])
        df_true = forward_model_multi_station(
            pmax=pmax, tpeak=tpeak, d=d, time=time,
            x_prime=x_prime, y_prime=y_prime,
            x0_prime=x0_prime, y0_prime=y0_prime,
            z=z, a=a_true, b=b_true, c=c_true,
            nu=nu, h=h, E=E_true, theta_deg=theta_true,
            alpha=alpha, station_names=station_names,
        )
    else:
        df_true = None

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

    colors = plt.cm.tab10(np.linspace(0, 1, len(COMPONENTS)))

    for station_name in station_names:
        fig, ax = plt.subplots(figsize=(17, 10), constrained_layout=True)

        for i, comp in enumerate(COMPONENTS):
            col = f"{comp}_{station_name}"

            param_curves = []
            total_curves = []

            for draw in posterior_subset:
                b_i_raw, E_i, theta_i = draw
                a_i, b_i, c_i = reconstruct_geometry_from_b(b_i_raw)

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
                total_curves.append(curve + rng.normal(0.0, sigma_noise, size=len(curve)))

            param_curves = np.asarray(param_curves)
            total_curves = np.asarray(total_curves)

            lower_param = np.percentile(param_curves, LOW, axis=0)
            upper_param = np.percentile(param_curves, HIGH, axis=0)
            lower_total = np.percentile(total_curves, LOW, axis=0)
            upper_total = np.percentile(total_curves, HIGH, axis=0)

            # Total uncertainty: light band
            ax.fill_between(
                time, lower_total, upper_total,
                color=colors[i], alpha=0.14, zorder=1
            )

            # Epistemic uncertainty: darker band
            ax.fill_between(
                time, lower_param, upper_param,
                color=colors[i], alpha=0.62, zorder=2
            )

            # Observed data
            ax.scatter(
                obs_df["time_s"].values.astype(float),
                obs_df[col].values.astype(float),
                facecolors="white",
                edgecolors="black",
                s=34,
                linewidths=0.8,
                zorder=5,
            )

            # MAP prediction
            ax.plot(
                time,
                df_map[col].values.astype(float),
                linestyle="--",
                color=colors[i],
                linewidth=2.8,
                zorder=4,
                label=f"{COMPONENT_LABELS[i]} (MAP)",
            )

            # True curve if needed
            if SHOW_TRUE and df_true is not None:
                ax.plot(
                    time,
                    df_true[col].values.astype(float),
                    color="black",
                    linewidth=2.2,
                    zorder=3,
                    label="True" if i == 0 else None,
                )

        ax.set_title(f"Station {station_name} — Posterior and total uncertainty")
        ax.set_xlabel("Time")
        ax.set_ylabel("Strain (nε)")
        ax.grid(True, alpha=0.30)

        epistemic_patch = Patch(facecolor="gray", alpha=0.45, label="Epistemic uncertainty")
        aleatoric_patch = Patch(facecolor="gray", alpha=0.10, label="Total uncertainty")

        handles, labels = ax.get_legend_handles_labels()
        handles.extend([epistemic_patch, aleatoric_patch])
        labels.extend(["Epistemic uncertainty", "Total uncertainty"])

        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )

        out_name = f"{PLOT_PREFIX}_{station_name}_uncertainty.png"
        plt.savefig(out_name, dpi=300)
        plt.close()

        print(f"[INFO] Saved {out_name}")

# ============================================================
# Summary JSON
# ============================================================

def save_summary(posterior_flat, logps):
    map_idx = int(np.argmax(logps))
    b_map_raw, E_map, theta_map = posterior_flat[map_idx]
    a_map, b_map, c_map = reconstruct_geometry_from_b(b_map_raw)

    pred_map = forward_model_multi_station(
        pmax=pmax, tpeak=tpeak, d=d, time=time,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0_prime, y0_prime=y0_prime,
        z=z, a=a_map, b=b_map, c=c_map,
        nu=nu, h=h, E=E_map, theta_deg=theta_map,
        alpha=alpha, station_names=station_names,
    )

    pred_vec = bi._flatten_df_to_vector(pred_map, EXPECTED_COLS)
    strain_r2_map = compute_r2(observed_vector, pred_vec)

    volume_r2 = None
    if volume_obs is not None:
        delta_P = pmax * np.sin(np.pi * time / np.max(time))
        volume_pred_map = bi.volume_forward(a_map, b_map, c_map, E_map, nu, delta_P)
        volume_r2 = compute_r2(volume_obs, volume_pred_map)

    summary = {
        "MAP": {
            "b": float(b_map_raw),
            "a": float(a_map),
            "c": float(c_map),
            "E": float(E_map),
            "theta_deg": float(theta_map),
        },
        "priors": {
            "b": [100.0, 600.0],
            "E": [0.5e9, 3.0e10],
            "theta_deg": [30.0, 100.0],
        },
        "fixed": {
            "a_fixed": float(a_fixed),
            "b_fixed": float(b_fixed),
            "c_fixed": float(c_fixed),
            "sigma_noise": float(sigma_noise),
        },
        "fit": {
            "strain_R2_MAP": float(strain_r2_map),
            "volume_R2_MAP": None if volume_r2 is None else float(volume_r2),
        },
        "n_posterior_draws_for_fit": int(min(N_POSTERIOR_SAMPLES, len(posterior_flat))),
        "top_percent_histogram": float(TOP_PERCENT),
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Saved summary JSON to {SUMMARY_JSON}")
    print(f"[INFO] MAP strain R^2: {strain_r2_map:.4f}")
    if volume_r2 is not None:
        print(f"[INFO] MAP volume R^2: {volume_r2:.4f}")

# ============================================================
# Main
# ============================================================

def main():
    _, posterior_flat, logps = load_posterior()
    plot_posterior_histograms(posterior_flat, logps)
    plot_predictive_fit(posterior_flat, logps)
    save_summary(posterior_flat, logps)
    print("[INFO] Post-processing complete.")

if __name__ == "__main__":
    main()