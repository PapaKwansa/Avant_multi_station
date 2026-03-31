#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full sensitivity + global search + Sobol analysis for the multi-station forward model.

Requirements:
- forward_model_multi_station.py
- multi_stations_input.py (with read_input())
- AVANT_stations.csv
- avant_cleaned_strain.csv
- (Optional) SALib for Sobol indices: pip install SALib
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_model_multi_station import forward_model_multi_station
from multi_stations_input import read_input

# Optional: Sobol analysis
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False
    print("[WARN] SALib not installed; Sobol analysis will be skipped.")

# ============================================================
# Presentation style
# ============================================================

plt.rcParams.update({
    "figure.figsize": (11, 7),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "font.size": 13,
    "font.weight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.4,
    "lines.markersize": 5,
})

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(__file__)
STATION_FILE = os.path.join(BASE_DIR, "AVANT_stations.csv")
OBSERVED_FILE = os.path.join(BASE_DIR, "avant_cleaned_strain.csv")
VOLUME_FILE = os.path.join(BASE_DIR, "observed_volume.csv")  # optional

OUT_DIR = os.path.join(BASE_DIR, "sensitivity_plots")
os.makedirs(OUT_DIR, exist_ok=True)

N_1D = 17
N_2D = 13
RUN_HEATMAPS = True
RUN_VOLUME_SENSITIVITY = True

# Global search
RUN_GLOBAL_SEARCH = True
N_GLOBAL_SAMPLES = 500  # adjust as needed

# Sobol
RUN_SOBOL = True and HAS_SALIB
N_SOBOL_SAMPLES = 512  # base sample size for Saltelli

# ============================================================
# Helpers
# ============================================================

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def safe_float(x, fallback):
    try:
        return float(x)
    except Exception:
        return float(fallback)

def savefig_with_legend_below(fig, ax, filename, ncol=3):
    """
    Place legend below the axes, then save.
    """
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.subplots_adjust(bottom=0.18)
        fig.legend(handles, labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, 0.02),
                   ncol=ncol,
                   frameon=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[INFO] Saved {path}")

def savefig_grid_with_legend_below(fig, axes, filename, ncol=3):
    """
    For multi-axes figures: put a single legend below using first axes that has handles.
    """
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if h:
            handles.extend(h)
            labels.extend(l)
    if handles:
        fig.subplots_adjust(bottom=0.12)
        fig.legend(handles, labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, 0.02),
                   ncol=ncol,
                   frameon=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[INFO] Saved {path}")

def pressure_time_series(pmax, tpeak, d, time):
    t = np.asarray(time, dtype=float)
    p = np.zeros_like(t, dtype=float)
    rising = t <= tpeak
    falling = t > tpeak
    p[rising] = (pmax / tpeak) * t[rising]
    p[falling] = pmax * np.exp(-d * (t[falling] / tpeak - 1.0))
    return p

def bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))

def volume_forward(a, b, c, E, nu, delta_P):
    v0 = a * b * c
    k = bulk_modulus(E, nu)
    return v0 * (1.0 + delta_P / k)

def get_strain_columns_in_order(obs_df):
    cols = [c for c in obs_df.columns if c not in ("time_s", "Volume")]
    if len(cols) == 0:
        raise RuntimeError("No strain columns found in observed file.")
    return cols

def model_block_columns(station_names):
    return [
        f"{comp}_{sname}"
        for comp in ["eXX", "eYY", "eXY", "eZZ"]
        for sname in station_names
    ]

def compute_component_rmse(pred_matrix, obs_matrix, station_names):
    out = {}
    n_stat = len(station_names)
    comps = ["eXX", "eYY", "eXY", "eZZ"]
    for ic, comp in enumerate(comps):
        out[comp] = {}
        for js, sname in enumerate(station_names):
            idx = ic * n_stat + js
            out[comp][sname] = rmse(pred_matrix[:, idx], obs_matrix[:, idx])
    return out

def reorder_stations_from_metadata(stations_df):
    stations_df = stations_df.copy()
    stations_df["station"] = stations_df["station"].astype(str).str.strip()
    required_cols = ["station", "x_prime", "y_prime", "depth"]
    missing = set(required_cols) - set(stations_df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in AVANT_stations.csv: {missing}")
    station_names = stations_df["station"].values
    x_prime = stations_df["x_prime"].values.astype(float)
    y_prime = stations_df["y_prime"].values.astype(float)
    z = stations_df["depth"].values.astype(float)
    return station_names, x_prime, y_prime, z

# ============================================================
# Load inputs
# ============================================================

params = read_input()

stations_df = pd.read_csv(STATION_FILE)
station_names, x_prime, y_prime, z = reorder_stations_from_metadata(stations_df)

obs_df = pd.read_csv(OBSERVED_FILE).drop_duplicates().reset_index(drop=True)
if "time_s" not in obs_df.columns:
    raise RuntimeError("Observed file must contain a 'time_s' column")

time = obs_df["time_s"].values.astype(float)

obs_cols = get_strain_columns_in_order(obs_df)
expected_cols = model_block_columns(station_names)

if len(obs_cols) != len(expected_cols):
    raise RuntimeError(
        f"Observed strain column count mismatch. "
        f"Found {len(obs_cols)}, expected {len(expected_cols)} "
        f"({4 * len(station_names)})."
    )

obs_matrix = obs_df[obs_cols].values.astype(float)

print(f"[INFO] Stations: {station_names}")
print(f"[INFO] Observed matrix shape: {obs_matrix.shape}")
print(f"[INFO] Observed min/max: {np.min(obs_matrix):.6g} / {np.max(obs_matrix):.6g}")
print(f"[INFO] Observed mean abs: {np.mean(np.abs(obs_matrix)):.6g}")
print(f"[INFO] Observed std: {np.std(obs_matrix):.6g}")

# ============================================================
# Baseline values from input
# ============================================================

nu0 = safe_float(params.get("nu", 0.25), 0.25)
alpha0 = safe_float(params.get("alpha", 0.8), 0.8)

a0 = safe_float(params.get("a_fixed", 150.0), 150.0)
b0 = safe_float(params.get("b_fixed", 290.0), 290.0)
c0 = safe_float(params.get("c_fixed", 2.5), 2.5)

x00 = safe_float(params.get("x0_prime", 0.0), 0.0)
y00 = safe_float(params.get("y0_prime", 0.0), 0.0)
h0 = safe_float(params.get("h", 518.29), 518.29)

pmax0 = safe_float(params.get("pmax", 2.0e6), 2.0e6)
tpeak0 = safe_float(params.get("tpeak", 393333.0), 393333.0)
d0 = safe_float(params.get("d", 0.4), 0.4)

E0 = safe_float(params.get("E", 2.0e9), 2.0e9)
theta0 = safe_float(params.get("theta_deg", 75.0), 75.0)

print("[INFO] Baseline values from input:")
print(f"       nu = {nu0:.6g}")
print(f"       alpha = {alpha0:.6g}")
print(f"       a = {a0:.6g}")
print(f"       b = {b0:.6g}")
print(f"       c = {c0:.6g}")
print(f"       x0_prime = {x00:.6g}")
print(f"       y0_prime = {y00:.6g}")
print(f"       h = {h0:.6g}")
print(f"       pmax = {pmax0:.6g}")
print(f"       tpeak = {tpeak0:.6g}")
print(f"       d = {d0:.6g}")
print(f"       E = {E0:.6g}")
print(f"       theta_deg = {theta0:.6g}")

# ============================================================
# Optional volume observations
# ============================================================

use_volume = False
volume_obs = None

if "Volume" in obs_df.columns:
    volume_obs = obs_df["Volume"].values.astype(float)
    if len(volume_obs) != len(time):
        raise RuntimeError("Volume column must have same length as time_s.")
    use_volume = True
    print("[INFO] Using Volume column from observed strain file.")
elif RUN_VOLUME_SENSITIVITY and os.path.exists(VOLUME_FILE):
    vol_df = pd.read_csv(VOLUME_FILE).drop_duplicates().reset_index(drop=True)
    if not {"time_s", "Volume"}.issubset(vol_df.columns):
        raise RuntimeError("observed_volume.csv must contain columns time_s and Volume.")
    volume_obs = np.interp(
        time,
        vol_df["time_s"].values.astype(float),
        vol_df["Volume"].values.astype(float),
    )
    use_volume = True
    print(f"[INFO] Loaded and interpolated volume observations from {VOLUME_FILE}.")
else:
    print("[WARN] No volume observations found; volume sensitivity will be skipped.")

# ============================================================
# Forward evaluation wrapper
# ============================================================

def eval_model(overrides=None):
    """
    Evaluate the forward model with optional parameter overrides.
    Returns a dict with predicted strain matrix, RMSE, and optional volume metrics.
    """
    p = {
        "nu": nu0,
        "alpha": alpha0,
        "a_fixed": a0,
        "b_fixed": b0,
        "c_fixed": c0,
        "x0_prime": x00,
        "y0_prime": y00,
        "h": h0,
        "pmax": pmax0,
        "tpeak": tpeak0,
        "d": d0,
        "E": E0,
        "theta_deg": theta0,
    }
    if overrides:
        p.update(overrides)

    df_pred = forward_model_multi_station(
        pmax=p["pmax"],
        tpeak=p["tpeak"],
        d=p["d"],
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=p["x0_prime"],
        y0_prime=p["y0_prime"],
        z=z,
        a=p["a_fixed"],
        b=p["b_fixed"],
        c=p["c_fixed"],
        nu=p["nu"],
        h=p["h"],
        E=p["E"],
        theta_deg=p["theta_deg"],
        alpha=p["alpha"],
        station_names=station_names,
        debug=False,
    )

    pred_matrix = df_pred[expected_cols].values.astype(float)
    strain_rmse = rmse(pred_matrix, obs_matrix)
    comp_rmse = compute_component_rmse(pred_matrix, obs_matrix, station_names)

    volume_pred = None
    volume_rmse = None
    if use_volume and volume_obs is not None:
        pressure = pressure_time_series(p["pmax"], p["tpeak"], p["d"], time)
        volume_pred = volume_forward(
            p["a_fixed"], p["b_fixed"], p["c_fixed"], p["E"], p["nu"], pressure
        )
        volume_rmse = rmse(volume_pred, volume_obs)

    return {
        "params": p,
        "df_pred": df_pred,
        "pred_matrix": pred_matrix,
        "strain_rmse": strain_rmse,
        "component_rmse": comp_rmse,
        "volume_pred": volume_pred,
        "volume_rmse": volume_rmse,
    }

# ============================================================
# Baseline diagnostics
# ============================================================

baseline = eval_model()

print("\n[INFO] Baseline diagnostics")
print(f"[INFO] baseline strain RMSE = {baseline['strain_rmse']:.6g}")
if baseline["volume_rmse"] is not None:
    print(f"[INFO] baseline volume RMSE = {baseline['volume_rmse']:.6g}")

# ============================================================
# 1D sweep definitions
# ============================================================

station_span = max(np.ptp(x_prime), np.ptp(y_prime))
center_window = max(1000.0, 2.0 * station_span)

sweep_defs = [
    dict(key="nu", label="Poisson ratio ν", kind="linear",
         low=0.15, high=0.35, n=N_1D, plot_scale=1.0),

    dict(key="alpha", label="Biot coefficient α", kind="linear",
         low=max(0.1, alpha0 * 0.4), high=min(1.5, alpha0 * 1.6), n=N_1D, plot_scale=1.0),

    dict(key="a_fixed", label="Semi-axis a (m)", kind="linear",
         low=50, high=300, n=N_1D, plot_scale=1.0),

    dict(key="b_fixed", label="Semi-axis b (m)", kind="linear",
         low=50, high=500, n=N_1D, plot_scale=1.0),

    dict(key="c_fixed", label="Semi-axis c (m)", kind="linear",
         low=1, high=20, n=N_1D, plot_scale=1.0),

    dict(key="x0_prime", label="Center x₀′ (m)", kind="linear",
         low=x00 - center_window, high=x00 + center_window, n=N_1D, plot_scale=1.0),

    dict(key="y0_prime", label="Center y₀′ (m)", kind="linear",
         low=y00 - center_window, high=y00 + center_window, n=N_1D, plot_scale=1.0),

    dict(key="h", label="Depth h (m)", kind="linear",
         low=max(1.0, h0 * 0.5), high=max(1.0, h0 * 1.5), n=N_1D, plot_scale=1.0),

    dict(key="pmax", label="Peak pressure pmax (MPa)", kind="linear",
         low=max(1e3, pmax0 * 0.25), high=max(1e3 * 1.001, pmax0 * 4.0), n=N_1D, plot_scale=1e6),

    dict(key="tpeak", label="Time to peak tpeak (hours)", kind="linear",
         low=max(1.0, tpeak0 * 0.5), high=max(1.0, tpeak0 * 1.5), n=N_1D, plot_scale=3600.0),

    dict(key="d", label="Decay rate d", kind="linear",
         low=max(0.01, d0 * 0.25), high=max(0.05, d0 * 4.0), n=N_1D, plot_scale=1.0),

    dict(key="E", label="Young’s modulus E (GPa)", kind="log",
         low=max(1e8, E0 * 0.25), high=max(1e8 * 1.001, E0 * 4.0), n=N_1D, plot_scale=1e9),

    dict(key="theta_deg", label="Rotation angle θ (deg)", kind="linear",
         low=-90.0, high=90.0, n=N_1D, plot_scale=1.0),
]

# ============================================================
# One-at-a-time sweeps
# ============================================================

sweep_records = []
sweep_results = {}

baseline_values = {
    "nu": nu0,
    "alpha": alpha0,
    "a_fixed": a0,
    "b_fixed": b0,
    "c_fixed": c0,
    "x0_prime": x00,
    "y0_prime": y00,
    "h": h0,
    "pmax": pmax0,
    "tpeak": tpeak0,
    "d": d0,
    "E": E0,
    "theta_deg": theta0,
}

for sd in sweep_defs:
    key = sd["key"]
    label = sd["label"]
    n = sd["n"]
    scale = sd["plot_scale"]

    base_val = baseline_values[key]

    if sd["kind"] == "log":
        values = np.logspace(np.log10(sd["low"]), np.log10(sd["high"]), n)
    else:
        values = np.linspace(sd["low"], sd["high"], n)

    strain_vals = []
    volume_vals = []

    print(f"\n[INFO] Sweeping {key} ({label})")

    for v in values:
        out = eval_model(overrides={key: float(v)})
        strain_vals.append(out["strain_rmse"])
        if use_volume and out["volume_rmse"] is not None:
            volume_vals.append(out["volume_rmse"])

        sweep_records.append({
            "parameter": key,
            "label": label,
            "value": float(v),
            "value_plot": float(v / scale),
            "strain_rmse": float(out["strain_rmse"]),
            "volume_rmse": None if out["volume_rmse"] is None else float(out["volume_rmse"]),
            "baseline_value": float(base_val),
            "baseline_value_plot": float(base_val / scale),
        })

    strain_vals = np.asarray(strain_vals, dtype=float)
    best_idx = int(np.argmin(strain_vals))
    best_val = float(values[best_idx])
    best_rmse = float(strain_vals[best_idx])

    sweep_results[key] = {
        "label": label,
        "values": values,
        "values_plot": values / scale,
        "strain_rmse": strain_vals,
        "baseline_value": base_val,
        "baseline_value_plot": base_val / scale,
        "best_value": best_val,
        "best_rmse": best_rmse,
    }

    if use_volume and len(volume_vals) == len(values):
        sweep_results[key]["volume_rmse"] = np.asarray(volume_vals, dtype=float)

    print(f"[RESULT] {key}: best={best_val:.6g}, RMSE={best_rmse:.6g}")

# Save sweep table
sweep_df = pd.DataFrame(sweep_records)
sweep_csv = os.path.join(OUT_DIR, "sensitivity_1d_sweep_results.csv")
sweep_df.to_csv(sweep_csv, index=False)
print(f"[INFO] Saved {sweep_csv}")

summary_rows = []
for key, res in sweep_results.items():
    summary_rows.append({
        "parameter": key,
        "label": res["label"],
        "baseline_value": float(res["baseline_value"]),
        "best_value": float(res["best_value"]),
        "baseline_rmse": float(baseline["strain_rmse"]),
        "best_rmse": float(res["best_rmse"]),
        "improvement_pct": 100.0 * (baseline["strain_rmse"] - res["best_rmse"]) /
                           max(baseline["strain_rmse"], 1e-12),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "sensitivity_1d_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"[INFO] Saved {summary_csv}")

# ============================================================
# 1D sensitivity plots (high quality, legend below)
# ============================================================

n_params = len(sweep_defs)
n_cols = 3
n_rows = int(math.ceil(n_params / n_cols))

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(18, 5.2 * n_rows),
                         squeeze=False)
axes_flat = axes.ravel()

for i, sd in enumerate(sweep_defs):
    key = sd["key"]
    res = sweep_results[key]
    ax = axes_flat[i]

    ax.plot(res["values_plot"], res["strain_rmse"],
            color="navy", marker="o", markersize=5,
            label="Strain RMSE")
    ax.axvline(res["baseline_value_plot"],
               color="crimson", linestyle="--", linewidth=2,
               label="Baseline")
    ax.set_title(res["label"])
    ax.set_xlabel(res["label"])
    ax.set_ylabel("Strain RMSE")
    ax.tick_params(width=1.5)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis("off")

fig.suptitle("One-at-a-time sensitivity of model inputs", y=0.995,
             fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0.08, 1, 0.96])
savefig_grid_with_legend_below(fig, axes, "sensitivity_1d_strain_rmse.png", ncol=3)

if use_volume and any("volume_rmse" in r for r in sweep_results.values()):
    figv, axesv = plt.subplots(n_rows, n_cols,
                               figsize=(18, 5.2 * n_rows),
                               squeeze=False)
    axesv_flat = axesv.ravel()

    for i, sd in enumerate(sweep_defs):
        key = sd["key"]
        res = sweep_results[key]
        ax = axesv_flat[i]

        if "volume_rmse" in res:
            ax.plot(res["values_plot"], res["volume_rmse"],
                    color="darkgreen", marker="o", markersize=5,
                    label="Volume RMSE")
            ax.axvline(res["baseline_value_plot"],
                       color="crimson", linestyle="--", linewidth=2,
                       label="Baseline")
            ax.set_title(res["label"] + " — volume")
            ax.set_xlabel(res["label"])
            ax.set_ylabel("Volume RMSE")
        else:
            ax.text(0.5, 0.5, "No volume metric",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    for j in range(i + 1, len(axesv_flat)):
        axesv_flat[j].axis("off")

    figv.suptitle("One-at-a-time sensitivity of volume fit", y=0.995,
                  fontsize=20, fontweight="bold")
    figv.tight_layout(rect=[0, 0.08, 1, 0.96])
    savefig_grid_with_legend_below(figv, axesv, "sensitivity_1d_volume_rmse.png", ncol=3)

# ============================================================
# 2D heatmaps (E vs theta, x0 vs y0)
# ============================================================

if RUN_HEATMAPS:
    print("\n[INFO] Running 2D heatmap for E vs theta_deg")
    E_grid = np.logspace(np.log10(max(1e8, E0 * 0.25)),
                         np.log10(max(1e8 * 1.001, E0 * 4.0)), N_2D)
    theta_grid = np.linspace(-90.0, 90.0, N_2D)
    rmse_grid = np.zeros((len(theta_grid), len(E_grid)))

    for i_th, th in enumerate(theta_grid):
        for j_E, E_val in enumerate(E_grid):
            out = eval_model(overrides={"E": float(E_val), "theta_deg": float(th)})
            rmse_grid[i_th, j_E] = out["strain_rmse"]

    fig2, ax2 = plt.subplots(figsize=(10.5, 7.5))
    E_mesh, th_mesh = np.meshgrid(E_grid / 1e9, theta_grid)
    hm = ax2.pcolormesh(E_mesh, th_mesh, rmse_grid,
                        shading="auto", cmap="viridis")
    cbar = fig2.colorbar(hm, ax=ax2)
    cbar.set_label("Strain RMSE", fontweight="bold")

    ax2.scatter([E0 / 1e9], [theta0],
                marker="x", s=120, color="red", linewidths=3,
                label="Baseline")
    ax2.set_xlabel("Young’s modulus E (GPa)")
    ax2.set_ylabel("Rotation angle θ (deg)")
    ax2.set_title("2D sensitivity: strain RMSE vs E and θ")
    fig2.tight_layout(rect=[0, 0.12, 1, 0.96])
    savefig_with_legend_below(fig2, ax2, "sensitivity_2d_E_theta.png", ncol=1)

    if use_volume:
        print("[INFO] Running 2D heatmap for E vs theta_deg (volume)")
        rmse_grid_v = np.zeros((len(theta_grid), len(E_grid)))
        for i_th, th in enumerate(theta_grid):
            for j_E, E_val in enumerate(E_grid):
                out = eval_model(overrides={"E": float(E_val), "theta_deg": float(th)})
                rmse_grid_v[i_th, j_E] = np.nan if out["volume_rmse"] is None else out["volume_rmse"]

        fig2v, ax2v = plt.subplots(figsize=(10.5, 7.5))
        hm2 = ax2v.pcolormesh(E_mesh, th_mesh, rmse_grid_v,
                              shading="auto", cmap="magma")
        cbar2 = fig2v.colorbar(hm2, ax=ax2v)
        cbar2.set_label("Volume RMSE", fontweight="bold")

        ax2v.scatter([E0 / 1e9], [theta0],
                     marker="x", s=120, color="cyan", linewidths=3,
                     label="Baseline")
        ax2v.set_xlabel("Young’s modulus E (GPa)")
        ax2v.set_ylabel("Rotation angle θ (deg)")
        ax2v.set_title("2D sensitivity: volume RMSE vs E and θ")
        fig2v.tight_layout(rect=[0, 0.12, 1, 0.96])
        savefig_with_legend_below(fig2v, ax2v, "sensitivity_2d_E_theta_volume.png", ncol=1)

    print("[INFO] Running 2D heatmap for x0_prime vs y0_prime")
    x_grid = np.linspace(x00 - center_window, x00 + center_window, N_2D)
    y_grid = np.linspace(y00 - center_window, y00 + center_window, N_2D)
    rmse_xy = np.zeros((len(y_grid), len(x_grid)))

    for i_y, yv in enumerate(y_grid):
        for j_x, xv in enumerate(x_grid):
            out = eval_model(overrides={"x0_prime": float(xv), "y0_prime": float(yv)})
            rmse_xy[i_y, j_x] = out["strain_rmse"]

    fig3, ax3 = plt.subplots(figsize=(10.5, 7.5))
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    hm3 = ax3.pcolormesh(X_mesh, Y_mesh, rmse_xy,
                         shading="auto", cmap="viridis")
    cbar3 = fig3.colorbar(hm3, ax=ax3)
    cbar3.set_label("Strain RMSE", fontweight="bold")

    ax3.scatter([x00], [y00],
                marker="x", s=120, color="red", linewidths=3,
                label="Baseline")
    ax3.set_xlabel("x0′ (m)")
    ax3.set_ylabel("y0′ (m)")
    ax3.set_title("2D sensitivity: strain RMSE vs inclusion center")
    fig3.tight_layout(rect=[0, 0.12, 1, 0.96])
    savefig_with_legend_below(fig3, ax3, "sensitivity_2d_x0_y0.png", ncol=1)

# ============================================================
# Representative time-series comparison
# ============================================================

rep_station = station_names[-1]
rep_component = "eZZ"
rep_col = f"{rep_component}_{rep_station}"
rep_idx = expected_cols.index(rep_col)

best_E = sweep_results["E"]["best_value"]
best_theta = sweep_results["theta_deg"]["best_value"]

base_df = baseline["df_pred"]
bestE_df = eval_model(overrides={"E": best_E})["df_pred"]
bestT_df = eval_model(overrides={"theta_deg": best_theta})["df_pred"]

fig4, ax4 = plt.subplots(figsize=(11.5, 6.5))
ax4.plot(time / 3600.0, obs_matrix[:, rep_idx],
         color="black", label=f"Observed {rep_col}")
ax4.plot(time / 3600.0, base_df[rep_col].values,
         color="royalblue", linestyle="--", label="Baseline")
ax4.plot(time / 3600.0, bestE_df[rep_col].values,
         color="darkgreen", linestyle="-.", label=f"Best E = {best_E/1e9:.2f} GPa")
ax4.plot(time / 3600.0, bestT_df[rep_col].values,
         color="firebrick", linestyle=":", label=f"Best θ = {best_theta:.1f}°")

ax4.set_xlabel("Time (hours)")
ax4.set_ylabel("Strain (nε)")
ax4.set_title(f"Representative channel sensitivity: {rep_col}")
fig4.tight_layout(rect=[0, 0.18, 1, 0.96])
savefig_with_legend_below(fig4, ax4, "sensitivity_time_series_example.png", ncol=2)

# ============================================================
# Global multi-parameter search (random / LHS style)
# ============================================================

global_best = {
    "rmse": baseline["strain_rmse"],
    "params": baseline["params"],
}

global_records = []

if RUN_GLOBAL_SEARCH:
    print("\n[INFO] Running global multi-parameter search to reduce RMSE")

    # Define reasonable ranges (edit as needed)
    search_ranges = {
        "E": (max(1e8, E0 * 0.25), max(1e8 * 1.001, E0 * 4.0)),
        "theta_deg": (-90.0, 90.0),
        "a_fixed": (50.0, 300.0),
        "b_fixed": (50.0, 500.0),
        "c_fixed": (1.0, 20.0),
        "x0_prime": (x00 - center_window, x00 + center_window),
        "y0_prime": (y00 - center_window, y00 + center_window),
        "h": (max(1.0, h0 * 0.5), max(1.0, h0 * 1.5)),
        "pmax": (max(1e3, pmax0 * 0.25), max(1e3 * 1.001, pmax0 * 4.0)),
        "tpeak": (max(1.0, tpeak0 * 0.5), max(1.0, tpeak0 * 1.5)),
        "d": (max(0.01, d0 * 0.25), max(0.05, d0 * 4.0)),
        "nu": (0.15, 0.35),
        "alpha": (max(0.1, alpha0 * 0.4), min(1.5, alpha0 * 1.6)),
    }

    keys = list(search_ranges.keys())
    lows = np.array([search_ranges[k][0] for k in keys], dtype=float)
    highs = np.array([search_ranges[k][1] for k in keys], dtype=float)

    # Simple random sampling (could be replaced with LHS)
    rng = np.random.default_rng(123)
    samples = rng.random((N_GLOBAL_SAMPLES, len(keys)))
    samples = lows + samples * (highs - lows)

    for i in range(N_GLOBAL_SAMPLES):
        overrides = {k: float(samples[i, j]) for j, k in enumerate(keys)}
        out = eval_model(overrides=overrides)
        r = out["strain_rmse"]

        rec = {"strain_rmse": r}
        rec.update(overrides)
        global_records.append(rec)

        if r < global_best["rmse"]:
            global_best["rmse"] = r
            global_best["params"] = overrides
            print(f"[GLOBAL] New best RMSE={r:.6g} at sample {i+1}/{N_GLOBAL_SAMPLES}")

    global_df = pd.DataFrame(global_records)
    global_csv = os.path.join(OUT_DIR, "global_search_results.csv")
    global_df.to_csv(global_csv, index=False)
    print(f"[INFO] Saved {global_csv}")
    print(f"[INFO] Global best RMSE = {global_best['rmse']:.6g}")
    print(f"[INFO] Global best params = {global_best['params']}")

    # Simple scatter plots for a few key parameters vs RMSE
    for k in ["E", "theta_deg", "a_fixed", "b_fixed", "c_fixed"]:
        if k not in global_df.columns:
            continue
        fig_g, ax_g = plt.subplots(figsize=(8, 6))
        ax_g.scatter(global_df[k], global_df["strain_rmse"],
                     s=18, alpha=0.6, color="navy", label="Samples")
        ax_g.set_xlabel(k)
        ax_g.set_ylabel("Strain RMSE")
        ax_g.set_title(f"Global search: RMSE vs {k}")
        fig_g.tight_layout(rect=[0, 0.18, 1, 0.96])
        savefig_with_legend_below(fig_g, ax_g,
                                  f"global_search_rmse_vs_{k}.png", ncol=1)

# ============================================================
# Sobol sensitivity analysis (if SALib available)
# ============================================================

sobol_results = None

if RUN_SOBOL:
    print("\n[INFO] Running Sobol sensitivity analysis (SALib)")

    # Choose a subset of parameters for Sobol (to keep dimension manageable)
    sobol_params = [
        "E", "theta_deg", "a_fixed", "b_fixed", "c_fixed",
        "x0_prime", "y0_prime", "h", "pmax", "tpeak", "d", "nu", "alpha"
    ]

    sobol_ranges = {
        "E": (max(1e8, E0 * 0.25), max(1e8 * 1.001, E0 * 4.0)),
        "theta_deg": (-90.0, 90.0),
        "a_fixed": (50.0, 300.0),
        "b_fixed": (50.0, 500.0),
        "c_fixed": (1.0, 20.0),
        "x0_prime": (x00 - center_window, x00 + center_window),
        "y0_prime": (y00 - center_window, y00 + center_window),
        "h": (max(1.0, h0 * 0.5), max(1.0, h0 * 1.5)),
        "pmax": (max(1e3, pmax0 * 0.25), max(1e3 * 1.001, pmax0 * 4.0)),
        "tpeak": (max(1.0, tpeak0 * 0.5), max(1.0, tpeak0 * 1.5)),
        "d": (max(0.01, d0 * 0.25), max(0.05, d0 * 4.0)),
        "nu": (0.15, 0.35),
        "alpha": (max(0.1, alpha0 * 0.4), min(1.5, alpha0 * 1.6)),
    }

    problem = {
        "num_vars": len(sobol_params),
        "names": sobol_params,
        "bounds": [sobol_ranges[p] for p in sobol_params],
    }

    # Generate samples
    X = saltelli.sample(problem, N_SOBOL_SAMPLES, calc_second_order=True)
    Y = np.zeros(X.shape[0], dtype=float)

    print(f"[INFO] Sobol sample size: {X.shape[0]}")

    for i in range(X.shape[0]):
        overrides = {sobol_params[j]: float(X[i, j]) for j in range(len(sobol_params))}
        out = eval_model(overrides=overrides)
        Y[i] = out["strain_rmse"]

    # Analyze
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    sobol_results = Si

    # Save raw Sobol results
    sobol_dict = {
        "S1": Si["S1"].tolist(),
        "S1_conf": Si["S1_conf"].tolist(),
        "ST": Si["ST"].tolist(),
        "ST_conf": Si["ST_conf"].tolist(),
        "S2": Si["S2"].tolist(),
        "S2_conf": Si["S2_conf"].tolist(),
        "names": sobol_params,
    }
    with open(os.path.join(OUT_DIR, "sobol_results.json"), "w") as f:
        json.dump(sobol_dict, f, indent=2)
    print(f"[INFO] Saved Sobol results to sobol_results.json")

    # Bar plot for first-order and total-order indices
    indices = np.arange(len(sobol_params))
    width = 0.35

    fig_s, ax_s = plt.subplots(figsize=(12, 7))
    ax_s.bar(indices - width/2, Si["S1"], width,
             yerr=Si["S1_conf"], color="steelblue", label="First-order S1")
    ax_s.bar(indices + width/2, Si["ST"], width,
             yerr=Si["ST_conf"], color="darkorange", label="Total-order ST")

    ax_s.set_xticks(indices)
    ax_s.set_xticklabels(sobol_params, rotation=45, ha="right")
    ax_s.set_ylabel("Sobol index")
    ax_s.set_title("Sobol sensitivity indices for strain RMSE")
    fig_s.tight_layout(rect=[0, 0.18, 1, 0.96])
    savefig_with_legend_below(fig_s, ax_s, "sobol_indices_strain_rmse.png", ncol=2)

# ============================================================
# Save global summary
# ============================================================

best_rows = []
for k, res in sweep_results.items():
    best_rows.append({
        "parameter": k,
        "label": res["label"],
        "baseline_value": float(res["baseline_value"]),
        "best_value": float(res["best_value"]),
        "baseline_rmse": float(baseline["strain_rmse"]),
        "best_rmse": float(res["best_rmse"]),
        "improvement_pct": 100.0 * (baseline["strain_rmse"] - res["best_rmse"]) /
                           max(baseline["strain_rmse"], 1e-12),
    })

best_df = pd.DataFrame(best_rows)
best_df.to_csv(os.path.join(OUT_DIR, "sensitivity_summary.csv"), index=False)

summary = {
    "baseline_strain_rmse": float(baseline["strain_rmse"]),
    "baseline_volume_rmse": None if baseline["volume_rmse"] is None else float(baseline["volume_rmse"]),
    "station_names": list(station_names),
    "fixed_values": {
        "nu": nu0,
        "alpha": alpha0,
        "a": a0,
        "b": b0,
        "c": c0,
        "x0_prime": x00,
        "y0_prime": y00,
        "h": h0,
        "pmax": pmax0,
        "tpeak": tpeak0,
        "d": d0,
        "E": E0,
        "theta_deg": theta0,
    },
    "heatmaps_enabled": bool(RUN_HEATMAPS),
    "volume_used": bool(use_volume),
    "global_best_rmse": float(global_best["rmse"]) if RUN_GLOBAL_SEARCH else None,
    "global_best_params": global_best["params"] if RUN_GLOBAL_SEARCH else None,
    "sobol_enabled": bool(RUN_SOBOL),
}

with open(os.path.join(OUT_DIR, "sensitivity_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n[INFO] Full sensitivity analysis complete.")
print(f"[INFO] Results saved in: {OUT_DIR}")
