
"""Hydraulic-to-analytical strain detectability benchmark.

Run from the AVANT repository root.

Uses:
    - Saved finite-difference hydraulic results
    - Verified nine-zone analytical pressure kernel

Produces:
    - Improved fine-grid and nine-zone pressure maps
    - Hydraulic pressure histories
    - Analytical strain histories
    - Channel and network signal-to-noise diagnostics
    - Numerical summary and CSV outputs

This is a fixed-geometry synthetic diagnostic.
No field observations or Bayesian pressure inversion are used.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path("results/pressure_superposition")

FD = ROOT / "injection_fd_001"
MECH = ROOT / "nine_uniform_noisy_001"
OUT = FD / "detectability"

OUT.mkdir(parents=True, exist_ok=True)

NOISE_NS = 2.0
SHUTIN_H = 19.0

# Hydraulic local x/y are identified with analytical xi/eta
# for this controlled synthetic benchmark.
# The analytical kernel already handles the 15-degree rotation.
HYDRAULIC_AXES_VERIFIED = True

# Values smaller than this are displayed as effectively zero.
# The underlying hydraulic data are never modified.
DISPLAY_THRESHOLD_KPA = 1e-6

# Selected hydraulic snapshots.
PLOT_TIMES_H = [0, 5, 19, 24, 48, 96]

# Visualization settings.
DPI = 300
CMAP = "viridis"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 13,
    "savefig.dpi": DPI,
})


# ============================================================
# 2. LOAD AND VALIDATE INPUTS
# ============================================================

if not HYDRAULIC_AXES_VERIFIED:
    raise RuntimeError(
        "Hydraulic local x/y must correspond to analytical "
        "xi/eta before calculating strain."
    )

required_files = [
    FD / "pressure_history.csv",
    FD / "hydraulic_fields.npz",
    MECH / "pressure_kernel_and_posterior.npz",
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )

history = pd.read_csv(FD / "pressure_history.csv")

with np.load(FD / "hydraulic_fields.npz") as f:
    field_times = np.asarray(f["times_h"], float)
    field_pa = np.asarray(f["pressure_change_pa"], float)
    x = np.asarray(f["x_m"], float)
    y = np.asarray(f["y_m"], float)

with np.load(
    MECH / "pressure_kernel_and_posterior.npz"
) as f:
    G = np.asarray(f["G_ns_per_pa"], float)
    B = np.asarray(f["basis"], float)
    channels = [str(v) for v in f["channels"]]
    training = np.asarray(
        f["training_channel_mask"], bool
    )

times = history["time_h"].to_numpy(float)

if len(times) < 2 or np.any(np.diff(times) <= 0):
    raise ValueError(
        "Hydraulic times must be strictly increasing."
    )

if len(x) < 2 or len(y) < 2:
    raise ValueError("Invalid hydraulic grid coordinates.")

if field_pa.shape != (
    len(field_times), len(y), len(x)
):
    raise ValueError(
        "Fine-grid pressure array dimensions do not match "
        "the saved coordinates."
    )

if not np.all(np.isfinite(field_pa)):
    raise ValueError("Non-finite hydraulic pressures found.")

if not (
    np.all(np.diff(x) > 0)
    and np.all(np.diff(y) > 0)
):
    raise ValueError(
        "Hydraulic x/y coordinates must increase."
    )

# Hydraulic row-major ordering:
# row0_col0, row0_col1, ..., row2_col2
hydraulic_columns = [
    f"zone_row{iy}_col{ix}_mpa"
    for iy in range(3)
    for ix in range(3)
]

missing = [
    col for col in hydraulic_columns
    if col not in history.columns
]

if missing:
    raise ValueError(
        f"Missing hydraulic pressure columns: {missing}"
    )

hydraulic_mpa = history[
    hydraulic_columns
].to_numpy(float)

if not np.all(np.isfinite(hydraulic_mpa)):
    raise ValueError(
        "Non-finite regional hydraulic pressures found."
    )

# Analytical ordering: x outermost, y innermost.
# Established from tile_cuboid() enumeration.
permutation = np.array(
    [0, 3, 6, 1, 4, 7, 2, 5, 8],
    dtype=int
)

analytical_pa = (
    hydraulic_mpa[:, permutation] * 1e6
)

H = G @ B

if H.shape != (32, 9):
    raise ValueError(
        f"Expected sensitivity matrix (32, 9), got {H.shape}"
    )

if len(channels) != 32:
    raise ValueError("Expected 32 strain channels.")

if training.shape != (32,) or training.sum() != 24:
    raise ValueError(
        "Expected 24 training and 8 withheld channels."
    )

if not np.allclose(B, np.eye(9)):
    raise ValueError(
        "Expected identity pressure basis for nine zones."
    )

if not np.all(np.isfinite(H)):
    raise ValueError(
        "Non-finite analytical sensitivities found."
    )

# Verify actual physical domain dimensions.
dx = float(np.mean(np.diff(x)))
dy = float(np.mean(np.diff(y)))

if not np.allclose(np.diff(x), dx):
    raise ValueError("Nonuniform hydraulic x grid.")

if not np.allclose(np.diff(y), dy):
    raise ValueError("Nonuniform hydraulic y grid.")

extent = [
    x[0] - dx / 2,
    x[-1] + dx / 2,
    y[0] - dy / 2,
    y[-1] + dy / 2,
]

print("Hydraulic-to-analytical mapping verified.")
print("Hydraulic pressure shape:", hydraulic_mpa.shape)
print("Analytical pressure shape:", analytical_pa.shape)
print("Sensitivity matrix shape:", H.shape)
print("Domain extent:", extent)


# ============================================================
# 3. CALCULATE ANALYTICAL STRAIN AND SNR
# ============================================================

# Strain in nanostrain.
strain = analytical_pa @ H.T

# Individual channel signal-to-noise ratio.
snr = np.abs(strain) / NOISE_NS

# Combined network SNR for 24 training channels.
network_snr = np.sqrt(
    np.sum(
        (strain[:, training] / NOISE_NS) ** 2,
        axis=1
    )
)

# RMS strain across training channels.
training_rms = np.sqrt(
    np.mean(strain[:, training] ** 2, axis=1)
)

# Withheld-channel network SNR.
holdout_snr = np.sqrt(
    np.sum(
        (strain[:, ~training] / NOISE_NS) ** 2,
        axis=1
    )
)

prediction = pd.DataFrame({
    "time_h": times,
    "network_snr_training": network_snr,
    "network_snr_holdout": holdout_snr,
    "training_rms_strain_ns": training_rms,
})

for j, channel in enumerate(channels):
    prediction[channel] = strain[:, j]

prediction.to_csv(
    OUT / "injection_strain_predictions.csv",
    index=False
)


# ============================================================
# 4. PRESSURE HISTORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5.5),
    constrained_layout=True
)

pressure_series = [
    ("well_cell_mpa", "Well cell"),
    ("zone_row1_col1_mpa", "Central zone average"),
    ("mean_mpa", "Domain average"),
]

for col, label in pressure_series:
    ax.plot(
        times,
        history[col].to_numpy(float) * 1000,
        linewidth=2,
        label=label
    )

ax.axvline(
    SHUTIN_H,
    color="black",
    linestyle="--",
    label="Injection stops"
)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Pressure change (kPa)")
ax.set_title(
    "30-gpm injection (19 h) and shut-in recovery (77 h)"
)
ax.grid(alpha=0.25)
ax.legend()

fig.savefig(
    OUT / "pressure_history_kpa.png"
)
plt.close(fig)


# ============================================================
# 5. IMPROVED FINE-GRID AND NINE-ZONE PRESSURE MAPS
# ============================================================

def plot_pressure_maps(target_h):

    idx = int(np.argmin(
        np.abs(field_times - target_h)
    ))

    if abs(field_times[idx] - target_h) > 0.01:
        print(
            f"Skipping {target_h} h: no saved FD snapshot."
        )
        return

    # For t=0 use zero pressure.
    if target_h == 0:
        zones_kpa = np.zeros((3, 3))
    else:
        row = int(np.argmin(
            np.abs(times - target_h)
        ))

        if abs(times[row] - target_h) > 0.01:
            raise ValueError(
                f"No pressure-history row at {target_h} h"
            )

        zones_kpa = (
            hydraulic_mpa[row].reshape(3, 3) * 1000
        )

    fine_kpa = field_pa[idx] / 1000

    peak_kpa = max(
        float(np.max(np.abs(fine_kpa))),
        float(np.max(np.abs(zones_kpa)))
    )

    negligible = peak_kpa < DISPLAY_THRESHOLD_KPA

    if negligible:
        fine_display = np.zeros_like(fine_kpa)
        zones_display = np.zeros_like(zones_kpa)
        vmax = DISPLAY_THRESHOLD_KPA
    else:
        fine_display = fine_kpa
        zones_display = zones_kpa
        vmax = peak_kpa

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13.5, 6),
        constrained_layout=True
    )

    norm = Normalize(vmin=0, vmax=vmax)

    axes[0].imshow(
        fine_display,
        origin="lower",
        extent=extent,
        cmap=CMAP,
        norm=norm,
        interpolation="nearest",
        aspect="equal"
    )

    axes[0].scatter(
        [0], [0],
        marker="+",
        color="red",
        s=140,
        linewidths=2,
        label="Injection well"
    )

    axes[0].legend(loc="upper right")

    axes[0].set_title(
        f"Fine-grid pressure change — {target_h:g} h"
    )

    im = axes[1].imshow(
        zones_display,
        origin="lower",
        extent=extent,
        cmap=CMAP,
        norm=norm,
        interpolation="nearest",
        aspect="equal"
    )

    axes[1].set_title(
        f"Nine zone-averaged pressures — {target_h:g} h"
    )

    # Label the nine hydraulic pressure regions.
    for iy in range(3):
        for ix in range(3):

            cx = extent[0] + (
                ix + 0.5
            ) * (extent[1] - extent[0]) / 3

            cy = extent[2] + (
                iy + 0.5
            ) * (extent[3] - extent[2]) / 3

            value = zones_kpa[iy, ix]

            if negligible:
                label = "~0"
            elif abs(value) >= 0.01:
                label = f"{value:.3f}"
            else:
                label = f"{value:.2e}"

            normalized = (
                zones_display[iy, ix] / vmax
                if vmax > 0 else 0
            )

            text_color = (
                "black" if normalized > 0.65
                else "white"
            )

            axes[1].text(
                cx, cy, label,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="medium"
            )

    # Plot 3x3 region boundaries.
    for ax in axes:

        for fraction in (1/3, 2/3):

            xx = extent[0] + fraction * (
                extent[1] - extent[0]
            )

            yy = extent[2] + fraction * (
                extent[3] - extent[2]
            )

            ax.axvline(
                xx, color="white",
                linewidth=0.8, alpha=0.5
            )

            ax.axhline(
                yy, color="white",
                linewidth=0.8, alpha=0.5
            )

        ax.set_xlabel("Local x (m)")
        ax.set_ylabel("Local y (m)")

    # Shared colorbar outside both subplots.
    cbar = fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.035,
        pad=0.035,
        shrink=0.85
    )

    cbar.set_label("Pressure change (kPa)")

    if negligible:
        fig.suptitle(
            f"{target_h:g} h: pressure below "
            f"{DISPLAY_THRESHOLD_KPA:g} kPa "
            "(displayed as zero)"
        )
    else:
        fig.suptitle(
            f"Injection pressure distribution at {target_h:g} h"
        )

    fig.savefig(
        OUT / f"pressure_maps_{target_h:g}h_kpa.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


for target in PLOT_TIMES_H:
    plot_pressure_maps(target)


# ============================================================
# 6. ANALYTICAL STRAIN HISTORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 5.5),
    constrained_layout=True
)

for j, channel in enumerate(channels):
    ax.plot(
        times,
        strain[:, j],
        linewidth=1,
        alpha=0.55
    )

ax.axvline(
    SHUTIN_H,
    color="black",
    linestyle="--",
    label="Shut-in"
)

ax.axhline(
    NOISE_NS,
    color="red",
    linestyle=":",
    label="+1 noise SD"
)

ax.axhline(
    -NOISE_NS,
    color="red",
    linestyle=":",
    label="-1 noise SD"
)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Predicted strain change (nanostrain)")
ax.set_title(
    "Predicted strain from injection-generated pressure"
)
ax.grid(alpha=0.2)
ax.legend()

fig.savefig(
    OUT / "predicted_strain_history.png"
)
plt.close(fig)


# Additional zoomed-in strain plot.
fig, ax = plt.subplots(
    figsize=(11, 5.5),
    constrained_layout=True
)

for j in range(len(channels)):
    ax.plot(
        times,
        strain[:, j],
        linewidth=1,
        alpha=0.6
    )

ax.axvline(
    SHUTIN_H,
    color="black",
    linestyle="--"
)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Predicted strain change (nanostrain)")
ax.set_title(
    "Predicted strain history — signal-scale view"
)
ax.grid(alpha=0.2)

fig.savefig(
    OUT / "predicted_strain_history_zoom.png"
)
plt.close(fig)


# ============================================================
# 7. NETWORK SIGNAL-TO-NOISE RATIO
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5.5),
    constrained_layout=True
)

ax.plot(
    times,
    network_snr,
    linewidth=2,
    label="Training network (24 channels)"
)

ax.plot(
    times,
    holdout_snr,
    linewidth=2,
    linestyle="--",
    label="Withheld network (8 channels)"
)

ax.axhline(
    1,
    color="red",
    linestyle=":",
    label="SNR = 1"
)

ax.axvline(
    SHUTIN_H,
    color="black",
    linestyle="--",
    label="Shut-in"
)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Combined network SNR")
ax.set_title("Strain signal detectability")
ax.legend()
ax.grid(alpha=0.2)

fig.savefig(
    OUT / "strain_snr_history.png"
)
plt.close(fig)


# ============================================================
# 8. PEAK STRAIN SNR BY STATION
# ============================================================

stations = sorted({
    channel.split("_")[-1]
    for channel in channels
})

station_peaks = {}
station_peak_strain = {}

for station in stations:

    indices = [
        j for j, channel in enumerate(channels)
        if channel.endswith("_" + station)
    ]

    station_peaks[station] = float(
        np.max(snr[:, indices])
    )

    station_peak_strain[station] = float(
        np.max(np.abs(strain[:, indices]))
    )

fig, ax = plt.subplots(
    figsize=(10, 5),
    constrained_layout=True
)

values = [
    station_peaks[s]
    for s in stations
]

ax.bar(stations, values)

ax.axhline(
    1,
    color="red",
    linestyle="--",
    label="SNR = 1"
)

ax.set_xlabel("Station")
ax.set_ylabel("Peak channel SNR")
ax.set_title("Peak strain detectability by station")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.savefig(
    OUT / "station_peak_strain_snr.png"
)
plt.close(fig)


# ============================================================
# 9. NUMERICAL SUMMARY
# ============================================================

summary = {
    "scope": (
        "Fixed-geometry hydraulic-to-analytical "
        "synthetic strain detectability"
    ),
    "injection_rate_gpm": 30.0,
    "injection_hours": SHUTIN_H,
    "total_hours": 96.0,
    "noise_std_ns": NOISE_NS,
    "n_hydraulic_times": len(times),
    "n_strain_channels": len(channels),
    "n_training_channels": int(training.sum()),
    "n_holdout_channels": int((~training).sum()),
    "peak_abs_strain_ns": float(
        np.max(np.abs(strain))
    ),
    "peak_training_rms_strain_ns": float(
        np.max(training_rms)
    ),
    "peak_network_snr": float(
        np.max(network_snr)
    ),
    "peak_holdout_network_snr": float(
        np.max(holdout_snr)
    ),
    "peak_channel_snr": float(
        np.max(snr)
    ),
    "peak_station_snr": station_peaks,
    "peak_station_strain_ns": station_peak_strain,
    "hydraulic_to_analytical_permutation": (
        permutation.tolist()
    ),
    "hydraulic_axes_verified": (
        HYDRAULIC_AXES_VERIFIED
    ),
    "pressure_display_threshold_kpa": (
        DISPLAY_THRESHOLD_KPA
    ),
    "warning": (
        "Same-model synthetic diagnostic only. "
        "No observed field strain or Bayesian "
        "pressure inversion is used."
    )
}

summary_path = (
    OUT / "injection_detectability_summary.json"
)

summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nDETECTABILITY SUMMARY")
print(json.dumps(summary, indent=2))
print("\nResults saved to:", OUT)
