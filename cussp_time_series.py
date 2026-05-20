import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as inp

# ============================================================
# USER SETTINGS
# ============================================================
OUTPUT_DIR = "poster_quality_plots"
PLOT_ALL_STATIONS = False   # set True to save all stations
STATION_TO_PLOT = "S1"      # change to the station you want on the poster

# ============================================================
# POSTER-QUALITY STYLE
#   - larger fonts
#   - regular weight (not bold everywhere)
#   - clearer legend
#   - thicker lines
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.linewidth": 1.8,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "legend.fontsize": 14,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
    "savefig.dpi": 600,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD INPUT PARAMETERS
# ============================================================
params = inp.read_input()

station_names = [str(s).strip() for s in params["station_names"]]
n_stations = len(station_names)

# ============================================================
# RUN FORWARD MODEL
# ============================================================
df_model = forward_model_multi_station(
    params["pmax"], params["tpeak"], params["d"], params["time"],
    params["x_prime"], params["y_prime"],
    params["x0_prime"], params["y0_prime"],
    params["z"], params["a_fixed"], params["b_fixed"], params["c_fixed"],
    params["nu"], params["h"], params["E"], params["theta_deg"],
    alpha=params.get("alpha", 0.8),
    station_names=params["station_names"]
)

# ============================================================
# LOAD OBSERVED DATA
# ============================================================
df_obs = pd.read_csv("avant_cleaned_strain.csv")

time_obs = df_obs["time_s"].values
time_model = df_model["time_s"].values

# ============================================================
# CHECK STATION / COLUMN CONSISTENCY
# ============================================================
obs_cols = [f"strain_{i}" for i in range(1, 4 * n_stations + 1)]
if len(obs_cols) != 4 * n_stations:
    raise ValueError(
        f"Observed file has {len(obs_cols)} strain columns, "
        f"but expected {4 * n_stations} for {n_stations} stations."
    )

components = ["eXX", "eYY", "eXY", "eZZ"]

# Cleaner, more readable legend labels
pretty_labels = {
    "eXX": r"$\epsilon_{xx}$",
    "eYY": r"$\epsilon_{yy}$",
    "eXY": r"$\epsilon_{xy}$",
    "eZZ": r"$\epsilon_{zz}$",
}

# More readable, distinct colors
colors = {
    "eXX": "#1f77b4",  # blue
    "eYY": "#ff7f0e",  # orange
    "eXY": "#2ca02c",  # green
    "eZZ": "#d62728",  # red
}

def plot_station_fit(station: str) -> None:
    station = str(station).strip()

    if station not in station_names:
        raise ValueError(
            f"Station '{station}' not found. Available stations: {station_names}"
        )

    i = station_names.index(station)

    # Map observed data columns for this station
    obs_map = {
        "eXX": f"strain_{i + 1}",
        "eYY": f"strain_{i + 1 + n_stations}",
        "eXY": f"strain_{i + 1 + 2 * n_stations}",
        "eZZ": f"strain_{i + 1 + 3 * n_stations}",
    }

    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)

    # Plot model and observed for each component
    for comp in components:
        model_col = f"{comp}_{station}"
        obs_col = obs_map[comp]

        if model_col not in df_model.columns:
            print(f"Warning: {model_col} not found.")
            continue

        if obs_col not in df_obs.columns:
            print(f"Warning: {obs_col} not found.")
            continue

        color = colors[comp]

        # Model
        ax.plot(
            time_model,
            df_model[model_col].values,
            color=color,
            linewidth=3.8,
            label=f"Model {pretty_labels[comp]}"
        )

        # Observed
        ax.plot(
            time_obs,
            df_obs[obs_col].values,
            linestyle="--",
            color=color,
            linewidth=3.0,
            alpha=0.95,
            label=f"Observed {pretty_labels[comp]}"
        )

    # Axis labels
    ax.set_xlabel("Time (s)", fontsize=22)
    ax.set_ylabel("Strain (nε)", fontsize=22)

    # Title
    ax.set_title(
        f"Model vs Observed Strain Tensor — Station {station}",
        fontsize=24,
        pad=16
    )

    # Grid
    ax.grid(True, linestyle="--", alpha=0.25)

    # Keep y-axis in plain numbers (no scientific notation clutter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    # Ticks
    ax.tick_params(axis="both", which="major", length=8, width=1.8, labelsize=16)

    # Spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    # Legend: larger and easier to read
    leg = ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
        handlelength=2.4,
        columnspacing=1.2,
        labelspacing=0.5,
        handletextpad=0.6
    )

    # Keep legend text normal weight for easier reading
    for text in leg.get_texts():
        text.set_fontweight("normal")

    # Slight margins so curves do not sit on the frame
    ax.margins(x=0.02, y=0.08)

    # Save files
    png_file = os.path.join(OUTPUT_DIR, f"model_vs_observed_{station}.png")
    pdf_file = os.path.join(OUTPUT_DIR, f"model_vs_observed_{station}.pdf")

    fig.savefig(
        png_file,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.08
    )
    fig.savefig(
        pdf_file,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.08
    )

    print(f"Saved: {png_file}")
    print(f"Saved: {pdf_file}")

    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
if PLOT_ALL_STATIONS:
    for station in station_names:
        plot_station_fit(station)
else:
    plot_station_fit(STATION_TO_PLOT)