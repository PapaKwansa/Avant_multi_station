import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as inp

# ============================================================
# GLOBAL PLOT STYLE (POSTER QUALITY)
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.0,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
    "savefig.dpi": 600,
})

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
output_dir = "poster_quality_plots"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# LOAD INPUT PARAMETERS
# ============================================================
params = inp.read_input()

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

# ============================================================
# TIME AXIS
# ============================================================
time_obs = df_obs["time_s"].values
time_model = df_model["time_s"].values

# ============================================================
# COMPONENT ORDER
# ============================================================
components = ["eXX", "eYY", "eXY", "eZZ"]

# ============================================================
# CHECK STATION COUNT
# ============================================================
n_stations = len(params["station_names"])
obs_cols = [f"strain_{i}" for i in range(1, 17)]

if len(obs_cols) != 4 * n_stations:
    raise ValueError(
        f"Observed file has {len(obs_cols)} strain columns, "
        f"but expected {4 * n_stations} "
        f"for {n_stations} stations and 4 components per station."
    )

# ============================================================
# COLOR MAP
# ============================================================
colors = {
    "eXX": "#1f77b4",  # blue
    "eYY": "#d62728",  # red
    "eXY": "#2ca02c",  # green
    "eZZ": "#9467bd",  # purple
}

# ============================================================
# PLOT MODEL + OBSERVED
# ============================================================
for i, station in enumerate(params["station_names"]):

    station = str(station).strip()

    # --------------------------------------------------------
    # OBSERVED COLUMN MAP
    # --------------------------------------------------------
    obs_map = {
        "eXX": f"strain_{i + 1}",
        "eYY": f"strain_{i + 1 + n_stations}",
        "eXY": f"strain_{i + 1 + 2 * n_stations}",
        "eZZ": f"strain_{i + 1 + 3 * n_stations}",
    }

    # --------------------------------------------------------
    # FIGURE SETUP
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    # --------------------------------------------------------
    # LOOP COMPONENTS
    # --------------------------------------------------------
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

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------
        ax.plot(
            time_model,
            df_model[model_col].values,
            color=color,
            linewidth=3.5,
            label=f"Model {comp}"
        )

        # ----------------------------------------------------
        # OBSERVED
        # ----------------------------------------------------
        ax.plot(
            time_obs,
            df_obs[obs_col].values,
            linestyle="--",
            color=color,
            linewidth=3,
            alpha=0.9,
            label=f"Observed {comp}"
        )

    # --------------------------------------------------------
    # AXIS LABELS
    # --------------------------------------------------------
    ax.set_xlabel("Time (s)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Strain", fontsize=20, fontweight="bold")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------
    ax.set_title(
        f"Model vs Observed Strain Tensor\nStation {station}",
        fontsize=22,
        fontweight="bold",
        pad=18
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------
    ax.grid(True, linestyle="--", alpha=0.25)

    # --------------------------------------------------------
    # SCIENTIFIC NOTATION
    # --------------------------------------------------------
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)

    # --------------------------------------------------------
    # THICKER SPINES
    # --------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------
    leg = ax.legend(
        ncol=2,
        loc="best",
        fontsize=13
    )

    for text in leg.get_texts():
        text.set_fontweight("bold")

    # --------------------------------------------------------
    # TICKS
    # --------------------------------------------------------
    ax.tick_params(axis='both', which='major',
                   length=8, width=2)

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------
    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE FIGURES
    # --------------------------------------------------------
    png_file = os.path.join(
        output_dir,
        f"model_vs_observed_{station}.png"
    )

    pdf_file = os.path.join(
        output_dir,
        f"model_vs_observed_{station}.pdf"
    )

    plt.savefig(
        png_file,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.savefig(
        pdf_file,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"Saved: {png_file}")
    print(f"Saved: {pdf_file}")

    plt.show()
