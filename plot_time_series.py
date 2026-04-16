import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as inp

# ------------------------------------------------------------
# LOAD INPUT PARAMETERS
# ------------------------------------------------------------
params = inp.read_input()

# ------------------------------------------------------------
# RUN FORWARD MODEL
# ------------------------------------------------------------
df_model = forward_model_multi_station(
    params["pmax"], params["tpeak"], params["d"], params["time"],
    params["x_prime"], params["y_prime"],
    params["x0_prime"], params["y0_prime"],
    params["z"], params["a_fixed"], params["b_fixed"], params["c_fixed"],
    params["nu"], params["h"], params["E"], params["theta_deg"],
    alpha=params.get("alpha", 0.8),
    station_names=params["station_names"]
)

# ------------------------------------------------------------
# LOAD OBSERVED DATA
# ------------------------------------------------------------
df_obs = pd.read_csv("avant_cleaned_strain.csv")

# ------------------------------------------------------------
# TIME ARRAYS IN SECONDS
# ------------------------------------------------------------
time_model = df_model["time_s"].values
time_obs = df_obs["time_s"].values

# ------------------------------------------------------------
# COMPONENT ORDER
# Must match the raw COMSOL export order
# ------------------------------------------------------------
components = ["eXX", "eYY", "eXY", "eZZ"]

# ------------------------------------------------------------
# OBSERVED COLUMN ORDER
# Assumes 4 columns per station in the same component order
# ------------------------------------------------------------
obs_cols = [col for col in df_obs.columns if col.startswith("strain_")]
obs_cols = sorted(obs_cols, key=lambda x: int(x.split("_")[1]))

n_stations_obs = len(obs_cols) // 4
if len(obs_cols) % 4 != 0:
    raise ValueError(
        f"Observed strain columns ({len(obs_cols)}) is not divisible by 4."
    )

if len(params["station_names"]) > n_stations_obs:
    raise ValueError(
        f"Model has {len(params['station_names'])} stations but observed file "
        f"only contains {n_stations_obs} station groups."
    )

# ------------------------------------------------------------
# PLOT SEPARATELY FOR EACH STATION
# ------------------------------------------------------------
# ---------------------------------------------
# The observed dataset has columns like:
# time_s, strain_eXX_station1, strain_eYY_station1, strain_eXY_station1, strain_eZZ_station1,
# strain_eXX_station2, strain_eYY_station2, strain_eXY_station2, strain_eZZ_station2, ...
# We need to group these columns by station to plot them correctly.
# ---------------------------------------------
for i, station in enumerate(params["station_names"]):
    station = str(station).strip()

    # observed columns for this station: 4 columns per station
    obs_start = 4 * i
    obs_end = obs_start + 4
    station_obs_cols = obs_cols[obs_start:obs_end]

    # safety check
    if len(station_obs_cols) < 4:
        print(f"Warning: not enough observed columns for station {station}, skipping.")
        continue

    # ---------------------------
    # MODEL PLOT
    # ---------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for comp in components:
        model_col = f"{comp}_{station}"
        if model_col in df_model.columns:
            ax.plot(time_model, df_model[model_col].values, label=f"Model {comp}")
        else:
            print(f"Warning: {model_col} not found in model output.")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain (nanostrain)")
    ax.set_title(f"Model Strain Tensor Time Series — Station {station}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"model_time_series_{station}.png", dpi=300)
    plt.show()

    # ---------------------------
    # OBSERVED PLOT
    # ---------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for comp, obs_col in zip(components, station_obs_cols):
        ax.plot(time_obs, df_obs[obs_col].values, label=f"Observed {comp}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain (nanostrain)")
    ax.set_title(f"Observed Strain Tensor Time Series — Station {station}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"observed_time_series_{station}.png", dpi=300)
    plt.show()