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
# TIME AXIS
# ------------------------------------------------------------
time_obs = df_obs["time_s"].values
time_model = df_model["time_s"].values

# ------------------------------------------------------------
# COMPONENT ORDER
# ------------------------------------------------------------
components = ["eXX", "eYY", "eXY", "eZZ"]

# ------------------------------------------------------------
# CHECK STATION COUNT
# ------------------------------------------------------------
n_stations = len(params["station_names"])
obs_cols = [f"strain_{i}" for i in range(1, 17)]

if len(obs_cols) != 4 * n_stations:
    raise ValueError(
        f"Observed file has {len(obs_cols)} strain columns, but expected {4 * n_stations} "
        f"for {n_stations} stations and 4 components per station."
    )

# ------------------------------------------------------------
# PLOT MODEL + OBSERVED ON THE SAME FIGURE FOR EACH STATION
# ------------------------------------------------------------
for i, station in enumerate(params["station_names"]):
    station = str(station).strip()

    # Observed columns are component-major, not station-major
    obs_map = {
        "eXX": f"strain_{i + 1}",
        "eYY": f"strain_{i + 1 + n_stations}",
        "eXY": f"strain_{i + 1 + 2 * n_stations}",
        "eZZ": f"strain_{i + 1 + 3 * n_stations}",
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    for comp in components:
        model_col = f"{comp}_{station}"
        obs_col = obs_map[comp]

        if model_col not in df_model.columns:
            print(f"Warning: {model_col} not found in model output.")
            continue

        if obs_col not in df_obs.columns:
            print(f"Warning: {obs_col} not found in observed data.")
            continue

        # Plot model
        line_model, = ax.plot(
            time_model,
            df_model[model_col].values,
            label=f"Model {comp}"
        )

        # Plot observed with same color, dashed
        ax.plot(
            time_obs,
            df_obs[obs_col].values,
            linestyle="--",
            color=line_model.get_color(),
            label=f"Obs {comp}"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain (nanostrain)")
    ax.set_title(f"Model vs Observed Strain Tensor — Station {station}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"model_vs_observed_{station}.png", dpi=300)
    plt.show()