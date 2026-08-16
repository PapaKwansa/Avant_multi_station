import numpy as np
import matplotlib.pyplot as plt

import multi_stations_input as input_parameters
from forward_model_multi_station import forward_model_multi_station

# ------------------------------------------------------------
# Run forward model and extract strain for ONE station
# ------------------------------------------------------------
def run_forward_station(params, station_index):
    df = forward_model_multi_station(
        pmax=params["pmax"],
        tpeak=params["tpeak"],
        d=params["d"],
        time=np.array(params["time"]),
        x_prime=np.array(params["x_prime"]),
        y_prime=np.array(params["y_prime"]),
        x0_prime=params["x0_prime"],
        y0_prime=params["y0_prime"],
        z=params["z"],
        a=params["a"],
        b=params["b"],
        c=params["c"],
        nu=params["nu"],
        h=params["h"],
        E=params["E"],
        theta_deg=params["theta_deg"],
        alpha=params.get("alpha", None),
    )

    tag = f"FS{station_index+1:02d}"
    cols = [f"Epsilon_{c}_nanostrain_{tag}" for c in ["XX","YY","ZZ","XY","XZ","YZ"]]

    return df[cols].copy()


# ------------------------------------------------------------
# Compute normalized sensitivity
# ------------------------------------------------------------
def compute_sensitivity(base_df, pert_df, delta_p):
    return (pert_df - base_df) / delta_p


# ------------------------------------------------------------
# Plot Δstrain
# ------------------------------------------------------------
def plot_delta(delta_df, time, pname, station):
    plt.figure(figsize=(13, 8))
    for i, col in enumerate(delta_df.columns):
        plt.subplot(2, 3, i + 1)
        plt.plot(time, delta_df[col], linewidth=2)
        plt.title(f"{col}")
        plt.grid(True)

    plt.suptitle(f"Δ Strain — {pname} — Station {station}")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Plot normalized sensitivity
# ------------------------------------------------------------
def plot_sensitivity(sens_df, time, pname, station):
    plt.figure(figsize=(13, 8))
    for i, col in enumerate(sens_df.columns):
        plt.subplot(2, 3, i + 1)
        plt.plot(time, sens_df[col], linewidth=2)
        plt.title(f"{col}")
        plt.grid(True)

    plt.suptitle(f"Normalized Sensitivity — {pname} — Station {station}")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------
if __name__ == "__main__":

    params = input_parameters.read_input()
    time = np.array(params["time"])

    stations = list(range(len(params["station_names"])))  # All stations

    vary_dict = {
        "a": 0.2,
        "b": 0.2,
        "c": 0.2,
        "h": 0.2,
        "E": 0.2,
        "nu": 0.05,
        "pmax": 0.2,
        "theta_deg": 60.0,
    }

    for station in stations:

        print(f"\n===== Station {params["station_names"][station]} =====")

        base_df = run_forward_station(params, station)

        for pname, delta in vary_dict.items():

            print(f"  Sensitivity to {pname}")

            test_params = params.copy()

            if pname == "theta_deg":
                test_params[pname] = params[pname] + delta
                delta_p = np.deg2rad(delta)
            else:
                test_params[pname] = params[pname] * (1 + delta)
                delta_p = params[pname] * delta

            pert_df = run_forward_station(test_params, station)

            delta_df = pert_df - base_df
            sens_df = compute_sensitivity(base_df, pert_df, delta_p)

            plot_delta(delta_df, time, pname, f"FS{station+1:02d}")
            plot_sensitivity(sens_df, time, pname, f"FS{station+1:02d}")
