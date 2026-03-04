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
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    params = input_parameters.read_input()
    time = np.array(params["time"])

    stations = list(range(len(params["station_names"])))  # All stations
    theta_list = [10, 30, 50, 70, 90]

    for station in stations:

        plt.figure(figsize=(14, 8))

        for theta in theta_list:
            test_params = params.copy()
            test_params["theta_deg"] = theta

            df = run_forward_station(test_params, station)

            # Plot only horizontal strains (most diagnostic)
            for comp in ["XX", "YY", "XY"]:
                col = f"Epsilon_{comp}_nanostrain_{params["station_names"][station]}"
                plt.plot(
                    time,
                    df[col],
                    label=f"{comp}, θ={theta}°"
                )

        plt.xlabel("Time (days)")
        plt.ylabel("Strain (nanostrain)")
        plt.title(f"Orientation sweep — FS{station+1:02d}")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
