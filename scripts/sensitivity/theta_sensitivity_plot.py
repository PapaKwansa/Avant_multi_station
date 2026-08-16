import numpy as np
import matplotlib.pyplot as plt
import os

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
    cols = {
        "xx": f"Epsilon_XX_nanostrain_{tag}",
        "yy": f"Epsilon_YY_nanostrain_{tag}",
        "zz": f"Epsilon_ZZ_nanostrain_{tag}",
        "xy": f"Epsilon_XY_nanostrain_{tag}",
        "xz": f"Epsilon_XZ_nanostrain_{tag}",
        "yz": f"Epsilon_YZ_nanostrain_{tag}",
    }

    return (
        df[cols["xx"]], df[cols["yy"]], df[cols["zz"]],
        df[cols["xy"]], df[cols["xz"]], df[cols["yz"]]
    )


# ------------------------------------------------------------
# Plot strain vs time for multiple theta values
# ------------------------------------------------------------
def plot_theta_sweep(time, strain_dict, component, station_tag):
    plt.figure(figsize=(10, 6))

    for theta, series in strain_dict.items():
        plt.plot(time, series, label=f"θ = {theta}°", linewidth=2)

    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (days)")
    plt.ylabel(f"{component} (nanostrain)")
    plt.title(f"{component} vs θ — {station_tag}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()  # Commented out for saving
    plt.savefig(f"theta_sensitivity_plots/{component}_{station_tag}_theta_sweep.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":

    # Create output folder
    os.makedirs('theta_sensitivity_plots', exist_ok=True)

    # Load parameters
    params = input_parameters.read_input()
    time = np.array(params["time"])

    # All stations
    num_stations = len(params["x_prime"])
    stations = list(range(num_stations))  # All stations

    # Absolute orientation angles to test
    theta_list = [10, 30, 50, 70, 90]  # Can expand

    for station in stations:

        station_tag = f"FS{station+1:02d}"
        print(f"\n===== Theta rotation sensitivity for {station_tag} =====")

        # Storage for all components
        eps_xx = {}
        eps_yy = {}
        eps_zz = {}
        eps_xy = {}
        eps_xz = {}
        eps_yz = {}

        for theta in theta_list:
            print(f"  Running θ = {theta}°")

            test_params = params.copy()
            test_params["theta_deg"] = theta

            exx, eyy, ezz, exy, exz, eyz = run_forward_station(test_params, station)

            eps_xx[theta] = exx
            eps_yy[theta] = eyy
            eps_zz[theta] = ezz
            eps_xy[theta] = exy
            eps_xz[theta] = exz
            eps_yz[theta] = eyz

        # Plot all components
        plot_theta_sweep(time, eps_xx, "εxx", station_tag)
        plot_theta_sweep(time, eps_yy, "εyy", station_tag)
        plot_theta_sweep(time, eps_zz, "εzz", station_tag)
        plot_theta_sweep(time, eps_xy, "εxy", station_tag)
        plot_theta_sweep(time, eps_xz, "εxz", station_tag)
        plot_theta_sweep(time, eps_yz, "εyz", station_tag)

    print("\nTheta rotation sensitivity analysis completed for all stations.")