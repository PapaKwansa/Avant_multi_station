import numpy as np
import matplotlib.pyplot as plt
import multi_stations_input as input_parameters
from forward_model_multi_station import forward_model_multi_station

# ------------------------------------------------------------
# Helper function to run the forward model for a SINGLE station
# ------------------------------------------------------------
def run_clean_forward(params, station_index=0):
    """Return clean forward-model DF for a selected station."""
    Ns = len(params["x_prime"])
    assert station_index < Ns, "station index out of range"

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

    # extract only this station’s 6 components
    strain_cols = [
        f"Epsilon_XX_nanostrain_FS{station_index+1:02d}",
        f"Epsilon_YY_nanostrain_FS{station_index+1:02d}",
        f"Epsilon_ZZ_nanostrain_FS{station_index+1:02d}",
        f"Epsilon_XY_nanostrain_FS{station_index+1:02d}",
        f"Epsilon_XZ_nanostrain_FS{station_index+1:02d}",
        f"Epsilon_YZ_nanostrain_FS{station_index+1:02d}",
    ]

    return df[strain_cols], params["time"]


# ------------------------------------------------------------
# PLOT: baseline and perturbed curves for each strain component
# ------------------------------------------------------------
def plot_sensitivity(baseline_df, varied_df, time, pname, factor):
    """Plot how varying a parameter changes time-series."""
    components = baseline_df.columns

    plt.figure(figsize=(13, 8))

    for i, comp in enumerate(components):
        plt.subplot(2, 3, i + 1)

        plt.plot(time, baseline_df[comp], label="Baseline", linewidth=2)
        plt.plot(time, varied_df[comp], "--", label=f"{pname} × {factor}", linewidth=2)

        plt.title(comp.replace("_", " "))
        plt.xlabel("Time [days]")
        plt.ylabel("Strain [nano-strain]")
        plt.grid(True)

        if i == 0:
            plt.legend()

    plt.suptitle(f"Time-Series Sensitivity: Parameter {pname}")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    params = input_parameters.read_input()
    base_params = params.copy()

    Station_index = 1  # Analyze second station (FS02)

    print("Running baseline forward model...")
    baseline_df, time = run_clean_forward(params, station_index=Station_index)

    # --- PARAMETERS TO TEST ---
    vary_dict = {
    # Geometry of pressurized zone
    "a": 0.20,          # +20%
    "b": 0.20,
    "c": 0.20,

    # Elastic properties
    "E": 0.20,
    "nu": 0.05,         # +5% (Poisson's ratio is small)

    # Depth and thickness
    "h": 0.20,          # +20%
    "z": 0.20,          # if z is variable

    # Loading parameters
    "pmax": 0.20,
    "tpeak": 0.20,
    "d": 0.20,
    "alpha": 0.20,

    # Source location parameters
    "x0_prime": 0.10,
    "y0_prime": 0.10,

    # Orientation
    "theta_deg": 10.0    # +5 degrees
}


    for pname, delta in vary_dict.items():
        print(f"\n--- VARYING PARAMETER: {pname} ---")

        # create modified copy
        test_params = params.copy()

        if pname == "theta_deg":
            test_params[pname] = params[pname] + delta  # add degrees
            factor_label = f"+{delta}°"
        else:
            test_params[pname] = params[pname] * (1 + delta)
            factor_label = f"+{int(delta * 100)}%"

        # run perturbed model
        varied_df, _ = run_clean_forward(test_params, station_index=Station_index)

        # PLOT
        plot_sensitivity(baseline_df, varied_df, time, pname, factor_label)

    print("\nSensitivity analysis completed.")
