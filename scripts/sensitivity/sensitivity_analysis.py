# sensitivity_analysis.py
import os
import numpy as np
import pandas as pd
from scipy.stats import qmc, pearsonr, spearmanr
import matplotlib.pyplot as plt

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input

# -------------------- Parameters --------------------
param_defs = {
    "a":   (50.0, 300.0),
    "b":   (10.0, 100.0),
    "c":   (50.0, 300.0),
    "pmax":(1e6, 2e7),
    "alpha": (0.2, 1.0),
    "E":   (1e9, 1e11),
    "nu":  (0.2, 0.35),
    "h":   (500.0, 5000.0),
    "theta_deg": (-30.0, 30.0),
    "x0_prime": (-500.0, 500.0),
    "y0_prime": (-500.0, 500.0),
}

param_names = list(param_defs.keys())

base_components = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

# -------------------- LHS Sampler --------------------
def sample_lhs(n_samples):
    lb = np.array([param_defs[n][0] for n in param_names])
    ub = np.array([param_defs[n][1] for n in param_names])
    sampler = qmc.LatinHypercube(d=len(param_names))
    u = sampler.random(n=n_samples)
    samples = qmc.scale(u, lb, ub)
    return samples

# -------------------- Model Wrapper --------------------
def model_scalar_summary(sample, fixed_args):
    kwargs = {name: float(val) for name, val in zip(param_names, sample)}
    model_out = forward_model_multi_station(
        pmax=kwargs.get("pmax", fixed_args["pmax"]),
        tpeak=fixed_args["tpeak"],
        d=fixed_args["d"],
        time=fixed_args["time"],
        x_prime=fixed_args["x_prime"],
        y_prime=fixed_args["y_prime"],
        x0_prime=kwargs.get("x0_prime", fixed_args["x0_prime"]),
        y0_prime=kwargs.get("y0_prime", fixed_args["y0_prime"]),
        z=fixed_args["z"],
        a=kwargs.get("a", fixed_args["a"]),
        b=kwargs.get("b", fixed_args["b"]),
        c=kwargs.get("c", fixed_args["c"]),
        nu=kwargs.get("nu", fixed_args["nu"]),
        h=kwargs.get("h", fixed_args["h"]),
        E=kwargs.get("E", fixed_args["E"]),
        theta_deg=kwargs.get("theta_deg", fixed_args["theta_deg"]),
        alpha=kwargs.get("alpha", fixed_args["alpha"]),
    )

    outputs = {}
    ts_list = []
    for comp in base_components:
        comp_cols = [c for c in model_out.columns if c.startswith(comp)]
        arr = model_out[comp_cols].values  # Nt x Ns
        outputs[f"{comp}_peak_abs_max"] = np.max(np.abs(arr))
        outputs[f"{comp}_rms_avg"] = np.sqrt(np.mean(arr**2))
        ts_list.append(model_out[comp_cols])  # Save time-series per component

    return outputs, ts_list

# -------------------- Run Experiment --------------------
def run_experiment(n_samples=200):
    params_input = multi_stations_input.read_input()
    fixed_args = {
        "pmax": params_input["pmax"],
        "tpeak": params_input["tpeak"],
        "d": params_input["d"],
        "time": np.array(params_input["time"]),
        "x_prime": np.array(params_input["x_prime"]),
        "y_prime": np.array(params_input["y_prime"]),
        "x0_prime": params_input["x0_prime"],
        "y0_prime": params_input["y0_prime"],
        "z": params_input["z"],
        "a": params_input["a"],
        "b": params_input["b"],
        "c": params_input["c"],
        "nu": params_input["nu"],
        "h": params_input["h"],
        "E": params_input["E"],
        "theta_deg": params_input["theta_deg"],
        "alpha": params_input["alpha"],
    }

    os.makedirs("sensitivity_outputs", exist_ok=True)
    out_dir = "sensitivity_outputs"

    samples = sample_lhs(n_samples)
    results = []
    ts_all = []

    print(f"[INFO] Sampling {n_samples} parameter sets using LHS...")
    for idx, s in enumerate(samples, 1):
        out, ts_list = model_scalar_summary(s, fixed_args)
        results.append(out)
        ts_all.append(ts_list)
        if idx % (n_samples//4) == 0:
            print(f"  evaluated {idx}/{n_samples} samples")

    # Save combined scalar results
    param_df = pd.DataFrame(samples, columns=param_names)
    scalar_df = pd.DataFrame(results)
    combined_df = pd.concat([param_df.reset_index(drop=True), scalar_df.reset_index(drop=True)], axis=1)
    combined_df.to_csv(os.path.join(out_dir, "sensitivity_samples_and_outputs.csv"), index=False)
    print("[INFO] Saved combined sample+scalar table.")

    # -------------------- Correlations --------------------
    pearson_corr = {}
    spearman_corr = {}
    for out_col in scalar_df.columns:
        pearson_corr[out_col] = {p: pearsonr(combined_df[p], combined_df[out_col])[0] for p in param_names}
        spearman_corr[out_col] = {p: spearmanr(combined_df[p], combined_df[out_col])[0] for p in param_names}

        # Tornado plot
        vals = pearson_corr[out_col]
        ranked = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)
        plt.figure(figsize=(8, 5))
        plt.bar([x[0] for x in ranked], [x[1] for x in ranked])
        plt.xticks(rotation=45)
        plt.ylabel("Pearson correlation")
        plt.title(f"Tornado plot for {out_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tornado_{out_col}.png"))
        plt.close()

    # -------------------- Time-series sensitivity --------------------
    print("[INFO] Computing time-series sensitivity...")
    for comp_idx, comp in enumerate(base_components):
        station_cols = [c for c in ts_all[0][comp_idx].columns]
        n_stations = len(station_cols)
        n_times = len(fixed_args["time"])
        corr_time = np.zeros((len(param_names), n_times, n_stations))

        for s_idx, col in enumerate(station_cols):
            ts_matrix = np.array([ts[comp_idx][col].values for ts in ts_all])  # n_samples x n_times
            for i, pname in enumerate(param_names):
                pvec = samples[:, i]
                for t in range(n_times):
                    try:
                        corr_time[i, t, s_idx] = pearsonr(pvec, ts_matrix[:, t])[0]
                    except Exception:
                        corr_time[i, t, s_idx] = 0.0

        # Plot average across stations
        plt.figure(figsize=(12, 6))
        for i, pname in enumerate(param_names):
            avg_corr = corr_time[i, :, :].mean(axis=1)
            plt.plot(fixed_args["time"], avg_corr, label=pname)
        plt.xlabel("Time (days)")
        plt.ylabel("Pearson correlation")
        plt.title(f"Time-series sensitivity: {comp} (average across stations)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"time_series_sensitivity_{comp}.png"))
        plt.close()

    return combined_df, pearson_corr, spearman_corr

# -------------------- Main --------------------
if __name__ == "__main__":
    combined, pearson_corr, spearman_corr = run_experiment(n_samples=200)
    # Print top parameters for the first component (example)
    first_out = list(pearson_corr.keys())[0]
    ranked = sorted(pearson_corr[first_out].items(), key=lambda x: abs(x[1]), reverse=True)
    print("[RESULT] Top parameters by Pearson (abs) for", first_out)
    for k, v in ranked[:10]:
        print(k, v)
