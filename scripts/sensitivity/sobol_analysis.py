# sobol_analysis.py
"""
Sobol sensitivity analysis for multi-station strain forward model.

Produces:
 - Scalar Sobol indices (S1, ST) for each component's peak and RMS.
 - Time-varying Sobol indices (S1, ST) for each component (averaged across stations by default).
 - CSVs and PNGs saved to ./sobol_outputs/

Notes:
 - Requires SALib: pip install SALib
 - Sampling cost can be high. Use small `N_base` (e.g., 64) for testing.
"""

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from functools import partial
from SALib.sample import saltelli
from SALib.analyze import sobol

# Import your model + input provider
from forward_model_multi_station import forward_model_multi_station
import multi_stations_input

# ---------------- USER CONFIG ----------------
OUTDIR = "sobol_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Parameters to include (match your param_defs earlier)
param_bounds = {
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

param_names = list(param_bounds.keys())
D = len(param_names)

# SALib problem dictionary
problem = {
    'num_vars': D,
    'names': param_names,
    'bounds': [param_bounds[n] for n in param_names],
}

# Which components to analyze
COMP_BASES = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

# Choose base sample size N (small to test; increase for stable indices)
# For SALib.saltelli.sample: total model evals = N*(D+2) when calc_second_order=False
N_base = 128  # TESTING: use 128 or 256. Increase for production.
calc_second_order = False

# Parallel workers (None -> mp.cpu_count())
N_PROCS = None

# ----------------------------------------------
def eval_single(params_row, fixed_args):
    """
    Evaluate model for a single parameter row (1D array) and return:
      - dict of scalar summaries {component+'_peak_abs_max', component+'_rms_avg'}
      - time-series dict {component: (Nt x Ns) numpy array}
    """
    # Map params to names
    kwargs = {name: float(val) for name, val in zip(param_names, params_row)}

    df = forward_model_multi_station(
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

    scalars = {}
    ts = {}
    for comp in COMP_BASES:
        comp_cols = [c for c in df.columns if c.startswith(comp)]
        arr = df[comp_cols].values  # Nt x Ns
        scalars[f"{comp}_peak_abs_max"] = np.max(np.abs(arr))
        scalars[f"{comp}_rms_avg"] = float(np.sqrt(np.mean(arr**2)))
        ts[comp] = arr.copy()  # Nt x Ns
    return scalars, ts

def worker_eval(X_rows, fixed_args):
    """Evaluate a block of samples (used by Pool.map). Returns list of (scalars, ts)."""
    out = [eval_single(row, fixed_args) for row in X_rows]
    return out

# ---------------- Main driver -------------------
def run_sobol(N_base=N_base, calc_second_order=calc_second_order, n_procs=N_PROCS):
    # load fixed args from multi_stations_input
    params_input = multi_stations_input.read_input()
    fixed_args = {
        "pmax": params_input["pmax"],
        "tpeak": params_input["tpeak"],
        "d": params_input["d"],
        "time": np.asarray(params_input["time"]),
        "x_prime": np.asarray(params_input["x_prime"]),
        "y_prime": np.asarray(params_input["y_prime"]),
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
        "alpha": params_input.get("alpha", None),
    }

    # Generate Saltelli samples
    print("[INFO] Generating Saltelli samples...")
    param_values = saltelli.sample(problem, N_base, calc_second_order=calc_second_order)
    Nsamp = param_values.shape[0]
    print(f"[INFO] Total parameter sets to evaluate: {Nsamp}")

    # Evaluate model for each sample (parallel)
    print("[INFO] Evaluating model for each sample (this may take time)...")
    if n_procs is None:
        n_procs = max(1, mp.cpu_count() - 1)

    # chunk param_values into batches per worker to reduce overhead
    chunks = np.array_split(param_values, n_procs)
    pool = mp.Pool(processes=n_procs)
    worker = partial(worker_eval, fixed_args=fixed_args)
    results = pool.map(worker, chunks)
    pool.close()
    pool.join()

    # flatten results (list of lists)
    flat = [item for sub in results for item in sub]  # length Nsamp; each item=(scalars, ts)
    # build scalar DataFrame and time-series array
    scalar_keys = list(flat[0][0].keys())
    scalar_vals = np.array([[item[0][k] for k in scalar_keys] for item in flat])  # Nsamp x n_scalars
    scalars_df = pd.DataFrame(scalar_vals, columns=scalar_keys)

    # time-series: for each comp -> array (Nsamp x Nt x Ns)
    # use first sample to get shapes
    Nt, Ns = flat[0][1][COMP_BASES[0]].shape
    ts_data = {comp: np.zeros((Nsamp, Nt, Ns), dtype=float) for comp in COMP_BASES}
    for i, item in enumerate(flat):
        ts_dict = item[1]
        for comp in COMP_BASES:
            ts_data[comp][i, :, :] = ts_dict[comp]

    # Save param matrix and scalars
    param_df = pd.DataFrame(param_values, columns=param_names)
    combined = pd.concat([param_df.reset_index(drop=True), scalars_df.reset_index(drop=True)], axis=1)
    combined.to_csv(os.path.join(OUTDIR, "sobol_param_scalar_table.csv"), index=False)
    print(f"[INFO] Saved parameter+scalar table to {OUTDIR}/sobol_param_scalar_table.csv")

    # ---------------- Scalar Sobol (peak & rms) ----------------
    print("[INFO] Running Sobol analysis on scalar outputs...")
    sobol_results = {}
    for col in scalar_keys:
        Y = scalars_df[col].values
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
        sobol_results[col] = Si
        # Save S1 & ST to CSV
        df_S = pd.DataFrame({
            'param': param_names,
            'S1': Si['S1'],
            'S1_conf': Si.get('S1_conf', [np.nan]*D),
            'ST': Si['ST'],
            'ST_conf': Si.get('ST_conf', [np.nan]*D)
        })
        df_S.to_csv(os.path.join(OUTDIR, f"sobol_scalar_{col}.csv"), index=False)

        # Plot S1 and ST bar charts
        x = np.arange(D)
        width = 0.35
        plt.figure(figsize=(10,5))
        plt.bar(x - width/2, Si['S1'], width, label='S1')
        plt.bar(x + width/2, Si['ST'], width, label='ST')
        plt.xticks(x, param_names, rotation=45, ha='right')
        plt.ylabel('Sobol index')
        plt.title(f"Sobol indices (scalar) — {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"sobol_bar_{col}.png"))
        plt.close()

    # ---------------- Time-varying Sobol ----------------
    # For each component, average across stations or do per-station analysis
    print("[INFO] Running time-varying Sobol (averaged across stations)...")
    time = fixed_args["time"]
    for comp in COMP_BASES:
        # aggregate per-sample by averaging across stations, for each time -> Y shape (Nsamp x Nt)
        Y_time = ts_data[comp].mean(axis=2)  # Nsamp x Nt
        # Prepare storage
        S1_t = np.zeros((Nt, D))
        ST_t = np.zeros((Nt, D))
        for t in range(Nt):
            Yt = Y_time[:, t]
            Si_t = sobol.analyze(problem, Yt, calc_second_order=False, print_to_console=False)
            S1_t[t, :] = Si_t['S1']
            ST_t[t, :] = Si_t['ST']
        # Save to CSV
        df_s1 = pd.DataFrame(S1_t, columns=param_names)
        df_st = pd.DataFrame(ST_t, columns=param_names)
        df_s1['time'] = time
        df_st['time'] = time
        df_s1.to_csv(os.path.join(OUTDIR, f"sobol_time_S1_{comp}.csv"), index=False)
        df_st.to_csv(os.path.join(OUTDIR, f"sobol_time_ST_{comp}.csv"), index=False)

        # Plot a few top parameters' S1 vs time (choose top 6 by max S1)
        maxS1 = S1_t.max(axis=0)
        top_idx = np.argsort(maxS1)[-6:][::-1]
        plt.figure(figsize=(10,5))
        for idx in top_idx:
            plt.plot(time, S1_t[:, idx], label=param_names[idx])
        plt.xlabel('Time (days)')
        plt.ylabel('S1 (first-order Sobol)')
        plt.title(f"Time-varying S1 (top params) — {comp}")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"sobol_time_S1_plot_{comp}.png"))
        plt.close()

    print("[INFO] All Sobol analysis finished. Results in", OUTDIR)
    return sobol_results, ts_data, combined

# Entry point
if __name__ == "__main__":
    sobol_results, ts_data, combined_table = run_sobol()
