"""
avant_sensitivity_sobol.py

Sobol sensitivity analysis for AVANT multi-station strain forward model.

- Uses SALib Saltelli sampling
- Computes scalar Sobol indices (S1, ST) for each strain component
- Computes time-varying Sobol indices (S1, ST) averaged across stations
- Saves CSVs and plots to ./sobol_outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from SALib.sample import saltelli  # Saltelli sampling
from SALib.analyze import sobol as sobol_analyze  # Sobol indices analysis

# ---------- Shared config + forward model wrapper ----------
from avant_sensitivity_config import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    COMP_BASES,
    load_fixed_args,
    extract_component_matrix,
    peak_abs_max,
    rms_avg,
    run_forward_model_from_params,
)

# =============================================================
OUTDIR = "sobol_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------- User config -----------------
N_BASE = 128           # Saltelli base sample size (increase for convergence)
CALC_SECOND_ORDER = False
N_PROCS = None         # Use all available CPUs by default

# =============================================================
# 1) Model evaluation wrapper for one row
# =============================================================

def eval_single(params_row, fixed_args):
    """
    Evaluate model for one parameter row and return:
    - dict of scalar summaries (peak, RMS)
    - dict of time-series arrays per component (Nt x Ns)
    """
    # Convert NumPy array -> dict for run_forward_model_from_params
    params_dict = {name: val for name, val in zip(PARAM_NAMES, params_row)}
    df = run_forward_model_from_params(params_dict, fixed_args)

    scalars = {}
    ts = {}
    for comp in COMP_BASES:
        arr = extract_component_matrix(df, comp)  # Nt x Ns
        scalars[f"{comp}_peak_abs_max"] = peak_abs_max(arr)
        scalars[f"{comp}_rms_avg"] = rms_avg(arr)
        ts[comp] = arr.copy()
    return scalars, ts

def worker_eval(rows_block, fixed_args):
    """Evaluate a block of parameter rows (for multiprocessing)"""
    return [eval_single(row, fixed_args) for row in rows_block]

# =============================================================
# 2) Run Sobol analysis
# =============================================================

def run_sobol(n_base=N_BASE, calc_second_order=CALC_SECOND_ORDER, n_procs=N_PROCS):
    fixed_args = load_fixed_args()

    # Saltelli sample
    problem = {
        'num_vars': len(PARAM_NAMES),
        'names': PARAM_NAMES,
        'bounds': [PARAM_BOUNDS[p] for p in PARAM_NAMES],
    }

    print("[INFO] Generating Saltelli samples...")
    param_values = saltelli.sample(problem, n_base, calc_second_order=calc_second_order)
    Nsamp = param_values.shape[0]
    print(f"[INFO] Total parameter sets to evaluate: {Nsamp}")

    # Parallel evaluation
    if n_procs is None:
        n_procs = max(1, mp.cpu_count() - 1)

    chunks = np.array_split(param_values, n_procs)
    pool = mp.Pool(n_procs)
    worker = partial(worker_eval, fixed_args=fixed_args)
    results = pool.map(worker, chunks)
    pool.close()
    pool.join()

    # Flatten results
    flat = [item for sublist in results for item in sublist]

    # Build scalar DataFrame
    scalar_keys = list(flat[0][0].keys())
    scalar_vals = np.array([[item[0][k] for k in scalar_keys] for item in flat])
    scalars_df = pd.DataFrame(scalar_vals, columns=scalar_keys)

    # Time-series data (per component)
    Nt, Ns = flat[0][1][COMP_BASES[0]].shape
    ts_data = {comp: np.zeros((Nsamp, Nt, Ns)) for comp in COMP_BASES}
    for i, item in enumerate(flat):
        for comp in COMP_BASES:
            ts_data[comp][i, :, :] = item[1][comp]

    # Save parameter + scalar table
    param_df = pd.DataFrame(param_values, columns=PARAM_NAMES)
    combined = pd.concat([param_df.reset_index(drop=True),
                          scalars_df.reset_index(drop=True)], axis=1)
    combined.to_csv(os.path.join(OUTDIR, "sobol_param_scalar_table.csv"), index=False)
    print(f"[INFO] Saved parameter + scalar table to {OUTDIR}/sobol_param_scalar_table.csv")

    # ----------------- Scalar Sobol indices -----------------
    print("[INFO] Computing scalar Sobol indices...")
    sobol_results = {}
    for col in scalar_keys:
        Y = scalars_df[col].values
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
        sobol_results[col] = Si

        # Save S1 & ST
        df_S = pd.DataFrame({
            "param": PARAM_NAMES,
            "S1": Si["S1"],
            "S1_conf": Si.get("S1_conf", [np.nan]*len(PARAM_NAMES)),
            "ST": Si["ST"],
            "ST_conf": Si.get("ST_conf", [np.nan]*len(PARAM_NAMES))
        })
        df_S.to_csv(os.path.join(OUTDIR, f"sobol_scalar_{col}.csv"), index=False)

        # Plot bar chart
        x = np.arange(len(PARAM_NAMES))
        width = 0.35
        plt.figure(figsize=(10,5))
        plt.bar(x - width/2, Si["S1"], width, label="S1")
        plt.bar(x + width/2, Si["ST"], width, label="ST")
        plt.xticks(x, PARAM_NAMES, rotation=45, ha="right")
        plt.ylabel("Sobol index")
        plt.title(f"Sobol indices (scalar) — {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"sobol_bar_{col}.png"))
        plt.close()

    # ----------------- Time-varying Sobol -----------------
    print("[INFO] Computing time-varying Sobol indices (avg across stations)...")
    time = fixed_args["time"]
    for comp in COMP_BASES:
        Y_time = ts_data[comp].mean(axis=2)  # average over stations -> Nsamp x Nt

        S1_t = np.zeros((len(time), len(PARAM_NAMES)))
        ST_t = np.zeros((len(time), len(PARAM_NAMES)))

        for t in range(len(time)):
            Si_t = sobol_analyze.analyze(problem, Y_time[:, t], calc_second_order=False, print_to_console=False)
            S1_t[t, :] = Si_t["S1"]
            ST_t[t, :] = Si_t["ST"]

        # Save CSV
        pd.DataFrame(S1_t, columns=PARAM_NAMES).assign(time=time).to_csv(
            os.path.join(OUTDIR, f"sobol_time_S1_{comp}.csv"), index=False
        )
        pd.DataFrame(ST_t, columns=PARAM_NAMES).assign(time=time).to_csv(
            os.path.join(OUTDIR, f"sobol_time_ST_{comp}.csv"), index=False
        )

        # Plot top 6 S1 params over time
        maxS1 = S1_t.max(axis=0)
        top_idx = np.argsort(maxS1)[-6:][::-1]

        plt.figure(figsize=(10,5))
        for idx in top_idx:
            plt.plot(time, S1_t[:, idx], label=PARAM_NAMES[idx])
        plt.xlabel("Time (days)")
        plt.ylabel("S1")
        plt.title(f"Time-varying S1 (top params) — {comp}")
        plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"sobol_time_S1_plot_{comp}.png"))
        plt.close()

    print("[INFO] Sobol sensitivity analysis complete. Results in", OUTDIR)
    return sobol_results, ts_data, combined

# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    sobol_results, ts_data, combined_table = run_sobol()
