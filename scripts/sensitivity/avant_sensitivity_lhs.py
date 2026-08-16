"""
avant_sensitivity_lhs.py

LHS-based (screening) sensitivity analysis for the AVANT multi-station
strain forward model.

What this script does:
- Latin Hypercube Sampling of parameters
- Calls YOUR existing forward model via avant_sensitivity_config
- Computes scalar metrics (peak, RMS) for each strain component
- Pearson + Spearman correlations (tornado plots)
- Time-varying Pearson correlations (averaged across stations)
- Saves everything to ./lhs_outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc, pearsonr, spearmanr

# --------- IMPORT SHARED CONFIG + YOUR MODEL WRAPPER ----------
from avant_sensitivity_config import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    COMP_BASES,
    load_fixed_args,
    classify_stations,
    extract_component_matrix,
    peak_abs_max,
    rms_avg,
    run_forward_model_from_params,
)

# ==============================================================
# 1) SAMPLING (LHS)
# ==============================================================

def sample_lhs(n_samples):
    lb = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    ub = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])

    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES))
    u = sampler.random(n=n_samples)
    samples = qmc.scale(u, lb, ub)

    return pd.DataFrame(samples, columns=PARAM_NAMES)

# ==============================================================
# 2) SCALAR METRICS FOR ONE MODEL RUN
# ==============================================================

def summarize_strain_scalars(df):
    """
    Given model output DataFrame, compute scalar summaries
    for each strain component.

    Returns: dict {metric_name: value}
    """
    outputs = {}

    for comp in COMP_BASES:
        arr = extract_component_matrix(df, comp)  # Nt x Ns

        outputs[f"{comp}_peak_abs_max"] = peak_abs_max(arr)
        outputs[f"{comp}_rms_avg"] = rms_avg(arr)

    return outputs

# ==============================================================
# 3) RUN LHS EXPERIMENT
# ==============================================================

def run_lhs_experiment(n_samples=200, out_dir="lhs_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # Load fixed AVANT inputs
    fixed_args = load_fixed_args()

    print(f"[INFO] Sampling {n_samples} parameter sets using LHS...")
    param_df = sample_lhs(n_samples)

    results = []
    ts_all = []  # store time-series for later time-varying sensitivity

    print("[INFO] Evaluating forward model...")
    for i in range(n_samples):
        if (i + 1) % max(1, n_samples // 5) == 0:
            print(f"  evaluated {i+1}/{n_samples} samples")

        # Call YOUR forward model
        df = run_forward_model_from_params(param_df.iloc[i], fixed_args)

        # Scalar summaries
        scalars = summarize_strain_scalars(df)
        results.append(scalars)

        # Store time-series per component
        ts_list = [extract_component_matrix(df, comp) for comp in COMP_BASES]
        ts_all.append(ts_list)

    # Combine parameters + outputs
    scalar_df = pd.DataFrame(results)
    combined = pd.concat([param_df.reset_index(drop=True),
                           scalar_df.reset_index(drop=True)], axis=1)

    combined_path = os.path.join(out_dir, "lhs_samples_and_outputs.csv")
    combined.to_csv(combined_path, index=False)
    print(f"[INFO] Saved combined table to {combined_path}")

    # ==============================================================
    # 4) CORRELATION ANALYSIS + TORNADO PLOTS
    # ==============================================================

    pearson_corr = {}
    spearman_corr = {}

    print("[INFO] Computing correlations and tornado plots...")

    for out_col in scalar_df.columns:
        pearson_corr[out_col] = {}
        spearman_corr[out_col] = {}

        for p in PARAM_NAMES:
            pearson_corr[out_col][p] = pearsonr(combined[p], combined[out_col])[0]
            spearman_corr[out_col][p] = spearmanr(combined[p], combined[out_col])[0]

        # Tornado plot (Pearson)
        vals = pearson_corr[out_col]
        ranked = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)

        plt.figure(figsize=(8, 5))
        plt.bar([x[0] for x in ranked], [x[1] for x in ranked])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Pearson correlation")
        plt.title(f"Tornado plot — {out_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tornado_{out_col}.png"))
        plt.close()

    # Save correlations
    pd.DataFrame(pearson_corr).to_csv(os.path.join(out_dir, "pearson_correlations.csv"))
    pd.DataFrame(spearman_corr).to_csv(os.path.join(out_dir, "spearman_correlations.csv"))

    # ==============================================================
    # 5) TIME-VARYING SENSITIVITY (PEARSON) — AVERAGED ACROSS STATIONS
    # ==============================================================

    print("[INFO] Computing time-varying sensitivity...")

    time = fixed_args["time"]
    n_times = len(time)

    for comp_idx, comp in enumerate(COMP_BASES):
        # Stack: (n_samples, n_times, n_stations)
        ts_stack = np.array([ts[comp_idx] for ts in ts_all])

        # Average across stations
        ts_mean = ts_stack.mean(axis=2)  # (n_samples, n_times)

        plt.figure(figsize=(12, 6))

        for i, pname in enumerate(PARAM_NAMES):
            corr_t = np.zeros(n_times)

            for t in range(n_times):
                try:
                    corr_t[t] = pearsonr(param_df[pname], ts_mean[:, t])[0]
                except Exception:
                    corr_t[t] = 0.0

            plt.plot(time, corr_t, label=pname)

        plt.xlabel("Time (days)")
        plt.ylabel("Pearson correlation")
        plt.title(f"Time-series sensitivity — {comp} (avg across stations)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"time_series_sensitivity_{comp}.png"))
        plt.close()

    print("[INFO] LHS sensitivity analysis complete.")
    return combined, pearson_corr, spearman_corr

# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    combined, pearson_corr, spearman_corr = run_lhs_experiment(n_samples=200)

    # Print top parameters for first output as a quick check
    first_out = list(pearson_corr.keys())[0]
    ranked = sorted(pearson_corr[first_out].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True)

    print("\n[RESULT] Top parameters by |Pearson| for", first_out)
    for k, v in ranked[:10]:
        print(f"{k}: {v:.3f}")
