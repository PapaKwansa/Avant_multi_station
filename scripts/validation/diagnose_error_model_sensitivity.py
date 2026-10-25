#!/usr/bin/env python3
"""Stage 5D: reuse audited Stage 5 Jacobians for error-covariance sensitivity.

Fast, NO new MCMC, NO new observations, NO hydraulic/mechanical recomputation.
Reweights local 2-parameter Fisher information under proposed observation
noise scales and within-channel exponential temporal covariance.

Run from AVANT repository root; see README_STAGE5D_ERROR_MODEL.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.coupled_dream import sha256_file
from avant_model.inversion.error_model_sensitivity import evaluate_error_case

PARAM_NAMES = ("log10_K", "log10_Ss")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identifiability-dir", required=True,
                   help="Stage 5 result with jacobians_training.npz and identifiability_summary.json")
    p.add_argument("--stage4a-dir", required=True,
                   help="Original Stage 4A output directory, for input hashes/timestamps")
    p.add_argument("--output-dir", required=True, help="New output directory")
    return p


def read_verified_inputs(stage5_dir, stage4a_dir):
    """Verify source hashes, training-only dimensions, and baseline Fisher."""
    old = json.loads((stage5_dir / "identifiability_summary.json").read_text(encoding="utf-8"))
    a = json.loads((stage4a_dir / "joint_summary.json").read_text(encoding="utf-8"))
    if not str(old.get("scope", "")).startswith("Stage 5 local identifiability"):
        raise ValueError("Expected original Stage 5 local-identifiability result")
    if not str(a.get("scope", "")).startswith("Stage 4A same-model synthetic joint"):
        raise ValueError("Expected original Stage 4A result")
    expected = old["sha256"]
    for key, report_key in (("hydraulic_config", "hydraulic_config_sha256"),
                            ("mechanical_config", "mechanical_config_sha256"),
                            ("gauges_csv", "gauge_csv_sha256")):
        if expected.get(key) != a.get(report_key):
            raise ValueError("Stage 5 and Stage 4A configuration hashes disagree: " + key)
    paths = {
        "stage4a_report": stage4a_dir / "joint_summary.json",
        "pressure_observations": stage4a_dir / "synthetic_pressure_observations.npz",
    }
    # Config paths are optional inputs to this quick analysis, but verified by
    # Stage 5 previously. Recheck actual Stage 4A data and Stage 1 observations.
    for key in ("stage4a_report", "pressure_observations"):
        if sha256_file(paths[key]) != expected[key]:
            raise ValueError("Stage 4A provenance mismatch: " + key)
    source_dir = Path(a["source_dir"])
    if sha256_file(source_dir / "synthetic_observations.npz") != a["strain_observations_sha256"]:
        raise ValueError("Original Stage 1 strain observations changed")
    with np.load(paths["pressure_observations"], allow_pickle=False) as p:
        times_p = np.asarray(p["times_h"], float)
        n_p = int(np.sum(p["train_mask"]))
    with np.load(source_dir / "synthetic_observations.npz", allow_pickle=False) as s:
        times_s = np.asarray(s["times_h"], float)
        n_s = int(np.sum(s["train_mask"]))
    if not np.array_equal(times_p, times_s):
        raise ValueError("Stage 1 and Stage 4A time vectors differ")
    if len(times_p) < 2 or not np.isfinite(times_p).all() or np.any(np.diff(times_p) <= 0):
        raise ValueError("Invalid original time vector")
    with np.load(stage5_dir / "jacobians_training.npz", allow_pickle=False) as f:
        js = np.asarray(f["sensitivity_strain"], float)
        jp = np.asarray(f["sensitivity_pressure"], float)
        jj = np.asarray(f["sensitivity_joint"], float)
        theta = np.asarray(f["theta_log10"], float)
    if n_s < 1 or n_p < 1:
        raise ValueError("Need training data from both modalities")
    if js.shape[0] != old.get("n_training_strain", js.shape[0]) or jp.shape[0] != old.get("n_training_pressure", jp.shape[0]):
        raise ValueError("Stage 5 report training counts disagree with saved Jacobians")
    if js.shape != (len(times_p) * n_s, 2) or jp.shape != (len(times_p) * n_p, 2):
        raise ValueError("Training Jacobian shapes disagree with immutable observations")
    if not np.array_equal(np.vstack((js, jp)), jj):
        raise ValueError("Joint training Jacobian must exactly stack its training blocks")
    if not np.array_equal(theta, np.asarray(old["reference_theta_log10"], float)):
        raise ValueError("Stage 5 Jacobian parameter reference differs from manifest")
    for key, jac in (("Strain only", js), ("Pressure only", jp), ("Joint", jj)):
        np.testing.assert_allclose(jac.T @ jac,
                                   np.asarray(old["information"][key]["fisher_information"]),
                                   atol=1e-7, rtol=1e-12)
    return old, js, jp, times_p, n_s, n_p


def case_rows(label, case, category):
    rows = []
    for j, name in enumerate(PARAM_NAMES):
        info = case["information"]
        rows.append({
            "scenario": label, "category": category, "parameter": name,
            "strain_sigma_factor": case["assumed_strain_sigma_multiplier"],
            "pressure_sigma_factor": case["assumed_pressure_sigma_multiplier"],
            "strain_tau_h": case["assumed_strain_time_correlation_hours"],
            "pressure_tau_h": case["assumed_pressure_time_correlation_hours"],
            "strain_only_sd_log10": (info["Strain only"]["local_sd_log10"] or [None, None])[j],
            "pressure_only_sd_log10": (info["Pressure only"]["local_sd_log10"] or [None, None])[j],
            "joint_sd_log10": (info["Joint"]["local_sd_log10"] or [None, None])[j],
            "joint_over_strain_sd_ratio": case["joint_over_strain_local_sd_ratio"][name],
            "joint_rank": info["Joint"]["numerical_rank"],
            "joint_min_singular": info["Joint"]["singular_values"][-1],
            "joint_condition_number": info["Joint"]["condition_number"],
        })
    return rows


def plot_noise_grid(rows, factors, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
    heatmaps = []
    for ax, name in zip(axes, PARAM_NAMES):
        mat = np.full((len(factors), len(factors)), np.nan)
        for r in rows:
            if r["category"] == "noise_scale" and r["parameter"] == name:
                i = factors.index(r["strain_sigma_factor"])
                j = factors.index(r["pressure_sigma_factor"])
                mat[i, j] = r["joint_over_strain_sd_ratio"]
        im = ax.imshow(mat, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(factors)), [f"{v:g}x" for v in factors])
        ax.set_yticks(range(len(factors)), [f"{v:g}x" for v in factors])
        ax.set(xlabel="Assumed pressure noise scale", ylabel="Assumed strain noise scale",
               title=f"Joint / strain-only local SD: {name}")
        for i in range(len(factors)):
            for j in range(len(factors)):
                value = mat[i, j]
                ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "n/a",
                        ha="center", va="center", fontsize=10,
                        color="white" if value < 0.57 else "black")
        heatmaps.append(im)
    fig.colorbar(heatmaps[0], ax=axes, label="Joint / strain-only SD (smaller = more gain)", shrink=.82)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"noise_scale_information_gain.{ext}", dpi=220)
    plt.close(fig)


def plot_correlation(rows, labels, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, param in zip(axes, PARAM_NAMES):
        filtered = [next(r for r in rows if r["scenario"] == label and r["parameter"] == param)
                    for label in labels]
        x = np.arange(len(filtered))
        width = .35
        ax.bar(x-width/2, [r["strain_only_sd_log10"] for r in filtered], width, label="Strain only")
        ax.bar(x+width/2, [r["joint_sd_log10"] for r in filtered], width, label="Joint")
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set(title=param, ylabel="Conditional local SD (log10 units)")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="upper left")
    fig.suptitle("Error covariance sensitivity (no refitting or new observations)")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"temporal_correlation_uncertainty.{ext}", dpi=220)
    plt.close(fig)


def main(argv=None):
    args = arguments().parse_args(argv)
    stage5_dir, stage4a_dir, output = (Path(args.identifiability_dir).resolve(),
                                       Path(args.stage4a_dir).resolve(), Path(args.output_dir).resolve())
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    old, js, jp, times, n_s, n_p = read_verified_inputs(stage5_dir, stage4a_dir)
    factors = [0.5, 1.0, 2.0]
    cases = {}
    rows = []
    for sf in factors:
        for pf in factors:
            label = f"strain_{sf:g}x_pressure_{pf:g}x"
            case = evaluate_error_case(js, jp, times, n_s, n_p,
                                       strain_factor=sf, pressure_factor=pf)
            cases[label] = case
            rows.extend(case_rows(label, case, "noise_scale"))
    labels = ["Independent", "Strain 6h", "Pressure 6h", "Both 6h", "Both 24h"]
    corr_args = [(0., 0.), (6., 0.), (0., 6.), (6., 6.), (24., 24.)]
    for label, (ts, tp) in zip(labels, corr_args):
        case = evaluate_error_case(js, jp, times, n_s, n_p,
                                   strain_tau_h=ts, pressure_tau_h=tp)
        cases[label] = case
        rows.extend(case_rows(label, case, "temporal_correlation"))
    base = cases["strain_1x_pressure_1x"]
    for key in ("Strain only", "Pressure only", "Joint"):
        np.testing.assert_allclose(base["information"][key]["fisher_information"],
                                   old["information"][key]["fisher_information"],
                                   atol=1e-7, rtol=1e-12)
        np.testing.assert_allclose(base["information"][key]["local_sd_log10"],
                                   old["information"][key]["local_sd_log10"],
                                   atol=1e-12, rtol=1e-12)
    output.mkdir(parents=True)
    (output / "error_model_sensitivity_summary.json").write_text(json.dumps({
        "scope": "Stage 5D local two-parameter error covariance sensitivity; SAME-MODEL synthetic, fixed geometry and mechanics",
        "interpretation": "Prospective local Fisher precision under alternative ASSUMED noise scales and time covariance; no data regenerated, no parameter refits, no DREAM, no coverage or field validation",
        "reference": old["reference"],
        "reference_theta_log10": old["reference_theta_log10"],
        "n_training_strain": int(js.shape[0]), "n_training_pressure": int(jp.shape[0]),
        "n_time_samples": int(len(times)), "n_strain_channels": n_s,
        "n_pressure_gauges": n_p,
        "temporal_covariance": "Within-channel exponential R(t,u)=exp(-|t-u|/tau_hours); independent channels and modalities; tau=0 identity",
        "baseline_sigma_strain_ns": float(json.loads((stage4a_dir / "joint_summary.json").read_text())["strain_sigma_ns"]),
        "baseline_sigma_pressure_pa": float(json.loads((stage4a_dir / "joint_summary.json").read_text())["pressure_sigma_pa"]),
        "prior_stage5_summary_sha256": sha256_file(stage5_dir / "identifiability_summary.json"),
        "prior_stage5_jacobians_sha256": sha256_file(stage5_dir / "jacobians_training.npz"),
        "stage4a_summary_sha256": sha256_file(stage4a_dir / "joint_summary.json"),
        "cases": cases,
        "limitations": [
            "First-order local sensitivity only; not posterior sampling or a likelihood refit.",
            "Observations and the reference best fit are held fixed in every case.",
            "Changing assumed covariance does not generate genuinely correlated data and cannot establish empirical coverage.",
            "No cross-station, cross-component, or pressure/strain cross-covariance is modeled.",
            "Error model includes only measurement noise scenarios, not systematic physical model discrepancy.",
            "Only homogeneous K/Ss vary; geometry, elasticity, boundaries, and source remain fixed.",
            "Results do not validate COMSOL or field performance or establish global uniqueness.",
        ],
    }, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "error_model_sensitivity_cases.csv", index=False)
    plot_noise_grid(rows, factors, output)
    plot_correlation(rows, labels, output)
    print("Saved Stage 5D assumed error-model sensitivity:", output, flush=True)
    print("Noise-scale scenarios:", len(factors)**2, "; temporal-correlation scenarios:", len(labels), flush=True)
    for label in labels:
        print(f"{label}: joint / strain-only SD ratios = {cases[label]['joint_over_strain_local_sd_ratio']}", flush=True)
    print("Not a DREAM posterior, empirical coverage study, or independent COMSOL validation.", flush=True)
    return output


if __name__ == "__main__":
    main()
