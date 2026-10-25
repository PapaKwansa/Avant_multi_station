#!/usr/bin/env python3
"""Stage 5: local identifiability of Stage 4A saved synthetic pressure/strain.

Reuses Stage 4A hash-validated observations and exact fixed-geometry forward
model. No observations are redrawn and no DREAM sampling is performed.

Use from the AVANT repository root with the (vrugt) environment:
  python scripts/validation/diagnose_joint_identifiability.py --help

The output is a local noise-weighted sensitivity analysis of ONLY log10(K) and
log10(Ss); it is not a ten-parameter mechanical/spatial or field inversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.joint_dream import (
    build_joint_likelihood, joint_mle_record, load_stage4a, validate_joint_mle,
)
from avant_model.inversion.coupled_dream import sha256_file
from avant_model.inversion.local_identifiability import (
    analyze_jacobian, central_difference_jacobian, compare_jacobian_steps,
)


NAMES = ("log10 K (m/s)", "log10 Ss (1/m)")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage4a-dir", required=True)
    p.add_argument("--hydraulic-config", required=True)
    p.add_argument("--mechanical-config", required=True)
    p.add_argument("--gauges-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--step-log10", type=float, default=0.002,
                   help="Centered finite difference step for each log10 parameter")
    p.add_argument("--reference", choices=("joint-mle", "synthetic-truth"), default="joint-mle")
    return p


def plot_spectrum(infos, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    labels = ("Strain only", "Pressure only", "Joint")
    for label in labels:
        values = np.asarray(infos[label]["singular_values"], float)
        ax.plot(np.arange(1, len(values)+1), values, "o-", label=label)
    ax.set(yscale="log", xlabel="Singular-value index", ylabel="Singular value of whitened Jacobian",
           title="Local information from pressure and strain")
    ax.set_xticks([1, 2]); ax.grid(alpha=0.25); ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(output/f"local_identifiability_singular_values.{ext}", dpi=220)
    plt.close(fig)


def plot_uncertainty(infos, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    width = 0.23
    xx = np.arange(2)
    for i, label in enumerate(("Strain only", "Pressure only", "Joint")):
        vals = infos[label]["local_sd_log10"]
        if vals is None:
            continue
        ax.bar(xx + (i-1)*width, vals, width, label=label)
    ax.set_xticks(xx, [r"$\log_{10} K$", r"$\log_{10} S_s$"])
    ax.set(ylabel="Local 1-sigma uncertainty (log10 units)",
           title="Linearized uncertainty; fixed geometry and error model")
    ax.grid(alpha=0.2, axis="y"); ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(output/f"local_identifiability_uncertainty.{ext}", dpi=220)
    plt.close(fig)


def main(argv=None):
    a = arguments().parse_args(argv)
    if not np.isfinite(a.step_log10) or not 0 < a.step_log10 <= 0.1:
        raise ValueError("--step-log10 must be positive and <=0.1")
    stage4a = Path(a.stage4a_dir).resolve()
    hc = Path(a.hydraulic_config).resolve()
    mc = Path(a.mechanical_config).resolve()
    gc = Path(a.gauges_csv).resolve()
    output = Path(a.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output folder: {output}")
    report, stage1, strain, pressure = load_stage4a(stage4a, hc, mc, gc)
    bounds = HydraulicBounds()
    print("Rebuilding Stage 4 fixed mechanical kernel...", flush=True)
    likelihood, _ = build_joint_likelihood(hc, mc, stage1, strain, pressure, bounds)
    joint_mle, joint_ll = validate_joint_mle(likelihood, report, bounds)
    theta = (joint_mle if a.reference == "joint-mle"
             else np.asarray(stage1["truth_log10_K_log10_Ss"], float))
    steps = np.full(2, a.step_log10)
    lower = np.asarray((bounds.log10_K[0], bounds.log10_Ss[0]), float)
    upper = np.asarray((bounds.log10_K[1], bounds.log10_Ss[1]), float)
    if np.any(theta - steps <= lower) or np.any(theta + steps >= upper):
        raise ValueError("finite-difference offsets would cross prior boundaries")
    st, pt = strain.training_mask, pressure.training_mask

    def predict_whitened_both(state):
        s, p = likelihood.predict(state)
        # Stable order: first time-major training strain, then time-major training pressure.
        return np.concatenate(((s[:, st] / strain.sigma_ns).ravel(),
                               (p[:, pt] / pressure.sigma_pa).ravel()))

    jac = central_difference_jacobian(predict_whitened_both, theta, steps)
    jac_half = central_difference_jacobian(predict_whitened_both, theta, steps/2)
    n_strain = int(len(strain.times_h) * st.sum())
    n_pressure = int(len(pressure.times_h) * pt.sum())
    if jac.shape != (n_strain + n_pressure, 2):
        raise RuntimeError(f"Unexpected Jacobian shape: {jac.shape}")
    j_strain, j_pressure = jac[:n_strain], jac[n_strain:]
    infos = {
        "Strain only": analyze_jacobian(j_strain),
        "Pressure only": analyze_jacobian(j_pressure),
        "Joint": analyze_jacobian(jac),
    }
    # Exact identity, useful as a regression/provenance check.
    fs = np.asarray(infos["Strain only"]["fisher_information"])
    fp = np.asarray(infos["Pressure only"]["fisher_information"])
    fj = np.asarray(infos["Joint"]["fisher_information"])
    if not np.allclose(fj, fs + fp, rtol=1e-12, atol=1e-6):
        raise RuntimeError("Joint information must equal sum of training data blocks")
    ratio = {}
    for i, name in enumerate(("log10_K", "log10_Ss")):
        sd_s = infos["Strain only"]["local_sd_log10"]
        sd_j = infos["Joint"]["local_sd_log10"]
        ratio[name] = (float(sd_j[i]/sd_s[i]) if sd_s is not None and sd_j is not None else None)
    result = {
        "scope": "Stage 5 local identifiability for SAME-MODEL synthetic 2-parameter hydraulic inverse; not independent COMSOL/field or full mechanical/spatial validation",
        "reference": a.reference,
        "reference_theta_log10": theta.tolist(),
        "joint_mle_theta_log10": joint_mle.tolist(),
        "stage4a_joint_mle_log_likelihood": float(joint_ll),
        "step_log10": float(a.step_log10),
        "jacobian_relative_change_half_step": compare_jacobian_steps(jac, jac_half),
        "n_training_strain": n_strain,
        "n_training_pressure": n_pressure,
        "parameter_names": ["log10_K_m_s", "log10_Ss_per_m"],
        "information": infos,
        "joint_over_strain_local_sd_ratio": ratio,
        "sha256": {
            "stage4a_report": sha256_file(stage4a / "joint_summary.json"),
            "pressure_observations": sha256_file(stage4a / "synthetic_pressure_observations.npz"),
            "hydraulic_config": sha256_file(hc),
            "mechanical_config": sha256_file(mc),
            "gauges_csv": sha256_file(gc),
        },
        "limitations": [
            "Rank and Fisher information are LOCAL and depend on the reference parameter vector.",
            "Independent Gaussian measurement errors with fixed sigma; no temporal or cross-channel covariance.",
            "Geometry and elastic properties are FIXED; no mechanical or spatial identifiability is tested.",
            "Same-model synthetic observations; results cannot independently validate forward physics.",
            "Linearized covariance is NOT a DREAM posterior and may disagree for nonlinear/multimodal models.",
            "Pressure sensors use full FV fields; strain uses fixed compressed cuboid kernel.",
            "A full-rank local Jacobian does not rule out globally nonunique parameter sets.",
        ],
    }
    output.mkdir(parents=True)
    (output/"identifiability_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output/"jacobians_training.npz", theta_log10=theta,
                        sensitivity_strain=j_strain, sensitivity_pressure=j_pressure,
                        sensitivity_joint=jac, step_log10=steps)
    plot_spectrum(infos, output)
    plot_uncertainty(infos, output)
    print("Saved identifiability diagnostics to", output, flush=True)
    print("Local rank (strain / pressure / joint):",
          *(infos[k]["numerical_rank"] for k in ("Strain only", "Pressure only", "Joint")), flush=True)
    print("Half-step sensitivity difference:", result["jacobian_relative_change_half_step"], flush=True)
    print("Joint / strain local standard-deviation ratios:", ratio, flush=True)
    print("Important: local two-parameter result only; not proof of global uniqueness.", flush=True)
    return result


if __name__ == "__main__":
    main()
