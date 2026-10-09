#!/usr/bin/env python3
"""Non-destructive regression checks and a SAME-MODEL synthetic pressure test.

Run from the repository root. No field observations are inverted by this script.
No import of multi_stations_input: geometry is always explicit in the JSON.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from avant_model.model import multi_station_strain as mechanics
from avant_model.model import multi_station_rotation as rotation
from avant_model.model import multi_station_coord_transform as coord
from avant_model.model import forward_model_multi_station as legacy
from avant_model.model import pressure_superposition as superposition
from avant_model.inversion import linear_pressure_inversion as inference
from avant_model.model.pressure_superposition import (
    Cuboid, ElasticProperties, build_pressure_kernel, coarse_block_basis)
from avant_model.inversion.linear_pressure_inversion import linear_gaussian_posterior


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "description" not in cfg:
        raise ValueError("Configuration must describe its provenance/purpose.")
    parent = Cuboid(**cfg["geometry"])
    elastic = ElasticProperties(**cfg["elastic"])
    if "station_file" in cfg and "stations" in cfg:
        raise ValueError("Specify stations OR station_file, not both.")
    if "station_file" in cfg:
        station_path = Path(cfg["station_file"])
        if not station_path.is_absolute():
            station_path = ROOT / station_path
        station_path = station_path.resolve()
        stations = pd.read_csv(station_path)
        station_source = {"path": str(station_path), "sha256": file_hash(station_path)}
    else:
        stations = pd.DataFrame(cfg["stations"])
        station_source = {"path": "inline in JSON", "sha256": file_hash(path)}
    required = ["station", "x", "y", "z"]
    if not set(required) <= set(stations.columns):
        raise ValueError("Station metadata must contain station,x,y,z.")
    if stations[required].isna().any().any():
        raise ValueError("Missing station metadata values.")
    stations = stations.loc[:, required].copy()
    stations["station"] = stations["station"].astype(str).str.strip()
    for key in ("x", "y", "z"):
        stations[key] = pd.to_numeric(stations[key], errors="raise")
    return cfg, parent, elastic, stations, station_source


def station_kwargs(stations, cfg):
    return dict(x_prime=stations.x.to_numpy(), y_prime=stations.y.to_numpy(),
                z=stations.z.to_numpy(), station_names=stations.station.tolist(),
                components=tuple(cfg.get("components", superposition.DEFAULT_COMPONENTS)))


def verify(cfg, parent, elastic, stations, out):
    # The old forward wrapper supports these four channels. Additional channels
    # can be used in the kernel, but need a separate independent reference.
    kwargs = station_kwargs(stations, cfg)
    if not set(kwargs["components"]) <= set(superposition.DEFAULT_COMPONENTS):
        raise ValueError("Legacy-wrapper regression currently supports eXX,eYY,eXY,eZZ.")
    times = np.array([0.0, 0.25, 1.0, 1.5])
    pmax = float(cfg["verification_pressure_pa"])
    pressure = legacy.pressure_time_series(pmax, 1.0, 0.4, times)
    reference = legacy.forward_model_multi_station(
        pmax=pmax, tpeak=1.0, d=0.4, time=times,
        x_prime=kwargs["x_prime"], y_prime=kwargs["y_prime"], z=kwargs["z"],
        station_names=kwargs["station_names"], **asdict(parent), **asdict(elastic))
    rows = []
    for splits in cfg["verification_splits"]:
        start = time.perf_counter()
        kernel = build_pressure_kernel(parent, elastic, splits=splits, **kwargs)
        predicted = kernel.predict(np.repeat(pressure[:, None], len(kernel.tiles), axis=1))
        expected = reference.loc[:, list(kernel.channels)].to_numpy()
        error = float(np.max(np.abs(predicted - expected)))
        scale = float(np.max(np.abs(expected)))
        ok = np.allclose(predicted, expected, rtol=cfg["verification_rtol"],
                         atol=cfg["verification_atol_ns"])
        rows.append({"splits": list(splits), "passed": bool(ok),
                     "max_abs_error_ns": error,
                     "max_abs_reference_ns": scale,
                     "relative_max_error": error / scale if scale else None,
                     "elapsed_seconds": time.perf_counter() - start})
        print(f"  splits={tuple(splits)} passed={bool(ok)} max_error={error:.4e} ns")
    result = {"scope": "Regression against supplied single-cuboid forward wrapper; not independent physics validation",
              "rtol": cfg["verification_rtol"], "atol_ns": cfg["verification_atol_ns"],
              "all_passed": all(row["passed"] for row in rows), "cases": rows}
    write_json(out / "verification_summary.json", result)
    if not result["all_passed"]:
        raise RuntimeError("Partition regression failed. Stop before pressure inversion; inspect the JSON.")
    return result


def synthetic(cfg, parent, elastic, stations, out, plots):
    kwargs = station_kwargs(stations, cfg)
    kernel = build_pressure_kernel(parent, elastic, splits=cfg["fine_splits"], **kwargs)
    B = coarse_block_basis(cfg["fine_splits"], cfg["coarse_splits"])
    H = kernel.G @ B
    nc = B.shape[1]
    truth = np.asarray(cfg["true_coefficients_pa"], dtype=float)
    if truth.shape != (nc,) or not np.isfinite(truth).all():
        raise ValueError("true_coefficients_pa must match the coarse grid and be finite.")
    prior_mean = np.broadcast_to(np.asarray(cfg["prior_mean_pa"], dtype=float), (nc,)).copy()
    prior_sd = np.broadcast_to(np.asarray(cfg["prior_sd_pa"], dtype=float), (nc,)).copy()
    if not np.isfinite(prior_sd).all() or np.any(prior_sd <= 0):
        raise ValueError("prior_sd_pa must be positive and finite.")
    noise = float(cfg["noise_std_ns"])
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise_std_ns must be positive and finite.")
    rng = np.random.default_rng(cfg["seed"])
    clean = H @ truth
    observed = clean + rng.normal(0.0, noise, clean.size)
    holdout = set(cfg.get("holdout_stations", []))
    if not holdout <= set(kernel.station_names):
        raise ValueError("Unknown holdout station(s).")
    channel_stations = [name for _ in kwargs["components"] for name in kernel.station_names]
    training = np.array([name not in holdout for name in channel_stations])
    if not training.any():
        raise ValueError("At least one training station is required.")
    posterior = linear_gaussian_posterior(
        H[training], observed[training], prior_mean_pa=prior_mean,
        prior_covariance_pa2=np.diag(prior_sd ** 2), noise_std_ns=noise)
    predicted = H @ posterior.mean_pa
    signal_var = np.einsum("ij,jk,ik->i", H, posterior.covariance_pa2, H)
    signal_sd = np.sqrt(np.maximum(signal_var, 0))
    total_sd = np.sqrt(signal_var + noise ** 2)
    interval_multiplier = 1.959963984540054
    summary_table = pd.DataFrame({
        "coefficient": np.arange(nc), "true_pa": truth,
        "prior_mean_pa": prior_mean, "prior_sd_pa": prior_sd,
        "posterior_mean_pa": posterior.mean_pa, "posterior_sd_pa": posterior.std_pa,
        "lower95_pa": posterior.mean_pa - interval_multiplier * posterior.std_pa,
        "upper95_pa": posterior.mean_pa + interval_multiplier * posterior.std_pa})
    summary_table.to_csv(out / "synthetic_pressure_posterior.csv", index=False)
    predictions = pd.DataFrame({
        "channel": kernel.channels, "station": channel_stations,
        "subset": np.where(training, "training", "holdout"),
        "true_clean_ns": clean, "synthetic_observed_ns": observed,
        "posterior_mean_ns": predicted, "parameter_sd_ns": signal_sd,
        "predictive_sd_ns": total_sd,
        "parameter_lower95_ns": predicted - interval_multiplier * signal_sd,
        "parameter_upper95_ns": predicted + interval_multiplier * signal_sd,
        "predictive_lower95_ns": predicted - interval_multiplier * total_sd,
        "predictive_upper95_ns": predicted + interval_multiplier * total_sd})
    predictions.to_csv(out / "synthetic_channel_predictions.csv", index=False)
    pd.DataFrame([asdict(t) for t in kernel.tiles]).to_csv(out / "tile_geometry.csv", index=False)
    fine_mean = B @ posterior.mean_pa
    fine_sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", B, posterior.covariance_pa2, B), 0))
    np.savez_compressed(out / "pressure_kernel_and_posterior.npz", G_ns_per_pa=kernel.G,
                        basis=B, channels=np.asarray(kernel.channels), true_coefficients_pa=truth,
                        posterior_mean_pa=posterior.mean_pa, posterior_covariance_pa2=posterior.covariance_pa2,
                        fine_mean_pa=fine_mean, fine_sd_pa=fine_sd,
                        prior_whitened_singular_values=posterior.prior_whitened_singular_values,
                        training_channel_mask=training)
    checks = ((truth >= summary_table.lower95_pa) & (truth <= summary_table.upper95_pa))
    metrics = {
        "scope": "Same-model synthetic test, fixed geometry/mechanics, Gaussian prior and known IID noise; not real-field or permeability inference",
        "n_stations": len(kernel.station_names), "n_channels": len(kernel.channels),
        "n_training_channels": int(training.sum()), "n_holdout_channels": int((~training).sum()),
        "fine_splits": list(cfg["fine_splits"]), "n_mechanical_tiles": len(kernel.tiles),
        "coarse_splits": list(cfg["coarse_splits"]), "n_unknown_coefficients": nc,
        "numerical_rank": posterior.numerical_rank, "likelihood_nullity": posterior.likelihood_nullity,
        "prior_whitened_singular_values": posterior.prior_whitened_singular_values.tolist(),
        "modes_with_prior_whitened_singular_value_gt_one": posterior.informative_modes,
        "pressure_rmse_pa": float(np.sqrt(np.mean((truth - posterior.mean_pa) ** 2))),
        "truth_inside_marginal95_count": int(checks.sum()),
        "coverage_note": "One realization and correlated coefficients do not establish coverage/calibration.",
        "training_rmse_ns": float(np.sqrt(np.mean((observed[training] - predicted[training]) ** 2))),
        "holdout_rmse_ns": (float(np.sqrt(np.mean((observed[~training] - predicted[~training]) ** 2)))
                            if (~training).any() else None),
        "training_noise_std_ns": noise,
        "posterior_std_pa": posterior.std_pa.tolist(),
        "prior_and_noise_are_illustrative": True
    }
    write_json(out / "synthetic_summary.json", metrics)
    print(f"  Synthetic pressure coefficients: {nc}; mechanical tiles: {len(kernel.tiles)}")
    print(f"  Numerical rank: {posterior.numerical_rank}/{nc}; pressure RMSE: {metrics['pressure_rmse_pa']:.3g} Pa")
    if plots:
        make_plots(out, parent, cfg, summary_table, predictions, posterior, fine_mean, fine_sd)
    return metrics


def make_plots(out, parent, cfg, table, predictions, posterior, fine_mean, fine_sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.errorbar(table.coefficient, table.posterior_mean_pa / 1e6,
                yerr=1.959963984540054 * table.posterior_sd_pa / 1e6,
                fmt="o", capsize=4, label="Posterior mean and marginal 95% interval")
    ax.plot(table.coefficient, table.true_pa / 1e6, "x", label="Synthetic truth")
    ax.set(xlabel="Coarse pressure coefficient (C-order)", ylabel="Pressure change (MPa)",
           title="Synthetic pressure recovery - conditional uncertainty")
    ax.set_xticks(table.coefficient)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / "synthetic_pressure_coefficients.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, rows in predictions.groupby("subset", sort=False):
        ax.scatter(rows.synthetic_observed_ns, rows.posterior_mean_ns, label=name)
    lo = min(predictions.synthetic_observed_ns.min(), predictions.posterior_mean_ns.min())
    hi = max(predictions.synthetic_observed_ns.max(), predictions.posterior_mean_ns.max())
    ax.plot([lo, hi], [lo, hi], "--", label="1:1")
    ax.set(xlabel="Synthetic observations (nanostrain)", ylabel="Predicted mean (nanostrain)",
           title="Training and withheld channels")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "synthetic_obs_pred.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    s = posterior.prior_whitened_singular_values
    ax.semilogy(np.arange(1, len(s)+1), np.maximum(s, np.finfo(float).tiny), "o-")
    ax.axhline(1.0, linestyle="--", label="Equal prior and likelihood precision in a mode")
    ax.set(xlabel="Mode", ylabel="Singular value", title="Noise- and prior-scaled sensitivity")
    ax.set_xticks(np.arange(1, len(s)+1)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "synthetic_singular_values.png", dpi=180); plt.close(fig)
    nx, ny, nz = cfg["fine_splits"]
    if nz == 1:
        for values, name, title in [(fine_mean, "mean", "Posterior mean pressure change"),
                                    (fine_sd, "sd", "Conditional posterior pressure standard deviation")]:
            fig, ax = plt.subplots(figsize=(6.0, 5.2))
            image = ax.pcolormesh(np.linspace(-parent.a, parent.a, nx+1),
                                 np.linspace(-parent.b, parent.b, ny+1),
                                 values.reshape(nx, ny).T / 1e6, shading="flat")
            ax.set(xlabel="Local source x (m)", ylabel="Local source y (m)",
                   title=title + "\nSynthetic test; block basis, not field-resolution map")
            ax.set_aspect("equal"); fig.colorbar(image, ax=ax, label="MPa")
            fig.tight_layout(); fig.savefig(out / f"synthetic_pressure_{name}.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verification-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg, parent, elastic, stations, station_source = load_config(config_path)
    out = args.output_dir.resolve()
    # Never overwrite an existing directory, even if it contains an earlier failure.
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration_used.json").write_bytes(config_path.read_bytes())
    stations.to_csv(out / "station_metadata_used.csv", index=False)
    paths = {"mechanics": mechanics.__file__, "rotation": rotation.__file__,
             "coordinate_transform": coord.__file__, "legacy_forward": legacy.__file__,
             "superposition": superposition.__file__, "linear_inference": inference.__file__,
             "runner": __file__}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head, status = None, "No accessible Git checkout"
    write_json(out / "provenance.json", {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "git_head": head,
        "git_status_porcelain": status, "station_source": station_source,
        "configuration_sha256": file_hash(config_path),
        "modules": {k: {"path": str(Path(v).resolve()), "sha256": file_hash(v)} for k, v in paths.items()}})
    print(f"Analytical kernel loaded from: {mechanics.__file__}")
    print(f"Purpose: {cfg['description']}")
    try:
        verify(cfg, parent, elastic, stations, out)
        if not args.verification_only:
            synthetic(cfg, parent, elastic, stations, out, plots=not args.no_plots)
    except Exception as exc:
        write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    print(f"Checks complete: {out}")
    print("These are regression/synthetic results, not independent field validation.")


if __name__ == "__main__":
    main()
