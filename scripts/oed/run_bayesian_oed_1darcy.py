#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized production Bayesian OED runner for the AVANT 1-Darcy analytical model.

Key production correction
-------------------------
For existing S01--S08 station-subset OED, this runner computes ONE full
8-station Jacobian per posterior geometry scenario, then slices that Jacobian
for every candidate station subset. It does NOT recompute a Jacobian for each
subset.

For N posterior scenarios and 8 existing stations:

    old approach: N * 255 Jacobians
    this approach: N Jacobians + inexpensive subset FIM calculations

Physical OED state
------------------
    [a, b, h, theta_deg, x0_prime, y0_prime]

Statistical nuisance parameter
------------------------------
    log10_sigma_strain

The nuisance parameter is not included in the physical Fisher state. It is
used only to define the predictive/observation-noise scale.

Modes
-----
validate
    Validate the Bayesian run contract and convergence status.

posterior_existing
    Posterior-informed robust OED for all existing S01--S08 subsets.

posterior_placement
    Posterior-informed robust OED for candidate spatial networks.

eig
    Monte Carlo expected-information-gain evaluation.

all
    Run all posterior-informed analyses.

A non-converged posterior is rejected by default. Use --allow-nonconverged
only for software/integration testing.

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avant_model.model.forward_model_multi_station import forward_model_multi_station

from avant_model.oed.bayesian_design import (
    bayesian_design_summary,
    estimate_expected_information_gain,
    posterior_sigma_samples,
    rank_expected_information_gain,
)
from avant_model.oed.fisher_information import (
    fisher_diagnostics,
    fisher_information_matrix,
    make_observation_weights,
)
from avant_model.oed.geometry_uncertainty import (
    GEOMETRY_PARAMETER_NAMES,
    posterior_geometry_scenarios,
)
from avant_model.oed.sensor_placement import (
    SensorLocation,
    SensorNetwork,
    generate_rectangular_grid,
    random_candidate_networks,
)


COMPONENTS = ("eXX", "eYY", "eZZ", "eXY")
PHYSICAL_PARAMETERS = tuple(GEOMETRY_PARAMETER_NAMES)
NUISANCE_PARAMETER = "log10_sigma_strain"


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optimized posterior-informed Bayesian OED."
    )

    parser.add_argument("--run-dir", type=Path, required=True)

    parser.add_argument(
        "--mode",
        choices=[
            "validate",
            "posterior_existing",
            "posterior_placement",
            "eig",
            "all",
        ],
        default="all",
    )

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--posterior-samples", type=int, default=500)
    parser.add_argument("--burn-in-fraction", type=float, default=0.20)

    parser.add_argument(
        "--sigma-statistic",
        choices=["median", "mean", "q05", "q95"],
        default="median",
    )

    parser.add_argument(
        "--conservative-quantile",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--fd-step",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-10,
    )

    parser.add_argument(
        "--existing-subset-sizes",
        type=str,
        default="1,2,3,4,5,6,7,8",
    )

    parser.add_argument("--placement-grid-nx", type=int, default=9)
    parser.add_argument("--placement-grid-ny", type=int, default=9)
    parser.add_argument("--placement-xmin", type=float, default=-900.0)
    parser.add_argument("--placement-xmax", type=float, default=900.0)
    parser.add_argument("--placement-ymin", type=float, default=-900.0)
    parser.add_argument("--placement-ymax", type=float, default=900.0)
    parser.add_argument("--placement-size", type=int, default=4)
    parser.add_argument("--placement-networks", type=int, default=100)
    parser.add_argument("--placement-min-spacing", type=float, default=100.0)

    parser.add_argument("--eig-designs", type=int, default=10)
    parser.add_argument("--eig-outer", type=int, default=100)
    parser.add_argument("--eig-inner", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="Integration testing only.",
    )

    return parser.parse_args()


# ============================================================================
# Generic utilities
# ============================================================================

def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: {0}".format(type(value).__name__))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.20,
        }
    )


def save_figure(fig, output_base: Path, handles=None, labels=None, ncol=2):
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if handles and labels:
        fig.subplots_adjust(bottom=0.20)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            frameon=False,
        )

    fig.savefig(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_subset_sizes(text: str, station_count: int) -> List[int]:
    sizes = sorted(
        set(
            int(item.strip())
            for item in text.split(",")
            if item.strip()
        )
    )

    if not sizes:
        raise ValueError("At least one subset size is required.")

    if any(size < 1 or size > station_count for size in sizes):
        raise ValueError(
            "Subset sizes must lie between 1 and {0}.".format(station_count)
        )

    return sizes


# ============================================================================
# Exact Bayesian run contract
# ============================================================================

def load_run_contract(run_dir: Path):
    run_dir = run_dir.resolve()

    required = (
        "run_manifest.json",
        "posterior_run_summary.json",
        "posterior_samples_flat.npy",
        "posterior_chains.npy",
        "posterior_logps.npy",
        "station_metadata_used.csv",
    )

    missing = [
        name
        for name in required
        if not (run_dir / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Bayesian run is missing:\n"
            + "\n".join("    {0}".format(name) for name in missing)
        )

    with open(run_dir / "run_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    with open(
        run_dir / "posterior_run_summary.json",
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)

    flat = np.load(
        run_dir / "posterior_samples_flat.npy",
        allow_pickle=False,
    )

    chains = np.load(
        run_dir / "posterior_chains.npy",
        allow_pickle=False,
    )

    logps = np.load(
        run_dir / "posterior_logps.npy",
        allow_pickle=False,
    ).reshape(-1)

    stations = pd.read_csv(
        run_dir / "station_metadata_used.csv"
    )

    return run_dir, manifest, summary, flat, chains, logps, stations


def validate_run_contract(
    manifest,
    summary,
    flat,
    chains,
    logps,
    stations,
) -> Dict[str, object]:

    expected_parameters = (
        *PHYSICAL_PARAMETERS,
        NUISANCE_PARAMETER,
    )

    parameter_names = tuple(
        manifest.get("parameter_names", [])
    )

    physical_names = tuple(
        manifest.get("physical_parameter_names", [])
    )

    nuisance_names = tuple(
        manifest.get("nuisance_parameter_names", [])
    )

    if parameter_names != expected_parameters:
        raise RuntimeError(
            "Unexpected parameter contract: {0}".format(parameter_names)
        )

    if physical_names != PHYSICAL_PARAMETERS:
        raise RuntimeError(
            "Unexpected physical parameter contract: {0}".format(
                physical_names
            )
        )

    if nuisance_names != (NUISANCE_PARAMETER,):
        raise RuntimeError(
            "Unexpected nuisance parameter contract: {0}".format(
                nuisance_names
            )
        )

    if flat.ndim != 2:
        raise RuntimeError("posterior_samples_flat.npy must be 2-D.")

    if chains.ndim != 3:
        raise RuntimeError("posterior_chains.npy must be 3-D.")

    if flat.shape[1] != len(parameter_names):
        raise RuntimeError("Flat posterior parameter dimension mismatch.")

    if chains.shape[2] != len(parameter_names):
        raise RuntimeError("Chain posterior parameter dimension mismatch.")

    if logps.size != flat.shape[0]:
        raise RuntimeError("posterior_logps size mismatch.")

    required_station_columns = {"station", "x", "y", "z"}
    missing_station_columns = required_station_columns - set(stations.columns)

    if missing_station_columns:
        raise RuntimeError(
            "Station metadata missing: "
            + ", ".join(sorted(missing_station_columns))
        )

    convergence = summary.get("convergence", {})
    converged = bool(convergence.get("converged", False))

    return {
        "parameter_names": parameter_names,
        "converged": converged,
        "status": str(summary.get("run_status", "unknown")),
        "final_rhat": convergence.get("final_rhat"),
        "n_flat_samples": int(flat.shape[0]),
        "n_chains": int(chains.shape[0]),
        "iterations_per_chain": int(chains.shape[1]),
        "n_stations": int(len(stations)),
    }


# ============================================================================
# Posterior preparation
# ============================================================================

def retained_flat_from_chains(
    chains: np.ndarray,
    burn_in_fraction: float,
) -> np.ndarray:
    if not 0.0 <= burn_in_fraction < 1.0:
        raise ValueError("--burn-in-fraction must satisfy 0 <= f < 1.")

    burn = int(
        math_floor(
            chains.shape[1] * burn_in_fraction
        )
    )

    retained = chains[:, burn:, :]

    if retained.shape[1] < 1:
        raise RuntimeError("No chain samples remain after burn-in.")

    return retained.reshape(
        -1,
        retained.shape[2],
    )


def math_floor(value: float) -> int:
    return int(np.floor(value))


def posterior_scenario_table(
    retained_samples: np.ndarray,
    parameter_names: Sequence[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("--posterior-samples must be positive.")

    return posterior_geometry_scenarios(
        posterior_samples=retained_samples,
        parameter_names=parameter_names,
        n_samples=n_samples,
        seed=seed,
        source="bayesian_posterior",
    )


def posterior_parameter_scales(
    scenarios: pd.DataFrame,
    prior_bounds: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """
    Dimensionless Fisher scaling.

    For p_scaled = p / scale:
        d y / d p_scaled = J * scale

    Posterior standard deviation is used when it is informative. A small
    fallback based on prior width prevents degenerate scales in tiny smoke
    tests or highly concentrated posteriors.
    """

    scales = []

    for name in PHYSICAL_PARAMETERS:
        posterior_std = float(
            scenarios[name].std(ddof=1)
        ) if len(scenarios) > 1 else 0.0

        prior_width = float(
            prior_bounds[name][1]
            - prior_bounds[name][0]
        )

        floor_scale = max(
            1.0e-6 * prior_width,
            1.0e-8,
        )

        scales.append(
            max(
                posterior_std,
                floor_scale,
            )
        )

    return np.asarray(scales, dtype=float)


def representative_sigma(
    retained_samples: np.ndarray,
    parameter_names: Sequence[str],
    statistic: str,
) -> float:
    sigma = posterior_sigma_samples(
        retained_samples,
        parameter_names,
    )

    if statistic == "median":
        return float(np.median(sigma))
    if statistic == "mean":
        return float(np.mean(sigma))
    if statistic == "q05":
        return float(np.percentile(sigma, 5.0))
    if statistic == "q95":
        return float(np.percentile(sigma, 95.0))

    raise ValueError("Unknown sigma statistic.")


# ============================================================================
# Canonical observed time and forward functions
# ============================================================================

def load_time_from_manifest(manifest: Mapping) -> np.ndarray:
    observed_path = Path(
        manifest["data_contract"]["observed_file"]
    )

    if not observed_path.exists():
        # Allow moving the repository after the original run.
        observed_path = (
            REPO_ROOT
            / "datasets"
            / "comsol"
            / "1darcy"
            / "processed"
            / "strain.csv"
        )

    if not observed_path.exists():
        raise FileNotFoundError(
            "Observed strain dataset not found:\n    {0}".format(
                observed_path
            )
        )

    observed = pd.read_csv(observed_path)

    if "time_s" not in observed.columns:
        raise RuntimeError("Observed dataset lacks time_s.")

    return observed["time_s"].to_numpy(float)


def vector_forward_factory(
    locations: Sequence[SensorLocation],
    time: np.ndarray,
    fixed_inputs: Mapping[str, float],
):
    station_names = np.asarray(
        [location.sensor_id for location in locations],
        dtype=str,
    )

    x_prime = np.asarray(
        [location.x for location in locations],
        dtype=float,
    )

    y_prime = np.asarray(
        [location.y for location in locations],
        dtype=float,
    )

    z = np.asarray(
        [location.z for location in locations],
        dtype=float,
    )

    channels = [
        "{0}_{1}".format(component, station)
        for component in COMPONENTS
        for station in station_names
    ]

    def forward_vector(parameter_vector: np.ndarray) -> np.ndarray:
        values = np.asarray(
            parameter_vector,
            dtype=float,
        ).reshape(-1)

        if values.size != len(PHYSICAL_PARAMETERS):
            raise ValueError("Physical parameter vector has wrong length.")

        geometry = {
            name: float(value)
            for name, value in zip(PHYSICAL_PARAMETERS, values)
        }

        prediction = forward_model_multi_station(
            pmax=float(fixed_inputs["pmax"]),
            tpeak=float(fixed_inputs["tpeak"]),
            d=float(fixed_inputs["d"]),
            time=time,
            x_prime=x_prime,
            y_prime=y_prime,
            x0_prime=geometry["x0_prime"],
            y0_prime=geometry["y0_prime"],
            z=z,
            a=geometry["a"],
            b=geometry["b"],
            c=float(fixed_inputs["c"]),
            nu=float(fixed_inputs["nu"]),
            h=geometry["h"],
            E=float(fixed_inputs["E"]),
            theta_deg=geometry["theta_deg"],
            alpha=float(fixed_inputs["alpha"]),
            station_names=station_names,
            debug=False,
        )

        return prediction[channels].to_numpy(float).reshape(-1)

    return forward_vector


def mapping_forward_factory(
    locations: Sequence[SensorLocation],
    time: np.ndarray,
    fixed_inputs: Mapping[str, float],
):
    vector_forward = vector_forward_factory(
        locations,
        time,
        fixed_inputs,
    )

    def forward_mapping(geometry: Mapping[str, float]) -> np.ndarray:
        vector = np.asarray(
            [geometry[name] for name in PHYSICAL_PARAMETERS],
            dtype=float,
        )

        return vector_forward(vector)

    return forward_mapping


# ============================================================================
# Numerical Jacobian
# ============================================================================

def numerical_jacobian(
    forward_vector,
    parameter_vector: np.ndarray,
    prior_bounds: Mapping[str, Sequence[float]],
    relative_step: float,
) -> np.ndarray:
    if relative_step <= 0.0:
        raise ValueError("--fd-step must be positive.")

    p = np.asarray(
        parameter_vector,
        dtype=float,
    ).reshape(-1)

    baseline = np.asarray(
        forward_vector(p),
        dtype=float,
    ).reshape(-1)

    J = np.zeros(
        (
            baseline.size,
            p.size,
        ),
        dtype=float,
    )

    for column, name in enumerate(PHYSICAL_PARAMETERS):
        prior_width = float(
            prior_bounds[name][1]
            - prior_bounds[name][0]
        )

        step = relative_step * max(
            abs(p[column]),
            0.01 * prior_width,
            1.0,
        )

        lower = float(prior_bounds[name][0])
        upper = float(prior_bounds[name][1])

        plus = p.copy()
        minus = p.copy()

        plus[column] = min(
            p[column] + step,
            upper,
        )

        minus[column] = max(
            p[column] - step,
            lower,
        )

        if plus[column] > p[column] and minus[column] < p[column]:
            y_plus = np.asarray(
                forward_vector(plus),
                dtype=float,
            ).reshape(-1)

            y_minus = np.asarray(
                forward_vector(minus),
                dtype=float,
            ).reshape(-1)

            J[:, column] = (
                y_plus - y_minus
            ) / (
                plus[column] - minus[column]
            )

        elif plus[column] > p[column]:
            y_plus = np.asarray(
                forward_vector(plus),
                dtype=float,
            ).reshape(-1)

            J[:, column] = (
                y_plus - baseline
            ) / (
                plus[column] - p[column]
            )

        elif minus[column] < p[column]:
            y_minus = np.asarray(
                forward_vector(minus),
                dtype=float,
            ).reshape(-1)

            J[:, column] = (
                baseline - y_minus
            ) / (
                p[column] - minus[column]
            )

        else:
            raise RuntimeError(
                "No feasible finite-difference step for {0}.".format(name)
            )

    if not np.all(np.isfinite(J)):
        raise RuntimeError("Jacobian contains non-finite values.")

    return J


# ============================================================================
# Correct transient station-row selector
# ============================================================================

def station_rows_time_series(
    n_times: int,
    station_count: int,
    station_indices: Sequence[int],
) -> np.ndarray:
    selected = tuple(
        sorted(
            set(int(index) for index in station_indices)
        )
    )

    if not selected:
        raise ValueError("At least one station is required.")

    if any(
        index < 0 or index >= station_count
        for index in selected
    ):
        raise IndexError("Invalid station index.")

    channel_indices = []

    for component_index in range(len(COMPONENTS)):
        offset = component_index * station_count
        channel_indices.extend(
            offset + station_index
            for station_index in selected
        )

    channel_indices = np.asarray(
        channel_indices,
        dtype=int,
    )

    rows = []

    channels_per_time = len(COMPONENTS) * station_count

    for time_index in range(n_times):
        rows.extend(
            time_index * channels_per_time
            + channel_indices
        )

    return np.asarray(rows, dtype=int)


# ============================================================================
# Fisher utility
# ============================================================================

def fisher_metrics_from_jacobian(
    jacobian: np.ndarray,
    sigma: float,
    parameter_scales: np.ndarray,
    regularization: float,
) -> Dict[str, float]:
    weights = make_observation_weights(
        jacobian.shape[0],
        sigma,
    )

    fisher = fisher_information_matrix(
        jacobian,
        observation_weights=weights,
        parameter_scales=parameter_scales,
    )

    result = fisher_diagnostics(
        fisher,
        regularization=regularization,
    )

    return {
        "d_optimality": float(result.log_determinant),
        "a_optimality": float(np.trace(result.covariance)),
        "e_optimality": float(result.min_eigenvalue),
        "condition_number": float(result.condition_number),
        "rank": int(result.rank),
        "trace_information": float(result.trace),
    }


# ============================================================================
# Optimized existing-network posterior OED
# ============================================================================

def run_posterior_existing(
    output_dir: Path,
    stations: pd.DataFrame,
    time: np.ndarray,
    fixed_inputs: Mapping[str, float],
    prior_bounds: Mapping[str, Sequence[float]],
    scenarios: pd.DataFrame,
    sigma: float,
    parameter_scales: np.ndarray,
    subset_sizes: Sequence[int],
    fd_step: float,
    regularization: float,
    conservative_quantile: float,
) -> pd.DataFrame:

    print(
        "\n[Bayesian OED] Optimized posterior existing-network analysis"
    )

    station_locations = [
        SensorLocation(
            sensor_id=str(row["station"]).strip(),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
        )
        for _, row in stations.iterrows()
    ]

    full_forward = vector_forward_factory(
        station_locations,
        time,
        fixed_inputs,
    )

    station_names = [
        location.sensor_id
        for location in station_locations
    ]

    subsets = []

    for subset_size in subset_sizes:
        for subset in combinations(
            range(len(station_locations)),
            subset_size,
        ):
            sensor_ids = tuple(
                station_names[index]
                for index in subset
            )

            subsets.append(
                (
                    subset,
                    sensor_ids,
                )
            )

    print(
        "  posterior scenarios = {0}".format(len(scenarios))
    )

    print(
        "  station subsets     = {0}".format(len(subsets))
    )

    print(
        "  expensive Jacobians = {0} (one per posterior scenario)".format(
            len(scenarios)
        )
    )

    records = []

    jacobian_times = []

    for scenario_number, (_, scenario_row) in enumerate(
        scenarios.iterrows(),
        start=1,
    ):
        scenario_start = _time.time()

        parameter_vector = np.asarray(
            [
                float(scenario_row[name])
                for name in PHYSICAL_PARAMETERS
            ],
            dtype=float,
        )

        full_jacobian = numerical_jacobian(
            full_forward,
            parameter_vector,
            prior_bounds,
            fd_step,
        )

        jacobian_elapsed = _time.time() - scenario_start
        jacobian_times.append(jacobian_elapsed)

        mean_jacobian_time = float(
            np.mean(jacobian_times)
        )

        remaining = (
            len(scenarios)
            - scenario_number
        ) * mean_jacobian_time

        print(
            "  Jacobian {0}/{1}: {2:.1f} s | estimated Jacobian time remaining {3:.1f} s".format(
                scenario_number,
                len(scenarios),
                jacobian_elapsed,
                remaining,
            )
        )

        for subset, sensor_ids in subsets:
            rows = station_rows_time_series(
                n_times=len(time),
                station_count=len(station_locations),
                station_indices=subset,
            )

            subset_jacobian = full_jacobian[
                rows,
                :,
            ]

            metrics = fisher_metrics_from_jacobian(
                subset_jacobian,
                sigma,
                parameter_scales,
                regularization,
            )

            records.append(
                {
                    "design_id": "+".join(sensor_ids),
                    "design_size": len(subset),
                    "sensor_ids": ",".join(sensor_ids),
                    "scenario_id": str(
                        scenario_row["scenario_id"]
                    ),
                    "scenario_weight": float(
                        scenario_row["weight"]
                    ),
                    **metrics,
                }
            )

    long_table = pd.DataFrame(records)

    long_table.to_csv(
        output_dir
        / "tables"
        / "posterior_existing_scenario_metrics.csv",
        index=False,
    )

    summaries = []

    for design_id, group in long_table.groupby(
        "design_id",
        sort=False,
    ):
        summary = bayesian_design_summary(
            design_id=design_id,
            design_size=int(group["design_size"].iloc[0]),
            utilities=group["d_optimality"].to_numpy(float),
            scenario_weights=group["scenario_weight"].to_numpy(float),
            metric="d_optimality",
            conservative_quantile=conservative_quantile,
        )

        # Also summarize A and E for reporting.
        a_summary = weighted_metric_summary(
            group["a_optimality"].to_numpy(float),
            group["scenario_weight"].to_numpy(float),
            higher_is_better=False,
            conservative_quantile=conservative_quantile,
        )

        e_summary = weighted_metric_summary(
            group["e_optimality"].to_numpy(float),
            group["scenario_weight"].to_numpy(float),
            higher_is_better=True,
            conservative_quantile=conservative_quantile,
        )

        summaries.append(
            {
                "design_id": summary.design_id,
                "design_size": summary.design_size,
                "sensor_ids": group["sensor_ids"].iloc[0],
                "d_expected": summary.expected_utility,
                "d_median": summary.median_utility,
                "d_conservative": summary.conservative_utility,
                "d_worst_case": summary.worst_case_utility,
                "d_std": summary.utility_std,
                "d_cv": summary.utility_cv,
                "a_expected": a_summary["expected"],
                "a_conservative": a_summary["conservative"],
                "a_worst_case": a_summary["worst_case"],
                "e_expected": e_summary["expected"],
                "e_conservative": e_summary["conservative"],
                "e_worst_case": e_summary["worst_case"],
                "n_scenarios": summary.n_scenarios,
            }
        )

    ranking = pd.DataFrame(summaries)

    ranking = ranking.sort_values(
        by=[
            "d_conservative",
            "d_expected",
            "design_size",
            "design_id",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ranking["posterior_rank"] = np.arange(
        1,
        len(ranking) + 1,
        dtype=int,
    )

    ranking.to_csv(
        output_dir
        / "tables"
        / "posterior_existing_network_ranking.csv",
        index=False,
    )

    best_by_size_rows = []

    for design_size, group in ranking.groupby(
        "design_size",
        sort=True,
    ):
        best_by_size_rows.append(
            group.sort_values(
                by=[
                    "d_conservative",
                    "d_expected",
                ],
                ascending=[
                    False,
                    False,
                ],
            ).iloc[0]
        )

    best_by_size = pd.DataFrame(
        best_by_size_rows
    ).reset_index(drop=True)

    best_by_size.to_csv(
        output_dir
        / "tables"
        / "posterior_best_existing_by_size.csv",
        index=False,
    )

    # Ranking figure.
    top = ranking.head(
        min(15, len(ranking))
    ).iloc[::-1]

    fig, ax = plt.subplots(
        figsize=(
            11.0,
            max(
                6.5,
                0.45 * len(top),
            ),
        )
    )

    ax.barh(
        top["design_id"],
        top["d_conservative"],
        label="Posterior conservative D-utility",
    )

    ax.scatter(
        top["d_expected"],
        top["design_id"],
        label="Posterior expected D-utility",
        zorder=3,
    )

    ax.set_xlabel(
        r"Posterior D-optimal utility, $\log\det(F)$"
    )

    ax.set_ylabel("Existing station network")

    ax.set_title(
        "Posterior-informed existing-network OED"
    )

    handles, labels = ax.get_legend_handles_labels()

    save_figure(
        fig,
        output_dir
        / "figures"
        / "posterior_existing_network_ranking",
        handles,
        labels,
        ncol=2,
    )

    # Best utility versus sensor count.
    fig, ax = plt.subplots(
        figsize=(9.0, 6.2)
    )

    ax.plot(
        best_by_size["design_size"],
        best_by_size["d_conservative"],
        marker="o",
        linewidth=2.2,
        label="Conservative posterior utility",
    )

    ax.plot(
        best_by_size["design_size"],
        best_by_size["d_expected"],
        marker="s",
        linewidth=2.0,
        label="Expected posterior utility",
    )

    ax.set_xlabel("Number of strainmeters")
    ax.set_ylabel(r"$\log\det(F)$")
    ax.set_title(
        "Posterior information versus existing-network size"
    )

    handles, labels = ax.get_legend_handles_labels()

    save_figure(
        fig,
        output_dir
        / "figures"
        / "posterior_information_vs_sensor_count",
        handles,
        labels,
        ncol=2,
    )

    print(
        "  best posterior network: {0}".format(
            ranking.iloc[0]["sensor_ids"]
        )
    )

    return ranking


def weighted_metric_summary(
    values: np.ndarray,
    weights: np.ndarray,
    higher_is_better: bool,
    conservative_quantile: float,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    weights = weights / np.sum(weights)

    expected = float(
        np.sum(weights * values)
    )

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)

    def quantile(q):
        index = int(
            np.searchsorted(
                cumulative,
                q,
                side="left",
            )
        )
        index = min(
            max(index, 0),
            len(sorted_values) - 1,
        )
        return float(sorted_values[index])

    if higher_is_better:
        conservative = quantile(
            conservative_quantile
        )
        worst_case = float(np.min(values))
    else:
        conservative = quantile(
            1.0 - conservative_quantile
        )
        worst_case = float(np.max(values))

    return {
        "expected": expected,
        "conservative": conservative,
        "worst_case": worst_case,
    }


# ============================================================================
# Posterior placement
# ============================================================================

def build_placement_networks(
    args: argparse.Namespace,
) -> Tuple[List[SensorLocation], List[SensorNetwork]]:
    candidate_locations = generate_rectangular_grid(
        [
            args.placement_xmin,
            args.placement_xmax,
            args.placement_ymin,
            args.placement_ymax,
        ],
        nx=args.placement_grid_nx,
        ny=args.placement_grid_ny,
        z=-40.0,
        prefix="BP",
        include_boundary=True,
    )

    networks = random_candidate_networks(
        candidate_locations,
        design_size=args.placement_size,
        n_networks=args.placement_networks,
        seed=args.seed,
        min_spacing=args.placement_min_spacing,
    )

    return candidate_locations, networks


def run_posterior_placement(
    output_dir: Path,
    networks: Sequence[SensorNetwork],
    candidate_locations: Sequence[SensorLocation],
    scenarios: pd.DataFrame,
    time: np.ndarray,
    fixed_inputs: Mapping[str, float],
    prior_bounds: Mapping[str, Sequence[float]],
    sigma: float,
    parameter_scales: np.ndarray,
    fd_step: float,
    regularization: float,
    conservative_quantile: float,
) -> pd.DataFrame:

    print(
        "\n[Bayesian OED] Posterior-informed free spatial placement"
    )

    records = []

    total_jacobians = len(networks) * len(scenarios)
    completed = 0
    timing = []

    for network in networks:
        forward = vector_forward_factory(
            network.sensor_locations,
            time,
            fixed_inputs,
        )

        for _, scenario_row in scenarios.iterrows():
            start = _time.time()

            parameter_vector = np.asarray(
                [
                    float(scenario_row[name])
                    for name in PHYSICAL_PARAMETERS
                ],
                dtype=float,
            )

            J = numerical_jacobian(
                forward,
                parameter_vector,
                prior_bounds,
                fd_step,
            )

            metrics = fisher_metrics_from_jacobian(
                J,
                sigma,
                parameter_scales,
                regularization,
            )

            records.append(
                {
                    "design_id": network.design_id,
                    "design_size": network.design_size,
                    "sensor_ids": ",".join(network.sensor_ids),
                    "scenario_id": str(
                        scenario_row["scenario_id"]
                    ),
                    "scenario_weight": float(
                        scenario_row["weight"]
                    ),
                    **metrics,
                }
            )

            completed += 1
            timing.append(_time.time() - start)

            if (
                completed == 1
                or completed % 25 == 0
                or completed == total_jacobians
            ):
                mean_time = float(np.mean(timing))
                remaining = (
                    total_jacobians - completed
                ) * mean_time

                print(
                    "  placement Jacobian {0}/{1} | estimated remaining {2:.1f} s".format(
                        completed,
                        total_jacobians,
                        remaining,
                    )
                )

    long_table = pd.DataFrame(records)

    long_table.to_csv(
        output_dir
        / "tables"
        / "posterior_placement_scenario_metrics.csv",
        index=False,
    )

    summaries = []

    for design_id, group in long_table.groupby(
        "design_id",
        sort=False,
    ):
        summary = bayesian_design_summary(
            design_id=design_id,
            design_size=int(group["design_size"].iloc[0]),
            utilities=group["d_optimality"].to_numpy(float),
            scenario_weights=group["scenario_weight"].to_numpy(float),
            metric="d_optimality",
            conservative_quantile=conservative_quantile,
        )

        summaries.append(
            {
                "design_id": summary.design_id,
                "design_size": summary.design_size,
                "sensor_ids": group["sensor_ids"].iloc[0],
                "d_expected": summary.expected_utility,
                "d_median": summary.median_utility,
                "d_conservative": summary.conservative_utility,
                "d_worst_case": summary.worst_case_utility,
                "d_std": summary.utility_std,
                "d_cv": summary.utility_cv,
            }
        )

    ranking = pd.DataFrame(summaries).sort_values(
        by=[
            "d_conservative",
            "d_expected",
        ],
        ascending=[
            False,
            False,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ranking["posterior_rank"] = np.arange(
        1,
        len(ranking) + 1,
        dtype=int,
    )

    ranking.to_csv(
        output_dir
        / "tables"
        / "posterior_placement_ranking.csv",
        index=False,
    )

    network_map = {
        network.design_id: network
        for network in networks
    }

    best = network_map[
        str(ranking.iloc[0]["design_id"])
    ]

    fig, ax = plt.subplots(
        figsize=(9.0, 8.5)
    )

    ax.scatter(
        [location.x for location in candidate_locations],
        [location.y for location in candidate_locations],
        s=18,
        alpha=0.35,
        label="Candidate locations",
    )

    ax.scatter(
        [location.x for location in best.sensor_locations],
        [location.y for location in best.sensor_locations],
        s=125,
        marker="*",
        label="Best posterior-informed network",
    )

    for location in best.sensor_locations:
        ax.annotate(
            location.sensor_id,
            (location.x, location.y),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_xlabel("$x'$ (m)")
    ax.set_ylabel("$y'$ (m)")
    ax.set_title(
        "Posterior-informed robust strainmeter placement"
    )
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()

    save_figure(
        fig,
        output_dir
        / "figures"
        / "posterior_best_placement_map",
        handles,
        labels,
        ncol=2,
    )

    print(
        "  best posterior placement: {0}".format(
            ranking.iloc[0]["sensor_ids"]
        )
    )

    return ranking


# ============================================================================
# EIG
# ============================================================================

def run_eig(
    output_dir: Path,
    networks: Sequence[SensorNetwork],
    scenarios: pd.DataFrame,
    time: np.ndarray,
    fixed_inputs: Mapping[str, float],
    sigma: float,
    eig_designs: int,
    eig_outer: int,
    eig_inner: int,
    seed: int,
) -> pd.DataFrame:

    print(
        "\n[Bayesian OED] Expected information gain"
    )

    selected_networks = list(networks)[
        :min(
            eig_designs,
            len(networks),
        )
    ]

    results = []

    for index, network in enumerate(
        selected_networks,
        start=1,
    ):
        print(
            "  EIG design {0}/{1}: {2}".format(
                index,
                len(selected_networks),
                network.design_id,
            )
        )

        forward = mapping_forward_factory(
            network.sensor_locations,
            time,
            fixed_inputs,
        )

        result = estimate_expected_information_gain(
            design_id=network.design_id,
            design_size=network.design_size,
            scenarios=scenarios,
            forward_function=forward,
            observation_sigma=sigma,
            n_outer_samples=eig_outer,
            n_inner_samples=eig_inner,
            seed=seed + index,
        )

        results.append(result)

    ranking = rank_expected_information_gain(
        results
    )

    ranking.to_csv(
        output_dir
        / "tables"
        / "expected_information_gain_ranking.csv",
        index=False,
    )

    return ranking


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    if args.posterior_samples <= 0:
        raise ValueError("--posterior-samples must be positive.")

    if not 0.0 < args.conservative_quantile < 0.5:
        raise ValueError(
            "--conservative-quantile must satisfy 0 < q < 0.5."
        )

    if args.regularization < 0.0:
        raise ValueError("--regularization cannot be negative.")

    configure_plot_style()

    (
        run_dir,
        manifest,
        run_summary,
        flat,
        chains,
        logps,
        stations,
    ) = load_run_contract(
        args.run_dir
    )

    validation = validate_run_contract(
        manifest,
        run_summary,
        flat,
        chains,
        logps,
        stations,
    )

    print("=" * 78)
    print("AVANT 1-DARCY OPTIMIZED BAYESIAN OED")
    print("=" * 78)

    print("\nBayesian run:")
    print("  {0}".format(run_dir))

    print("\nPosterior:")
    print(
        "  flat samples         = {0}".format(
            validation["n_flat_samples"]
        )
    )
    print(
        "  chains               = {0}".format(
            validation["n_chains"]
        )
    )
    print(
        "  iterations per chain = {0}".format(
            validation["iterations_per_chain"]
        )
    )
    print(
        "  converged            = {0}".format(
            validation["converged"]
        )
    )
    print(
        "  status               = {0}".format(
            validation["status"]
        )
    )

    if not validation["converged"]:
        print(
            "\nWARNING: Bayesian run is not converged."
        )

        if args.mode == "validate":
            print(
                "Validation only: contract is valid; exiting."
            )
            return

        if not args.allow_nonconverged:
            raise RuntimeError(
                "Posterior-informed OED requires a converged Bayesian run. "
                "Use --allow-nonconverged only for integration testing."
            )

    if args.mode == "validate":
        print(
            "\nBayesian run contract: VALID"
        )
        return

    if args.output_dir is None:
        output_dir = run_dir / "bayesian_oed"
    else:
        output_dir = args.output_dir.resolve()

    (output_dir / "tables").mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_names = manifest["parameter_names"]
    prior_bounds = manifest["priors"]
    fixed_inputs = manifest["fixed_model_inputs"]

    retained = retained_flat_from_chains(
        chains,
        args.burn_in_fraction,
    )

    scenarios = posterior_scenario_table(
        retained,
        parameter_names,
        args.posterior_samples,
        args.seed,
    )

    scenarios.to_csv(
        output_dir
        / "tables"
        / "posterior_geometry_scenarios.csv",
        index=False,
    )

    parameter_scales = posterior_parameter_scales(
        scenarios,
        prior_bounds,
    )

    sigma = representative_sigma(
        retained,
        parameter_names,
        args.sigma_statistic,
    )

    time = load_time_from_manifest(
        manifest
    )

    subset_sizes = parse_subset_sizes(
        args.existing_subset_sizes,
        len(stations),
    )

    save_json(
        output_dir
        / "bayesian_oed_manifest.json",
        {
            "source_bayesian_run": run_dir,
            "mode": args.mode,
            "bayesian_run_converged": validation["converged"],
            "burn_in_fraction": args.burn_in_fraction,
            "retained_samples": int(retained.shape[0]),
            "posterior_scenarios_used": int(len(scenarios)),
            "physical_parameters": PHYSICAL_PARAMETERS,
            "parameter_scales": parameter_scales,
            "sigma_statistic": args.sigma_statistic,
            "representative_sigma_nstrain": sigma,
            "finite_difference_step": args.fd_step,
            "regularization": args.regularization,
            "existing_subset_sizes": subset_sizes,
            "seed": args.seed,
        },
    )

    print("\nBayesian OED configuration:")
    print(
        "  retained chain samples = {0}".format(
            retained.shape[0]
        )
    )
    print(
        "  posterior scenarios     = {0}".format(
            len(scenarios)
        )
    )
    print(
        "  representative sigma    = {0:.6g} nstrain".format(
            sigma
        )
    )
    print(
        "  parameter scales        = {0}".format(
            np.array2string(
                parameter_scales,
                precision=4,
            )
        )
    )

    start = _time.time()

    if args.mode in (
        "posterior_existing",
        "all",
    ):
        run_posterior_existing(
            output_dir=output_dir,
            stations=stations,
            time=time,
            fixed_inputs=fixed_inputs,
            prior_bounds=prior_bounds,
            scenarios=scenarios,
            sigma=sigma,
            parameter_scales=parameter_scales,
            subset_sizes=subset_sizes,
            fd_step=args.fd_step,
            regularization=args.regularization,
            conservative_quantile=args.conservative_quantile,
        )

    candidate_locations = None
    placement_networks = None

    if args.mode in (
        "posterior_placement",
        "eig",
        "all",
    ):
        (
            candidate_locations,
            placement_networks,
        ) = build_placement_networks(
            args
        )

    if args.mode in (
        "posterior_placement",
        "all",
    ):
        run_posterior_placement(
            output_dir=output_dir,
            networks=placement_networks,
            candidate_locations=candidate_locations,
            scenarios=scenarios,
            time=time,
            fixed_inputs=fixed_inputs,
            prior_bounds=prior_bounds,
            sigma=sigma,
            parameter_scales=parameter_scales,
            fd_step=args.fd_step,
            regularization=args.regularization,
            conservative_quantile=args.conservative_quantile,
        )

    if args.mode in (
        "eig",
        "all",
    ):
        run_eig(
            output_dir=output_dir,
            networks=placement_networks,
            scenarios=scenarios,
            time=time,
            fixed_inputs=fixed_inputs,
            sigma=sigma,
            eig_designs=args.eig_designs,
            eig_outer=args.eig_outer,
            eig_inner=args.eig_inner,
            seed=args.seed,
        )

    elapsed = _time.time() - start

    save_json(
        output_dir
        / "completion_summary.json",
        {
            "completed": True,
            "mode": args.mode,
            "elapsed_seconds": elapsed,
            "bayesian_run_converged": validation["converged"],
            "source_bayesian_run": run_dir,
        },
    )

    print("\n" + "=" * 78)
    print("OPTIMIZED BAYESIAN OED COMPLETE")
    print("=" * 78)
    print(
        "Elapsed time: {0:.1f} s".format(
            elapsed
        )
    )
    print(
        "Output directory:\n    {0}".format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
