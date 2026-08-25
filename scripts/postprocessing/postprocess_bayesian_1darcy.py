#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production post-processing for the AVANT 1-Darcy transient analytical
Bayesian inversion.

This module is deliberately separated from the sampler. It consumes a
completed PyDREAM run and produces reproducible posterior summaries,
diagnostics, posterior-predictive uncertainty, and publication-quality
station plots.

Canonical posterior state
-------------------------
    [a, b, h, theta_deg, x0_prime, y0_prime, log10_sigma_strain]

Fixed analytical-model inputs are read from the run manifest produced by
run_inversion_multi_station.py:

    c
    E
    nu
    pmax
    tpeak
    d
    alpha

The nuisance parameter is transformed to physical strain-noise units as

    sigma_strain = 10**log10_sigma_strain

Uncertainty terminology
-----------------------
1. Parameter uncertainty (epistemic):
       spread caused by uncertainty in the six inferred physical
       parameters.

2. Total predictive uncertainty:
       parameter uncertainty plus a Gaussian residual term using the
       sigma_strain value carried by each posterior draw.

The total predictive band is therefore the posterior predictive band under
the Gaussian likelihood used by the Bayesian inversion.

The observed-versus-MAP residual is reported separately. It is not itself
treated as an independent stochastic noise process.

Canonical data contract
-----------------------
The run directory must contain:

    run_manifest.json
    posterior_chains.npy
    posterior_logps.npy
    posterior_samples_flat.npy
    convergence_history.json
    station_metadata_used.csv

The transient observed dataset is resolved from the run manifest:

    datasets/comsol/1darcy/processed/strain.csv

The station coordinates are resolved from the run manifest:

    datasets/comsol/1darcy/metadata/stations.csv

or, preferentially for reproducibility, from the
station_metadata_used.csv saved with the run.

Outputs
-------
postprocessed/
    posterior_parameter_summary.csv
    posterior_parameter_summary.json
    posterior_correlation_matrix.csv
    convergence_summary.json
    predictive_metrics_by_station_component.csv
    posterior_traces.png
    posterior_distributions.png
    posterior_correlation_matrix.png
    posterior_pairwise.png
    station_S01_observed_vs_posterior.png
    ...
    station_S08_observed_vs_posterior.png
    station_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================================
# Repository imports
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avant_model.model.forward_model_multi_station import (
    forward_model_multi_station,
)


# ============================================================================
# Constants
# ============================================================================

PARAMETER_NAMES = (
    "a",
    "b",
    "h",
    "theta_deg",
    "x0_prime",
    "y0_prime",
    "log10_sigma_strain",
)

PHYSICAL_PARAMETER_NAMES = (
    "a",
    "b",
    "h",
    "theta_deg",
    "x0_prime",
    "y0_prime",
)

COMPONENTS = (
    "eXX",
    "eYY",
    "eZZ",
    "eXY",
)

COMPONENT_LABELS = {
    "eXX": r"$\varepsilon_{xx}$",
    "eYY": r"$\varepsilon_{yy}$",
    "eZZ": r"$\varepsilon_{zz}$",
    "eXY": r"$\varepsilon_{xy}$",
}

PARAMETER_LABELS = {
    "a": r"$a$ (m)",
    "b": r"$b$ (m)",
    "h": r"$h$ (m)",
    "theta_deg": r"$\theta$ (deg)",
    "x0_prime": r"$x_0'$ (m)",
    "y0_prime": r"$y_0'$ (m)",
    "log10_sigma_strain": r"$\log_{10}\sigma_\mathrm{strain}$",
}

OBSERVED_LABEL = "Observed"
MAP_LABEL_SUFFIX = "MAP analytical"
PARAM_BAND_LABEL = "95% parameter uncertainty"
TOTAL_BAND_LABEL = "95% total predictive uncertainty"

DEFAULT_BURN_IN = 0.20
DEFAULT_N_PREDICTIVE = 500
DEFAULT_SEED = 42
DEFAULT_MAX_PAIRWISE = 5000

# Plot colors are intentionally fixed for reproducible publication figures.
COMPONENT_COLORS = {
    "eXX": "#1f77b4",
    "eYY": "#ff7f0e",
    "eZZ": "#2ca02c",
    "eXY": "#d62728",
}

# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a completed AVANT 1-Darcy PyDREAM Bayesian run."
        )
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "PyDREAM run directory containing posterior_chains.npy, "
            "posterior_logps.npy, and run_manifest.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for post-processing outputs. "
            "Default: <run-dir>/postprocessed."
        ),
    )

    parser.add_argument(
        "--burn-in-fraction",
        type=float,
        default=DEFAULT_BURN_IN,
        help=(
            "Fraction of each chain removed as burn-in before posterior "
            "summaries and predictive propagation."
        ),
    )

    parser.add_argument(
        "--n-predictive-draws",
        type=int,
        default=DEFAULT_N_PREDICTIVE,
        help=(
            "Number of posterior draws propagated through the forward model."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for posterior subsampling and predictive noise.",
    )

    parser.add_argument(
        "--max-pairwise-samples",
        type=int,
        default=DEFAULT_MAX_PAIRWISE,
        help="Maximum samples used for the pairwise posterior plot.",
    )

    parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "Stop with an error if the sampler did not report convergence. "
            "By default, non-converged runs are processed but explicitly "
            "marked as diagnostic-only."
        ),
    )

    return parser.parse_args()


# ============================================================================
# General helpers
# ============================================================================


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n    {path}"
        )


def percentile_summary(
    values: np.ndarray,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "q02_5": float(np.percentile(values, 2.5)),
        "q05": float(np.percentile(values, 5.0)),
        "q25": float(np.percentile(values, 25.0)),
        "q75": float(np.percentile(values, 75.0)),
        "q95": float(np.percentile(values, 95.0)),
        "q97_5": float(np.percentile(values, 97.5)),
    }


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(observed) - np.asarray(predicted)
    return float(np.sqrt(np.mean(residual**2)))


def mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(observed) - np.asarray(predicted)
    return float(np.mean(np.abs(residual)))


def bias(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(observed) - np.asarray(predicted)
    return float(np.mean(residual))


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot <= 0.0:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def coverage(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    observed = np.asarray(observed, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    inside = (
        (observed >= lower)
        & (observed <= upper)
    )

    return float(np.mean(inside))


def format_station_label(station: str) -> str:
    return str(station).strip()


# ============================================================================
# Run loading and validation
# ============================================================================


def load_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()

    require_file(
        run_dir / "run_manifest.json",
        "run manifest",
    )

    require_file(
        run_dir / "posterior_chains.npy",
        "posterior chains",
    )

    require_file(
        run_dir / "posterior_logps.npy",
        "posterior log-probabilities",
    )

    require_file(
        run_dir / "convergence_history.json",
        "convergence history",
    )

    manifest = load_json(
        run_dir / "run_manifest.json"
    )

    chains = np.load(
        run_dir / "posterior_chains.npy",
        allow_pickle=False,
    )

    logps = np.load(
        run_dir / "posterior_logps.npy",
        allow_pickle=False,
    )

    convergence_history = load_json(
        run_dir / "convergence_history.json"
    )

    if chains.ndim != 3:
        raise RuntimeError(
            "posterior_chains.npy must have shape "
            "(nchains, niterations, nparameters). "
            f"Received {chains.shape}."
        )

    if chains.shape[2] != len(PARAMETER_NAMES):
        raise RuntimeError(
            "Posterior parameter count does not match the canonical "
            f"seven-parameter state. Expected {len(PARAMETER_NAMES)}, "
            f"received {chains.shape[2]}."
        )

    expected_total_logps = (
        chains.shape[0] * chains.shape[1]
    )

    logps = np.asarray(
        logps,
        dtype=float,
    ).reshape(-1)

    if logps.size != expected_total_logps:
        raise RuntimeError(
            "posterior_logps.npy is inconsistent with posterior_chains.npy.\n"
            f"Expected {expected_total_logps} values, received {logps.size}."
        )

    manifest_parameter_names = tuple(
        manifest.get("parameter_names", [])
    )

    if manifest_parameter_names != PARAMETER_NAMES:
        raise RuntimeError(
            "Run manifest parameter ordering does not match the canonical "
            "production Bayesian state.\n"
            f"Expected: {PARAMETER_NAMES}\n"
            f"Found:    {manifest_parameter_names}"
        )

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "chains": chains,
        "logps": logps.reshape(
            chains.shape[0],
            chains.shape[1],
        ),
        "convergence_history": convergence_history,
    }


# ============================================================================
# Convergence handling
# ============================================================================


def convergence_status(run: dict) -> dict:
    manifest = run["manifest"]
    convergence = manifest.get(
        "convergence",
        {},
    )

    # Current runner writes convergence status into posterior_run_summary,
    # but the authoritative run manifest may not yet contain it. Therefore
    # also inspect the convergence-history records.
    converged = convergence.get(
        "converged",
        None,
    )

    status = convergence.get(
        "status",
        None,
    )

    threshold = convergence.get(
        "threshold",
        manifest.get(
            "sampler",
            {},
        ).get(
            "rhat_threshold",
            np.nan,
        ),
    )

    history = run[
        "convergence_history"
    ]

    latest_rhat = None
    criterion_reached = False

    if history:
        latest = history[-1]

        latest_rhat = latest.get(
            "rhat",
            None,
        )

        criterion_reached = bool(
            latest.get(
                "all_below_threshold",
                False,
            )
        )

    if converged is None:
        converged = criterion_reached

    if status is None:
        status = (
            "converged"
            if converged
            else "not_converged"
        )

    return {
        "status": str(status),
        "converged": bool(converged),
        "criterion": "all_rhat_below_threshold",
        "rhat_threshold": float(threshold),
        "latest_rhat": latest_rhat,
        "criterion_reached": bool(
            criterion_reached
        ),
        "history": history,
    }


# ============================================================================
# Observations and station metadata
# ============================================================================


def load_station_metadata(
    run: dict,
) -> pd.DataFrame:
    run_dir = run["run_dir"]

    used_path = (
        run_dir / "station_metadata_used.csv"
    )

    if used_path.exists():
        stations = pd.read_csv(
            used_path
        )
    else:
        station_file = Path(
            run["manifest"]["data_contract"][
                "station_file"
            ]
        )

        require_file(
            station_file,
            "station metadata",
        )

        stations = pd.read_csv(
            station_file
        )

    stations.columns = [
        str(column).strip()
        for column in stations.columns
    ]

    # Canonical production metadata is station,x,y,z.
    required = {
        "station",
        "x",
        "y",
        "z",
    }

    missing = (
        required
        - set(stations.columns)
    )

    if missing:
        raise RuntimeError(
            "Station metadata is missing required columns: "
            + ", ".join(sorted(missing))
        )

    stations = stations.loc[
        :,
        [
            "station",
            "x",
            "y",
            "z",
        ],
    ].copy()

    stations["station"] = (
        stations["station"]
        .astype(str)
        .str.strip()
    )

    for column in ("x", "y", "z"):
        stations[column] = pd.to_numeric(
            stations[column],
            errors="raise",
        )

    manifest_stations = tuple(
        run["manifest"]["data_contract"].get(
            "station_names",
            [],
        )
    )

    actual_stations = tuple(
        stations["station"].tolist()
    )

    if manifest_stations and (
        actual_stations != manifest_stations
    ):
        raise RuntimeError(
            "Station metadata ordering does not match the run manifest.\n"
            f"Manifest: {manifest_stations}\n"
            f"Loaded:   {actual_stations}"
        )

    return stations


def load_observed_dataset(
    run: dict,
    stations: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    manifest_observed = Path(
        run["manifest"]["data_contract"][
            "observed_file"
        ]
    )

    require_file(
        manifest_observed,
        "observed transient strain dataset",
    )

    observed = pd.read_csv(
        manifest_observed
    )

    observed.columns = [
        str(column).strip()
        for column in observed.columns
    ]

    if "time_s" not in observed.columns:
        raise RuntimeError(
            "Observed strain dataset must contain 'time_s'."
        )

    station_names = (
        stations["station"].tolist()
    )

    expected_columns = [
        f"{component}_{station}"
        for component in COMPONENTS
        for station in station_names
    ]

    missing = [
        column
        for column in expected_columns
        if column not in observed.columns
    ]

    if missing:
        raise RuntimeError(
            "Observed dataset is missing canonical strain channels:\n"
            + "\n".join(
                f"    {column}"
                for column in missing
            )
        )

    unexpected = [
        column
        for column in observed.columns
        if column not in {
            "time_s",
            *expected_columns,
        }
    ]

    if unexpected:
        raise RuntimeError(
            "Observed dataset contains unexpected columns:\n"
            + "\n".join(
                f"    {column}"
                for column in unexpected
            )
        )

    observed = observed.loc[
        :,
        [
            "time_s",
            *expected_columns,
        ],
    ].copy()

    for column in observed.columns:
        observed[column] = pd.to_numeric(
            observed[column],
            errors="raise",
        )

    time_values = observed[
        "time_s"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(time_values)
    ):
        raise RuntimeError(
            "Observed time vector contains non-finite values."
        )

    if not np.all(
        np.diff(time_values) >= 0.0
    ):
        raise RuntimeError(
            "Observed time vector must be monotonic non-decreasing."
        )

    return observed, expected_columns


# ============================================================================
# Posterior preparation
# ============================================================================


def apply_burn_in(
    chains: np.ndarray,
    logps: np.ndarray,
    burn_in_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if not (
        0.0 <= burn_in_fraction < 1.0
    ):
        raise ValueError(
            "burn-in fraction must satisfy 0 <= fraction < 1."
        )

    burn_in = int(
        chains.shape[1]
        * burn_in_fraction
    )

    post_chains = chains[
        :,
        burn_in:,
        :,
    ]

    post_logps = logps[
        :,
        burn_in:,
    ]

    if post_chains.shape[1] < 2:
        raise RuntimeError(
            "Too few post-burn-in iterations remain for posterior "
            "analysis. Reduce burn-in fraction or run more iterations."
        )

    posterior = post_chains.reshape(
        -1,
        chains.shape[2],
    )

    post_logps_flat = post_logps.reshape(
        -1
    )

    return (
        posterior,
        post_logps_flat,
        burn_in,
    )


def map_sample(
    posterior: np.ndarray,
    logps: np.ndarray,
) -> Tuple[np.ndarray, int]:
    index = int(
        np.argmax(logps)
    )

    return (
        posterior[index],
        index,
    )


# ============================================================================
# Forward-model context
# ============================================================================


def build_forward_context(
    run: dict,
    stations: pd.DataFrame,
    observed: pd.DataFrame,
) -> dict:
    fixed = run[
        "manifest"
    ]["fixed_model_inputs"]

    station_names = (
        stations["station"].to_numpy(
            dtype=str
        )
    )

    return {
        "pmax": float(
            fixed["pmax"]
        ),
        "E": float(
            fixed["E"]
        ),
        "c": float(
            fixed["c"]
        ),
        "nu": float(
            fixed["nu"]
        ),
        "tpeak": float(
            fixed["tpeak"]
        ),
        "d": float(
            fixed["d"]
        ),
        "alpha": float(
            fixed["alpha"]
        ),
        "time": observed[
            "time_s"
        ].to_numpy(
            dtype=float
        ),
        "x_prime": stations[
            "x"
        ].to_numpy(
            dtype=float
        ),
        "y_prime": stations[
            "y"
        ].to_numpy(
            dtype=float
        ),
        "z": stations[
            "z"
        ].to_numpy(
            dtype=float
        ),
        "station_names": station_names,
    }


def forward_from_sample(
    sample: np.ndarray,
    context: dict,
) -> pd.DataFrame:
    (
        a,
        b,
        h,
        theta_deg,
        x0_prime,
        y0_prime,
        log10_sigma_strain,
    ) = [
        float(value)
        for value in sample
    ]

    return forward_model_multi_station(
        pmax=context["pmax"],
        tpeak=context["tpeak"],
        d=context["d"],
        time=context["time"],
        x_prime=context["x_prime"],
        y_prime=context["y_prime"],
        x0_prime=x0_prime,
        y0_prime=y0_prime,
        z=context["z"],
        a=a,
        b=b,
        c=context["c"],
        nu=context["nu"],
        h=h,
        E=context["E"],
        theta_deg=theta_deg,
        alpha=context["alpha"],
        station_names=context["station_names"],
    )


# ============================================================================
# Parameter posterior summaries
# ============================================================================


def posterior_summary_dataframe(
    posterior: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for index, name in enumerate(
        PARAMETER_NAMES
    ):
        values = posterior[
            :,
            index
        ]

        statistics = percentile_summary(
            values
        )

        row = {
            "parameter": name,
            **statistics,
        }

        rows.append(
            row
        )

    # Add physical sigma in nstrain.
    sigma_values = (
        10.0
        ** posterior[
            :,
            PARAMETER_NAMES.index(
                "log10_sigma_strain"
            ),
        ]
    )

    sigma_statistics = (
        percentile_summary(
            sigma_values
        )
    )

    rows.append(
        {
            "parameter": "sigma_strain",
            **sigma_statistics,
        }
    )

    return pd.DataFrame(
        rows
    )


def posterior_correlation_dataframe(
    posterior: np.ndarray,
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        posterior,
        columns=PARAMETER_NAMES,
    )

    # Add sigma in physical nstrain units as a separate interpretable
    # quantity while keeping the sampled log parameter in the matrix.
    dataframe["sigma_strain"] = (
        10.0
        ** dataframe[
            "log10_sigma_strain"
        ]
    )

    return dataframe.corr()


# ============================================================================
# Publication-quality parameter plots
# ============================================================================


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def plot_traces(
    chains: np.ndarray,
    output_path: Path,
) -> None:
    n_chains, n_iterations, _ = (
        chains.shape
    )

    fig, axes = plt.subplots(
        len(PARAMETER_NAMES),
        1,
        figsize=(
            11,
            1.9 * len(PARAMETER_NAMES),
        ),
        sharex=True,
    )

    if len(PARAMETER_NAMES) == 1:
        axes = [axes]

    for index, name in enumerate(
        PARAMETER_NAMES
    ):
        ax = axes[index]

        for chain_index in range(
            n_chains
        ):
            ax.plot(
                np.arange(
                    n_iterations
                ),
                chains[
                    chain_index,
                    :,
                    index,
                ],
                linewidth=0.8,
                alpha=0.65,
            )

        ax.set_ylabel(
            PARAMETER_LABELS[name]
        )
        ax.grid(
            True,
            alpha=0.20,
            linestyle="--",
        )

    axes[-1].set_xlabel(
        "Iteration"
    )

    fig.suptitle(
        "PyDREAM parameter traces",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.99]
    )

    fig.savefig(
        output_path
    )

    plt.close(fig)


def plot_posterior_distributions(
    posterior: np.ndarray,
    map_sample_vector: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 11),
    )

    axes = axes.ravel()

    for index, name in enumerate(
        PARAMETER_NAMES
    ):

        values = posterior[
            :,
            index,
        ]

        ax = axes[index]

        ax.hist(
            values,
            bins=50,
            density=True,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.5,
        )

        median = np.median(
            values
        )

        q02_5, q97_5 = np.percentile(
            values,
            [2.5, 97.5],
        )

        mean = np.mean(
            values
        )

        map_value = (
            map_sample_vector[index]
        )

        ax.axvspan(
            q02_5,
            q97_5,
            alpha=0.12,
            label="95% credible interval",
        )

        ax.axvline(
            mean,
            linestyle="--",
            linewidth=1.8,
            label="Posterior mean",
        )

        ax.axvline(
            median,
            linestyle="-.",
            linewidth=1.5,
            label="Posterior median",
        )

        ax.axvline(
            map_value,
            linestyle=":",
            linewidth=2.0,
            label="MAP",
        )

        ax.set_xlabel(
            PARAMETER_LABELS[name]
        )

        ax.set_ylabel(
            "Density"
        )

        ax.grid(
            True,
            alpha=0.20,
            linestyle="--",
        )

    # Physical sigma in the last panel.
    sigma_index = (
        PARAMETER_NAMES.index(
            "log10_sigma_strain"
        )
    )

    sigma_values = (
        10.0
        ** posterior[
            :,
            sigma_index,
        ]
    )

    ax = axes[
        sigma_index
    ]

    ax.clear()

    sigma_map = (
        10.0
        ** map_sample_vector[
            sigma_index
        ]
    )

    sigma_mean = (
        np.mean(
            sigma_values
        )
    )

    sigma_median = (
        np.median(
            sigma_values
        )
    )

    sigma_q02_5, sigma_q97_5 = (
        np.percentile(
            sigma_values,
            [2.5, 97.5],
        )
    )

    ax.hist(
        sigma_values,
        bins=50,
        density=True,
        alpha=0.65,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axvspan(
        sigma_q02_5,
        sigma_q97_5,
        alpha=0.12,
        label="95% credible interval",
    )

    ax.axvline(
        sigma_mean,
        linestyle="--",
        linewidth=1.8,
        label="Posterior mean",
    )

    ax.axvline(
        sigma_median,
        linestyle="-.",
        linewidth=1.5,
        label="Posterior median",
    )

    ax.axvline(
        sigma_map,
        linestyle=":",
        linewidth=2.0,
        label="MAP",
    )

    ax.set_xlabel(
        r"$\sigma_{\mathrm{strain}}$ (n$\varepsilon$)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.grid(
        True,
        alpha=0.20,
        linestyle="--",
    )

    # Remove unused panels.
    for index in range(
        len(PARAMETER_NAMES),
        len(axes),
    ):
        axes[index].axis(
            "off"
        )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.01,
        ),
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        "Posterior parameter distributions",
        y=0.99,
    )

    fig.subplots_adjust(
        bottom=0.11,
        top=0.93,
        hspace=0.38,
    )

    fig.savefig(
        output_path
    )

    plt.close(fig)


def plot_correlation_matrix(
    correlation: pd.DataFrame,
    output_path: Path,
) -> None:
    matrix = correlation.to_numpy(
        dtype=float
    )

    labels = [
        r"$a$",
        r"$b$",
        r"$h$",
        r"$\theta$",
        r"$x_0'$",
        r"$y_0'$",
        r"$\log_{10}\sigma$",
        r"$\sigma$",
    ]

    fig, ax = plt.subplots(
        figsize=(9, 8),
    )

    image = ax.imshow(
        matrix,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        labels,
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.85,
    )

    colorbar.set_label(
        "Posterior correlation"
    )

    ax.set_title(
        "Posterior parameter correlation matrix"
    )

    fig.tight_layout()

    fig.savefig(
        output_path
    )

    plt.close(fig)


def plot_pairwise(
    posterior: np.ndarray,
    output_path: Path,
    max_samples: int,
    seed: int,
) -> None:
    dataframe = pd.DataFrame(
        posterior,
        columns=PARAMETER_NAMES,
    )

    if len(dataframe) > max_samples:
        dataframe = dataframe.sample(
            n=max_samples,
            random_state=seed,
        )

    sigma_values = (
        10.0
        ** dataframe[
            "log10_sigma_strain"
        ]
    )

    dataframe = dataframe.drop(
        columns=[
            "log10_sigma_strain"
        ]
    )

    dataframe[
        "sigma_strain"
    ] = sigma_values.to_numpy()

    names = [
        "a",
        "b",
        "h",
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "sigma_strain",
    ]

    labels = {
        "a": r"$a$",
        "b": r"$b$",
        "h": r"$h$",
        "theta_deg": r"$\theta$",
        "x0_prime": r"$x_0'$",
        "y0_prime": r"$y_0'$",
        "sigma_strain": r"$\sigma$",
    }

    n = len(names)

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(15, 15),
    )

    for row in range(n):
        for column in range(n):
            ax = axes[row, column]

            x_values = dataframe[
                names[column]
            ].to_numpy()

            y_values = dataframe[
                names[row]
            ].to_numpy()

            if row == column:

                ax.hist(
                    x_values,
                    bins=40,
                    density=True,
                    alpha=0.70,
                    edgecolor="black",
                    linewidth=0.4,
                )

            else:

                ax.scatter(
                    x_values,
                    y_values,
                    s=5,
                    alpha=0.22,
                    rasterized=True,
                )

            if row == n - 1:
                ax.set_xlabel(
                    labels[
                        names[column]
                    ]
                )
            else:
                ax.set_xticklabels([])

            if column == 0:
                ax.set_ylabel(
                    labels[
                        names[row]
                    ]
                )
            else:
                ax.set_yticklabels([])

            ax.grid(
                True,
                alpha=0.12,
                linestyle="--",
            )

    fig.suptitle(
        "Posterior pairwise relationships",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.985]
    )

    fig.savefig(
        output_path
    )

    plt.close(fig)


# ============================================================================
# Posterior predictive propagation
# ============================================================================


def sample_posterior_draws(
    posterior: np.ndarray,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        seed
    )

    if n_draws <= 0:
        raise ValueError(
            "Number of predictive draws must be positive."
        )

    if posterior.shape[0] <= n_draws:
        indices = np.arange(
            posterior.shape[0]
        )
    else:
        indices = rng.choice(
            posterior.shape[0],
            size=n_draws,
            replace=False,
        )

    return posterior[
        indices
    ]


def propagate_posterior(
    posterior_draws: np.ndarray,
    context: dict,
    observed_columns: Sequence[str],
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Returns:
        parameter_ensemble:
            shape (draws, time, channels)

        total_predictive_ensemble:
            shape (draws, time, channels)

        sigma_draws:
            shape (draws,)
    """

    n_draws = (
        posterior_draws.shape[0]
    )

    n_time = (
        context["time"].size
    )

    n_channels = len(
        observed_columns
    )

    parameter_ensemble = np.zeros(
        (
            n_draws,
            n_time,
            n_channels,
        ),
        dtype=float,
    )

    total_predictive_ensemble = np.zeros_like(
        parameter_ensemble
    )

    sigma_draws = np.zeros(
        n_draws,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    for index, sample in enumerate(
        posterior_draws
    ):

        prediction_df = (
            forward_from_sample(
                sample,
                context,
            )
        )

        try:
            parameter_curve = (
                prediction_df[
                    list(observed_columns)
                ].to_numpy(
                    dtype=float
                )
            )
        except KeyError as exc:
            raise RuntimeError(
                "Forward-model output is missing one or more canonical "
                "strain columns."
            ) from exc

        sigma = float(
            10.0
            ** sample[
                PARAMETER_NAMES.index(
                    "log10_sigma_strain"
                )
            ]
        )

        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=parameter_curve.shape,
        )

        parameter_ensemble[
            index
        ] = parameter_curve

        total_predictive_ensemble[
            index
        ] = (
            parameter_curve
            + noise
        )

        sigma_draws[
            index
        ] = sigma

    return (
        parameter_ensemble,
        total_predictive_ensemble,
        sigma_draws,
    )


# ============================================================================
# Station/component metrics
# ============================================================================


def predictive_metrics(
    observed: pd.DataFrame,
    observed_columns: Sequence[str],
    parameter_ensemble: np.ndarray,
    total_predictive_ensemble: np.ndarray,
    map_prediction: pd.DataFrame,
    stations: Sequence[str],
) -> pd.DataFrame:
    lower_parameter = np.percentile(
        parameter_ensemble,
        2.5,
        axis=0,
    )

    upper_parameter = np.percentile(
        parameter_ensemble,
        97.5,
        axis=0,
    )

    lower_total = np.percentile(
        total_predictive_ensemble,
        2.5,
        axis=0,
    )

    upper_total = np.percentile(
        total_predictive_ensemble,
        97.5,
        axis=0,
    )

    rows = []

    time_count = len(
        observed
    )

    for component_index, component in enumerate(
        COMPONENTS
    ):
        for station_index, station in enumerate(
            stations
        ):

            channel_index = (
                component_index
                * len(stations)
                + station_index
            )

            column = observed_columns[
                channel_index
            ]

            obs = observed[
                column
            ].to_numpy(
                dtype=float
            )

            parameter_mean = np.mean(
                parameter_ensemble[
                    :,
                    :,
                    channel_index,
                ],
                axis=0,
            )

            total_mean = np.mean(
                total_predictive_ensemble[
                    :,
                    :,
                    channel_index,
                ],
                axis=0,
            )

            map_curve = (
                map_prediction[
                    column
                ].to_numpy(
                    dtype=float
                )
            )

            rows.append(
                {
                    "station": station,
                    "component": component,
                    "channel": column,
                    "n_time": time_count,

                    "map_rmse_nstrain": rmse(
                        obs,
                        map_curve,
                    ),

                    "map_mae_nstrain": mae(
                        obs,
                        map_curve,
                    ),

                    "map_bias_nstrain": bias(
                        obs,
                        map_curve,
                    ),

                    "map_r2": r_squared(
                        obs,
                        map_curve,
                    ),

                    "posterior_mean_rmse_nstrain": rmse(
                        obs,
                        parameter_mean,
                    ),

                    "posterior_mean_mae_nstrain": mae(
                        obs,
                        parameter_mean,
                    ),

                    "posterior_mean_bias_nstrain": bias(
                        obs,
                        parameter_mean,
                    ),

                    "total_predictive_mean_rmse_nstrain": rmse(
                        obs,
                        total_mean,
                    ),

                    "parameter_95pct_coverage": coverage(
                        obs,
                        lower_parameter[
                            :,
                            channel_index,
                        ],
                        upper_parameter[
                            :,
                            channel_index,
                        ],
                    ),

                    "total_95pct_coverage": coverage(
                        obs,
                        lower_total[
                            :,
                            channel_index,
                        ],
                        upper_total[
                            :,
                            channel_index,
                        ],
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# High-quality station figures
# ============================================================================


def station_plot(
    station: str,
    observed: pd.DataFrame,
    observed_columns: Sequence[str],
    time: np.ndarray,
    parameter_ensemble: np.ndarray,
    total_predictive_ensemble: np.ndarray,
    map_prediction: pd.DataFrame,
    stations: Sequence[str],
    output_path: Path,
) -> None:
    station_index = stations.index(
        station
    )

    fig, ax = plt.subplots(
        figsize=(13.5, 8.0),
    )

    parameter_lower = np.percentile(
        parameter_ensemble,
        2.5,
        axis=0,
    )

    parameter_upper = np.percentile(
        parameter_ensemble,
        97.5,
        axis=0,
    )

    total_lower = np.percentile(
        total_predictive_ensemble,
        2.5,
        axis=0,
    )

    total_upper = np.percentile(
        total_predictive_ensemble,
        97.5,
        axis=0,
    )

    component_handles = []

    for component_index, component in enumerate(
        COMPONENTS
    ):
        color = COMPONENT_COLORS[
            component
        ]

        channel_index = (
            component_index
            * len(stations)
            + station_index
        )

        column = observed_columns[
            channel_index
        ]

        observed_values = (
            observed[
                column
            ].to_numpy(
                dtype=float
            )
        )

        map_values = (
            map_prediction[
                column
            ].to_numpy(
                dtype=float
            )
        )

        # Total predictive uncertainty: light outer envelope.
        ax.fill_between(
            time,
            total_lower[
                :,
                channel_index,
            ],
            total_upper[
                :,
                channel_index,
            ],
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )

        # Parameter-only uncertainty: darker inner envelope.
        ax.fill_between(
            time,
            parameter_lower[
                :,
                channel_index,
            ],
            parameter_upper[
                :,
                channel_index,
            ],
            color=color,
            alpha=0.28,
            linewidth=0,
            zorder=2,
        )

        # MAP analytical forward-model prediction.
        ax.plot(
            time,
            map_values,
            color=color,
            linewidth=2.4,
            linestyle="-",
            zorder=4,
        )

        # Observed dataset.
        ax.scatter(
            time,
            observed_values,
            s=18,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            alpha=0.95,
            zorder=5,
        )

        component_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.4,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markersize=5.5,
                label=(
                    f"{COMPONENT_LABELS[component]} "
                    f"(observed + MAP)"
                ),
            )
        )

    uncertainty_handles = [
        Patch(
            facecolor="0.55",
            alpha=0.28,
            edgecolor="none",
            label=PARAM_BAND_LABEL,
        ),
        Patch(
            facecolor="0.55",
            alpha=0.10,
            edgecolor="none",
            label=TOTAL_BAND_LABEL,
        ),
    ]

    handles = (
        component_handles
        + uncertainty_handles
    )

    labels = [
        handle.get_label()
        for handle in handles
    ]

    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.17,
        ),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.8,
        handletextpad=0.7,
        borderaxespad=0.0,
    )

    for text in legend.get_texts():
        text.set_fontsize(
            10.5
        )

    ax.set_title(
        f"{station}: observed strain and Bayesian posterior predictive uncertainty"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Strain (nε)"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.20,
    )

    ax.tick_params(
        direction="out",
        length=5,
    )

    ax.margins(
        x=0.015,
        y=0.08,
    )

    fig.subplots_adjust(
        bottom=0.25,
        left=0.09,
        right=0.98,
        top=0.90,
    )

    fig.savefig(
        output_path
    )

    # Also provide a vector PDF for manuscripts/posters.
    pdf_path = output_path.with_suffix(
        ".pdf"
    )

    fig.savefig(
        pdf_path,
    )

    plt.close(fig)


# ============================================================================
# Residual diagnostics
# ============================================================================


def station_residual_plot(
    station: str,
    observed: pd.DataFrame,
    observed_columns: Sequence[str],
    time: np.ndarray,
    map_prediction: pd.DataFrame,
    sigma_map: float,
    stations: Sequence[str],
    output_path: Path,
) -> None:
    station_index = stations.index(
        station
    )

    fig, axes = plt.subplots(
        len(COMPONENTS),
        1,
        figsize=(12, 10),
        sharex=True,
    )

    for component_index, component in enumerate(
        COMPONENTS
    ):
        ax = axes[
            component_index
        ]

        channel_index = (
            component_index
            * len(stations)
            + station_index
        )

        column = observed_columns[
            channel_index
        ]

        observed_values = (
            observed[
                column
            ].to_numpy(
                dtype=float
            )
        )

        map_values = (
            map_prediction[
                column
            ].to_numpy(
                dtype=float
            )
        )

        residual = (
            observed_values
            - map_values
        )

        color = COMPONENT_COLORS[
            component
        ]

        ax.plot(
            time,
            residual,
            color=color,
            linewidth=1.6,
        )

        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

        ax.axhline(
            sigma_map,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
        )

        ax.axhline(
            -sigma_map,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
        )

        ax.set_ylabel(
            f"{COMPONENT_LABELS[component]} residual\n(nε)"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.18,
        )

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        f"{station}: observed minus MAP analytical prediction",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0.01, 1, 0.98]
    )

    fig.savefig(
        output_path
    )

    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    configure_publication_style()

    run_dir = (
        args.run_dir
        .resolve()
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "postprocessed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run = load_run(
        run_dir
    )

    convergence = convergence_status(
        run
    )

    print(
        "=" * 78
    )
    print(
        "AVANT 1-DARCY BAYESIAN POST-PROCESSING"
    )
    print(
        "=" * 78
    )

    print(
        f"\nRun directory:\n    {run_dir}"
    )

    print(
        "\nConvergence status:"
    )

    print(
        f"    {convergence['status'].upper()}"
    )

    print(
        f"    R-hat threshold = "
        f"{convergence['rhat_threshold']}"
    )

    if convergence[
        "latest_rhat"
    ] is not None:
        print(
            "    latest R-hat = "
            + np.array2string(
                np.asarray(
                    convergence[
                        "latest_rhat"
                    ],
                    dtype=float,
                ),
                precision=4,
                separator=", ",
            )
        )

    if (
        args.require_converged
        and not convergence["converged"]
    ):
        raise RuntimeError(
            "The requested run has not converged. "
            "Re-run without --require-converged for diagnostic "
            "post-processing, or complete a converged production run."
        )

    if not convergence[
        "converged"
    ]:
        print(
            "\n[WARNING] This run is NOT CONVERGED."
        )
        print(
            "[WARNING] The generated posterior summaries and predictive "
            "plots are diagnostic only and must not be presented as final "
            "Bayesian inference."
        )

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    stations_df = load_station_metadata(
        run
    )

    station_names = (
        stations_df[
            "station"
        ].tolist()
    )

    observed, observed_columns = (
        load_observed_dataset(
            run,
            stations_df,
        )
    )

    context = build_forward_context(
        run,
        stations_df,
        observed,
    )

    chains = run[
        "chains"
    ]

    logps = run[
        "logps"
    ]

    posterior, post_logps, burn_in = (
        apply_burn_in(
            chains,
            logps,
            args.burn_in_fraction,
        )
    )

    map_vector, map_index = (
        map_sample(
            posterior,
            post_logps,
        )
    )

    map_prediction = (
        forward_from_sample(
            map_vector,
            context,
        )
    )

    sigma_map = float(
        10.0
        ** map_vector[
            PARAMETER_NAMES.index(
                "log10_sigma_strain"
            )
        ]
    )

    print(
        f"\nPosterior samples after burn-in: "
        f"{posterior.shape[0]}"
    )

    print(
        f"Burn-in iterations per chain: "
        f"{burn_in}"
    )

    print(
        f"MAP posterior draw index: "
        f"{map_index}"
    )

    print(
        f"MAP sigma_strain: "
        f"{sigma_map:.6g} nstrain"
    )

    # ------------------------------------------------------------------
    # Save convergence summary
    # ------------------------------------------------------------------

    convergence_summary = {
        **convergence,
        "burn_in_fraction": float(
            args.burn_in_fraction
        ),
        "burn_in_iterations_per_chain": int(
            burn_in
        ),
        "n_chains": int(
            chains.shape[0]
        ),
        "n_iterations_per_chain": int(
            chains.shape[1]
        ),
        "n_post_burn_samples": int(
            posterior.shape[0]
        ),
    }

    save_json(
        output_dir
        / "convergence_summary.json",
        convergence_summary,
    )

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    summary_df = (
        posterior_summary_dataframe(
            posterior
        )
    )

    summary_df.to_csv(
        output_dir
        / "posterior_parameter_summary.csv",
        index=False,
    )

    posterior_correlation = (
        posterior_correlation_dataframe(
            posterior
        )
    )

    posterior_correlation.to_csv(
        output_dir
        / "posterior_correlation_matrix.csv"
    )

    summary_json = {
        "run_status": convergence[
            "status"
        ],
        "burn_in_fraction": float(
            args.burn_in_fraction
        ),
        "n_post_burn_samples": int(
            posterior.shape[0]
        ),
        "map": {
            name: float(
                value
            )
            for name, value in zip(
                PARAMETER_NAMES,
                map_vector,
            )
        },
        "sigma_map_nstrain": sigma_map,
        "posterior_summary": (
            summary_df.to_dict(
                orient="records"
            )
        ),
    }

    save_json(
        output_dir
        / "posterior_parameter_summary.json",
        summary_json,
    )

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    plot_traces(
        chains,
        output_dir
        / "posterior_traces.png",
    )

    plot_posterior_distributions(
        posterior,
        map_vector,
        output_dir
        / "posterior_distributions.png",
    )

    plot_correlation_matrix(
        posterior_correlation,
        output_dir
        / "posterior_correlation_matrix.png",
    )

    plot_pairwise(
        posterior,
        output_dir
        / "posterior_pairwise.png",
        max_samples=args.max_pairwise_samples,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Posterior predictive propagation
    # ------------------------------------------------------------------

    predictive_draws = (
        sample_posterior_draws(
            posterior,
            args.n_predictive_draws,
            args.seed,
        )
    )

    (
        parameter_ensemble,
        total_predictive_ensemble,
        sigma_draws,
    ) = propagate_posterior(
        predictive_draws,
        context,
        observed_columns,
        args.seed,
    )

    # Save raw predictive arrays for downstream analysis.
    np.save(
        output_dir
        / "posterior_predictive_parameter_ensemble.npy",
        parameter_ensemble,
    )

    np.save(
        output_dir
        / "posterior_predictive_total_ensemble.npy",
        total_predictive_ensemble,
    )

    np.save(
        output_dir
        / "posterior_predictive_sigma_draws.npy",
        sigma_draws,
    )

    # ------------------------------------------------------------------
    # Predictive metrics
    # ------------------------------------------------------------------

    metrics_df = predictive_metrics(
        observed,
        observed_columns,
        parameter_ensemble,
        total_predictive_ensemble,
        map_prediction,
        station_names,
    )

    metrics_df.to_csv(
        output_dir
        / "predictive_metrics_by_station_component.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Station figures
    # ------------------------------------------------------------------

    for station in station_names:

        station_output = (
            output_dir
            / f"station_{station}_observed_vs_posterior.png"
        )

        station_plot(
            station,
            observed,
            observed_columns,
            context["time"],
            parameter_ensemble,
            total_predictive_ensemble,
            map_prediction,
            station_names,
            station_output,
        )

        residual_output = (
            output_dir
            / f"station_{station}_residuals.png"
        )

        station_residual_plot(
            station,
            observed,
            observed_columns,
            context["time"],
            map_prediction,
            sigma_map,
            station_names,
            residual_output,
        )

    # ------------------------------------------------------------------
    # Predictive summary
    # ------------------------------------------------------------------

    predictive_summary = {
        "n_predictive_draws": int(
            predictive_draws.shape[0]
        ),
        "sigma_map_nstrain": sigma_map,
        "sigma_posterior": percentile_summary(
            sigma_draws
        ),
        "run_status": convergence[
            "status"
        ],
        "station_names": station_names,
        "components": list(
            COMPONENTS
        ),
    }

    save_json(
        output_dir
        / "posterior_predictive_summary.json",
        predictive_summary,
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    if convergence[
        "converged"
    ]:
        print(
            "POSTERIOR POST-PROCESSING COMPLETE"
        )
    else:
        print(
            "POSTERIOR DIAGNOSTIC POST-PROCESSING COMPLETE - NOT CONVERGED"
        )

    print(
        "=" * 78
    )

    print(
        f"\nOutput directory:\n    {output_dir}"
    )

    print(
        "\nGenerated:"
    )

    print(
        "    posterior_parameter_summary.csv"
    )

    print(
        "    posterior_parameter_summary.json"
    )

    print(
        "    posterior_correlation_matrix.csv"
    )

    print(
        "    convergence_summary.json"
    )

    print(
        "    posterior_traces.png"
    )

    print(
        "    posterior_distributions.png"
    )

    print(
        "    posterior_correlation_matrix.png"
    )

    print(
        "    posterior_pairwise.png"
    )

    print(
        "    station_S01 ... station_S08 observed-vs-posterior figures"
    )

    print(
        "    station_S01 ... station_S08 residual figures"
    )

    print(
        "    posterior predictive ensemble arrays"
    )

    print(
        "\nImportant:"
    )

    if convergence[
        "converged"
    ]:
        print(
            "    This posterior run is marked CONVERGED."
        )
    else:
        print(
            "    This posterior run is marked NOT CONVERGED."
        )
        print(
            "    Treat all posterior and predictive outputs as diagnostic."
        )


if __name__ == "__main__":
    main()
