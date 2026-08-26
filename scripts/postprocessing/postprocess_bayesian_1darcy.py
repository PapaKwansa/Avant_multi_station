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
    "log10_sigma_strain": r"$\log_{10}\sigma_{\mathrm{strain}}$",
    "sigma_strain": r"$\sigma_{\mathrm{strain}}$ (n$\varepsilon$)",
}

PLOT_PARAMETER_NAMES = (
    "a",
    "b",
    "theta_deg",
    "x0_prime",
    "y0_prime",
    "sigma_strain",
)


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

    parameterization = manifest.get(
        "parameterization",
        {},
    )

    manifest_parameter_names = tuple(
        parameterization.get(
            "canonical_parameter_names",
         [],
        )
    )

    if manifest_parameter_names != PARAMETER_NAMES:
        raise RuntimeError(
            "Run manifest canonical parameter ordering does not match "
            "the AVANT post-processing contract.\n"
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
    """Return correlations for scientifically independent interpreted variables."""
    dataframe = pd.DataFrame({
        "a": posterior[:, PARAMETER_NAMES.index("a")],
        "b": posterior[:, PARAMETER_NAMES.index("b")],
        "theta_deg": posterior[:, PARAMETER_NAMES.index("theta_deg")],
        "x0_prime": posterior[:, PARAMETER_NAMES.index("x0_prime")],
        "y0_prime": posterior[:, PARAMETER_NAMES.index("y0_prime")],
        "sigma_strain": 10.0 ** posterior[:, PARAMETER_NAMES.index("log10_sigma_strain")],
    })
    return dataframe.corr()


# ============================================================================
# Publication-quality parameter plots
# ============================================================================


def configure_publication_style() -> None:
    """Configure a clean, bold, symposium/manuscript plotting style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13.5,
        "axes.labelsize": 15.0,
        "axes.titlesize": 16.0,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 11.5,
        "legend.frameon": False,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def save_figure(fig, output_path: Path) -> None:
    """Save both high-resolution PNG and vector PDF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    fig.savefig(output_path.with_suffix(".pdf"))


def _physical_series_from_chains(chains: np.ndarray, name: str) -> np.ndarray:
    """Return a (nchains, niterations) physical parameter series."""
    index = PARAMETER_NAMES.index(name)
    if name == "log10_sigma_strain":
        return 10.0 ** chains[:, :, index]
    return chains[:, :, index]


def _physical_values(posterior: np.ndarray, name: str) -> np.ndarray:
    index = PARAMETER_NAMES.index(name)
    if name == "log10_sigma_strain":
        return 10.0 ** posterior[:, index]
    return posterior[:, index]


def plot_traces(
    chains: np.ndarray,
    output_path: Path,
) -> None:
    """Plot traces for the six scientifically interpreted parameters."""
    n_chains, n_iterations, _ = chains.shape
    names = (
        "a",
        "b",
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "log10_sigma_strain",
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.5, 11.0),
        sharex=True,
    )
    axes = axes.ravel()

    for panel_index, name in enumerate(names):
        ax = axes[panel_index]
        values = _physical_series_from_chains(chains, name)

        for chain_index in range(n_chains):
            ax.plot(
                np.arange(n_iterations),
                values[chain_index],
                linewidth=1.0,
                alpha=0.75,
            )

        label = PARAMETER_LABELS[
            "sigma_strain" if name == "log10_sigma_strain" else name
        ]
        ax.set_ylabel(label, fontweight="bold")
        ax.grid(True, alpha=0.16, linestyle="--")
        ax.tick_params(direction="out")
        ax.set_title(
            label,
            fontweight="bold",
            pad=8,
        )

    axes[-2].set_xlabel("Iteration", fontweight="bold")
    axes[-1].set_xlabel("Iteration", fontweight="bold")

    handles = [
        Line2D([0], [0], linewidth=1.2, label=f"Chain {i + 1}")
        for i in range(n_chains)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(n_chains, 4),
        frameon=False,
    )

    fig.suptitle(
        "Bayesian posterior traces",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.93,
        hspace=0.34,
        wspace=0.22,
    )
    save_figure(fig, output_path)
    plt.close(fig)


def plot_posterior_distributions(
    posterior: np.ndarray,
    map_sample_vector: np.ndarray,
    output_path: Path,
) -> None:
    """
    Create the publication/symposium-style 3x2 posterior distribution figure.

    Statistical markers:
        mean   = red dashed
        median = green dash-dot
        MAP    = purple dotted

    The six plotted physical quantities are:
        a
        b
        theta
        x0'
        y0'
        sigma_strain
    """

    names = PLOT_PARAMETER_NAMES

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.8, 8.8),
        squeeze=False,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------
    # Statistical marker colors
    # ------------------------------------------------------------------

    mean_color = "#d62728"      # red
    median_color = "#2ca02c"    # green
    map_color = "#9467bd"       # purple

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    legend_handles = [
        Patch(
            facecolor="0.65",
            edgecolor="none",
            alpha=0.18,
            label="95% credible interval",
        ),
        Line2D(
            [0],
            [0],
            color=mean_color,
            linestyle="--",
            linewidth=2.2,
            label="Posterior mean",
        ),
        Line2D(
            [0],
            [0],
            color=median_color,
            linestyle="-.",
            linewidth=2.2,
            label="Posterior median",
        ),
        Line2D(
            [0],
            [0],
            color=map_color,
            linestyle=":",
            linewidth=2.7,
            label="MAP",
        ),
    ]

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    for panel_index, name in enumerate(names):

        ax = axes[panel_index]

        # --------------------------------------------------------------
        # Physical posterior values
        # --------------------------------------------------------------

        if name == "sigma_strain":

            values = _physical_values(
                posterior,
                "log10_sigma_strain",
            )

            map_value = float(
                10.0
                ** map_sample_vector[
                    PARAMETER_NAMES.index(
                        "log10_sigma_strain"
                    )
                ]
            )

            xlabel = PARAMETER_LABELS[
                "sigma_strain"
            ]

        else:

            values = _physical_values(
                posterior,
                name,
            )

            map_value = float(
                map_sample_vector[
                    PARAMETER_NAMES.index(name)
                ]
            )

            xlabel = PARAMETER_LABELS[name]

        mean_value = float(
            np.mean(values)
        )

        median_value = float(
            np.median(values)
        )

        q025, q975 = np.percentile(
            values,
            [2.5, 97.5],
        )

        # --------------------------------------------------------------
        # Histogram
        # --------------------------------------------------------------

        ax.hist(
            values,
            bins=35,
            density=True,
            alpha=0.72,
            edgecolor="black",
            linewidth=0.65,
            zorder=2,
        )

        # --------------------------------------------------------------
        # 95% credible interval
        # --------------------------------------------------------------

        ax.axvspan(
            q025,
            q975,
            facecolor="0.65",
            alpha=0.16,
            edgecolor="none",
            zorder=0,
        )

        # --------------------------------------------------------------
        # Posterior mean
        # --------------------------------------------------------------

        ax.axvline(
            mean_value,
            color=mean_color,
            linestyle="--",
            linewidth=2.2,
            zorder=5,
        )

        # --------------------------------------------------------------
        # Posterior median
        # --------------------------------------------------------------

        ax.axvline(
            median_value,
            color=median_color,
            linestyle="-.",
            linewidth=2.2,
            zorder=5,
        )

        # --------------------------------------------------------------
        # MAP
        # --------------------------------------------------------------

        ax.axvline(
            map_value,
            color=map_color,
            linestyle=":",
            linewidth=2.7,
            zorder=6,
        )

        # --------------------------------------------------------------
        # Labels and styling
        # --------------------------------------------------------------

        ax.set_xlabel(
            xlabel,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Posterior density",
            fontweight="bold",
        )

        ax.set_title(
            xlabel,
            fontweight="bold",
            fontsize=17,
            pad=8,
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.16,
            linestyle="--",
        )

        ax.tick_params(
            direction="out",
            width=1.3,
            length=5.5,
        )

        # Make all four borders visibly strong.
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    fig.suptitle(
        "Posterior distributions of inferred AVANT parameters",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )

    # ------------------------------------------------------------------
    # Shared legend below all six panels
    # ------------------------------------------------------------------

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        columnspacing=2.0,
        handlelength=3.0,
        handletextpad=0.7,
        fontsize=11.5,
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.13,
        top=0.90,
        hspace=0.34,
        wspace=0.25,
    )

    save_figure(
        fig,
        output_path,
    )

    plt.close(fig)


def plot_correlation_matrix(
    correlation: pd.DataFrame,
    output_path: Path,
) -> None:
    matrix = correlation.to_numpy(dtype=float)
    labels = [
        r"$a$",
        r"$b$",
        r"$\theta$",
        r"$x_0'$",
        r"$y_0'$",
        r"$\sigma$",
    ]

    fig, ax = plt.subplots(
        figsize=(9.8, 8.6),
    )

    image = ax.imshow(
        matrix,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=13, rotation=30, ha="right", fontweight="bold")
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Posterior correlation",
        fontsize=13,
        fontweight="bold",
    )
    colorbar.ax.tick_params(labelsize=11)

    ax.set_title(
        "Posterior parameter correlations",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_pairwise(
    posterior: np.ndarray,
    output_path: Path,
    max_samples: int,
    seed: int,
) -> None:
    dataframe = pd.DataFrame({
        "a": posterior[:, PARAMETER_NAMES.index("a")],
        "b": posterior[:, PARAMETER_NAMES.index("b")],
        "theta_deg": posterior[:, PARAMETER_NAMES.index("theta_deg")],
        "x0_prime": posterior[:, PARAMETER_NAMES.index("x0_prime")],
        "y0_prime": posterior[:, PARAMETER_NAMES.index("y0_prime")],
        "sigma_strain": 10.0 ** posterior[:, PARAMETER_NAMES.index("log10_sigma_strain")],
    })

    if len(dataframe) > max_samples:
        dataframe = dataframe.sample(
            n=max_samples,
            random_state=seed,
        )

    names = [
        "a",
        "b",
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "sigma_strain",
    ]

    labels = {
        "a": r"$a$",
        "b": r"$b$",
        "theta_deg": r"$\theta$",
        "x0_prime": r"$x_0'$",
        "y0_prime": r"$y_0'$",
        "sigma_strain": r"$\sigma$",
    }

    n = len(names)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(16, 16),
    )

    for row in range(n):
        for column in range(n):
            ax = axes[row, column]
            x_values = dataframe[names[column]].to_numpy()
            y_values = dataframe[names[row]].to_numpy()

            if row == column:
                ax.hist(
                    x_values,
                    bins=36,
                    density=True,
                    alpha=0.72,
                    edgecolor="black",
                    linewidth=0.45,
                )
            else:
                ax.scatter(
                    x_values,
                    y_values,
                    s=7,
                    alpha=0.18,
                    rasterized=True,
                )

            if row == n - 1:
                ax.set_xlabel(labels[names[column]], fontweight="bold")
            else:
                ax.set_xticklabels([])

            if column == 0:
                ax.set_ylabel(labels[names[row]], fontweight="bold")
            else:
                ax.set_yticklabels([])

            ax.grid(
                True,
                alpha=0.10,
                linestyle="--",
            )
            ax.tick_params(labelsize=9)

    fig.suptitle(
        "Posterior pairwise relationships",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.055,
        top=0.955,
        wspace=0.05,
        hspace=0.05,
    )
    save_figure(fig, output_path)
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
    """
    Create a publication-quality 2x2 station figure.

    Panels:
        top-left     = eXX
        top-right    = eYY
        bottom-left  = eZZ
        bottom-right = eXY

    Visual semantics:
        open circles          = observed data
        solid colored line    = MAP analytical prediction
        darker band           = 95% parameter uncertainty
        lighter band          = 95% total predictive uncertainty

    Component colors identify the strain component.
    Band opacity identifies uncertainty type.
    """

    station_index = stations.index(
        station
    )

    # ------------------------------------------------------------------
    # Posterior uncertainty envelopes
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 2x2 layout
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.0),
        sharex=True,
        squeeze=False,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------
    # Plot each strain component in its own panel
    # ------------------------------------------------------------------

    for component_index, component in enumerate(
        COMPONENTS
    ):

        ax = axes[component_index]

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

        # --------------------------------------------------------------
        # Total predictive uncertainty
        # Lighter outer band.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Parameter uncertainty
        # Darker inner band.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # MAP analytical prediction
        # --------------------------------------------------------------

        ax.plot(
            time,
            map_values,
            color=color,
            linewidth=2.6,
            linestyle="-",
            zorder=4,
        )

        # --------------------------------------------------------------
        # Observed data
        # --------------------------------------------------------------

        ax.scatter(
            time,
            observed_values,
            s=24,
            facecolor="white",
            edgecolor=color,
            linewidth=1.1,
            alpha=0.95,
            zorder=5,
        )

        # --------------------------------------------------------------
        # Panel title
        # --------------------------------------------------------------

        ax.set_title(
            COMPONENT_LABELS[
                component
            ],
            fontsize=17,
            fontweight="bold",
            pad=8,
        )

        ax.set_ylabel(
            "Strain (nε)",
            fontsize=14,
            fontweight="bold",
        )

        # Only bottom row gets x-axis labels.
        if component_index >= 2:
            ax.set_xlabel(
                "Time (s)",
                fontsize=14,
                fontweight="bold",
            )

        # --------------------------------------------------------------
        # Styling
        # --------------------------------------------------------------

        ax.grid(
            True,
            linestyle="--",
            alpha=0.18,
        )

        ax.tick_params(
            direction="out",
            width=1.25,
            length=5,
            labelsize=11.5,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.margins(
            x=0.015,
            y=0.08,
        )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    semantic_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6.5,
            markeredgewidth=1.1,
            label="Observed",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.6,
            linestyle="-",
            label="MAP analytical prediction",
        ),
        Patch(
            facecolor="0.45",
            alpha=0.28,
            edgecolor="none",
            label="95% parameter uncertainty",
        ),
        Patch(
            facecolor="0.45",
            alpha=0.10,
            edgecolor="none",
            label="95% total predictive uncertainty",
        ),
    ]

    component_handles = [
        Line2D(
            [0],
            [0],
            color=COMPONENT_COLORS[
                component
            ],
            linewidth=2.6,
            label=COMPONENT_LABELS[
                component
            ],
        )
        for component in COMPONENTS
    ]

    # First legend: semantics.
    semantic_legend = fig.legend(
        handles=semantic_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.048),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=2.8,
        columnspacing=1.7,
    )

    # Second legend: component colors.
    component_legend = fig.legend(
        handles=component_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=2.8,
        columnspacing=1.7,
    )

    fig.add_artist(
        semantic_legend
    )

    # ------------------------------------------------------------------
    # Overall station title
    # ------------------------------------------------------------------

    fig.suptitle(
        f"{station}: observed strain and Bayesian posterior predictive uncertainty",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.16,
        top=0.90,
        hspace=0.28,
        wspace=0.20,
    )

    save_figure(
        fig,
        output_path,
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

    # Validate constrained-run metadata when available.
    parameterization = run["manifest"].get(
        "parameterization",
        {},
    )

    fixed_parameters = parameterization.get(
        "fixed_parameters",
        {},
    )

    if "h" in fixed_parameters:
        fixed_h = float(fixed_parameters["h"])
        h_index = PARAMETER_NAMES.index("h")

        if not np.allclose(
            chains[:, :, h_index],
            fixed_h,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "Saved constrained-run posterior contains h values that "
                "do not equal the fixed depth recorded in the run manifest."
            )

    geometric_constraints = parameterization.get(
        "geometric_constraints",
        {},
    )

    if geometric_constraints.get(
        "b_greater_than_a",
        False,
    ):
        a_index = PARAMETER_NAMES.index("a")
        b_index = PARAMETER_NAMES.index("b")

        if not np.all(
            chains[:, :, b_index] > chains[:, :, a_index]
        ):
            raise RuntimeError(
                "Saved constrained-run posterior contains samples "
                "violating the physical constraint b > a."
            )

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
        "    posterior_traces.png / posterior_traces.pdf"
    )

    print(
        "    posterior_distributions.png / posterior_distributions.pdf"
    )

    print(
        "    posterior_correlation_matrix.png / posterior_correlation_matrix.pdf"
    )

    print(
        "    posterior_pairwise.png / posterior_pairwise.pdf"
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
