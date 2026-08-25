#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production PyDREAM runner for the AVANT 1-Darcy transient analytical
Bayesian inversion.

==============================================================================
CANONICAL MCMC STATE
==============================================================================

Physical parameters of interest:

    a
    b
    h
    theta_deg
    x0_prime
    y0_prime

Statistical nuisance parameter:

    log10_sigma_strain

Complete sampled state:

    [
        a,
        b,
        h,
        theta_deg,
        x0_prime,
        y0_prime,
        log10_sigma_strain,
    ]

==============================================================================
CANONICAL 1-DARCY TRANSIENT DATA CONTRACT
==============================================================================

Station metadata:

    datasets/comsol/1darcy/metadata/stations.csv

Expected columns:

    station,x,y,z

Expected stations:

    S01, S02, ..., S08

Observed transient strain:

    datasets/comsol/1darcy/processed/strain.csv

Expected columns:

    time_s

    eXX_S01 ... eXX_S08
    eYY_S01 ... eYY_S08
    eZZ_S01 ... eZZ_S08
    eXY_S01 ... eXY_S08

The older datasets/AVANT_stations.csv file is intentionally ignored.
That file belongs to an older four-station workflow and must never be
silently selected for the current 1-Darcy transient inversion.

==============================================================================
SCIENTIFIC LIKELIHOOD
==============================================================================

The reusable likelihood is implemented in:

    src/avant_model/inversion/bayesian_inversion_multi_station.py

This runner supplies the data and fixed analytical-model parameters to that
likelihood.

==============================================================================
RESPONSIBILITIES
==============================================================================

This script is responsible for:

    1. Loading the canonical 1-Darcy transient station metadata.
    2. Loading and validating the transient strain dataset.
    3. Loading fixed analytical-model parameters.
    4. Defining the PyDREAM priors.
    5. Running the PyDREAM sampler.
    6. Monitoring Gelman-Rubin convergence.
    7. Saving posterior samples and provenance metadata.

Posterior plotting and posterior-predictive analysis are intentionally
implemented later in a separate post-processing module.

==============================================================================
MULTIPROCESSING
==============================================================================

The PyDREAM likelihood wrapper is defined at module scope.

This is required for Windows multiprocessing with the "spawn" start method:
nested functions such as main.<locals>.likelihood_wrapper cannot be pickled.
"""

from __future__ import annotations

# ============================================================================
# Numerical thread limits
# ============================================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================================
# Standard library
# ============================================================================

import argparse
import json
import multiprocessing
import platform
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path


# ============================================================================
# Third-party packages
# ============================================================================

import numpy as np
import pandas as pd
from scipy.stats import uniform

from pydream.convergence import Gelman_Rubin
from pydream.core import run_dream
from pydream.parameters import SampledParam


# ============================================================================
# Repository imports
# ============================================================================

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from avant_model.data import (
    multi_stations_input as input_data,
)

from avant_model.inversion.bayesian_inversion_multi_station import (
    PHYSICAL_PARAMETER_NAMES,
    NUISANCE_PARAMETER_NAMES,
    SAMPLED_PARAMETER_NAMES,
    likelihood,
)


# ============================================================================
# Canonical definitions
# ============================================================================

PHYSICAL_NAMES = tuple(
    PHYSICAL_PARAMETER_NAMES
)

NUISANCE_NAMES = tuple(
    NUISANCE_PARAMETER_NAMES
)

PARAM_NAMES = tuple(
    SAMPLED_PARAMETER_NAMES
)

COMPONENTS = (
    "eXX",
    "eYY",
    "eZZ",
    "eXY",
)

EXPECTED_STATIONS = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
)


# ============================================================================
# Canonical 1-Darcy transient data paths
# ============================================================================

STATION_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "metadata"
    / "stations.csv"
)

OBSERVED_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "processed"
    / "strain.csv"
)


# ============================================================================
# Output configuration
# ============================================================================

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "bayesian"
    / "1darcy"
)

DEFAULT_MODEL_NAME = (
    "pydream_1darcy_ab_h_theta_center_sigma"
)


# ============================================================================
# PyDREAM defaults
# ============================================================================

DEFAULT_MAX_ITER = 50000
DEFAULT_BATCH_SIZE = 5000
DEFAULT_NCHAINS = 4
DEFAULT_RHAT_THRESHOLD = 1.10
DEFAULT_SEED = 42


# ============================================================================
# Command-line arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the production PyDREAM Bayesian inversion "
            "for the canonical AVANT 1-Darcy transient dataset."
        )
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Unique model name used for the output directory "
            "and PyDREAM history."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "results/bayesian/1darcy/<model-name>"
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Maximum PyDREAM iterations per chain.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of iterations in each PyDREAM batch.",
    )

    parser.add_argument(
        "--nchains",
        type=int,
        default=DEFAULT_NCHAINS,
        help="Number of PyDREAM chains.",
    )

    parser.add_argument(
        "--rhat-threshold",
        type=float,
        default=DEFAULT_RHAT_THRESHOLD,
        help="Gelman-Rubin convergence threshold.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="NumPy random seed.",
    )

    parser.add_argument(
        "--mp-start",
        choices=(
            "spawn",
            "fork",
            "forkserver",
        ),
        default="spawn",
        help="Multiprocessing start method.",
    )

    return parser.parse_args()


# ============================================================================
# JSON helper
# ============================================================================

def save_json(path, payload):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


# ============================================================================
# Station metadata
# ============================================================================

def load_station_metadata():
    """
    Load the canonical eight-station 1-Darcy metadata.

    No fallback station-file search is permitted.
    """

    if not STATION_FILE.exists():
        raise FileNotFoundError(
            "Canonical 1-Darcy station metadata was not found:\n"
            f"    {STATION_FILE}"
        )

    stations = pd.read_csv(
        STATION_FILE
    )

    stations.columns = [
        str(column).strip()
        for column in stations.columns
    ]

    required_columns = {
        "station",
        "x",
        "y",
        "z",
    }

    missing = (
        required_columns
        - set(stations.columns)
    )

    if missing:
        raise RuntimeError(
            "Canonical station metadata is missing required columns: "
            + ", ".join(sorted(missing))
            + f"\nFound columns: {list(stations.columns)}"
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

    for column in (
        "x",
        "y",
        "z",
    ):
        stations[column] = pd.to_numeric(
            stations[column],
            errors="raise",
        )

    actual_station_order = tuple(
        stations["station"].tolist()
    )

    if actual_station_order != EXPECTED_STATIONS:
        raise RuntimeError(
            "Unexpected canonical 1-Darcy station metadata.\n"
            f"Expected: {EXPECTED_STATIONS}\n"
            f"Found:    {actual_station_order}"
        )

    if stations["station"].duplicated().any():
        raise RuntimeError(
            "Station names are duplicated."
        )

    coordinates = stations[
        [
            "x",
            "y",
            "z",
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(coordinates)
    ):
        raise RuntimeError(
            "Station metadata contains non-finite coordinates."
        )

    return stations


# ============================================================================
# Observed transient dataset
# ============================================================================

def expected_component_columns():
    """
    Return the exact canonical 32-channel ordering.
    """

    return [
        f"{component}_{station}"
        for component in COMPONENTS
        for station in EXPECTED_STATIONS
    ]


def load_observed_dataset():
    """
    Load the canonical eight-station transient strain dataset.

    No alternate observed dataset is accepted.
    """

    if not OBSERVED_FILE.exists():
        raise FileNotFoundError(
            "Canonical 1-Darcy transient strain dataset was not found:\n"
            f"    {OBSERVED_FILE}"
        )

    observed = pd.read_csv(
        OBSERVED_FILE
    )

    observed.columns = [
        str(column).strip()
        for column in observed.columns
    ]

    expected_columns = [
        "time_s",
        *expected_component_columns(),
    ]

    missing = [
        column
        for column in expected_columns
        if column not in observed.columns
    ]

    if missing:
        raise RuntimeError(
            "The canonical transient strain dataset "
            "is missing these required columns:\n"
            + "\n".join(
                f"    {column}"
                for column in missing
            )
        )

    unexpected = [
        column
        for column in observed.columns
        if column not in expected_columns
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected columns found in the canonical "
            "transient strain dataset:\n"
            + "\n".join(
                f"    {column}"
                for column in unexpected
            )
        )

    observed = observed.loc[
        :,
        expected_columns,
    ].copy()

    for column in expected_columns:
        observed[column] = pd.to_numeric(
            observed[column],
            errors="raise",
        )

    if observed.empty:
        raise RuntimeError(
            "Canonical transient strain dataset is empty."
        )

    values = observed.to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "Canonical transient strain dataset contains "
            "non-finite values."
        )

    time_values = observed[
        "time_s"
    ].to_numpy(
        dtype=float
    )

    if time_values.size < 2:
        raise RuntimeError(
            "At least two time steps are required."
        )

    if not np.all(
        np.diff(time_values) >= 0.0
    ):
        raise RuntimeError(
            "time_s must be monotonically non-decreasing."
        )

    return observed


# ============================================================================
# Fixed analytical-model inputs
# ============================================================================

def load_fixed_model_inputs(observed_time):
    """
    Load fixed analytical-model inputs.

    The observed dataset supplies the authoritative time vector.
    The existing model configuration is checked against it.
    """

    params = input_data.read_input()

    required = (
        "pmax",
        "E",
        "c_fixed",
        "nu",
        "tpeak",
        "d",
        "alpha",
    )

    missing = [
        name
        for name in required
        if name not in params
    ]

    if missing:
        raise RuntimeError(
            "multi_stations_input.read_input() is missing "
            "required fixed Bayesian parameters:\n"
            + "\n".join(
                f"    {name}"
                for name in missing
            )
        )

    configured_time = np.asarray(
        params.get(
            "time",
            [],
        ),
        dtype=float,
    )

    if configured_time.size:

        if configured_time.shape != observed_time.shape:
            raise RuntimeError(
                "Configured analytical-model time vector and "
                "observed time vector have different lengths.\n"
                f"Configured: {configured_time.shape}\n"
                f"Observed:   {observed_time.shape}"
            )

        if not np.allclose(
            configured_time,
            observed_time,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError(
                "Configured analytical-model time vector does not "
                "match the canonical transient strain dataset."
            )

    return {
        "pmax": float(params["pmax"]),
        "E": float(params["E"]),
        "c": float(params["c_fixed"]),
        "nu": float(params["nu"]),
        "tpeak": float(params["tpeak"]),
        "d": float(params["d"]),
        "alpha": float(params["alpha"]),
        "time": observed_time.copy(),
    }


# ============================================================================
# Priors
# ============================================================================

def make_uniform_prior(
    low,
    high,
    name,
):
    if not (
        np.isfinite(low)
        and np.isfinite(high)
        and high > low
    ):
        raise ValueError(
            f"Invalid prior bounds for {name}: "
            f"[{low}, {high}]"
        )

    return SampledParam(
        uniform,
        loc=float(low),
        scale=float(
            high - low
        ),
    )


def build_priors():
    """
    Current broad Bayesian-development prior set.

    These ranges are retained for the first production refactor.
    We will later move the finalized narrow/current/broad scenarios
    into explicit configuration files.
    """

    bounds = {
        "a": [
            40.0,
            900.0,
        ],
        "b": [
            20.0,
            1000.0,
        ],
        "h": [
            50.0,
            900.0,
        ],
        "theta_deg": [
            -90.0,
            90.0,
        ],
        "x0_prime": [
            -900.0,
            900.0,
        ],
        "y0_prime": [
            -900.0,
            900.0,
        ],
        "log10_sigma_strain": [
            0.0,
            np.log10(2000.0),
        ],
    }

    priors = [
        make_uniform_prior(
            *bounds["a"],
            name="a",
        ),
        make_uniform_prior(
            *bounds["b"],
            name="b",
        ),
        make_uniform_prior(
            *bounds["h"],
            name="h",
        ),
        make_uniform_prior(
            *bounds["theta_deg"],
            name="theta_deg",
        ),
        make_uniform_prior(
            *bounds["x0_prime"],
            name="x0_prime",
        ),
        make_uniform_prior(
            *bounds["y0_prime"],
            name="y0_prime",
        ),
        make_uniform_prior(
            *bounds["log10_sigma_strain"],
            name="log10_sigma_strain",
        ),
    ]

    return priors, bounds


# ============================================================================
# Vector helpers
# ============================================================================

def flatten_observed_vector(
    observed,
    component_columns,
):
    return np.concatenate(
        [
            observed[column].to_numpy(
                dtype=float
            )
            for column in component_columns
        ]
    )


def normalize_sampled_params(
    sampled_params,
):
    array = np.asarray(
        sampled_params
    )

    if array.ndim == 3:
        return [
            np.asarray(
                array[index],
                dtype=float,
            )
            for index in range(
                array.shape[0]
            )
        ]

    if array.ndim == 2:
        return [
            np.asarray(
                array,
                dtype=float,
            )
        ]

    raise RuntimeError(
        "Unexpected PyDREAM sampled-parameter shape: "
        f"{array.shape}"
    )


# ============================================================================
# TOP-LEVEL PICKLEABLE PYDREAM LIKELIHOOD
# ============================================================================

def pydream_likelihood(
    sampled_parameters,
    *,
    observed_vector,
    time_vector,
    x_prime,
    y_prime,
    z,
    c,
    E,
    nu,
    pmax,
    tpeak,
    d,
    alpha,
    component_columns,
    station_names,
):
    """
    Top-level pickleable likelihood wrapper for PyDREAM.

    IMPORTANT
    ---------
    This function must remain at module scope.

    Windows multiprocessing uses the "spawn" method by default.
    Functions defined inside main() are local objects and cannot be
    pickled for transfer to PyDREAM worker processes.
    """

    try:
        return likelihood(
            sampled_parameters,
            observed_vector,
            time=time_vector,
            x_prime=x_prime,
            y_prime=y_prime,
            z=z,
            c=c,
            E=E,
            nu=nu,
            pmax=pmax,
            tpeak=tpeak,
            d=d,
            alpha=alpha,
            component_cols=component_columns,
            station_names=station_names,
        )

    except Exception as exc:
        print(
            "[ERROR] Likelihood evaluation failed."
        )

        print(
            f"        parameters = {sampled_parameters}"
        )

        print(
            f"        error = {exc}"
        )

        return -np.inf


# ============================================================================
# Manifest
# ============================================================================

def create_run_manifest(
    *,
    args,
    output_dir,
    stations,
    observed,
    fixed,
    prior_bounds,
):
    return {
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "repository_root": str(
            REPO_ROOT
        ),
        "runner": str(
            Path(__file__).resolve()
        ),
        "model_name": args.model_name,
        "data_contract": {
            "station_file": str(
                STATION_FILE
            ),
            "observed_file": str(
                OBSERVED_FILE
            ),
            "station_names": (
                stations[
                    "station"
                ].tolist()
            ),
            "n_stations": int(
                len(stations)
            ),
            "n_time_steps": int(
                len(observed)
            ),
            "n_strain_channels": int(
                len(observed.columns) - 1
            ),
        },
        "parameter_names": list(
            PARAM_NAMES
        ),
        "physical_parameter_names": list(
            PHYSICAL_NAMES
        ),
        "nuisance_parameter_names": list(
            NUISANCE_NAMES
        ),
        "sampler": {
            "name": "PyDREAM",
            "nchains": int(
                args.nchains
            ),
            "max_iter": int(
                args.max_iter
            ),
            "batch_size": int(
                args.batch_size
            ),
            "rhat_threshold": float(
                args.rhat_threshold
            ),
            "seed": int(
                args.seed
            ),
            "mp_start": args.mp_start,
        },
        "fixed_model_inputs": {
            "pmax": fixed["pmax"],
            "E": fixed["E"],
            "c": fixed["c"],
            "nu": fixed["nu"],
            "tpeak": fixed["tpeak"],
            "d": fixed["d"],
            "alpha": fixed["alpha"],
        },
        "priors": prior_bounds,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "output_directory": str(
            output_dir
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate run controls
    # ------------------------------------------------------------------

    if args.max_iter <= 0:
        raise ValueError(
            "--max-iter must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.nchains < 2:
        raise ValueError(
            "--nchains must be at least 2."
        )

    if not (
        1.0
        < args.rhat_threshold
        < 2.0
    ):
        raise ValueError(
            "--rhat-threshold must be between 1 and 2."
        )

    if args.batch_size > args.max_iter:
        args.batch_size = args.max_iter

    np.random.seed(
        args.seed
    )

    # ------------------------------------------------------------------
    # Load canonical 1-Darcy transient inputs
    # ------------------------------------------------------------------

    stations = (
        load_station_metadata()
    )

    observed = (
        load_observed_dataset()
    )

    station_names = (
        stations[
            "station"
        ].to_numpy(
            dtype=str
        )
    )

    # The metadata file uses x,y,z.
    # These are the station coordinates supplied to the analytical
    # forward model as x_prime, y_prime, z.
    x_prime = (
        stations[
            "x"
        ].to_numpy(
            dtype=float
        )
    )

    y_prime = (
        stations[
            "y"
        ].to_numpy(
            dtype=float
        )
    )

    z = (
        stations[
            "z"
        ].to_numpy(
            dtype=float
        )
    )

    component_columns = (
        expected_component_columns()
    )

    observed_vector = (
        flatten_observed_vector(
            observed,
            component_columns,
        )
    )

    observed_time = (
        observed[
            "time_s"
        ].to_numpy(
            dtype=float
        )
    )

    # ------------------------------------------------------------------
    # Fixed analytical inputs
    # ------------------------------------------------------------------

    fixed = (
        load_fixed_model_inputs(
            observed_time
        )
    )

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    priors, prior_bounds = (
        build_priors()
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    if args.output_dir is None:

        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / args.model_name
        )

    else:

        output_dir = (
            args.output_dir
        )

    output_dir = (
        output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print(
        "=" * 78
    )

    print(
        "AVANT 1-DARCY BAYESIAN PARAMETER INFERENCE"
    )

    print(
        "=" * 78
    )

    print(
        "\nCANONICAL TRANSIENT DATA"
    )

    print(
        f"  station metadata:\n"
        f"    {STATION_FILE}"
    )

    print(
        f"  strain data:\n"
        f"    {OBSERVED_FILE}"
    )

    print(
        "\nSTATIONS"
    )

    print(
        f"  count = {len(station_names)}"
    )

    print(
        f"  names = {list(station_names)}"
    )

    print(
        "\nTRANSIENT DATA"
    )

    print(
        f"  time steps = {len(observed)}"
    )

    print(
        f"  strain channels = "
        f"{len(component_columns)}"
    )

    print(
        f"  total observations = "
        f"{observed_vector.size}"
    )

    print(
        "\nPHYSICAL PARAMETERS"
    )

    for name in PHYSICAL_NAMES:
        print(
            f"  {name}"
        )

    print(
        "\nSTATISTICAL NUISANCE PARAMETER"
    )

    print(
        "  log10_sigma_strain"
    )

    print(
        "\nFIXED ANALYTICAL-MODEL INPUTS"
    )

    for name in (
        "pmax",
        "E",
        "c",
        "nu",
        "tpeak",
        "d",
        "alpha",
    ):
        print(
            f"  {name:8s} = "
            f"{fixed[name]}"
        )

    print(
        "\nPRIORS"
    )

    for name in PARAM_NAMES:
        print(
            f"  {name:20s} = "
            f"{prior_bounds[name]}"
        )

    # ------------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------------

    manifest = (
        create_run_manifest(
            args=args,
            output_dir=output_dir,
            stations=stations,
            observed=observed,
            fixed=fixed,
            prior_bounds=prior_bounds,
        )
    )

    save_json(
        output_dir
        / "run_manifest.json",
        manifest,
    )

    stations.to_csv(
        output_dir
        / "station_metadata_used.csv",
        index=False,
    )

    save_json(
        output_dir
        / "component_column_order.json",
        component_columns,
    )

    # ------------------------------------------------------------------
    # Build PICKLEABLE likelihood callable
    # ------------------------------------------------------------------

    pydream_likelihood_fn = partial(
        pydream_likelihood,

        observed_vector=(
            observed_vector
        ),

        time_vector=(
            fixed["time"]
        ),

        x_prime=(
            x_prime
        ),

        y_prime=(
            y_prime
        ),

        z=(
            z
        ),

        c=(
            fixed["c"]
        ),

        E=(
            fixed["E"]
        ),

        nu=(
            fixed["nu"]
        ),

        pmax=(
            fixed["pmax"]
        ),

        tpeak=(
            fixed["tpeak"]
        ),

        d=(
            fixed["d"]
        ),

        alpha=(
            fixed["alpha"]
        ),

        component_columns=(
            component_columns
        ),

        station_names=(
            station_names
        ),
    )

    # ------------------------------------------------------------------
    # Multiprocessing
    # ------------------------------------------------------------------

    mp_context = (
        multiprocessing.get_context(
            args.mp_start
        )
    )

    # ------------------------------------------------------------------
    # Sampling state
    # ------------------------------------------------------------------

    chain_collection = None

    log_probability_collection = []

    total_iterations = 0

    batch_number = 0

    convergence_history = []

    converged = False
    convergence_iteration = None
    final_rhat = None

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PYDREAM SAMPLING"
    )

    print(
        "=" * 78
    )

    # ------------------------------------------------------------------
    # PyDREAM loop
    # ------------------------------------------------------------------

    while (
        total_iterations
        < args.max_iter
    ):

        batch_number += 1

        batch_iterations = min(
            args.batch_size,
            args.max_iter
            - total_iterations,
        )

        print(
            f"\n[INFO] Batch {batch_number}: "
            f"{batch_iterations} iterations"
        )

        start_time = time.time()

        sampled_params, logps = (
            run_dream(
                parameters=priors,

                likelihood=(
                    pydream_likelihood_fn
                ),

                niterations=(
                    batch_iterations
                ),

                nchains=(
                    args.nchains
                ),

                mp_context=(
                    mp_context
                ),

                snooker=True,

                adapt_gamma=True,

                save_history=True,

                model_name=(
                    f"{args.model_name}"
                    f"_batch{batch_number:03d}"
                ),
            )
        )

        batch_chains = (
            normalize_sampled_params(
                sampled_params
            )
        )

        if len(batch_chains) != (
            args.nchains
        ):
            raise RuntimeError(
                "PyDREAM returned "
                f"{len(batch_chains)} chains, "
                f"expected {args.nchains}."
            )

        if chain_collection is None:

            chain_collection = [
                chain.copy()
                for chain in batch_chains
            ]

        else:

            for chain_index in range(
                args.nchains
            ):

                chain_collection[
                    chain_index
                ] = np.vstack(
                    [
                        chain_collection[
                            chain_index
                        ],
                        batch_chains[
                            chain_index
                        ],
                    ]
                )

        log_probability_collection.append(
            np.asarray(
                logps,
                dtype=float,
            ).ravel()
        )

        total_iterations += (
            batch_iterations
        )

        print(
            f"[INFO] Batch finished in "
            f"{time.time() - start_time:.2f} s"
        )

        # --------------------------------------------------------------
        # Gelman-Rubin
        # --------------------------------------------------------------

        try:

            rhat = np.asarray(
                Gelman_Rubin(
                    [
                        np.asarray(
                            chain,
                            dtype=float,
                        )
                        for chain in (
                            chain_collection
                        )
                    ]
                ),
                dtype=float,
            )

            print(
                "[INFO] R-hat =",
                np.round(
                    rhat,
                    4,
                ),
            )

            convergence_entry = {
                "batch": int(
                    batch_number
                ),
                "total_iterations": int(
                    total_iterations
                ),
                "rhat": [
                    float(value)
                    for value in rhat
                ],
                "all_below_threshold": bool(
                    np.all(
                        rhat
                        < args.rhat_threshold
                    )
                ),
            }

            convergence_history.append(
                convergence_entry
            )

            if convergence_entry[
                "all_below_threshold"
            ]:

                converged = True
                convergence_iteration = total_iterations
                final_rhat = rhat.copy()

                print(
                    "[INFO] Gelman-Rubin convergence criterion reached."
                )
                print(
                    f"[INFO] Converged after {total_iterations} iterations per chain."
                )

                break

        except Exception as exc:

            print(
                "[WARN] Gelman-Rubin calculation failed:"
            )

            print(
                f"       {exc}"
            )

            convergence_history.append(
                {
                    "batch": int(
                        batch_number
                    ),
                    "total_iterations": int(
                        total_iterations
                    ),
                    "rhat": None,
                    "all_below_threshold": False,
                    "error": str(exc),
                }
            )

    # ------------------------------------------------------------------
    # Final chain validation
    # ------------------------------------------------------------------

    if chain_collection is None:

        raise RuntimeError(
            "No posterior chains were returned."
        )

    if len(chain_collection) != (
        args.nchains
    ):

        raise RuntimeError(
            "Final chain count does not match "
            "requested nchains."
        )

    posterior_chains = np.stack(
        chain_collection,
        axis=0,
    )

    posterior_logps = (
        np.concatenate(
            log_probability_collection
        )
    )

    flat_samples = (
        posterior_chains.reshape(
            -1,
            posterior_chains.shape[-1],
        )
    )

    if flat_samples.shape[1] != (
        len(PARAM_NAMES)
    ):

        raise RuntimeError(
            "Posterior dimensionality mismatch.\n"
            f"Expected: {len(PARAM_NAMES)}\n"
            f"Received: {flat_samples.shape[1]}\n"
            f"Parameters: {PARAM_NAMES}"
        )

    if posterior_logps.size != (
        flat_samples.shape[0]
    ):

        raise RuntimeError(
            "Posterior log-probability count does not "
            "match flattened posterior samples."
        )

    # ------------------------------------------------------------------
    # Save arrays
    # ------------------------------------------------------------------

    np.save(
        output_dir
        / "posterior_chains.npy",
        posterior_chains,
    )

    np.save(
        output_dir
        / "posterior_logps.npy",
        posterior_logps,
    )

    np.save(
        output_dir
        / "posterior_samples_flat.npy",
        flat_samples,
    )

    save_json(
        output_dir
        / "convergence_history.json",
        convergence_history,
    )

    # ------------------------------------------------------------------
    # MAP estimate
    # ------------------------------------------------------------------

    map_index = int(
        np.argmax(
            posterior_logps
        )
    )

    map_vector = (
        flat_samples[
            map_index
        ]
    )

    map_record = {
        name: float(value)
        for name, value in zip(
            PARAM_NAMES,
            map_vector,
        )
    }

    map_record[
        "sigma_strain"
    ] = float(
        10.0
        ** map_record[
            "log10_sigma_strain"
        ]
    )

    # ------------------------------------------------------------------
    # Basic posterior moments
    # ------------------------------------------------------------------

    posterior_mean = {
        name: float(
            np.mean(
                flat_samples[
                    :,
                    index,
                ]
            )
        )
        for index, name in enumerate(
            PARAM_NAMES
        )
    }

    posterior_std = {
        name: float(
            np.std(
                flat_samples[
                    :,
                    index,
                ],
                ddof=1,
            )
        )
        for index, name in enumerate(
            PARAM_NAMES
        )
    }

    sigma_index = PARAM_NAMES.index(
        "log10_sigma_strain"
    )

    sigma_samples = (
        10.0
        ** flat_samples[
            :,
            sigma_index,
        ]
    )

    posterior_mean[
        "sigma_strain"
    ] = float(
        np.mean(
            sigma_samples
        )
    )

    posterior_std[
        "sigma_strain"
    ] = float(
        np.std(
            sigma_samples,
            ddof=1,
        )
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "parameter_names": list(
            PARAM_NAMES
        ),
        "physical_parameter_names": list(
            PHYSICAL_NAMES
        ),
        "nuisance_parameter_names": list(
            NUISANCE_NAMES
        ),
        "n_chains": int(
            posterior_chains.shape[0]
        ),
        "iterations_per_chain": int(
            posterior_chains.shape[1]
        ),
        "total_flat_samples": int(
            flat_samples.shape[0]
        ),
        "map": map_record,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "fixed_model_inputs": {
            "pmax": fixed["pmax"],
            "E": fixed["E"],
            "c": fixed["c"],
            "nu": fixed["nu"],
            "tpeak": fixed["tpeak"],
            "d": fixed["d"],
            "alpha": fixed["alpha"],
        },
        "prior_bounds": prior_bounds,
        "run_status": (
            "converged"
            if converged
            else "not_converged"
        ),
        "convergence": {
            "status": (
                "converged"
                if converged
                else "not_converged"
            ),
            "criterion": "all_rhat_below_threshold",
            "threshold": float(
                args.rhat_threshold
            ),
            "converged": bool(
                converged
            ),
            "convergence_iterations_per_chain": (
                int(convergence_iteration)
                if convergence_iteration is not None
                else None
            ),
            "final_rhat": (
                [
                    float(value)
                    for value in final_rhat
                ]
                if final_rhat is not None
                else (
                    convergence_history[-1]["rhat"]
                    if (
                        convergence_history
                        and convergence_history[-1]["rhat"] is not None
                    )
                    else None
                )
            ),
        },
    }

    save_json(
        output_dir
        / "posterior_run_summary.json",
        summary,
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    if converged:
        print(
            "BAYESIAN INVERSION CONVERGED"
        )
    else:
        print(
            "BAYESIAN SAMPLING COMPLETE - NOT CONVERGED"
        )

    print(
        "=" * 78
    )

    print(
        f"\nOutput directory:\n"
        f"    {output_dir}"
    )

    print(
        "\nConvergence status:"
    )

    if converged:
        print(
            "    CONVERGED"
        )
        print(
            f"    iterations/chain = {convergence_iteration}"
        )
    else:
        print(
            "    NOT CONVERGED"
        )
        print(
            f"    maximum iterations reached = {total_iterations}"
        )

    print(
        f"    R-hat threshold = {args.rhat_threshold}"
    )

    if final_rhat is not None:
        print(
            "    final R-hat = "
            + np.array2string(
                np.asarray(final_rhat),
                precision=4,
                separator=", ",
            )
        )
    elif convergence_history:
        last_rhat = convergence_history[-1].get("rhat")
        if last_rhat is not None:
            print(
                "    final R-hat = "
                + np.array2string(
                    np.asarray(last_rhat),
                    precision=4,
                    separator=", ",
                )
            )

    print(
        "\nPosterior dimensions:"
    )

    print(
        f"    chains             = "
        f"{posterior_chains.shape[0]}"
    )

    print(
        f"    iterations/chain   = "
        f"{posterior_chains.shape[1]}"
    )

    print(
        f"    total samples      = "
        f"{flat_samples.shape[0]}"
    )

    print(
        "\nMAP:"
    )

    for name in PARAM_NAMES:

        print(
            f"    {name:22s} = "
            f"{map_record[name]:.8g}"
        )

    print(
        f"    {'sigma_strain':22s} = "
        f"{map_record['sigma_strain']:.8g} nstrain"
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()