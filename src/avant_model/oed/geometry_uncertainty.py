#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geometry-uncertainty utilities for the AVANT analytical-model OED workflow.

Purpose
-------
The body of interest is not assumed to have known dimensions, depth,
orientation, center, or boundaries before strainmeter deployment.

The canonical uncertain physical geometry state is

    [a, b, h, theta_deg, x0_prime, y0_prime]

where:
    a, b        horizontal semi-dimensions of the analytical body
    h           body-center depth used by the analytical solution
    theta_deg   horizontal orientation
    x0_prime    body-center x coordinate
    y0_prime    body-center y coordinate

The thickness/semi-thickness ``c`` remains fixed in the current production
Bayesian/OED parameterization. It can be promoted into the uncertain state
later without changing the overall architecture.

This module provides scenario generation for:

1. Nominal/local OED
   A single reference geometry.

2. Prior-domain robust OED
   Space-filling samples across physically plausible bounds.

3. Posterior-informed robust OED
   Geometry realizations drawn from the Bayesian posterior.

4. Adaptive/Bayesian sensor placement
   Posterior draws can be re-used by later Bayesian design utilities.

This module does not evaluate the forward model, calculate Fisher matrices,
rank sensor designs, or make plots. Those responsibilities belong to the
other OED modules.

Python compatibility
--------------------
Python 3.9 compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Canonical geometry state
# ============================================================================

GEOMETRY_PARAMETER_NAMES = (
    "a",
    "b",
    "h",
    "theta_deg",
    "x0_prime",
    "y0_prime",
)

DEFAULT_REFERENCE_GEOMETRY = {
    "a": 190.0,
    "b": 325.0,
    "h": 520.0,
    "theta_deg": 15.0,
    "x0_prime": 45.0,
    "y0_prime": 170.0,
}


# ============================================================================
# Data containers
# ============================================================================


@dataclass(frozen=True)
class GeometryScenario:
    """One uncertain analytical-body realization."""

    scenario_id: str
    source: str
    a: float
    b: float
    h: float
    theta_deg: float
    x0_prime: float
    y0_prime: float
    weight: float = 1.0

    def parameter_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.a,
                self.b,
                self.h,
                self.theta_deg,
                self.x0_prime,
                self.y0_prime,
            ],
            dtype=float,
        )

    def parameter_dict(self) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(
                GEOMETRY_PARAMETER_NAMES,
                self.parameter_vector(),
            )
        }


# ============================================================================
# Validation
# ============================================================================


def validate_bounds(
    bounds: Mapping[str, Sequence[float]],
) -> None:
    missing = [
        name
        for name in GEOMETRY_PARAMETER_NAMES
        if name not in bounds
    ]

    if missing:
        raise ValueError(
            "Geometry bounds are missing parameters: "
            + ", ".join(missing)
        )

    for name in GEOMETRY_PARAMETER_NAMES:
        values = np.asarray(
            bounds[name],
            dtype=float,
        ).reshape(-1)

        if values.size != 2:
            raise ValueError(
                "Bounds for {0} must contain exactly two values.".format(
                    name
                )
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Bounds for {0} contain non-finite values.".format(
                    name
                )
            )

        if not values[1] > values[0]:
            raise ValueError(
                "Upper bound must exceed lower bound for {0}.".format(
                    name
                )
            )

    # Physical constraints for dimensions/depth.
    for name in (
        "a",
        "b",
        "h",
    ):
        if float(
            bounds[name][0]
        ) <= 0.0:
            raise ValueError(
                "Lower bound for {0} must be positive.".format(
                    name
                )
            )


def validate_geometry(
    geometry: Mapping[str, float],
    bounds: Optional[
        Mapping[str, Sequence[float]]
    ] = None,
) -> None:
    missing = [
        name
        for name in GEOMETRY_PARAMETER_NAMES
        if name not in geometry
    ]

    if missing:
        raise ValueError(
            "Geometry is missing parameters: "
            + ", ".join(missing)
        )

    for name in GEOMETRY_PARAMETER_NAMES:
        value = float(
            geometry[name]
        )

        if not np.isfinite(value):
            raise ValueError(
                "Geometry parameter {0} is non-finite.".format(
                    name
                )
            )

    for name in (
        "a",
        "b",
        "h",
    ):
        if float(
            geometry[name]
        ) <= 0.0:
            raise ValueError(
                "Geometry parameter {0} must be positive.".format(
                    name
                )
            )

    if bounds is not None:
        validate_bounds(
            bounds
        )

        for name in GEOMETRY_PARAMETER_NAMES:
            value = float(
                geometry[name]
            )

            low = float(
                bounds[name][0]
            )

            high = float(
                bounds[name][1]
            )

            if (
                value < low
                or value > high
            ):
                raise ValueError(
                    "Geometry parameter {0}={1} lies outside "
                    "bounds [{2}, {3}].".format(
                        name,
                        value,
                        low,
                        high,
                    )
                )


def normalize_weights(
    weights: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            "At least one scenario weight is required."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "Scenario weights contain non-finite values."
        )

    if np.any(
        array < 0.0
    ):
        raise ValueError(
            "Scenario weights must be non-negative."
        )

    total = float(
        np.sum(array)
    )

    if total <= 0.0:
        raise ValueError(
            "At least one scenario weight must be positive."
        )

    return (
        array
        / total
    )


# ============================================================================
# Scenario conversion
# ============================================================================


def scenario_from_mapping(
    scenario_id: str,
    source: str,
    geometry: Mapping[str, float],
    weight: float = 1.0,
) -> GeometryScenario:
    validate_geometry(
        geometry
    )

    if not np.isfinite(
        float(weight)
    ) or float(weight) < 0.0:
        raise ValueError(
            "Scenario weight must be finite and non-negative."
        )

    return GeometryScenario(
        scenario_id=str(
            scenario_id
        ),
        source=str(
            source
        ),
        a=float(
            geometry["a"]
        ),
        b=float(
            geometry["b"]
        ),
        h=float(
            geometry["h"]
        ),
        theta_deg=float(
            geometry["theta_deg"]
        ),
        x0_prime=float(
            geometry["x0_prime"]
        ),
        y0_prime=float(
            geometry["y0_prime"]
        ),
        weight=float(
            weight
        ),
    )


def scenarios_to_dataframe(
    scenarios: Sequence[GeometryScenario],
    normalize: bool = True,
) -> pd.DataFrame:
    if not scenarios:
        raise ValueError(
            "At least one GeometryScenario is required."
        )

    records = [
        asdict(
            scenario
        )
        for scenario in scenarios
    ]

    dataframe = pd.DataFrame(
        records
    )

    if dataframe[
        "scenario_id"
    ].duplicated().any():
        raise ValueError(
            "scenario_id values must be unique."
        )

    if normalize:
        dataframe[
            "weight"
        ] = normalize_weights(
            dataframe[
                "weight"
            ].to_numpy(
                dtype=float
            )
        )

    return dataframe


def dataframe_to_scenarios(
    dataframe: pd.DataFrame,
) -> Sequence[GeometryScenario]:
    required = {
        "scenario_id",
        "source",
        "weight",
        *GEOMETRY_PARAMETER_NAMES,
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Scenario table is missing columns: "
            + ", ".join(sorted(missing))
        )

    scenarios = []

    for _, row in dataframe.iterrows():
        scenarios.append(
            scenario_from_mapping(
                scenario_id=str(
                    row["scenario_id"]
                ),
                source=str(
                    row["source"]
                ),
                geometry={
                    name: float(
                        row[name]
                    )
                    for name
                    in GEOMETRY_PARAMETER_NAMES
                },
                weight=float(
                    row["weight"]
                ),
            )
        )

    return scenarios


# ============================================================================
# Reference / nominal geometry
# ============================================================================


def reference_scenario(
    geometry: Optional[
        Mapping[str, float]
    ] = None,
    scenario_id: str = "reference",
) -> GeometryScenario:
    if geometry is None:
        geometry = (
            DEFAULT_REFERENCE_GEOMETRY
        )

    return scenario_from_mapping(
        scenario_id=scenario_id,
        source="reference",
        geometry=geometry,
        weight=1.0,
    )


# ============================================================================
# Latin hypercube sampling
# ============================================================================


def _latin_hypercube_unit(
    n_samples: int,
    n_dimensions: int,
    seed: int,
) -> np.ndarray:
    """
    Generate a simple randomized Latin hypercube in [0, 1]^D.

    Implemented locally to avoid requiring scipy.stats.qmc in older
    Python/SciPy installations.
    """

    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    if n_dimensions <= 0:
        raise ValueError(
            "n_dimensions must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    result = np.empty(
        (
            n_samples,
            n_dimensions,
        ),
        dtype=float,
    )

    for dimension in range(
        n_dimensions
    ):
        permutation = rng.permutation(
            n_samples
        )

        jitter = rng.random(
            n_samples
        )

        result[
            :,
            dimension
        ] = (
            permutation
            + jitter
        ) / float(
            n_samples
        )

    return result


def sample_prior_lhs(
    bounds: Mapping[str, Sequence[float]],
    n_samples: int,
    seed: int = 42,
    source: str = "prior_lhs",
) -> pd.DataFrame:
    """
    Generate space-filling geometry scenarios over prior/design bounds.
    """

    validate_bounds(
        bounds
    )

    unit_samples = (
        _latin_hypercube_unit(
            n_samples=n_samples,
            n_dimensions=len(
                GEOMETRY_PARAMETER_NAMES
            ),
            seed=seed,
        )
    )

    physical = np.zeros_like(
        unit_samples
    )

    for index, name in enumerate(
        GEOMETRY_PARAMETER_NAMES
    ):
        low = float(
            bounds[name][0]
        )

        high = float(
            bounds[name][1]
        )

        physical[
            :,
            index
        ] = (
            low
            + unit_samples[
                :,
                index
            ]
            * (
                high
                - low
            )
        )

    weight = (
        1.0
        / float(
            n_samples
        )
    )

    records = []

    for index in range(
        n_samples
    ):
        record = {
            "scenario_id": (
                "{0}_{1:06d}".format(
                    source,
                    index + 1,
                )
            ),
            "source": source,
            "weight": weight,
        }

        for parameter_index, name in enumerate(
            GEOMETRY_PARAMETER_NAMES
        ):
            record[
                name
            ] = float(
                physical[
                    index,
                    parameter_index,
                ]
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================================
# Independent uniform random sampling
# ============================================================================


def sample_prior_random(
    bounds: Mapping[str, Sequence[float]],
    n_samples: int,
    seed: int = 42,
    source: str = "prior_random",
) -> pd.DataFrame:
    """
    Generate independent uniform random geometry scenarios.
    """

    validate_bounds(
        bounds
    )

    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    records = []

    weight = (
        1.0
        / float(
            n_samples
        )
    )

    for index in range(
        n_samples
    ):
        record = {
            "scenario_id": (
                "{0}_{1:06d}".format(
                    source,
                    index + 1,
                )
            ),
            "source": source,
            "weight": weight,
        }

        for name in (
            GEOMETRY_PARAMETER_NAMES
        ):
            low = float(
                bounds[name][0]
            )

            high = float(
                bounds[name][1]
            )

            record[
                name
            ] = float(
                rng.uniform(
                    low,
                    high,
                )
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================================
# Boundary-focused scenarios
# ============================================================================


def boundary_scenarios(
    bounds: Mapping[str, Sequence[float]],
    reference: Optional[
        Mapping[str, float]
    ] = None,
    include_reference: bool = True,
) -> pd.DataFrame:
    """
    Generate deterministic one-parameter-at-a-time boundary scenarios.

    These are useful for stress-testing sensor networks against plausible
    extremes of body size, depth, orientation, and center.

    This is not a substitute for LHS/posterior ensembles; it is an explicit
    edge-case diagnostic.
    """

    validate_bounds(
        bounds
    )

    if reference is None:
        reference = {
            name: (
                0.5
                * (
                    float(
                        bounds[name][0]
                    )
                    + float(
                        bounds[name][1]
                    )
                )
            )
            for name in GEOMETRY_PARAMETER_NAMES
        }

    validate_geometry(
        reference,
        bounds=bounds,
    )

    records = []

    if include_reference:
        records.append(
            {
                "scenario_id": "boundary_reference",
                "source": "boundary_test",
                "weight": 1.0,
                **{
                    name: float(
                        reference[name]
                    )
                    for name
                    in GEOMETRY_PARAMETER_NAMES
                },
            }
        )

    for name in (
        GEOMETRY_PARAMETER_NAMES
    ):
        for side, bound_index in (
            ("low", 0),
            ("high", 1),
        ):
            geometry = {
                parameter: float(
                    reference[
                        parameter
                    ]
                )
                for parameter
                in GEOMETRY_PARAMETER_NAMES
            }

            geometry[
                name
            ] = float(
                bounds[
                    name
                ][
                    bound_index
                ]
            )

            records.append(
                {
                    "scenario_id": (
                        "boundary_{0}_{1}".format(
                            name,
                            side,
                        )
                    ),
                    "source": "boundary_test",
                    "weight": 1.0,
                    **geometry,
                }
            )

    dataframe = pd.DataFrame(
        records
    )

    dataframe[
        "weight"
    ] = normalize_weights(
        dataframe[
            "weight"
        ].to_numpy(
            dtype=float
        )
    )

    return dataframe


# ============================================================================
# Posterior scenarios
# ============================================================================


def posterior_geometry_scenarios(
    posterior_samples: np.ndarray,
    parameter_names: Sequence[str],
    n_samples: Optional[int] = None,
    seed: int = 42,
    weights: Optional[
        Sequence[float]
    ] = None,
    source: str = "bayesian_posterior",
) -> pd.DataFrame:
    """
    Extract geometry scenarios from posterior samples.

    Extra posterior columns, such as log10_sigma_strain, are ignored.

    Parameters
    ----------
    posterior_samples
        Shape (n_samples, n_parameters).

    parameter_names
        Names corresponding to posterior columns.

    n_samples
        Optional posterior subsample size. If omitted, all samples are used.

    weights
        Optional posterior weights. For ordinary MCMC draws these should
        normally be omitted because retained draws are already distributed
        according to the posterior.
    """

    samples = np.asarray(
        posterior_samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "posterior_samples must be a 2-D array."
        )

    if not np.all(
        np.isfinite(samples)
    ):
        raise ValueError(
            "posterior_samples contains non-finite values."
        )

    names = [
        str(name)
        for name in parameter_names
    ]

    if len(
        names
    ) != samples.shape[1]:
        raise ValueError(
            "parameter_names length does not match posterior columns."
        )

    missing = [
        name
        for name in GEOMETRY_PARAMETER_NAMES
        if name not in names
    ]

    if missing:
        raise ValueError(
            "Posterior does not contain geometry parameters: "
            + ", ".join(missing)
        )

    geometry_indices = [
        names.index(
            name
        )
        for name in GEOMETRY_PARAMETER_NAMES
    ]

    geometry_samples = samples[
        :,
        geometry_indices,
    ]

    n_available = (
        geometry_samples.shape[0]
    )

    if n_samples is None:
        selected_indices = np.arange(
            n_available
        )
    else:
        if n_samples <= 0:
            raise ValueError(
                "n_samples must be positive."
            )

        rng = np.random.default_rng(
            seed
        )

        if n_samples >= n_available:
            selected_indices = np.arange(
                n_available
            )
        else:
            selected_indices = rng.choice(
                n_available,
                size=int(
                    n_samples
                ),
                replace=False,
            )

            selected_indices.sort()

    selected = geometry_samples[
        selected_indices
    ]

    if weights is None:
        selected_weights = np.full(
            selected.shape[0],
            1.0
            / float(
                selected.shape[0]
            ),
            dtype=float,
        )
    else:
        full_weights = np.asarray(
            weights,
            dtype=float,
        ).reshape(-1)

        if full_weights.size != n_available:
            raise ValueError(
                "weights length does not match posterior sample count."
            )

        selected_weights = normalize_weights(
            full_weights[
                selected_indices
            ]
        )

    records = []

    for row_index in range(
        selected.shape[0]
    ):
        record = {
            "scenario_id": (
                "{0}_{1:06d}".format(
                    source,
                    row_index + 1,
                )
            ),
            "source": source,
            "weight": float(
                selected_weights[
                    row_index
                ]
            ),
        }

        for parameter_index, name in enumerate(
            GEOMETRY_PARAMETER_NAMES
        ):
            record[
                name
            ] = float(
                selected[
                    row_index,
                    parameter_index,
                ]
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


def load_posterior_scenarios_from_run(
    run_dir,
    n_samples: Optional[int] = None,
    burn_in_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load posterior geometry scenarios directly from a production Bayesian run.

    Expected files:
        posterior_chains.npy
        run_manifest.json

    The nuisance parameter is automatically ignored.
    """

    from pathlib import Path
    import json

    run_path = Path(
        run_dir
    ).resolve()

    chains_path = (
        run_path
        / "posterior_chains.npy"
    )

    manifest_path = (
        run_path
        / "run_manifest.json"
    )

    if not chains_path.exists():
        raise FileNotFoundError(
            "posterior_chains.npy not found:\n    {0}".format(
                chains_path
            )
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "run_manifest.json not found:\n    {0}".format(
                manifest_path
            )
        )

    if not (
        0.0
        <= burn_in_fraction
        < 1.0
    ):
        raise ValueError(
            "burn_in_fraction must satisfy 0 <= fraction < 1."
        )

    chains = np.load(
        chains_path,
        allow_pickle=False,
    )

    if chains.ndim != 3:
        raise ValueError(
            "posterior_chains.npy must have shape "
            "(nchains, niterations, nparameters)."
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    parameter_names = manifest.get(
        "parameter_names",
        [],
    )

    if len(
        parameter_names
    ) != chains.shape[2]:
        raise ValueError(
            "Manifest parameter_names does not match posterior dimension."
        )

    burn_in = int(
        chains.shape[1]
        * burn_in_fraction
    )

    retained = chains[
        :,
        burn_in:,
        :,
    ]

    if retained.shape[1] < 1:
        raise ValueError(
            "No posterior samples remain after burn-in."
        )

    flat = retained.reshape(
        -1,
        retained.shape[2],
    )

    return posterior_geometry_scenarios(
        posterior_samples=flat,
        parameter_names=parameter_names,
        n_samples=n_samples,
        seed=seed,
        source="bayesian_posterior",
    )


# ============================================================================
# Mixed prior/posterior scenario sets
# ============================================================================


def combine_scenario_tables(
    tables: Sequence[pd.DataFrame],
    source_weights: Optional[
        Sequence[float]
    ] = None,
) -> pd.DataFrame:
    """
    Combine scenario ensembles and normalize their weights.

    ``source_weights`` can deliberately balance ensembles, for example:
        0.5 prior-robust scenarios
        0.5 posterior-informed scenarios

    This is useful when designing a network that should remain robust to
    broader geological uncertainty while exploiting current Bayesian
    information.
    """

    if not tables:
        raise ValueError(
            "At least one scenario table is required."
        )

    required = {
        "scenario_id",
        "source",
        "weight",
        *GEOMETRY_PARAMETER_NAMES,
    }

    checked = []

    for index, table in enumerate(
        tables
    ):
        missing = (
            required
            - set(
                table.columns
            )
        )

        if missing:
            raise ValueError(
                "Scenario table {0} is missing columns: {1}".format(
                    index,
                    ", ".join(
                        sorted(
                            missing
                        )
                    ),
                )
            )

        checked.append(
            table.copy()
        )

    if source_weights is None:
        source_weights_array = np.full(
            len(checked),
            1.0
            / float(
                len(checked)
            ),
            dtype=float,
        )
    else:
        if len(
            source_weights
        ) != len(
            checked
        ):
            raise ValueError(
                "source_weights length must match the number of tables."
            )

        source_weights_array = normalize_weights(
            source_weights
        )

    combined_parts = []

    for table_index, table in enumerate(
        checked
    ):
        local_weights = normalize_weights(
            table[
                "weight"
            ].to_numpy(
                dtype=float
            )
        )

        table[
            "weight"
        ] = (
            local_weights
            * source_weights_array[
                table_index
            ]
        )

        # Guarantee globally unique scenario IDs after combining sources.
        table[
            "scenario_id"
        ] = [
            "set{0}_{1}".format(
                table_index + 1,
                scenario_id,
            )
            for scenario_id
            in table[
                "scenario_id"
            ].astype(str)
        ]

        combined_parts.append(
            table
        )

    combined = pd.concat(
        combined_parts,
        axis=0,
        ignore_index=True,
    )

    combined[
        "weight"
    ] = normalize_weights(
        combined[
            "weight"
        ].to_numpy(
            dtype=float
        )
    )

    return combined


# ============================================================================
# Scenario diagnostics
# ============================================================================


def scenario_summary(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return descriptive statistics for each uncertain geometry parameter.
    """

    required = {
        *GEOMETRY_PARAMETER_NAMES,
    }

    missing = (
        required
        - set(
            scenarios.columns
        )
    )

    if missing:
        raise ValueError(
            "Scenario table is missing geometry columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    rows = []

    for name in (
        GEOMETRY_PARAMETER_NAMES
    ):
        values = scenarios[
            name
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "parameter": name,
                "mean": float(
                    np.mean(
                        values
                    )
                ),
                "std": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if values.size > 1
                else 0.0,
                "minimum": float(
                    np.min(
                        values
                    )
                ),
                "q05": float(
                    np.percentile(
                        values,
                        5.0,
                    )
                ),
                "q25": float(
                    np.percentile(
                        values,
                        25.0,
                    )
                ),
                "median": float(
                    np.median(
                        values
                    )
                ),
                "q75": float(
                    np.percentile(
                        values,
                        75.0,
                    )
                ),
                "q95": float(
                    np.percentile(
                        values,
                        95.0,
                    )
                ),
                "maximum": float(
                    np.max(
                        values
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def effective_scenario_count(
    scenarios: pd.DataFrame,
) -> float:
    """
    Effective sample size implied by scenario weights:

        N_eff = 1 / sum(w_i^2)
    """

    if "weight" not in scenarios.columns:
        raise ValueError(
            "Scenario table must contain a weight column."
        )

    weights = normalize_weights(
        scenarios[
            "weight"
        ].to_numpy(
            dtype=float
        )
    )

    return float(
        1.0
        / np.sum(
            weights**2
        )
    )
