#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensor-placement utilities for the AVANT analytical-model OED workflow.

This module handles the spatial-design layer of OED.  It does not evaluate
the analytical forward model or compute Fisher information directly.
Instead, it provides reusable tools for constructing and validating
candidate strainmeter locations and candidate sensor networks.

Supported design problems
-------------------------
1. Existing-network selection
   Select subsets of the currently available S01--S08 stations.

2. Free spatial placement
   Generate candidate sensor locations in a user-defined x-y deployment
   domain.

3. Constrained placement
   Enforce:
       - minimum inter-sensor spacing
       - rectangular/circular deployment bounds
       - excluded regions
       - fixed/required sensors

4. Candidate-network generation
   Generate all combinations for small candidate sets or bounded numbers of
   random/greedy candidate networks for larger design spaces.

5. Geometry-relative candidate locations
   When the body geometry is uncertain, candidate locations remain defined
   in the external deployment coordinate system.  The true body boundary is
   never required to generate the candidate network.

This is intentional: the practical field-design problem is often to place
strainmeters before the body's exact shape and boundaries are known.

The resulting candidate networks can be passed to the robust-design module,
which evaluates their information content over uncertain geometry or
Bayesian posterior scenarios.

Python compatibility
--------------------
Python 3.9 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Data containers
# ============================================================================


@dataclass(frozen=True)
class SensorLocation:
    """One candidate strainmeter location."""

    sensor_id: str
    x: float
    y: float
    z: float = 0.0
    fixed: bool = False
    label: str = ""

    def as_tuple(self) -> Tuple[float, float, float]:
        return (
            float(self.x),
            float(self.y),
            float(self.z),
        )


@dataclass(frozen=True)
class SensorNetwork:
    """A candidate strainmeter network."""

    design_id: str
    sensor_ids: Tuple[str, ...]
    sensor_locations: Tuple[SensorLocation, ...]
    design_size: int

    @property
    def xy_coordinates(self) -> np.ndarray:
        return np.asarray(
            [
                [sensor.x, sensor.y]
                for sensor in self.sensor_locations
            ],
            dtype=float,
        )


# ============================================================================
# Validation helpers
# ============================================================================


def validate_sensor_locations(
    locations: Sequence[SensorLocation],
) -> None:
    if not locations:
        raise ValueError(
            "At least one sensor location is required."
        )

    ids = [
        str(location.sensor_id)
        for location in locations
    ]

    if len(
        ids
    ) != len(
        set(ids)
    ):
        raise ValueError(
            "Sensor IDs must be unique."
        )

    for location in locations:
        values = np.asarray(
            [
                location.x,
                location.y,
                location.z,
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Sensor {0} contains non-finite coordinates.".format(
                    location.sensor_id
                )
            )


def validate_sensor_network(
    network: SensorNetwork,
) -> None:
    validate_sensor_locations(
        network.sensor_locations
    )

    if network.design_size != len(
        network.sensor_locations
    ):
        raise ValueError(
            "design_size does not match the number of sensor locations."
        )

    if tuple(
        network.sensor_ids
    ) != tuple(
        location.sensor_id
        for location in network.sensor_locations
    ):
        raise ValueError(
            "sensor_ids do not match sensor_locations."
        )


def validate_deployment_bounds(
    bounds: Sequence[float],
) -> Tuple[float, float, float, float]:
    values = np.asarray(
        bounds,
        dtype=float,
    ).reshape(-1)

    if values.size != 4:
        raise ValueError(
            "Rectangular bounds must be [xmin, xmax, ymin, ymax]."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Deployment bounds contain non-finite values."
        )

    xmin, xmax, ymin, ymax = (
        float(value)
        for value in values
    )

    if xmax <= xmin:
        raise ValueError(
            "xmax must exceed xmin."
        )

    if ymax <= ymin:
        raise ValueError(
            "ymax must exceed ymin."
        )

    return (
        xmin,
        xmax,
        ymin,
        ymax,
    )


def validate_min_spacing(
    min_spacing: float,
) -> float:
    value = float(
        min_spacing
    )

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "min_spacing must be finite and non-negative."
        )

    return value


# ============================================================================
# Location conversion
# ============================================================================


def locations_to_dataframe(
    locations: Sequence[SensorLocation],
) -> pd.DataFrame:
    validate_sensor_locations(
        locations
    )

    rows = []

    for location in locations:
        rows.append(
            {
                "sensor_id": str(
                    location.sensor_id
                ),
                "x": float(
                    location.x
                ),
                "y": float(
                    location.y
                ),
                "z": float(
                    location.z
                ),
                "fixed": bool(
                    location.fixed
                ),
                "label": str(
                    location.label
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def dataframe_to_locations(
    dataframe: pd.DataFrame,
) -> List[SensorLocation]:
    required = {
        "sensor_id",
        "x",
        "y",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Sensor-location table is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    locations = []

    for _, row in dataframe.iterrows():
        locations.append(
            SensorLocation(
                sensor_id=str(
                    row["sensor_id"]
                ),
                x=float(
                    row["x"]
                ),
                y=float(
                    row["y"]
                ),
                z=float(
                    row.get(
                        "z",
                        0.0,
                    )
                ),
                fixed=bool(
                    row.get(
                        "fixed",
                        False,
                    )
                ),
                label=str(
                    row.get(
                        "label",
                        "",
                    )
                ),
            )
        )

    validate_sensor_locations(
        locations
    )

    return locations


# ============================================================================
# Existing station networks
# ============================================================================


def networks_from_station_table(
    stations: pd.DataFrame,
    *,
    subset_sizes: Optional[
        Sequence[int]
    ] = None,
    include_all: bool = True,
) -> List[SensorNetwork]:
    """
    Convert an existing station table into candidate sensor networks.

    Expected columns:
        station,x,y
    Optional:
        z
    """

    required = {
        "station",
        "x",
        "y",
    }

    missing = (
        required
        - set(
            stations.columns
        )
    )

    if missing:
        raise ValueError(
            "Station table is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    locations = []

    for _, row in stations.iterrows():
        locations.append(
            SensorLocation(
                sensor_id=str(
                    row["station"]
                ).strip(),
                x=float(
                    row["x"]
                ),
                y=float(
                    row["y"]
                ),
                z=float(
                    row.get(
                        "z",
                        0.0,
                    )
                ),
            )
        )

    return enumerate_sensor_networks(
        locations,
        subset_sizes=subset_sizes,
        include_all=include_all,
    )


def enumerate_sensor_networks(
    locations: Sequence[SensorLocation],
    *,
    subset_sizes: Optional[
        Sequence[int]
    ] = None,
    include_all: bool = True,
    prefix: str = "network",
) -> List[SensorNetwork]:
    """
    Enumerate all candidate networks from a finite candidate-location set.

    For eight existing stations, enumerating all 255 non-empty subsets is
    inexpensive and appropriate.

    For large candidate-location grids, use candidate_networks_random or
    greedy-network generation instead.
    """

    validate_sensor_locations(
        locations
    )

    n_locations = len(
        locations
    )

    if subset_sizes is None:
        sizes = list(
            range(
                1,
                n_locations + 1,
            )
        )
    else:
        sizes = sorted(
            set(
                int(size)
                for size in subset_sizes
            )
        )

        if not sizes:
            raise ValueError(
                "subset_sizes cannot be empty."
            )

        if any(
            size < 1
            or size > n_locations
            for size in sizes
        ):
            raise ValueError(
                "subset_sizes must lie between 1 and the number "
                "of candidate locations."
            )

    networks = []

    design_number = 0

    for size in sizes:
        for indices in combinations(
            range(
                n_locations
            ),
            size,
        ):
            selected = tuple(
                locations[index]
                for index in indices
            )

            if size == n_locations and not include_all:
                continue

            design_number += 1

            network = SensorNetwork(
                design_id=(
                    "{0}_{1:05d}".format(
                        prefix,
                        design_number,
                    )
                ),
                sensor_ids=tuple(
                    location.sensor_id
                    for location in selected
                ),
                sensor_locations=selected,
                design_size=size,
            )

            validate_sensor_network(
                network
            )

            networks.append(
                network
            )

    return networks


# ============================================================================
# Geometry-independent deployment masks
# ============================================================================


def point_in_rectangle(
    x: float,
    y: float,
    bounds: Sequence[float],
) -> bool:
    xmin, xmax, ymin, ymax = (
        validate_deployment_bounds(
            bounds
        )
    )

    return bool(
        xmin <= x <= xmax
        and ymin <= y <= ymax
    )


def point_in_circle(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    radius: float,
) -> bool:
    radius = float(
        radius
    )

    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError(
            "radius must be finite and non-negative."
        )

    return bool(
        (
            float(x) - float(center_x)
        )
        ** 2
        + (
            float(y) - float(center_y)
        )
        ** 2
        <= radius**2
    )


def point_is_excluded(
    x: float,
    y: float,
    excluded_regions: Optional[
        Sequence[Mapping[str, object]]
    ] = None,
) -> bool:
    """
    Return True when a point lies inside an explicitly excluded region.

    Supported region dictionaries:
        {"type": "rectangle", "bounds": [xmin,xmax,ymin,ymax]}
        {"type": "circle", "center": [x,y], "radius": r}
    """

    if not excluded_regions:
        return False

    for region in excluded_regions:
        region_type = str(
            region.get(
                "type",
                "",
            )
        ).lower()

        if region_type == "rectangle":
            if point_in_rectangle(
                x,
                y,
                region["bounds"],
            ):
                return True

        elif region_type == "circle":
            center = region.get(
                "center"
            )

            if (
                center is None
                or len(center) != 2
            ):
                raise ValueError(
                    "Circle exclusion region requires center=[x,y]."
                )

            if point_in_circle(
                x,
                y,
                center[0],
                center[1],
                region["radius"],
            ):
                return True

        else:
            raise ValueError(
                "Unsupported exclusion-region type: {0}".format(
                    region_type
                )
            )

    return False


def location_allowed(
    location: SensorLocation,
    deployment_bounds: Optional[
        Sequence[float]
    ] = None,
    excluded_regions: Optional[
        Sequence[Mapping[str, object]]
    ] = None,
) -> bool:
    if deployment_bounds is not None:
        if not point_in_rectangle(
            location.x,
            location.y,
            deployment_bounds,
        ):
            return False

    if point_is_excluded(
        location.x,
        location.y,
        excluded_regions,
    ):
        return False

    return True


# ============================================================================
# Spacing constraints
# ============================================================================


def euclidean_distance(
    first: SensorLocation,
    second: SensorLocation,
) -> float:
    return float(
        hypot(
            first.x - second.x,
            first.y - second.y,
        )
    )


def pairwise_distances(
    locations: Sequence[SensorLocation],
) -> pd.DataFrame:
    validate_sensor_locations(
        locations
    )

    rows = []

    for index_a, index_b in combinations(
        range(
            len(locations)
        ),
        2,
    ):
        first = locations[
            index_a
        ]

        second = locations[
            index_b
        ]

        rows.append(
            {
                "sensor_id_i": first.sensor_id,
                "sensor_id_j": second.sensor_id,
                "distance": euclidean_distance(
                    first,
                    second,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def satisfies_minimum_spacing(
    locations: Sequence[SensorLocation],
    min_spacing: float,
) -> bool:
    spacing = validate_min_spacing(
        min_spacing
    )

    validate_sensor_locations(
        locations
    )

    if len(
        locations
    ) < 2:
        return True

    distances = pairwise_distances(
        locations
    )

    return bool(
        np.all(
            distances[
                "distance"
            ].to_numpy(
                dtype=float
            )
            >= spacing
        )
    )


# ============================================================================
# Candidate grid generation
# ============================================================================


def generate_rectangular_grid(
    bounds: Sequence[float],
    nx: int,
    ny: int,
    *,
    z: float = 0.0,
    prefix: str = "C",
    include_boundary: bool = True,
    excluded_regions: Optional[
        Sequence[Mapping[str, object]]
    ] = None,
) -> List[SensorLocation]:
    """
    Generate a deterministic rectangular candidate grid.
    """

    xmin, xmax, ymin, ymax = (
        validate_deployment_bounds(
            bounds
        )
    )

    if nx < 1 or ny < 1:
        raise ValueError(
            "nx and ny must be positive."
        )

    if include_boundary:
        x_values = np.linspace(
            xmin,
            xmax,
            int(nx),
        )

        y_values = np.linspace(
            ymin,
            ymax,
            int(ny),
        )

    else:
        x_values = np.linspace(
            xmin,
            xmax,
            int(nx) + 2,
        )[1:-1]

        y_values = np.linspace(
            ymin,
            ymax,
            int(ny) + 2,
        )[1:-1]

    locations = []

    counter = 0

    for y in y_values:
        for x in x_values:
            if point_is_excluded(
                x,
                y,
                excluded_regions,
            ):
                continue

            counter += 1

            locations.append(
                SensorLocation(
                    sensor_id=(
                        "{0}{1:04d}".format(
                            prefix,
                            counter,
                        )
                    ),
                    x=float(x),
                    y=float(y),
                    z=float(z),
                )
            )

    return locations


def filter_locations(
    locations: Sequence[SensorLocation],
    deployment_bounds: Optional[
        Sequence[float]
    ] = None,
    excluded_regions: Optional[
        Sequence[Mapping[str, object]]
    ] = None,
    min_spacing: float = 0.0,
) -> List[SensorLocation]:
    """
    Filter a candidate set against deployment constraints and minimum spacing.

    Minimum spacing is enforced greedily in the input order.  For more
    sophisticated spatial optimization, the returned locations should be
    passed to a design-selection optimizer rather than relying on this
    greedy filter alone.
    """

    validate_sensor_locations(
        locations
    )

    spacing = validate_min_spacing(
        min_spacing
    )

    accepted = []

    for location in locations:
        if not location_allowed(
            location,
            deployment_bounds,
            excluded_regions,
        ):
            continue

        if spacing > 0.0:
            if any(
                euclidean_distance(
                    location,
                    existing,
                )
                < spacing
                for existing
                in accepted
            ):
                continue

        accepted.append(
            location
        )

    return accepted


# ============================================================================
# Random candidate placement
# ============================================================================


def sample_random_locations(
    bounds: Sequence[float],
    n_locations: int,
    *,
    seed: int = 42,
    z: float = 0.0,
    prefix: str = "R",
    excluded_regions: Optional[
        Sequence[Mapping[str, object]]
    ] = None,
    min_spacing: float = 0.0,
    max_attempts: Optional[int] = None,
) -> List[SensorLocation]:
    """
    Generate random admissible candidate locations.

    Rejection sampling is used for excluded regions and spacing.
    """

    xmin, xmax, ymin, ymax = (
        validate_deployment_bounds(
            bounds
        )
    )

    if n_locations <= 0:
        raise ValueError(
            "n_locations must be positive."
        )

    spacing = validate_min_spacing(
        min_spacing
    )

    if max_attempts is None:
        max_attempts = max(
            1000,
            int(
                100
                * n_locations
            ),
        )

    rng = np.random.default_rng(
        seed
    )

    locations = []

    attempts = 0

    while (
        len(locations)
        < n_locations
        and attempts < max_attempts
    ):

        attempts += 1

        x = float(
            rng.uniform(
                xmin,
                xmax,
            )
        )

        y = float(
            rng.uniform(
                ymin,
                ymax,
            )
        )

        candidate = SensorLocation(
            sensor_id=(
                "{0}{1:05d}".format(
                    prefix,
                    len(locations) + 1,
                )
            ),
            x=x,
            y=y,
            z=float(z),
        )

        if not location_allowed(
            candidate,
            deployment_bounds=bounds,
            excluded_regions=excluded_regions,
        ):
            continue

        if spacing > 0.0:
            if any(
                euclidean_distance(
                    candidate,
                    existing,
                )
                < spacing
                for existing
                in locations
            ):
                continue

        locations.append(
            candidate
        )

    if len(
        locations
    ) < n_locations:
        raise RuntimeError(
            "Unable to generate the requested number of admissible "
            "locations within max_attempts. "
            "Relax spacing/exclusion constraints or increase max_attempts."
        )

    return locations


# ============================================================================
# Candidate network generation for large spatial designs
# ============================================================================


def random_candidate_networks(
    locations: Sequence[SensorLocation],
    design_size: int,
    n_networks: int,
    *,
    seed: int = 42,
    required_sensor_ids: Optional[
        Sequence[str]
    ] = None,
    min_spacing: float = 0.0,
) -> List[SensorNetwork]:
    """
    Generate unique random network candidates.

    Required sensors are included in every network.

    This is preferable to exhaustive enumeration when the candidate
    location set is large.
    """

    validate_sensor_locations(
        locations
    )

    if design_size <= 0:
        raise ValueError(
            "design_size must be positive."
        )

    if design_size > len(
        locations
    ):
        raise ValueError(
            "design_size exceeds the number of candidate locations."
        )

    if n_networks <= 0:
        raise ValueError(
            "n_networks must be positive."
        )

    spacing = validate_min_spacing(
        min_spacing
    )

    location_by_id = {
        location.sensor_id: location
        for location in locations
    }

    if required_sensor_ids is None:
        required_ids = []
    else:
        required_ids = [
            str(sensor_id)
            for sensor_id in required_sensor_ids
        ]

    missing_required = [
        sensor_id
        for sensor_id in required_ids
        if sensor_id not in location_by_id
    ]

    if missing_required:
        raise ValueError(
            "Required sensors are not present in candidate locations: "
            + ", ".join(
                missing_required
            )
        )

    if len(
        required_ids
    ) > design_size:
        raise ValueError(
            "Number of required sensors exceeds design_size."
        )

    rng = np.random.default_rng(
        seed
    )

    optional_locations = [
        location
        for location in locations
        if location.sensor_id
        not in set(
            required_ids
        )
    ]

    unique_signatures = set()

    networks = []

    max_attempts = max(
        1000,
        50 * n_networks,
    )

    attempts = 0

    while (
        len(networks)
        < n_networks
        and attempts < max_attempts
    ):

        attempts += 1

        optional_count = (
            design_size
            - len(
                required_ids
            )
        )

        if optional_count == 0:
            selected_optional = []
        else:
            selected_indices = rng.choice(
                len(
                    optional_locations
                ),
                size=optional_count,
                replace=False,
            )

            selected_optional = [
                optional_locations[
                    int(index)
                ]
                for index
                in selected_indices
            ]

        selected = [
            location_by_id[
                sensor_id
            ]
            for sensor_id
            in required_ids
        ] + selected_optional

        selected = sorted(
            selected,
            key=lambda location:
                str(
                    location.sensor_id
                ),
        )

        signature = tuple(
            location.sensor_id
            for location in selected
        )

        if signature in unique_signatures:
            continue

        if not satisfies_minimum_spacing(
            selected,
            spacing,
        ):
            continue

        unique_signatures.add(
            signature
        )

        networks.append(
            SensorNetwork(
                design_id=(
                    "random_network_{0:05d}".format(
                        len(networks) + 1
                    )
                ),
                sensor_ids=signature,
                sensor_locations=tuple(
                    selected
                ),
                design_size=len(
                    selected
                ),
            )
        )

    if len(
        networks
    ) < n_networks:
        raise RuntimeError(
            "Only generated {0} of {1} requested random networks. "
            "Increase candidate density or relax constraints.".format(
                len(networks),
                n_networks,
            )
        )

    return networks


# ============================================================================
# Greedy network generation
# ============================================================================


def greedy_network_by_distance(
    locations: Sequence[SensorLocation],
    design_size: int,
    *,
    seed_sensor_id: Optional[str] = None,
    min_spacing: float = 0.0,
) -> SensorNetwork:
    """
    Construct a deterministic spatially dispersed candidate network.

    This is a geometry-only baseline and is NOT an information-optimal
    algorithm.  It is useful as a reference design against which OED results
    can be compared.

    Selection proceeds by repeatedly adding the candidate farthest from the
    already selected set.
    """

    validate_sensor_locations(
        locations
    )

    if not (
        1 <= design_size <= len(
            locations
        )
    ):
        raise ValueError(
            "design_size must be within the candidate-location count."
        )

    spacing = validate_min_spacing(
        min_spacing
    )

    location_by_id = {
        location.sensor_id: location
        for location in locations
    }

    if seed_sensor_id is None:
        seed = locations[0]
    else:
        if seed_sensor_id not in location_by_id:
            raise ValueError(
                "seed_sensor_id is not present in candidate locations."
            )

        seed = location_by_id[
            seed_sensor_id
        ]

    selected = [
        seed
    ]

    remaining = [
        location
        for location in locations
        if location.sensor_id
        != seed.sensor_id
    ]

    while (
        len(selected)
        < design_size
    ):

        admissible = []

        for location in remaining:

            nearest = min(
                euclidean_distance(
                    location,
                    selected_location,
                )
                for selected_location
                in selected
            )

            if nearest >= spacing:
                admissible.append(
                    (
                        nearest,
                        location,
                    )
                )

        if not admissible:
            raise RuntimeError(
                "Could not satisfy the requested design size under "
                "the minimum-spacing constraint."
            )

        admissible.sort(
            key=lambda item:
                (
                    -item[0],
                    str(
                        item[1].sensor_id
                    ),
                )
        )

        selected.append(
            admissible[0][1]
        )

        remaining = [
            location
            for location in remaining
            if location.sensor_id
            != admissible[0][1].sensor_id
        ]

    selected = tuple(
        selected
    )

    network = SensorNetwork(
        design_id=(
            "greedy_distance_{0}".format(
                design_size
            )
        ),
        sensor_ids=tuple(
            location.sensor_id
            for location in selected
        ),
        sensor_locations=selected,
        design_size=design_size,
    )

    validate_sensor_network(
        network
    )

    return network


# ============================================================================
# Network tables
# ============================================================================


def networks_to_dataframe(
    networks: Sequence[SensorNetwork],
) -> pd.DataFrame:
    if not networks:
        raise ValueError(
            "At least one network is required."
        )

    records = []

    for network in networks:
        validate_sensor_network(
            network
        )

        record = {
            "design_id": network.design_id,
            "design_size": network.design_size,
            "sensor_ids": ",".join(
                network.sensor_ids
            ),
        }

        for index, location in enumerate(
            network.sensor_locations,
            start=1,
        ):
            record[
                "sensor_{0}_id".format(
                    index
                )
            ] = location.sensor_id

            record[
                "sensor_{0}_x".format(
                    index
                )
            ] = float(
                location.x
            )

            record[
                "sensor_{0}_y".format(
                    index
                )
            ] = float(
                location.y
            )

            record[
                "sensor_{0}_z".format(
                    index
                )] = float(
                location.z
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================================
# Spatial baseline comparison
# ============================================================================


def network_minimum_spacing(
    network: SensorNetwork,
) -> float:
    validate_sensor_network(
        network
    )

    if network.design_size < 2:
        return float(
            np.inf
        )

    distances = pairwise_distances(
        network.sensor_locations
    )

    return float(
        distances[
            "distance"
        ].min()
    )


def network_centroid(
    network: SensorNetwork,
) -> Tuple[float, float]:
    validate_sensor_network(
        network
    )

    coordinates = network.xy_coordinates

    return (
        float(
            np.mean(
                coordinates[:, 0]
            )
        ),
        float(
            np.mean(
                coordinates[:, 1]
            )
        ),
    )


def network_spatial_spread(
    network: SensorNetwork,
) -> float:
    """
    Root-mean-square radial distance from the network centroid.

    Used as a descriptive geometric metric only; it is not an information
    metric.
    """

    validate_sensor_network(
        network
    )

    cx, cy = network_centroid(
        network
    )

    radii = np.sqrt(
        (
            network.xy_coordinates[:, 0]
            - cx
        )
        ** 2
        + (
            network.xy_coordinates[:, 1]
            - cy
        )
        ** 2
    )

    return float(
        np.sqrt(
            np.mean(
                radii**2
            )
        )
    )


def add_network_geometry_metrics(
    table: pd.DataFrame,
    networks: Sequence[SensorNetwork],
) -> pd.DataFrame:
    """
    Add purely geometric network descriptors to a design-results table.
    """

    network_map = {
        network.design_id: network
        for network in networks
    }

    dataframe = table.copy()

    if "design_id" not in dataframe.columns:
        raise ValueError(
            "Design result table must contain design_id."
        )

    dataframe[
        "network_min_spacing"
    ] = [
        network_minimum_spacing(
            network_map[
                design_id
            ]
        )
        for design_id
        in dataframe[
            "design_id"
        ]
    ]

    dataframe[
        "network_spatial_spread"
    ] = [
        network_spatial_spread(
            network_map[
                design_id
            ]
        )
        for design_id
        in dataframe[
            "design_id"
        ]
    ]

    centroid_x = []
    centroid_y = []

    for design_id in dataframe[
        "design_id"
    ]:
        cx, cy = network_centroid(
            network_map[
                design_id
            ]
        )
        centroid_x.append(
            cx
        )
        centroid_y.append(
            cy
        )

    dataframe[
        "network_centroid_x"
    ] = centroid_x

    dataframe[
        "network_centroid_y"
    ] = centroid_y

    return dataframe
