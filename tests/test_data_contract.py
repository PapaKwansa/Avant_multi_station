"""Tests for the canonical AVANT 1-Darcy data contract."""

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

STATION_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "metadata"
    / "stations.csv"
)

STRAIN_FILE = (
    REPO_ROOT
    / "datasets"
    / "comsol"
    / "1darcy"
    / "processed"
    / "strain.csv"
)

COMPONENTS = ("eXX", "eYY", "eZZ", "eXY")


def test_canonical_files_exist():
    assert STATION_FILE.is_file()
    assert STRAIN_FILE.is_file()


def test_station_metadata_contract():
    stations = pd.read_csv(STATION_FILE)

    assert list(stations.columns) == [
        "station",
        "x",
        "y",
        "z",
    ]

    assert len(stations) == 8

    assert stations["station"].tolist() == [
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
        "S07",
        "S08",
    ]

    assert stations["station"].is_unique

    coordinates = stations[
        ["x", "y", "z"]
    ].to_numpy(dtype=float)

    assert np.isfinite(coordinates).all()


def test_strain_dataset_contract():
    stations = pd.read_csv(STATION_FILE)
    strain = pd.read_csv(STRAIN_FILE)

    assert "time_s" in strain.columns
    assert len(strain) == 107

    expected_columns = [
        "time_s",
        *[
            f"{component}_{station}"
            for component in COMPONENTS
            for station in stations["station"]
        ],
    ]

    assert strain.columns.tolist() == expected_columns

    # 1 time column + 4 components x 8 stations.
    assert strain.shape == (107, 33)

    values = strain.to_numpy(dtype=float)

    assert np.isfinite(values).all()


def test_time_contract():
    strain = pd.read_csv(STRAIN_FILE)

    time = strain["time_s"].to_numpy(dtype=float)

    assert time.size == 107
    assert np.isfinite(time).all()
    assert np.all(time >= 0.0)


def test_strain_channel_count():
    strain = pd.read_csv(STRAIN_FILE)

    strain_columns = [
        column
        for column in strain.columns
        if column != "time_s"
    ]

    assert len(strain_columns) == 32