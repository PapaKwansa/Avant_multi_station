"""Regression tests for the AVANT multi-station analytical forward model."""

from pathlib import Path

import numpy as np
import pandas as pd

from avant_model.model.forward_model_multi_station import (
    forward_model_multi_station,
)


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


def load_reference_inputs():
    stations = pd.read_csv(STATION_FILE)
    observed = pd.read_csv(STRAIN_FILE)

    return {
        "time": observed["time_s"].to_numpy(dtype=float),
        "station_names": stations["station"].astype(str).to_numpy(),
        "x_prime": stations["x"].to_numpy(dtype=float),
        "y_prime": stations["y"].to_numpy(dtype=float),
        "z": stations["z"].to_numpy(dtype=float),
    }


def run_reference_forward_model():
    data = load_reference_inputs()

    return forward_model_multi_station(
        pmax=999310.0,
        tpeak=350000.0,
        d=0.4,
        time=data["time"],
        x_prime=data["x_prime"],
        y_prime=data["y_prime"],
        x0_prime=45.0,
        y0_prime=170.0,
        z=data["z"],
        a=190.0,
        b=325.0,
        c=3.125,
        nu=0.35,
        h=520.0,
        E=8.0e9,
        theta_deg=15.0,
        alpha=0.8,
        station_names=data["station_names"],
        debug=False,
    )


def test_forward_model_returns_expected_number_of_rows():
    prediction = run_reference_forward_model()

    assert len(prediction) == 107


def test_forward_model_has_time_column():
    prediction = run_reference_forward_model()

    assert "time_s" in prediction.columns


def test_forward_model_component_contract():
    prediction = run_reference_forward_model()
    data = load_reference_inputs()

    expected = [
        f"{component}_{station}"
        for component in COMPONENTS
        for station in data["station_names"]
    ]

    for column in expected:
        assert column in prediction.columns

    assert len(expected) == 32


def test_forward_model_outputs_are_finite():
    prediction = run_reference_forward_model()

    strain_columns = [
        column
        for column in prediction.columns
        if column != "time_s"
    ]

    values = prediction[strain_columns].to_numpy(dtype=float)

    assert np.isfinite(values).all()


def test_forward_model_is_deterministic():
    prediction_1 = run_reference_forward_model()
    prediction_2 = run_reference_forward_model()

    strain_columns = [
        column
        for column in prediction_1.columns
        if column != "time_s"
    ]

    np.testing.assert_allclose(
        prediction_1[strain_columns].to_numpy(dtype=float),
        prediction_2[strain_columns].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def test_forward_model_time_matches_observed_contract():
    prediction = run_reference_forward_model()
    data = load_reference_inputs()

    np.testing.assert_allclose(
        prediction["time_s"].to_numpy(dtype=float),
        data["time"],
        rtol=0.0,
        atol=0.0,
    )


def test_reference_prediction_is_nontrivial():
    prediction = run_reference_forward_model()

    strain_columns = [
        column
        for column in prediction.columns
        if column != "time_s"
    ]

    values = prediction[strain_columns].to_numpy(dtype=float)

    assert np.max(np.abs(values)) > 0.0
    assert np.std(values) > 0.0