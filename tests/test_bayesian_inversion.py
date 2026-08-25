"""Regression tests for the AVANT Bayesian inversion likelihood module."""

import numpy as np
import pandas as pd
import pytest

from avant_model.inversion.bayesian_inversion_multi_station import (
    PHYSICAL_PARAMETER_NAMES,
    NUISANCE_PARAMETER_NAMES,
    SAMPLED_PARAMETER_NAMES,
    bulk_modulus,
    cuboid_volume,
    flatten_dataframe_columns,
    gaussian_log_likelihood,
    parameters_are_valid,
    unpack_sampled_parameters,
    volume_forward,
)


def test_parameter_contract():
    assert PHYSICAL_PARAMETER_NAMES == (
        "a",
        "b",
        "h",
        "theta_deg",
        "x0_prime",
        "y0_prime",
    )

    assert NUISANCE_PARAMETER_NAMES == (
        "log10_sigma_strain",
    )

    assert SAMPLED_PARAMETER_NAMES == (
        "a",
        "b",
        "h",
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "log10_sigma_strain",
    )


def test_unpack_sampled_parameters_and_sigma_transform():
    sample = [
        190.0,
        325.0,
        520.0,
        15.0,
        45.0,
        170.0,
        2.0,
    ]

    result = unpack_sampled_parameters(sample)

    assert result["a"] == 190.0
    assert result["b"] == 325.0
    assert result["h"] == 520.0
    assert result["theta_deg"] == 15.0
    assert result["x0_prime"] == 45.0
    assert result["y0_prime"] == 170.0
    assert result["log10_sigma_strain"] == 2.0

    # sigma = 10^2 = 100 nstrain.
    assert result["sigma_strain"] == 100.0


def test_unpack_rejects_wrong_parameter_count():
    with pytest.raises(ValueError):
        unpack_sampled_parameters(
            [190.0, 325.0, 520.0]
        )


def test_valid_parameter_state():
    parameters = unpack_sampled_parameters(
        [
            190.0,
            325.0,
            520.0,
            15.0,
            45.0,
            170.0,
            2.0,
        ]
    )

    assert parameters_are_valid(parameters)


def test_nonpositive_physical_parameter_is_invalid():
    parameters = unpack_sampled_parameters(
        [
            -190.0,
            325.0,
            520.0,
            15.0,
            45.0,
            170.0,
            2.0,
        ]
    )

    assert not parameters_are_valid(parameters)


def test_bulk_modulus_reference_value():
    E = 8.0e9
    nu = 0.35

    expected = E / (
        3.0 * (1.0 - 2.0 * nu)
    )

    assert np.isclose(
        bulk_modulus(E, nu),
        expected,
    )


def test_bulk_modulus_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        bulk_modulus(-1.0, 0.35)

    with pytest.raises(ValueError):
        bulk_modulus(8.0e9, 0.5)


def test_cuboid_volume_uses_full_dimensions():
    """
    a, b and c are semi-dimensions, so V0 = 8abc.
    """

    volume = cuboid_volume(
        190.0,
        325.0,
        3.125,
    )

    expected = (
        8.0
        * 190.0
        * 325.0
        * 3.125
    )

    assert np.isclose(
        volume,
        expected,
    )


def test_cuboid_volume_rejects_nonpositive_dimension():
    with pytest.raises(ValueError):
        cuboid_volume(
            190.0,
            0.0,
            3.125,
        )


def test_volume_forward_zero_pressure_returns_initial_volume():
    initial_volume = cuboid_volume(
        190.0,
        325.0,
        3.125,
    )

    predicted = volume_forward(
        a=190.0,
        b=325.0,
        c=3.125,
        E=8.0e9,
        nu=0.35,
        delta_P=0.0,
    )

    assert np.isclose(
        float(predicted),
        initial_volume,
    )


def test_gaussian_log_likelihood_matches_closed_form():
    residual = np.array(
        [1.0, -2.0, 0.5]
    )

    sigma = 2.5

    expected = (
        -0.5
        * np.sum(
            (residual / sigma) ** 2
        )
        - residual.size * np.log(sigma)
        - 0.5
        * residual.size
        * np.log(2.0 * np.pi)
    )

    actual = gaussian_log_likelihood(
        residual,
        sigma,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_gaussian_log_likelihood_rejects_invalid_sigma():
    residual = np.array(
        [1.0, 2.0]
    )

    assert gaussian_log_likelihood(
        residual,
        0.0,
    ) == -np.inf

    assert gaussian_log_likelihood(
        residual,
        -1.0,
    ) == -np.inf


def test_gaussian_log_likelihood_rejects_nonfinite_residual():
    residual = np.array(
        [1.0, np.nan]
    )

    assert gaussian_log_likelihood(
        residual,
        1.0,
    ) == -np.inf


def test_gaussian_log_likelihood_rejects_empty_residual():
    with pytest.raises(ValueError):
        gaussian_log_likelihood(
            [],
            1.0,
        )


def test_flatten_dataframe_columns_preserves_requested_order():
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [10.0, 20.0],
            "c": [100.0, 200.0],
        }
    )

    result = flatten_dataframe_columns(
        dataframe,
        ["b", "a"],
    )

    expected = np.array(
        [
            10.0,
            20.0,
            1.0,
            2.0,
        ]
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_flatten_dataframe_columns_rejects_missing_column():
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0],
        }
    )

    with pytest.raises(KeyError):
        flatten_dataframe_columns(
            dataframe,
            ["a", "missing"],
        )