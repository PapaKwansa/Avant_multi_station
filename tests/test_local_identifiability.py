"""Lightweight unit tests for numerical identifiability helpers."""
import numpy as np
import pytest

from avant_model.inversion.local_identifiability import (
    analyze_jacobian, central_difference_jacobian, compare_jacobian_steps,
)


def test_linear_vector_jacobian():
    matrix = np.array([[2.0, 0.3], [-5.0, 1.2], [7.0, -2.0]])
    jac = central_difference_jacobian(lambda x: matrix @ x, [-6.0, -6.0], [1e-3, 1e-3])
    np.testing.assert_allclose(jac, matrix, atol=1e-10)


def test_matrix_shaped_observables_flattened():
    jac = central_difference_jacobian(lambda x: np.array([[x[0], x[1]], [x[0]+x[1], 2*x[0]]]),
                                      [2.0, 3.0], [0.001, 0.001])
    np.testing.assert_allclose(jac, [[1, 0], [0, 1], [1, 1], [2, 0]], atol=1e-10)


def test_nonlinear_derivative_and_step_check():
    f = lambda x: np.array([x[0]**2, np.exp(x[1])])
    j1 = central_difference_jacobian(f, [1.0, 0.2], [0.002, 0.002])
    j2 = central_difference_jacobian(f, [1.0, 0.2], [0.001, 0.001])
    np.testing.assert_allclose(j2, [[2, 0], [0, np.exp(0.2)]], rtol=1e-6, atol=1e-6)
    assert compare_jacobian_steps(j1, j2) < 1e-6


def test_complementarity_recovers_rank():
    s = analyze_jacobian([[2.0, 0.0], [3.0, 0.0]])
    p = analyze_jacobian([[0.0, 5.0], [0.0, 7.0]])
    joint = analyze_jacobian([[2.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 7.0]])
    assert s["numerical_rank"] == 1 and p["numerical_rank"] == 1
    assert joint["numerical_rank"] == 2
    assert s["local_covariance"] is None
    np.testing.assert_allclose(joint["local_correlation"], np.eye(2), atol=1e-12)


def test_information_additive_and_variance_contraction():
    strain = np.array([[1., 0.1], [2., 1.], [-1., 2.]])
    pressure = np.array([[4., 0.0], [0., 4.]])
    a = analyze_jacobian(strain)
    b = analyze_jacobian(pressure)
    joint = analyze_jacobian(np.vstack((strain, pressure)))
    np.testing.assert_allclose(joint["fisher_information"],
                               np.asarray(a["fisher_information"]) + np.asarray(b["fisher_information"]))
    assert all(j < s for j, s in zip(joint["local_sd_log10"], a["local_sd_log10"]))


def test_tradeoff_negative_correlation():
    result = analyze_jacobian([[2., 2.], [2., 2.2], [2., 1.8]])
    assert result["numerical_rank"] == 2
    assert result["local_correlation"][0][1] < -0.95


def test_underdetermined_no_false_full_rank():
    result = analyze_jacobian([[2.0, 3.0]])
    assert result["numerical_rank"] == 1
    assert result["condition_number"] is None
    assert result["local_covariance"] is None


@pytest.mark.parametrize("theta,steps", [([0, 1], [0, 0.1]), ([0, 1], [0.1]),
                                            ([np.nan, 1], [0.1, 0.1])])
def test_rejects_invalid_difference_inputs(theta, steps):
    with pytest.raises(ValueError):
        central_difference_jacobian(lambda x: x, theta, steps)


def test_rejects_differing_output_shape():
    with pytest.raises(ValueError):
        central_difference_jacobian(lambda x: np.ones(2 if x[0] > 0 else 3), [0., 1.], [0.1, 0.1])


def test_rejects_nonfinite_jacobian():
    with pytest.raises(ValueError):
        analyze_jacobian([[np.nan, 1.]])
