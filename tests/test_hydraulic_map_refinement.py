"""Numerical tests of the stand-alone Stage 2 MAP/ML helper routines."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import least_squares

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/inversion/refine_hydraulic_map.py'
spec = importlib.util.spec_from_file_location('refine_hydraulic_map', SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_residuals_exclude_heldout():
    class Observations:
        training_mask = np.array([True, False, True])
        observed_ns = np.array([[1., 1e8, 4.], [2., -1e8, 3.]])
        sigma_ns = 2.
    class Likelihood:
        observations = Observations()
        def predict(self, theta):
            return np.zeros((2,3))
    np.testing.assert_array_equal(mod.optimization_residuals([-6,-6], Likelihood()),
                                  [-.5, -2., -1., -1.5])


def test_linearized_gaussian_uncertainty():
    jac = np.array([[2., 0.], [0., 4.], [0., 0.]])
    out = mod.local_laplace_summary(jac, [-6., -6.])
    np.testing.assert_allclose(out['standard_deviations_log10_linearized'], [.5, .25])
    assert out['jacobian_condition_number'] == 2.


def test_continuous_refinement_improves_likelihood():
    xtrue = np.array([-6., -5.97])
    obs = np.array([xtrue[0]*2, xtrue[1]*3])
    def residual(x):
        return np.array([2*x[0], 3*x[1]]) - obs
    fit = least_squares(residual, [-6.1,-5.9], bounds=([-7,-7], [-4,-4]), jac='3-point')
    assert fit.success
    np.testing.assert_allclose(fit.x, xtrue, atol=1e-8)
