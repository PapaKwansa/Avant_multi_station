"""Deterministic small-matrix smoke test, not an AVANT or COMSOL inversion."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from avant_model.inversion.marginalized_pressure_likelihood import marginalized_pressure_likelihood

H = np.array([[1e-5, 2e-6], [3e-6, -4e-6], [0., 7e-6]])
d = np.array([2., -1., .5])
mu = np.zeros(2)
C = np.eye(2) * (1e6)**2
R = np.eye(3) * 4.
result = marginalized_pressure_likelihood(H, d, prior_mean_pa=mu,
    prior_covariance_pa2=C, noise_covariance_ns2=R)
print('Synthetic marginal log likelihood:', result.log_likelihood)
print('Conditional pressure means (Pa):', result.conditional_mean_pa)
print('Conditional pressure SD (Pa):', np.sqrt(np.diag(result.conditional_covariance_pa2)))
print('This tests the statistical module only; no DREAM or COMSOL data are used.')
