import numpy as np
import pytest
from avant_model.inversion.marginalized_pressure_likelihood import (
    marginalized_pressure_likelihood as marginal,
    reference_difference_covariance,
)


def example():
    H = np.array([[1e-5, 2e-6], [3e-6, -4e-6], [0., 7e-6]])
    d = np.array([2., -1., 0.5])
    mu = np.array([100000., -200000.])
    C = np.array([[2e10, 3e9], [3e9, 1e10]])
    R = np.array([[4., .5, 0.], [.5, 9., 1.], [0., 1., 5.]])
    return H, d, mu, C, R


def test_against_direct_marginal_normal():
    H, d, mu, C, R = example()
    result = marginal(H, d, prior_mean_pa=mu, prior_covariance_pa2=C, noise_covariance_ns2=R)
    S = R + H @ C @ H.T
    residual = d - H @ mu
    sign, ld = np.linalg.slogdet(S)
    assert sign == 1
    expected = -.5 * (len(d)*np.log(2*np.pi) + ld + residual @ np.linalg.solve(S, residual))
    np.testing.assert_allclose(result.log_likelihood, expected, rtol=1e-12, atol=1e-12)


def test_conditional_posterior():
    H, d, mu, C, R = example()
    result = marginal(H, d, prior_mean_pa=mu, prior_covariance_pa2=C, noise_covariance_ns2=R)
    S = R + H @ C @ H.T
    mean = mu + C @ H.T @ np.linalg.solve(S, d - H @ mu)
    cov = C - C @ H.T @ np.linalg.solve(S, H @ C)
    np.testing.assert_allclose(result.conditional_mean_pa, mean, rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(result.conditional_covariance_pa2, cov, rtol=1e-12, atol=1e-5)


def test_null_column_keeps_prior():
    H, d, mu, C, R = example()
    H[:, 1] = 0
    C = np.diag([1e10, 2e10])
    result = marginal(H, d, prior_mean_pa=mu, prior_covariance_pa2=C, noise_covariance_ns2=R)
    assert result.conditional_mean_pa[1] == pytest.approx(mu[1])
    assert result.conditional_covariance_pa2[1,1] == pytest.approx(C[1,1])


def test_bad_covariances_rejected():
    H, d, mu, C, R = example()
    with pytest.raises(ValueError):
        marginal(H,d,prior_mean_pa=mu,prior_covariance_pa2=-C,noise_covariance_ns2=R)
    with pytest.raises(ValueError):
        marginal(H,d,prior_mean_pa=mu,prior_covariance_pa2=C,noise_covariance_ns2=np.zeros_like(R))
    with pytest.raises(ValueError):
        marginal(H,d,prior_mean_pa=mu,prior_covariance_pa2=C,noise_covariance_ns2=R[:2,:2])


def test_shared_reference_blocks():
    R0 = np.diag([2., 3.])
    R = reference_difference_covariance(R0, 3)
    assert R.shape == (6,6)
    np.testing.assert_allclose(R[:2,:2], 2*R0)
    np.testing.assert_allclose(R[:2,2:4], R0)
    np.linalg.cholesky(R)


def test_joint_time_marginal_against_direct():
    H, d, mu, C, R = example()
    T = 3
    HH = np.kron(np.eye(T), H)
    dd = np.tile(d, T)
    mm = np.tile(mu, T)
    CC = np.kron(np.eye(T), C)
    RR = reference_difference_covariance(R, T)
    res = marginal(HH,dd,prior_mean_pa=mm,prior_covariance_pa2=CC,noise_covariance_ns2=RR)
    S = RR + HH @ CC @ HH.T
    z = dd - HH @ mm
    expected = -.5*(len(dd)*np.log(2*np.pi)+np.linalg.slogdet(S)[1]+z@np.linalg.solve(S,z))
    np.testing.assert_allclose(res.log_likelihood,expected,rtol=1e-12)
