"""Statistical reference tests independent of the analytical mechanics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from avant_model.inversion.linear_pressure_inversion import linear_gaussian_posterior


def test_scalar_known_posterior():
    post = linear_gaussian_posterior([[2.]], [3.], prior_mean_pa=[0.],
                                    prior_covariance_pa2=[[4.]], noise_std_ns=1.)
    np.testing.assert_allclose(post.mean_pa, [6./4.25])
    np.testing.assert_allclose(post.covariance_pa2, [[1./4.25]])


def test_null_space_keeps_prior_uncertainty():
    post = linear_gaussian_posterior([[1.,1.]], [2.], prior_mean_pa=[0.,0.],
                                    prior_covariance_pa2=np.eye(2), noise_std_ns=.1)
    null = np.array([1.,-1.])/np.sqrt(2.)
    assert post.likelihood_nullity == 1
    assert null @ post.covariance_pa2 @ null == pytest.approx(1.)
    assert null @ post.mean_pa == pytest.approx(0., abs=1e-14)


def test_zero_kernel_returns_prior():
    C0 = np.array([[2.,.3],[.3,3.]])
    mean = [1.,2.]
    post = linear_gaussian_posterior(np.zeros((3,2)), [3.,4.,5.],
                                    prior_mean_pa=mean, prior_covariance_pa2=C0, noise_std_ns=1.)
    np.testing.assert_allclose(post.mean_pa, mean)
    np.testing.assert_allclose(post.covariance_pa2, C0)
    assert post.numerical_rank == 0


def test_correlated_errors_match_dense_gaussian_update():
    rng = np.random.default_rng(2)
    H = rng.normal(size=(7,4))*1e-4
    L = rng.normal(size=(4,4))
    C0 = (L@L.T+np.eye(4))*1e10
    m0 = rng.normal(size=4)*1e5
    R = np.eye(7)*2. + .4*np.ones((7,7))
    d = rng.normal(size=7)
    post = linear_gaussian_posterior(H,d,prior_mean_pa=m0,prior_covariance_pa2=C0,
                                    noise_covariance_ns2=R)
    W = C0@H.T@np.linalg.inv(R+H@C0@H.T)
    np.testing.assert_allclose(post.mean_pa, m0+W@(d-H@m0), rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(post.covariance_pa2, C0-W@H@C0, rtol=1e-7, atol=.05)
    np.testing.assert_allclose(post.predicted_observation_covariance_ns2,
                               post.predicted_signal_covariance_ns2+R)


def test_noise_vector_matches_covariance():
    H=np.array([[1.,2.],[3.,4.],[4.,1.]])
    d=[1.,2.,3.]; sd=np.array([1.,2.,3.])
    kw=dict(prior_mean_pa=[0.,0.],prior_covariance_pa2=np.eye(2))
    a=linear_gaussian_posterior(H,d,noise_std_ns=sd,**kw)
    b=linear_gaussian_posterior(H,d,noise_covariance_ns2=np.diag(sd**2),**kw)
    np.testing.assert_allclose(a.mean_pa,b.mean_pa)
    np.testing.assert_allclose(a.covariance_pa2,b.covariance_pa2)


def test_higher_noise_increases_conditional_uncertainty():
    H=np.array([[1.,2.],[3.,4.],[4.,1.]])
    kw=dict(prior_mean_pa=[0.,0.],prior_covariance_pa2=np.eye(2))
    a=linear_gaussian_posterior(H,[1.,2.,3.],noise_std_ns=.5,**kw)
    b=linear_gaussian_posterior(H,[1.,2.,3.],noise_std_ns=5.,**kw)
    assert np.all(b.std_pa > a.std_pa)


@pytest.mark.parametrize("noise", [0., -1., np.nan, [1.,2.,3.]])
def test_invalid_noise_rejected(noise):
    with pytest.raises(ValueError):
        linear_gaussian_posterior(np.eye(2),[1.,2.],prior_mean_pa=[0.,0.],
                                  prior_covariance_pa2=np.eye(2),noise_std_ns=noise)


def test_improper_prior_not_silently_fixed():
    with pytest.raises(ValueError):
        linear_gaussian_posterior(np.eye(2),[1.,2.],prior_mean_pa=[0.,0.],
                                  prior_covariance_pa2=np.zeros((2,2)),noise_std_ns=1.)


def test_exactly_one_noise_model():
    with pytest.raises(ValueError):
        linear_gaussian_posterior(np.eye(2),[1.,2.],prior_mean_pa=[0.,0.],
                                  prior_covariance_pa2=np.eye(2),noise_std_ns=1.,
                                  noise_covariance_ns2=np.eye(2))
