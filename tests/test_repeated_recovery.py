"""Tests for error draws and generalized least-squares synthetic refitting."""
from __future__ import annotations

import numpy as np
import pytest

from avant_model.inversion.repeated_recovery import (
    ErrorSpec, gaussian_temporal_noise, synthetic_observations, whiten_time_major,
    wald_covariance, fit_replicate, wilson_interval,
)
from avant_model.inversion.error_model_sensitivity import temporal_correlation_matrix


def test_error_specs_reject_invalid_scales():
    for cfg in [(-1,10,0,0),(1,0,0,0),(1,10,-1,0),(1,10,0,float("nan"))]:
        with pytest.raises(ValueError): ErrorSpec(*cfg)


def test_noise_reproducible():
    t=[0.,1.,3.,7.]
    a=gaussian_temporal_noise(t,2,3.,4.,np.random.default_rng(22))
    b=gaussian_temporal_noise(t,2,3.,4.,np.random.default_rng(22))
    np.testing.assert_array_equal(a,b)
    assert a.shape==(4,2)


def test_noise_matches_temporal_covariance_with_many_channels():
    t=np.array([0.,1.,4.])
    noises=gaussian_temporal_noise(t,35000,2.,3.,np.random.default_rng(3))
    expected=4*temporal_correlation_matrix(t,3.)
    np.testing.assert_allclose(np.cov(noises),expected,rtol=.04,atol=.05)


def test_no_cross_channel_identical_noise():
    n=gaussian_temporal_noise([0.,2.,5.],3,1.,6.,np.random.default_rng(10))
    assert not np.array_equal(n[:,0],n[:,1])


def test_noise_rejects_bad_channel_count():
    with pytest.raises(ValueError): gaussian_temporal_noise([0.,1.],0,1.,0.,np.random.default_rng(1))
    with pytest.raises(ValueError): gaussian_temporal_noise([0.,1.],1.4,1.,0.,np.random.default_rng(1))


def test_observation_shape_and_truth_unchanged():
    true_s=np.ones((3,2));true_p=np.ones((3,3))*5
    s,p=synthetic_observations(true_s,true_p,[0.,2.,5.],ErrorSpec(2.,20.,4.,5.),np.random.default_rng(1))
    assert s.shape==(3,2) and p.shape==(3,3)
    np.testing.assert_equal(true_s,np.ones((3,2)))
    np.testing.assert_equal(true_p,np.ones((3,3))*5)


def test_whitening_matches_explicit_covariance_inverse():
    t=np.array([0.,1.,4.,7.])
    rng=np.random.default_rng(12)
    e=rng.normal(size=(len(t),3))
    sigma=4.;tau=3.
    w=whiten_time_major(e,t,sigma,tau)
    c=temporal_correlation_matrix(t,tau)*sigma**2
    expected=sum(float(e[:,j].T@np.linalg.solve(c,e[:,j])) for j in range(3))
    np.testing.assert_allclose(np.sum(w*w),expected,rtol=1e-12)


def test_white_noise_identity():
    e=np.arange(12.).reshape(4,3)
    np.testing.assert_equal(whiten_time_major(e,[0,1,3,4],2.,0.),e/2.)


def test_wald_covariance_svd_matches_inverse():
    j=np.array([[3.,1.],[1.,4.],[0.,2.]])
    np.testing.assert_allclose(wald_covariance(j),np.linalg.inv(j.T@j),atol=1e-14)


def test_rank_deficient_wald_returns_none():
    assert wald_covariance(np.array([[1.,2.],[2.,4.],[3.,6.]])) is None


def test_wilson_handles_extremes():
    lo,hi=wilson_interval(0,10)
    assert lo==0 and hi<.5
    lo,hi=wilson_interval(10,10)
    assert hi==1 and lo>.5
    with pytest.raises(ValueError):wilson_interval(11,10)


def test_fit_linear_synthetic_models():
    times=np.array([0.,1.,3.,8.,15.])
    # Designed linear surrogate in theta space, independent from AVANT physics.
    def predict(x):
        p=np.outer(np.array([1.,2.,3.,4.,5.]),np.array([1.,2.,3.]))*x[0]
        s=np.outer(np.array([1.,3.,2.,4.,2.]),np.array([2.,-1.,3.,4.]))*x[1]
        return s,p
    truth=np.array([-.3,.7]);s,p=predict(truth)
    s+=.001*np.random.default_rng(2).normal(size=s.shape)
    p+=.001*np.random.default_rng(3).normal(size=p.shape)
    result=fit_replicate(predict,times,s,p,np.array([True,True,True,False]),
                         np.array([True,True,False]),ErrorSpec(.01,.01),
                         [0.,0.],[-2.,-2.],[2.,2.],max_nfev=40)
    np.testing.assert_allclose(result["estimate_log10"],truth,atol=.001)
    assert all(result["local_sd_log10"]>0)
    assert result["heldout_strain_rmse_ns"] is not None
    assert result["heldout_pressure_rmse_pa"] is not None


def test_fit_correlated_good_assumptions():
    times=np.array([0.,1.,3.,8.,15.])
    def predict(x):
        t=times[:,None]
        return t*(x[0]+1)+2*x[1]*np.ones((1,2)),t*(x[1]+1)+3*x[0]*np.ones((1,2))
    truth=np.array([-.3,.7]);s,p=predict(truth)
    result=fit_replicate(predict,times,s,p,[True,False],[True,False],
                         ErrorSpec(.01,.01,6.,6.),[0.,0.],[-2.,-2.],[2.,2.],mode="joint")
    np.testing.assert_allclose(result["estimate_log10"],truth,atol=1e-8)


def test_invalid_fitter_mode():
    with pytest.raises(ValueError):
        fit_replicate(lambda x:(np.ones((2,2)),np.ones((2,2))),[0.,1.],
                      np.ones((2,2)),np.ones((2,2)),[True,False],[True,False],
                      ErrorSpec(1.,1.),[0.,0.],[-1.,-1.],[1.,1.],mode="bad")
