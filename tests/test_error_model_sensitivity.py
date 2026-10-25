"""Unit tests for covariance sensitivity, separate from live Stage 4A data."""
from __future__ import annotations

import numpy as np
import pytest

from avant_model.inversion.error_model_sensitivity import (
    temporal_correlation_matrix,
    whiten_saved_jacobian,
    evaluate_error_case,
)


@pytest.fixture
def example():
    times = np.array([0., 0.5, 2., 5., 12.])
    s = np.array([
        [1., .1], [1.2, .3],
        [2., .2], [.9, .4],
        [1.6, .8], [1.1, .5],
        [1.4, .6], [1., .9],
        [1.3, 1.2], [1.2, 1.5],
    ])
    p = np.array([[2., .2], [1., 1.], [3., .5], [2., 1.5], [1., 2.]])
    return times, s, p


def test_independence_correlation_is_identity(example):
    t, _, _ = example
    np.testing.assert_array_equal(temporal_correlation_matrix(t, 0), np.eye(len(t)))


def test_exponential_correlation_uses_real_time_distance():
    r = temporal_correlation_matrix([0., 1., 5.], 2.)
    np.testing.assert_allclose(r[0,1], np.exp(-.5))
    np.testing.assert_allclose(r[1,2], np.exp(-2.))
    np.testing.assert_allclose(r[0,2], np.exp(-2.5))
    assert np.linalg.eigvalsh(r).min() > 0


def test_no_timestep_assumption_or_channel_mixing(example):
    t, s, _ = example
    out = whiten_saved_jacobian(s, t, 2, correlation_tau_h=2.)
    lower = np.linalg.cholesky(temporal_correlation_matrix(t, 2.))
    expected = np.linalg.solve(lower, s.reshape(5, 2, 2)[:, 0, :])
    np.testing.assert_allclose(out.reshape(5,2,2)[:,0,:], expected)


def test_diagonal_noise_scaling(example):
    t, s, _ = example
    np.testing.assert_allclose(whiten_saved_jacobian(s,t,2,2.),s/2)


def test_explicit_covariance_inverse_fisher(example):
    t,s,_=example
    corr=temporal_correlation_matrix(t,1.)
    w=whiten_saved_jacobian(s,t,2,correlation_tau_h=1.)
    j=s.reshape(5,2,2)
    # Independent channels, each with its own correlated times.
    f_exp=sum(j[:,c,:].T@np.linalg.solve(corr,j[:,c,:]) for c in range(2))
    np.testing.assert_allclose(w.T@w, f_exp,rtol=1e-12,atol=1e-12)


def test_baseline_equals_stacked_independent_analysis(example):
    t,s,p=example
    case=evaluate_error_case(s,p,t,2,1)
    np.testing.assert_allclose(case['information']['Joint']['fisher_information'],
                               np.vstack((s,p)).T@np.vstack((s,p)))
    assert case['information']['Joint']['numerical_rank']==2


def test_information_additive_for_correlated_case(example):
    t,s,p=example
    case=evaluate_error_case(s,p,t,2,1,strain_tau_h=2.,pressure_tau_h=1.)
    a=np.asarray(case['information']['Strain only']['fisher_information'])
    b=np.asarray(case['information']['Pressure only']['fisher_information'])
    j=np.asarray(case['information']['Joint']['fisher_information'])
    np.testing.assert_allclose(j,a+b,rtol=1e-12)


def test_joint_sd_not_greater_than_strain_only(example):
    t,s,p=example
    case=evaluate_error_case(s,p,t,2,1,pressure_factor=5.,strain_tau_h=2.)
    sdj=case['information']['Joint']['local_sd_log10']
    sds=case['information']['Strain only']['local_sd_log10']
    assert np.all(np.asarray(sdj)<=np.asarray(sds)+1e-12)


def test_double_both_noise_scales_doubles_all_sds(example):
    t,s,p=example
    a=evaluate_error_case(s,p,t,2,1)
    b=evaluate_error_case(s,p,t,2,1,strain_factor=2.,pressure_factor=2.)
    for key in ('Strain only','Pressure only','Joint'):
        np.testing.assert_allclose(b['information'][key]['local_sd_log10'],
                                   2*np.asarray(a['information'][key]['local_sd_log10']),rtol=1e-12)


def test_strain_and_pressure_tau_asymmetric(example):
    t,s,p=example
    a=evaluate_error_case(s,p,t,2,1,strain_tau_h=6.)
    b=evaluate_error_case(s,p,t,2,1,pressure_tau_h=6.)
    assert not np.allclose(a['information']['Joint']['fisher_information'],
                           b['information']['Joint']['fisher_information'])


@pytest.mark.parametrize('times',[[0.,0.,1.], [0.,3.,2.], [0.,np.nan,1.], [0.], [[0.,1.]]])
def test_reject_bad_times(times):
    with pytest.raises(ValueError):
        temporal_correlation_matrix(times)


@pytest.mark.parametrize('tau',[-1.,float('nan'),float('inf')])
def test_reject_bad_correlation_scale(tau):
    with pytest.raises(ValueError):
        temporal_correlation_matrix([0.,2.],tau)


@pytest.mark.parametrize('mult',[0.,-1.,float('nan'),float('inf')])
def test_reject_bad_noise_multiplier(example,mult):
    t,s,_=example
    with pytest.raises(ValueError):
        whiten_saved_jacobian(s,t,2,noise_multiplier=mult)


def test_reject_channel_mismatch(example):
    t,s,_=example
    with pytest.raises(ValueError):
        whiten_saved_jacobian(s,t,3)


def test_reject_nonfinite_jacobian(example):
    t,s,_=example
    bad=s.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):
        whiten_saved_jacobian(bad,t,2)


def test_reject_noninteger_channels(example):
    t,s,_=example
    with pytest.raises(ValueError):
        whiten_saved_jacobian(s,t,2.5)
