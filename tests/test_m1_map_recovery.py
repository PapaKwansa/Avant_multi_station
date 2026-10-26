"""Exact, deterministic mathematical tests for bounded M1 synthetic MAP recovery.

The tests use lightweight mathematical stand-ins for the optimizer interface;
production-kernel integration occurs in the user's real AVANT repository.
"""
import numpy as np
import pytest

from avant_model.inversion.m1_map_recovery import (
    PARAMETER_NAMES, SEARCH_BOUNDS, CONSTRAINED_FIXED,
    ParameterSubset, ObservationLayout, check_bounds,
    station_train_holdout_masks, add_independent_noise,
    fit_multi_start, parameter_error_table,
)

TRUTH=np.array([-6.,np.log10(3e-7),-6.,0.,215.,400.,15.,53.,141.])


def channel_names():
    return [f'{c}_{s}' for c in ('eXX','eYY','eXY','eZZ')
            for s in ('S01','S02','S03','S04','S05','S06','S07','S08')]


def layout():
    t,h=station_train_holdout_masks(channel_names())
    return ObservationLayout(t,h,(0,1),(2,))


def test_search_bounds_physically_ordered_and_include_truth():
    b=check_bounds()
    assert b.shape==(9,2)
    assert np.all(b[:,0]<TRUTH)
    assert np.all(TRUTH<b[:,1])
    assert b[4,1] < b[5,0]
    assert b[0,0]>b[1,1]


def test_invalid_bounds_rejected():
    bad=dict(SEARCH_BOUNDS)
    bad['a_m']=(400,600)
    with pytest.raises(ValueError,match='canonical'):check_bounds(bad)


def test_subset_parameter_roundtrip():
    obj=ParameterSubset(TRUTH,CONSTRAINED_FIXED)
    assert len(obj.free_indices)==6
    assert obj.free_names==tuple(x for x in PARAMETER_NAMES if x not in CONSTRAINED_FIXED)
    test=TRUTH.copy();test[0]+=.02;test[2]-=.02
    assert np.allclose(obj.expand(obj.normalize(test)),test)


def test_constrained_oracle_keeps_exact_fixed_parameters():
    obj=ParameterSubset(TRUTH,CONSTRAINED_FIXED)
    for z in (np.zeros(6),np.ones(6),np.full(6,.5)):
        est=obj.expand(z)
        for nm in CONSTRAINED_FIXED:
            ix=PARAMETER_NAMES.index(nm)
            assert est[ix]==TRUTH[ix]


def test_full_parameterization():
    obj=ParameterSubset(TRUTH,())
    assert len(obj.free_indices)==9
    assert np.allclose(obj.expand(obj.normalize(TRUTH)),TRUTH)


@pytest.mark.parametrize('bad',[(),('b_m','b_m'),('not_a_parameter',)])
def test_bad_fixed_names(bad):
    if bad==():
        assert len(ParameterSubset(TRUTH,bad).free_indices)==9
    else:
        with pytest.raises(ValueError):ParameterSubset(TRUTH,bad)


def test_invalid_reference_values_rejected():
    with pytest.raises(ValueError):ParameterSubset(np.ones(8))
    bad=TRUTH.copy();bad[4]=1e4
    with pytest.raises(ValueError):ParameterSubset(bad)


def test_train_holdout_station_masks_do_not_overlap():
    t,h=station_train_holdout_masks(channel_names())
    assert t.sum()==24 and h.sum()==8
    assert not np.any(t&h)


def test_missing_station_invalid():
    with pytest.raises(ValueError):station_train_holdout_masks(['eXX_S01','eXX_S02'])


def test_observation_vector_contains_correct_gauges_and_channels():
    lay=layout();s=np.arange(64.).reshape(2,32);p=np.arange(6.).reshape(2,3)
    train=lay.extract(s,p,normalized=False)
    held=lay.extract(s,p,subset='holdout')
    assert train.shape==(2*24+2*2,)
    assert held.shape==(2*8+2*1,)
    assert np.array_equal(train[-4:],[0.,1.,3.,4.])
    assert np.array_equal(held[-2:],[2.,5.])
    std=lay.extract(s,p,normalized=True)
    assert np.allclose(std[:48],train[:48]/2.)
    assert np.allclose(std[-4:],train[-4:]/20000.)


def test_heldout_values_do_not_enter_training_objective():
    lay=layout()
    s=np.zeros((3,32));p=np.zeros((3,3))
    s2=s.copy();p2=p.copy()
    s2[:,lay.channel_holdout_mask]=1e7;p2[:,2]=1e9
    assert np.array_equal(lay.extract(s,p),lay.extract(s2,p2))
    assert not np.array_equal(lay.extract(s,p,subset='holdout'),lay.extract(s2,p2,subset='holdout'))


def test_noise_is_reproducible_and_independent_across_scales():
    lay=layout();s=np.zeros((7,32));p=np.zeros((7,3))
    a=add_independent_noise(s,p,lay,123)
    b=add_independent_noise(s,p,lay,123)
    c=add_independent_noise(s,p,lay,124)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert not np.array_equal(a[0],c[0])


def test_nonfinite_observation_rejected():
    lay=layout();s=np.ones((1,32));p=np.ones((1,3))
    s[0,0]=np.nan
    with pytest.raises(ValueError):lay.extract(s,p)


def test_parameter_error_table_exact():
    est=TRUTH.copy();est[4]+=5
    rows=parameter_error_table(TRUTH,est)
    assert len(rows)==9
    assert rows[4]['difference']==5.
    assert rows[1]['difference']==0.


def test_optimizer_recovers_small_linear_analog_with_two_starts():
    """Pass-through optimizer uses normalized physical bounds and true train split.

    The forward is a synthetic linear stand-in, not a real AVANT mechanics test.
    """
    lay=layout()
    truth=TRUTH.copy()
    subset=ParameterSubset(truth,CONSTRAINED_FIXED)
    def pred(theta):
        # Easily identifiable free parameters, no accidental signal in holdouts.
        dif=(theta[subset.free_indices]-truth[subset.free_indices])
        s=np.zeros((4,32));p=np.zeros((4,3))
        # Distinct identifiable signatures for each of the six free params.
        train_cols=np.flatnonzero(lay.channel_train_mask)
        s[0,train_cols[:6]]=dif * 10.0
        p[0,0]=dif[0] * 50000.
        p[0,1]=dif[1] * 50000.
        return s,p
    obs=pred(truth)
    best,all_starts,n_calls=fit_multi_start(pred,obs,lay,subset,n_starts=2,max_nfev=80,seed=1)
    assert len(all_starts)==2
    assert best is not None
    assert best['chi2'] < 1e-8
    assert n_calls>=best['nfev']
    got=np.asarray(best['fitted_parameters'])
    assert np.linalg.norm(got[subset.free_indices]-truth[subset.free_indices])<.1


@pytest.mark.parametrize('n,maxev,step',[(0,10,.01),(2,1,.01),(2,20,0),(2,20,.6)])
def test_invalid_optimization_options(n,maxev,step):
    lay=layout();subset=ParameterSubset(TRUTH)
    obs=(np.zeros((1,32)),np.zeros((1,3)))
    with pytest.raises(ValueError):fit_multi_start(lambda t:obs,obs,lay,subset,
                                                   n_starts=n,max_nfev=maxev,diff_step=step)


def test_no_optimizers_are_silent_success():
    lay=layout();subset=ParameterSubset(TRUTH)
    obs=(np.zeros((1,32)),np.zeros((1,3)))
    def bad(theta):
        raise ValueError('test failure')
    best,all_runs, calls = fit_multi_start(bad,obs,lay,subset,n_starts=1,max_nfev=5)
    assert best is None
    assert not all_runs[0]['success']
    assert 'test failure' in all_runs[0]['message']
