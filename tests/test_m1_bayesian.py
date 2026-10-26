"""M1 Bayesian tests; no physical inference from the local stand-in kernel."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from avant_model.inversion.m1_bayesian import (
    WorkerM1Likelihood, assert_worker_pickles, generate_chain_starts,
    load_saved_m1, log_prior, M1BayesianLikelihood, normalize_dream,
    read_prior_config, sample_prior, sha256, validate_map_likelihood, M1GammaCachedForward,
)
from avant_model.inversion.m1_map_recovery import (
    PARAMETER_NAMES, ObservationLayout,
)
from avant_model.inversion.m1_posterior_diagnostics import (
    diagnose, predictive_bands, heldout_score,
)

PRIOR_PATH=Path(__file__).parents[1]/'configs/pressure_superposition/stage6/m1_bayesian_priors_001.json'
TRUTH=np.array([-6.,np.log10(3e-7),-6.,0.,215.,400.,15.,53.,141.])


def prior():return read_prior_config(PRIOR_PATH)


def test_priors_are_nine_free_and_explicitly_illustrative():
    p=prior()
    assert p['prior_provenance']=='illustrative_not_field_measured'
    assert tuple(p['parameters'])==PARAMETER_NAMES
    assert len(p['parameters'])==9
    assert p['parameters']['b_m']['mean'] != TRUTH[5]
    assert p['parameters']['y0_prime_m']['mean'] != TRUTH[8]


def test_prior_rejects_missing_parameter(tmp_path):
    doc=prior();doc['parameters'].pop('a_m')
    path=tmp_path/'bad.json';path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='nine'):read_prior_config(path)


def test_prior_rejects_unknown_source(tmp_path):
    doc=prior();doc['prior_provenance']='field_measurement'
    path=tmp_path/'bad.json';path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='illustrative'):read_prior_config(path)


@pytest.mark.parametrize('name',PARAMETER_NAMES)
def test_prior_support_is_valid(name):
    item=prior()['parameters'][name]
    lo,hi=item['bounds']
    assert lo<TRUTH[PARAMETER_NAMES.index(name)]<hi


def test_normalized_prior_and_out_of_bounds():
    p=prior();ll=log_prior(TRUTH,p)
    assert np.isfinite(ll)
    z=TRUTH.copy();z[0]=-1.
    assert log_prior(z,p)==-np.inf
    assert log_prior(np.ones(8),p)==-np.inf


def test_soft_prior_is_not_truth_delta_function():
    p=prior();rng=np.random.default_rng(33)
    samples=np.stack([sample_prior(p,rng) for _ in range(80)])
    assert samples.shape==(80,9)
    assert np.std(samples[:,5])>10
    assert np.std(samples[:,8])>10
    assert np.all(np.isfinite([log_prior(s,p) for s in samples]))


def test_soft_prior_normalization_numerical():
    p=prior()
    from scipy.integrate import trapezoid
    from scipy.stats import truncnorm
    for name in ('log10_K_outside','b_m','y0_prime_m'):
        item=p['parameters'][name];lo,hi=item['bounds']
        d=truncnorm((lo-item['mean'])/item['sd'],(hi-item['mean'])/item['sd'],
                    loc=item['mean'],scale=item['sd'])
        xx=np.linspace(lo,hi,10001)
        assert trapezoid(d.pdf(xx),xx)==pytest.approx(1.,rel=1e-7)


def test_seeded_chain_starts_are_varied_no_oracle_fixing():
    p=prior()
    a=generate_chain_starts(TRUTH,p,nchains=4,seed=11)
    b=generate_chain_starts(TRUTH,p,nchains=4,seed=11)
    assert np.array_equal(a,b)
    assert a.shape==(4,9)
    assert np.min(np.std(a,axis=0))>0
    assert all(np.isfinite(log_prior(row,p)) for row in a)


def test_bad_chain_count_rejected():
    with pytest.raises(ValueError,match='4 chains'):generate_chain_starts(TRUTH,prior(),nchains=3)


def test_worker_is_windows_pickleable():
    w=WorkerM1Likelihood('C:/safe/m1','C:/safe/config.json')
    assert assert_worker_pickles(w)
    assert pickle.loads(pickle.dumps(w))==w


def layout():
    return ObservationLayout(np.r_[np.ones(24,bool),np.zeros(8,bool)],
                             np.r_[np.zeros(24,bool),np.ones(8,bool)],(0,1),(2,))


class FakeSaved:
    def __init__(self):
        self.layout=layout()
        self.observed_strain_ns=np.zeros((3,32))
        self.observed_pressure_pa=np.zeros((3,3))


def test_likelihood_uses_train_only_and_noise_normalization():
    saved=FakeSaved()
    def pred(theta):
        s=np.full((3,32),2.0)
        p=np.full((3,3),20000.0)
        return s,p
    l=M1BayesianLikelihood(saved,pred)
    n=3*(24+2)
    assert l(TRUTH)==pytest.approx(-.5*n)
    saved.observed_strain_ns[:,24:]=1e10
    saved.observed_pressure_pa[:,2]=1e10
    assert l(TRUTH)==pytest.approx(-.5*n)


def test_likelihood_rejects_bad_parameters():
    l=M1BayesianLikelihood(FakeSaved(),lambda x:(np.zeros((3,32)),np.zeros((3,3))))
    assert l(np.ones(8))==-np.inf
    assert l(np.full(9,np.nan))==-np.inf
    assert l(np.r_[1.,TRUTH[1:]])==-np.inf


def test_existing_map_chi2_can_be_reproduced():
    saved=FakeSaved();x=TRUTH.copy()
    record={'fits':{'full':{'best':{'fitted_parameters':x.tolist(),'chi2':0.0}}}}
    ll=M1BayesianLikelihood(saved,lambda x:(np.zeros((3,32)),np.zeros((3,3))))
    theta,loglike=validate_map_likelihood(ll,record)
    assert np.array_equal(theta,x)
    assert loglike==0
    record['fits']['full']['best']['chi2']=1
    with pytest.raises(ValueError,match='mismatch'):validate_map_likelihood(ll,record)


def test_chain_shapes_checked():
    chains=[np.ones((30,9)) for _ in range(4)]
    lp=[np.ones(30) for _ in range(4)]
    a,b=normalize_dream(chains,lp,30,4)
    assert a.shape==(4,30,9) and b.shape==(4,30)
    with pytest.raises(ValueError):normalize_dream(chains,lp,31,4)


def test_converged_independent_chains_and_stuck_chains():
    rng=np.random.default_rng(12)
    x=rng.normal(size=(4,1200,9))
    d=diagnose(x,200)
    assert d['pass']
    for p in PARAMETER_NAMES:assert d['metrics'][p]['rank_folded_split_rhat']<1.01
    y=x.copy();y[0,:,0]+=8
    assert not diagnose(y,200)['pass']


def test_smoke_never_passes_convergence_minimum():
    rng=np.random.default_rng(101)
    d=diagnose(rng.normal(size=(4,80,9)),40)
    assert not d['pass']


def test_predictive_bands_and_heldout_coverage():
    rng=np.random.default_rng(4)
    x=np.zeros((40,3,4))
    b=predictive_bands(x,1.,rng,12)
    assert b['mean'].shape==(3,4)
    assert np.all(b['noisy_q95']>b['noisy_q05'])
    res=heldout_score(b,np.zeros((3,4)),np.array([False,True,False,True]))
    assert res['heldout_RMSE']==0
    assert res['heldout_count']==6


def test_sha_of_existing_prior():
    assert len(sha256(PRIOR_PATH))==64


def test_gamma_cache_only_recomputes_hydraulic_state_when_needed():
    from types import SimpleNamespace
    class FakeModel:
        def __init__(self):self.count=0
        def predict(self,state,times_h):
            self.count+=1
            return SimpleNamespace(strain_ns=np.ones((len(times_h),2))*3,
                                   pressure_gauges_pa=np.ones((len(times_h),3))*40)
    model=FakeModel()
    fn=M1GammaCachedForward(model,[0.,1.,2.])
    a,p=fn(TRUTH)
    q=TRUTH.copy();q[3]=np.log10(2.)
    b,p2=fn(q)
    assert np.array_equal(b,2*a)
    assert np.array_equal(p,p2)
    assert fn.full_forward_solves==1
    q[0]+=.01
    fn(q)
    assert fn.full_forward_solves==2
