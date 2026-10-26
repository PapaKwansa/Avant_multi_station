"""Mathematical tests for add-only M1 publication and independent Sobol.

No production analytical strain kernel is replaced or modified.
"""
import numpy as np
import pytest
from scripts.postprocessing.m1_publication_core import (
    membership_probability, posterior_predictive_bands, sobol_jansen_saltelli,
    sobol_bootstrap, summarize, transformed_independent_priors,
)


def states(n=10):
    x=np.zeros((n,9))
    x[:,4]=1.;x[:,5]=2.;x[:,6]=0.;x[:,7]=0.;x[:,8]=0.
    return x


def test_summary_shape_and_truth():
    x=states()
    s=summarize(x)
    assert s['median'].shape==(9,)
    assert s['q025'][4]==1
    assert s['q975'][5]==2


def test_summary_rejects_wrong_n():
    with pytest.raises(ValueError):summarize(np.ones((1,9)))


def test_geometry_membership_exact_axes():
    p=membership_probability(states(),[-2.,0.,2.],[-3.,0.,3.])
    assert p.shape==(3,3)
    assert p[1,1]==1.
    assert np.count_nonzero(p)==1


def test_geometry_membership_probabilistic_mixture():
    x=states(10)
    x[5:,7]=10.
    p=membership_probability(x,[0],[0])
    assert p[0,0]==pytest.approx(.5)


def test_geometry_orientation_and_translation():
    x=states()
    x[:,6]=90.
    x[:,7]=3.;x[:,8]=2.
    p=membership_probability(x,[3.,5.],[2.])
    assert np.array_equal(p,[[1.,1.]])


def test_geometry_rejects_invalid_b_a():
    x=states();x[:,5]=.5
    with pytest.raises(ValueError):membership_probability(x,[0],[0])


def test_gaussian_latent_vs_noisy_width():
    s=np.full((100,3,2),20.)
    p=np.full((100,3,1),1e5)
    bands=posterior_predictive_bands(s,p,2.,2e4,seed=42,replicas=8)
    assert np.all(bands['strain']['latent_q'][2]==20.)
    assert np.all(bands['strain']['noisy_q'][2]>20.)
    assert np.all(bands['pressure']['noisy_q'][0]<100000.)


def test_predictive_reproducible():
    s=np.zeros((11,2,3));p=np.ones((11,2,1))
    a=posterior_predictive_bands(s,p,2.,2e4,seed=5)
    b=posterior_predictive_bands(s,p,2.,2e4,seed=5)
    assert np.array_equal(a['strain']['noisy_q'],b['strain']['noisy_q'])


def test_predictive_invalid_shapes():
    with pytest.raises(ValueError):
        posterior_predictive_bands(np.zeros((4,2,3)),np.ones((3,2,1)),2.,2e4)


def test_sobol_constant_other_inputs():
    # Additive f=x0, measured under independent standard uniform inputs.
    rng=np.random.default_rng(123)
    a=rng.random((20000,3));b=rng.random((20000,3))
    fa=a[:,0,None];fb=b[:,0,None]
    fab=np.empty((20000,3,1))
    for i in range(3):
        ab=a.copy();ab[:,i]=b[:,i];fab[:,i,0]=ab[:,0]
    s1,st=sobol_jansen_saltelli(fa,fb,fab)
    assert s1[0,0]==pytest.approx(1.,abs=.045)
    assert st[0,0]==pytest.approx(1.,abs=.045)
    assert np.max(abs(st[1:,0]))<.03
    assert np.max(abs(s1[1:,0]))<.03


def test_sobol_additive_known_weights():
    rng=np.random.default_rng(44)
    a=rng.random((30000,2));b=rng.random((30000,2))
    f=lambda x:(x[:,0]+2*x[:,1])[:,None]
    fab=np.stack([f(np.column_stack([b[:,0],a[:,1]])),
                  f(np.column_stack([a[:,0],b[:,1]]))],axis=1)
    s1,st=sobol_jansen_saltelli(f(a),f(b),fab)
    assert s1[:,0]==pytest.approx([.2,.8],abs=.05)
    assert st[:,0]==pytest.approx([.2,.8],abs=.05)


def test_sobol_zero_variance_rejected():
    a=np.ones((8,1));b=a.copy();ab=np.ones((8,3,1))
    with pytest.raises(ValueError):sobol_jansen_saltelli(a,b,ab)


def test_bootstrap_shape_and_reproducibility():
    rng=np.random.default_rng(85);a=rng.random((300,2));b=rng.random((300,2))
    ya=a[:,0,None];yb=b[:,0,None]
    ab=np.stack([b[:,0,None],a[:,0,None]],axis=1)
    c1,c2=sobol_bootstrap(ya,yb,ab,draws=40,seed=12)
    d1,d2=sobol_bootstrap(ya,yb,ab,draws=40,seed=12)
    assert c1.shape==(2,2,1)
    assert c2.shape==(2,2,1)
    assert np.array_equal(c1,d1) and np.array_equal(c2,d2)


def test_uniform_transform():
    names=('a','b')
    priors={'parameters':{'a':{'bounds':[2.,4.], 'distribution':'uniform'},
                          'b':{'bounds':[10.,20.], 'distribution':'uniform'}}}
    t=transformed_independent_priors([[0.,1.],[.5,.5]],priors,names)
    assert np.allclose(t,[[2.,20.],[3.,15.]])


def test_truncated_normal_transform_not_truth_frozen():
    priors={'parameters':{'a':{'bounds':[0.,10.], 'distribution':'truncated_normal',
                               'mean':5.,'sd':2.}}}
    x=transformed_independent_priors([[.1],[.5],[.9]],priors,('a',))
    assert x[0,0]<x[1,0]<x[2,0]
    assert x[1,0]==pytest.approx(5.)


def test_unrecognized_measure_rejected():
    with pytest.raises(ValueError):
        transformed_independent_priors([[.5]],{'parameters':{}},('a',),measure='posterior')
