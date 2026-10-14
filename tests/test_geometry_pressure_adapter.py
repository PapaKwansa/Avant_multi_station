import numpy as np
import pytest

from avant_model.inversion.geometry_pressure_adapter import evaluate_geometry_pressure
from avant_model.inversion.marginalized_pressure_likelihood import marginalized_pressure_likelihood
from avant_model.model.pressure_superposition import Cuboid, ElasticProperties, coarse_block_basis


def inputs():
    geom = Cuboid(215, 400, 2.5, 522.5, 53, 141, 15)
    elastic = ElasticProperties(8e9, .25, .7)
    names = ('S01', 'S02')
    channels = tuple(f'{c}_{s}' for c in ('eXX','eYY','eXY','eZZ') for s in names)
    return dict(geometry=geom, elastic=elastic,
                x_prime=np.array([250., -500.]), y_prime=np.array([433., 0.]),
                z=np.array([30., 30.]), station_names=names,
                observed_ns=np.arange(8., dtype=float), observed_channels=channels,
                fine_splits=(2,2,1), pressure_splits=(2,2,1),
                prior_mean_pa=np.zeros(4), prior_covariance_pa2=np.eye(4)*1e12,
                noise_covariance_ns2=np.eye(8)*8.)


def test_exact_likelihood_and_posterior_against_direct():
    kw=inputs()
    result=evaluate_geometry_pressure(**kw)
    direct=marginalized_pressure_likelihood(result.design_ns_per_pa,kw['observed_ns'],
        prior_mean_pa=kw['prior_mean_pa'], prior_covariance_pa2=kw['prior_covariance_pa2'],
        noise_covariance_ns2=kw['noise_covariance_ns2'])
    assert np.isclose(result.log_likelihood,direct.log_likelihood)
    np.testing.assert_allclose(result.conditional_mean_pa,direct.conditional_mean_pa)
    np.testing.assert_allclose(result.conditional_covariance_pa2,direct.conditional_covariance_pa2)


def test_geometry_changes_design():
    kw=inputs()
    a=evaluate_geometry_pressure(**kw)
    kw['geometry']=Cuboid(190, 400, 2.5, 522.5, 53, 141, 15)
    b=evaluate_geometry_pressure(**kw)
    assert not np.allclose(a.design_ns_per_pa,b.design_ns_per_pa,atol=0,rtol=1e-6)


def test_channel_order_mismatch_fails():
    kw=inputs()
    kw['observed_channels']=tuple(reversed(kw['observed_channels']))
    with pytest.raises(ValueError,match='channel order'):
        evaluate_geometry_pressure(**kw)


def test_wrong_prior_size_fails():
    kw=inputs()
    kw['prior_mean_pa']=np.zeros(3)
    with pytest.raises(ValueError):
        evaluate_geometry_pressure(**kw)


def test_invalid_coarse_partition_fails():
    kw=inputs()
    kw['pressure_splits']=(3,2,1)
    with pytest.raises(ValueError):
        evaluate_geometry_pressure(**kw)
