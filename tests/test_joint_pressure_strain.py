from types import SimpleNamespace
import numpy as np
import pytest
from avant_model.model.pressure_gauges import PressureGaugeOperator
from avant_model.inversion.coupled_likelihood import gaussian_log_likelihood
from avant_model.inversion.coupled_observations import StrainObservations
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.joint_pressure_strain import (
    JointPressureStrainLikelihood, PressureObservations, pressure_gauge_masks,
    synthetic_pressure_observations,
)


class DummyFV:
    def __init__(self):
        self.x_m = np.array([-1., 0., 1.])
        self.y_m = np.array([-1., 0., 1.])
        self.calls = 0

    def solve(self, K, Ss, times):
        self.calls += 1
        x,y = np.meshgrid(self.x_m,self.y_m)
        # Stress the xy->flatten convention; all gauges and mechanics share this array.
        p = np.stack([(K*1.e6*(1+x) + Ss*1.e6*(2+y))*10*(1+t) for t in times])
        return SimpleNamespace(times_h=np.asarray(times, float), x_m=self.x_m, y_m=self.y_m,
                               pressure_change_pa=p)


class DummyModel:
    def __init__(self):
        self.hydraulic = DummyFV()
        self.P = np.eye(9)
        self.H = np.array([[0,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0]],float)
        self.channels = ('eXX_S01', 'eXX_S07')

    def predict(self, K, Ss, times):
        fv = self.hydraulic.solve(K,Ss,times)
        result = fv.pressure_change_pa.reshape(len(times),-1) @ self.P.T @ self.H.T
        return SimpleNamespace(strain_ns=result)


def fixture_likelihood():
    model = DummyModel()
    times = np.array([0.,.5,1.])
    truth = model.predict(1e-6,1e-6,times).strain_ns
    strain = StrainObservations(times, model.channels, truth.copy(), 2.0,
                                np.array([True,False]), np.array([False,True]))
    op = PressureGaugeOperator(model.hydraulic.x_m, model.hydraulic.y_m,
                               np.array([-.5,.5]), np.array([.5,-.5]))
    pressure_truth = op.sample(model.hydraulic.solve(1e-6,1e-6,times).pressure_change_pa)
    pressure = PressureObservations(times, ('P01','P03'), [-.5,.5],[.5,-.5],
                                    pressure_truth.copy(), 4.0, [True,False],[False,True])
    return model, strain, pressure


def test_joint_likelihood_is_sum_of_normalized_blocks():
    model, strain, pressure = fixture_likelihood()
    ll = JointPressureStrainLikelihood(model,strain,pressure)
    before = model.hydraulic.calls
    got = ll([-6.,-6.])
    assert model.hydraulic.calls == before + 1
    expected = gaussian_log_likelihood(strain.observed_ns[:,[0]],strain.observed_ns[:,[0]],2.)
    expected += gaussian_log_likelihood(pressure.observed_pa[:,[0]],pressure.observed_pa[:,[0]],4.)
    assert got == pytest.approx(expected, abs=1e-12)


def test_joint_pressure_and_strain_see_same_single_fv_field():
    model, strain, pressure = fixture_likelihood()
    ll = JointPressureStrainLikelihood(model,strain,pressure)
    theta = [-6.1,-5.9]
    before = model.hydraulic.calls
    strain_pred, pressure_pred = ll.predict(theta)
    assert model.hydraulic.calls == before+1
    expected_strain = model.predict(10**theta[0],10**theta[1],strain.times_h).strain_ns
    np.testing.assert_allclose(strain_pred,expected_strain,rtol=0,atol=1e-12)
    assert pressure_pred.shape == pressure.observed_pa.shape


def test_holdout_values_excluded_from_likelihood():
    model,strain,pressure = fixture_likelihood()
    altered_s = strain.observed_ns.copy();altered_s[:,1] += 1.e6
    altered_p = pressure.observed_pa.copy();altered_p[:,1] += 1.e6
    new_s = StrainObservations(strain.times_h,strain.channels,altered_s,2.,
                                strain.training_mask,strain.holdout_mask)
    new_p = PressureObservations(pressure.times_h,pressure.gauge_names,
                                pressure.gauge_x_m,pressure.gauge_y_m,altered_p,4.,
                                pressure.training_mask,pressure.holdout_mask)
    assert JointPressureStrainLikelihood(model,strain,pressure)([-6,-6]) == pytest.approx(
           JointPressureStrainLikelihood(model,new_s,new_p)([-6,-6]))


def test_pressure_only_and_strain_only_ablation():
    model,strain,pressure = fixture_likelihood()
    both = JointPressureStrainLikelihood(model,strain,pressure)
    only_s = JointPressureStrainLikelihood(model,strain,pressure,include_pressure=False)
    only_p = JointPressureStrainLikelihood(model,strain,pressure,include_strain=False)
    theta = [-6.1,-5.9]
    assert both(theta) == pytest.approx(only_s(theta) + only_p(theta))
    with pytest.raises(ValueError):
        JointPressureStrainLikelihood(model,strain,pressure,include_strain=False,include_pressure=False)
    assert only_s.training_residuals(theta).size == 3
    assert only_p.training_residuals(theta).size == 3
    assert both.training_residuals(theta).size == 6


def test_invalid_parameter_returns_negative_infinity():
    model,strain,pressure = fixture_likelihood()
    ll = JointPressureStrainLikelihood(model,strain,pressure)
    assert ll([-8,-6]) == -np.inf


def test_disjoint_times_rejected():
    model,strain,pressure = fixture_likelihood()
    wrong = PressureObservations([0,.5,1.5], pressure.gauge_names,
                                 pressure.gauge_x_m,pressure.gauge_y_m,
                                 pressure.observed_pa,pressure.sigma_pa,
                                 pressure.training_mask,pressure.holdout_mask)
    with pytest.raises(ValueError,match='matching'):
        JointPressureStrainLikelihood(model,strain,wrong)


def test_synthetic_pressure_is_reproducible_and_noise_fixed():
    model,strain,_ = fixture_likelihood()
    kwargs = dict(gauge_names=('P01','P03'),gauge_x_m=[-.5,.5],gauge_y_m=[.5,-.5],
                  sigma_pa=10.,seed=123,holdout_names=('P03',))
    a,truth1=synthetic_pressure_observations(model,strain.times_h,1e-6,1e-6,**kwargs)
    b,truth2=synthetic_pressure_observations(model,strain.times_h,1e-6,1e-6,**kwargs)
    np.testing.assert_array_equal(a.observed_pa,b.observed_pa)
    np.testing.assert_array_equal(truth1,truth2)
    assert not np.array_equal(a.observed_pa,truth1)


def test_bad_pressure_gauge_masks_rejected():
    with pytest.raises(ValueError):
        pressure_gauge_masks(('P01','P03'),holdout_names=('P99',))
    with pytest.raises(ValueError):
        pressure_gauge_masks(('P01','P03'),holdout_names=('P01','P03'))


def test_duplicate_gauge_names_rejected():
    with pytest.raises(ValueError,match='unique'):
        PressureObservations([0,1], ('P01','P01'), [0.,1.], [0.,1.],
                             np.ones((2,2)),2.,[True,False],[False,True])
