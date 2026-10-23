from types import SimpleNamespace
import numpy as np
import pytest
from avant_model.inversion.coupled_likelihood import (
    CoupledStrainLikelihood, gaussian_log_likelihood,
)
from avant_model.inversion.coupled_observations import StrainObservations


class DummyCoupledForward:
    channels = ('eXX_S01', 'eYY_S07', 'eXX_S02', 'eYY_S08')

    def predict(self, K_m_s, Ss_per_m, observation_times_h):
        t = np.asarray(observation_times_h, dtype=float)
        # Depends on both parameters; test is independent of numerical FV code.
        y = np.column_stack((t * K_m_s * 1e6,
                             t * Ss_per_m * 1e6,
                             t * (K_m_s + Ss_per_m) * 1e6,
                             t * Ss_per_m * 1e6))
        return SimpleNamespace(times_h=t, channels=self.channels, strain_ns=y)


def observations():
    model = DummyCoupledForward()
    predicted = model.predict(1e-6, 1e-6, [1., 2.])
    obs = predicted.strain_ns.copy()
    obs[:, 1] += 10.0  # holdout-only residual must not change likelihood
    return StrainObservations(predicted.times_h, predicted.channels, obs, 2.0,
                              np.array([True, False, True, False]),
                              np.array([False, True, False, True]))


def test_gaussian_normalization():
    expected = -0.5 * (3 / 2)**2 - np.log(2) - 0.5 * np.log(2*np.pi)
    assert gaussian_log_likelihood([3.], [0.], 2.) == pytest.approx(expected)


def test_likelihood_uses_only_training_channels():
    obs = observations()
    like = CoupledStrainLikelihood(DummyCoupledForward(), obs)
    expected = gaussian_log_likelihood(obs.observed_ns[:, obs.training_mask],
                                       DummyCoupledForward().predict(1e-6, 1e-6, obs.times_h).strain_ns[:, obs.training_mask],
                                       2.0)
    assert like([-6., -6.]) == pytest.approx(expected)


def test_invalid_parameter_returns_negative_infinity():
    like = CoupledStrainLikelihood(DummyCoupledForward(), observations())
    assert like([-8., -6.]) == -np.inf


def test_channel_mismatch_rejected():
    obs = observations()
    bad = DummyCoupledForward()
    bad.channels = tuple(reversed(bad.channels))
    with pytest.raises(ValueError, match='channel ordering'):
        CoupledStrainLikelihood(bad, obs)


def test_truth_beats_strongly_incorrect_candidate_without_training_noise():
    like = CoupledStrainLikelihood(DummyCoupledForward(), observations())
    assert like([-6., -6.]) > like([-5.0, -6.0])
