from types import SimpleNamespace
import numpy as np
import pytest
from avant_model.inversion.coupled_observations import (
    station_masks, make_synthetic_observations, StrainObservations,
)


CHANNELS = ('eXX_S01', 'eYY_S07', 'eXY_S02', 'eZZ_S08')


def test_holdout_mask_does_not_mix_channels():
    train, hold = station_masks(CHANNELS)
    assert train.tolist() == [True, False, True, False]
    assert hold.tolist() == [False, True, False, True]


def test_missing_holdout_station_rejected():
    with pytest.raises(ValueError):
        station_masks(CHANNELS, ('S07', 'S09'))


def test_synthetic_noise_repeatable_and_shape():
    prediction = SimpleNamespace(times_h=np.array([1., 2.]), channels=CHANNELS,
                                 strain_ns=np.full((2, 4), 2.0))
    a = make_synthetic_observations(prediction, seed=42)
    b = make_synthetic_observations(prediction, seed=42)
    np.testing.assert_array_equal(a.observed_ns, b.observed_ns)
    assert a.observed_ns.shape == (2, 4)
    assert not np.array_equal(a.observed_ns, prediction.strain_ns)


def test_duplicate_times_rejected():
    with pytest.raises(ValueError):
        StrainObservations(np.array([1., 1.]), CHANNELS, np.zeros((2, 4)),
                           2.0, np.array([1, 0, 1, 0], bool),
                           np.array([0, 1, 0, 1], bool))
