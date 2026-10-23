import numpy as np
import pytest
from avant_model.inversion.coupled_parameters import HydraulicBounds


def test_log_parameter_transform():
    bounds = HydraulicBounds()
    assert bounds.contains([-6.0, -6.0])
    assert bounds.to_physical([-6.0, -6.0]) == pytest.approx((1e-6, 1e-6))


@pytest.mark.parametrize('bad', [[-8, -6], [-6, -3], [np.nan, -6], [-6], [-6, -6, -6]])
def test_invalid_parameter_rejected(bad):
    bounds = HydraulicBounds()
    assert not bounds.contains(bad)
    with pytest.raises(ValueError):
        bounds.to_physical(bad)


def test_bad_bounds_rejected():
    with pytest.raises(ValueError):
        HydraulicBounds(log10_K=(-4, -7))
