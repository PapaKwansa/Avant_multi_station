import numpy as np
import pytest
from avant_model.model.pressure_gauges import PressureGaugeOperator


def test_affine_field_exact_at_interior_and_corners():
    x = np.array([-3., -1., 2., 5.])
    y = np.array([-4., 0., 7.])
    gx = np.array([0.4, -3., 5., 2.])
    gy = np.array([-1.2, -4., 7., 0.])
    op = PressureGaugeOperator(x, y, gx, gy)
    xx, yy = np.meshgrid(x, y)
    field = 123. + 8.*xx - 11.*yy
    np.testing.assert_allclose(op.sample(field), 123+8*gx-11*gy, atol=1e-12)


def test_stacked_time_array_shape_and_weights():
    x = np.arange(4., dtype=float)
    y = np.arange(3., dtype=float)
    op = PressureGaugeOperator(x, y, np.array([.5, 2.5]), np.array([.5, 1.5]))
    field = np.stack([np.ones((3,4))*v for v in (0., 4., 8.)])
    got = op.sample(field)
    assert got.shape == (3,2)
    np.testing.assert_allclose(got, np.array([[0,0],[4,4],[8,8]]))
    np.testing.assert_allclose(op.weights.sum(axis=1), 1.)


@pytest.mark.parametrize('gx,gy', [([-0.1], [0.5]), ([3.1], [.5]), ([.5], [-.01]), ([.5], [2.1])])
def test_gauges_outside_cell_centers_rejected(gx, gy):
    with pytest.raises(ValueError, match='within'):
        PressureGaugeOperator(np.arange(4.), np.arange(3.), gx, gy)


def test_invalid_axes_and_shape_rejected():
    with pytest.raises(ValueError):
        PressureGaugeOperator([1,0,2], [0,1], [.5], [.5])
    op = PressureGaugeOperator([0,1], [0,1], [.5], [.5])
    with pytest.raises(ValueError, match='Expected FV pressure'):
        op.sample(np.zeros((2,2,2,2)))


def test_no_silent_nearest_neighbor_extrapolation():
    op = PressureGaugeOperator([0.,2.], [0.,2.], [1.], [1.])
    value = op.sample(np.array([[0.,2.],[2.,4.]]))
    np.testing.assert_allclose(value, [2.])
