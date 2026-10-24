"""Pressure-change gauge observation operator for the cell-centered AVANT FV grid.

Gauge x/y positions are measured in the FV **local** hydraulic x/y system,
which Stage 3 assumes is aligned to the analytical cuboid's xi/eta axes.
No rotation, nearest-neighbor extrapolation, or boundary interpretation is
performed here.  Samples are bilinear interpolations of FV cell-center values.
"""
from dataclasses import dataclass
import numpy as np


def _validate_axis(x, name):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
        raise ValueError(f'{name} must be an increasing finite 1D cell-center axis (>=2)')
    return x


@dataclass(frozen=True)
class PressureGaugeOperator:
    """Precomputed bilinear weights for gauges strictly inside FV cell centers.

    Results have shape ``(n_times, n_gauges)`` for input ``(n_times, ny, nx)``.
    A 2D ``(ny,nx)`` input returns a 1D vector. This operator does not apply
    the mechanical regional pressure projection; gauges see full FV fields.
    """
    x_m: np.ndarray
    y_m: np.ndarray
    gauge_x_m: np.ndarray
    gauge_y_m: np.ndarray

    def __post_init__(self):
        x = _validate_axis(self.x_m, 'x_m')
        y = _validate_axis(self.y_m, 'y_m')
        gx = np.asarray(self.gauge_x_m, float)
        gy = np.asarray(self.gauge_y_m, float)
        if gx.ndim != 1 or gx.size < 1 or gx.shape != gy.shape:
            raise ValueError('gauge_x_m and gauge_y_m must be equal-length nonempty 1D vectors')
        if not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
            raise ValueError('Gauge positions must be finite')
        # Do not silently extrapolate into boundary half-cells or beyond domain.
        if np.any(gx < x[0]) or np.any(gx > x[-1]) or np.any(gy < y[0]) or np.any(gy > y[-1]):
            raise ValueError('Gauge must lie within the FV cell-center interpolation rectangle')
        ix = np.clip(np.searchsorted(x, gx, side='right') - 1, 0, x.size - 2)
        iy = np.clip(np.searchsorted(y, gy, side='right') - 1, 0, y.size - 2)
        tx = (gx - x[ix]) / (x[ix + 1] - x[ix])
        ty = (gy - y[iy]) / (y[iy + 1] - y[iy])
        w = np.stack(((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty), axis=1)
        idx = np.stack((iy*x.size+ix, iy*x.size+ix+1,
                        (iy+1)*x.size+ix, (iy+1)*x.size+ix+1), axis=1)
        for name, value in [('x_m', x), ('y_m', y), ('gauge_x_m', gx), ('gauge_y_m', gy),
                            ('weights', w), ('indices', idx)]:
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def sample(self, pressure_change_pa):
        p = np.asarray(pressure_change_pa, dtype=float)
        expected = (len(self.y_m), len(self.x_m))
        if p.ndim == 2 and p.shape == expected:
            return np.sum(p.ravel()[self.indices] * self.weights, axis=1)
        if p.ndim == 3 and p.shape[1:] == expected:
            return np.sum(p.reshape(p.shape[0], -1)[:, self.indices] * self.weights[None, :, :], axis=2)
        raise ValueError(f'Expected FV pressure shape {expected} or (times,{expected[0]},{expected[1]}), got {p.shape}')
