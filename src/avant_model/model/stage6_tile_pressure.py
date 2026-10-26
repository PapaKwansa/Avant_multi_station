"""Stage 6B-2: quadrature integration of global FV pressure over moving cuboid tiles.

This adds a pressure mapping operator only. It does not evaluate mechanics, modify
legacy AVANT cuboid routines, solve flow, or infer parameters. All coordinates
are global survey (primed) except the source's own local xi/eta coordinates.

One-layer tiles are ordered ix outermost, iy innermost, matching
pressure_superposition.tile_cuboid(parent, (nx,ny,1)). The pressure in each tile
is a positive-area-weighted average of the *bilinearly interpolated FV field*.
For fixed geometry the mapping is linear in pressure and can be cached.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from avant_model.model.pressure_gauges import PressureGaugeOperator
from avant_model.model.stage6_coordinates import (
    SourceRectangle, local_to_global, source_tile_centers_global,
    source_corners_global,
)


def _positive_int(value, label, max_value=32):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{label} must be an integer')
    if not 1 <= value <= max_value:
        raise ValueError(f'{label} must lie in [1,{max_value}]')
    return int(value)


def _fv_axis(axis, name):
    arr = np.asarray(axis, dtype=float)
    if arr.ndim != 1 or arr.size < 2 or not np.isfinite(arr).all():
        raise ValueError(f'{name} must be a finite 1D FV center axis with at least 2 elements')
    step = np.diff(arr)
    if not (step > 0).all() or not np.allclose(step, step[0], rtol=1e-10, atol=1e-10):
        raise ValueError(f'{name} must have strictly increasing, uniform FV cell centers')
    return arr


@dataclass(frozen=True)
class PressureSupportSummary:
    """Pressure-magnitude **proxy**, not a strain or material mass-fraction result.

    Both numerator and denominator have units Pa*m^2; thickness cancels for
    a spatially uniform aquifer thickness. Numerator integrates |p_bilinear|
    over the cuboid footprint via tile quadrature. Denominator is the FV
    midpoint integral of |p_cell_center| over the entire 2D global FV domain.
    Different integration rules produce numerical quadrature differences;
    ratios are not guaranteed to lie exactly in [0,1]. Do not interpret as a
    fraction of induced strain, fluid mass, pore volume, or stored energy.
    """
    inside_abs_pressure_pa_m2: np.ndarray
    global_abs_pressure_pa_m2: np.ndarray
    ratio_inside_over_global: np.ndarray
    note: str = ('Pressure-magnitude integral proxy only; does not quantify '
                 'strain truncation or Biot coupling. NaN when total pressure is zero.')


class RotatedTilePressureOperator:
    """Geometry-specific, reusable global FV pressure-to-tile averaging operator.

    All sample points and entire source footprint must lie within the FV
    cell-center interpolation rectangle. No outside-domain extrapolation.
    Bilinear spatial interpolation is performed BEFORE area quadrature;
    results are not exact cellwise FV-volume integration.
    """
    def __init__(self, x_prime_m, y_prime_m, source: SourceRectangle,
                 nx: int, ny: int, quadrature_order: int = 4):
        if not isinstance(source, SourceRectangle):
            raise TypeError('source must be a validated Stage 6A SourceRectangle')
        self.source = source
        self.x_prime_m = _fv_axis(x_prime_m, 'x_prime_m').copy()
        self.y_prime_m = _fv_axis(y_prime_m, 'y_prime_m').copy()
        self.nx = _positive_int(nx, 'nx', max_value=1000)
        self.ny = _positive_int(ny, 'ny', max_value=1000)
        self.order = _positive_int(quadrature_order, 'quadrature_order')
        self.n_tiles = self.nx * self.ny
        self.tile_area_m2 = (2*source.a_m/self.nx)*(2*source.b_m/self.ny)
        self.tile_centers_prime_m = source_tile_centers_global(source, self.nx, self.ny)
        self.parent_corners_prime_m = source_corners_global(source)
        xg, yg = self.parent_corners_prime_m.T
        if (np.any(xg < self.x_prime_m[0]) or np.any(xg > self.x_prime_m[-1])
                or np.any(yg < self.y_prime_m[0]) or np.any(yg > self.y_prime_m[-1])):
            raise ValueError('Entire rotated source must be inside FV center interpolation rectangle; no extrapolation')
        abscissas, weights = np.polynomial.legendre.leggauss(self.order)
        normalized_weights = np.outer(weights, weights).ravel() / 4.0
        if not np.isclose(normalized_weights.sum(), 1., rtol=0, atol=1e-14):
            raise RuntimeError('Quadrature weights do not sum to one')
        # Iterate in the exact tile order of pressure_superposition.tile_cuboid.
        xi_centers = -source.a_m + (np.arange(self.nx)+0.5)*(2*source.a_m/self.nx)
        eta_centers = -source.b_m + (np.arange(self.ny)+0.5)*(2*source.b_m/self.ny)
        xi = np.repeat(xi_centers, self.ny)
        eta = np.tile(eta_centers, self.nx)
        local_x = xi[:, None, None] + source.a_m/self.nx*abscissas[None, :, None]
        local_y = eta[:, None, None] + source.b_m/self.ny*abscissas[None, None, :]
        local_x, local_y = np.broadcast_arrays(local_x, local_y)
        xs, ys = local_to_global(local_x.ravel(), local_y.ravel(), source)
        self._sample_op = PressureGaugeOperator(self.x_prime_m,self.y_prime_m,xs,ys)
        self._quad_weights = np.array(normalized_weights, copy=True)
        # Immutable metadata by default; numerical operators remain cached.
        for attr in ('x_prime_m', 'y_prime_m', 'tile_centers_prime_m',
                     'parent_corners_prime_m', '_quad_weights'):
            getattr(self, attr).setflags(write=False)

    def _quadrature_values(self, pressure_change_pa):
        vals = np.asarray(pressure_change_pa, dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError('Pressure field contains nonfinite values')
        sampled = self._sample_op.sample(vals)
        return sampled.reshape((-1, self.n_tiles, self.order*self.order)) if vals.ndim == 3 else sampled.reshape(self.n_tiles,self.order*self.order)

    def mean_pressure_pa(self, pressure_change_pa):
        """Return shape (n_tiles,) or (n_times,n_tiles) of tile pressure means."""
        return np.einsum('...tk,k->...t',self._quadrature_values(pressure_change_pa),self._quad_weights)

    def integrated_pressure_pa_m2(self, pressure_change_pa):
        """Return the signed spatial integral (Pa*m^2) over the entire source."""
        means = self.mean_pressure_pa(pressure_change_pa)
        return self.tile_area_m2*np.sum(means,axis=-1)

    def pressure_support_summary(self, pressure_change_pa):
        """Approximate coverage of ABSOLUTE pressure integral by source footprint.

        This is a *pressure support* proxy, never an estimate of strain error.
        For a zero field the coverage ratio is NaN by design.
        """
        p=np.asarray(pressure_change_pa,dtype=float)
        values=self._quadrature_values(p)
        inside = self.tile_area_m2*np.sum(
            np.einsum('...tk,k->...t',np.abs(values),self._quad_weights), axis=-1)
        dx=float(self.x_prime_m[1]-self.x_prime_m[0])
        dy=float(self.y_prime_m[1]-self.y_prime_m[0])
        total = dx*dy*np.sum(np.abs(p),axis=(-2,-1))
        ratio=np.full_like(np.asarray(total,dtype=float),np.nan)
        np.divide(inside,total,out=ratio,where=total>0)
        return PressureSupportSummary(inside,total,ratio)
