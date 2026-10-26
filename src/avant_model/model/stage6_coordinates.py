"""Stage 6 global/source coordinate contract (no change to legacy AVANT kernels).

The 'primed' frame is the physical global horizontal survey frame of the
AVANT benchmark. Analytical z is positive *downward*; this module never
converts COMSOL elevations. Local cuboid axes (xi, eta) are obtained by the
existing AVANT rotation convention:
  xi  = (x' - x0') cos(theta) - (y' - y0') sin(theta)
  eta = (x' - x0') sin(theta) + (y' - y0') cos(theta)

The inverse, used when mapping local mechanical tiles into a *fixed*
global FV grid, is explicitly tested against this convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SourceRectangle:
    a_m: float
    b_m: float
    theta_deg: float
    x0_prime_m: float
    y0_prime_m: float

    def __post_init__(self):
        vals = (self.a_m, self.b_m, self.theta_deg,
                self.x0_prime_m, self.y0_prime_m)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError('Source rectangle values must be finite')
        if not 0 < self.a_m < self.b_m:
            raise ValueError('Canonical source rectangle requires 0 < a < b')
        if not 0 <= self.theta_deg < 180:
            raise ValueError('Canonical theta_deg must be in [0,180)')


def global_to_local(x_prime_m, y_prime_m, source: SourceRectangle):
    """Physical global/survey x',y' -> rotated source-local xi,eta."""
    x, y = np.broadcast_arrays(np.asarray(x_prime_m, dtype=float),
                               np.asarray(y_prime_m, dtype=float))
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Nonfinite coordinates')
    dx, dy = x - source.x0_prime_m, y - source.y0_prime_m
    c = math.cos(math.radians(source.theta_deg))
    s = math.sin(math.radians(source.theta_deg))
    return c * dx - s * dy, s * dx + c * dy


def local_to_global(xi_m, eta_m, source: SourceRectangle):
    """Inverse map of global_to_local: source-local -> physical survey frame."""
    xi, eta = np.broadcast_arrays(np.asarray(xi_m, dtype=float),
                                  np.asarray(eta_m, dtype=float))
    if not (np.isfinite(xi).all() and np.isfinite(eta).all()):
        raise ValueError('Nonfinite local coordinates')
    c = math.cos(math.radians(source.theta_deg))
    s = math.sin(math.radians(source.theta_deg))
    return source.x0_prime_m + c*xi + s*eta, source.y0_prime_m - s*xi + c*eta


def source_corners_global(source: SourceRectangle):
    """Four physical corners, counterclockwise in local (xi,eta) order."""
    xi = np.array([-source.a_m, source.a_m, source.a_m, -source.a_m])
    eta = np.array([-source.b_m, -source.b_m, source.b_m, source.b_m])
    x, y = local_to_global(xi, eta, source)
    return np.column_stack([x, y])


def source_tile_centers_global(source: SourceRectangle, nx: int, ny: int):
    """(nx*ny,2) tile centers with x-index outermost, y-index innermost.

    This matches the tile ordering in ``pressure_superposition.tile_cuboid``
    for a one-layer mechanical partition (nx,ny,1).
    """
    for v in (nx, ny):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError('nx and ny must be positive integers')
    xis = -source.a_m + (np.arange(nx)+.5)*(2*source.a_m/nx)
    etas = -source.b_m + (np.arange(ny)+.5)*(2*source.b_m/ny)
    xloc, yloc = np.meshgrid(xis, etas, indexing='ij')
    xg, yg = local_to_global(xloc.ravel(), yloc.ravel(), source)
    return np.column_stack([xg, yg])


def contains_global_point(source: SourceRectangle, x_prime_m: float, y_prime_m: float,
                          tolerance_m: float = 1e-9) -> bool:
    xi, eta = global_to_local(x_prime_m, y_prime_m, source)
    if np.ndim(xi) != 0 or np.ndim(eta) != 0:
        raise ValueError('Point containment requires scalar coordinates')
    t = float(tolerance_m)
    if not np.isfinite(t) or t < 0:
        raise ValueError('tolerance_m must be finite and nonnegative')
    return bool(abs(float(xi)) <= source.a_m + t and abs(float(eta)) <= source.b_m + t)
