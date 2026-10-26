"""M1: conservative confined FV flow with a geologically defined rotated K contrast.

Uses exactly the Stage 6 fixed global grid, well weights, storage and time stepping.
The geological zone is NOT a mechanical strain support. Elasticity remains uniform.

A rotated rectangular compartment is intersected with each FV cell in *area*;
partially covered cells use an arithmetic-area effective conductivity. Face
transmissibility is the harmonic mean of adjacent cell values. This is a
coarse-grid approximation of an oblique sharp material contact, NOT an exact
cut-cell flux solution; require grid-resolution sensitivity before inference.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from avant_model.model.confined_fv import FVResult
from avant_model.model.stage6_coordinates import SourceRectangle, global_to_local
from avant_model.model.stage6_global_fv import GlobalConfinedFVSolver


def _clip_polygon(poly, axis, bound, keep_greater):
    """Clip convex polygon to one coordinate half-space (Sutherland-Hodgman)."""
    if not poly:
        return []
    def inside(pt):
        return (pt[axis] >= bound) if keep_greater else (pt[axis] <= bound)
    out = []
    previous = poly[-1]
    previous_ok = inside(previous)
    for current in poly:
        current_ok = inside(current)
        if current_ok != previous_ok:
            denom = current[axis] - previous[axis]
            fraction = (bound - previous[axis]) / denom
            out.append((previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1])))
        if current_ok:
            out.append(current)
        previous, previous_ok = current, current_ok
    return out


def _clipped_area(poly, a, b):
    for axis, bound, keep_greater in ((0, -a, True), (0, a, False),
                                      (1, -b, True), (1, b, False)):
        poly = _clip_polygon(poly, axis, bound, keep_greater)
        if len(poly) < 3:
            return 0.0
    arr = np.asarray(poly, dtype=float)
    return abs(float(np.dot(arr[:, 0], np.roll(arr[:, 1], -1)) -
                     np.dot(arr[:, 1], np.roll(arr[:, 0], -1)))) / 2.0


def zone_cell_fractions(x_centers, y_centers, dx, dy, zone: SourceRectangle):
    """Exact geometric area fractions of rectangle-zone overlap per FV cell.

    The clipped-area operation is exact for straight cell/zone boundaries up to
    floating precision. The *hydraulic* mixed-cell approximation is still
    lower-order and must be checked through observable convergence.
    """
    x = np.asarray(x_centers, dtype=float)
    y = np.asarray(y_centers, dtype=float)
    if not isinstance(zone, SourceRectangle):
        raise TypeError('zone must be a validated SourceRectangle')
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError('FV axes must be nonempty regular 1D grids')
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or dx <= 0 or dy <= 0:
        raise ValueError('Invalid FV grid')
    a, b = zone.a_m, zone.b_m
    xx, yy = np.meshgrid(x, y)
    corners = []
    for ox, oy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
        xl, yl = global_to_local(xx + ox * dx, yy + oy * dy, zone)
        corners.append(np.stack([xl, yl], axis=-1))
    poly = np.stack(corners, axis=-2)  # (ny, nx, 4, 2)
    minx, maxx = poly[..., 0].min(axis=-1), poly[..., 0].max(axis=-1)
    miny, maxy = poly[..., 1].min(axis=-1), poly[..., 1].max(axis=-1)
    inside = (minx >= -a) & (maxx <= a) & (miny >= -b) & (maxy <= b)
    outside = (maxx <= -a) | (minx >= a) | (maxy <= -b) | (miny >= b)
    result = inside.astype(float)
    for iy, ix in np.argwhere(~(inside | outside)):
        points = [tuple(z) for z in poly[iy, ix]]
        result[iy, ix] = min(1.0, max(0.0, _clipped_area(points, a, b) / (dx * dy)))
    return result


def harmonic(a, b):
    return 2.0 * a * b / (a + b)


@dataclass(frozen=True)
class TwoZoneFVResult:
    flow: FVResult
    zone_fraction: np.ndarray
    effective_K_m_s: np.ndarray
    max_mass_balance_residual_m3_s: float


class TwoZoneGlobalFVSolver(GlobalConfinedFVSolver):
    """Conservative two-zone K solver; same one-way Ss and fixed global well."""
    def solve_two_zone(self, K_inside_m_s, K_outside_m_s, Ss_per_m,
                       zone: SourceRectangle, output_times_h=None):
        K1, K0, Ss = map(float, (K_inside_m_s, K_outside_m_s, Ss_per_m))
        if not all(np.isfinite(v) and v > 0 for v in (K1, K0, Ss)):
            raise ValueError('K_inside, K_outside and Ss must be finite and positive')
        if not isinstance(zone, SourceRectangle):
            raise TypeError('zone must be a SourceRectangle')
        self.domain.validate_source(zone)
        fractions = zone_cell_fractions(self.x_m, self.y_m, self.dx, self.dy, zone)
        K = K0 + (K1 - K0) * fractions
        if np.any(K <= 0):
            raise RuntimeError('Nonpositive effective cell conductivity')
        # Off-diagonal FV fluxes: one common conductance per shared face.
        ids = np.arange(self.n).reshape(self.ny, self.nx)
        ix = ids[:, :-1].ravel(); jx = ids[:, 1:].ravel()
        iy = ids[:-1, :].ravel(); jy = ids[1:, :].ravel()
        gx = (self.b * self.dy / self.dx) * harmonic(K[:, :-1], K[:, 1:]).ravel()
        gy = (self.b * self.dx / self.dy) * harmonic(K[:-1, :], K[1:, :]).ravel()
        ii = np.concatenate([ix, jx, iy, jy])
        jj = np.concatenate([jx, ix, jy, iy])
        vv = np.concatenate([-gx, -gx, -gy, -gy])
        diagonal = np.zeros(self.n, dtype=float)
        for p, q, g in ((ix, jx, gx), (iy, jy, gy)):
            np.add.at(diagonal, p, g)
            np.add.at(diagonal, q, g)
        # Exactly the legacy half-cell zero-head-change outer BC.
        boundary = np.zeros((self.ny, self.nx), dtype=float)
        boundary[:, 0] += 2 * K[:, 0] * self.b * self.dy / self.dx
        boundary[:, -1] += 2 * K[:, -1] * self.b * self.dy / self.dx
        boundary[0, :] += 2 * K[0, :] * self.b * self.dx / self.dy
        boundary[-1, :] += 2 * K[-1, :] * self.b * self.dx / self.dy
        diagonal += boundary.ravel()
        rows = np.concatenate([ii, np.arange(self.n)])
        cols = np.concatenate([jj, np.arange(self.n)])
        data = np.concatenate([vv, diagonal])
        A = coo_matrix((data, (rows, cols)), shape=(self.n, self.n)).tocsc()
        if output_times_h is None:
            times = np.arange(round(self.tend / self.dt_h) + 1) * self.dt_h
        else:
            times = np.asarray(output_times_h, dtype=float)
            if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
                raise ValueError('Output times must be finite and nonempty')
            steps = np.rint(times / self.dt_h).astype(int)
            if (np.any(times < 0) or np.any(times > self.tend + 1e-9) or
                not np.allclose(times / self.dt_h, steps, atol=1e-8, rtol=0) or
                len(np.unique(steps)) != len(steps)):
                raise ValueError('Output times must be distinct valid solver steps')
        step_ids = np.rint(times / self.dt_h).astype(int)
        coef = Ss * self.volume / (self.dt_h * 3600.0)
        lu = splu(A + diags(np.full(self.n, coef), format='csc'))
        head = np.zeros(self.n, dtype=float)
        saved = {0: head.copy()} if 0 in step_ids else {}
        source = self.q * self.source
        max_residual = 0.0
        for step in range(1, round(self.tend / self.dt_h) + 1):
            injecting = step * self.dt_h <= self.tinj + 1e-10
            old = head
            head = lu.solve(coef * old + (source if injecting else 0.0))
            residual = (self.q if injecting else 0.0) - coef * float(np.sum(head - old)) - float(np.sum(A @ head))
            max_residual = max(max_residual, abs(residual))
            if step in step_ids:
                saved[step] = head.copy()
        stack = np.stack([saved[int(t)].reshape(self.ny, self.nx) for t in step_ids])
        result = FVResult(times.copy(), self.x_m.copy(), self.y_m.copy(), stack,
                          stack * self.rho * self.g, max_residual, self.q * self.tinj * 3600.0)
        return TwoZoneFVResult(result, fractions, K, max_residual)
