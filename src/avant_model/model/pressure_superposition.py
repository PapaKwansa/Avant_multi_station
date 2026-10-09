"""Tiled pressure-change loading using the existing AVANT analytical routines.

This add-on does NOT change the analytical solution or infer permeability.
Coordinates: global/primed horizontal coordinates; depth z positive downward
from the original free surface. Geometry semi-dimensions and positions: metres.
Pressure changes: Pa. Kernel: nanostrain/Pa. Tensor shear, not engineering shear.
Initial supported domain: receivers strictly outside the parent source, including
surface receivers. Interior/interface strain evaluation is deliberately excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Sequence, Tuple

import numpy as np

from . import multi_station_coord_transform as coord
from . import multi_station_rotation as rotation
from . import multi_station_strain as mechanics

DEFAULT_COMPONENTS = ("eXX", "eYY", "eXY", "eZZ")
SUPPORTED_COMPONENTS = DEFAULT_COMPONENTS + ("eXZ", "eYZ")


class NumericalKernelError(RuntimeError):
    """An existing analytical routine returned a non-finite or invalid tensor."""


@dataclass(frozen=True)
class Cuboid:
    a: float
    b: float
    c: float
    h: float
    x0_prime: float
    y0_prime: float
    theta_deg: float

    def __post_init__(self):
        values = (self.a, self.b, self.c, self.h,
                  self.x0_prime, self.y0_prime, self.theta_deg)
        if not np.isfinite(values).all():
            raise ValueError("All cuboid values must be finite.")
        if min(self.a, self.b, self.c) <= 0 or self.h <= self.c:
            raise ValueError("Require a,b,c > 0 and h > c (source below surface).")
        # b > a is an optional constraint on the PARENT inversion, not a
        # mathematical requirement on every tile. Do not impose it here.


@dataclass(frozen=True)
class ElasticProperties:
    E: float
    nu: float
    alpha: float

    def __post_init__(self):
        if not np.isfinite((self.E, self.nu, self.alpha)).all():
            raise ValueError("Elastic inputs must be finite.")
        if self.E <= 0 or not (-1 < self.nu < 0.5):
            raise ValueError("Require E > 0 and -1 < nu < 0.5.")
        if not 0 <= self.alpha <= 1:
            raise ValueError("This saturated-material interface requires 0 <= alpha <= 1.")


@dataclass(frozen=True)
class Tile:
    ix: int
    iy: int
    iz: int
    xi: float
    eta: float
    zeta: float
    a: float
    b: float
    c: float
    h: float


@dataclass(frozen=True)
class PressureKernel:
    G: np.ndarray
    channels: Tuple[str, ...]
    tiles: Tuple[Tile, ...]
    parent: Cuboid
    elastic: ElasticProperties
    splits: Tuple[int, int, int]
    station_names: Tuple[str, ...]

    def __post_init__(self):
        array = np.array(self.G, dtype=float, copy=True)
        if array.shape != (len(self.channels), len(self.tiles)):
            raise ValueError("Kernel shape does not match channel/tile metadata.")
        if not np.isfinite(array).all():
            raise ValueError("Kernel contains non-finite values.")
        array.setflags(write=False)
        object.__setattr__(self, "G", array)

    def predict(self, pressure_pa):
        """(N,) -> (M,), or (T,N) -> (T,M); pressures are changes, not increments."""
        p = np.asarray(pressure_pa, dtype=float)
        if p.ndim not in (1, 2) or p.shape[-1] != self.G.shape[1]:
            raise ValueError("pressure_pa must have shape (n_tiles,) or (n_times,n_tiles).")
        if not np.isfinite(p).all():
            raise ValueError("Pressure changes must be finite.")
        with np.errstate(over="raise", invalid="raise"):
            return p @ self.G.T

    def dataframe(self, time_s, pressure_pa):
        """Explicit named columns matching the existing component-major convention."""
        import pandas as pd
        t = np.asarray(time_s, dtype=float)
        p = np.asarray(pressure_pa, dtype=float)
        if (t.ndim != 1 or t.size == 0 or not np.isfinite(t).all()
                or (t.size > 1 and np.any(np.diff(t) <= 0))):
            raise ValueError("time_s must be finite, nonempty, and strictly increasing.")
        if p.ndim != 2 or p.shape[0] != t.size:
            raise ValueError("For a time series provide pressure_pa with shape (T,N).")
        frame = pd.DataFrame(self.predict(p), columns=self.channels)
        frame.insert(0, "time_s", t)
        return frame


def checked_splits(splits: Sequence[int]) -> Tuple[int, int, int]:
    s = tuple(splits)
    if len(s) != 3 or any(isinstance(v, (bool, np.bool_)) or
                          not isinstance(v, Integral) or v < 1 for v in s):
        raise ValueError("splits must contain three positive integers.")
    return tuple(int(v) for v in s)


def tile_cuboid(parent: Cuboid, splits=(2, 2, 1)) -> Tuple[Tile, ...]:
    """Exact regular partition. Ordering: x outermost, z innermost (C order)."""
    nx, ny, nz = checked_splits(splits)
    a, b, c = parent.a / nx, parent.b / ny, parent.c / nz
    tiles = []
    for ix, iy, iz in product(range(nx), range(ny), range(nz)):
        xi = -parent.a + (2 * ix + 1) * a
        eta = -parent.b + (2 * iy + 1) * b
        zeta = -parent.c + (2 * iz + 1) * c
        tiles.append(Tile(ix, iy, iz, xi, eta, zeta, a, b, c, parent.h + zeta))
    return tuple(tiles)


def coarse_block_basis(fine_splits, coarse_splits):
    """Map coarse-zone pressures to a finer mechanical tiling, p_fine = B @ a.

    Coarse dimensions must divide fine dimensions. This is an explicit block
    basis, NOT smoothing, and imposes identical pressures within each block.
    """
    fine = checked_splits(fine_splits)
    coarse = checked_splits(coarse_splits)
    if any(f % c for f, c in zip(fine, coarse)):
        raise ValueError("Each coarse split must divide the corresponding fine split.")
    factors = tuple(f // c for f, c in zip(fine, coarse))
    B = np.zeros((int(np.prod(fine)), int(np.prod(coarse))))
    for row, index in enumerate(product(*(range(n) for n in fine))):
        coarse_index = tuple(i // f for i, f in zip(index, factors))
        B[row, np.ravel_multi_index(coarse_index, coarse)] = 1.0
    return B


def build_pressure_kernel(parent: Cuboid, elastic: ElasticProperties, *,
                          x_prime, y_prime, z, station_names=None,
                          splits=(2, 2, 1), components=DEFAULT_COMPONENTS,
                          exterior_tolerance_m=1e-8) -> PressureKernel:
    """Evaluate 1-Pa finite-cuboid responses; never multiply by volume again.

    Original mechanics/coordinate/rotation routines are reused, not approximated.
    Existing numerical warnings are not suppressed. Non-finite final tensors
    fail with tile/station context; no clipping, atan2 replacement, or zero-fill.
    Geometry/material changes require rebuilding this kernel.
    """
    s = checked_splits(splits)
    x, y, z_obs = [np.asarray(v, dtype=float) for v in (x_prime, y_prime, z)]
    if any(v.ndim != 1 or v.size == 0 for v in (x, y, z_obs)):
        raise ValueError("Receiver coordinates must be nonempty one-dimensional arrays.")
    if not (x.shape == y.shape == z_obs.shape):
        raise ValueError("Receiver coordinate shapes must agree.")
    if not all(np.isfinite(v).all() for v in (x, y, z_obs)) or np.any(z_obs < 0):
        raise ValueError("Coordinates must be finite and z must be depth >= 0.")
    n = x.size
    names = (tuple(f"S{i+1:02d}" for i in range(n)) if station_names is None
             else tuple(str(v).strip() for v in station_names))
    if len(names) != n or len(set(names)) != n or any(not v for v in names):
        raise ValueError("Provide one unique nonempty station name per receiver.")
    components = tuple(components)
    if (not components or len(set(components)) != len(components)
            or any(v not in SUPPORTED_COMPONENTS for v in components)):
        raise ValueError(f"Choose unique components from {SUPPORTED_COMPONENTS}.")
    tol = float(exterior_tolerance_m)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("exterior_tolerance_m must be finite and nonnegative.")
    x_local, y_local = coord.primed_to_unprimed(
        x, y, parent.x0_prime, parent.y0_prime, parent.theta_deg)
    in_source = ((np.abs(x_local) <= parent.a + tol)
                 & (np.abs(y_local) <= parent.b + tol)
                 & (np.abs(z_obs - parent.h) <= parent.c + tol))
    if np.any(in_source):
        bad = [names[i] for i in np.flatnonzero(in_source)]
        raise ValueError(f"Interior/on-source receivers not supported in this starter: {bad}")
    tiles = tile_cuboid(parent, s)
    channels = tuple(f"{c}_{name}" for c in components for name in names)
    G = np.empty((len(channels), len(tiles)))
    ec = mechanics.charac_strain(
        mechanics.linear_trans(elastic.alpha, elastic.nu, 1.0, elastic.E), elastic.nu)
    for j, tile in enumerate(tiles):
        for i in range(n):
            try:
                eps = np.asarray(mechanics.strain(
                    x=x_local[i] - tile.xi, y=y_local[i] - tile.eta,
                    z=z_obs[i],  # Keep the original free-surface reference.
                    a=tile.a, b=tile.b, c=tile.c, ec=ec, h=tile.h, nu=elastic.nu),
                    dtype=float)
            except (FloatingPointError, ZeroDivisionError) as exc:
                raise NumericalKernelError(f"Tile {j}, station {names[i]}: {exc}") from exc
            if eps.shape != (3, 3) or not np.isfinite(eps).all():
                raise NumericalKernelError(
                    f"Non-finite/invalid analytical tensor: tile {j}, station {names[i]}. "
                    "Check corner alignments; do not silently alter the verified kernel.")
            values = rotation.rotate_strain_tensor(
                eps[0, 0], eps[1, 1], eps[0, 1], eps[0, 2], eps[1, 2],
                eps[2, 2], parent.theta_deg)
            by_name = dict(zip(("eXX", "eYY", "eXY", "eXZ", "eYZ", "eZZ"), values))
            for k, component in enumerate(components):
                G[k * n + i, j] = by_name[component] * 1e9
    return PressureKernel(G, channels, tiles, parent, elastic, s, names)
