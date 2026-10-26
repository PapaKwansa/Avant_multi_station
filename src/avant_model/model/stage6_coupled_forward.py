"""Stage 6B-3: fixed-global FV pressure -> moving tiles -> unchanged AVANT strain.

This is a one-way effective-storage forward adapter, not a 3-D coupled Biot
solver, a calibrated geologic source model, or an inverse sampler. Physical
horizontal coordinates are AVANT's global/survey primed coordinates, and
receiver z is already positive-downward analytical depth.

The pressure field is independent of the chosen mechanical support. Source
geometry controls only the tiling, pressure-to-tile mapping, and analytical
mechanical kernel. Pressure gauges sample the uncompressed global FV field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

from avant_model.inversion.stage6_parameters import Stage6Contract, Stage6State
from avant_model.model.stage6_coordinates import SourceRectangle
from avant_model.model.stage6_global_fv import GlobalConfinedFVSolver
from avant_model.model.stage6_tile_pressure import RotatedTilePressureOperator
from avant_model.model.pressure_gauges import PressureGaugeOperator
from avant_model.model.pressure_superposition import (
    Cuboid, ElasticProperties, build_pressure_kernel,
)

DEFAULT_COMPONENTS = ('eXX', 'eYY', 'eXY', 'eZZ')


@dataclass(frozen=True)
class Stage6ForwardResult:
    times_h: np.ndarray
    channels: tuple[str, ...]
    strain_ns: np.ndarray            # (times, channels)
    tile_pressure_pa: np.ndarray     # (times, nx*ny)
    gauge_pressure_pa: np.ndarray | None  # (times, gauges) or None
    fv_pressure_change_pa: np.ndarray  # (times, y, x)
    max_mass_balance_residual_m3_s: float
    tile_splits: tuple[int, int]


@dataclass(frozen=True)
class SupportComparison:
    """Different *mechanical loading supports* for identical FV pressures.

    Reference is the baseline finite-support prediction, not independent
    COMSOL truth. Differences are support-sensitivity diagnostics, not an
    estimate of the true mechanical modeling error.
    """
    factors: tuple[float, ...]
    splits: tuple[tuple[int, int], ...]
    strain_ns: tuple[np.ndarray, ...]
    times_h: np.ndarray
    relative_l2_difference_from_base: tuple[float, ...]
    relative_l2_difference_by_time: np.ndarray


class Stage6CoupledForward:
    """Reuse fixed-global FV and original analytical cuboid without changing either.

    Receiver depths MUST have already been converted into the analytical
    positive-downward convention. `stations` is a mapping with station/x/y/z.
    Gamma scales alpha_ref/E_ref; it does not separately infer E or alpha.
    """

    def __init__(self, contract: Stage6Contract, hydraulic_config: Mapping,
                 stations: Mapping, *, splits=(6, 6), quadrature_order=4,
                 components=DEFAULT_COMPONENTS, gauges_prime_m=None):
        if not isinstance(contract, Stage6Contract):
            raise TypeError('contract must be Stage6Contract')
        self.contract = contract
        self.hydraulic = GlobalConfinedFVSolver(hydraulic_config, contract.domain)
        self.splits = tuple(splits)
        if len(self.splits) != 2 or any(isinstance(v, (bool, np.bool_)) or
           not isinstance(v, (int, np.integer)) or v <= 0 for v in self.splits):
            raise ValueError('splits must have two positive integers')
        if isinstance(quadrature_order, (bool, np.bool_)):
            raise ValueError('quadrature order must be an integer')
        self.quadrature_order = int(quadrature_order)
        if self.quadrature_order != quadrature_order or not 1 <= self.quadrature_order <= 32:
            raise ValueError('quadrature order must be an integer from 1 to 32')
        self.components = tuple(components)
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError('components must be unique and nonempty')
        self.stations = {}
        for key in ('station', 'x', 'y', 'z'):
            if key not in stations:
                raise ValueError('missing station coordinate field: '+key)
            self.stations[key] = (tuple(str(v) for v in stations[key]) if key == 'station'
                                  else np.asarray(stations[key], dtype=float).copy())
        n = len(self.stations['station'])
        if n < 1 or len(set(self.stations['station'])) != n or any(not v for v in self.stations['station']):
            raise ValueError('station IDs must be unique and nonempty')
        if any(np.asarray(self.stations[key]).shape != (n,) or
               not np.isfinite(self.stations[key]).all()
               for key in ('x', 'y', 'z')):
            raise ValueError('station coordinates must be finite vectors of equal length')
        if np.any(self.stations['z'] < 0):
            raise ValueError('receiver depths must be positive downward')
        # This 2-D hydraulic approximation assumes a uniform aquifer thickness
        # equal to the mechanical source thickness in the first benchmark.
        thickness = 2 * contract.mechanics.source_c_m
        if not np.isclose(self.hydraulic.b, thickness, rtol=0, atol=1e-10):
            raise ValueError('Stage 6B benchmark requires FV thickness = 2*source_c_m')
        self.gauges_prime_m = None
        self.gauge_op = None
        if gauges_prime_m is not None:
            xy = np.asarray(gauges_prime_m, dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 1 or not np.isfinite(xy).all():
                raise ValueError('gauges_prime_m must be a finite nonempty (N,2) array')
            self.gauges_prime_m = xy.copy()
            self.gauge_op = PressureGaugeOperator(
                self.hydraulic.x_m, self.hydraulic.y_m, xy[:, 0], xy[:, 1])
        # Mechanics only depend on geometry and nu, never K, Ss, or gamma.
        self._mechanics_cache = {}

    def _mechanics(self, source: SourceRectangle, nu, splits):
        key = (source, float(nu), tuple(splits))
        if key not in self._mechanics_cache:
            tile_op = RotatedTilePressureOperator(
                self.hydraulic.x_m, self.hydraulic.y_m,
                source, splits[0], splits[1], self.quadrature_order)
            ref = self.contract.mechanics
            parent = Cuboid(source.a_m, source.b_m, ref.source_c_m, ref.source_h_m,
                            source.x0_prime_m, source.y0_prime_m, source.theta_deg)
            elastic = ElasticProperties(ref.E_ref_Pa, float(nu), ref.alpha_ref)
            kernel = build_pressure_kernel(parent, elastic,
                x_prime=self.stations['x'], y_prime=self.stations['y'],
                z=self.stations['z'], station_names=self.stations['station'],
                splits=(splits[0], splits[1], 1), components=self.components)
            expected = splits[0] * splits[1]
            if kernel.G.shape[1] != expected or tile_op.n_tiles != expected:
                raise RuntimeError('Tile-pressure/operator dimensions disagree')
            # The existing tile_cuboid ordering is ix outermost, iy innermost.
            # Verify the metadata rather than merely assume its ordering.
            for j, tile in enumerate(kernel.tiles):
                if (tile.ix, tile.iy, tile.iz) != (j // splits[1], j % splits[1], 0):
                    raise RuntimeError('Analytical kernel tile ordering mismatch')
            self._mechanics_cache[key] = (tile_op, kernel)
        return self._mechanics_cache[key]

    def _strain_from_pressure(self, p_fields, source, nu, gamma_rel, splits):
        tile_op, kernel = self._mechanics(source, nu, splits)
        pressure = tile_op.mean_pressure_pa(p_fields)
        strain = kernel.predict(pressure) * float(gamma_rel)
        if strain.shape != (p_fields.shape[0], len(kernel.channels)):
            raise RuntimeError('Invalid coupled forward output dimensions')
        return np.asarray(strain), np.asarray(pressure), tuple(kernel.channels)

    def predict(self, state: Stage6State, times_h):
        state = self.contract.validate_state(state)
        fv = self.hydraulic.solve(state.K_m_s, state.Ss_per_m, times_h)
        strain, tile_pressure, channels = self._strain_from_pressure(
            fv.pressure_change_pa, state.source, state.nu,
            state.gamma_rel, self.splits)
        gauges = (self.gauge_op.sample(fv.pressure_change_pa)
                  if self.gauge_op is not None else None)
        return Stage6ForwardResult(
            fv.times_h.copy(), channels, strain, tile_pressure, gauges,
            fv.pressure_change_pa.copy(), fv.max_mass_balance_residual_m3_s,
            self.splits)

    def compare_support(self, state: Stage6State, times_h,
                        factors=(1.0, 1.5, 2.0)) -> SupportComparison:
        """Expand mechanical pressure support about the SAME source center.

        Keep cell size approximately fixed by requiring factor*splits to be
        integer. All compared cases use the SAME global FV pressure solution.
        Larger support means more pressure-loaded cuboids, NOT a re-solved
        hydraulic problem. This does not independently verify the mechanics.
        """
        state = self.contract.validate_state(state)
        factors = tuple(float(v) for v in factors)
        if len(factors) < 2 or abs(factors[0]-1.0) > 1e-12 or any(
                not np.isfinite(v) or v < 1.0 for v in factors):
            raise ValueError('factors must start at 1.0 and be >= 1')
        if any(b <= a for a, b in zip(factors, factors[1:])):
            raise ValueError('support factors must be strictly increasing')
        fv = self.hydraulic.solve(state.K_m_s, state.Ss_per_m, times_h)
        curves, grids = [], []
        for factor in factors:
            sizes = tuple(int(round(v * factor)) for v in self.splits)
            if any(abs(k * factor - n) > 1e-8 for k,n in zip(self.splits, sizes)):
                raise ValueError('support factors must preserve initial tile sizes')
            if sizes[0] * sizes[1] > 1024:
                raise ValueError('support comparison exceeds 1024 mechanical tiles')
            source = SourceRectangle(
                state.a_m*factor, state.b_m*factor, state.theta_deg,
                state.x0_prime_m, state.y0_prime_m)
            strain, _, channels = self._strain_from_pressure(
                fv.pressure_change_pa, source, state.nu, state.gamma_rel, sizes)
            curves.append(strain)
            grids.append(sizes)
        base = curves[0]
        norm = np.linalg.norm(base)
        ratios = tuple(float(np.linalg.norm(s-base) / norm) if norm > 0 else float('nan')
                       for s in curves)
        by_time = np.full((len(fv.times_h),len(factors)), np.nan)
        for k, (a, b) in enumerate(zip(base, zip(*curves))):
            n = np.linalg.norm(a)
            if n > 0:
                by_time[k] = [np.linalg.norm(x-a)/n for x in b]
        return SupportComparison(factors, tuple(grids), tuple(curves),
                                 fv.times_h.copy(), ratios, by_time)
