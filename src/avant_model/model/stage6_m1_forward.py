"""M1 pilot: geological K-contrast zone + independent, FIXED mechanical loading.

The rectangle affects ONLY hydraulic conductivity. It is NOT the limit of
poroelastic loading. The fixed mechanical cuboids sample pressure inside and
outside the zone via the validated legacy cuboid kernel (unchanged). Elastic
moduli and alpha are uniform. The pressure solution is one-way with effective Ss.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from avant_model.inversion.stage6_parameters import FixedHydraulicDomain, ReferenceMechanics
from avant_model.model.stage6_coordinates import SourceRectangle
from avant_model.model.stage6_m1_two_zone_fv import TwoZoneGlobalFVSolver
from avant_model.model.stage6_tile_pressure import RotatedTilePressureOperator
from avant_model.model.pressure_gauges import PressureGaugeOperator
from avant_model.model.pressure_superposition import Cuboid, ElasticProperties, build_pressure_kernel


@dataclass(frozen=True)
class M1State:
    K_inside_m_s: float
    K_outside_m_s: float
    Ss_per_m: float
    gamma_rel: float
    zone: SourceRectangle

    def validate(self):
        vals = (self.K_inside_m_s, self.K_outside_m_s, self.Ss_per_m, self.gamma_rel)
        if not all(np.isfinite(v) and v > 0 for v in vals):
            raise ValueError('All hydraulic properties and mechanical amplitude must be positive and finite')
        if not isinstance(self.zone, SourceRectangle):
            raise TypeError('zone must be a SourceRectangle')
        return self


@dataclass(frozen=True)
class M1Result:
    times_h: np.ndarray
    strain_ns: np.ndarray
    channels: tuple[str, ...]
    pressure_gauges_pa: np.ndarray | None
    fv_pressure_pa: np.ndarray
    zone_fraction: np.ndarray
    conductivity_field_m_s: np.ndarray
    max_mass_balance_residual_m3_s: float


class Stage6M1Forward:
    def __init__(self, hydraulic_config: Mapping, domain: FixedHydraulicDomain,
                 mechanics: ReferenceMechanics, stations: Mapping,
                 mechanical_support: SourceRectangle, *, splits=(6, 8),
                 quadrature_order=4, components=('eXX','eYY','eXY','eZZ'),
                 gauges_prime_m=None):
        self.hydraulic = TwoZoneGlobalFVSolver(hydraulic_config, domain)
        self.mechanics = mechanics
        if not isinstance(mechanics, ReferenceMechanics):
            raise TypeError('mechanics must be a Stage6 ReferenceMechanics')
        if not isinstance(mechanical_support, SourceRectangle):
            raise TypeError('mechanical_support must be a SourceRectangle')
        self.support = mechanical_support
        self.splits = tuple(splits)
        self.quadrature_order = quadrature_order
        stations = dict(stations)
        for k in ('station','x','y','z'):
            if k not in stations:
                raise ValueError('Missing station field '+k)
        names = tuple(str(k) for k in stations['station'])
        if not names or len(names) != len(set(names)):
            raise ValueError('Stations must have unique names')
        axes = {k:np.asarray(stations[k],dtype=float) for k in ('x','y','z')}
        if any(v.shape != (len(names),) or not np.isfinite(v).all() for v in axes.values()):
            raise ValueError('Bad station coordinates')
        if np.any(axes['z'] < 0):
            raise ValueError('Station z is already positive downward depth')
        if not np.isclose(self.hydraulic.b,2*mechanics.source_c_m,rtol=0,atol=1e-10):
            raise ValueError('2D hydraulic layer thickness must equal mechanical pressurized layer thickness')
        self.tile_op = RotatedTilePressureOperator(self.hydraulic.x_m, self.hydraulic.y_m,
                                                   mechanical_support, self.splits[0],
                                                   self.splits[1], self.quadrature_order)
        parent = Cuboid(mechanical_support.a_m, mechanical_support.b_m,
                        mechanics.source_c_m, mechanics.source_h_m,
                        mechanical_support.x0_prime_m, mechanical_support.y0_prime_m,
                        mechanical_support.theta_deg)
        elast = ElasticProperties(mechanics.E_ref_Pa, mechanics.nu_ref, mechanics.alpha_ref)
        self.kernel = build_pressure_kernel(parent, elast, x_prime=axes['x'], y_prime=axes['y'],
                                            z=axes['z'], station_names=names,
                                            splits=(*self.splits,1),components=components)
        if self.kernel.G.shape[1] != self.tile_op.n_tiles:
            raise RuntimeError('Mechanical/FV tile ordering or count mismatch')
        for j, tile in enumerate(self.kernel.tiles):
            if (tile.ix, tile.iy, tile.iz) != (j//self.splits[1],j%self.splits[1],0):
                raise RuntimeError('Incorrect mechanical tile order')
        self.channels = tuple(self.kernel.channels)
        self.gauge_op = None
        if gauges_prime_m is not None:
            g = np.asarray(gauges_prime_m,dtype=float)
            if g.ndim != 2 or g.shape[1] != 2 or len(g) < 1 or not np.isfinite(g).all():
                raise ValueError('gauges_prime_m must have shape (n,2)')
            self.gauge_op = PressureGaugeOperator(self.hydraulic.x_m,self.hydraulic.y_m,g[:,0],g[:,1])

    def predict(self, state: M1State, times_h):
        state.validate()
        fv = self.hydraulic.solve_two_zone(state.K_inside_m_s,state.K_outside_m_s,
                                           state.Ss_per_m,state.zone,times_h)
        p = fv.flow.pressure_change_pa
        tile_pressure = self.tile_op.mean_pressure_pa(p)
        strain = self.kernel.predict(tile_pressure) * state.gamma_rel
        gauges = self.gauge_op.sample(p) if self.gauge_op is not None else None
        return M1Result(fv.flow.times_h, strain, self.channels, gauges, p,
                        fv.zone_fraction, fv.effective_K_m_s,
                        fv.max_mass_balance_residual_m3_s)
