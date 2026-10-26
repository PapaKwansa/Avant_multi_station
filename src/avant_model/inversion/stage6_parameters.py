"""Stage 6A parameter/physics contract for AVANT hydraulic-mechanical-spatial inversion.

Contract only: does NOT solve flow, calculate strain, perform MCMC, or claim
that the generalized Stage 6B forward model is implemented. The primary
parameterization fits the *identifiable* mechanical amplitude alpha/E (as a
ratio to a fixed reference) rather than fitting E and alpha independently.
The hydraulic Ss is explicitly an *effective one-way* storage coefficient.

Sampled parameters in canonical order (default eight):
  log10_K_m_s, log10_Ss_per_m, log10_gamma_rel,
  a_m, b_m, theta_deg, x0_prime_m, y0_prime_m.
Optionally infer nu after a separate identifiability analysis; otherwise
it is fixed. Neither E nor alpha can be freely inferred from this model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Mapping

import numpy as np

from avant_model.model.stage6_coordinates import (
    SourceRectangle, source_corners_global, contains_global_point,
)

PRIMARY_PARAMETER_NAMES = (
    'log10_K_m_s', 'log10_Ss_per_m', 'log10_gamma_rel',
    'a_m', 'b_m', 'theta_deg', 'x0_prime_m', 'y0_prime_m',
)
ALLOWED_PARAMETER_NAMES = PRIMARY_PARAMETER_NAMES + ('nu',)
DISALLOWED_SEPARATE_PARAMETERS = frozenset({'E', 'alpha', 'log10_E', 'log10_alpha',
                                          'E_Pa', 'alpha_biot', 'log10_E_Pa'})


@dataclass(frozen=True)
class FixedHydraulicDomain:
    xmin_prime_m: float
    xmax_prime_m: float
    ymin_prime_m: float
    ymax_prime_m: float
    well_x_prime_m: float
    well_y_prime_m: float
    source_margin_m: float = 0.0
    require_well_in_source: bool = True

    def __post_init__(self):
        values = (self.xmin_prime_m, self.xmax_prime_m, self.ymin_prime_m,
                  self.ymax_prime_m, self.well_x_prime_m, self.well_y_prime_m,
                  self.source_margin_m)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Domain and well coordinates must be finite')
        if not (self.xmin_prime_m < self.xmax_prime_m and
                self.ymin_prime_m < self.ymax_prime_m):
            raise ValueError('Invalid global hydraulic domain extents')
        if not (self.xmin_prime_m < self.well_x_prime_m < self.xmax_prime_m and
                self.ymin_prime_m < self.well_y_prime_m < self.ymax_prime_m):
            raise ValueError('Well must be strictly inside the hydraulic domain')
        if self.source_margin_m < 0:
            raise ValueError('source_margin_m cannot be negative')
        if not isinstance(self.require_well_in_source, bool):
            raise ValueError('require_well_in_source must be boolean')

    def validate_source(self, source: SourceRectangle) -> None:
        corners = source_corners_global(source)
        margin = self.source_margin_m
        if (np.min(corners[:, 0]) < self.xmin_prime_m + margin or
            np.max(corners[:, 0]) > self.xmax_prime_m - margin or
            np.min(corners[:, 1]) < self.ymin_prime_m + margin or
            np.max(corners[:, 1]) > self.ymax_prime_m - margin):
            raise ValueError('Rotated mechanical source extends beyond fixed hydraulic support/margin')
        if self.require_well_in_source and not contains_global_point(
            source, self.well_x_prime_m, self.well_y_prime_m):
            raise ValueError('Configured well must lie in the mechanical source')

    def extent(self):
        return (self.xmin_prime_m, self.xmax_prime_m,
                self.ymin_prime_m, self.ymax_prime_m)


@dataclass(frozen=True)
class ReferenceMechanics:
    E_ref_Pa: float
    alpha_ref: float
    nu_ref: float
    source_c_m: float
    source_h_m: float

    def __post_init__(self):
        values = (self.E_ref_Pa, self.alpha_ref, self.nu_ref,
                  self.source_c_m, self.source_h_m)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Reference mechanics must be finite')
        if not self.E_ref_Pa > 0 or not 0 < self.alpha_ref <= 1:
            raise ValueError('Reference modulus and Biot coefficient invalid')
        if not 0 <= self.nu_ref < 0.5:
            raise ValueError('Reference Poisson ratio must be in [0,0.5)')
        if not 0 < self.source_c_m < self.source_h_m:
            raise ValueError('Reference source must lie strictly below free surface')


@dataclass(frozen=True)
class Stage6State:
    log10_K_m_s: float
    log10_Ss_per_m: float
    log10_gamma_rel: float
    a_m: float
    b_m: float
    theta_deg: float
    x0_prime_m: float
    y0_prime_m: float
    nu: float

    @property
    def source(self) -> SourceRectangle:
        return SourceRectangle(self.a_m, self.b_m, self.theta_deg,
                               self.x0_prime_m, self.y0_prime_m)

    @property
    def K_m_s(self):
        return float(10.0 ** self.log10_K_m_s)

    @property
    def Ss_per_m(self):
        return float(10.0 ** self.log10_Ss_per_m)

    @property
    def gamma_rel(self):
        return float(10.0 ** self.log10_gamma_rel)

    def effective_alpha_over_E_per_Pa(self, reference: ReferenceMechanics):
        return self.gamma_rel * reference.alpha_ref / reference.E_ref_Pa

    def E_if_alpha_fixed_Pa(self, reference: ReferenceMechanics):
        """Conditional interpretation if alpha=alpha_ref is externally fixed."""
        return reference.E_ref_Pa / self.gamma_rel

    def as_dict(self):
        return {name: float(getattr(self, name)) for name in ALLOWED_PARAMETER_NAMES}


@dataclass(frozen=True)
class Stage6Contract:
    free_names: tuple
    fixed: Mapping[str, float]
    bounds: Mapping[str, tuple]
    reference_state: Mapping[str, float]
    domain: FixedHydraulicDomain
    mechanics: ReferenceMechanics
    hydraulic_storage_mode: str
    geometry_semantics: str

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError('Contract JSON must be an object')
        if data.get('schema_version') != 'stage6a-1':
            raise ValueError('Expected schema_version stage6a-1')
        p = data['parameters']
        if not isinstance(p, dict):
            raise ValueError('parameters must be an object')
        free = tuple(p['free'])
        if not free or len(set(free)) != len(free):
            raise ValueError('Free parameter names must be unique and nonempty')
        if any(not isinstance(name, str) for name in free):
            raise ValueError('Parameter names must be strings')
        if any(name in DISALLOWED_SEPARATE_PARAMETERS for name in free):
            raise ValueError('E and alpha cannot both be independently estimated by this reduced-order model')
        invalid = set(free) - set(ALLOWED_PARAMETER_NAMES)
        if invalid:
            raise ValueError('Unsupported free parameter names: ' + str(sorted(invalid)))
        fixed = dict(p['fixed'])
        if set(fixed).intersection(free):
            raise ValueError('Parameters cannot be simultaneously fixed and free')
        if set(fixed) - set(ALLOWED_PARAMETER_NAMES):
            raise ValueError('Unsupported fixed parameter name')
        if set(free) | set(fixed) != set(ALLOWED_PARAMETER_NAMES):
            raise ValueError('Every allowed parameter must be either fixed or free')
        if tuple(name for name in ALLOWED_PARAMETER_NAMES if name in free) != free:
            raise ValueError('Free parameters must follow canonical parameter ordering')
        bounds = {k: tuple(v) for k, v in p['bounds'].items()}
        if set(bounds) != set(free):
            raise ValueError('Specify bounds for exactly the free parameters')
        for name, pair in bounds.items():
            if len(pair) != 2 or not all(math.isfinite(float(v)) for v in pair) or not pair[0] < pair[1]:
                raise ValueError('Invalid bounds for ' + name)
        ref = dict(p['reference_state'])
        if set(ref) != set(ALLOWED_PARAMETER_NAMES):
            raise ValueError('reference_state must provide every parameter')
        if any(not math.isfinite(float(v)) for v in ref.values()):
            raise ValueError('reference_state must be finite')
        for name, val in fixed.items():
            if not math.isfinite(float(val)) or float(val) != float(ref[name]):
                raise ValueError('Fixed value differs from reference_state: ' + name)
        mechanics = ReferenceMechanics(**data['reference_mechanics'])
        if float(ref['nu']) != float(mechanics.nu_ref):
            raise ValueError('reference_state.nu and reference_mechanics.nu_ref disagree')
        domain = FixedHydraulicDomain(**data['global_hydraulic_domain'])
        storage = data['hydraulic_storage_mode']
        if storage != 'effective_one_way':
            raise ValueError('Stage 6A supports only explicitly labeled effective_one_way storage')
        semantics = data['geometry_semantics']
        if semantics != 'finite_pressurized_mechanical_support':
            raise ValueError('Source geometry must explicitly describe finite pressurized mechanical support')
        instance = cls(free, fixed, bounds, ref, domain, mechanics, storage, semantics)
        # Reference point must be feasible and inside the sampling bounds.
        for name in free:
            lo, hi = bounds[name]
            x = ref[name]
            if not lo <= x <= hi or (name == 'theta_deg' and x == 180):
                raise ValueError('Reference parameter outside bounds: ' + name)
        instance.reference()
        return instance

    @classmethod
    def from_json(cls, path):
        with Path(path).open('r', encoding='utf-8') as handle:
            return cls.from_dict(json.load(handle))

    def validate_state(self, state: Stage6State):
        for name in ALLOWED_PARAMETER_NAMES:
            val = float(getattr(state, name))
            if not math.isfinite(val):
                raise ValueError('Nonfinite parameter: ' + name)
            if name in self.bounds:
                lo, hi = self.bounds[name]
                if not lo <= val <= hi:
                    raise ValueError('Parameter outside declared bounds: ' + name)
            elif val != float(self.fixed[name]):
                raise ValueError('A fixed parameter has changed: ' + name)
        if not 0 <= state.nu < 0.5:
            raise ValueError('nu outside physical range [0,0.5)')
        if not (state.K_m_s > 0 and math.isfinite(state.K_m_s) and
                state.Ss_per_m > 0 and math.isfinite(state.Ss_per_m) and
                state.gamma_rel > 0 and math.isfinite(state.gamma_rel)):
            raise ValueError('Nonphysical derived hydraulic or mechanical amplitude')
        self.domain.validate_source(state.source)
        return state

    def state_from_vector(self, values):
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(self.free_names),) or not np.isfinite(arr).all():
            raise ValueError(f'Expected finite vector of length {len(self.free_names)}')
        data = {**self.fixed, **dict(zip(self.free_names, map(float, arr)))}
        return self.validate_state(Stage6State(**data))

    def reference(self):
        return self.validate_state(Stage6State(**self.reference_state))

    def vector_from_state(self, state: Stage6State):
        self.validate_state(state)
        return np.array([getattr(state, name) for name in self.free_names], dtype=float)

    def manifest(self):
        state = self.reference()
        return {
            'schema_version': 'stage6a-1',
            'scope': 'Parameter/coordinates contract only; no new generalized forward inversion',
            'free_parameter_order': list(self.free_names),
            'reference_parameters': state.as_dict(),
            'reference_derived': {
                'K_m_s': state.K_m_s,
                'Ss_per_m': state.Ss_per_m,
                'gamma_rel': state.gamma_rel,
                'effective_alpha_over_E_per_Pa': state.effective_alpha_over_E_per_Pa(self.mechanics),
                'E_Pa_only_if_alpha_fixed_at_reference': state.E_if_alpha_fixed_Pa(self.mechanics),
            },
            'hydraulic_domain_global_prime_m': list(self.domain.extent()),
            'well_global_prime_m': [self.domain.well_x_prime_m, self.domain.well_y_prime_m],
            'source_corners_global_prime_m': source_corners_global(state.source).tolist(),
            'storage_mode': self.hydraulic_storage_mode,
            'geometry_semantics': self.geometry_semantics,
            'caveats': [
                'One-way effective storage; this does not enforce a two-way Biot constitutive storage relation.',
                'E and alpha are structurally non-identifiable separately in current forward mechanics.',
                'Changing nu requires a new analytical kernel; change of gamma_rel alone scales its amplitude.',
                'Stage 6B must implement fixed-global FV well and boundary conditions and validated pressure-to-tile integration.',
                'A finite pressurized mechanical support may not capture strain from pressure outside that support.',
                'These illustrative bounds are not field priors and do not establish parameter identifiability.',
            ],
        }
