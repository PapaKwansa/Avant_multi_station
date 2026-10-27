"""Add-on OED for the nine-parameter M1 pressure/strain forward model.

No physics or legacy OED routines are changed. Observation rows are explicit:
all strain (time-major, then forward-kernel channel order), followed by all
pressure (time-major, then gauge order). This deliberately does NOT use the
legacy component-major station-subset slicer on time-expanded M1 observations.

Information scores are local and conditional on an independent-error model and
chosen reference point. Unit prior precision in normalized parameter space is
only a declared numerical design regularizer, NOT an inferred field prior or a
calibrated parameter posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from avant_model.inversion.m1_identifiability_gate import central_difference
from avant_model.inversion.m1_map_recovery import PARAMETER_NAMES, check_bounds
from avant_model.oed.fisher_information import fisher_information_matrix


@dataclass(frozen=True)
class ObservationRow:
    index: int
    kind: str                   # strain or pressure
    time_index: int
    time_h: float
    station: str | None
    component: str | None
    gauge: str | None


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    v = tuple(str(x) for x in values)
    if not v or any(not x for x in v) or len(set(v)) != len(v):
        raise ValueError(f'{label} must contain unique, nonempty labels')
    return v


class M1ObservationCatalog:
    """Stable row indexing for arbitrary M1 station, gauge and time subsets."""

    def __init__(self, times_h, channels: Sequence[str], gauges: Sequence[str],
                 *, strain_sigma_ns: float, pressure_sigma_pa: float):
        t = np.asarray(times_h, float)
        if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('times_h must be finite and strictly increasing')
        self.times_h = t.copy()
        self.channels = _names(channels, 'channels')
        self.gauges = _names(gauges, 'gauges')
        for name, value in (('strain_sigma_ns', strain_sigma_ns),
                            ('pressure_sigma_pa', pressure_sigma_pa)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.strain_sigma_ns = float(strain_sigma_ns)
        self.pressure_sigma_pa = float(pressure_sigma_pa)
        channel_parts = []
        for c in self.channels:
            if '_' not in c:
                raise ValueError('Strain channels must use component_station labels')
            component, station = c.rsplit('_', 1)
            if not component or not station:
                raise ValueError('Invalid strain channel: ' + c)
            channel_parts.append((component, station))
        self.components = tuple(dict.fromkeys(x[0] for x in channel_parts))
        self.stations = tuple(dict.fromkeys(x[1] for x in channel_parts))
        rows = []
        for it, time in enumerate(t):
            for comp, station in channel_parts:
                rows.append(ObservationRow(len(rows), 'strain', it, float(time),
                                           station, comp, None))
        for it, time in enumerate(t):
            for gauge in self.gauges:
                rows.append(ObservationRow(len(rows), 'pressure', it, float(time),
                                           None, None, gauge))
        self.rows = tuple(rows)
        self.sigma = np.array([self.strain_sigma_ns if r.kind == 'strain'
                               else self.pressure_sigma_pa for r in rows], dtype=float)

    def flatten(self, strain_ns, pressure_pa) -> np.ndarray:
        s = np.asarray(strain_ns, float)
        p = np.asarray(pressure_pa, float)
        if s.shape != (len(self.times_h), len(self.channels)):
            raise ValueError('Unexpected M1 strain array shape')
        if p.shape != (len(self.times_h), len(self.gauges)):
            raise ValueError('Unexpected M1 pressure array shape')
        y = np.concatenate((s.ravel(), p.ravel()))
        if not np.isfinite(y).all():
            raise ValueError('Non-finite M1 response')
        return y

    def _subset(self, spec: Mapping, key: str, allowed: tuple[str, ...]) -> set[str]:
        value = spec.get(key, 'all')
        if value == 'all':
            return set(allowed)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'{key} must be "all" or a list')
        result = set(value)
        if result - set(allowed):
            raise ValueError(f'Unknown {key} labels: {sorted(result-set(allowed))}')
        return result

    def indices(self, spec: Mapping) -> np.ndarray:
        stations = self._subset(spec, 'stations', self.stations)
        components = self._subset(spec, 'components', self.components)
        gauges = self._subset(spec, 'gauges', self.gauges)
        times = spec.get('time_indices', 'all')
        if times == 'all':
            time_set = set(range(len(self.times_h)))
        else:
            if not isinstance(times, (list, tuple)) or any(type(i) is not int for i in times):
                raise ValueError('time_indices must be "all" or an integer list')
            time_set = set(times)
            if time_set - set(range(len(self.times_h))):
                raise ValueError('time_indices out of range')
        ids = [r.index for r in self.rows if r.time_index in time_set and (
            (r.kind == 'strain' and r.station in stations and r.component in components) or
            (r.kind == 'pressure' and r.gauge in gauges))]
        if not ids:
            raise ValueError('Observation design selects no measurements')
        return np.asarray(ids, dtype=int)


def make_m1_jacobian(predict, reference_theta, catalog: M1ObservationCatalog,
                     steps: Sequence[float]) -> np.ndarray:
    """Compute one full-network M1 Jacobian, and reuse its rows for all designs.

    `predict(theta)` must return `(strain_ns, pressure_pa)` on the catalog network.
    Step sizes are absolute in nine canonical log/physical parameter coordinates.
    """
    theta = np.asarray(reference_theta, float)
    h = np.asarray(steps, float)
    if theta.shape != (9,) or h.shape != (9,) or not np.isfinite(theta).all() or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError('Expected finite nine-parameter reference and positive steps')
    bounds = check_bounds()
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        raise ValueError('Reference outside verified M1 pilot bounds')
    if np.any(theta-h < bounds[:, 0]) or np.any(theta+h > bounds[:, 1]):
        raise ValueError('Finite-difference steps cross M1 bounds; choose a feasible reference or smaller steps')

    def forward(x):
        strain, pressure = predict(x)
        return catalog.flatten(strain, pressure)

    J = central_difference(forward, theta, h)
    if J.shape != (len(catalog.rows), 9) or not np.isfinite(J).all():
        raise ValueError('Invalid full-network nine-parameter Jacobian')
    return J


def score_design(J_full: np.ndarray, catalog: M1ObservationCatalog,
                 spec: Mapping, *, parameter_scales: Sequence[float],
                 prior_precision: float = 1.0, rank_rtol: float = 1e-3) -> dict:
    """Evaluate design using local, independent-error Fisher information.

    Information gain = log det(P+F) - log det(P) where P=lambda*I in the
    *scaled parameter coordinates*.  This is an illustrative regularizer,
    not an empirical Bayesian prior. Report raw rank/conditioning separately.
    """
    J = np.asarray(J_full, float)
    s = np.asarray(parameter_scales, float)
    if J.shape != (len(catalog.rows), 9) or not np.isfinite(J).all():
        raise ValueError('J_full shape or finiteness invalid')
    if s.shape != (9,) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('Nine strictly positive parameter scales required')
    if not np.isfinite(prior_precision) or prior_precision <= 0:
        raise ValueError('prior_precision must be strictly positive')
    if not 0 < rank_rtol < 1:
        raise ValueError('rank_rtol must lie in (0,1)')
    idx = catalog.indices(spec)
    Jsub = J[idx]
    weights = 1.0 / catalog.sigma[idx]**2
    F = fisher_information_matrix(Jsub, observation_weights=weights, parameter_scales=s)
    if F.shape != (9, 9) or not np.isfinite(F).all():
        raise RuntimeError('Bad Fisher information from existing OED utility')
    WJ = (Jsub * s[None, :]) / catalog.sigma[idx, None]
    singular = np.linalg.svd(WJ, compute_uv=False)
    rank = int(np.sum(singular > rank_rtol * singular[0])) if len(singular) and singular[0] > 0 else 0
    raw_condition = (float(singular[0] / singular[-1]) if rank == 9 and len(singular) == 9 and singular[-1] > 0
                     else float('inf'))
    P = prior_precision * np.eye(9)
    sign, logdet = np.linalg.slogdet(P + F)
    if sign <= 0:
        raise RuntimeError('Non-positive design information determinant')
    info_gain = float(logdet - 9 * np.log(prior_precision))
    # The diagonal inverse is a regularized *information proxy*, not a posterior CI.
    regularized_inverse = np.linalg.solve(P + F, np.eye(9))
    contributions = {'strain': int(sum(catalog.rows[int(i)].kind == 'strain' for i in idx)),
                     'pressure': int(sum(catalog.rows[int(i)].kind == 'pressure' for i in idx))}
    return {
        'name': str(spec.get('name', 'unnamed')),
        'n_observations': int(len(idx)),
        'n_strain': contributions['strain'], 'n_pressure': contributions['pressure'],
        'rank': rank, 'rank_tolerance_relative': rank_rtol,
        'condition_number_unregularized': raw_condition if np.isfinite(raw_condition) else None,
        'singular_values': singular.tolist(),
        'information_gain_logdet': info_gain,
        'a_score_prior_regularized': float(np.trace(regularized_inverse)),
        'e_score_prior_regularized': float(np.min(np.linalg.eigvalsh(P+F))),
        'parameter_uncertainty_proxy_scaled': np.sqrt(np.maximum(np.diag(regularized_inverse), 0)).tolist(),
        'selected_row_indices': idx.tolist(),
    }


def compare_designs(J_full, catalog, designs: Sequence[Mapping], *, parameter_scales,
                    prior_precision=1.0, rank_rtol=1e-3) -> list[dict]:
    result = []
    seen = set()
    for design in designs:
        name = str(design.get('name', ''))
        if not name or name in seen:
            raise ValueError('All designs must have distinct nonempty names')
        seen.add(name)
        result.append(score_design(J_full, catalog, design, parameter_scales=parameter_scales,
                                   prior_precision=prior_precision, rank_rtol=rank_rtol))
    return result


def greedy_add_sensors(J_full, catalog: M1ObservationCatalog, *, base_design: Mapping,
                       candidate_stations: Sequence[str] = (), candidate_gauges: Sequence[str] = (),
                       max_additions: int = 2, parameter_scales: Sequence[float],
                       prior_precision: float = 1.0) -> list[dict]:
    """One-at-a-time greedy augmentation from an existing measurement design.

    A station adds its selected components at all selected times; a gauge adds
    one pressure observation at each selected time. Marginal information gain
    *per new observation* is used when candidate groups have different sizes.
    This is not global combinatorial placement optimality.
    """
    if type(max_additions) is not int or max_additions < 0:
        raise ValueError('max_additions must be a nonnegative integer')
    state = dict(base_design)
    st = catalog._subset(state, 'stations', catalog.stations)
    ga = catalog._subset(state, 'gauges', catalog.gauges)
    options = [('station', x) for x in candidate_stations if x not in st]
    options += [('gauge', x) for x in candidate_gauges if x not in ga]
    if len(options) != len(set(options)):
        raise ValueError('Duplicate candidate sensors')
    if any(k == 'station' and x not in catalog.stations or k == 'gauge' and x not in catalog.gauges for k, x in options):
        raise ValueError('Candidate sensor not included in forward-network catalog')
    current = score_design(J_full, catalog, state, parameter_scales=parameter_scales,
                           prior_precision=prior_precision)
    history = []
    for iteration in range(max_additions):
        candidates = []
        for kind, name in options:
            modified = dict(state)
            if kind == 'station':
                modified['stations'] = sorted(st | {name})
            else:
                modified['gauges'] = sorted(ga | {name})
            new_score = score_design(J_full, catalog, modified, parameter_scales=parameter_scales,
                                     prior_precision=prior_precision)
            count = new_score['n_observations'] - current['n_observations']
            if count <= 0:
                continue
            gain = new_score['information_gain_logdet'] - current['information_gain_logdet']
            candidates.append((gain/count, gain, kind, name, new_score))
        if not candidates:
            break
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
        per_row, gain, kind, name, new_score = candidates[0]
        if gain <= 1e-12:
            break
        if kind == 'station':
            st.add(name)
            state['stations'] = sorted(st)
        else:
            ga.add(name)
            state['gauges'] = sorted(ga)
        options.remove((kind, name))
        history.append({'step': iteration+1, 'added_kind': kind, 'added_name': name,
                        'information_gain_increase': gain, 'increase_per_new_observation': per_row,
                        'new_information_gain': new_score['information_gain_logdet'],
                        'new_effective_rank': new_score['rank'],
                        'n_observations': new_score['n_observations']})
        current = new_score
    return history
