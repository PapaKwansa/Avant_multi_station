"""Bounded M1 same-model synthetic MAP recovery; original mechanics untouched.

A MAP point here means the maximum of a *uniform bounded prior* times an
independent Gaussian likelihood; inside the bounds it equals constrained MLE.
No observed COMSOL data, real-world calibration, posterior interval or field
boundary recovery is implied by this synthetic benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import least_squares

from avant_model.inversion.m1_identifiability_gate import PARAMETERS

# Illustrative geological/physical search ranges around the M1 pilot; NOT field priors.
# K_inside>K_outside is ensured throughout these ranges. Canonical b>a holds.
SEARCH_BOUNDS = {
    'log10_K_inside': (-6.18, -5.78),
    'log10_K_outside': (-6.93, -6.30),
    'log10_Ss': (-6.35, -5.65),
    'log10_gamma_rel': (-0.24, 0.24),
    'a_m': (180.0, 255.0),
    'b_m': (325.0, 485.0),
    'theta_deg': (0.0, 38.0),
    'x0_prime_m': (-20.0, 130.0),
    'y0_prime_m': (60.0, 220.0),
}

# Benchmark deliberately assumes these values are independently known.
# For synthetic tests they equal the generating truth (ORACLE); cannot claim
# that a real geological survey gives such precise information.
CONSTRAINED_FIXED = ('log10_K_outside', 'b_m', 'y0_prime_m')
PARAMETER_NAMES = tuple(PARAMETERS)


def check_bounds(bounds=SEARCH_BOUNDS):
    if set(bounds) != set(PARAMETER_NAMES):
        raise ValueError('M1 bounds must contain exactly the nine canonical parameter names')
    arr = np.asarray([bounds[k] for k in PARAMETER_NAMES], dtype=float)
    if arr.shape != (9, 2) or not np.isfinite(arr).all() or np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError('Each bound must be a finite increasing pair')
    if not arr[4, 1] < arr[5, 0]:
        raise ValueError('Search bounds must enforce canonical a<b')
    if not arr[0, 0] > arr[1, 1]:
        raise ValueError('Search bounds must preserve a nonzero conductivity contrast')
    if not (0 <= arr[6, 0] < arr[6, 1] < 180):
        raise ValueError('Rotation must be canonical')
    return arr


@dataclass(frozen=True)
class ParameterSubset:
    """A priori fixed parameter subset; normalized optimization coordinates in [0,1]."""
    reference: np.ndarray
    fixed_names: tuple[str, ...] = ()

    def __post_init__(self):
        ref = np.asarray(self.reference, dtype=float)
        if ref.shape != (9,) or not np.isfinite(ref).all():
            raise ValueError('Reference must contain nine finite parameters')
        bounds = check_bounds()
        if np.any(ref < bounds[:, 0]) or np.any(ref > bounds[:, 1]):
            raise ValueError('Reference outside search bounds')
        fixed = tuple(self.fixed_names)
        if len(set(fixed)) != len(fixed) or set(fixed) - set(PARAMETER_NAMES):
            raise ValueError('Invalid fixed parameter names')
        if len(fixed) == len(PARAMETER_NAMES):
            raise ValueError('At least one free parameter required')
        object.__setattr__(self, 'reference', ref.copy())
        object.__setattr__(self, 'fixed_names', fixed)

    @property
    def free_indices(self):
        return np.array([i for i, n in enumerate(PARAMETER_NAMES) if n not in self.fixed_names], dtype=int)

    @property
    def free_names(self):
        return tuple(PARAMETER_NAMES[i] for i in self.free_indices)

    def expand(self, normalized):
        z = np.asarray(normalized, dtype=float)
        if z.shape != (len(self.free_indices),) or not np.isfinite(z).all():
            raise ValueError('Normalized vector shape or values invalid')
        if np.any(z < -1e-10) or np.any(z > 1 + 1e-10):
            raise ValueError('Normalized parameters outside bounds')
        limits = check_bounds()[self.free_indices]
        theta = self.reference.copy()
        theta[self.free_indices] = limits[:, 0] + np.clip(z, 0., 1.) * (limits[:, 1] - limits[:, 0])
        return theta

    def normalize(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (9,) or not np.isfinite(theta).all():
            raise ValueError('Parameter vector must be nine finite numbers')
        lim = check_bounds()[self.free_indices]
        return (theta[self.free_indices] - lim[:, 0]) / (lim[:, 1] - lim[:, 0])


def station_train_holdout_masks(channels, train=('S01', 'S02', 'S03', 'S04', 'S05', 'S06'),
                                holdout=('S07', 'S08')):
    channels = tuple(channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError('Channels must be nonempty and unique')
    train = set(train)
    held = set(holdout)
    if not train or not held or train & held:
        raise ValueError('Training and held-out station groups must be disjoint')
    parts = []
    for channel in channels:
        pieces = channel.rsplit('_', 1)
        if len(pieces) != 2 or not pieces[0] or not pieces[1]:
            raise ValueError('Expected component_station channel naming')
        parts.append(pieces[-1])
    arr = np.asarray(parts)
    mtrain, mhold = np.isin(arr, list(train)), np.isin(arr, list(held))
    if not mtrain.any() or not mhold.any() or np.any(~(mtrain | mhold)):
        raise ValueError('Missing or unexpected channels in requested train/holdout split')
    return mtrain, mhold


@dataclass(frozen=True)
class ObservationLayout:
    channel_train_mask: np.ndarray
    channel_holdout_mask: np.ndarray
    gauge_train_indices: tuple[int, ...]
    gauge_holdout_indices: tuple[int, ...]
    strain_sigma_ns: float = 2.0
    pressure_sigma_pa: float = 20000.0

    def __post_init__(self):
        a, b = np.asarray(self.channel_train_mask, dtype=bool), np.asarray(self.channel_holdout_mask, dtype=bool)
        if a.ndim != 1 or b.shape != a.shape or not a.any() or not b.any() or np.any(a & b):
            raise ValueError('Invalid strain training and held-out masks')
        gt, gh = self.gauge_train_indices, self.gauge_holdout_indices
        if not gt or not gh or set(gt) & set(gh) or min((*gt, *gh)) < 0:
            raise ValueError('Invalid pressure gauge training/holdout indices')
        if any(not np.isfinite(v) or v <= 0 for v in (self.strain_sigma_ns, self.pressure_sigma_pa)):
            raise ValueError('Gaussian noise sigmas must be positive and finite')

    def extract(self, strain, pressure, *, subset='train', normalized=False):
        s, p = np.asarray(strain, dtype=float), np.asarray(pressure, dtype=float)
        if s.ndim != 2 or p.ndim != 2 or s.shape[0] != p.shape[0] or s.shape[1] != len(self.channel_train_mask):
            raise ValueError('Mismatched strain/pressure shapes')
        if p.shape[1] <= max((*self.gauge_train_indices, *self.gauge_holdout_indices)):
            raise ValueError('Missing pressure gauge channels')
        if not np.isfinite(s).all() or not np.isfinite(p).all():
            raise ValueError('Nonfinite observed or predicted response')
        if subset == 'train':
            mask, gauges = self.channel_train_mask, self.gauge_train_indices
        elif subset == 'holdout':
            mask, gauges = self.channel_holdout_mask, self.gauge_holdout_indices
        else:
            raise ValueError('subset must be train or holdout')
        svec, pvec = s[:, mask].ravel(), p[:, list(gauges)].ravel()
        if normalized:
            svec, pvec = svec / self.strain_sigma_ns, pvec / self.pressure_sigma_pa
        return np.concatenate([svec, pvec])

    def rmse(self, strain_a, pressure_a, strain_b, pressure_b, *, subset):
        mask = self.channel_train_mask if subset == 'train' else self.channel_holdout_mask
        gauges = self.gauge_train_indices if subset == 'train' else self.gauge_holdout_indices
        ds = np.asarray(strain_a)[:, mask] - np.asarray(strain_b)[:, mask]
        dp = np.asarray(pressure_a)[:, list(gauges)] - np.asarray(pressure_b)[:, list(gauges)]
        return float(np.sqrt(np.mean(ds**2))), float(np.sqrt(np.mean(dp**2)) / 1000.)


def add_independent_noise(strain, pressure, layout: ObservationLayout, seed: int):
    """Same-model, known-sigma synthetic data. Truth and noise seed saved by runner."""
    rng = np.random.default_rng(seed)
    s, p = np.asarray(strain, float), np.asarray(pressure, float)
    if s.ndim != 2 or p.ndim != 2 or s.shape[0] != p.shape[0]:
        raise ValueError('Invalid truth response shapes')
    return (s + rng.normal(0., layout.strain_sigma_ns, s.shape),
            p + rng.normal(0., layout.pressure_sigma_pa, p.shape))


def fit_multi_start(predict: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
                    observations: tuple[np.ndarray, np.ndarray],
                    layout: ObservationLayout, subset: ParameterSubset,
                    *, n_starts: int = 2, max_nfev: int = 50, seed: int = 314159,
                    diff_step: float = 0.02):
    """Bounded Gaussian MAP with uniform bounded prior, all starts reported.

    ``predict(theta)`` returns (strain_ns, pressure_pa) on all channels/times.
    ``diff_step`` is the relative SciPy numerical derivative step in normalized
    [0,1] optimization coordinates. Check stability if the optimizer stalls.
    """
    if not isinstance(n_starts, int) or not 1 <= n_starts <= 20:
        raise ValueError('n_starts must be 1..20')
    if not isinstance(max_nfev, int) or not 2 <= max_nfev <= 1000:
        raise ValueError('max_nfev must be 2..1000')
    if not np.isfinite(diff_step) or not 0 < diff_step < .5:
        raise ValueError('diff_step must be positive and below 0.5')
    s_obs, p_obs = observations
    data = layout.extract(s_obs, p_obs, normalized=True)
    rng = np.random.default_rng(seed)
    n_calls = 0
    results = []

    def residual(z):
        nonlocal n_calls
        n_calls += 1
        theta = subset.expand(z)
        s_pred, p_pred = predict(theta)
        return layout.extract(s_pred, p_pred, normalized=True) - data

    for start_id in range(n_starts):
        # Midpoint is a repeatable neutral first start; further starts sample
        # independently of the (hidden) generating truth.
        initial = np.full(len(subset.free_indices), .5) if start_id == 0 else rng.uniform(.12, .88, len(subset.free_indices))
        c0 = n_calls
        try:
            opt = least_squares(residual, initial, bounds=(0., 1.),
                                method='trf', jac='2-point', diff_step=diff_step,
                                max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            theta_fit = subset.expand(opt.x)
            results.append(dict(
                start_id=start_id, success=bool(opt.success), status=int(opt.status),
                message=str(opt.message), nfev=int(opt.nfev), njev=None if opt.njev is None else int(opt.njev),
                total_predict_calls=int(n_calls - c0),
                initial_parameters=subset.expand(initial).tolist(),
                fitted_parameters=theta_fit.tolist(),
                cost=float(opt.cost), chi2=float(np.sum(opt.fun**2)),
                residual_norm=float(np.linalg.norm(opt.fun)),
                at_bounds=bool(np.any(np.asarray(opt.x) < 1e-4) or np.any(np.asarray(opt.x) > 1-1e-4)),
                min_normalized_bound_margin=float(np.min(np.minimum(opt.x, 1-opt.x))),
            ))
        except (FloatingPointError, ValueError, RuntimeError, OverflowError) as exc:
            results.append(dict(start_id=start_id, success=False, status=-9,
                                message=f'{type(exc).__name__}: {exc}',
                                total_predict_calls=int(n_calls - c0),
                                initial_parameters=subset.expand(initial).tolist(),
                                fitted_parameters=None, cost=None, chi2=None,
                                at_bounds=None))
    winners = [r for r in results if r['success'] and np.isfinite(r['chi2'])]
    best = min(winners, key=lambda r: r['chi2']) if winners else None
    return best, results, n_calls


def parameter_error_table(truth, estimate):
    t, e = np.asarray(truth, float), np.asarray(estimate, float)
    if t.shape != (9,) or e.shape != (9,):
        raise ValueError('Expected nine-dimensional vectors')
    return [dict(parameter=n, generating_value=float(a), fitted_value=float(b),
                 difference=float(b-a)) for n, a, b in zip(PARAMETER_NAMES, t, e)]
