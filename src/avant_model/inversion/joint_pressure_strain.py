"""AVANT Stage 4: joint pressure-gauge / strain inference, fixed geometry.

A single FV pressure solve drives (a) bilinear pressure-gauge predictions and
(b) the verified 27-coefficient pressure projection and analytical strain
superposition, using the existing compressed mechanical matrix H = G @ A.

Independent Gaussian pressure and strain error blocks are a controlled
synthetic assumption, not a field calibration model. The pressure source is
FV pressure_change_pa; never substitute the 27 pressure coefficients for gauges.
"""
from dataclasses import dataclass
import numpy as np
from avant_model.model.pressure_gauges import PressureGaugeOperator
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.coupled_likelihood import gaussian_log_likelihood


@dataclass(frozen=True)
class PressureObservations:
    times_h: np.ndarray
    gauge_names: tuple
    gauge_x_m: np.ndarray
    gauge_y_m: np.ndarray
    observed_pa: np.ndarray
    sigma_pa: float
    training_mask: np.ndarray
    holdout_mask: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.times_h, float)
        names = tuple(map(str, self.gauge_names))
        gx = np.asarray(self.gauge_x_m, float)
        gy = np.asarray(self.gauge_y_m, float)
        values = np.asarray(self.observed_pa, float)
        train = np.asarray(self.training_mask, bool)
        held = np.asarray(self.holdout_mask, bool)
        n = len(names)
        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError('Pressure observation times must be finite, increasing, unique')
        if n < 2 or len(set(names)) != n or any(not name for name in names):
            raise ValueError('Need >=2 unique nonempty gauge names')
        if gx.shape != (n,) or gy.shape != (n,) or not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
            raise ValueError('Pressure gauge coordinates must be finite and shape (n_gauges,)')
        if values.shape != (t.size, n) or not np.all(np.isfinite(values)):
            raise ValueError('Pressure observation array has incorrect shape or nonfinite entries')
        if train.shape != (n,) or held.shape != (n,) or not np.array_equal(~train, held) or not train.any() or not held.any():
            raise ValueError('Pressure training/holdout masks must be complementary and nonempty')
        sigma = float(self.sigma_pa)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Pressure noise standard deviation must be finite and >0')
        for key, value in [('times_h', t), ('gauge_x_m', gx), ('gauge_y_m', gy),
                           ('observed_pa', values), ('training_mask', train), ('holdout_mask', held)]:
            object.__setattr__(self, key, value.copy())
        object.__setattr__(self, 'gauge_names', names)
        object.__setattr__(self, 'sigma_pa', sigma)


def pressure_gauge_masks(gauge_names, holdout_names=('P03',)):
    names = tuple(map(str, gauge_names))
    holdout = set(holdout_names)
    if not holdout or not holdout.issubset(names):
        raise ValueError('Pressure holdout names must occur in gauge_names')
    held = np.array([name in holdout for name in names], bool)
    if not held.any() or held.all():
        raise ValueError('Pressure training and holdout must both be nonempty')
    return ~held, held


class JointPressureStrainLikelihood:
    """Callable joint log likelihood with a single FV solve per proposal.

    ``include_strain`` / ``include_pressure`` permit controlled ablations.
    ``predict`` returns two matrices (strain_ns, gauge_pressure_change_pa).
    """
    def __init__(self, model, strain_observations, pressure_observations, bounds=None,
                 include_strain=True, include_pressure=True):
        self.model = model
        self.strain = strain_observations
        self.pressure = pressure_observations
        self.bounds = bounds if bounds is not None else HydraulicBounds()
        self.include_strain = bool(include_strain)
        self.include_pressure = bool(include_pressure)
        if not (self.include_strain or self.include_pressure):
            raise ValueError('At least one observation type must be included')
        if tuple(model.channels) != tuple(self.strain.channels):
            raise ValueError('Strain channel ordering differs from the mechanical model')
        if not np.array_equal(self.strain.times_h, self.pressure.times_h):
            raise ValueError('Stage 4 requires exactly matching pressure and strain time vectors')
        self.gauge_operator = PressureGaugeOperator(
            model.hydraulic.x_m, model.hydraulic.y_m,
            self.pressure.gauge_x_m, self.pressure.gauge_y_m,
        )

    def predict(self, theta):
        K, Ss = self.bounds.to_physical(theta)
        fv = self.model.hydraulic.solve(K, Ss, self.strain.times_h)
        if not np.allclose(fv.times_h, self.strain.times_h, rtol=0, atol=1e-10):
            raise ValueError('FV prediction and observations have mismatched times')
        flat = fv.pressure_change_pa.reshape(len(fv.times_h), -1)
        coeff = flat @ self.model.P.T
        predicted_strain = coeff @ self.model.H.T
        predicted_pressure = self.gauge_operator.sample(fv.pressure_change_pa)
        if predicted_strain.shape != self.strain.observed_ns.shape:
            raise ValueError('Strain prediction has unexpected shape')
        if predicted_pressure.shape != self.pressure.observed_pa.shape:
            raise ValueError('Pressure prediction has unexpected shape')
        return predicted_strain, predicted_pressure

    def training_residuals(self, theta):
        strain, pressure = self.predict(theta)
        blocks = []
        if self.include_strain:
            blocks.append(((strain[:, self.strain.training_mask]
                            - self.strain.observed_ns[:, self.strain.training_mask])
                           / self.strain.sigma_ns).ravel())
        if self.include_pressure:
            blocks.append(((pressure[:, self.pressure.training_mask]
                            - self.pressure.observed_pa[:, self.pressure.training_mask])
                           / self.pressure.sigma_pa).ravel())
        return np.concatenate(blocks)

    def log_likelihood_blocks(self, theta):
        strain, pressure = self.predict(theta)
        l_strain = gaussian_log_likelihood(
            self.strain.observed_ns[:, self.strain.training_mask],
            strain[:, self.strain.training_mask], self.strain.sigma_ns)
        l_pressure = gaussian_log_likelihood(
            self.pressure.observed_pa[:, self.pressure.training_mask],
            pressure[:, self.pressure.training_mask], self.pressure.sigma_pa)
        return float(l_strain), float(l_pressure)

    def __call__(self, theta):
        if not self.bounds.contains(theta):
            return -np.inf
        try:
            values = self.log_likelihood_blocks(theta)
            return float(values[0] * int(self.include_strain) + values[1] * int(self.include_pressure))
        except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
            return -np.inf


def synthetic_pressure_observations(model, times_h, K_true, Ss_true,
                                    gauge_names, gauge_x_m, gauge_y_m,
                                    *, sigma_pa=20000.0, seed=314159,
                                    holdout_names=('P03',)):
    """Generate ONCE, then save; no re-drawing noise inside likelihood calls."""
    fv = model.hydraulic.solve(K_true, Ss_true, times_h)
    op = PressureGaugeOperator(fv.x_m, fv.y_m, gauge_x_m, gauge_y_m)
    truth = op.sample(fv.pressure_change_pa)
    noise = np.random.default_rng(seed).normal(0.0, sigma_pa, truth.shape)
    train, held = pressure_gauge_masks(gauge_names, holdout_names)
    obs = PressureObservations(fv.times_h, tuple(gauge_names), gauge_x_m, gauge_y_m,
                               truth + noise, sigma_pa, train, held)
    return obs, truth
