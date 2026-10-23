"""Validated strain observations and explicit station training/holdout masks.

No measured pressure is inferred from the regional pressure coefficients.
Pressure-gauge observation operators will be implemented separately.
"""
from dataclasses import dataclass
import numpy as np


def station_masks(channels, holdout_stations=("S07", "S08")):
    channels = tuple(str(c) for c in channels)
    holdout = set(holdout_stations)
    if not channels or len(channels) != len(set(channels)) or not holdout:
        raise ValueError("Channels must be unique and holdout station list nonempty")
    names = []
    for channel in channels:
        if "_" not in channel:
            raise ValueError(f"Invalid strain channel: {channel}")
        component, station = channel.rsplit("_", 1)
        if not component or not station:
            raise ValueError(f"Invalid strain channel: {channel}")
        names.append(station)
    if not holdout.issubset(set(names)):
        raise ValueError(f"Holdout stations not present: {sorted(holdout - set(names))}")
    is_holdout = np.array([s in holdout for s in names], dtype=bool)
    if not is_holdout.any() or is_holdout.all():
        raise ValueError("Must have both training and holdout channels")
    return ~is_holdout, is_holdout


@dataclass(frozen=True)
class StrainObservations:
    times_h: np.ndarray
    channels: tuple
    observed_ns: np.ndarray
    sigma_ns: float
    training_mask: np.ndarray
    holdout_mask: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.times_h, dtype=float)
        y = np.asarray(self.observed_ns, dtype=float)
        c = tuple(self.channels)
        train = np.asarray(self.training_mask, dtype=bool)
        holdout = np.asarray(self.holdout_mask, dtype=bool)
        if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Observation times must be finite, increasing, unique")
        if y.shape != (len(t), len(c)) or not np.isfinite(y).all():
            raise ValueError("Strain observations must have shape (n_times, n_channels) and be finite")
        if len(c) != len(set(c)) or len(c) == 0:
            raise ValueError("Strain channel names must be unique and nonempty")
        if train.shape != (len(c),) or holdout.shape != (len(c),):
            raise ValueError("Station masks must have one entry per channel")
        if not np.array_equal(~train, holdout) or not train.any() or not holdout.any():
            raise ValueError("Training and holdout masks must be complementary and nonempty")
        if not np.isfinite(self.sigma_ns) or self.sigma_ns <= 0:
            raise ValueError("sigma_ns must be finite and positive")
        object.__setattr__(self, "times_h", t.copy())
        object.__setattr__(self, "observed_ns", y.copy())
        object.__setattr__(self, "channels", c)
        object.__setattr__(self, "training_mask", train.copy())
        object.__setattr__(self, "holdout_mask", holdout.copy())


def make_synthetic_observations(forward_result, *, sigma_ns=2.0, seed=42,
                                holdout_stations=("S07", "S08")):
    """Independent Gaussian noise is a controlled synthetic assumption only."""
    truth = np.asarray(forward_result.strain_ns, dtype=float)
    if not np.isfinite(sigma_ns) or sigma_ns <= 0 or not np.isfinite(truth).all():
        raise ValueError("Invalid synthetic prediction or observation noise")
    train, holdout = station_masks(forward_result.channels, holdout_stations)
    rng = np.random.default_rng(seed)
    observed = truth + rng.normal(0.0, sigma_ns, size=truth.shape)
    return StrainObservations(forward_result.times_h, forward_result.channels,
                              observed, float(sigma_ns), train, holdout)
