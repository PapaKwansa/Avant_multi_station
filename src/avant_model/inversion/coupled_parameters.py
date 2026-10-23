"""Bounded homogeneous hydraulic-property parameterization for AVANT.

The first inversion estimates log10(K [m/s]) and log10(Ss [1/m]).
This model deliberately holds geometry, mechanics and aquifer thickness fixed.
"""
from dataclasses import dataclass
import numpy as np

PARAMETER_NAMES = ("log10_K_m_s", "log10_Ss_per_m")
DEFAULT_BOUNDS = ((-7.0, -4.0), (-7.0, -4.0))


@dataclass(frozen=True)
class HydraulicBounds:
    log10_K: tuple = DEFAULT_BOUNDS[0]
    log10_Ss: tuple = DEFAULT_BOUNDS[1]

    def __post_init__(self):
        for name, pair in (("K", self.log10_K), ("Ss", self.log10_Ss)):
            x = np.asarray(pair, dtype=float)
            if x.shape != (2,) or not np.isfinite(x).all() or x[0] >= x[1]:
                raise ValueError(f"Invalid log10 {name} bounds: {pair}")

    def contains(self, theta):
        try:
            x = np.asarray(theta, dtype=float)
        except (ValueError, TypeError):
            return False
        if x.shape != (2,) or not np.isfinite(x).all():
            return False
        return bool(self.log10_K[0] <= x[0] <= self.log10_K[1]
                    and self.log10_Ss[0] <= x[1] <= self.log10_Ss[1])

    def to_physical(self, theta):
        if not self.contains(theta):
            raise ValueError("Hydraulic log10 parameters are invalid or outside prior bounds")
        x = np.asarray(theta, dtype=float)
        return float(10.0 ** x[0]), float(10.0 ** x[1])
