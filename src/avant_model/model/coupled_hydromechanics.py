"""Fixed-geometry, homogeneous FV -> cuboid mean/gradient strain forward model.

Pressure in Pa; gradients in Pa/m; strain in nanostrain.
The hydraulic grid axes MUST coincide with the cuboid's local xi/eta axes.
"""
from dataclasses import dataclass
import numpy as np
from avant_model.model.confined_fv import ConfinedFVSolver
from avant_model.model.pressure_superposition import build_pressure_kernel


def regional_design(x, y, nzone=3):
    """Return a mapping from FV cell pressures to region [mean,gx,gy].

    Uses centered ordinary least squares on equal-area cell centers.
    Region ordering: x outermost, y innermost.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx % nzone or ny % nzone:
        raise ValueError('FV grid must be divisible by number of regions')
    result = np.zeros((3*nzone*nzone, nx*ny))
    sx, sy = nx//nzone, ny//nzone
    for ix in range(nzone):
        for iy in range(nzone):
            xs = np.arange(ix*sx, (ix+1)*sx)
            ys = np.arange(iy*sy, (iy+1)*sy)
            xx, yy = np.meshgrid(x[xs], y[ys])
            X = np.column_stack([np.ones(xx.size), (xx-xx.mean()).ravel(), (yy-yy.mean()).ravel()])
            pinv = np.linalg.pinv(X)
            indices = (ys[:,None]*nx + xs[None,:]).ravel()
            result[3*(ix*nzone+iy):3*(ix*nzone+iy)+3, indices] = pinv
    return result


def tile_gradient_design(tiles, lx, ly, fine_n, nzone=3):
    """Map regional [mean,gx,gy] to fine-tile center pressures."""
    if fine_n % nzone:
        raise ValueError('Fine mechanical splits must divide into pressure regions')
    A = np.zeros((len(tiles), 3*nzone*nzone))
    for k, tile in enumerate(tiles):
        ix, iy = tile.ix // (fine_n//nzone), tile.iy // (fine_n//nzone)
        j = 3*(ix*nzone+iy)
        cx = -lx/2 + (ix+0.5)*lx/nzone
        cy = -ly/2 + (iy+0.5)*ly/nzone
        A[k, j:j+3] = [1., tile.xi-cx, tile.eta-cy]
    return A


@dataclass
class CoupledResult:
    times_h: np.ndarray
    strain_ns: np.ndarray
    pressure_coefficients: np.ndarray
    channels: tuple
    max_mass_balance_residual_m3_s: float


class CoupledHydroMechanicalForward:
    def __init__(self, hydraulic_config, geometry, elastic, stations,
                 mechanical_splits=(18,18,1), components=('eXX','eYY','eXY','eZZ')):
        self.hydraulic = ConfinedFVSolver(hydraulic_config)
        fine = tuple(mechanical_splits)
        if fine[0] != fine[1] or fine[2] != 1:
            raise ValueError('This adapter currently supports (N,N,1) only')
        if not np.isclose(self.hydraulic.lx, 2*geometry.a) or not np.isclose(self.hydraulic.ly, 2*geometry.b):
            raise ValueError('FV domain and cuboid local dimensions disagree')
        self.fine_n = fine[0]
        self.kernel = build_pressure_kernel(
            geometry, elastic, x_prime=np.asarray(stations['x'],float),
            y_prime=np.asarray(stations['y'],float), z=np.asarray(stations['z'],float),
            station_names=tuple(stations['station']), splits=fine, components=components)
        self.P = regional_design(self.hydraulic.x_m, self.hydraulic.y_m)
        self.A = tile_gradient_design(self.kernel.tiles, self.hydraulic.lx,
                                      self.hydraulic.ly, self.fine_n)
        self.H = self.kernel.G @ self.A
        self.channels = tuple(self.kernel.channels)

    def predict(self, K_m_s, Ss_per_m, observation_times_h):
        fv = self.hydraulic.solve(K_m_s, Ss_per_m, observation_times_h)
        flat = fv.pressure_change_pa.reshape(len(fv.times_h), -1)
        q = flat @ self.P.T
        strain = q @ self.H.T
        return CoupledResult(fv.times_h, strain, q, self.channels,
                             fv.max_mass_balance_residual_m3_s)
