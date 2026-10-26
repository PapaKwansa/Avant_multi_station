"""Numerical diagnostics for a geology-informed, two-zone AVANT forward model.

This module does *not* solve inverse problems or claim global identifiability.
The local design matrix is whitened by assumed observation errors and scaled
by explicitly declared physically meaningful parameter perturbation scales.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PARAMETERS = ('log10_K_inside', 'log10_K_outside', 'log10_Ss',
              'log10_gamma_rel', 'a_m', 'b_m', 'theta_deg',
              'x0_prime_m', 'y0_prime_m')


@dataclass(frozen=True)
class LocalInformation:
    singular_values: np.ndarray
    column_norms: np.ndarray
    column_cosines: np.ndarray
    weakest_direction: np.ndarray
    sensitivity_rank: int
    rank_tolerance: float
    conditional_sd_scaled: np.ndarray | None
    numerical_condition: float


def scaled_information(jacobian, scales, *, rtol=1e-3):
    """Analyze data-standardized Jacobian with columns scaled to relevant ranges.

    Rank cutoff is a *screening heuristic*; it cannot establish structural or
    global uniqueness. ``conditional_sd_scaled`` is Wald/Fisher-only and is
    omitted for rank-deficient matrices.
    """
    J = np.asarray(jacobian, dtype=float)
    s = np.asarray(scales, dtype=float)
    if J.ndim != 2 or J.shape[1] == 0 or J.shape[0] < J.shape[1]:
        raise ValueError('Jacobian must have observations >= parameters')
    if s.shape != (J.shape[1],) or not (np.isfinite(s).all() and np.all(s > 0)):
        raise ValueError('Finite, positive scales required, one per parameter')
    if not np.isfinite(J).all():
        raise ValueError('Jacobian must be finite')
    if not np.isfinite(rtol) or not (0 < rtol < 1):
        raise ValueError('rank tolerance must be in (0,1)')
    Js = J * s[None, :]
    u, singular, vt = np.linalg.svd(Js, full_matrices=False)
    threshold = float(rtol * singular[0]) if singular[0] else 0.0
    rank = int(np.sum(singular > threshold)) if singular[0] else 0
    norm = np.linalg.norm(Js, axis=0)
    gram = Js.T @ Js
    cos = np.divide(gram, norm[:,None]*norm[None,:], out=np.zeros_like(gram),
                    where=(norm[:,None]*norm[None,:]) > 0)
    sd = (np.sqrt(np.sum((vt.T / singular)**2, axis=1))
          if rank == J.shape[1] and singular[-1] > 0 else None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf')
    return LocalInformation(singular, norm, cos, vt[-1,:], rank, threshold,
                            sd, condition)


def relative_change(reference, comparator, *, eps=1e-12):
    """Euclidean change in common data units; not a measurement-error claim."""
    a = np.asarray(reference, dtype=float)
    b = np.asarray(comparator, dtype=float)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite arrays of matching shape required')
    return float(np.linalg.norm(b-a)/max(np.linalg.norm(a),eps))


def rms_difference(reference, comparator):
    a,b = np.asarray(reference, dtype=float),np.asarray(comparator,dtype=float)
    if a.shape!=b.shape or a.size==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite arrays of matching shape required')
    return float(np.sqrt(np.mean((b-a)**2)))


def central_difference(fn, theta, step, *, indices=None):
    """Central-difference a forward response; return shape (n_obs, n_free)."""
    x = np.asarray(theta, dtype=float)
    h = np.asarray(step, dtype=float)
    if x.ndim != 1 or h.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError('Finite theta and positive derivative steps required')
    ix = range(len(x)) if indices is None else tuple(indices)
    cols = []
    for i in ix:
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(x):
            raise ValueError('Invalid parameter index')
        plus, minus = x.copy(), x.copy()
        plus[i] += h[i]
        minus[i] -= h[i]
        yp, ym = np.asarray(fn(plus), dtype=float), np.asarray(fn(minus),dtype=float)
        if yp.ndim!=1 or yp.shape != ym.shape or not np.isfinite(yp).all() or not np.isfinite(ym).all():
            raise ValueError('Forward response must be finite, consistent 1D vectors')
        cols.append((yp-ym)/(2*h[i]))
    return np.column_stack(cols)


def relative_column_changes(a,b):
    x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if x.shape!=y.shape or x.ndim!=2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite 2D matching matrices required')
    return np.linalg.norm(y-x,axis=0)/np.maximum(np.linalg.norm(x,axis=0),1e-14)
