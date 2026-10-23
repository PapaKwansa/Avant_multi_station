"""In-memory version of AVANT's homogeneous confined, cell-centered FV model.

Same stencil and backward Euler timing as scripts/validation/run_injection_fd.py.
Zero head-change on all four outer boundaries; centered cell injection.
No plotting, file writes, or dependence on AVANT imports.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import splu

GPM_TO_M3S = 0.003785411784 / 60.0

@dataclass
class FVResult:
    times_h: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    head_change_m: np.ndarray
    pressure_change_pa: np.ndarray
    max_mass_balance_residual_m3_s: float
    injected_volume_m3: float

class ConfinedFVSolver:
    def __init__(self, config):
        self.cfg = dict(config)
        c = self.cfg
        self.nx, self.ny = int(c['nx']), int(c['ny'])
        if self.nx < 3 or self.ny < 3 or self.nx % 2 != 1 or self.ny % 2 != 1:
            raise ValueError('nx and ny must be odd integers >= 3 (centered well)')
        self.lx, self.ly, self.b = (float(c[k]) for k in ('length_x_m','length_y_m','thickness_m'))
        self.dt_h = float(c['dt_hours'])
        self.tinj = float(c['injection_hours'])
        self.tend = float(c['total_hours'])
        self.rho = float(c['rho_kg_m3'])
        self.g = float(c['gravity_m_s2'])
        self.q = float(c['injection_rate_gpm']) * GPM_TO_M3S
        vals = (self.lx,self.ly,self.b,self.dt_h,self.tinj,self.tend,self.rho,self.g)
        if not all(np.isfinite(v) and v > 0 for v in vals) or not 0 < self.tinj < self.tend or not np.isfinite(self.q) or self.q < 0:
            raise ValueError('Invalid physical configuration')
        for t in (self.tinj,self.tend):
            if not np.isclose(t/self.dt_h,round(t/self.dt_h),rtol=0,atol=1e-9):
                raise ValueError('Injection and total times must be multiples of dt_hours')
        self.dx, self.dy = self.lx/self.nx, self.ly/self.ny
        self.x_m = (np.arange(self.nx)+.5)*self.dx-self.lx/2
        self.y_m = (np.arange(self.ny)+.5)*self.dy-self.ly/2
        self.n = self.nx*self.ny
        self.well = (self.ny//2)*self.nx+self.nx//2
        self.volume = self.dx*self.dy*self.b
        # 1D Laplacian: half-cell Dirichlet boundaries add 2*t at the boundary.
        def axis_matrix(n, conductance):
            diagonal = np.full(n, 2.0*conductance)
            diagonal[0] += conductance
            diagonal[-1] += conductance
            return diags(([-conductance*np.ones(n-1),diagonal,-conductance*np.ones(n-1)]),(-1,0,1),format='csc')
        self.A_unit = (kron(eye(self.ny,format='csc'),axis_matrix(self.nx,self.b*self.dy/self.dx),format='csc')+
                       kron(axis_matrix(self.ny,self.b*self.dx/self.dy),eye(self.nx,format='csc'),format='csc'))

    def solve(self, K_m_s, Ss_per_m, output_times_h=None):
        K, Ss = float(K_m_s),float(Ss_per_m)
        if not (np.isfinite(K) and K > 0 and np.isfinite(Ss) and Ss > 0):
            raise ValueError('K and Ss must be finite and positive')
        if output_times_h is None:
            requested = np.arange(round(self.tend/self.dt_h)+1,dtype=float)*self.dt_h
        else:
            requested = np.asarray(output_times_h,dtype=float)
            if requested.ndim != 1 or requested.size == 0 or not np.isfinite(requested).all():
                raise ValueError('Output times must be a nonempty finite 1D sequence')
            if np.any(requested < 0) or np.any(requested > self.tend+1e-9):
                raise ValueError('Output times outside simulation')
            if not np.allclose(requested/self.dt_h,np.rint(requested/self.dt_h),rtol=0,atol=1e-8):
                raise ValueError('Output times must coincide with solver steps; no temporal interpolation')
            if len(np.unique(np.rint(requested/self.dt_h).astype(int))) != len(requested):
                raise ValueError('Output times must be distinct')
        step_ids = np.rint(requested/self.dt_h).astype(int)
        storage = Ss*self.volume
        coef = storage/(self.dt_h*3600.)
        A = K*self.A_unit
        lu = splu((A+diags(np.full(self.n,coef),format='csc')).tocsc())
        head = np.zeros(self.n)
        saved = {0:head.copy()} if 0 in step_ids else {}
        source = np.zeros(self.n); source[self.well] = self.q
        max_residual = 0.
        for step in range(1,round(self.tend/self.dt_h)+1):
            qstep = self.q if step*self.dt_h <= self.tinj+1e-10 else 0.
            old = head
            head = lu.solve(coef*old+(source if qstep else 0.))
            residual = qstep - coef*np.sum(head-old) - np.sum(A@head)
            max_residual = max(max_residual,abs(float(residual)))
            if step in step_ids:
                saved[step] = head.copy()
        stack = np.stack([saved[i].reshape(self.ny,self.nx) for i in step_ids])
        return FVResult(requested.copy(),self.x_m.copy(),self.y_m.copy(),stack,stack*self.rho*self.g,
                        max_residual,self.q*self.tinj*3600.)
