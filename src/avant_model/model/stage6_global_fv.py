"""Stage 6B-1: fixed-global homogeneous confined FV; does not touch legacy mechanics.

This solver reproduces the existing cell-centered backward-Euler and half-cell
zero-head-change boundary stencil on a FIXED survey-coordinate grid.
The physical well remains fixed and uses normalized bilinear deposition to
up to four surrounding cell centers, which conserves injected volume and avoids
jumping between cells when spatial resolution is changed. It is NOT a line-well
model, 3D poroelastic flow solver, or variable-K solver.

The solve method returns the existing FVResult and is independent of sampled
mechanical source geometry. In particular, source a,b,theta,x0,y0 are not used.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import splu

from avant_model.model.confined_fv import FVResult, GPM_TO_M3S
from avant_model.model.pressure_gauges import PressureGaugeOperator
from avant_model.inversion.stage6_parameters import FixedHydraulicDomain


@dataclass(frozen=True)
class WellDiscretization:
    requested_x_prime_m: float
    requested_y_prime_m: float
    cell_indices_flat: tuple[int, ...]
    weights: tuple[float, ...]
    effective_x_prime_m: float
    effective_y_prime_m: float


class GlobalConfinedFVSolver:
    """Fixed-coordinate pressure solver with a spatially located point-source approximation.

    The well must be inside the cell-CENTER interpolation rectangle. This is
    deliberately stricter than simply falling inside the physical domain:
    bilinear source deposition is undefined at edge half-cells.
    """
    def __init__(self, config: Mapping, domain: FixedHydraulicDomain):
        c = dict(config)
        self.cfg = c
        if not isinstance(domain, FixedHydraulicDomain):
            raise TypeError('domain must be a Stage 6 FixedHydraulicDomain')
        self.domain = domain
        self.nx, self.ny = int(c['nx']), int(c['ny'])
        if self.nx < 3 or self.ny < 3:
            raise ValueError('nx and ny must each be >= 3')
        self.lx = domain.xmax_prime_m-domain.xmin_prime_m
        self.ly = domain.ymax_prime_m-domain.ymin_prime_m
        self.b = float(c['thickness_m'])
        self.dt_h = float(c['dt_hours'])
        self.tinj = float(c['injection_hours'])
        self.tend = float(c['total_hours'])
        self.rho = float(c['rho_kg_m3'])
        self.g = float(c['gravity_m_s2'])
        self.q = float(c['injection_rate_gpm'])*GPM_TO_M3S
        vals=(self.b,self.dt_h,self.tinj,self.tend,self.rho,self.g)
        if any(not np.isfinite(v) or v<=0 for v in vals) or not 0<self.tinj<self.tend or not np.isfinite(self.q) or self.q<0:
            raise ValueError('Invalid hydraulic configuration')
        for t in (self.tinj,self.tend):
            if not np.isclose(t/self.dt_h,round(t/self.dt_h),rtol=0,atol=1e-9):
                raise ValueError('injection_hours and total_hours must be time step multiples')
        self.dx,self.dy=self.lx/self.nx,self.ly/self.ny
        self.x_m=domain.xmin_prime_m+(np.arange(self.nx)+0.5)*self.dx
        self.y_m=domain.ymin_prime_m+(np.arange(self.ny)+0.5)*self.dy
        self.n=self.nx*self.ny
        self.volume=self.dx*self.dy*self.b
        self.well_operator=PressureGaugeOperator(
            self.x_m,self.y_m,
            np.array([domain.well_x_prime_m]),
            np.array([domain.well_y_prime_m]))
        self.source=np.zeros(self.n)
        indices=self.well_operator.indices[0]
        weights=self.well_operator.weights[0]
        np.add.at(self.source,indices,weights)
        # Source stencil must be a non-negative partition of unity.
        if np.min(self.source)<-1e-14 or not np.isclose(self.source.sum(),1,rtol=0,atol=2e-14):
            raise RuntimeError('Well source weights violate conservation')
        nz=np.flatnonzero(self.source>1e-14)
        w=self.source[nz]
        wx=float(np.sum(w*self.x_m[nz%self.nx]))
        wy=float(np.sum(w*self.y_m[nz//self.nx]))
        self.well_discretization=WellDiscretization(
            domain.well_x_prime_m,domain.well_y_prime_m,
            tuple(int(i) for i in nz),tuple(float(v) for v in w),wx,wy)
        if abs(wx-domain.well_x_prime_m)>self.dx*1e-10 or abs(wy-domain.well_y_prime_m)>self.dy*1e-10:
            raise RuntimeError('Well deposition does not preserve physical coordinates')
        # Same half-cell Dirichlet boundary stencil as legacy confined_fv.py.
        def axis_matrix(n,t):
            diagonal=np.full(n,2*t)
            diagonal[0]+=t;diagonal[-1]+=t
            return diags(([-t*np.ones(n-1),diagonal,-t*np.ones(n-1)]),(-1,0,1),format='csc')
        self.A_unit=(kron(eye(self.ny,format='csc'),axis_matrix(self.nx,self.b*self.dy/self.dx),format='csc') +
                     kron(axis_matrix(self.ny,self.b*self.dx/self.dy),eye(self.nx,format='csc'),format='csc'))

    def solve(self,K_m_s,Ss_per_m,output_times_h=None):
        K,Ss=float(K_m_s),float(Ss_per_m)
        if not np.isfinite(K) or not np.isfinite(Ss) or K<=0 or Ss<=0:
            raise ValueError('K and Ss must be positive and finite')
        if output_times_h is None:
            times=np.arange(round(self.tend/self.dt_h)+1,dtype=float)*self.dt_h
        else:
            times=np.asarray(output_times_h,dtype=float)
            if times.ndim!=1 or times.size==0 or not np.isfinite(times).all():
                raise ValueError('output times must be a finite nonempty vector')
            if np.any(times<0) or np.any(times>self.tend+1e-9):
                raise ValueError('requested output time outside simulation')
            if not np.allclose(times/self.dt_h,np.rint(times/self.dt_h),rtol=0,atol=1e-8):
                raise ValueError('output times must coincide with solver steps')
            if len(np.unique(np.rint(times/self.dt_h).astype(int)))!=len(times):
                raise ValueError('requested times must be distinct')
        ids=np.rint(times/self.dt_h).astype(int)
        coef=Ss*self.volume/(self.dt_h*3600.0)
        A=K*self.A_unit
        lu=splu((A+diags(np.full(self.n,coef),format='csc')).tocsc())
        head=np.zeros(self.n)
        saved={0:head.copy()} if 0 in ids else {}
        max_residual=0.0
        source_term=self.q*self.source
        steps=round(self.tend/self.dt_h)
        for step in range(1,steps+1):
            is_injection=step*self.dt_h<=self.tinj+1e-10
            old=head
            head=lu.solve(coef*old+(source_term if is_injection else 0.0))
            qstep=self.q if is_injection else 0.0
            residual=qstep-coef*np.sum(head-old)-np.sum(A@head)
            max_residual=max(max_residual,abs(float(residual)))
            if step in ids:
                saved[step]=head.copy()
        stack=np.stack([saved[int(i)].reshape(self.ny,self.nx) for i in ids])
        return FVResult(times.copy(),self.x_m.copy(),self.y_m.copy(),stack,
                        stack*self.rho*self.g,max_residual,self.q*self.tinj*3600.0)
