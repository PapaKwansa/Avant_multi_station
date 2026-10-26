"""Tests for bounded M1 pilot; actual legacy kernel used when installed in AVANT."""
import numpy as np
import pytest

from avant_model.inversion.stage6_parameters import FixedHydraulicDomain, ReferenceMechanics
from avant_model.model.stage6_coordinates import SourceRectangle
from avant_model.model.stage6_global_fv import GlobalConfinedFVSolver
from avant_model.model.stage6_m1_two_zone_fv import TwoZoneGlobalFVSolver, zone_cell_fractions
from avant_model.model.stage6_m1_forward import Stage6M1Forward, M1State


def config(nx=21, ny=25):
    return dict(nx=nx,ny=ny, thickness_m=5.,dt_hours=.5,
                injection_hours=1.,total_hours=2.,rho_kg_m3=1000.,
                gravity_m_s2=9.81,injection_rate_gpm=30.)


def domain():
    return FixedHydraulicDomain(-600.,600.,-800.,800.,0.,0.,0.,True)


def zone(a=160,b=300,theta=15,x0=0,y0=0):
    return SourceRectangle(a,b,theta,x0,y0)


def mechanics():
    return ReferenceMechanics(8.e9,.7,.25,2.5,522.5)


def stations():
    return dict(station=['S01','S02'],x=[400.,-450.],y=[500.,-500.],z=[30.,30.])


def test_zone_fraction_axis_aligned_exact_one_cell():
    x = np.array([-1.,0.,1.]);y=np.array([-1.,0.,1.])
    z=zone(a=.25,b=.3,theta=0.,x0=0,y0=0)
    r=zone_cell_fractions(x,y,1.,1.,z)
    assert r[1,1]==pytest.approx(4*.25*.3)
    assert np.count_nonzero(r)==1


def test_fraction_area_is_rotationally_consistent():
    x=np.arange(-9.5,10.,1.);y=np.arange(-9.5,10.,1.)
    for th in (0,15,43,89,179):
        z=zone(a=3,b=6,theta=th,x0=0,y0=0)
        f=zone_cell_fractions(x,y,1.,1.,z)
        assert np.all((f>=0)&(f<=1))
        assert f.sum()==pytest.approx(4*z.a_m*z.b_m,rel=1e-12,abs=1e-10)


def test_fraction_translation_on_translated_grid():
    x=np.arange(-9.5,10.,1.);y=np.arange(-9.5,10.,1.)
    f1=zone_cell_fractions(x,y,1.,1.,zone(a=3,b=6,theta=34))
    f2=zone_cell_fractions(x+150,y-250,1.,1.,zone(a=3,b=6,theta=34,x0=150,y0=-250))
    assert np.allclose(f1,f2,rtol=0,atol=1e-12)


def test_homogeneous_limit_matches_stage6_global_solver():
    c=config();d=domain();s=TwoZoneGlobalFVSolver(c,d)
    ref=GlobalConfinedFVSolver(c,d).solve(1e-6,1e-6,[0,.5,1,2])
    for theta in (0,15,69,179):
        new=s.solve_two_zone(1e-6,1e-6,1e-6,zone(theta=theta),[0,.5,1,2])
        assert np.allclose(new.flow.pressure_change_pa,ref.pressure_change_pa,rtol=3e-11,atol=1e-6)
        assert new.max_mass_balance_residual_m3_s<1e-12


def test_uniform_contrast_yields_positive_pressure_and_mass_balance():
    fv=TwoZoneGlobalFVSolver(config(),domain()).solve_two_zone(1e-6,3e-7,1e-6,zone(),[0,1,2])
    assert np.max(fv.flow.pressure_change_pa[1])>0
    assert np.min(fv.flow.pressure_change_pa)>=-1e-8
    assert fv.max_mass_balance_residual_m3_s < 1e-12
    assert np.max(fv.effective_K_m_s)<=1e-6
    assert np.min(fv.effective_K_m_s)>=3e-7
    assert np.any((fv.zone_fraction>0)&(fv.zone_fraction<1))


def test_zone_moves_pressure_when_there_is_a_contrast():
    s=TwoZoneGlobalFVSolver(config(),domain())
    p1=s.solve_two_zone(1e-6,2e-7,1e-6,zone(theta=0),[1]).flow.pressure_change_pa
    p2=s.solve_two_zone(1e-6,2e-7,1e-6,zone(theta=60),[1]).flow.pressure_change_pa
    assert not np.allclose(p1,p2,rtol=1e-5,atol=1e-6)


def test_no_contrast_geometry_invisible_to_hydraulic_model():
    s=TwoZoneGlobalFVSolver(config(),domain())
    p1=s.solve_two_zone(1e-6,1e-6,1e-6,zone(theta=0),[1]).flow.pressure_change_pa
    p2=s.solve_two_zone(1e-6,1e-6,1e-6,zone(theta=60),[1]).flow.pressure_change_pa
    assert np.allclose(p1,p2,rtol=5e-12,atol=1e-7)


def test_invalid_K_Ss_fail_fast():
    s=TwoZoneGlobalFVSolver(config(),domain())
    for v in (-1,0,np.nan,np.inf):
        with pytest.raises(ValueError):s.solve_two_zone(v,1e-6,1e-6,zone(),[1])
        with pytest.raises(ValueError):s.solve_two_zone(1e-6,v,1e-6,zone(),[1])
        with pytest.raises(ValueError):s.solve_two_zone(1e-6,1e-6,v,zone(),[1])


def test_invalid_times_fail_fast():
    s=TwoZoneGlobalFVSolver(config(),domain())
    for t in ([.25],[3],[-1],[1,1],[]):
        with pytest.raises(ValueError):s.solve_two_zone(1e-6,1e-6,1e-6,zone(),t)


def forward():
    return Stage6M1Forward(config(),domain(),mechanics(),stations(),
                           zone(a=400,b=550,theta=0),splits=(2,3),quadrature_order=2,
                           gauges_prime_m=[[0.,0.],[200.,0.]])


def test_integrated_forward_has_output_and_zero_initial():
    model=forward()
    r=model.predict(M1State(1e-6,3e-7,1e-6,1.0,zone()),[0.,1.,2.])
    assert r.fv_pressure_pa.shape==(3,25,21)
    assert r.strain_ns.shape==(3,8)
    assert r.pressure_gauges_pa.shape==(3,2)
    assert np.allclose(r.strain_ns[0],0)
    assert np.allclose(r.pressure_gauges_pa[0],0)
    assert np.isfinite(r.strain_ns).all()
    assert r.max_mass_balance_residual_m3_s<1e-12


def test_gamma_scales_strain_not_pressure():
    model=forward()
    s1=M1State(1e-6,2e-7,1e-6,1.,zone())
    s2=M1State(1e-6,2e-7,1e-6,2.,zone())
    a,b=model.predict(s1,[1,2]),model.predict(s2,[1,2])
    assert np.allclose(a.fv_pressure_pa,b.fv_pressure_pa)
    assert np.allclose(b.strain_ns,2*a.strain_ns,rtol=1e-12,atol=1e-8)


def test_geology_is_not_mechanical_support():
    model=forward()
    a,b=model.predict(M1State(1e-6,1e-6,1e-6,1.,zone(theta=5)),[1]),model.predict(
        M1State(1e-6,1e-6,1e-6,1.,zone(theta=72)),[1])
    assert np.allclose(a.strain_ns,b.strain_ns,rtol=1e-12,atol=1e-7)
    assert model.support != zone()


def test_pressure_outside_geology_is_loaded():
    model=forward()
    r=model.predict(M1State(1e-6,3e-7,1e-6,1.,zone()),[1])
    p=r.fv_pressure_pa[0]
    assert np.any(p[r.zone_fraction<.1]>0)
    assert model.tile_op.n_tiles==6


def test_negative_gamma_rejected():
    with pytest.raises(ValueError):M1State(1e-6,1e-6,1e-6,-1,zone()).validate()
