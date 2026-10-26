"""Additive mathematical and M1-interface safeguards; no kernel modifications."""
import numpy as np
import pytest
from avant_model.inversion.m1_identifiability_gate import (
    PARAMETERS, scaled_information, central_difference,
    relative_change, rms_difference, relative_column_changes,
)


def test_primary_parameter_count():
    assert len(PARAMETERS) == 9
    assert len(set(PARAMETERS)) == 9


def test_scaled_information_exact_two_parameter_case():
    J=np.diag([1.,2.,3.])
    result=scaled_information(J,np.array([1.,.5,2.]))
    assert result.sensitivity_rank==3
    assert np.allclose(result.singular_values, [6.,1.,1.])
    assert np.allclose(result.column_norms, [1.,1.,6.])
    assert np.allclose(result.column_cosines,np.eye(3))
    assert np.allclose(result.conditional_sd_scaled,[1.,1.,1/6])


def test_information_rank_deficiency_returns_no_finite_inverse():
    J=np.array([[1,1],[2,2],[3,3]],dtype=float)
    info=scaled_information(J,[1.,1.],rtol=1e-3)
    assert info.sensitivity_rank==1
    assert info.conditional_sd_scaled is None
    assert info.singular_values[-1] < 1e-10


def test_information_zero_jacobian():
    info=scaled_information(np.zeros((6,2)),[1.,2.])
    assert info.sensitivity_rank==0
    assert info.conditional_sd_scaled is None


@pytest.mark.parametrize('shape,scales', [((2,3),[1,1,1]),((2,2),[1]),((2,2),[0,1]),((2,2),[1,float('nan')])])
def test_invalid_information_inputs(shape,scales):
    with pytest.raises(ValueError):scaled_information(np.ones(shape),scales)


def test_central_difference_analytic_jacobian():
    def fun(x):return np.array([3*x[0]**2+2*x[1],x[0]-x[1]**2,3.])
    J=central_difference(fun,[1.,2.],[1e-3,1e-3])
    assert np.allclose(J,[[6,2],[1,-4],[0,0]],rtol=1e-9,atol=1e-10)
    assert central_difference(fun,[1.,2.],[1e-3,1e-3],indices=[1]).shape == (3,1)


def test_derivative_shape_or_step_invalid():
    with pytest.raises(ValueError):central_difference(lambda x:np.ones(3),[0,1],[0.1,0])
    with pytest.raises(ValueError):central_difference(lambda x:np.ones((2,2)),[0],[0.1])
    with pytest.raises(ValueError):central_difference(lambda x:np.ones(2),[0],[0.1],indices=[2])


def test_relative_change_rms():
    assert rms_difference(np.array([0,0]),[3,4])==pytest.approx(np.sqrt(12.5))
    assert relative_change([3,4],[3,4])==pytest.approx(0)
    assert relative_change([3,4],[6,8])==pytest.approx(1)
    with pytest.raises(ValueError):relative_change([1,2],[1,2,3])


def test_column_change_catches_unstable_direction():
    a=np.eye(2)
    b=a.copy(); b[:,1]*=1.2
    assert np.allclose(relative_column_changes(a,b),[0.,.2])


def test_m1_geology_difference_degeneracy_in_forward():
    # Full legacy-kernel inference requires Windows repo.  This short test
    # checks only that the two-zone FV loses geometric sensitivity when K1=K0.
    from avant_model.model.stage6_coordinates import SourceRectangle
    from avant_model.inversion.stage6_parameters import FixedHydraulicDomain
    from avant_model.model.stage6_m1_two_zone_fv import TwoZoneGlobalFVSolver
    hc=dict(nx=13,ny=15,thickness_m=5.,dt_hours=.5,
            injection_hours=1.,total_hours=2.,rho_kg_m3=1000.,
            gravity_m_s2=9.81,injection_rate_gpm=30.)
    domain=FixedHydraulicDomain(-400,400,-600,600,0,0,0,True)
    sol=TwoZoneGlobalFVSolver(hc,domain)
    z1=SourceRectangle(100,200,10,0,0)
    z2=SourceRectangle(110,230,55,0,0)
    a=sol.solve_two_zone(1e-6,1e-6,1e-6,z1,[1,2]).flow.pressure_change_pa
    b=sol.solve_two_zone(1e-6,1e-6,1e-6,z2,[1,2]).flow.pressure_change_pa
    assert np.allclose(a,b,rtol=1e-11,atol=1e-7)
