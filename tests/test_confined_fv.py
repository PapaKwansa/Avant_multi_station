"""Regression tests for the AVANT in-memory confined FV core."""
import numpy as np
import pytest
from avant_model.model.confined_fv import ConfinedFVSolver

CFG = dict(nx=9,ny=9,length_x_m=430.,length_y_m=800.,thickness_m=5.,
           dt_hours=.5,injection_hours=19.,total_hours=24.,rho_kg_m3=1000.,
           gravity_m_s2=9.81,injection_rate_gpm=30.)

def test_initial_and_mass_balance():
    result = ConfinedFVSolver(CFG).solve(1e-6,1e-6,[0,5,19,24])
    assert np.array_equal(result.pressure_change_pa[0],np.zeros((9,9)))
    assert result.max_mass_balance_residual_m3_s < 1e-10
    assert result.injected_volume_m3 == pytest.approx(129.4611,rel=1e-5)
    assert np.isfinite(result.pressure_change_pa).all()

def test_symmetry_and_center_peak():
    p = ConfinedFVSolver(CFG).solve(1e-6,1e-6,[19]).pressure_change_pa[0]
    np.testing.assert_allclose(p,p[::-1,:],rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(p,p[:,::-1],rtol=1e-11,atol=1e-8)
    assert p[4,4] == pytest.approx(p.max())

def test_invalid_times():
    with pytest.raises(ValueError,match='coincide'):
        ConfinedFVSolver(CFG).solve(1e-6,1e-6,[19.25])

def test_repeatability():
    solver=ConfinedFVSolver(CFG)
    a=solver.solve(1e-6,1e-6,[5,19]).pressure_change_pa
    b=solver.solve(1e-6,1e-6,[5,19]).pressure_change_pa
    np.testing.assert_array_equal(a,b)
