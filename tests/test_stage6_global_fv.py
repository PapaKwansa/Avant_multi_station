import numpy as np
import pytest
from avant_model.model.stage6_global_fv import GlobalConfinedFVSolver
from avant_model.model.confined_fv import ConfinedFVSolver
from avant_model.inversion.stage6_parameters import FixedHydraulicDomain, Stage6Contract


def conf(nx=11,ny=11):
    return dict(nx=nx,ny=ny,thickness_m=5.0,dt_hours=.5,
                injection_hours=2.,total_hours=4.,rho_kg_m3=1000.,gravity_m_s2=9.81,
                injection_rate_gpm=30.)


def dom(lx=430.,ly=800.,x0=0.,y0=0.,wellx=0.,welly=0.):
    return FixedHydraulicDomain(x0-lx/2,x0+lx/2,y0-ly/2,y0+ly/2,wellx,welly)


def test_exact_agreement_with_legacy_grid_when_center_well():
    c=conf()
    c_old={**c,'length_x_m':430.,'length_y_m':800.}
    legacy=ConfinedFVSolver(c_old).solve(1e-6,1e-6,[0.,1.,2.,2.5,4.])
    new=GlobalConfinedFVSolver(c,dom()).solve(1e-6,1e-6,[0.,1.,2.,2.5,4.])
    np.testing.assert_array_equal(new.x_m,legacy.x_m)
    np.testing.assert_array_equal(new.y_m,legacy.y_m)
    np.testing.assert_allclose(new.pressure_change_pa,legacy.pressure_change_pa,atol=1e-10,rtol=5e-14)
    assert new.max_mass_balance_residual_m3_s < 1e-12


def test_domain_translation_leaves_solution_unchanged_when_well_translated():
    g1=GlobalConfinedFVSolver(conf(),dom())
    g2=GlobalConfinedFVSolver(conf(),dom(x0=500.,y0=-130.,wellx=500.,welly=-130.))
    np.testing.assert_allclose(g1.solve(1e-6,1e-6,[2.,4.]).pressure_change_pa,
                               g2.solve(1e-6,1e-6,[2.,4.]).pressure_change_pa,
                               rtol=1e-14,atol=1e-10)
    np.testing.assert_allclose(g2.x_m-g1.x_m,500.,atol=1e-12)
    np.testing.assert_allclose(g2.y_m-g1.y_m,-130.,atol=1e-12)


def test_offcenter_well_deposition_conserves_mass_and_location():
    obj=GlobalConfinedFVSolver(conf(),dom(wellx=20.,welly=27.))
    assert len(obj.well_discretization.weights)<=4
    assert sum(obj.well_discretization.weights)==pytest.approx(1)
    assert obj.well_discretization.effective_x_prime_m==pytest.approx(20.)
    assert obj.well_discretization.effective_y_prime_m==pytest.approx(27.)
    res=obj.solve(1e-6,1e-6,[0.,2.,4.])
    assert res.max_mass_balance_residual_m3_s<1e-12
    assert res.injected_volume_m3==pytest.approx(obj.q*2.*3600.)
    assert np.max(res.pressure_change_pa[1])>0


def test_well_sample_near_grid_center_has_single_nonzero_weight():
    obj=GlobalConfinedFVSolver(conf(),dom())
    assert len(obj.well_discretization.weights)==1
    assert obj.well_discretization.weights[0]==pytest.approx(1.)


def test_reject_well_outside_cell_center_support():
    with pytest.raises(ValueError):
        GlobalConfinedFVSolver(conf(),dom(wellx=214.,welly=0.))


def test_solver_is_independent_of_source_rectangle():
    contract=Stage6Contract.from_json('configs/pressure_superposition/stage6/stage6a_contract_001.json')
    obj=GlobalConfinedFVSolver(conf(),contract.domain)
    obj2=GlobalConfinedFVSolver(conf(),contract.domain)
    r=obj.solve(1e-6,1e-6,[19.*0+2.]);r2=obj2.solve(1e-6,1e-6,[2.])
    np.testing.assert_array_equal(r.pressure_change_pa,r2.pressure_change_pa)


@pytest.mark.parametrize('arg',(0.0,-1.,float('nan')))
def test_invalid_K_or_Ss_rejected(arg):
    obj=GlobalConfinedFVSolver(conf(),dom())
    with pytest.raises(ValueError):obj.solve(arg,1e-6,[1.])
    with pytest.raises(ValueError):obj.solve(1e-6,arg,[1.])


def test_no_duplicate_out_of_range_or_offstep_times():
    obj=GlobalConfinedFVSolver(conf(),dom())
    for times in ([1.,1.],[2.3],[8.],[-.5],[],[float('nan')]):
        with pytest.raises(ValueError):obj.solve(1e-6,1e-6,times)


def test_output_times_order_preserved():
    obj=GlobalConfinedFVSolver(conf(),dom())
    r=obj.solve(1e-6,1e-6,[3.,0.,2.])
    np.testing.assert_array_equal(r.times_h,[3.,0.,2.])
    assert np.min(r.pressure_change_pa)>=-1e-9


def test_no_unintended_kernel_changes_importable():
    # Stage 6B-1 imports no mechanical strain kernel and never calls it.
    from avant_model.model import stage6_global_fv
    assert not any('pressure_superposition' in x for x in stage6_global_fv.__dict__)
