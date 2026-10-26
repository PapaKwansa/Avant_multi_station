"""Stage 6B-3 forward assembly tests; no modifications to existing mechanics."""
from dataclasses import replace
import copy
import numpy as np
import pytest

from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.model.stage6_coupled_forward import Stage6CoupledForward


def contract():
    return Stage6Contract.from_dict({
        'schema_version':'stage6a-1',
        'parameters':{
            'free':['log10_K_m_s','log10_Ss_per_m','log10_gamma_rel',
                    'a_m','b_m','theta_deg','x0_prime_m','y0_prime_m'],
            'fixed':{'nu':0.25},
            'bounds':{
                'log10_K_m_s':[-7.,-4.], 'log10_Ss_per_m':[-7.,-4.],
                'log10_gamma_rel':[-0.5,0.5], 'a_m':[140.,270.],
                'b_m':[300.,500.], 'theta_deg':[0.,179.999],
                'x0_prime_m':[-100.,200.], 'y0_prime_m':[-50.,250.]},
            'reference_state':{
                'log10_K_m_s':-6.,'log10_Ss_per_m':-6.,'log10_gamma_rel':0.,
                'a_m':215.,'b_m':400.,'theta_deg':15.,
                'x0_prime_m':53.,'y0_prime_m':141.,'nu':.25}},
        'reference_mechanics':{'E_ref_Pa':8.e9,'alpha_ref':.7,'nu_ref':.25,
                               'source_c_m':2.5,'source_h_m':522.5},
        'global_hydraulic_domain':{'xmin_prime_m':-1200.,'xmax_prime_m':1200.,
             'ymin_prime_m':-1600.,'ymax_prime_m':1600.,
             'well_x_prime_m':53.,'well_y_prime_m':141.,
             'source_margin_m':0.,'require_well_in_source':True},
        'hydraulic_storage_mode':'effective_one_way',
        'geometry_semantics':'finite_pressurized_mechanical_support'})


def cfg(nx=15,ny=17):
    return dict(nx=nx,ny=ny,thickness_m=5.,dt_hours=1.,
                injection_hours=1.,total_hours=2.,rho_kg_m3=1000.,
                gravity_m_s2=9.81,injection_rate_gpm=30.)


def stations():
    return dict(station=['S01','S02','S03'],x=[250.,-500.,20.],
                y=[433.,0.,0.],z=[30.,30.,30.])


def model(**kw):
    settings={'splits':(2,2),'quadrature_order':3}
    settings.update(kw)
    return Stage6CoupledForward(contract(),cfg(),stations(),**settings)


def test_reference_forward_shapes_and_zero_initial_pressure():
    m=model()
    out=m.predict(m.contract.reference(),[0,1,2])
    assert out.strain_ns.shape==(3,12)
    assert out.tile_pressure_pa.shape==(3,4)
    assert out.fv_pressure_change_pa.shape==(3,17,15)
    assert out.channels[0]=='eXX_S01'
    assert out.channels[-1]=='eZZ_S03'
    assert np.max(np.abs(out.strain_ns[0]))==0
    assert np.max(np.abs(out.tile_pressure_pa[0]))==0
    assert out.gauge_pressure_pa is None
    assert out.max_mass_balance_residual_m3_s<1e-10


def test_gamma_rel_scales_mechanics_only():
    m=model();s=m.contract.reference()
    a=m.predict(s,[0,1,2]);b=m.predict(replace(s,log10_gamma_rel=np.log10(2.)),[0,1,2])
    assert np.array_equal(a.fv_pressure_change_pa,b.fv_pressure_change_pa)
    assert np.array_equal(a.tile_pressure_pa,b.tile_pressure_pa)
    assert np.allclose(b.strain_ns,2*a.strain_ns,rtol=1e-13,atol=1e-12)
    assert len(m._mechanics_cache)==1


def test_geometry_changes_strain_mapping_without_moving_fv_well():
    m=model();s=m.contract.reference()
    a=m.predict(s,[1,2]);b=m.predict(replace(s,x0_prime_m=70.),[1,2])
    assert np.array_equal(a.fv_pressure_change_pa,b.fv_pressure_change_pa)
    assert np.linalg.norm(a.strain_ns-b.strain_ns)>0
    assert len(m._mechanics_cache)==2
    assert m.hydraulic.well_discretization.effective_x_prime_m==pytest.approx(53.)


def test_hydraulic_change_requires_new_field_not_new_kernel():
    m=model();s=m.contract.reference()
    a=m.predict(s,[1]);b=m.predict(replace(s,log10_K_m_s=-5.8),[1])
    assert not np.array_equal(a.fv_pressure_change_pa,b.fv_pressure_change_pa)
    assert len(m._mechanics_cache)==1


def test_identical_prediction_is_deterministic():
    m=model();s=m.contract.reference()
    a=m.predict(s,[1,2]);b=m.predict(s,[1,2])
    assert np.array_equal(a.strain_ns,b.strain_ns)
    assert np.array_equal(a.fv_pressure_change_pa,b.fv_pressure_change_pa)


def test_gauges_sample_full_fv_independent_of_gamma():
    m=model(gauges_prime_m=[[55.,145.],[-250.,200.]])
    s=m.contract.reference();a=m.predict(s,[1]);b=m.predict(replace(s,log10_gamma_rel=.1),[1])
    assert a.gauge_pressure_pa.shape==(1,2)
    assert np.array_equal(a.gauge_pressure_pa,b.gauge_pressure_pa)
    assert np.all(np.isfinite(a.gauge_pressure_pa))


def test_support_comparison_factor_one_is_same_forward():
    m=model();s=m.contract.reference()
    a=m.predict(s,[0,1,2]);c=m.compare_support(s,[0,1,2],factors=(1.,1.5,2.))
    assert c.splits==((2,2),(3,3),(4,4))
    assert np.array_equal(c.strain_ns[0],a.strain_ns)
    assert c.relative_l2_difference_from_base[0]==0
    assert np.allclose(c.relative_l2_difference_by_time[1:,0],0)
    assert np.isnan(c.relative_l2_difference_by_time[0]).all()
    assert np.isfinite(c.relative_l2_difference_by_time[1:]).all()


@pytest.mark.parametrize('factors',[(1.,1.),(2.,),(.9,1.5),(1.,1.3),
                                    (1.,2.,1.5),(1.,float('nan'))])
def test_unsupported_support_factors_rejected(factors):
    m=model()
    with pytest.raises(ValueError):m.compare_support(m.contract.reference(),[1],factors)


def test_invalid_station_coordinates_rejected():
    s=stations();s['z'][1]=-1
    with pytest.raises(ValueError,match='positive downward'):
        Stage6CoupledForward(contract(),cfg(),s,splits=(2,2))


def test_duplicate_station_identifiers_rejected():
    s=stations();s['station'][2]='S01'
    with pytest.raises(ValueError,match='unique'):
        Stage6CoupledForward(contract(),cfg(),s,splits=(2,2))


def test_mismatched_vertical_thickness_rejected():
    c=cfg();c['thickness_m']=3
    with pytest.raises(ValueError,match='thickness'):
        Stage6CoupledForward(contract(),c,stations(),splits=(2,2))


def test_invalid_state_not_silently_clipped():
    m=model();s=m.contract.reference()
    with pytest.raises(ValueError):m.predict(replace(s,a_m=1000.),[1])


def test_bad_quadrature_order_rejected():
    with pytest.raises(ValueError):model(quadrature_order=True)
