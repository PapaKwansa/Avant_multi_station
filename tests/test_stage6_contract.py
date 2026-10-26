"""Stage 6A parameter contract tests; no forward solve or inverse sampling."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from avant_model.inversion.stage6_parameters import (
    Stage6Contract, Stage6State, FixedHydraulicDomain, ReferenceMechanics,
    PRIMARY_PARAMETER_NAMES,
)

CONFIG_PATH = (Path(__file__).resolve().parents[1] / 'configs' /
               'pressure_superposition' / 'stage6' / 'stage6a_contract_001.json')


@pytest.fixture
def config():
    return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))


@pytest.fixture
def contract(config):
    return Stage6Contract.from_dict(config)


def test_exact_eight_parameter_order(contract):
    assert contract.free_names == PRIMARY_PARAMETER_NAMES
    assert contract.fixed == {'nu':0.25}


def test_reference_matches_larry_benchmark(contract):
    s=contract.reference()
    assert (s.a_m,s.b_m,s.theta_deg,s.x0_prime_m,s.y0_prime_m)==(215,400,15,53,141)
    assert s.K_m_s == pytest.approx(1e-6)
    assert s.Ss_per_m == pytest.approx(1e-6)
    assert s.nu==.25


def test_gamma_one_at_benchmark(contract):
    s=contract.reference()
    assert s.gamma_rel==pytest.approx(1)
    assert s.effective_alpha_over_E_per_Pa(contract.mechanics)==pytest.approx(.7/8e9)
    assert s.E_if_alpha_fixed_Pa(contract.mechanics)==pytest.approx(8e9)


def test_sample_vector_roundtrip_and_dict(contract):
    s=contract.reference()
    x=contract.vector_from_state(s)
    assert x.shape==(8,)
    assert contract.state_from_vector(x)==s
    assert s.as_dict()['nu']==.25


def test_no_independent_alpha_E_can_be_requested(config):
    config['parameters']['free'].append('E')
    with pytest.raises(ValueError,match='E and alpha'):
        Stage6Contract.from_dict(config)


def test_noncanonical_parameter_order_rejected(config):
    config['parameters']['free'][0],config['parameters']['free'][1]=(config['parameters']['free'][1],config['parameters']['free'][0])
    with pytest.raises(ValueError,match='canonical'):
        Stage6Contract.from_dict(config)


def test_fixed_nu_not_also_sampled(config):
    config['parameters']['free'].append('nu')
    with pytest.raises(ValueError,match='simultaneously'):
        Stage6Contract.from_dict(config)


def test_optional_nu_free_can_be_configured(config):
    config['parameters']['free'].append('nu')
    del config['parameters']['fixed']['nu']
    config['parameters']['bounds']['nu']=[0.1,0.4]
    c=Stage6Contract.from_dict(config)
    assert c.vector_from_state(c.reference()).shape==(9,)


def test_reject_nonphysical_storage_mode(config):
    config['hydraulic_storage_mode']='fully_coupled'
    with pytest.raises(ValueError,match='effective_one_way'):
        Stage6Contract.from_dict(config)


def test_reject_unsupported_geometry_semantics(config):
    config['geometry_semantics']='aquifer_boundary'
    with pytest.raises(ValueError,match='finite pressurized mechanical support'):
        Stage6Contract.from_dict(config)


def test_gamma_rescaling_at_fixed_alpha(contract):
    s=contract.reference()
    moved=Stage6State(**{**s.as_dict(), 'log10_gamma_rel': np.log10(2.0)})
    contract.validate_state(moved)
    assert moved.effective_alpha_over_E_per_Pa(contract.mechanics)==pytest.approx(2*.7/8e9)
    assert moved.E_if_alpha_fixed_Pa(contract.mechanics)==pytest.approx(4e9)


def test_reference_nu_must_match_mechanical_reference(config):
    config['reference_mechanics']['nu_ref']=0.30
    with pytest.raises(ValueError,match='nu_ref disagree'):
        Stage6Contract.from_dict(config)


def test_source_domain_and_well_are_fixed(contract):
    s=contract.reference()
    contract.domain.validate_source(s.source)
    assert contract.domain.well_x_prime_m == 53
    assert contract.domain.well_y_prime_m == 141


def test_geometry_can_move_without_domain_moving(contract):
    s=contract.reference()
    moved=Stage6State(**{**s.as_dict(), 'x0_prime_m': 75., 'theta_deg': 25.})
    contract.validate_state(moved)
    assert contract.domain.extent()==(-1200,1200,-1600,1600)


def test_reject_source_outside_domain(contract):
    s=contract.reference()
    bad=Stage6State(**{**s.as_dict(),'x0_prime_m':1e4})
    with pytest.raises(ValueError):
        contract.validate_state(bad)


def test_reject_source_that_does_not_enclose_well(contract):
    s=contract.reference()
    bad=Stage6State(**{**s.as_dict(),'x0_prime_m': 200., 'y0_prime_m':250.,
                      'a_m':150., 'b_m':300., 'theta_deg':140.})
    # The well lies outside the short dimension for this moved, rotated source.
    with pytest.raises(ValueError,match='well'):
        contract.validate_state(bad)


def test_reject_inverted_geometry_even_with_valid_individual_bounds(contract):
    s=contract.reference()
    bad=Stage6State(**{**s.as_dict(), 'a_m':275., 'b_m':300.})
    # Still b>a; this pair should be valid (does not forbid modest aspect ratio).
    contract.validate_state(bad)
    bad=Stage6State(**{**s.as_dict(), 'a_m':301., 'b_m':300.})
    with pytest.raises(ValueError):
        contract.validate_state(bad)


def test_reject_missing_and_nonnumeric_samples(contract):
    with pytest.raises(ValueError):
        contract.state_from_vector(np.zeros(7))
    with pytest.raises(ValueError):
        contract.state_from_vector([float('nan')]*8)


def test_reject_sample_outside_bounds(contract):
    x=contract.vector_from_state(contract.reference())
    x[0]=-8
    with pytest.raises(ValueError,match='outside declared bounds'):
        contract.state_from_vector(x)


def test_reference_outside_bounds_rejected(config):
    config['parameters']['bounds']['a_m']=[230.,275.]
    with pytest.raises(ValueError,match='Reference parameter outside'):
        Stage6Contract.from_dict(config)


def test_fixed_value_must_equal_reference(config):
    config['parameters']['fixed']['nu']=.27
    with pytest.raises(ValueError,match='Fixed value differs'):
        Stage6Contract.from_dict(config)


def test_domain_well_outside_rejected():
    with pytest.raises(ValueError,match='Well'):
        FixedHydraulicDomain(-1,1,-1,1,2,0)


def test_source_depth_below_surface_required():
    with pytest.raises(ValueError):
        ReferenceMechanics(8e9,.7,.25,4.,2.)


def test_invalid_nu_rejected(config):
    config['parameters']['fixed']['nu']=0.5
    config['parameters']['reference_state']['nu']=0.5
    with pytest.raises(ValueError):
        Stage6Contract.from_dict(config)


def test_missing_parameter_rejected(config):
    del config['parameters']['reference_state']['log10_gamma_rel']
    with pytest.raises(ValueError,match='every parameter'):
        Stage6Contract.from_dict(config)


def test_manifest_explicitly_denies_completed_forward_model(contract):
    report=contract.manifest()
    assert 'no new generalized forward inversion' in report['scope']
    assert report['reference_derived']['gamma_rel']==1
    assert 'one_way' in report['storage_mode']
    assert len(report['source_corners_global_prime_m'])==4
