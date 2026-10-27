"""Focused M1 OED regression tests: no field data or heavy M1 solves needed."""
import numpy as np
import pytest

from avant_model.oed.m1_joint_design import (
    M1ObservationCatalog, compare_designs, greedy_add_sensors,
    make_m1_jacobian, score_design,
)
from avant_model.inversion.m1_map_recovery import check_bounds


@pytest.fixture
def catalog():
    return M1ObservationCatalog(
        [2., 19., 96.],
        ('eXX_S01', 'eXX_S02', 'eYY_S01', 'eYY_S02'),
        ('P01', 'P02'), strain_sigma_ns=2., pressure_sigma_pa=20000.)


@pytest.fixture
def random_J(catalog):
    rng = np.random.default_rng(300)
    return rng.normal(size=(len(catalog.rows), 9)) * catalog.sigma[:,None]


def test_m1_row_order_is_time_major_and_pressure_last(catalog):
    assert len(catalog.rows) == 18
    assert catalog.rows[0].kind == 'strain'
    assert catalog.rows[1].station == 'S02'
    assert catalog.rows[3].component == 'eYY'
    assert catalog.rows[4].time_index == 1
    assert catalog.rows[12].kind == 'pressure'
    assert catalog.rows[13].gauge == 'P02'
    assert catalog.rows[14].time_index == 1
    assert catalog.rows[17].time_index == 2


def test_catalog_flatten_matches_manual(catalog):
    strain = np.arange(12).reshape(3,4)
    pressure = np.arange(6).reshape(3,2)+100
    y = catalog.flatten(strain, pressure)
    np.testing.assert_array_equal(y, np.r_[np.arange(12), np.arange(6)+100])


def test_station_and_component_masks_select_correct_rows(catalog):
    ids = catalog.indices({'stations':['S02'], 'components':['eYY'], 'gauges':[], 'time_indices':[0,2]})
    np.testing.assert_array_equal(ids, [3,11])


def test_pressure_mask_selects_times_and_gauges(catalog):
    ids = catalog.indices({'stations':[], 'gauges':['P02'], 'time_indices':[1,2]})
    np.testing.assert_array_equal(ids, [15,17])


def test_all_means_all_without_duplicating_rows(catalog):
    ids = catalog.indices({'stations':'all','components':'all','gauges':'all'})
    np.testing.assert_array_equal(ids, np.arange(18))


@pytest.mark.parametrize('bad', [
    {'stations':['S88'],'gauges':[]},
    {'stations':[],'gauges':['P99']},
    {'stations':[],'gauges':[]},
    {'stations':['S01'],'gauges':[],'time_indices':[100]},
    {'stations':['S01'],'gauges':[],'time_indices':[True]},
    {'stations':['S01'],'components':['eZQ'],'gauges':[]},
])
def test_invalid_selection_rejected(catalog,bad):
    with pytest.raises(ValueError):catalog.indices(bad)


def test_fisher_equals_manual_weighted_matrix(catalog, random_J):
    spec = {'name':'joint','stations':['S01'],'components':'all','gauges':['P02'],'time_indices':[0,2]}
    scales = np.arange(1,10)
    out = score_design(random_J,catalog,spec,parameter_scales=scales)
    idx = catalog.indices(spec)
    whiten = random_J[idx] * scales / catalog.sigma[idx,None]
    F = whiten.T @ whiten
    sign, logdet = np.linalg.slogdet(np.eye(9)+F)
    assert sign == 1
    assert out['n_observations'] == 6
    assert out['n_strain'] == 4
    assert out['n_pressure'] == 2
    assert out['rank'] <= 6
    assert out['condition_number_unregularized'] is None
    assert out['information_gain_logdet'] == pytest.approx(logdet,rel=1e-11)


def test_joint_information_not_less_than_subsets(catalog, random_J):
    scales = np.ones(9)
    specs = [
        {'name':'strain','gauges':[]},
        {'name':'pressure','stations':[],'gauges':'all'},
        {'name':'joint'},
    ]
    r = compare_designs(random_J, catalog, specs,parameter_scales=scales)
    assert r[2]['information_gain_logdet'] >= r[0]['information_gain_logdet']-1e-12
    assert r[2]['information_gain_logdet'] >= r[1]['information_gain_logdet']-1e-12
    assert r[0]['n_pressure'] == 0
    assert r[1]['n_strain'] == 0
    assert r[2]['n_observations'] == len(catalog.rows)


def test_greedy_addition_increases_information(catalog, random_J):
    baseline = {'name':'training','stations':['S01'],'gauges':['P01'],'components':'all'}
    steps = greedy_add_sensors(random_J,catalog,base_design=baseline,
                               candidate_stations=['S02'], candidate_gauges=['P02'],
                               max_additions=2,parameter_scales=np.ones(9))
    assert len(steps) == 2
    assert len({s['added_name'] for s in steps}) == 2
    assert steps[1]['new_information_gain'] >= steps[0]['new_information_gain']


def test_central_difference_recovers_linear_nine_parameter_forward(catalog):
    theta = check_bounds().mean(axis=1)
    steps = np.array([.006,.006,.006,.006,2,2,.5,2,2.])
    rng = np.random.default_rng(3000)
    A = rng.normal(size=(len(catalog.rows),9))
    calls = []
    def predict(x):
        calls.append(np.asarray(x,float).copy())
        y = A @ np.asarray(x,float)
        return y[:12].reshape(3,4),y[12:].reshape(3,2)
    j = make_m1_jacobian(predict,theta,catalog,steps)
    np.testing.assert_allclose(j,A,rtol=1e-10,atol=1e-8)
    assert len(calls) == 18


def test_baseline_near_bound_protects_fd(catalog):
    theta = check_bounds().mean(axis=1)
    theta[4] = check_bounds()[4,0] + 0.001
    with pytest.raises(ValueError,match='cross M1 bounds'):
        make_m1_jacobian(lambda _: None,theta,catalog,[.006,.006,.006,.006,2,2,.5,2,2])


def test_zero_sensitivity_pressure_only_rank_deficient(catalog):
    J = np.ones((len(catalog.rows),9))
    J[:,3] = 0
    out = score_design(J,catalog,{'name':'pressure','stations':[]},parameter_scales=np.ones(9))
    assert out['rank'] < 9
    assert out['condition_number_unregularized'] is None


def test_duplicate_names_rejected(catalog):
    J = np.zeros((len(catalog.rows),9))
    with pytest.raises(ValueError, match='distinct'):
        compare_designs(J,catalog,[{'name':'x'},{'name':'x'}],parameter_scales=np.ones(9))
