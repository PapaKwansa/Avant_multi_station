"""Fast regression tests for the AVANT OED modules."""

import numpy as np

from avant_model.oed.fisher_information import (
    enumerate_station_subsets,
    fisher_diagnostics,
    fisher_information_matrix,
    make_observation_weights,
    subset_jacobian_by_stations,
)
from avant_model.oed.design_metrics import (
    metric_direction,
    robust_design_score,
)
from avant_model.oed.geometry_uncertainty import (
    GEOMETRY_PARAMETER_NAMES,
    boundary_scenarios,
    normalize_weights,
    sample_prior_lhs,
)
from avant_model.oed.robust_design import (
    weighted_quantile,
    weighted_robust_summary,
)
from avant_model.oed.sensor_placement import (
    generate_rectangular_grid,
    greedy_network_by_distance,
    network_minimum_spacing,
    random_candidate_networks,
)
from avant_model.oed.bayesian_design import (
    bayesian_design_summary,
    observation_sigma_vector,
    posterior_sigma_samples,
    representative_posterior_sigma,
)


BOUNDS = {
    "a": [40.0, 900.0],
    "b": [20.0, 1000.0],
    "h": [50.0, 900.0],
    "theta_deg": [-90.0, 90.0],
    "x0_prime": [-900.0, 900.0],
    "y0_prime": [-900.0, 900.0],
}


# ============================================================================
# FISHER INFORMATION
# ============================================================================


def test_fisher_information_matches_jtj():
    jacobian = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    fisher = fisher_information_matrix(jacobian)

    expected = jacobian.T @ jacobian

    np.testing.assert_allclose(
        fisher,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_fisher_information_is_symmetric():
    jacobian = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
        ]
    )

    fisher = fisher_information_matrix(jacobian)

    np.testing.assert_allclose(
        fisher,
        fisher.T,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_fisher_information_is_positive_semidefinite():
    rng = np.random.default_rng(42)
    jacobian = rng.normal(size=(20, 4))

    fisher = fisher_information_matrix(jacobian)

    eigenvalues = np.linalg.eigvalsh(fisher)

    assert np.all(eigenvalues >= -1.0e-10)


def test_fisher_diagnostics_detects_full_rank():
    fisher = np.diag(
        [1.0, 2.0, 3.0]
    )

    result = fisher_diagnostics(fisher)

    assert result.rank == 3


def test_observation_weights_scalar_sigma():
    weights = make_observation_weights(
        4,
        sigma=2.0,
    )

    np.testing.assert_allclose(
        weights,
        np.full(4, 0.25),
    )


# ============================================================================
# STATION SUBSETS
# ============================================================================


def test_enumerate_station_subsets_for_three_stations():
    subsets = list(
        enumerate_station_subsets(3)
    )

    assert subsets == [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]


def test_subset_jacobian_preserves_component_major_order():
    """
    Four components, three stations, one observation per channel.

    Rows are ordered:
        component 1: S0 S1 S2
        component 2: S0 S1 S2
        component 3: S0 S1 S2
        component 4: S0 S1 S2
    """

    jacobian = np.arange(
        24,
        dtype=float,
    ).reshape(12, 2)

    subset = subset_jacobian_by_stations(
        jacobian,
        station_count=3,
        station_indices=[0, 2],
        n_components=4,
    )

    expected_rows = [
        0, 2,
        3, 5,
        6, 8,
        9, 11,
    ]

    np.testing.assert_array_equal(
        subset,
        jacobian[expected_rows],
    )


# ============================================================================
# DESIGN METRICS
# ============================================================================


def test_metric_directions():
    assert metric_direction(
        "d_optimality"
    ) is True

    assert metric_direction(
        "condition_number"
    ) is False


def test_robust_design_score_summary():
    score = robust_design_score(
        "network_A",
        4,
        [10.0, 11.0, 9.0, 12.0, 8.0],
        "d_optimality",
    )

    assert score.design_id == "network_A"
    assert score.design_size == 4
    assert score.n_scenarios == 5
    assert np.isclose(
        score.primary_expected,
        10.0,
    )
    assert score.higher_is_better is True


# ============================================================================
# GEOMETRY UNCERTAINTY
# ============================================================================


def test_geometry_parameter_contract():
    assert tuple(
        GEOMETRY_PARAMETER_NAMES
    ) == (
        "a",
        "b",
        "h",
        "theta_deg",
        "x0_prime",
        "y0_prime",
    )


def test_lhs_is_reproducible():
    first = sample_prior_lhs(
        BOUNDS,
        10,
        seed=42,
    )

    second = sample_prior_lhs(
        BOUNDS,
        10,
        seed=42,
    )

    assert first.equals(second)


def test_lhs_samples_stay_inside_bounds():
    scenarios = sample_prior_lhs(
        BOUNDS,
        20,
        seed=123,
    )

    for parameter, limits in BOUNDS.items():
        assert (
            scenarios[parameter]
            >= limits[0]
        ).all()

        assert (
            scenarios[parameter]
            <= limits[1]
        ).all()


def test_boundary_scenario_count():
    scenarios = boundary_scenarios(
        BOUNDS
    )

    # Reference + low/high perturbation for each
    # of six geometry parameters.
    assert len(scenarios) == 13


def test_normalize_weights_sums_to_one():
    weights = normalize_weights(
        [2.0, 3.0, 5.0]
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )

    np.testing.assert_allclose(
        weights,
        [0.2, 0.3, 0.5],
    )


# ============================================================================
# ROBUST DESIGN
# ============================================================================


def test_weighted_quantile_simple_case():
    result = weighted_quantile(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        0.5,
    )

    assert np.isclose(
        result,
        2.0,
    )


def test_weighted_robust_summary():
    summary = weighted_robust_summary(
        [1.0, 2.0, 3.0],
        [0.2, 0.3, 0.5],
    )

    assert np.isclose(
        summary["weighted_mean"],
        2.3,
    )

    assert summary["n_scenarios"] == 3


# ============================================================================
# SENSOR PLACEMENT
# ============================================================================


def test_rectangular_grid_size():
    locations = generate_rectangular_grid(
        [-500.0, 500.0, -500.0, 500.0],
        5,
        5,
    )

    assert len(locations) == 25


def test_greedy_network_has_requested_size():
    locations = generate_rectangular_grid(
        [-500.0, 500.0, -500.0, 500.0],
        5,
        5,
    )

    network = greedy_network_by_distance(
        locations,
        4,
    )

    assert len(network.sensor_ids) == 4
    assert len(set(network.sensor_ids)) == 4


def test_random_candidate_networks_are_reproducible():
    locations = generate_rectangular_grid(
        [-500.0, 500.0, -500.0, 500.0],
        5,
        5,
    )

    first = random_candidate_networks(
        locations,
        design_size=4,
        n_networks=3,
        seed=42,
    )

    second = random_candidate_networks(
        locations,
        design_size=4,
        n_networks=3,
        seed=42,
    )

    first_ids = [
        network.sensor_ids
        for network in first
    ]

    second_ids = [
        network.sensor_ids
        for network in second
    ]

    assert first_ids == second_ids


def test_greedy_network_has_positive_spacing():
    locations = generate_rectangular_grid(
        [-500.0, 500.0, -500.0, 500.0],
        5,
        5,
    )

    network = greedy_network_by_distance(
        locations,
        4,
    )

    assert (
        network_minimum_spacing(network)
        > 0.0
    )


# ============================================================================
# BAYESIAN OED
# ============================================================================


def test_posterior_sigma_transform():
    samples = np.array(
        [
            [1, 2, 3, 4, 5, 6, 2.0],
            [1, 2, 3, 4, 5, 6, 3.0],
        ],
        dtype=float,
    )

    names = [
        "a",
        "b",
        "h",
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "log10_sigma_strain",
    ]

    sigma = posterior_sigma_samples(
        samples,
        names,
    )

    np.testing.assert_allclose(
        sigma,
        [100.0, 1000.0],
    )


def test_representative_posterior_sigma_median():
    samples = np.array(
        [
            [1, 2, 3, 4, 5, 6, 2.0],
            [1, 2, 3, 4, 5, 6, 2.3],
        ],
        dtype=float,
    )

    names = [
        "a",
        "b",
        "h",
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "log10_sigma_strain",
    ]

    result = representative_posterior_sigma(
        samples,
        names,
        statistic="median",
    )

    expected = np.median(
        10.0 ** samples[:, -1]
    )

    assert np.isclose(
        result,
        expected,
    )


def test_observation_sigma_vector_from_scalar():
    sigma = observation_sigma_vector(
        5.0,
        observation_dimension=4,
    )

    np.testing.assert_allclose(
        sigma,
        np.full(4, 5.0),
    )


def test_bayesian_design_summary():
    summary = bayesian_design_summary(
        "test",
        4,
        [10.0, 11.0, 9.0, 12.0],
        [0.25, 0.25, 0.25, 0.25],
    )

    assert summary.design_id == "test"
    assert summary.design_size == 4
    assert summary.n_scenarios == 4

    assert np.isclose(
        summary.expected_utility,
        10.5,
    )

    assert summary.higher_is_better is True