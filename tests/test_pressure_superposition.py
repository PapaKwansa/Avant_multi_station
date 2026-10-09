"""Regression checks of the add-on against the existing repository mechanics."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from avant_model.model import multi_station_strain as mechanics
from avant_model.model import multi_station_rotation as rotation
from avant_model.model.forward_model_multi_station import (
    forward_model_multi_station, pressure_time_series)
from avant_model.model.pressure_superposition import (
    Cuboid, ElasticProperties, NumericalKernelError, build_pressure_kernel,
    tile_cuboid, coarse_block_basis)

PARENT = Cuboid(190., 325., 3.125, 520., 45., 170., 15.)
ELASTIC = ElasticProperties(8e9, .35, .8)
STATIONS = dict(x_prime=np.array([-420.3, -250.7, 50.3, 450.4]),
                y_prime=np.array([-380.7, 210.2, -520.1, -120.8]),
                z=np.array([0., 25., 0., 18.]),
                station_names=["S01", "S02", "S03", "S04"])


def build(parent=PARENT, elastic=ELASTIC, **kwargs):
    station_args = dict(STATIONS)
    station_args.update(kwargs)
    return build_pressure_kernel(parent, elastic, **station_args)


@pytest.mark.parametrize("splits", [(1, 1, 1), (2, 2, 1), (2, 3, 2), (4, 4, 1)])
@pytest.mark.parametrize("theta", [0., 17., -41., 90.])
def test_partition_matches_legacy_time_series(splits, theta):
    parent = replace(PARENT, theta_deg=theta)
    kernel = build(parent=parent, splits=splits)
    times = np.array([0., .25, 1., 1.5])
    pressure = pressure_time_series(1e6, 1., .4, times)
    legacy = forward_model_multi_station(
        pmax=1e6, tpeak=1., d=.4, time=times, **STATIONS,
        a=parent.a, b=parent.b, c=parent.c, h=parent.h,
        x0_prime=parent.x0_prime, y0_prime=parent.y0_prime, theta_deg=theta,
        E=ELASTIC.E, nu=ELASTIC.nu, alpha=ELASTIC.alpha)
    prediction = kernel.predict(np.repeat(pressure[:, None], len(kernel.tiles), axis=1))
    np.testing.assert_allclose(prediction, legacy[list(kernel.channels)].to_numpy(),
                               rtol=1e-7, atol=1e-7)


def test_volume_and_tile_depths():
    tiles = tile_cuboid(PARENT, (2, 3, 2))
    assert len(tiles) == 12
    assert sum(8*t.a*t.b*t.c for t in tiles) == pytest.approx(8*PARENT.a*PARENT.b*PARENT.c)
    assert min(t.h-t.c for t in tiles) == pytest.approx(PARENT.h-PARENT.c)
    assert max(t.h+t.c for t in tiles) == pytest.approx(PARENT.h+PARENT.c)
    assert any(t.b < t.a for t in tile_cuboid(PARENT, (1, 4, 1)))


def test_coarse_basis_matches_coarse_kernel():
    fine = build(splits=(4, 4, 2))
    coarse = build(splits=(2, 2, 1))
    B = coarse_block_basis((4, 4, 2), (2, 2, 1))
    np.testing.assert_array_equal(B.sum(axis=1), np.ones(32))
    np.testing.assert_allclose((fine.G @ B)*1e6, coarse.G*1e6, rtol=1e-7, atol=1e-7)


def test_signed_pressure_superposition_and_zero():
    k = build()
    p = np.array([3., -4., 7., 2.])*1e5
    q = np.array([2., 1., -1., 3.])*1e5
    np.testing.assert_allclose(k.predict(p+q), k.predict(p)+k.predict(q), atol=1e-10)
    np.testing.assert_allclose(k.predict(-p), -k.predict(p), atol=1e-10)
    np.testing.assert_array_equal(k.predict(np.zeros(4)), np.zeros(16))


def test_horizontal_translation_invariance():
    a = build()
    b = build(parent=replace(PARENT, x0_prime=PARENT.x0_prime+127.,
                            y0_prime=PARENT.y0_prime-67.),
              x_prime=STATIONS['x_prime']+127., y_prime=STATIONS['y_prime']-67.)
    np.testing.assert_allclose(a.G, b.G, rtol=1e-8, atol=1e-15)


def test_material_amplitude_scaling():
    a = build()
    double_E = build(elastic=replace(ELASTIC, E=ELASTIC.E*2))
    half_alpha = build(elastic=replace(ELASTIC, alpha=ELASTIC.alpha/2))
    np.testing.assert_allclose(double_E.G, a.G/2, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(half_alpha.G, a.G/2, rtol=1e-13, atol=1e-16)


def test_named_columns_and_existing_flattening_convention():
    k = build(components=("eZZ", "eXY", "eXX", "eYY"))
    p = np.array([[0., 0., 0., 0.], [1e6, 3e5, 4e5, -2e5]])
    df = k.dataframe([0., 1.], p)
    assert list(df.columns) == ["time_s"]+list(k.channels)
    np.testing.assert_allclose(df[list(k.channels)].to_numpy().T.reshape(-1),
                               np.concatenate([df[c].to_numpy() for c in k.channels]))


def test_six_component_rotation_matches_matrix_formula():
    # Independent tensor transformation check of the posted coordinate convention.
    eps = np.array([[1., 2., 3.], [2., 5., 6.], [3., 6., 9.]])
    angle = 23.
    t = np.deg2rad(angle)
    R = np.array([[np.cos(t), -np.sin(t), 0.], [np.sin(t), np.cos(t), 0.], [0., 0., 1.]])
    global_eps = R.T @ eps @ R
    actual = rotation.rotate_strain_tensor(eps[0,0], eps[1,1], eps[0,1], eps[0,2], eps[1,2], eps[2,2], angle)
    expected = global_eps[[0,1,0,0,1,2], [0,1,1,2,2,2]]
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert build(components=("eXX", "eYY", "eXY", "eXZ", "eYZ", "eZZ")).G.shape == (24,4)


@pytest.mark.parametrize("splits", [(0,1,1), (1.,2,1), (True,1,1), (1,2), (-1,2,1)])
def test_invalid_splits(splits):
    with pytest.raises(ValueError):
        tile_cuboid(PARENT, splits)


def test_invalid_grid_basis():
    with pytest.raises(ValueError):
        coarse_block_basis((3,3,1),(2,2,1))


def test_source_and_surface_guards():
    with pytest.raises(ValueError):
        replace(PARENT, h=PARENT.c)
    with pytest.raises(ValueError, match="Interior"):
        build(x_prime=[45.], y_prime=[170.], z=[520.], station_names=["inside"])
    with pytest.raises(ValueError):
        build(z=[-1., 25., 0., 18.])


def test_bad_station_inputs():
    with pytest.raises(ValueError):
        build(station_names=["same"]*4)
    with pytest.raises(ValueError):
        build(x_prime=[1., 2.])
    with pytest.raises(ValueError):
        build(components=("eXX", "eXX"))


def test_nonfinite_tensor_is_not_silently_repaired(monkeypatch):
    monkeypatch.setattr(mechanics, "strain", lambda **kw: np.full((3,3), np.nan))
    with pytest.raises(NumericalKernelError, match="station"):
        build()


def test_invalid_pressure_and_time_inputs():
    k = build()
    for bad in [np.ones(3), np.ones((4,1)), np.full(4,np.nan)]:
        with pytest.raises(ValueError):
            k.predict(bad)
    with pytest.raises(ValueError):
        k.dataframe([0.,0.], np.ones((2,4)))
    with pytest.raises(ValueError):
        k.dataframe([0.,1.], np.ones(4))


def test_readonly_kernel():
    with pytest.raises(ValueError):
        build().G[0,0] = 0.


def test_free_surface_traction_conditions():
    k = build(z=np.zeros(4), components=("eXX","eYY","eXY","eXZ","eYZ","eZZ"))
    tensors = k.predict(np.array([8.,5.,3.,9.])*1e5).reshape(6,4)
    np.testing.assert_allclose(tensors[3], 0., atol=1e-7)
    np.testing.assert_allclose(tensors[4], 0., atol=1e-7)
    sigma_zz_over_2mu = (ELASTIC.nu/(1-2*ELASTIC.nu)*
                        (tensors[0]+tensors[1]+tensors[5])+tensors[5])
    np.testing.assert_allclose(sigma_zz_over_2mu, 0., atol=1e-7)


def test_noiseless_coarse_pressure_least_squares_recovery():
    k = build()
    truth = np.array([8.,5.5,3.,9.5])*1e5
    # This is a deterministic consistency test, not a zero-variance likelihood.
    estimate, _, rank, _ = np.linalg.lstsq(k.G, k.predict(truth), rcond=None)
    assert rank == 4
    np.testing.assert_allclose(estimate, truth, rtol=1e-8, atol=1e-3)
