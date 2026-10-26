"""Stage 6A coordinate mappings; source geometries are always global/survey registered."""
from dataclasses import replace
import numpy as np
import pytest

from avant_model.model.stage6_coordinates import (
    SourceRectangle, global_to_local, local_to_global,
    source_corners_global, source_tile_centers_global, contains_global_point,
)


@pytest.fixture
def source():
    return SourceRectangle(215., 400., 15., 53., 141.)


def test_source_center_is_origin(source):
    xi, eta = global_to_local(53., 141., source)
    assert float(xi) == pytest.approx(0)
    assert float(eta) == pytest.approx(0)


@pytest.mark.parametrize('theta', [0., 15., 89.9, 90., 135., 179.999])
def test_roundtrip_for_rotated_grids(source, theta):
    geom = replace(source, theta_deg=theta)
    x = np.array([[1., 5., 8.],[-70., 90., 250.]])
    y = np.array([[400., 5., 150.],[-120., -50., 2.]])
    xi, eta = global_to_local(x, y, geom)
    out_x, out_y = local_to_global(xi, eta, geom)
    np.testing.assert_allclose(out_x, x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out_y, y, rtol=0, atol=1e-12)


def test_theta_zero_matches_shift(source):
    geom=replace(source, theta_deg=0)
    x,y = global_to_local([53, 153],[141,241],geom)
    np.testing.assert_allclose(x,[0,100]);np.testing.assert_allclose(y,[0,100])


def test_theta_ninety_matches_production_sign(source):
    geom=replace(source,theta_deg=90)
    x,y=global_to_local(geom.x0_prime_m+10,geom.y0_prime_m+20,geom)
    assert float(x) == pytest.approx(-20)
    assert float(y) == pytest.approx(10)


def test_rectangle_corners_transform_to_exact_local_rectangle(source):
    corners = source_corners_global(source)
    xi, eta=global_to_local(corners[:,0],corners[:,1],source)
    np.testing.assert_allclose(xi,[-215,215,215,-215],atol=1e-12)
    np.testing.assert_allclose(eta,[-400,-400,400,400],atol=1e-12)


def test_tile_center_order_is_x_outer_y_inner(source):
    centers=source_tile_centers_global(source,2,3)
    assert centers.shape==(6,2)
    xi, eta=global_to_local(centers[:,0],centers[:,1],source)
    np.testing.assert_allclose(xi,[-107.5]*3+[107.5]*3,atol=1e-12)
    np.testing.assert_allclose(eta,[-400+800/6,0,400-800/6]*2,atol=1e-12)


def test_point_in_source(source):
    assert contains_global_point(source,53,141)
    assert not contains_global_point(source,1e6,141)


@pytest.mark.parametrize('bad',[(-1,400,15,53,141),(215,215,15,53,141),
                                (400,215,15,53,141),(215,400,-1,53,141),
                                (215,400,180,53,141),(215,400,float('nan'),53,141)])
def test_invalid_source_rejected(bad):
    with pytest.raises(ValueError):
        SourceRectangle(*bad)


def test_reject_coordinate_nan_and_bad_sampling(source):
    with pytest.raises(ValueError):
        global_to_local(np.nan,1,source)
    with pytest.raises(ValueError):
        local_to_global(1,np.inf,source)
    with pytest.raises(ValueError):
        source_tile_centers_global(source,2.5,3)
    with pytest.raises(ValueError):
        source_tile_centers_global(source,True,3)


def test_legacy_transform_equivalence_if_available(source):
    """Runs against production transform when installed in the actual repo."""
    legacy = pytest.importorskip('avant_model.model.multi_station_coord_transform')
    px,py=np.array([53.,90.,-100.]),np.array([141.,-20., 450.])
    x0,y0=legacy.primed_to_unprimed(px,py,source.x0_prime_m,source.y0_prime_m,source.theta_deg)
    x1,y1=global_to_local(px,py,source)
    np.testing.assert_allclose(x0,x1,rtol=0,atol=1e-12)
    np.testing.assert_allclose(y0,y1,rtol=0,atol=1e-12)


def test_tile_centers_match_production_tiling_if_available(source):
    """Additional integration check in actual AVANT repository (skips standalone)."""
    legacy = pytest.importorskip('avant_model.model.pressure_superposition')
    cuboid = legacy.Cuboid(a=source.a_m, b=source.b_m, c=2.5, h=522.5,
                           x0_prime=source.x0_prime_m, y0_prime=source.y0_prime_m,
                           theta_deg=source.theta_deg)
    tiles = legacy.tile_cuboid(cuboid, splits=(6, 6, 1))
    expected_local = np.array([(tile.xi, tile.eta) for tile in tiles])
    centers = source_tile_centers_global(source, 6, 6)
    xi, eta = global_to_local(centers[:,0], centers[:,1], source)
    np.testing.assert_allclose(np.column_stack([xi,eta]), expected_local,
                               rtol=0, atol=1e-12)
