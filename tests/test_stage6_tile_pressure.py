"""Scientific and shape checks for Stage 6B-2 without modifying legacy mechanics."""
import numpy as np
import pytest
from avant_model.model.stage6_coordinates import SourceRectangle, local_to_global
from avant_model.model.stage6_tile_pressure import RotatedTilePressureOperator


def domain():
    return np.linspace(-1000,1000,101),np.linspace(-1200,1200,121)


def src(theta=0,x0=40,y0=-40,a=160,b=300):
    return SourceRectangle(a,b,theta,x0,y0)


def oper(theta=0,nx=3,ny=5,order=4):
    x,y=domain()
    return RotatedTilePressureOperator(x,y,src(theta),nx,ny,order)


def test_constant_pressure_any_rotation():
    x,y=domain()
    for theta in (0,15,42,89,179):
        op=oper(theta)
        p=np.full((len(y),len(x)),123456.0)
        assert np.allclose(op.mean_pressure_pa(p),123456.,rtol=0,atol=1e-9)
        assert op.integrated_pressure_pa_m2(p) == pytest.approx(
        123456 * 4 * 160 * 300,
        rel=1e-14,
        abs=1e-6,
    )


def test_linear_field_exact_tile_means_for_rotated_rectangles():
    x,y=domain(); xx,yy=np.meshgrid(x,y)
    p=2e5+3.5*xx-2.7*yy
    for theta in (0,15,45,120,179):
        op=oper(theta,4,6,3)
        centers=op.tile_centers_prime_m
        true=2e5+3.5*centers[:,0]-2.7*centers[:,1]
        assert np.allclose(op.mean_pressure_pa(p),true,rtol=0,atol=1e-9)


def test_bilinear_field_exact_tile_average():
    x,y=domain();xx,yy=np.meshgrid(x,y)
    p=1e5+4*xx+2*yy+0.02*xx*yy
    op=oper(29,6,8,4)
    ctr=op.tile_centers_prime_m
    # A rotated rectangular tile has nonzero cross-covariance in global axes.
    # Compare with high-order numerical integration, not just center evaluation.
    ref=oper(29,6,8,12)
    assert np.allclose(op.mean_pressure_pa(p),ref.mean_pressure_pa(p),rtol=1e-13,atol=1e-8)


def test_time_batch_matches_separate_single_snapshots():
    x,y=domain();xx,yy=np.meshgrid(x,y)
    p=np.stack([xx*0,xx*2+yy*3,100+xx*5-yy])
    op=oper(41)
    assert op.mean_pressure_pa(p).shape==(3,15)
    assert np.allclose(op.mean_pressure_pa(p),np.stack([op.mean_pressure_pa(t) for t in p]))
    assert op.integrated_pressure_pa_m2(p).shape==(3,)


def test_zero_field_support_fraction_is_nan_and_nonzero_is_finite():
    x,y=domain();op=oper(25)
    p=np.stack([np.zeros((len(y),len(x))),np.ones((len(y),len(x)))])
    sup=op.pressure_support_summary(p)
    assert np.isnan(sup.ratio_inside_over_global[0])
    assert 0<sup.ratio_inside_over_global[1]<1
    expected=(4*160*300)/((x[1]-x[0])*(y[1]-y[0])*len(x)*len(y))
    assert sup.ratio_inside_over_global[1]==pytest.approx(expected,rel=1e-13)


def test_source_tile_order_x_outermost():
    op=oper(0,nx=2,ny=3,order=2)
    c=op.tile_centers_prime_m
    assert np.allclose(c[:,0],[src().x0_prime_m-80]*3+[src().x0_prime_m+80]*3)
    assert np.allclose(c[:,1],[-240,-40,160]*2)


def test_rotation_preserves_local_to_global_inverse():
    op=oper(15,4,5)
    s=src(15)
    xs=-s.a_m+(np.repeat(np.arange(4),5)+.5)*2*s.a_m/4
    ys=-s.b_m+(np.tile(np.arange(5),4)+.5)*2*s.b_m/5
    xg,yg=local_to_global(xs,ys,s)
    assert np.allclose(op.tile_centers_prime_m,np.column_stack([xg,yg]))


def test_translation_invariance():
    x,y=domain();xx,yy=np.meshgrid(x,y);p=xx**2+2*yy
    o=oper(15,4,5)
    shiftx,shifty=1234.,-2431.
    o2=RotatedTilePressureOperator(x+shiftx,y+shifty,src(15,x0=40+shiftx,y0=-40+shifty),4,5,4)
    assert np.allclose(o.mean_pressure_pa(p),o2.mean_pressure_pa(p),rtol=0,atol=1e-8)


def test_out_of_bounds_source_never_extrapolates():
    x,y=domain()
    with pytest.raises(ValueError,match='Entire rotated source'):
        RotatedTilePressureOperator(x,y,src(0,x0=900),3,3,2)


def test_shapes_and_finiteness_fail_fast():
    x,y=domain();o=oper(0)
    with pytest.raises(ValueError):o.mean_pressure_pa(np.ones((2,2)))
    bad=np.ones((len(y),len(x)));bad[0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):o.mean_pressure_pa(bad)


@pytest.mark.parametrize('bad',[0,-1,1.2,True,'4',33])
def test_quadrature_invalid(bad):
    x,y=domain()
    with pytest.raises(ValueError):RotatedTilePressureOperator(x,y,src(),3,3,bad)


@pytest.mark.parametrize('bad',[0,-1,1.2,True,'3'])
def test_tile_dimensions_invalid(bad):
    x,y=domain()
    with pytest.raises(ValueError):RotatedTilePressureOperator(x,y,src(),bad,4,3)


def test_nonuniform_grid_rejected():
    x,y=domain();x[17]+=.1
    with pytest.raises(ValueError,match='uniform'):RotatedTilePressureOperator(x,y,src(),3,3,2)


def test_tile_integral_matches_expected_domain_linear_integral():
    x,y=domain();xx,yy=np.meshgrid(x,y)
    p=2e5+3.5*xx-2.7*yy
    op=oper(15,5,7,4)
    expected=(2e5+3.5*40-2.7*(-40))*(4*160*300)
    assert op.integrated_pressure_pa_m2(p)==pytest.approx(expected,rel=1e-13)


def test_quadrature_refinement_is_accurate_for_smooth_nontrivial_field():
    x,y=domain();xx,yy=np.meshgrid(x,y)
    p=1e6*np.exp(-((xx-40)**2+(yy+40)**2)/200**2)
    op2=oper(20,6,6,2);op4=oper(20,6,6,4);op12=oper(20,6,6,12)
    e2=np.linalg.norm(op2.mean_pressure_pa(p)-op12.mean_pressure_pa(p))
    e4=np.linalg.norm(op4.mean_pressure_pa(p)-op12.mean_pressure_pa(p))
    assert e4 < e2


def test_does_not_change_inputs():
    x,y=domain(); p=np.ones((len(y),len(x)))
    before=p.copy();op=oper(18)
    op.mean_pressure_pa(p);op.pressure_support_summary(p)
    assert np.array_equal(p,before)


def test_global_pressure_integral_uses_area_not_mean():
    x,y=domain();op=oper(0)
    p=np.ones((len(y),len(x)))
    summary=op.pressure_support_summary(p)
    assert summary.global_abs_pressure_pa_m2==pytest.approx(101*121*20*20)


def test_pressure_quadrature_weight_sum_per_tile():
    op=oper(0,4,5,6)
    assert abs(op._quad_weights.sum()-1)<1e-14
    assert len(op._sample_op.gauge_x_m)==20*36


def test_production_analytical_tile_order_when_available():
    """Integration against unchanged AVANT cuboid partition (runs in real repo)."""
    try:
        from avant_model.model.pressure_superposition import Cuboid, tile_cuboid
    except ImportError:
        pytest.skip('Production pressure_superposition.py not installed in standalone fixture')
    x,y=domain();s=src(21)
    op=RotatedTilePressureOperator(x,y,s,3,4,3)
    parent=Cuboid(s.a_m,s.b_m,2.5,522.5,s.x0_prime_m,s.y0_prime_m,s.theta_deg)
    actual=tile_cuboid(parent,(3,4,1))
    xs,ys=local_to_global([t.xi for t in actual],[t.eta for t in actual],s)
    assert np.allclose(op.tile_centers_prime_m,np.column_stack([xs,ys]),rtol=0,atol=1e-12)
