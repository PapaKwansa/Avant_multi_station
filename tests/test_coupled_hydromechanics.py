"""Fast tests for pressure projection and compressed-gradient algebra."""
import numpy as np
from avant_model.model.coupled_hydromechanics import regional_design, tile_gradient_design


def test_region_coefficients_constant_and_linear():
    x = np.linspace(-215+215/81, 215-215/81, 81)
    y = np.linspace(-400+400/81, 400-400/81, 81)
    xx, yy = np.meshgrid(x,y)
    field = 2e5 + 100*xx - 50*yy
    coeff = regional_design(x,y) @ field.ravel()
    assert np.allclose(coeff[1::3],100,atol=1e-9)
    assert np.allclose(coeff[2::3],-50,atol=1e-9)
    assert np.isclose(coeff[0::3].mean(),field.mean())


def test_compression_identity():
    from types import SimpleNamespace
    tiles = [SimpleNamespace(ix=ix,iy=iy,xi=-215+(ix+.5)*430/18,
                             eta=-400+(iy+.5)*800/18)
             for ix in range(18) for iy in range(18)]
    A = tile_gradient_design(tiles,430,800,18)
    rng = np.random.default_rng(3)
    G = rng.normal(size=(32,324))
    q = rng.normal(size=27)
    assert np.allclose(G@(A@q),(G@A)@q,rtol=1e-12,atol=1e-11)
