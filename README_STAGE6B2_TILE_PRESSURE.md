# AVANT Stage 6B-2 — Global FV pressure integration onto rotating mechanical tiles

**Scope:** Add-only intermediate pressure mapper. It does **not** alter the original Larry analytical mechanics, `pressure_superposition.py`, legacy FV, or DREAM. It does **not** compute strain, infer geometry, or validate physics against COMSOL. It uses the existing Stage 6B-1 saved **global** FV pressure field, with the fixed Stage 6A source geometry.

## Why this step matters

The Stage 6B-1 FV solution is on a fixed physical survey grid (x',y'), while each analytical mechanical cuboid belongs to a locally rotated, translated source. Stage 6B-2 integrates the spatial FV pressure over each moving/rotated cuboid footprint. This supplies `p_tile(t)` for the *unchanged* `build_pressure_kernel` in Stage 6B-3. Once parameters vary, the `RotatedTilePressureOperator` is reconstructed for a new source geometry; for fixed geometry its bilinear interpolation weights can be cached.

## Scientific meaning and limitations

- **Pressure is a change from the reference condition, in Pa**; axes are physical global primed survey x',y' in meters, not COMSOL z.
- Uses positive Gauss–Legendre quadrature **inside each rotated tile**, with bilinear interpolation of *FV cell-center* pressure. Tile ordering is x-outermost, y-innermost (same as `pressure_superposition.tile_cuboid` for `(nx,ny,1)`).
- Every corner of the proposed source must be inside the FV **cell-center** interpolation rectangle; the operator **fails rather than extrapolates**.
- The quadrature refinement report compares e.g. 4-point versus 8-point Gauss integration **of the same interpolated field**. It does not test FV grid convergence. Coarse FV cells can underresolve gradients near the injection well.
- The support-proxy ratio compares `integral_source |p_interpolated| dA` with the midpoint `integral_global |p_FV| dA`. This is **not** a fraction of pressure-induced strain, mechanical energy, injected fluid mass, or storage. The two integration rules can differ slightly. For all-zero pressure it is undefined (`null` in JSON).
- A significant pressure integral outside the source does **not** directly measure strain error, but motivates the Stage 6B-3 expanded-support mechanical response benchmark. Likewise, a high inside-pressure ratio would not prove small strain truncation.
- Current mechanics assume a vertically uniform pressure distribution across source thickness. Stage 6B-3 must test this approximation when independent 3D pressure data become available.

## Additive files

```
src/avant_model/model/stage6_tile_pressure.py
scripts/validation/check_stage6_tile_pressure.py
tests/test_stage6_tile_pressure.py
README_STAGE6B2_TILE_PRESSURE.md
```

## Windows commands

From the AVANT repository root, after verifying and committing Stage 6B-1:

```
tar -xf avant_stage6b2_tile_pressure.zip
python -m pytest -q tests\test_stage6_tile_pressure.py
python -m pytest -q
python scripts\validation\check_stage6_tile_pressure.py --contract configs\pressure_superposition\stage6\stage6a_contract_001.json --fields results\validation\stage6b1_global_fv_001\stage6b1_pressure_fields.npz --splits-x 6 --splits-y 6 --quadrature-order 4 --reference-order 8 --output-dir results\validation\stage6b2_tile_pressure_001
type results\validation\stage6b2_tile_pressure_001\stage6b2_tile_pressure_report.json
start results\validation\stage6b2_tile_pressure_001\stage6b2_pressure_tile_mapping.png
start results\validation\stage6b2_tile_pressure_001\stage6b2_pressure_support.png
```

Re-runs require a fresh output directory because experiment results are immutable.

## Acceptance gate into Stage 6B-3

1. All existing and new tests pass; existing mechanical and FV modules remain untouched.
2. Constant, affine, bilinear, and rotated-field pressure tests pass, with correct tile ordering, sign, units, area, translation, and batch behavior.
3. Source outside the FV interpolation rectangle fails loudly. No unstated extrapolation or pressure clipping.
4. Numerical quadrature differences and the pressure-support diagnostic are inspected. No universal fixed 'accuracy threshold' is claimed without comparing to the intended observation/model discrepancy scale.
5. In 6B-3, multiply these tile pressures by the unchanged `build_pressure_kernel(...)` and compare against a higher-resolution **mechanical** partition and **expanded pressurized support**. Also preserve direct global gauge observations and run legacy-compatibility tests under identical domains and source physics.

Larry's spatial COMSOL pressure reference remains an independent validation requirement; no implication of field-level or full two-way poroelastic validation is made.
