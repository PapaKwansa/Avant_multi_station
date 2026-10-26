# AVANT Stage 6B-1 — fixed-global hydraulic solver

**Scope:** An add-only hydraulic interface for Stage 6B. It solves the *same* confined homogeneous one-way diffusion equation and half-cell zero-head-change Dirichlet boundaries as Stage 4, but on a fixed physical **global survey** coordinate grid with an independently fixed physical well location. It does **not** change the original analytical strain equations, coordinate or rotation routines, or existing `confined_fv.py`. It does **not** yet implement a geometry-varying mechanical source, spatial pressure-to-tile integration, or DREAM. Those are the next Stage 6B-2/6B-3 steps.

## Why

Stage 4's local FV domain is constrained to `lx=2*a, ly=2*b`. That arrangement becomes unsuitable for Stage 6 because geometry proposals would move artificial hydraulic boundaries. This module keeps fixed global bounds and physical well coordinates from the committed Stage 6A contract, independent of `a,b,theta,x0,y0`.

## Point-well discretization

At off-center wells, rate is distributed to up to four adjacent cell centers using nonnegative **bilinear shape-function weights** that add to one and preserve the prescribed physical well centroid. The well must fall inside the cell-center interpolation rectangle. This is an approximate source representation; the actual borehole radius/skin and near-well singular pressure require special treatment if relevant.

When the domain and well precisely match the old centered 81x81 Stage 4 benchmark, the source collapses to one central cell. A regression test compares the resulting pressure arrays against the original unchanged `ConfinedFVSolver` to tight numerical tolerances.

## Additive files

```
src/avant_model/model/stage6_global_fv.py
scripts/validation/check_stage6_global_fv.py
tests/test_stage6_global_fv.py
README_STAGE6B1_GLOBAL_FV.md
```

The archive contains **only** these files. It depends on committed Stage 6A `stage6_parameters.py`, plus unchanged `confined_fv.py` and `pressure_gauges.py`.

## Windows CMD, from AVANT repository root

```
tar -xf avant_stage6b1_global_fv.zip
python -m pytest -q tests\test_stage6_global_fv.py
python -m pytest -q
python scripts\validation\check_stage6_global_fv.py --contract configs\pressure_superposition\stage6\stage6a_contract_001.json --hydraulic-config configs\pressure_superposition\experiments\injection_fd_30gpm.json --output-dir results\validation\stage6b1_global_fv_001
type results\validation\stage6b1_global_fv_001\stage6b1_global_fv_report.json
start results\validation\stage6b1_global_fv_001\stage6b1_global_pressure_map.png
```

The referenced Stage 4 hydraulic configuration supplies **time step, rate, fluid, and grid count only**. K and Ss are taken from the Stage 6A *reference state* (both 1e-6), NOT the illustrative `K_m_s` and `Ss_per_m` keys in that older input file. The global domain comes from Stage 6A, NOT the old `length_x_m`, `length_y_m` keys.

Stage 6B-1 outputs a self-describing NPZ, JSON manifest (including input SHA256), and PNG/PDF pressure map. These are benchmarks, not paper validation or inferred permeability maps.

## Acceptance criteria for moving to 6B-2

1. New unit tests and full existing regression suite pass.
2. Exactly matching local-domain and centered-well conditions reproduce Stage 4 pressures.
3. Source weights sum to one and preserve the physical well centroid at off-center locations.
4. Global translation invariance, exact time-index sampling and mass balance pass.
5. Boundary location, source resolution and well discretization are explicitly recognized as model approximations.
6. The fixed-global hydraulic domain does not depend on proposed geometry.

## Still necessary in Stage 6B

- **6B-2:** Geometry-dependent rotated-tile pressure integration over the global hydraulic field (quadrature/area averaging, no silent extrapolation), reuse *unchanged* `build_pressure_kernel`, and preserve correct strain-channel ordering.
- **6B-3:** Integrate geometry and gamma into a generalized coupled forward model; quantify pressure/strain support truncation, grid/time/boundary sensitivity, and old/new compatibility.
- **Later:** Stage 6C identifiability, multistart optimization and posterior sampling.
- Larry's independent COMSOL spatial-pressure reference remains a separate scientific validation requirement.
