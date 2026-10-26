# Stage 6B-3 — Integrated fixed-global hydraulic–mechanical forward adapter

**Purpose:** Connect the Stage 6B-1 fixed-global FV solution and Stage 6B-2
rotated-tile area-averaging operator to the **unchanged** AVANT analytical
pressure-superposition strain kernel. This is the **last forward-assembly
checkpoint** before Stage 6C identifiability and Stage 6D/6E inference.

## Scope and non-negotiable limitations

- Add-only implementation. **Do not modify** `multi_station_strain.py`,
  `multi_station_rotation.py`, `multi_station_coord_transform.py`,
  `pressure_superposition.py`, `confined_fv.py`, or older inversion runners.
- 2D homogeneous confined **one-way effective-storage** hydraulic model.
  No claim of full two-way Biot consolidation or spatial permeability inference.
- Preserves Larry transient's already-verified analytical receiver depths:
  COMSOL global z = +490 m corresponds to analytical depth 30 m. Do not
  convert `stations_analytical_depth.csv` a second time.
- The global fixed FV pressure is **not restricted** to the variable mechanical
  source. Finite mechanical pressure loading only includes pressure in the
  chosen cuboid support. This can omit important deformation during recovery.
- Expanded support comparison checks how analytical predicted strain changes
  when pressure-driven cuboids cover a larger area. It is **not independent
  truth**. It uses identical FV pressure and the same analytical kernel.
- A 1x / 1.5x / 2x support comparison approximately preserves cuboid size by
  increasing the number of tiles proportionally. Do not treat it as a formal
  continuum convergence study; the baseline tile quadrature remains approximate.
- The provisional global Dirichlet boundary and well representation still
  need independent physical benchmark validation when COMSOL data arrive.
- No DREAM, no fitting, no synthetic recovery, no full-eight-parameter inference.

## Installation — Windows CMD from repository root

Start on `feature/full-hydromechanical-spatial-inversion`; Stage 6A, 6B-1,
6B-2 must already be extracted and passing. Extract the archive:

```
tar -xf avant_stage6b3_coupled_forward.zip
```

It adds **four files only**:

```
src/avant_model/model/stage6_coupled_forward.py
scripts/validation/check_stage6_coupled_forward.py
tests/test_stage6_coupled_forward.py
README_STAGE6B3_COUPLED_FORWARD.md
```

Run the new tests and complete regression suite:

```
python -m pytest -q tests\test_stage6_coupled_forward.py
python -m pytest -q
```

There were **348 passing tests** before this add-on. With 18 new tests,
expect **366 passing tests**, if the existing repository is unchanged.

Run ONE integrated benchmark (uses the existing Larry station locations but
**not** his observed COMSOL strain data):

```
python scripts\validation\check_stage6_coupled_forward.py --contract configs\pressure_superposition\stage6\stage6a_contract_001.json --hydraulic-config configs\pressure_superposition\experiments\injection_fd_30gpm.json --stations-csv configs\pressure_superposition\larry_network_001\stations_analytical_depth.csv --splits-x 6 --splits-y 6 --quadrature-order 4 --times-h 0 9.5 19 24 96 --support-factors 1 1.5 2 --output-dir results\validation\stage6b3_coupled_forward_001
```

The runner **refuses to overwrite** the output directory. To rerun, choose
another output path.

Review:

```
type results\validation\stage6b3_coupled_forward_001\stage6b3_coupled_forward_report.json
start results\validation\stage6b3_coupled_forward_001\stage6b3_pressure_and_source.png
start results\validation\stage6b3_coupled_forward_001\stage6b3_support_strain_comparison.png
```

Outputs include a JSON report, compressed arrays (time, strain, tile pressure,
FV pressure and predictions under expanded support), and PNG/PDF figures.

## What counts as Stage 6B acceptance?

1. All 366 tests pass and none of the protected mechanics modules has changed.
2. Initial pressure/strain is zero; pressure at physical gauges is sampled from
   the global FV field, not from compressed mechanical tiles.
3. Changing `gamma_rel` scales the strain but does not change pressure; moving
   geometry changes mechanical mapping but not the fixed FV hydraulic field.
4. The expansion sensitivity is **quantified** and interpreted. If 1x vs 2x
   support creates a significant strain change, **do not interpret the finite
   source dimensions as a uniquely identifiable physical source boundary**.
   This may require a better mechanical-loading-support definition before
   attempting source-geometry inversion. No arbitrary pass threshold is hidden.
5. After evaluating the result, commit Stage 6B-3 and proceed to Stage 6C
   multi-parameter identifiability. Do not launch eight-dimensional DREAM
   merely because this forward code runs.

## Provenance and prior results

Stage 6B-2 mapped the existing 81x81 FV field to 6x6 rotated tiles with an
order-4 versus order-8 RMS pressure discrepancy of 309.81 Pa (relative
0.001006) for the tested times. The global absolute-pressure integral inside
baseline support was about 57.1% at 19 h and 12.1% at 96 h. These are **not
strain-error percentages**. This script measures actual predictions from the
unchanged analytical kernel under expanded support; physical validity still
requires independent COMSOL validation.

Standalone local validation of this ZIP used a **test-double kernel** (to
exercise integration, parameter transformations, flow and outputs). It passed
18 tests and the file-output smoke. The original Larry mechanics are verified
only by running these tests and the benchmark in the user's AVANT repository.
