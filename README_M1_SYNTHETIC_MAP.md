# AVANT M1 — one bounded synthetic MAP recovery experiment

**Purpose:** Complete the *next* research decision. Test whether the two-zone
hydraulic, fixed-support analytical-mechanics M1 model can recover a geological
compartment and hydraulic/mechanical parameters from its own noisy simulated
pressure and strain observations. This package **does not** change Larry's
mechanical kernel or previously committed FV code. **Not** COMSOL/field validation.

## Files added (four only)

```
src/avant_model/inversion/m1_map_recovery.py
scripts/inversion/run_m1_synthetic_map.py
tests/test_m1_map_recovery.py
README_M1_SYNTHETIC_MAP.md
```

Requires the existing M1 forward pilot (6 files) and M1 local-identifiability
gate (4 files), both present in the working AVANT repository. Stage 6A/B files
and original cuboid strain source remain unchanged.

## First preserve your passing M1 gate

On `feature/full-hydromechanical-spatial-inversion` check `git status --short`.
If the M1 gate's three Python files and README remain untracked, stage exactly:

```bat
git add README_M1_GO_NO_GO.md
git add src/avant_model/inversion/m1_identifiability_gate.py
git add scripts/validation/run_m1_go_no_go.py
git add tests/test_m1_identifiability_gate.py
git diff --cached --check
git diff --cached --stat
git commit -m "Add M1 local identifiability and numerical go no go screen"
git push origin feature/full-hydromechanical-spatial-inversion
```

Use `git diff --cached --stat` to confirm *four* expected files. Do not stage ZIPs,
patches, or exploratory COMSOL scripts. Previously tested suite: 393 passing.

## Installation and one bounded experiment

Unzip into repository root and run:

```bat
tar -xf avant_m1_synthetic_map_recovery.zip
python -m pytest -q tests\test_m1_map_recovery.py
python -m pytest -q
```

Expected **415 total tests** if the previous 393-test baseline is unchanged
(22 new unit tests). Existing matplotlib/Pillow warning and certain legacy
cuboid `arctan` intermediate divide-by-zero warnings may persist; they must
not be silently suppressed. Final forward outputs must remain finite.

Cheap integration smoke (uses actual AVANT cuboid kernel, unlike local test
fixture; no valid physical gate at this coarse grid):

```bat
python scripts\inversion\run_m1_synthetic_map.py --config configs\pressure_superposition\stage6\stage6_m1_pilot_001.json --grid-n 41 --mode constrained --starts 1 --max-nfev 20 --output-dir results\inversion\m1_map_smoke_001
```

If smoke works, perform the **single bounded 81x81 recovery benchmark**:

```bat
python scripts\inversion\run_m1_synthetic_map.py --config configs\pressure_superposition\stage6\stage6_m1_pilot_001.json --grid-n 81 --mode both --starts 2 --max-nfev 80 --output-dir results\inversion\m1_map_full_001
```

The script is sequential and does not overwrite directories. A single fit may
perform hundreds of full forward solves and can be expensive; start with the
41x41 smoke. If a mode fails to converge, the output records **NO successful
fit**, not a fictitious success. A nonconverged numerical attempt does not prove
physical non-identifiability; inspect its diagnostics before deciding whether
one carefully justified rerun is warranted.

Read:

```bat
type results\inversion\m1_map_full_001\m1_map_recovery_report.json
start results\inversion\m1_map_full_001\m1_map_parameter_errors.png
start results\inversion\m1_map_full_001\m1_map_heldout_predictions.png
start results\inversion\m1_map_full_001\m1_map_k_fields.png
```

## Exact experimental design

- The nine-dimensional **generating truth** comes from the M1 pilot configuration:
  `log10 K_in=-6`, `log10 K_out=log10(3e-7)`, `log10 Ss=-6`,
  `log10 gamma_rel=0`, `a=215 m`, `b=400 m`, `theta=15 degrees`,
  `x0'=53 m`, `y0'=141 m`. This is an illustrative synthetic truth, not a field result.
- The same unchanged analytical half-space strain kernel, same FV grid, same
  numerical pressure mapping and same fixed mechanical support generate and
  fit the observations: an **inverse crime** (same-model synthetic benchmark).
- Time samples: 17 from 2 to 96 hours. Strain training stations S01-S06;
  held out S07-S08. All four strain components used.
- Pressure training gauges: (-185,25) and (300,150) m in global primed
  coordinates (invented earlier). **Additional held-out gauge:** (0,-350) m
  (also invented). No wellbore pressure or measured COMSOL data used.
- Noise: independent synthetic Gaussian with 2 nanostrain and 20 kPa SD;
  random seed recorded, observations saved in NPZ. Real correlated errors
  can produce undercoverage, as Stage 5D demonstrated.
- Both modes fit **exactly the same noisy observations**, in normalized bounded
  coordinates. Bounded uniform parameter prior + independent Gaussian
  likelihood -> bounded MAP = MLE within the bounds.
- **Constrained:** six free parameters; K_out, b, and y0' fixed to the true
  synthetic values. This is an explicit *oracle* test, optimistic relative to
  actual geological information.
- **Full:** nine free parameters and two independent starts by default. Their
  convergence and boundary estimates are **not guaranteed**. The strong
  correlations in the prior local Jacobian make poor recovery plausible.
- Search bounds are purpose-built for the synthetic reference and guarantee
  canonical source dimensions and a nonzero K contrast. They are **not**
  defensible field priors and restrict the possible source geometry.

The report includes fit success/failure for *each* start, the chi-squared
training objective, normalized-bound hits, all nine fitted and generating
values, held-out strain and pressure RMSE, and file SHA256 provenance.

## Scientific decision

- **If the constrained fit recovers a useful subset but the full model trades
  parameters:** Make only the subset a supported inversion feature. Preserve
  the full M1 model as an optional forward simulator.
- **If both fits are stable and show held-out agreement:** One carefully
  bounded, converged Bayesian run could be warranted, after checking that
  numerical discretization error is sufficiently small across the fitted region.
- **If neither can recover the key geological properties:** Stop expanding
  M1 boundary inversion and finalize the stronger established AVANT framework.

Never treat a same-model recovered field as proof that a real geological
boundary or its uncertainty has been determined. A physically realistic
held-out COMSOL or field benchmark is still required for that claim.

## Physical scope

The method remains a 2D confined one-way hydraulic solver plus homogeneous,
linear-elastic half-space analytical mechanics. It cannot exactly model elastic
heterogeneity, 3D hydraulic effects, fault slip, poroelastic two-way feedback,
or an unknown external reservoir boundary. The K mixed-cell approximation,
outer Dirichlet boundaries, finite mechanical pressure support, and point-well
approximation remain potential sources of model discrepancy. E and alpha are
not separately identifiable in the current mechanics; gamma_rel represents
only their effective ratio.
