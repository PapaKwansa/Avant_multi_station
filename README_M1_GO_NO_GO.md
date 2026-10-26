# AVANT M1 — one bounded identifiability and numerical-accuracy gate

**Purpose:** Determine whether the existing two-zone hydraulic-compartment model is ready for a *restricted* synthetic MAP inversion, not whether a field boundary has been recovered.

## Installation and Windows CMD execution

From `feature/full-hydromechanical-spatial-inversion` with the six M1 pilot files already installed (380 passing tests):

```bat
python -m pytest -q tests\test_m1_identifiability_gate.py
python -m pytest -q
python scripts\validation\run_m1_go_no_go.py --config configs\pressure_superposition\stage6\stage6_m1_pilot_001.json --output-dir results\validation\stage6_m1_gate_001

type results\validation\stage6_m1_gate_001\m1_go_no_go_report.json
start results\validation\stage6_m1_gate_001\m1_information.png
```

The runner refuses to overwrite its output. Use a new directory for a subsequent run. It does **not** use Larry's COMSOL observed pressure/strain as calibration data.

## What is tested

- **Local parameter sensitivity:** finite-difference, error-standardized sensitivity of 32 strain channels at 8 existing stations and *two hypothetical off-well pressure monitors* across 17 observation times, for the 9 parameters `log10(K_in), log10(K_out), log10(Ss), log10(gamma_rel), a, b, theta, x0', y0'`. The well-located synthetic pressure monitor is excluded because a point well does not have converged physical wellbore pressure.
- **Numerical derivative stability:** compare geometry parameter finite differences at one step and half-step. This detects discontinuities/noisy derivatives from geometric interface discretization.
- **No-contrast degeneracy:** when `K_in == K_out`, the five boundary geometry derivatives should be zero to numerical precision. This prevents false identification from a purely drawn rectangle.
- **One FV spatial resolution screen:** 81 x 81 versus ~122 x 122 on the fixed global domain, using the exact same off-well gauge positions, times, geometry and *fixed mechanical support*. Compare outputs, not pixel values.
- **One mechanical loading support screen:** extend the support half-dimensions by +150 m and use approximately the same tile sizes. This checks whether the selected finite pressure-integration region still materially affects strain.

## Decision criteria and limitations

Predeclared **engineering screening** budgets: strain RMS change should be <= 1 ns (one half of a hypothetical 2 ns measurement error) and off-well pressure RMS change should be <= 10 kPa (one half of a hypothetical 20 kPa error). Geometry Jacobian relative half-step change should be < 0.1 per column. Whitened/scaled Jacobian effective rank uses a relative SVD cutoff of 1e-3. **All three are conditional screening rules, not a proof of physical accuracy or global identifiability.** If any fail, `pilot_decision` is `NOT READY FOR FULL M1 DREAM`; do not simply loosen the thresholds until they pass. Examine whether numerical errors can be controlled cheaply, or restrict the inferred parameter subset and finish with the simpler AVANT model.

For a quick executable check that does not spend the time on spatial/support comparisons, optional `--skip-accuracy` is available, but **its output cannot qualify as a GO**.

**Critical scientific caveats:**

1. Single local rank at a chosen conductivity contrast, sensor layout, parameter reference and prior-like scaling is *not* global uniqueness and not an inversion. If `K_in` approaches `K_out`, compartment geometry becomes invisible.
2. The current FV solver uses *arithmetic volume/area mixing* of conductivity in intersected cells and harmonic adjacent-face fluxes. This is conservative but not an exact sharp oblique-interface solver. A single grid pair is a screen, not full mesh convergence.
3. Mechanical strain is from the existing cuboid superposition in a mechanically homogeneous half-space. The fixed integration region can truncate deformation; its extent must be checked. No assertion of heterogeneous-elastic or fully two-way poroelastic behavior.
4. Independent normal noise with 2 ns / 20 kPa is an *assumption* for this local information experiment, not a measured M1 error covariance. Previous 100-replicate tests showed covariance misspecification can under-cover.
5. Off-well gauges here are **synthetic hypothetical locations** from the M1 pilot config, not the measured pressure histories of Larry's COMSOL run or field gauges.
6. The numerical boundary is an artificial zero-head-change Dirichlet condition. Its position and influence are not validated against COMSOL in this experiment.
7. Any passing result merely authorizes a **bounded synthetic MAP attempt on the identifiable subset**. Independent COMSOL comparison and further coverage checks are required before real-world claims.
8. No modification to Larry's original mechanics equations or previously committed M1 modules is needed.

This package adds only a pure diagnostic module, a runner, tests, and this README. Its output reports enough numerical detail to make a deliberate *GO / restricted GO / NO GO* decision before expensive multi-parameter DREAM.
