# Stage 5D: conditional observation-error sensitivity

This add-only package extends the committed Stage 5 identifiability diagnostic.
**It does not modify or re-run the finite-volume solver, analytical cuboid kernel,
DREAM sampling, observations, or prior Stage 5 source code.**

## Purpose

Investigate how local two-parameter Fisher information for `log10 K` and
`log10 Ss` changes when the ASSUMED strain and pressure observation errors
have different standard deviations and independent, within-channel temporal
correlation. All observations and the parameter reference remain unchanged.

This is a **prospective error-model sensitivity calculation**, not a posterior,
not repeated-noise parameter recovery, not an independent numerical benchmark,
and not a validation of actual correlated measurement errors.

Time correlation is an exponential model using actual observation times:

`Corr(error_i(t),error_i(u)) = exp(-abs(t-u)/tau_hours)`.

Errors are assumed independent between different strain channels, different
pressure gauges, and between pressure and strain. `tau_hours=0` means temporal
independence. Whitening is done with a Cholesky factor in **time-major**
observation ordering.

## Install and run (Windows CMD)

Extract the ZIP in the AVANT repository root on branch
`feature/independent-physical-validation`. Run from that root with `(vrugt)`:

```bat
python -m pytest -q tests\test_error_model_sensitivity.py
python -m pytest -q
python scripts\validation\diagnose_error_model_sensitivity.py --identifiability-dir results\validation\stage5_identifiability_001 --stage4a-dir results\pressure_superposition\joint_pressure_strain_stage4a_001 --output-dir results\validation\stage5d_error_model_001
```

All Stage 4A input hashes and the existing Stage 5 Jacobian Fisher matrices
are checked before calculations. The existing Stage 5 Jacobians are already
noise-weighted using the original (2 ns, 20 kPa) independent error model.

This script does **not** rerun the likelihood or re-estimate an MLE. It applies
new Gaussian error covariance assumptions to the same local Jacobian.

## Files

- `src/avant_model/inversion/error_model_sensitivity.py`: reusable covariance
  and Jacobian-whitening calculations.
- `scripts/validation/diagnose_error_model_sensitivity.py`: audited experiment.
- `tests/test_error_model_sensitivity.py`: numerical and invalid-input tests.

## Outputs

- `error_model_sensitivity_summary.json`: complete conditional information
  diagnostics and provenance for every scenario.
- `error_model_sensitivity_cases.csv`: per-parameter local SD and joint/strain
  gain under each assumed error model.
- `noise_scale_information_gain.png/.pdf`: 3x3 grid of relative assumed noise
  scales, using no time correlation.
- `temporal_correlation_uncertainty.png/.pdf`: scenarios with 6 h or 24 h
  assumed within-channel time-correlation length.

## Interpretation limits

- Reweighting a fixed Jacobian **cannot** test biased estimates or empirical
  credible-interval coverage under correlated or misspecified errors.
- A local Fisher precision estimate is conditional on fixed geometry,
  mechanics, hydraulic boundaries, parameter values, and the error model.
- The Stage 4B DREAM posterior uses a different actual likelihood when
  error assumptions change; it is not automatically updated here.
- The next separate Stage 5D step must regenerate observations under a
  different error or physical truth model and actually refit/invert them.
- Independent COMSOL coupled validation remains pending an appropriate
  spatial-pressure reference dataset.
