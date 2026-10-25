# AVANT Stage 5: local identifiability diagnostic (add-only)

This package adds a local, noise-whitened sensitivity/Fisher-information audit
for the **already completed Stage 4A same-model synthetic joint inversion**.
It does not modify existing FV, mechanics, pressure-gauge, DREAM, or data files.

Copy the `src/`, `scripts/`, and `tests/` folders into the repository root.
Use the `(vrugt)` environment and start from the repository root.

```bat
python -m pytest -q tests\test_local_identifiability.py
python -m pytest -q
python scripts\validation\diagnose_joint_identifiability.py --stage4a-dir results\pressure_superposition\joint_pressure_strain_stage4a_001 --hydraulic-config configs\pressure_superposition\experiments\injection_fd_30gpm.json --mechanical-config configs\pressure_superposition\larry_network_001\configuration.json --gauges-csv configs\pressure_superposition\experiments\stage4_gauges_local.csv --output-dir results\validation\stage5_identifiability_001
```

If your Stage 4A saved directory has a different name, replace the
`--stage4a-dir` value with the directory containing `joint_summary.json` and
`synthetic_pressure_observations.npz`.

Outputs:

- `identifiability_summary.json`: ranks, sensitivity column norms and cosines,
  Fisher matrices, SVD and local covariance/correlation (if full rank).
- `jacobians_training.npz`: whitened, **training-only** strain, pressure, and
  concatenated joint Jacobians with respect to `(log10 K, log10 Ss)`.
- `local_identifiability_singular_values.png/.pdf`.
- `local_identifiability_uncertainty.png/.pdf`.

No new noise is generated; Stage 4A's hash-validated observations and joint MLE
are reused unchanged. All station/gauge holdout observations remain excluded.
Compare the local standard deviations with the sampled Stage 4B posterior
*qualitatively*; they are not required to be equal.

## Interpretation limits

1. These are LOCAL first-derivative diagnostics for TWO hydraulic log parameters
   with fixed geometry, elasticity, boundaries and independently specified noise.
2. A full-rank local Jacobian is neither proof of global uniqueness nor proof
   of field identifiability.
3. This is an internal same-model synthetic audit; it cannot replace matched
   independent COMSOL spatial-pressure/strain comparisons.
4. A single finite-difference step test does not bound truncation or modeling
   error. Inspect `jacobian_relative_change_half_step` and repeat at other
   reference vectors when necessary.
5. If local covariance is absent for a rank-deficient block, do not report
   parameter error bars from that block.
6. Do not interpret the numbers as identifiability of `E, nu, alpha, a, b,
   theta, x0, y0`, which are fixed here.
