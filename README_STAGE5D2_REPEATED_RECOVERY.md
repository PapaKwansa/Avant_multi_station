# AVANT Stage 5D-2: repeated synthetic hydraulic recovery

**Scope:** Add-only experimental runner with new independent Gaussian noise realizations,
correlated Gaussian noise realizations, **actual maximum-likelihood refitting**,
and approximate parameter confidence-interval coverage. Reuses the immutable
Stage 4A input artifacts and exact fixed-geometry FV-to-analytical forward model.

This experiment is **same-model synthetic**, not COMSOL verification, field
validation, full Bayesian posterior coverage, or hydraulic–mechanical–spatial
inversion. Geometry and mechanics remain fixed; only `log10 K` and `log10 Ss`
are estimated. S07/S08 and P03 remain held out of all optimization fits.

## Package contents

- `src/avant_model/inversion/repeated_recovery.py` – correlated Gaussian draws,
  time-domain Cholesky whitening, bounded nonlinear GLS fitting, and local
  known-error Wald covariance/interval calculations.
- `scripts/validation/run_repeated_hydraulic_recovery.py` – saved-data provenance,
  repeated paired experiments, reports, and PNG/PDF figures.
- `tests/test_repeated_recovery.py` – lightweight numerical tests for noise,
  whitening, local covariance, and refitting.
- This README.

The `error_model_sensitivity.py` module committed in Stage 5D is a dependency
and **is not duplicated or modified** in this ZIP.

## Windows CMD instructions

Unzip from your AVANT repository root in the `(vrugt)` environment:

```bat
 tar -xf avant_stage5d2_repeated_recovery.zip
 python -m pytest -q tests\test_repeated_recovery.py
 python -m pytest -q
```

With the previous 247-test baseline, expect **261** tests if no others changed.

Run a small **functionality and smoke test** first; do NOT interpret 8 repeats
as reliable empirical coverage:

```bat
python scripts\validation\run_repeated_hydraulic_recovery.py --stage4a-dir results\pressure_superposition\joint_pressure_strain_stage4a_001 --hydraulic-config configs\pressure_superposition\experiments\injection_fd_30gpm.json --mechanical-config configs\pressure_superposition\larry_network_001\configuration.json --gauges-csv configs\pressure_superposition\experiments\stage4_gauges_local.csv --replicates 8 --output-dir results\validation\stage5d2_recovery_smoke_001
```

Inspect:

```bat
type results\validation\stage5d2_recovery_smoke_001\recovery_summary.csv
start results\validation\stage5d2_recovery_smoke_001\recovery_rmse.png
start results\validation\stage5d2_recovery_smoke_001\wald_coverage.png
```

For a more meaningful Monte Carlo empirical coverage experiment, use at least
100 replicates (a separate NEW directory), after reviewing the smoke output:

```bat
python scripts\validation\run_repeated_hydraulic_recovery.py --stage4a-dir results\pressure_superposition\joint_pressure_strain_stage4a_001 --hydraulic-config configs\pressure_superposition\experiments\injection_fd_30gpm.json --mechanical-config configs\pressure_superposition\larry_network_001\configuration.json --gauges-csv configs\pressure_superposition\experiments\stage4_gauges_local.csv --replicates 100 --output-dir results\validation\stage5d2_recovery_100_001
```

The full run makes nine fits per replicate: three data-generation/error-assumption
cases x three observation modes. Each fit performs repeated FV solves, so the
100-replicate experiment is substantially more computationally expensive than
Stage 5D's Jacobian reweighting. Reduce `--replicates` during smoke testing.

## Scenarios

| Case | Generating errors | Error model fitted |
| --- | --- | --- |
| `iid_correct` | Independent normal noise in time/channels | Independent Gaussian errors |
| `corr_correct` | Exponential temporal correlation (`--correlation-hours`, default 6 h) | The same known correlation |
| `corr_ignored` | **The identical correlated observations** as `corr_correct` | Incorrect independent Gaussian errors |

The pressure and strain noise standard deviations are inherited from the
verified Stage 4A observations (2 ns and 20 kPa in the existing experiment).
The modeled correlation is independent **between channels and modalities** and
acts only along time within each station/gauge channel.

## Outputs

- `repeated_recovery_manifest.json` – seeds, source SHA-256 hashes, modeling
  scope, known error assumptions, parameter truth, and limitations.
- `synthetic_replicates.npz` – all newly generated synthetic observations,
  the generating truth, times, and training masks for paired-data audit.
- `replicate_fits.csv` – individual estimates, errors, conditional local Wald
  standard deviations, interval inclusion, holdout RMSE, failure reasons.
- `recovery_summary.csv` – empirical bias, RMSE, interval coverage and Wilson
  binomial 95% bounds, for each parameter/method/scenario.
- `recovery_rmse.png/.pdf` – empirical parameter error versus error assumption.
- `wald_coverage.png/.pdf` – empirical coverage with Wilson uncertainty bars.

## Scientific interpretation

- The truth and inversion use the **same** homogeneous FV and analytical model.
  This tests observation-noise robustness only, **not structural model error**.
- Reported intervals are local Gaussian/Wald confidence intervals, calculated
  from weighted residual Jacobians at each nonlinear optimum with **known**
  assumed observation standard deviations; they are not DREAM credible intervals.
- Correctly specified covariance does not guarantee exactly 95% empirical
  coverage in finite nonlinear samples. A handful of replicates cannot assess
  coverage precisely. Inspect Wilson bounds.
- Failed optimizations are retained and counted, not silently discarded.
- Numerical convergence, posterior multimodality, geometry/mechanics
  identifiability, cross-channel covariance, and field readiness are **not**
  established by these experiments.
- The next separate independent experiment should use a matched COMSOL or
  other high-fidelity simulation to generate data, so the inversion cannot
  assume its own model is the truth.
