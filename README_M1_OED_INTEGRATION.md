# AVANT M1: add-only nine-parameter pressure/strain OED integration

**Scope:** Local and explicitly scenario-robust design analysis using the **existing** M1 flow/strain forward solver and **existing** AVANT OED Fisher implementation. No old analytical, hydraulic, Bayesian, or OED module is changed. This is **not** a converged-posterior Bayesian OED and **not** COMSOL/field validation.

## New files

- `src/avant_model/oed/m1_joint_design.py`: stable time-major pressure/strain observation row catalog; nine-parameter central-difference Jacobian; scaled/whitened Fisher-information-based scores; greedy station/gauge augmentation.
- `scripts/oed/run_m1_oed.py`: immutable MAP provenance check, one candidate-network forward model, shared Jacobian, design scores, optional derivative check, optional explicitly supplied parameter scenarios, saved CSV/NPZ/JSON/figures.
- `configs/pressure_superposition/stage6/m1_oed_001.json`: illustrative synthetic station/gauge design and noise assumptions, including proposed hypothetical S09/P04/P05.
- `tests/test_m1_joint_design.py`: focused ordering, selection, finite difference, Fisher and greedy tests.
- `slurm/avant_m1_oed81.slurm`: compute-node submission example (store in `$HOME/avant_m1_runs/` before submitting).

## Physical parameters (canonical order)

`log10_K_inside, log10_K_outside, log10_Ss, log10_gamma_rel, a_m, b_m, theta_deg, x0_prime_m, y0_prime_m`

The M1 geometry affects **hydraulic K only**. The existing mechanical support is fixed and uses the unchanged pressure-superposition analytical kernel.

## Design score

Let `J_d` be rows selected from the full candidate-network Jacobian, `R_d=diag(sigma_i^2)` the **assumed** independent observation covariance, and `S=diag(parameter_scales)`.

`F_d = (J_d S)^T R_d^{-1} (J_d S)`.

The primary comparable design score is **log det(I + F_d)** for `prior_precision=1` (or log det(P+F)-log det(P) for a declared dimensionless `P=λI`). This is a *regularized local information score*, **not** a calibrated Bayesian posterior. We separately report the rank and condition of the *unregularized weighted/scaled Jacobian*, so rank-deficient pressure-only and strain-only designs cannot be mistaken for full identification.

Old `subset_jacobian_by_stations()` assumes only component-major strain rows. **Do not call it on the M1 time-major joint Jacobian**; use the named observation catalog instead.

## Install on an AVANT checkout

Work on a **new branch** from the verified `feature/full-hydromechanical-spatial-inversion` commit `af72327`. Extract the supplied ZIP at the repo root, preserving its subdirectories. ZIP contains new files only; existing files remain unchanged. Commit after verifying with the actual repository dependencies.

```bash
cd "$HOME/Avant_multi_station"
git switch feature/full-hydromechanical-spatial-inversion
git switch -c feature/m1-nine-parameter-oed
# Transfer the ZIP onto Palmetto (e.g. with scp) before extracting.
unzip -n /path/to/avant_m1_oed_integration.zip
# Run tests through a Palmetto compute allocation, not here on login.
```

**Cluster rule:** Python tests and all model/OED calculations must be run on a **compute node**, not on the login node. A SLURM batch script may invoke them. Do not `source` the activation script on login for computational execution.

The optional batch script runs the focused tests and then the M1 OED computation on a compute node. It should be placed in `$HOME/avant_m1_runs/` (so its relative `logs/` path resolves when submitted from that directory):

```bash
cp slurm/avant_m1_oed81.slurm "$HOME/avant_m1_runs/"
cd "$HOME/avant_m1_runs"
mkdir -p logs
sbatch avant_m1_oed81.slurm
```

The sample assumes the immutable synthetic MAP run `results/inversion/m1_map_palmetto_81_18290828` still exists. Change only `--map-dir` to another verified `--mode full` MAP folder if necessary. The OED runner verifies the saved report, observations, and original M1 configuration hashes.

For a first low-cost on-compute-node input check, run the script with `--dry-run` and a new output directory. `--dry-run` builds the candidate kernel and validates selection metadata, but does not solve the hydraulic forward problem or write results.

## Example outputs

`results/oed/m1_local_JOBID/`:

- `m1_oed_report.json`: per-design effective rank, information scores, assumptions, augmented sensor choices, provenance and limitations;
- `m1_oed_jacobian.npz`: unwhitened full-network Jacobian with explicit row labels, time indices, noise scale;
- `m1_oed_design_scores.csv`: comparison table;
- `m1_oed_design_comparison.pdf` / `.png`: comparison chart.

## Assumptions and cautions

- The saved MAP reference comes from **same-model synthetic** data, not independent geology.
- All added gauges and strain stations are **hypothetical**; the existing P01-P03 were also invented for the MAP pilot.
- Published field studies may have far fewer available strain components than the synthetic network.
- The independent-error assumption is for initial design comparison only. Real temporal correlations require a generalized whitening/covariance extension.
- Greedy augmentation optimizes information gain per added measurement from a finite candidate list; it is not global sensor-placement optimality.
- Robust scenarios, when supplied, are independently specified **parameter vectors**, NOT draws from the nonconverged DREAM posterior.
- Finite-difference derivatives for a moving conductivity boundary should be checked at multiple steps and hydraulic grid resolutions.
- Neither Bayesian expected-information gain nor posterior-averaged OED should be reported as validated until posterior convergence and scientific priors are established.
- No new COMSOL/field observation adapter is included; that remains a separate independent-validation task.

## Verification status of supplied package

The pure numerical and observation-selection tests were run locally using a **temporary Fisher API-compatible test shim** (the shim is excluded from the ZIP). **Do not claim the complete AVANT test suite or an 81-grid OED computation has passed until tests run against the actual repository on Palmetto.** No GitHub branch or commit is created automatically by this ZIP.
