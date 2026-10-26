# AVANT M1: Publication-ready posterior figures and independent global sensitivity

**Scope:** Add-only postprocessing and optional independent-input Sobol sensitivity for the **nine-parameter M1 two-zone hydraulic / effective elastic-coupling / geological-rectangle model**. Never alter Larry's analytical strain kernel, validated coordinate mapping, saved MAP observations, Bayes priors, DREAM chains, or existing scripts. This package contains **5 new files only**.

The two scripts are deliberately separate. **DREAM samples cannot legitimately be used directly to estimate ordinary Sobol first/total-order indices**, because the posterior is typically correlated. Sobol indices describe the variance of specific outputs under an *explicit independent input distribution*. That is a different experiment requiring new model runs. Synthetic generating truths and invented gauge sites are used only for benchmark scoring; not field knowledge.

## 0. Status and constraints

- Tested: **16 numerical unit tests** and end-to-end plotting/independent Sobol interface smoke tests using a local *stand-in analytical kernel*.
- **Not tested:** complete production `multi_station_strain.py` in your Windows `(vrugt)` environment. Your run is the integration test. Never use the fixture's sensitivity values scientifically.
- Does not modify a full M1 DREAM run that is already executing. Run this postprocessing only **after sampling finishes**, chain-file inspection, and the official `postprocess_m1_bayesian_dream.py` check.
- Does **not** improve sampler convergence, change the physical model, fit parameters, or provide independent COMSOL/field validation.
- By default, reads only saved full-run chains and saved MAP observations. Forward predictions cost `--posterior-draws` fresh M1 flow solves. Geometry probability uses *all* retained draws by default without costly flow solves.
- Exports **editable vector PDF** and **300-DPI PNG** by default; use `--dpi 600` for a particular journal's raster requirement. Typography and figure aspect are more important than DPI alone.
- A nonconverged run is labeled `EXPLORATORY: CHAINS NOT CONVERGED` on each figure and reported as nonconverged in JSON.

## 1. Install and regression tests (Windows CMD in repo root)

```bat
rem Ensure the current full DREAM run has finished before loading its output.
tar -xf avant_m1_publication_sensitivity.zip
python -m pytest -q tests\test_m1_publication_analysis.py
python -m pytest -q
```

Expected new tests: **16 passed**. Assuming the prior 442-test baseline is unchanged: **458 passed**. Python packages needed: the existing numpy, scipy, matplotlib, plus the installed AVANT package. No SALib dependency is required.

## 2. First finish the ordinary M1 posterior diagnostics

Use existing commands from the Bayesian package. Verify that your `m1_bayes_full_001.log` has no `Traceback`, `ImportError`, or worker failure. Verify saved chain shapes (4,1600,9) and finite log posterior. Then run:

```bat
python scripts\postprocessing\postprocess_m1_bayesian_dream.py --run-dir results\inversion\m1_bayes_full_001 --posterior-draws 40 --noise-replicates 16

type results\inversion\m1_bayes_full_001\m1_posterior_summary.json
```

**Do not claim convergence just because the process exits.** Inspect all nine rank-folded split R-hats and bulk/tail ESS. If `diagnostic_pass` is false, the new publication suite generates only labeled exploratory diagnostics. Do not use intervals as calibrated posteriors.

## 3. Publication figure suite (after full run and official diagnostic)

```bat
python scripts\postprocessing\postprocess_m1_publication.py --run-dir results\inversion\m1_bayes_full_001 --posterior-draws 60 --boundary-draws 0 --noise-replicates 6 --dpi 300 --local-report results\validation\stage6_m1_gate_001\m1_go_no_go_report.json
```

If the pre-existing `stage6_m1_gate_001` local report is unavailable, **omit** the `--local-report` option. It uses the previous synthetic/selected-sensor **Jacobian column norms** and is explicitly labeled local sensitivity; these are *not* Sobol indices. Its setup differs from the Bayesian training subset, so use it as a separate supplemental check.

Outputs in `results/inversion/m1_bayes_full_001/publication/`:

| File | Purpose |
|---|---|
| `m1_pub_traces.png/pdf` | All nine parameter chain-mixing traces |
| `m1_pub_marginals.png/pdf` | Prior versus posterior sample density, medians, 95% percentiles, synthetic truth |
| `m1_pub_correlations.png/pdf` | Annotated posterior *correlation* matrix; not sensitivity |
| `m1_pub_zone_probability.png/pdf` | Geological membership probability using all retained geometry draws, with reference boundary and probability contours |
| `m1_pub_station_S01.png/pdf` ... `S08` | Four strain components at each station; training versus held-out; 95% latent and noisy predictive bands |
| `m1_pub_pressure_gauges.png/pdf` | Both training pressure gauges and the independently held-out invented pressure gauge |
| `m1_pub_local_sensitivity.png/pdf` | Optional diagnostic using the *prior* local numerical Jacobian report |
| `m1_pub_predictive_metrics.csv` | Training/all and held-out RMSE, pointwise 95% latent/noisy coverage |
| `m1_pub_report.json` | Diagnosed status, sample counts, posterior summaries, normalization and limitations |

To repeat with a different draw count **choose a fresh output directory** to preserve your first report:

```bat
python scripts\postprocessing\postprocess_m1_publication.py --run-dir results\inversion\m1_bayes_full_001 --output-dir results\inversion\m1_bayes_full_001\publication_v2 --posterior-draws 120 --noise-replicates 8 --dpi 600
```

This outputs PNG/PDF; 600-DPI multi-panel raster files may use significant local memory. Most journals prefer the vector PDFs.

## 4. Optional Sobol forward sensitivity (NOT part of MCMC postprocessing)

A methodologically valid Sobol run requires an **independent input design**. You must declare what uncertainty distribution you mean:

- `--measure illustrative-priors` (default) draws independent values from the existing *illustrative* uniform/truncated-normal Bayesian input priors. It does **not** use the **correlated posterior**.
- `--measure uniform-bounds` explores the validated nine-parameter bounds uniformly, independent by construction. Both measures may generate quite different rankings.

Sobol outputs are **four fixed, physically interpretable scalar responses at 19 h**: S07 eXX strain, held-out off-well gauge pressure, training-network strain RMS, and mean training pressure-gauge pressure. Effects are output-, time-, and prior-range-specific.

First do the lightweight preflight (no flow solve):

```bat
python scripts\postprocessing\run_m1_sobol.py --run-dir results\inversion\m1_bayes_full_001 --output-dir results\analysis\m1_sobol_preflight_001 --base-n 32 --dry-run
```

Then optional **exploratory pilot** (32 base pairs x 11 = **352 M1 flow simulations**):

```bat
python scripts\postprocessing\run_m1_sobol.py --run-dir results\inversion\m1_bayes_full_001 --output-dir results\analysis\m1_sobol_pilot_001 --base-n 32 --bootstrap 300 --measure illustrative-priors
```

Do **NOT** run this at the same time as the full Bayesian inversion on a RAM-constrained Windows host.

If the paper truly needs global rankings and the 32-point pilot indicates useful indices, use an HPC allocation and run at **N=256** (2,816 forward evaluations), then, if practical, **N=512** (5,632 evaluations). Compare first/total estimates, their bootstrap 95% intervals and N/2-versus-N differences; **no index is called converged just because the job finishes**.

```bat
python scripts\postprocessing\run_m1_sobol.py --run-dir results\inversion\m1_bayes_full_001 --output-dir results\analysis\m1_sobol_256_001 --base-n 256 --bootstrap 300 --measure illustrative-priors
```

Outputs: `m1_sobol_indices.csv`, `m1_sobol_report.json`, `m1_sobol_first_total_indices.pdf/png`, `m1_sobol_n_doubling.pdf/png`. First order uses a Saltelli-type estimator, total order Jansen; a **paired base-sample bootstrap** estimates interval uncertainty. Raw negative or >1 finite-sample estimates are *not* silently clipped. Bootstrap on scrambled QMC is approximate, so its intervals and the N-doubling check are *screens*, not proof of statistical convergence. `exploratory_or_unstable_screen` marks N<256, high half/full discrepancy, or wide intervals. The Sobol output measure is defined by the *prior*, and its indices are **not posterior parameter identifiability or parameter-estimation uncertainties**.

## 5. What belongs in the paper?

**Main figures** if DREAM passes convergence: (1) synthetic design / geology and gauge positions; (2) eight/nine parameter priors versus posterior marginals and 95% intervals; (3) correlation and identifiability; (4) posterior zone membership; (5) joint held-out pressure–strain posterior predictive checks with separate latent and measurement bands; (6) model/parameter recovery and numerical discrepancy. Supplement: all-station components, complete traces/R-hats/ESS, full forward FVM numerical checks, conditional-prior sensitivity, and **optional** globally converged Sobol rankings. All figures must distinguish invented synthetic measurements, assumed Gaussian noise, and the unresolved field validation.

**Important:** Prior-versus-posterior overlay is *not* a proper **prior robustness study**. That requires refitting under justified alternative geological priors, or a validated importance-sampling reweighting study with adequate reweighting ESS. Such a study is worth conducting if your M1 conclusions prove sensitive to the illustrative priors; do not manufacture it from the original posterior alone.

## 6. Version control

After Windows tests and a production postprocessing run, stage ONLY the four new files:

```bat
git add README_M1_PUBLICATION_AND_SOBOL.md
git add scripts/postprocessing/m1_publication_core.py
git add scripts/postprocessing/postprocess_m1_publication.py
git add scripts/postprocessing/run_m1_sobol.py
git add tests/test_m1_publication_analysis.py
```

Exactly five add-only files are listed above, including this README and the tests. Verify with `git diff --cached --stat`; do not `git add .` because historic ZIPs and patches are intentionally untracked.

## Scientific references

- Gelman et al. (2020/2021), *Bayesian Workflow* and *Bayesian statistics and modelling*: prior, likelihood, computation, predictive checks and uncertainty must all be checked, not just plotting a posterior. https://sites.stat.columbia.edu/gelman/research/unpublished/Bayesian_Workflow_article.pdf
- Bayesian Analysis Reporting Guidelines, *Nature Human Behaviour* (2021): justify priors, conduct predictive checks and consider posterior robustness to priors. https://doi.org/10.1038/s41562-021-01177-7
- Saxton et al., *PLOS Computational Biology* 20 (2024), e1011946: sensitivity rankings require sample-size/estimator-convergence analysis. https://doi.org/10.1371/journal.pcbi.1011946

This package is a postprocessing and optional scientific-design add-on to the existing AVANT M1 workflow, **not a new geology/flow/strain physics implementation**.
