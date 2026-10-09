# AVANT pressure superposition: first testing implementation

## Decision and scope

This is an additive extension, not a replacement of the analytical solution.
It uses the existing `multi_station_strain`, `multi_station_coord_transform`,
`multi_station_rotation`, and `forward_model_multi_station` modules. No original
module, input file, PyDREAM runner, postprocessor, or dataset is included or changed.
No latent diffusion, neural surrogate, hydraulic solver, or permeability inversion
is included. These tests are small and should be run locally before any HPC job.

The implemented forward model is `d = G @ p`. Here pressure changes are in Pa and
G is in nanostrain/Pa. An optional block basis gives `p = B @ coefficients`, with
many mechanical tiles but fewer pressure unknowns. This version uses an explicit
piecewise-constant block basis; a smooth basis can be added after resolution tests.

The first pressure inversion is exactly Gaussian under its stated assumptions:
fixed geometry/mechanics, a proper Gaussian coefficient prior, and a known Gaussian
observation covariance. It calculates posterior means/covariances using whitening
and SVD, rather than running MCMC unnecessarily. It permits signed pressure changes.
It does not impose positivity or automatically choose the prior from the data.

## Contents

- `src/avant_model/model/pressure_superposition.py`: tiling, unit-pressure kernel,
  named predictions, and coarse-to-fine block basis.
- `src/avant_model/inversion/linear_pressure_inversion.py`: Gaussian inversion,
  null-space uncertainty, signal and observation predictive covariances.
- `scripts/validation/run_pressure_superposition_checks.py`: regression tests,
  synthetic inversion, withheld-station predictions, and provenance/output writing.
- `tests/test_pressure_superposition.py`: mechanics-wrapper regression and guards.
- `tests/test_linear_pressure_inversion.py`: statistical reference tests.
- `configs/pressure_superposition_demo.json`: explicitly synthetic experiment.
- This guide and a local testing report.

## Install from the add-only Git patch

Place `avant_pressure_superposition_starter.patch` in your repository root.
Review the status before creating the new branch; preserve all existing work.
Run commands separately and stop if any command reports an error.

```bat
git status
git switch -c feature/pressure-superposition
git apply --stat avant_pressure_superposition_starter.patch
git apply --check avant_pressure_superposition_starter.patch
git apply avant_pressure_superposition_starter.patch
```

The patch adds new files only. It does not create a commit or push to a remote.
The check should fail rather than overwrite an already existing file at a new
path. Do not use force, reset, clean, or reject options to work around a conflict.
An alternative is to copy the ZIP's contents into the repository, but do not
apply both the ZIP and patch. The ZIP is an overlay, not a complete repository.

## Run the tests

Use the existing Python environment. The added tests need NumPy, pandas, and
pytest. The demonstration plots additionally use Matplotlib. No PyTorch,
diffusion package, or new PyDREAM setup is required. Syntax was checked for
Python 3.9; local execution used the versions documented in the test report.
Do not upgrade your working environment merely to run these tests.

```bat
python -m pytest -q tests/test_pressure_superposition.py tests/test_linear_pressure_inversion.py
```

The local scratch-harness result was 48 passed. Repeat this against your actual
repository: a scratch transcription is not verification of your checkout.

Then run the illustrative experiment, from the repository root:

```bat
python scripts/validation/run_pressure_superposition_checks.py --config configs/pressure_superposition_demo.json --output-dir results/pressure_superposition/demo_001
```

The same command works on Linux. Use a new output directory for each run; the
runner intentionally refuses to overwrite one. For no plots add `--no-plots`.
For regression checks alone add `--verification-only`.

This runner always performs regression before inversion. Failed regression writes
`failure.json` and stops. It prints the actual path of the analytical module being
used, and records hashes of all participating modules and the configuration.

## Demonstration design: do not mistake this for field inference

The demonstration has eight INVENTED station locations, 16 mechanical cuboids,
four pressure coefficients, six training stations, and two withheld stations.
The true pressure coefficients and noise are invented. No measured strain CSV is
read. Depth h=520 m and semi-thickness c=3.125 m are explicit in the JSON; other
dimensions/location/orientation are illustrative baseline values, not a newly
calibrated geometry. E=8 GPa, nu=0.35, alpha=0.8 are adopted test values, not newly
verified field estimates.

A single pressure snapshot is inverted. Multi-time predictions are supported by
the kernel and checked against the old ramp/exponential history in regression,
but this runner does not fit a field pressure history or model fluid diffusion.

The pressure Gaussian prior has mean 0.5 MPa and standard deviation 0.5 MPa for
each independent coefficient. Observation noise has known standard deviation
2 nanostrain. These are declared test assumptions, not measured error estimates.

The reported credible intervals condition on the geometry, material properties,
basis, prior, and noise. They omit uncertainty in those assumptions. The code
also distinguishes uncertainty of the predicted mean signal from predictive
uncertainty including new Gaussian measurement error.

## Main outputs

- `verification_summary.json`: partition sizes, tolerances, and numerical errors.
- `synthetic_summary.json`: pressure error, training/holdout misfit, rank, and
  prior-whitened singular values. Singular values greater than one depend on the
  chosen prior and noise scales; they are not an absolute count of resolved cells.
- `synthetic_pressure_posterior.csv`: true, prior, and posterior pressures in Pa.
- `synthetic_channel_predictions.csv`: channel names, split, uncertainty and fit.
- `pressure_kernel_and_posterior.npz`: kernel, basis, posterior mean and full
  coefficient covariance, channel order, masks, and fine-grid summary.
- `tile_geometry.csv`: offsets, semi-dimensions and absolute center depths.
- `station_metadata_used.csv`, `configuration_used.json`, `provenance.json`.
- Five PNGs: coefficients/intervals, fit, singular values, pressure mean and SD.

Do not feed these files into the old seven-parameter geometry postprocessor.

## Scientific and numerical safeguards

1. Every tile uses the existing characteristic-strain convention. This addition
   does not decide between the code and old manuscript's differing prefactors.
   Resolve that normalization using the authoritative derivation/independent
   benchmark before interpreting absolute field pressure. A shared prefactor
   error is invisible to the uniform-partition regression.
2. Tile pressure is not divided by tile count. Finite-cuboid responses must not
   be multiplied by volume again. Equal pressure in every tile reproduces the
   parent pressure loading.
3. `z` stays referenced to the original free surface. A tile's center depth is
   `h_parent + zeta`; the station depth is never shifted a second time.
4. Initial support is receivers outside the parent inclusion. Receivers inside
   or on the source are rejected pending a separate interior/interface audit.
   Sources must be completely below the free surface and station depth z >= 0.
5. The original arctangent/logarithm expressions are not modified. Warnings are
   not suppressed; a non-finite final tensor raises an error identifying the
   station and tile. A partition introduces alignments not covered by old scans.
6. `b > a` is not imposed on each tile. That is a parent-inversion convention,
   not a requirement that every subdivided tile be longer along local y.
7. Kernels have explicit component-major names, by default eXX,eYY,eXY,eZZ.
   Tensor shear is used, not engineering shear. Column-name selection prevents
   silently swapping eXY and eZZ relative to other scripts' listings.
8. A kernel is valid only for the geometry, stations, mechanics and components
   used to build it. Rebuild it when any of them changes.
9. No import of `multi_stations_input` occurs. In particular, the alternate
   unconstrained h near 121.677 m in that input is not silently inherited.
10. Passing tests is not independent physical validation or proof of unique
    pressure recovery. Tests include conservation of source partition volume,
    uniform-loading equivalence, pressure linearity, coordinate consistency,
    free-surface traction conditions, and retention of prior variance in an
    unobserved pressure mode. They cannot establish all source normalization,
    interior-source validity, or applicability to mechanically layered fields.

## Use the actual station geometry next

Copy the JSON to a separately named configuration. Remove its `stations` array
and add, for example:

```json
"station_file": "datasets/comsol/1darcy/metadata/stations.csv"
```

Relative station paths are resolved from the repository root. Change the
`holdout_stations` names to match this CSV (or use an empty array for an initial
all-station synthetic check). Confirm that x,y are in the expected global frame
and z is downward depth, not elevation. Do not silently flip coordinate signs.
Confirm every geometry/material value; the example does not select a best fit.

IMPORTANT: Using actual station positions still creates SYNTHETIC strain in
this runner. It does not turn the experiment into a measured-data inversion.
That deliberate separation lets us measure network recoverability first.
For actual data inversion, a new loader must verify the observed strain units,
reference state, time alignment, instrument components, calibration and error
model. The original default scalar error scale is not automatically an adequate
real-data covariance.

## Next scientific tests

First repeat the synthetic tests using the actual network, several pressure
patterns, multiple independent noise realizations and different noise levels.
Compare one uniform coefficient against four zones and then a smooth basis.
Assess withheld observations, full pressure covariance, prior sensitivity,
geometry/material uncertainty, and sensitivity to source support extent.

Then compare an independently supplied spatial COMSOL pressure field and its
strain response under matched mechanics. Single-monitor pressure time series
are not spatial pressure fields. Test deliberately mismatched mechanics and
boundaries separately so improved fitting cannot hide model discrepancy.

For permeability inference, add a justified hydraulic mass-balance model and
known/uncertain forcing, storage, initial/boundary conditions and, preferably,
independent pressure observations. The mechanical superposition alone contains
no permeability sensitivity. Do not claim otherwise from the pressure posterior.

Latent diffusion is optional research for a later high-dimensional geological
prior, not a dependency of this pressure inversion. Establish a simple Gaussian
or low-dimensional Bayesian baseline before adding a learned prior.

## Git after validation

Review and stage the added source/configuration/test/documentation files by
name. Inspect `git diff --cached --stat` and `git diff --cached` before committing.
Do not stage unrelated previous edits or generated results with `git add .`.
Preserve the tested commit and environment information when later running on
Palmetto. No Slurm submission is needed for this initial test package.

## Context references (not claims of originality)

Vasco, Karasaki and Kishida (2001), A coupled inversion of pressure and surface
displacement, Water Resources Research, doi:10.1029/2001WR000391.

Cornelissen, Meulenbroek and Jansen (2024), On the derivation of closed-form
expressions for displacements, strains, and stresses inside poroelastic
reservoirs, JGR Solid Earth, doi:10.1029/2023JB027733.

Abdollahi et al. (2025), Reservoir pressure monitoring via surface deformation
inversion, Australian Energy Producers Journal, doi:10.1071/EP24037.

Wang and Tartakovsky (2026), Latent Diffusion Posterior Sampling with Surrogate
Likelihood Guidance for PDE Inverse Problems, arXiv:2606.26592 (preprint).
