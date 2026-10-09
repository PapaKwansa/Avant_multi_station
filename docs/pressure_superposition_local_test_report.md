# Local test report - AVANT pressure superposition starter

Date: 2026-10-08. This is a programming and same-model synthetic test report,
not an AVANT field result, new parameter calibration, or independent physics validation.

## What was actually executed

The new modules were tested in a temporary package. The original analytical
strain, coordinate, and tensor-rotation functions were transcribed from the
user's chat message. The original multi-station forward wrapper was obtained
from the uploaded markdown, removing markdown backslash escaping and replacing
nonbreaking spaces with ordinary spaces. No change to the numerical formulae
was intended. These original fixtures are NOT shipped in the add-on.

The actual Git checkout, its complete existing tests, actual station CSVs and
COMSOL spatial pressure fields were not available in this execution. Therefore
the same tests must still run in the user's environment. The user's files and
remote repository were not modified.

## Results

Automated tests: **48 passed**. The suite covers 16 angle/partition combinations,
zero and signed pressure, tile volumes/depths, translation, tensor rotation,
material scaling, component ordering, source-domain guards, non-finite response
handling, a deterministic noise-free pressure recovery, free-surface traction
conditions, exact Gaussian reference calculations, correlated error covariance,
and preservation of uncertainty in a likelihood null space.

Four separate demonstration partition regressions passed. At a 1 MPa uniform
loading, the maximum discrepancy with the original wrapper was
**5.0040e-12 nanostrain** in the
demonstration. This measures consistency between two uses of the SAME kernel;
it does not establish an absolute error relative to the physical solution.

Demonstration setup: eight invented station positions, 32 channels, six training
stations and two withheld stations, 16 calculation tiles and four pressure
coefficients. Noise standard deviation: 2 nanostrain. Prior coefficient mean:
0.5 MPa; prior standard deviation: 0.5 MPa. Fixed geometry and material properties.

- Pressure mean RMSE against the invented truth: 0.2223 MPa.
- Training strain RMSE: 1.4175 nanostrain.
- Withheld strain RMSE: 2.0445 nanostrain.
- Conditional coefficient SDs: 0.1473, 0.1633, 0.1332, 0.1362 MPa.
- Numerical rank: 4/4 for this particular synthetic network/design.
- All four true values fall in their marginal 95% intervals in this ONE realization.
  This is not a coverage-calibration study or a guarantee for other realizations.

Interpretation: near-noise-level strain fits coexist with appreciable pressure
uncertainty and imperfect point estimates. That is why uncertainty and resolution
must be part of the result, rather than treating a fitted pressure map as exact.

## Environment

Python: 3.13.5
NumPy: 2.3.5
Pandas: 2.2.3
Pytest: 9.0.2

The new Python sources were also parsed against Python 3.9 syntax. This is not
an execution test under Python 3.9 or on Windows/Palmetto. No dependency upgrades
are prescribed. Output provenance records the exact executed module hashes.
