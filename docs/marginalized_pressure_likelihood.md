# AVANT pressure-marginalized likelihood — stage 1

This is an **add-only statistical building block**, not the joint geometry–pressure
DREAM runner. It preserves the original analytical strain, coordinate routines,
existing geometry likelihood, and current pressure-only inversion.

At each proposed geometry, construct the tile pressure-to-strain matrix using
`pressure_superposition.py`, then form the coarse-basis design `H = G @ B`
with **exactly the same station/component/time ordering as the observations**.
For fixed geometry, model `d = H q + e`, with Gaussian `q ~ N(m0,C0)` and
`e ~ N(0,R)`. The marginalized distribution is
`d ~ N(H@m0, R + H@C0@H.T)`.
The module returns this normalized log density and the conditional Gaussian
pressure posterior, evaluated through small pressure-space linear algebra.

## Install and test

Apply the accompanying add-only patch from the repository root, then:

```bat
python -m pytest -q tests/test_marginalized_pressure_likelihood.py
python -m pytest -q
python scripts/validation/check_marginalized_pressure_likelihood.py
```

No COMSOL input files are needed for these tests. They use synthetic small
matrices, and cannot establish geometry identifiability or field validity.

## Important limitations before the DREAM integration

1. Build `H(gamma)` for each geometry proposal. Preserve the constrained
   geometry parameterization of the production runner; do not mix its depth
   settings with the Larry transient benchmark.
2. For stacked reference-differenced observations, shared reference noise
   produces cross-time covariance. The helper
   `reference_difference_covariance(R0,T)` constructs this covariance only
   under **independent raw-time errors with the same per-time covariance**.
3. The existing COMSOL residuals show systematic station/component biases.
   A full likelihood needs a defensible model-discrepancy covariance or a
   sensitivity study. Do not interpret IID 2 ns errors as established.
4. An independent per-time pressure prior permits unrelated pressure values
   at each time. A physical temporal prior must be selected and documented
   before a full 106-time joint fit. The module accepts any proper SPD C0.
5. Rebuilding the mechanical kernel for every DREAM proposal may be costly.
   Benchmark a small chain and cache only genuinely reusable calculations.
6. Keep pressure amplitude versus Young's modulus scaling ambiguity in view.
7. The optional legacy volume likelihood is **not** included; it requires
   a separate physical interpretation under nonuniform pressure.

## Next implementation stage

Create an **additive** joint DREAM runner with explicit geometry parameter
mapping, fixed transient station conversion, observation ordering, temporal
pressure basis/prior, residual covariance, and posterior reconstruction.
Verify synthetic truth recovery and compare the marginal log likelihood with
direct dense Gaussian calculations before submitting any Palmetto job.
