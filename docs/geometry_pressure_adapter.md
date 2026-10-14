# Stage 2: geometry-dependent marginalized pressure likelihood

This is an **add-only, single-snapshot** adapter. It recomputes the existing
pressure-superposition kernel for every proposed `Cuboid`, maps mechanical
tiles to pressure zones with `coarse_block_basis`, and evaluates the already
tested `marginalized_pressure_likelihood`.

## Explicit inputs

- Geometry is a `Cuboid(a,b,c,h,x0_prime,y0_prime,theta_deg)` with center depth `h`.
- Elastic constants are an `ElasticProperties(E,nu,alpha)`.
- `x_prime`, `y_prime` and `z` are station arrays; **z is verified analytical positive-downward depth**.
- `observed_ns` is one snapshot of strain *changes* in nanostrain.
- `observed_channels` must match the exact component-major kernel order.
- `prior_mean_pa`, `prior_covariance_pa2`, `noise_covariance_ns2` must be supplied explicitly.
- Only training stations should be passed if holdouts are desired.

## Limits

This module is **not** a DREAM runner, a COMSOL validation, or a field inversion.
It does not assign geometry priors, enforce `b>a`, or sample parameters. Those
are responsibilities of the future constrained DREAM adapter. Invalid proposed
geometries raise errors; a future sampler wrapper should map rejected proposals
to minus infinity according to explicit parameter bounds.

For a single reference-differenced snapshot, supply the correct covariance
(e.g. `2*R0` if raw observation errors at the reference and current time are
independent with common covariance `R0`). For multiple times, use a stacked
design and the correct cross-time reference covariance; **do not independently
multiply snapshot likelihoods when they share a noisy reference**. A
physically justified temporal pressure prior is also needed.

Run `python -m pytest -q tests/test_geometry_pressure_adapter.py`.
The tests use the actual unchanged mechanical kernel with a tiny two-station
synthetic setup; they are not COMSOL or geometry-identifiability validation.
