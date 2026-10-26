# AVANT Stage 6A — hydraulic / mechanical / spatial parameter and coordinate contract

**Scope:** Additive design-and-validation layer. **Not a new generalized forward solver,
not Stage 6 DREAM, not independent COMSOL validation and not a field inversion.**
This stage precedes 6B and must preserve Stages 1–5.

## Physical decision

The existing analytical strain kernel is linear in `alpha/E`, while the
current one-way FV pressure solve uses only `K, Ss`. Thus `E` and `alpha`
are exactly structurally confounded if freely estimated together in this
model. The primary eight-parameter state is:

```
[log10_K_m_s, log10_Ss_per_m, log10_gamma_rel,
 a_m, b_m, theta_deg, x0_prime_m, y0_prime_m]
```

`gamma_rel = (alpha/E) / (alpha_ref/E_ref)` is dimensionless, with
`alpha_ref=0.7` and `E_ref=8e9 Pa` for Larry's established benchmark. It
is an *effective mechanical amplitude*, not an independent measurement of
E and alpha. If alpha is independently fixed at alpha_ref, then
`E = E_ref/gamma_rel`; otherwise report the inferred alpha/E combination.
Poisson ratio `nu` is fixed at 0.25 initially and may be activated as a
ninth free parameter **only after sensitivity testing**.

Effective `Ss` is a one-way hydraulic storage coefficient. It is NOT
claimed to enforce full two-way Biot storage consistency.

## Global reference frame and geometry

The existing verified AVANT COMSOL transient convention is retained:
COMSOL surface global `z=+520 m`, strain plane `z=+490 m`, analytical
receiver depth `30 m`; inclusion `h=522.5 m` positive downward and
`c=2.5 m`. **Do not apply a second z transformation to the existing
analytical-depth station CSV.** Stage 6A never performs z conversion.

Physical global horizontal coordinates use the existing primed/survey
`x', y'` convention, while cuboid local `xi, eta` axes rotate according
to the unchanged `multi_station_coord_transform.py` convention. An
explicit inverse transformation and rotated-corner containment check are
provided for Stage 6B's FV-field-to-mechanical-tile mapping.

- The **global FV domain and hydraulic boundaries must remain fixed**
  when the geometry parameter vector changes.
- The injection well and observation gauges remain at **physical global
  positions**, not moving with the cuboid.
- The variable mechanical cuboid is an explicitly defined *finite
  pressurized mechanical support*, not a geological aquifer boundary.
  Modeling its pressure/strain response may neglect pressure-induced
  strain outside that support; Stage 6B must quantify this truncation.
- A well-in-support check is enabled for the illustrative first benchmark,
  but is a configurable modeling assumption, not a universal requirement.
- The existing Stage 4 finite-volume solver has *centered-well and
  fixed-local-domain assumptions*. Stage 6A does not replace them.

`configs/pressure_superposition/stage6/stage6a_contract_001.json`
provides an **illustrative proposed global domain and parameter bounds**;
these are NOT externally validated North Avant values or scientifically
justified field priors. Domain edges (x' +/- 1200 m, y' +/- 1600 m) were
chosen simply to keep the benchmark source well inside the computational
support. Reassess well position, boundary influence, time histories,
loading schedule and material assumptions before any physical inference.

The canonical rectangle has `0 < a < b` and `theta_deg` in `[0,180)`.
Sampling `a,b,theta` without a symmetry convention would introduce
redundant representations. A nearly square rectangle still has weakly
identifiable orientation. Geometry bounds can include some infeasible
combinations, which the contract rejects explicitly rather than silently
adjusting them. Prior density, including the implications of transforms
and geometric constraints, is a Stage 6C/6D task.

## Installation and verification (Windows CMD, from repository root)

Extract the ZIP into the repository root. It adds only these files:

```
src/avant_model/inversion/stage6_parameters.py
src/avant_model/model/stage6_coordinates.py
configs/pressure_superposition/stage6/stage6a_contract_001.json
scripts/validation/check_stage6_contract.py
tests/test_stage6_contract.py
tests/test_stage6_coordinates.py
README_STAGE6A_CONTRACT.md
```

Run:

```
python -m pytest -q tests\test_stage6_contract.py tests\test_stage6_coordinates.py
python -m pytest -q
python scripts\validation\check_stage6_contract.py --config configs\pressure_superposition\stage6\stage6a_contract_001.json --output-dir results\validation\stage6a_contract_001
```

Outputs: `stage6a_contract_report.json`, `stage6a_reference_geometry.png`,
`stage6a_reference_geometry.pdf`. These are contract visualizations, **not
inversion result plots**.

## Gate into Stage 6B

1. Stage 6A inputs and all tests pass, without changing legacy physics.
2. All Stage 4 reference predictions are later reproduced to a documented
   tolerance in a LEGACY_COMPATIBILITY MODE; do not require the new larger
   fixed global domain to produce identical pressure histories.
3. Stage 6B implements a fixed-global FV solution, a physically placed
   source/well, physical gauge sampling and geometry-dependent finite-tile
   pressure integration. Compare against an independent high-resolution
   tile integration and pressure/strain analytical checks.
4. Verify the pressure outside the mechanical source and quantify the
   support-truncation effect on mechanical predictions.
5. Only then extend the observed-data likelihood, sensitivity analysis,
   multi-start maximum likelihood, and DREAM sampling to Stage 6.

Stage 6A **does not** imply independent full coupled COMSOL validation.
That remains pending Larry's spatial pressure reference.
