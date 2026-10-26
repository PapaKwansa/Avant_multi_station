# AVANT M1 pilot — geological conductivity contrast, not a strain cutoff

## Scientific purpose and go/no-go status

This is a **bounded, same-model forward-model pilot**. It does **not** perform
hydraulic/geometry parameter estimation, field inference or independent COMSOL
validation. It is not a full two-way hydrogeomechanical or heterogeneous elastic
solver. It has been developed add-only, retaining the analytical cuboid equations
unchanged.

**What is physically represented:** A rotated, 2D, thin aquifer compartment
with different **hydraulic conductivity** from its surroundings, within a
**homogeneous, isotropic, linear elastic half-space**. The pressure change in
the relevant aquifer layer, inside *and outside* the geological compartment,
loads a fixed thin poroelastic layer represented by a *separate, constant*
mechanical integration rectangle tiled into many cuboids.

**What it cannot represent:** A block with different elastic properties,
mechanical contact, fault slip, anisotropic/layered solid mechanics, near-well
flow details, fully coupled Biot consolidation, or unknown physical outer flow
boundaries. A geological compartment that is a *thin sealing fault* would need
an interface-resistance model, not simply two zone K values.

## Model structure

With constant aquifer thickness `b_thick`, one effective `Ss`, a prescribed
rate well and fixed zero-head-change external FV boundaries:

    Ss * b_thick * cell_area * dh/dt + sum(face_transmissibility * (h_i-h_j)) = well_flow_i

The conductivity field is `K_out + (K_in-K_out)*f_i`, where `f_i` is the
geometrically **exact area fraction** of grid cell `i` overlapped by the
rotated geological rectangle. The *hydraulic* effective-conductivity formula
in mixed cells is approximate. Neighboring cell face conductances use the
harmonic mean. A shared face uses exactly one conductance, so internal FV flux
cancels in the global mass balance. The physical well uses the existing Stage
6B-1 bilinear source weights.

Mechanical strain uses the unchanged `build_pressure_kernel` from
`pressure_superposition.py` on a **fixed loading rectangle** wholly distinct
from the geological conductivity boundary, and the Stage 6B-2 quadrature
pressure sampler. The one-way coupling assumes elastic properties and alpha
are uniform, pressure is effectively uniform over the 5 m aquifer thickness,
and other layers have negligible incremental pore pressure.

For fixed mechanical grid and nu, kernel G is built **only once**:

    strain_ns = gamma_rel * (P_tile_from_FV @ G.T)

Changing geology modifies the pressure field but not the mechanical
integration support. Pressure gauges sample the original full FV field.

## Files added

```
src/avant_model/model/stage6_m1_two_zone_fv.py
src/avant_model/model/stage6_m1_forward.py
configs/pressure_superposition/stage6/stage6_m1_pilot_001.json
scripts/validation/check_stage6_m1_forward.py
tests/test_stage6_m1.py
README_STAGE6_M1_PILOT.md
```

No existing production mechanics, legacy FV, data or DREAM files are replaced.

## Windows CMD commands, repository root

Run on `feature/full-hydromechanical-spatial-inversion` after existing Stage 6B
modules are committed and passing:

```
tar -xf avant_stage6_m1_forward_pilot.zip
python -m pytest -q tests\test_stage6_m1.py
python -m pytest -q
python scripts\validation\check_stage6_m1_forward.py --config configs\pressure_superposition\stage6\stage6_m1_pilot_001.json --output-dir results\validation\stage6_m1_pilot_001
type results\validation\stage6_m1_pilot_001\m1_forward_report.json
start results\validation\stage6_m1_pilot_001\m1_k_and_pressure.png
start results\validation\stage6_m1_pilot_001\m1_strain_comparison.png
```

The original baseline is 366 tests; this package adds 14, for an **expected
380** in the unmodified repository. This number is not a claim of prior run
in Windows or with the production strain kernel. Use a new output directory
if the pilot already exists. Keep historical ZIPs/patches untracked.

## Three hard acceptance conditions before M1 inverse work

1. **Numerical and code regression:** full pytest suite green; `K_in=K_out`
   reproduces Stage 6B-1 global FV at matching grid/well/BC/times; mass
   balance small; mechanical equations unchanged; the hydraulic zone
   disappears from both pressure and strain predictions as `K_in->K_out`.
2. **Physical observation fidelity:** refine the FV grid and mechanical tile
   support/resolution until changes at *actual pressure and strain sensors*
   are smaller than a predeclared *numerical accuracy budget* (chosen relative
   to observation error). In particular, check oblique material-interface
   staircasing and the fixed loading region. The first pilot does **not**
   establish these convergence criteria.
3. **Identifiability:** check the scaled Jacobian for at least `log10 K_in`,
   `log10 K_out`, `log10 Ss`, `log10 gamma_rel` and the selected source
   geometry dimensions. If the near-null direction is unacceptable, reduce
   the parameter subset or fix it from geology. **Do not launch full DREAM**
   simply because the forward model runs.

If (2) cannot be satisfied at reasonable computational cost, or (3) shows
that geology/pressure/strain cannot distinguish a compartment, **stop M1
extension** and finish the robust homogeneous AVANT method/package instead.

## Critical restrictions and interpretation

- The **geological boundary** is a *material K contrast*, not a general
  reservoir/elastic contact. `K_in==K_out` makes its location unobservable to
  this model.
- `Ss` is uniform *effective, one-way* storage. Pressure is not fed back
  from deformation. `gamma_rel` represents the identifiable alpha/E amplitude
  ratio, not independent measurements of E and alpha.
- Numerical boundaries are provisional zero-*change*-in-head. A physical
  boundary cannot be inferred by moving this artificial domain wall.
- The pressure-to-strain cuboids all share the reference c=2.5 m and h=522.5
  m, from the verified Larry transient geometry. Station z has **already**
  been transformed to positive-downward depth (30 m), do not transform twice.
- **Exact geometrical area fraction is not exact hydraulic interface flux.**
  Arithmetic mixed-cell K and harmonic face K are first-pass conservative
  approximations; a slanted sharp contact may be anisotropic at the cut cell.
  Numerical refinement / an external comparison is required for spatial
  boundary estimates.
- The pressure-to-strain support remains finite. A fixed support avoids
  source-boundary confounding, but does not automatically eliminate omitted
  far-field strain; it must be justified and converged against a larger
  *fixed* support.
- No model can guarantee a unique boundary from finite noisy data. Use
  geological priors and report unresolved directions, not false precision.
- The resulting `K(x,y)` is a **two-zone field**; not a freely reconstructed
  arbitrary heterogeneous permeability field. To convert hydraulic
  conductivity K to intrinsic permeability k, require viscosity, fluid density
  and g (`k = mu*K/(rho*g)`).

## Proposed end of implementation

After passing the hard gates above: one geology-constrained MAP/identifiability
experiment, one converged DREAM run on the identifiable subset, and a feature
freeze. Independent COMSOL comparison remains a required gate for real-world
accuracy claims, not an excuse to indefinitely expand the codebase.
