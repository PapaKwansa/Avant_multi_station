#!/usr/bin/env python3
"""
Calculate the AVANT analytical strain tensor for the Leonid point benchmark.

This is intentionally a very small independent benchmark wrapper around the
authoritative production analytical strain kernel.

IMPORTANT
---------
The x, y, z values below are the coordinates used directly by the analytical
strain equations. No coordinate transformation or tensor rotation is applied.

This script does not:
    - perform inversion or fitting;
    - generate a time series;
    - modify the analytical model;
    - transform coordinates;
    - compare against external results automatically.

It only evaluates the strain tensor for the specified Leonid benchmark state.
"""

from __future__ import annotations

import numpy as np

from avant_model.model.multi_station_strain import (
    charac_strain,
    linear_trans,
    strain,
)


# ============================================================================
# LEONID BENCHMARK INPUTS
# ============================================================================

# Material properties
NU = 0.25
E = 1.0e10       # Pa
ALPHA = 0.8

# Inclusion geometry / depth
H = 2500.0       # m
A = 175.0        # m
B = 25.0         # m
C = 125.0        # m

# Analytical-frame observation coordinates.
# These are used directly by strain(); do NOT transform them.
X = -67.04       # m
Y = -12.0        # m
Z = 2510.0       # m

# Pressure
P = 1.0e7        # Pa = 10 MPa


def main() -> None:
    """Evaluate and report the Leonid benchmark strain tensor."""

    # ------------------------------------------------------------------
    # Material-to-characteristic-strain conversion
    # ------------------------------------------------------------------
    linear_response = linear_trans(
        ALPHA,
        NU,
        P,
        E,
    )

    ec = charac_strain(
        linear_response,
        NU,
    )

    # ------------------------------------------------------------------
    # Authoritative analytical strain calculation
    # ------------------------------------------------------------------
    tensor = strain(
        x=X,
        y=Y,
        z=Z,
        a=A,
        b=B,
        c=C,
        ec=ec,
        h=H,
        nu=NU,
    )

    tensor = np.asarray(
        tensor,
        dtype=float,
    )

    if tensor.shape != (3, 3):
        raise RuntimeError(
            f"Expected a 3x3 strain tensor; got shape {tensor.shape}."
        )

    if not np.all(np.isfinite(tensor)):
        raise RuntimeError(
            "Analytical strain tensor contains non-finite values."
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 72)
    print("AVANT / LEONID POINT-STRAIN BENCHMARK")
    print("=" * 72)

    print("\nInputs")
    print(f"  nu       = {NU:.12g}")
    print(f"  E        = {E:.12g} Pa")
    print(f"  alpha    = {ALPHA:.12g}")
    print(f"  pressure = {P:.12g} Pa")
    print(f"  h        = {H:.12g} m")
    print(f"  a        = {A:.12g} m")
    print(f"  b        = {B:.12g} m")
    print(f"  c        = {C:.12g} m")
    print(f"  x        = {X:.12g} m")
    print(f"  y        = {Y:.12g} m")
    print(f"  z        = {Z:.12g} m")

    print("\nIntermediate quantities")
    print(f"  linear_trans    = {linear_response:.12e}")
    print(f"  characteristic  = {ec:.12e}")

    print("\nStrain tensor (dimensionless)")
    print(tensor)

    print("\nStrain components")
    print(f"  epsilon_xx = {tensor[0, 0]: .12e}")
    print(f"  epsilon_yy = {tensor[1, 1]: .12e}")
    print(f"  epsilon_zz = {tensor[2, 2]: .12e}")
    print(f"  epsilon_xy = {tensor[0, 1]: .12e}")
    print(f"  epsilon_xz = {tensor[0, 2]: .12e}")
    print(f"  epsilon_yz = {tensor[1, 2]: .12e}")

    print("\nStrain components (nanostrain)")
    print(f"  epsilon_xx = {tensor[0, 0] * 1.0e9: .6f} nstrain")
    print(f"  epsilon_yy = {tensor[1, 1] * 1.0e9: .6f} nstrain")
    print(f"  epsilon_zz = {tensor[2, 2] * 1.0e9: .6f} nstrain")
    print(f"  epsilon_xy = {tensor[0, 1] * 1.0e9: .6f} nstrain")
    print(f"  epsilon_xz = {tensor[0, 2] * 1.0e9: .6f} nstrain")
    print(f"  epsilon_yz = {tensor[1, 2] * 1.0e9: .6f} nstrain")

    print("\n" + "=" * 72)
    print("CALCULATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()