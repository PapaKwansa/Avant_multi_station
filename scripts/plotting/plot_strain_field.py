#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-quality plan-view AVANT analytical strain fields.

The script evaluates the authoritative production analytical strain kernel
at a user-specified physical depth and plots:

    epsilon_xx
    epsilon_yy
    epsilon_xy
    epsilon_zz

in nanostrain.

Coordinate convention
---------------------
The command-line x/y coordinates describe the global horizontal plotting
frame. The inclusion center is (x0, y0), and theta_deg rotates the inclusion
relative to that frame.

Depth is positive downward:

    surface       depth = 0
    inclusion     centered at depth = h

The exact free surface is intentionally excluded by this plotting utility.

No analytical equations are duplicated here. The calculation delegates to:

    avant_model.model.multi_station_strain
    avant_model.model.multi_station_rotation

Outputs
-------
A high-resolution PNG and optional vector PDF are written to the requested
output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from avant_model.model.multi_station_rotation import rotate_strain_tensor
from avant_model.model.multi_station_strain import (
    charac_strain,
    linear_trans,
    strain,
)


COMPONENTS = (
    ("xx", r"$\epsilon_{xx}$"),
    ("yy", r"$\epsilon_{yy}$"),
    ("xy", r"$\epsilon_{xy}$"),
    ("zz", r"$\epsilon_{zz}$"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot publication-quality plan-view analytical strain fields "
            "at a specified physical depth."
        )
    )

    # Geometry
    parser.add_argument("--a", type=float, required=True,
                        help="Inclusion x semi-dimension [m].")
    parser.add_argument("--b", type=float, required=True,
                        help="Inclusion y semi-dimension [m].")
    parser.add_argument("--c", type=float, required=True,
                        help="Inclusion vertical semi-dimension [m].")
    parser.add_argument("--h", type=float, required=True,
                        help="Inclusion-center depth, positive downward [m].")

    parser.add_argument(
        "--theta",
        "--theta-deg",
        dest="theta_deg",
        type=float,
        default=0.0,
        help="Horizontal inclusion orientation [degrees]. Default: 0.",
    )

    parser.add_argument(
        "--x0",
        type=float,
        default=0.0,
        help="Global x coordinate of inclusion center [m]. Default: 0.",
    )
    parser.add_argument(
        "--y0",
        type=float,
        default=0.0,
        help="Global y coordinate of inclusion center [m]. Default: 0.",
    )

    # Material / pressure
    parser.add_argument("--pressure", "--p", dest="pressure",
                        type=float, required=True,
                        help="Pore-pressure perturbation [Pa].")
    parser.add_argument("--E", type=float, required=True,
                        help="Young's modulus [Pa].")
    parser.add_argument("--nu", type=float, required=True,
                        help="Poisson ratio.")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Biot coefficient.")

    # Requested plane
    parser.add_argument(
        "--depth",
        type=float,
        required=True,
        help="Evaluation depth, positive downward from surface [m].",
    )

    # Horizontal domain
    parser.add_argument("--xmin", type=float, default=-600.0,
                        help="Minimum global x [m]. Default: -600.")
    parser.add_argument("--xmax", type=float, default=600.0,
                        help="Maximum global x [m]. Default: 600.")
    parser.add_argument("--ymin", type=float, default=-600.0,
                        help="Minimum global y [m]. Default: -600.")
    parser.add_argument("--ymax", type=float, default=600.0,
                        help="Maximum global y [m]. Default: 600.")

    parser.add_argument(
        "--grid",
        type=int,
        default=181,
        help="Number of grid points in each horizontal direction. Default: 181.",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=31,
        help="Number of filled-contour levels. Default: 31.",
    )

    # Plot/output controls
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "strain_fields",
        help="Output directory. Default: plots/strain_fields.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional output basename.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG resolution. Default: 400 dpi.",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save a vector PDF.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )

    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    scalar_values = {
        "a": args.a,
        "b": args.b,
        "c": args.c,
        "h": args.h,
        "theta": args.theta_deg,
        "x0": args.x0,
        "y0": args.y0,
        "pressure": args.pressure,
        "E": args.E,
        "nu": args.nu,
        "alpha": args.alpha,
        "depth": args.depth,
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
    }

    for name, value in scalar_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite; received {value!r}."
            )

    for name in ("a", "b", "c", "h", "E"):
        if getattr(args, name) <= 0.0:
            raise ValueError(
                f"{name} must be strictly positive."
            )

    if args.pressure == 0.0:
        raise ValueError(
            "pressure must be nonzero for a strain-field plot."
        )

    if not (-1.0 < args.nu < 0.5):
        raise ValueError(
            "nu must satisfy -1 < nu < 0.5 for the elastic formulation."
        )

    if args.grid < 25:
        raise ValueError(
            "--grid must be at least 25."
        )

    if args.levels < 5:
        raise ValueError(
            "--levels must be at least 5."
        )

    if args.xmin >= args.xmax:
        raise ValueError(
            "xmin must be smaller than xmax."
        )

    if args.ymin >= args.ymax:
        raise ValueError(
            "ymin must be smaller than ymax."
        )

    # User-requested plotting-domain restrictions.
    if args.depth <= 0.0:
        raise ValueError(
            "depth must be > 0 m. This plotting utility intentionally "
            "does not evaluate the exact free surface (depth = 0)."
        )

    deepest_allowed = args.h + args.c

    if args.depth > deepest_allowed:
        raise ValueError(
            f"Requested depth {args.depth:g} m is deeper than the "
            f"permitted plotting limit h + c = {deepest_allowed:g} m."
        )


def global_to_inclusion(
    x_global: np.ndarray,
    y_global: np.ndarray,
    x0: float,
    y0: float,
    theta_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert global horizontal coordinates to inclusion-frame coordinates.

    This follows the same horizontal convention used by the production
    multi-station workflow.
    """

    theta = np.deg2rad(theta_deg)

    dx = x_global - x0
    dy = y_global - y0

    x_local = (
        dx * np.cos(theta)
        - dy * np.sin(theta)
    )

    y_local = (
        dx * np.sin(theta)
        + dy * np.cos(theta)
    )

    return x_local, y_local


def evaluate_fields(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Evaluate the four reported strain fields in nanostrain.
    """

    x_values = np.linspace(
        args.xmin,
        args.xmax,
        args.grid,
    )

    y_values = np.linspace(
        args.ymin,
        args.ymax,
        args.grid,
    )

    X, Y = np.meshgrid(
        x_values,
        y_values,
        indexing="xy",
    )

    X_local, Y_local = global_to_inclusion(
        X,
        Y,
        args.x0,
        args.y0,
        args.theta_deg,
    )

    linear_response = linear_trans(
        args.alpha,
        args.nu,
        args.pressure,
        args.E,
    )

    ec = charac_strain(
        linear_response,
        args.nu,
    )

    fields = {
        "xx": np.empty_like(X, dtype=float),
        "yy": np.empty_like(X, dtype=float),
        "xy": np.empty_like(X, dtype=float),
        "zz": np.empty_like(X, dtype=float),
    }

    flat_x = X_local.ravel()
    flat_y = Y_local.ravel()

    flat_fields = {
        key: value.ravel()
        for key, value in fields.items()
    }

    for index, (x_local, y_local) in enumerate(
        zip(flat_x, flat_y)
    ):
        tensor = strain(
            x=float(x_local),
            y=float(y_local),
            z=float(args.depth),
            a=float(args.a),
            b=float(args.b),
            c=float(args.c),
            ec=float(ec),
            h=float(args.h),
            nu=float(args.nu),
        )

        tensor = np.asarray(
            tensor,
            dtype=float,
        )

        if tensor.shape != (3, 3):
            raise RuntimeError(
                "Analytical strain kernel did not return a 3x3 tensor."
            )

        (
            exx,
            eyy,
            exy,
            _exz,
            _eyz,
            ezz,
        ) = rotate_strain_tensor(
            tensor[0, 0],
            tensor[1, 1],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 2],
            tensor[2, 2],
            args.theta_deg,
        )

        # Dimensionless strain -> nanostrain.
        flat_fields["xx"][index] = exx * 1.0e9
        flat_fields["yy"][index] = eyy * 1.0e9
        flat_fields["xy"][index] = exy * 1.0e9
        flat_fields["zz"][index] = ezz * 1.0e9

    for name, values in fields.items():
        if not np.all(np.isfinite(values)):
            count = int(
                np.size(values)
                - np.count_nonzero(np.isfinite(values))
            )

            raise RuntimeError(
                f"{name} field contains {count} non-finite values. "
                "Check whether the requested grid intersects a "
                "singular analytical location."
            )

    return X, Y, fields


def symmetric_levels(
    values: np.ndarray,
    n_levels: int,
) -> tuple[np.ndarray, TwoSlopeNorm]:
    """
    Construct zero-centered symmetric levels for a signed strain field.
    """

    maximum = float(
        np.nanmax(
            np.abs(values)
        )
    )

    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0

    levels = np.linspace(
        -maximum,
        maximum,
        n_levels,
    )

    norm = TwoSlopeNorm(
        vmin=-maximum,
        vcenter=0.0,
        vmax=maximum,
    )

    return levels, norm


def inclusion_footprint(
    args: argparse.Namespace,
    n_points: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the rotated rectangular inclusion footprint in global coordinates.
    """

    # Closed local rectangle.
    x_local = np.array(
        [
            -args.a,
            args.a,
            args.a,
            -args.a,
            -args.a,
        ],
        dtype=float,
    )

    y_local = np.array(
        [
            -args.b,
            -args.b,
            args.b,
            args.b,
            -args.b,
        ],
        dtype=float,
    )

    theta = np.deg2rad(args.theta_deg)

    x_global = (
        x_local * np.cos(theta)
        + y_local * np.sin(theta)
        + args.x0
    )

    y_global = (
        -x_local * np.sin(theta)
        + y_local * np.cos(theta)
        + args.y0
    )

    return x_global, y_global


def make_figure(
    X: np.ndarray,
    Y: np.ndarray,
    fields: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> plt.Figure:
    """
    Build the publication-quality 2x2 strain-field figure.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 9.0),
        constrained_layout=False,
    )

    axes = axes.ravel()

    footprint_x, footprint_y = inclusion_footprint(
        args
    )

    for ax, (component, label) in zip(
        axes,
        COMPONENTS,
    ):
        values = fields[component]

        levels, norm = symmetric_levels(
            values,
            args.levels,
        )

        contour = ax.contourf(
            X,
            Y,
            values,
            levels=levels,
            norm=norm,
            cmap="RdBu_r",
            extend="both",
        )

        # Light contour lines help reveal field structure without
        # overwhelming the filled contours.
        ax.contour(
            X,
            Y,
            values,
            levels=levels[::2],
            norm=norm,
            colors="k",
            linewidths=0.22,
            alpha=0.12,
        )

        # Inclusion center.
        ax.plot(
            args.x0,
            args.y0,
            marker="+",
            markersize=9,
            markeredgewidth=1.3,
            color="black",
            zorder=5,
        )

        # Projected inclusion footprint.
        ax.plot(
            footprint_x,
            footprint_y,
            linestyle="--",
            linewidth=1.1,
            color="black",
            alpha=0.75,
            zorder=4,
        )

        ax.set_title(
            label,
            fontsize=14,
            pad=8,
        )

        ax.set_xlabel(
            "x (m)",
            fontsize=11,
        )

        ax.set_ylabel(
            "y (m)",
            fontsize=11,
        )

        ax.set_xlim(
            args.xmin,
            args.xmax,
        )

        ax.set_ylim(
            args.ymin,
            args.ymax,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.tick_params(
            axis="both",
            labelsize=9,
        )

        ax.grid(
            True,
            linewidth=0.45,
            alpha=0.18,
        )

        colorbar = fig.colorbar(
            contour,
            ax=ax,
            fraction=0.046,
            pad=0.035,
        )

        colorbar.set_label(
            "Nanostrain",
            fontsize=10,
            fontweight="bold",
        )

        colorbar.ax.tick_params(
            labelsize=8,
        )

    fig.suptitle(
        f"Plan-view strain fields at depth {args.depth:g} m",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )

    parameter_text = (
        rf"$a={args.a:g}$ m, "
        rf"$b={args.b:g}$ m, "
        rf"$c={args.c:g}$ m, "
        rf"$h={args.h:g}$ m, "
        rf"$\theta={args.theta_deg:g}^\circ$, "
        rf"$x_0={args.x0:g}$ m, "
        rf"$y_0={args.y0:g}$ m"
        "\n"
        rf"$P={args.pressure:.3g}$ Pa, "
        rf"$E={args.E:.3g}$ Pa, "
        rf"$\nu={args.nu:g}$, "
        rf"$\alpha={args.alpha:g}$"
    )

    fig.text(
        0.5,
        0.018,
        parameter_text,
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.96,
        bottom=0.11,
        top=0.91,
        wspace=0.28,
        hspace=0.28,
    )

    return fig


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_inputs(args)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("AVANT PLAN-VIEW STRAIN FIELD")
    print("=" * 72)

    print("\nGeometry")
    print(f"  a       = {args.a:g} m")
    print(f"  b       = {args.b:g} m")
    print(f"  c       = {args.c:g} m")
    print(f"  h       = {args.h:g} m")
    print(f"  theta   = {args.theta_deg:g} deg")
    print(f"  center  = ({args.x0:g}, {args.y0:g}) m")

    print("\nMaterial / pressure")
    print(f"  pressure = {args.pressure:.12g} Pa")
    print(f"  E        = {args.E:.12g} Pa")
    print(f"  nu       = {args.nu:g}")
    print(f"  alpha    = {args.alpha:g}")

    print("\nEvaluation")
    print(f"  depth    = {args.depth:g} m")
    print(
        f"  x range  = [{args.xmin:g}, {args.xmax:g}] m"
    )
    print(
        f"  y range  = [{args.ymin:g}, {args.ymax:g}] m"
    )
    print(
        f"  grid     = {args.grid} x {args.grid}"
    )

    X, Y, fields = evaluate_fields(
        args
    )

    print("\nField ranges [nanostrain]")

    for component, _label in COMPONENTS:
        values = fields[component]

        print(
            f"  epsilon_{component}: "
            f"{np.min(values): .6g} to "
            f"{np.max(values): .6g}"
        )

    fig = make_figure(
        X,
        Y,
        fields,
        args,
    )

    if args.name:
        basename = args.name
    else:
        depth_token = (
            f"{args.depth:g}"
            .replace(".", "p")
            .replace("-", "m")
        )

        basename = (
            f"strain_field_depth_{depth_token}m"
        )

    png_path = (
        args.output_dir
        / f"{basename}.png"
    )

    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    print(f"\nSaved PNG:\n  {png_path.resolve()}")

    if args.pdf:
        pdf_path = (
            args.output_dir
            / f"{basename}.pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        print(f"\nSaved PDF:\n  {pdf_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("\n" + "=" * 72)
    print("STRAIN-FIELD PLOT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()