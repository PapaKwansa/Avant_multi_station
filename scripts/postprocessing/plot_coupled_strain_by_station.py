
"""Publication-quality transient strain plots for AVANT stations.

Reads the wide-format coupled_strain_history.csv.
Produces eight individual station figures and an optional overview.
No hydraulic or mechanical calculations are performed.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENTS = ("eXX", "eYY", "eXY", "eZZ")

LABELS = {
    "eXX": r"$\varepsilon_{XX}$",
    "eYY": r"$\varepsilon_{YY}$",
    "eXY": r"$\varepsilon_{XY}$",
    "eZZ": r"$\varepsilon_{ZZ}$",
}

COLORS = {
    "eXX": "#2166AC",
    "eYY": "#D95F02",
    "eXY": "#7570B3",
    "eZZ": "#1B9E77",
}


def validate_data(df):
    stations = [f"S{i:02d}" for i in range(1, 9)]

    required = ["time_h"] + [
        f"{comp}_{station}"
        for station in stations
        for comp in COMPONENTS
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.time_h.duplicated().any():
        raise ValueError("Duplicate times found.")

    if not np.isfinite(df[required].to_numpy(float)).all():
        raise ValueError("Non-finite values detected.")

    return stations


def format_axes(ax, shutin_h):
    ax.axvline(
        shutin_h,
        color="#444444",
        linestyle="--",
        linewidth=1.4,
        label="Injection stops",
    )

    ax.axhline(
        0,
        color="#777777",
        linewidth=0.9,
    )

    ax.grid(
        True,
        alpha=0.2,
        linewidth=0.7,
    )

    ax.set_xlabel("Time since injection start (h)", fontsize=12)
    ax.set_ylabel("Strain (nanostrain)", fontsize=12)
    ax.tick_params(labelsize=11)


def plot_individual(df, station, out, shutin_h):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    for comp in COMPONENTS:
        ax.plot(
            df.time_h,
            df[f"{comp}_{station}"],
            color=COLORS[comp],
            linewidth=2.2,
            label=LABELS[comp],
        )

    format_axes(ax, shutin_h)

    ax.set_title(
        f"Transient hydraulic-cuboid strain | {station}",
        fontsize=15,
        pad=15,
        weight="bold",
    )

    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=11,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    fig.savefig(
        out / f"coupled_strain_{station}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out / f"coupled_strain_{station}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_overview(df, stations, out, shutin_h):
    fig, axes = plt.subplots(
        4, 2,
        figsize=(13, 17),
        sharex=True,
        sharey=True,
    )

    for ax, station in zip(axes.flat, stations):
        for comp in COMPONENTS:
            ax.plot(
                df.time_h,
                df[f"{comp}_{station}"],
                color=COLORS[comp],
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=LABELS[comp],
            )

        ax.axvline(
            shutin_h,
            color="#444444",
            linestyle="--",
            linewidth=1,
        )

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(station, fontsize=13, weight="bold")
        ax.grid(alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Time (h)", fontsize=11)

    for ax in axes[:, 0]:
        ax.set_ylabel("Strain (nanostrain)", fontsize=11)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=12,
        frameon=False,
    )

    fig.suptitle(
        "Transient hydraulic-cuboid strain at eight stations",
        fontsize=17,
        y=0.99,
    )

    fig.subplots_adjust(
        top=0.91,
        bottom=0.06,
        left=0.09,
        right=0.97,
        hspace=0.25,
        wspace=0.16,
    )

    fig.savefig(
        out / "coupled_strain_overview.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shutin-h", type=float, default=19.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.sort_values("time_h")

    stations = validate_data(df)

    for station in stations:
        plot_individual(
            df, station, out, args.shutin_h
        )

    plot_overview(
        df, stations, out, args.shutin_h
    )

    print("Generated eight station figures and overview.")
    print("Saved to:", out)


if __name__ == "__main__":
    main()
