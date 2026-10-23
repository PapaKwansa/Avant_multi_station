
"""Compare hydraulic pressure representations using AVANT cuboid strain.

Run from repository root.

Uses saved finite-volume pressure snapshots and the unchanged analytical
cuboid kernel. No field observations or Bayesian inversion are used.

Three pressure representations:
1. Nine uniform regional means
2. Nine regional mean-plus-gradient fits
3. Independently averaged pressure on every fine mechanical tile

The refined result is a discretized reference, not an exact solution.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.model.pressure_superposition import (
    Cuboid,
    ElasticProperties,
    build_pressure_kernel,
    coarse_block_basis,
)

def block_average(field, nx, ny):
    """
    Conservative area-weighted averaging from a uniform hydraulic
    grid onto a regular mechanical tile grid.

    Works even when hydraulic grid dimensions are not divisible
    by the number of mechanical subdivisions.

    field[j, i] is indexed [hydraulic y, hydraulic x].
    """
    field = np.asarray(field, dtype=float)
    nrows, ncols = field.shape

    if nx <= 0 or ny <= 0:
        raise ValueError("Target subdivisions must be positive.")

    def overlap_weights(nsource, ntarget):
        source_edges = np.linspace(0.0, 1.0, nsource + 1)
        target_edges = np.linspace(0.0, 1.0, ntarget + 1)

        left = np.maximum(
            target_edges[:-1, None],
            source_edges[None, :-1],
        )
        right = np.minimum(
            target_edges[1:, None],
            source_edges[None, 1:],
        )

        overlap = np.maximum(right - left, 0.0)

        # Normalize each target row to preserve constant fields.
        return overlap / overlap.sum(axis=1, keepdims=True)

    wx = overlap_weights(ncols, nx)
    wy = overlap_weights(nrows, ny)

    result = wy @ field @ wx.T

    # For equal-area target tiles, the domain average is preserved.
    np.testing.assert_allclose(
        result.mean(),
        field.mean(),
        rtol=1e-12,
        atol=1e-8,
    )

    return result

def fit_pressure_regions(field, x, y, nzone=3):
    """Fit mean and local x/y gradients to each hydraulic region.

    Returns coefficients indexed [region_y, region_x, coefficient].
    Coefficients are [mean pressure, dp/dx, dp/dy], in Pa and Pa/m.
    """
    ny, nx = field.shape

    if ny % nzone or nx % nzone:
        raise ValueError("Grid must divide into pressure regions.")

    sy, sx = ny // nzone, nx // nzone
    coefficients = np.empty((nzone, nzone, 3))

    for jy in range(nzone):
        for ix in range(nzone):
            ys = slice(jy * sy, (jy + 1) * sy)
            xs = slice(ix * sx, (ix + 1) * sx)

            xx, yy = np.meshgrid(x[xs], y[ys])
            values = field[ys, xs].ravel()

            # Center the design matrix so its intercept is the mean.
            X = np.column_stack((
                np.ones(values.size),
                (xx - xx.mean()).ravel(),
                (yy - yy.mean()).ravel(),
            ))

            coefficients[jy, ix] = np.linalg.lstsq(
                X, values, rcond=None
            )[0]

    return coefficients


def pressure_vectors(field, x, y, tiles, fine_n=9, coarse_n=3):
    """Project FV pressures into analytical x-major tile ordering."""
    ny, nx = field.shape

    if nx != len(x) or ny != len(y):
        raise ValueError("Hydraulic coordinate dimensions disagree.")



    fine = block_average(field, fine_n, fine_n)
    coarse = block_average(field, coarse_n, coarse_n)
    coefficients = fit_pressure_regions(field, x, y, coarse_n)

    # Analytical tiles are ordered x outermost, y innermost.
    refined = fine.T.reshape(-1)

    uniform = np.empty(len(tiles))
    gradient = np.empty(len(tiles))

    lx = (x[-1] - x[0]) + (x[1] - x[0])
    ly = (y[-1] - y[0]) + (y[1] - y[0])

    coarse_dx = lx / coarse_n
    coarse_dy = ly / coarse_n

    for k, tile in enumerate(tiles):
        ix = tile.ix // (fine_n // coarse_n)
        iy = tile.iy // (fine_n // coarse_n)

        if ix >= coarse_n or iy >= coarse_n:
            raise ValueError("Invalid analytical tile mapping.")

        mean, gx, gy = coefficients[iy, ix]

        cx = -lx / 2 + (ix + 0.5) * coarse_dx
        cy = -ly / 2 + (iy + 0.5) * coarse_dy

        uniform[k] = coarse[iy, ix]
        gradient[k] = (
            mean
            + gx * (tile.xi - cx)
            + gy * (tile.eta - cy)
        )

    # Conservation of regional mean pressure.
    B = coarse_block_basis(
        (fine_n, fine_n, 1),
        (coarse_n, coarse_n, 1)
    )
    expected = B @ coarse.T.reshape(-1)

    np.testing.assert_allclose(
        uniform, expected, rtol=1e-12, atol=1e-7
    )

    np.testing.assert_allclose(
        uniform.mean(), refined.mean(), rtol=1e-12, atol=1e-7
    )

    np.testing.assert_allclose(
        gradient.mean(), refined.mean(), rtol=1e-12, atol=1e-7
    )

    return uniform, gradient, refined


def rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fine-n", type=int, default=9)
    args = ap.parse_args()

    if args.fine_n % 3:
        raise ValueError("Fine mechanical splits must be divisible by 3.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text())
    stations = pd.read_csv(cfg["station_file"])

    required = {"station", "x", "y", "z"}
    if not required.issubset(stations.columns):
        raise ValueError("Station coordinate columns are missing.")

    geom = Cuboid(**cfg["geometry"])
    elastic = ElasticProperties(**cfg["elastic"])

    print("Building analytical pressure kernel...", flush=True)

    kernel = build_pressure_kernel(
        geom,
        elastic,
        x_prime=stations.x.to_numpy(float),
        y_prime=stations.y.to_numpy(float),
        z=stations.z.to_numpy(float),
        station_names=stations.station.astype(str).tolist(),
        splits=(args.fine_n, args.fine_n, 1),
        components=tuple(cfg["components"]),
    )

    G = np.asarray(kernel.G, dtype=float)

    print("Kernel shape:", G.shape, flush=True)

    with np.load(args.fields) as f:
        x = np.asarray(f["x_m"], dtype=float)
        y = np.asarray(f["y_m"], dtype=float)
        times = np.asarray(f["times_h"], dtype=float)
        pressure = np.asarray(f["pressure_change_pa"], dtype=float)

    if pressure.shape != (len(times), len(y), len(x)):
        raise ValueError("Unexpected hydraulic pressure array shape.")

    if not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError("Hydraulic coordinates must be increasing.")

    # The hydraulic axes are assumed to be local cuboid xi/eta.
    # Cuboid geometry handles rotation to the observation frame.
    if not np.isclose(x[1] - x[0], x[-1] - x[-2]):
        raise ValueError("Hydraulic x spacing is not uniform.")

    if not np.isclose(y[1] - y[0], y[-1] - y[-2]):
        raise ValueError("Hydraulic y spacing is not uniform.")

    lx = x[-1] - x[0] + (x[1] - x[0])
    ly = y[-1] - y[0] + (y[1] - y[0])

    if not np.isclose(lx, 2 * geom.a) or not np.isclose(ly, 2 * geom.b):
        raise ValueError("Hydraulic dimensions do not match cuboid.")

    predictions = []
    metrics = []

    for t, field in zip(times, pressure):
        uniform, gradient, refined = pressure_vectors(
            field, x, y, kernel.tiles, args.fine_n
        )

        eu = G @ uniform
        eg = G @ gradient
        er = G @ refined

        du = eu - er
        dg = eg - er

        ru = rmse(du)
        rg = rmse(dg)

        metrics.append({
            "time_h": float(t),
            "uniform_strain_rmse_ns": ru,
            "gradient_strain_rmse_ns": rg,
            "improvement_pct": (
                100 * (1 - rg / ru) if ru > 1e-14 else np.nan
            ),
            "uniform_max_abs_error_ns": float(np.max(np.abs(du))),
            "gradient_max_abs_error_ns": float(np.max(np.abs(dg))),
            "reference_peak_abs_strain_ns": float(np.max(np.abs(er))),
        })

        for j, channel in enumerate(kernel.channels):
            predictions.append({
                "time_h": float(t),
                "channel": channel,
                "uniform_ns": float(eu[j]),
                "gradient_ns": float(eg[j]),
                "refined_ns": float(er[j]),
                "uniform_error_ns": float(du[j]),
                "gradient_error_ns": float(dg[j]),
            })

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.DataFrame(predictions)

    metrics_df.to_csv(out / "strain_representation_metrics.csv", index=False)
    pred_df.to_csv(out / "strain_predictions.csv", index=False)

    print("\nSTRAIN REPRESENTATION METRICS")
    print(metrics_df.to_string(index=False))

    # Figure 1: strain approximation error over saved times.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        metrics_df.time_h,
        metrics_df.uniform_strain_rmse_ns,
        "o-", label="Nine uniform-pressure regions"
    )
    ax.plot(
        metrics_df.time_h,
        metrics_df.gradient_strain_rmse_ns,
        "s-", label="Nine mean + gradient regions"
    )

    ax.axvline(19, color="black", ls="--", label="Shut-in")
    ax.set(
        xlabel="Time (hours)",
        ylabel="Strain RMSE (nanostrain)",
        title="Strain approximation error versus refined cuboid reference"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "strain_error_history.png", dpi=250)
    plt.close(fig)

    # Figure 2: channel-level comparison at 19 hours.
    available = times[np.argmin(np.abs(times - 19.0))]
    selected = pred_df[np.isclose(pred_df.time_h, available)]

    fig, ax = plt.subplots(figsize=(11, 5))
    xx = np.arange(len(selected))

    ax.plot(
        xx, selected.refined_ns, "ko-",
        markersize=3, label="Refined reference"
    )
    ax.plot(
        xx, selected.uniform_ns, "o--",
        markersize=3, label="Uniform"
    )
    ax.plot(
        xx, selected.gradient_ns, "s--",
        markersize=3, label="Mean + gradient"
    )

    ax.set_xticks(xx)
    ax.set_xticklabels(selected.channel, rotation=90, fontsize=7)
    ax.set_ylabel("Predicted strain (nanostrain)")
    ax.set_title(f"Hydraulic-to-cuboid strain comparison at {available:g} h")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "strain_channels_19h.png", dpi=250)
    plt.close(fig)

    # Figure 3: station-specific error at 19 hours.
    station_rows = []
    for station in stations.station.astype(str):
        sub = selected[selected.channel.str.endswith("_" + station)]
        station_rows.append({
            "station": station,
            "uniform_rmse_ns": rmse(sub.uniform_error_ns),
            "gradient_rmse_ns": rmse(sub.gradient_error_ns),
        })

    station_df = pd.DataFrame(station_rows)
    station_df.to_csv(out / "station_strain_errors_19h.csv", index=False)

    xx = np.arange(len(station_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        xx - width / 2,
        station_df.uniform_rmse_ns,
        width, label="Uniform"
    )
    ax.bar(
        xx + width / 2,
        station_df.gradient_rmse_ns,
        width, label="Mean + gradient"
    )

    ax.set_xticks(xx, station_df.station)
    ax.set_ylabel("Strain RMSE (nanostrain)")
    ax.set_title("Pressure-representation strain error by station at 19 h")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "station_strain_error_19h.png", dpi=250)
    plt.close(fig)

    summary = {
        "scope": "Same-model synthetic hydraulic-to-analytical strain comparison",
        "fine_splits": [args.fine_n, args.fine_n, 1],
        "coarse_splits": [3, 3, 1],
        "n_channels": len(kernel.channels),
        "n_times": len(times),
        "pressure_units": "Pa",
        "strain_units": "nanostrain",
        "hydraulic_axes_assumption": "local x/y = analytical xi/eta",
        "note": (
            "Refined mechanical tiles are a numerical reference, "
            "not a converged exact solution. "
            "Only saved hydraulic snapshots are compared. "
            "No field observations or Bayesian inversion."
        ),
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
