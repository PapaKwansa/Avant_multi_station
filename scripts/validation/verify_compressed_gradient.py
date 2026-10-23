"""Verify compressed mean-plus-gradient operator against explicit cuboid superposition.

Run from the AVANT repository root. Uses the existing analytical kernel unchanged.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from avant_model.model.pressure_superposition import (
    Cuboid, ElasticProperties, build_pressure_kernel,
)


def gradient_mapping(tiles, fine_n, coarse_n, lx, ly):
    """Map [mean Pa, gx Pa/m, gy Pa/m] for each x-major zone to tile Pa."""
    if fine_n % coarse_n:
        raise ValueError('fine_n must be divisible by coarse_n')
    A = np.zeros((len(tiles), 3 * coarse_n * coarse_n))
    factor = fine_n // coarse_n
    for k, tile in enumerate(tiles):
        ix = tile.ix // factor
        iy = tile.iy // factor
        zone = ix * coarse_n + iy
        cx = -lx / 2 + (ix + .5) * lx / coarse_n
        cy = -ly / 2 + (iy + .5) * ly / coarse_n
        A[k, 3 * zone:3 * zone + 3] = (1., tile.xi - cx, tile.eta - cy)
    return A


def fit_coefficients(field, x, y, coarse_n):
    """Independent centered least-squares fit for each hydraulic zone."""
    ny, nx = field.shape
    if ny % coarse_n or nx % coarse_n:
        raise ValueError('hydraulic grid must divide into coarse zones')
    sy, sx = ny // coarse_n, nx // coarse_n
    coeff = np.empty(3 * coarse_n * coarse_n)
    for ix in range(coarse_n):
        for iy in range(coarse_n):
            xs = slice(ix * sx, (ix + 1) * sx)
            ys = slice(iy * sy, (iy + 1) * sy)
            xx, yy = np.meshgrid(x[xs], y[ys])
            p = field[ys, xs].ravel()
            design = np.column_stack((np.ones(p.size), (xx - xx.mean()).ravel(), (yy - yy.mean()).ravel()))
            zone = ix * coarse_n + iy
            coeff[3 * zone:3 * zone + 3] = np.linalg.lstsq(design, p, rcond=None)[0]
    return coeff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fields', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--fine-n', type=int, default=18)
    ap.add_argument('--coarse-n', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10000)
    args = ap.parse_args()
    if args.fine_n % args.coarse_n:
        raise ValueError('fine-n must divide evenly by coarse-n')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    stations = pd.read_csv(cfg['station_file'])
    geom = Cuboid(**cfg['geometry'])
    elastic = ElasticProperties(**cfg['elastic'])
    print('Building analytical kernel...', flush=True)
    t0 = time.perf_counter()
    kernel = build_pressure_kernel(geom, elastic,
        x_prime=stations.x.to_numpy(float), y_prime=stations.y.to_numpy(float),
        z=stations.z.to_numpy(float), station_names=stations.station.astype(str).tolist(),
        splits=(args.fine_n, args.fine_n, 1), components=tuple(cfg['components']))
    build_s = time.perf_counter() - t0
    with np.load(args.fields) as f:
        x, y = np.asarray(f['x_m'], float), np.asarray(f['y_m'], float)
        times = np.asarray(f['times_h'], float)
        fields = np.asarray(f['pressure_change_pa'], float)
    dx, dy = x[1]-x[0], y[1]-y[0]
    lx, ly = x[-1]-x[0]+dx, y[-1]-y[0]+dy
    if not np.isclose(lx, 2*geom.a) or not np.isclose(ly, 2*geom.b):
        raise ValueError('Hydraulic domain does not match analytical cuboid footprint')
    A = gradient_mapping(kernel.tiles, args.fine_n, args.coarse_n, lx, ly)
    G = np.asarray(kernel.G, float)
    H = G @ A
    rows = []
    coeffs = []
    for t, field in zip(times, fields):
        q = fit_coefficients(field, x, y, args.coarse_n)
        explicit = G @ (A @ q)
        compressed = H @ q
        err = np.max(np.abs(explicit-compressed))
        rows.append({'time_h': float(t), 'max_abs_difference_ns': float(err),
                     'rms_difference_ns': float(np.sqrt(np.mean((explicit-compressed)**2)))})
        coeffs.append(q)
    results = pd.DataFrame(rows)
    results.to_csv(out/'compression_verification.csv', index=False)
    q0 = np.asarray(coeffs)[np.argmin(np.abs(times-19.))]
    def direct():
        return G @ (A @ q0)
    def fast():
        return H @ q0
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        direct()
    direct_s = time.perf_counter()-t0
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        fast()
    compressed_s = time.perf_counter()-t0
    summary = {
        'scope': 'Algebraic compression of the existing analytical cuboid kernel; synthetic test only',
        'fine_n': args.fine_n, 'coarse_n': args.coarse_n,
        'G_shape': list(G.shape), 'A_shape': list(A.shape), 'H_shape': list(H.shape),
        'kernel_build_seconds': build_s,
        'max_abs_verification_difference_ns': float(results.max_abs_difference_ns.max()),
        'benchmark_repeats': args.repeats,
        'direct_seconds': direct_s, 'compressed_seconds': compressed_s,
        'speedup_pressure_to_strain_only': direct_s/compressed_s if compressed_s else None,
        'warning': 'Excludes hydraulic solve, regional least-squares fit, and kernel rebuild for changed geometry or elastic properties.'
    }
    np.savez_compressed(out/'compressed_gradient_operator.npz', H_ns_per_coeff=H,
        A_tile_pressure_per_coeff=A, channels=np.asarray(kernel.channels),
        times_h=times, coefficients=np.asarray(coeffs))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
