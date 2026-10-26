#!/usr/bin/env python3
"""Validate Stage 6A physical/coordinate contract; create manifest + illustrative plot.

No forward-model solve, data fit, DREAM, or claims of Stage 6B readiness.
Run from repository root with installed -e . or configured PYTHONPATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from avant_model.inversion.stage6_parameters import Stage6Contract
from avant_model.model.stage6_coordinates import (
    global_to_local, local_to_global, source_corners_global,
    source_tile_centers_global,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)
    config_path = Path(args.config).resolve()
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Output directory exists: {out}')
    contract = Stage6Contract.from_json(config_path)
    reference = contract.reference()
    vec = contract.vector_from_state(reference)
    recovered = contract.state_from_vector(vec)
    assert reference == recovered
    tile_centers = source_tile_centers_global(reference.source, 6, 6)
    xx, yy = global_to_local(tile_centers[:, 0], tile_centers[:, 1], reference.source)
    xr, yr = local_to_global(xx, yy, reference.source)
    max_roundtrip = max(float(np.max(np.abs(xr-tile_centers[:,0]))),
                        float(np.max(np.abs(yr-tile_centers[:,1]))))
    if max_roundtrip > 1e-9:
        raise AssertionError('Global/source roundtrip precision failed')
    manifest = contract.manifest()
    manifest['source_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest['six_by_six_tile_center_roundtrip_error_m'] = max_roundtrip
    corners = source_corners_global(reference.source)
    closed = np.vstack([corners, corners[0]])
    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    xlo, xhi, ylo, yhi = contract.domain.extent()
    ax.plot([xlo,xhi,xhi,xlo,xlo], [ylo,ylo,yhi,yhi,ylo],
            color='#205a87', linewidth=2.4, label='Fixed global FV domain')
    ax.fill(closed[:,0], closed[:,1], color='#d8a44e', alpha=.25)
    ax.plot(closed[:,0], closed[:,1], color='#b27824', linewidth=2.4,
            label='Variable mechanical source (reference)')
    ax.scatter([contract.domain.well_x_prime_m], [contract.domain.well_y_prime_m],
               color='#b82929', marker='*', s=140, label='Fixed injection well', zorder=4)
    ax.scatter(tile_centers[:,0], tile_centers[:,1], marker='.', s=10, color='#937231',
               alpha=.65, label='Illustrative 6 x 6 tile centers')
    ax.set(xlabel="Global x' (m)", ylabel="Global y' (m)",
           title='Stage 6A coordinate contract (illustrative)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.20)
    ax.legend(fontsize=8, loc='upper right')
    out.mkdir(parents=True)
    (out / 'stage6a_contract_report.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'stage6a_reference_geometry.{extension}', dpi=240)
    plt.close(fig)
    print('Stage 6A contract accepted:', config_path)
    print('Free parameters:', ', '.join(contract.free_names))
    print('Reference K, Ss:', reference.K_m_s, reference.Ss_per_m)
    print('Global/source transform roundtrip max error (m):', max_roundtrip)
    print('Stage 6B generalized forward model NOT YET IMPLEMENTED')
    print('Saved:', out)
    return out


if __name__ == '__main__':
    main()
