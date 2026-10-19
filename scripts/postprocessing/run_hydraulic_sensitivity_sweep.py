"""12-scenario AVANT hydraulic sensitivity and analytical strain detectability.

Run from repository root; requires the existing run_injection_fd.py and the
nine-zone analytical sensitivity NPZ. No Bayesian inversion is performed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K_VALUES = [1e-3, 1e-4, 1e-5, 1e-6]
SS_VALUES = [1e-5, 1e-6, 1e-7]
PERMUTATION = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
ZONE_COLUMNS = [f'zone_row{iy}_col{ix}_mpa' for iy in range(3) for ix in range(3)]


def read_mechanics(path):
    with np.load(path) as f:
        G = np.asarray(f['G_ns_per_pa'], float)
        B = np.asarray(f['basis'], float)
        channels = [str(x) for x in f['channels']]
        mask = np.asarray(f['training_channel_mask'], bool)
    if G.shape != (32, 9) or B.shape != (9, 9) or mask.shape != (32,):
        raise ValueError('Unexpected analytical matrix or training-mask shape')
    if not np.allclose(B, np.eye(9)) or mask.sum() != 24:
        raise ValueError('Expected nine independent zones and 24 training channels')
    if not np.isfinite(G).all():
        raise ValueError('Nonfinite analytical sensitivity matrix')
    return G @ B, channels, mask


def make_plots(df, out, scenarios, noise):
    for metric, title, units, filename in [
        ('peak_strain_ns', 'Peak absolute predicted strain', 'nanostrain', 'peak_strain_heatmap.png'),
        ('peak_network_snr', 'Peak training-network SNR', 'dimensionless', 'network_snr_heatmap.png'),
        ('well_peak_kpa', 'Peak well-cell pressure rise', 'kPa', 'well_pressure_heatmap.png'),
        ('central_19h_kpa', 'Central-zone pressure at 19 h', 'kPa', 'central_pressure_heatmap.png'),
    ]:
        mat = np.array([[df.loc[(df.K_m_s == k) & (df.Ss_per_m == ss), metric].iloc[0]
                         for ss in SS_VALUES] for k in K_VALUES])
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.imshow(mat, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(SS_VALUES)), [f'{v:.0e}' for v in SS_VALUES])
        ax.set_yticks(range(len(K_VALUES)), [f'{v:.0e}' for v in K_VALUES])
        ax.set(xlabel='Specific storage Ss (1/m)', ylabel='Hydraulic conductivity K (m/s)', title=title)
        for i in range(len(K_VALUES)):
            for j in range(len(SS_VALUES)):
                ax.text(j, i, f'{mat[i,j]:.3g}', ha='center', va='center',
                        color='white' if mat[i,j] < (mat.min()+mat.max())/2 else 'black', fontsize=10)
        fig.colorbar(im, ax=ax, label=units)
        fig.savefig(out / filename, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for item in scenarios:
        h = item['history']
        ax.plot(h.time_h, h.well_cell_mpa * 1000, label=item['id'])
    ax.axvline(19, ls='--', color='black', lw=1, label='Shut-in')
    ax.set(xlabel='Time (hours)', ylabel='Well-cell pressure change (kPa)',
           title='30 gpm injection: hydraulic sensitivity')
    ax.legend(ncol=3, fontsize=7)
    ax.grid(alpha=.2)
    fig.savefig(out / 'well_pressure_histories.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for item in scenarios:
        ax.plot(item['history'].time_h, item['network_snr'], label=item['id'])
    ax.axhline(1, ls=':', color='red', label='SNR = 1')
    ax.axvline(19, ls='--', color='black', lw=1, label='Shut-in')
    ax.set(xlabel='Time (hours)', ylabel=f'Network SNR (noise SD = {noise:g} ns)',
           title='Training-network strain detectability')
    ax.legend(ncol=3, fontsize=7)
    ax.grid(alpha=.2)
    fig.savefig(out / 'network_snr_histories.png', dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(4, 3, figsize=(11, 14), constrained_layout=True)
    # Independent scales per panel deliberately show each spatial pattern;
    # each panel is annotated with its own maximum to avoid false comparisons.
    for item, ax in zip(scenarios, axs.flat):
        with np.load(item['path'] / 'hydraulic_fields.npz') as f:
            ts = np.asarray(f['times_h'])
            idx = int(np.argmin(abs(ts - 19.0)))
            if abs(ts[idx]-19.0) > 1e-8:
                raise ValueError('Missing 19 h snapshot')
            p = np.asarray(f['pressure_change_pa'][idx]) / 1000
            x = np.asarray(f['x_m']); y = np.asarray(f['y_m'])
        dx = x[1]-x[0]; dy = y[1]-y[0]
        extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]
        ax.imshow(p, origin='lower', extent=extent, cmap='viridis', aspect='auto', vmin=0)
        ax.plot(0, 0, '+', color='red', markersize=8)
        ax.set_title(f"{item['id']}\nmax {p.max():.3g} kPa", fontsize=9)
        ax.set_xlabel('x (m)', fontsize=8); ax.set_ylabel('y (m)', fontsize=8)
    fig.suptitle('19 h fine-grid pressure fields (individual panel scales)', fontsize=13)
    fig.savefig(out / 'pressure_fields_19h.png', dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pressure_superposition/experiments/injection_fd_30gpm.json')
    ap.add_argument('--kernel', default='results/pressure_superposition/nine_uniform_noisy_001/pressure_kernel_and_posterior.npz')
    ap.add_argument('--output-dir', default='results/pressure_superposition/hydraulic_sensitivity_001')
    ap.add_argument('--noise-ns', type=float, default=2.0)
    ap.add_argument('--limit', type=int, default=12, help='Optional first-N smoke test (1..12)')
    args = ap.parse_args()
    if args.noise_ns <= 0 or not 1 <= args.limit <= 12:
        ap.error('noise-ns must be positive and limit must be between 1 and 12')
    config_path = Path(args.config).resolve()
    kernel_path = Path(args.kernel).resolve()
    solver_path = Path('scripts/validation/run_injection_fd.py').resolve()
    if not solver_path.exists():
        raise FileNotFoundError(solver_path)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if (float(cfg['injection_rate_gpm']) != 30 or float(cfg['injection_hours']) != 19
            or float(cfg['total_hours']) != 96):
        raise ValueError('Expected 30 gpm, 19 h injection and 96 h total')
    H, channels, training = read_mechanics(kernel_path)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, scenarios = [], []
    combos = [(k, ss) for k in K_VALUES for ss in SS_VALUES][:args.limit]
    for k, ss in combos:
        ident = f'K{k:.0e}_Ss{ss:.0e}'
        case_dir = out / ident
        case_cfg = dict(cfg, K_m_s=k, Ss_per_m=ss)
        case_cfg_path = out / f'{ident}.json'
        case_cfg_path.write_text(json.dumps(case_cfg, indent=2), encoding='utf-8')
        if not (case_dir / 'summary.json').exists():
            print('Running', ident, flush=True)
            subprocess.run([sys.executable, str(solver_path), '--config', str(case_cfg_path),
                            '--output-dir', str(case_dir)], check=True,
                           stdout=(out / f'{ident}.log').open('w'), stderr=subprocess.STDOUT)
        history = pd.read_csv(case_dir / 'pressure_history.csv')
        balance = pd.read_csv(case_dir / 'mass_balance.csv')
        report = json.loads((case_dir / 'summary.json').read_text())
        if not set(ZONE_COLUMNS).issubset(history.columns):
            raise ValueError('Hydraulic zone columns missing')
        p_pa = history[ZONE_COLUMNS].to_numpy(float)[:, PERMUTATION] * 1e6
        strain = p_pa @ H.T
        net = np.linalg.norm(strain[:, training] / args.noise_ns, axis=1)
        holdout = np.linalg.norm(strain[:, ~training] / args.noise_ns, axis=1)
        t = history.time_h.to_numpy(float)
        ix19 = int(np.argmin(abs(t-19)))
        if abs(t[ix19]-19) > 1e-8:
            raise ValueError('No 19 h output')
        p19 = history.iloc[ix19][ZONE_COLUMNS].to_numpy(float).reshape(3, 3) * 1000
        rows.append(dict(scenario=ident, K_m_s=k, Ss_per_m=ss, diffusivity_m2_s=k/ss,
                         well_peak_kpa=float(history.well_cell_mpa.max()*1000),
                         central_19h_kpa=float(p19[1,1]),
                         peak_strain_ns=float(abs(strain).max()),
                         peak_network_snr=float(net.max()),
                         peak_holdout_snr=float(holdout.max()),
                         network_snr_19h=float(net[ix19]),
                         max_balance_residual_m3_s=float(report['max_mass_balance_residual_m3_s']),
                         injected_volume_m3=float(report['injected_volume_m3'])))
        scenarios.append(dict(id=ident, path=case_dir, history=history, network_snr=net))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'hydraulic_sensitivity_summary.csv', index=False)
    if args.limit == 12:
        make_plots(df, out, scenarios, args.noise_ns)
    print('\nSWEEP RESULTS\n', df.to_string(index=False, float_format=lambda x: f'{x:.5g}'))
    print('\nSaved:', out)
    if args.limit != 12:
        print('Smoke test only: run without --limit to produce the full 12-scenario plots.')


if __name__ == '__main__':
    main()
