"""Verify a 2-D confined finite-volume injection solver against Theis.

Run from repository root:
 python scripts/validation/verify_injection_against_theis.py --output-dir results/pressure_superposition/theis_verification_003

Theis assumes an infinite, homogeneous confined aquifer and a line source.
The FV model has a finite square domain, fixed-head boundaries, and a cell source.
Comparisons are at fixed physical radii and common half-hour output times.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import splu
from scipy.special import exp1

GPM_TO_M3S = 0.003785411784 / 60.0


def theis_pressure(r, t_h, K, Ss, b, q, rho, g, shutin_h):
    """Pressure change (Pa) from Theis injection plus shut-in superposition."""
    t = np.atleast_1d(np.asarray(t_h, dtype=float)) * 3600.0
    r = float(r)
    if r <= 0 or min(K, Ss, b) <= 0:
        raise ValueError('Radius, K, Ss and thickness must be positive')
    if shutin_h <= 0:
        raise ValueError('Shut-in time must be positive')
    D = K / Ss
    T = K * b
    answer = np.zeros_like(t)
    active = t > 0
    answer[active] = exp1(r**2 / (4.0 * D * t[active]))
    stopped = t > shutin_h * 3600.0
    if np.any(stopped):
        elapsed = t[stopped] - shutin_h * 3600.0
        answer[stopped] -= exp1(r**2 / (4.0 * D * elapsed))
    return rho * g * q / (4.0 * np.pi * T) * answer


def solve_fd(n, dt_h, cfg, radii):
    """Cell-centered FV, backward Euler, fixed-head boundaries.

    Interpolate the computed head along y=0 at the exact requested radii.
    Report every 0.5 hours regardless of the internal timestep.
    """
    L = float(cfg['domain_length_m'])
    K = float(cfg['K_m_s'])
    Ss = float(cfg['Ss_per_m'])
    b = float(cfg['thickness_m'])
    rho = float(cfg['rho_kg_m3'])
    g = float(cfg['gravity_m_s2'])
    q = float(cfg['injection_rate_gpm']) * GPM_TO_M3S
    shutin = float(cfg['injection_hours'])
    total = float(cfg['total_hours'])
    dt_h = float(dt_h)
    requested = np.atleast_1d(np.asarray(radii, dtype=float))
    if n < 5 or n % 2 != 1:
        raise ValueError('Grid size must be odd and at least 5')
    if min(L, K, Ss, b, rho, g, q, dt_h) <= 0 or not 0 < shutin < total:
        raise ValueError('Invalid positive parameters or injection schedule')
    for value in (total / dt_h, shutin / dt_h, 0.5 / dt_h):
        if not np.isclose(value, round(value), rtol=0, atol=1e-9):
            raise ValueError('dt must divide total time, shut-in time, and 0.5 h')
    dx = L / n
    if np.any(~np.isfinite(requested)) or np.any(requested <= 0) or np.any(requested >= (n//2 - 1) * dx):
        raise ValueError('Observation radii outside interpolation region')
    storage = Ss * dx**2 * b
    trans = K * b
    size = n * n

    def index(i, j):
        return j * n + i

    A = lil_matrix((size, size), dtype=float)
    for j in range(n):
        for i in range(n):
            k = index(i, j)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    A[k, k] += trans
                    A[k, index(ii, jj)] -= trans
                else:
                    A[k, k] += 2.0 * trans
    A = A.tocsc()
    coefficient = storage / (dt_h * 3600.0)
    lu = splu(A + diags(np.full(size, coefficient), format='csc'))
    head = np.zeros(size)
    source = np.zeros(size)
    source[index(n // 2, n // 2)] = q
    x_centers = (np.arange(n) - n // 2) * dx
    j_center = n // 2
    output_stride = int(round(0.5 / dt_h))
    records = []
    max_balance = 0.0
    start = time.perf_counter()

    for step in range(1, int(round(total / dt_h)) + 1):
        t_h = step * dt_h
        injection = q if t_h <= shutin + 1e-10 else 0.0
        old = head
        head = lu.solve(coefficient * old + (source if injection else 0.0))
        storage_rate = coefficient * np.sum(head - old)
        boundary_rate = float(np.sum(A @ head))
        max_balance = max(max_balance, abs(injection - storage_rate - boundary_rate))
        if step % output_stride:
            continue
        row = head.reshape(n, n)[j_center, :]
        pressure_fv_kpa = np.interp(requested, x_centers, row) * rho * g / 1000.0
        for r, p_fd in zip(requested, pressure_fv_kpa):
            p_ref = float(theis_pressure(r, [t_h], K, Ss, b, q, rho, g, shutin)[0] / 1000.0)
            records.append(dict(grid_n=n, dt_h=dt_h, time_h=round(t_h, 8),
                                requested_r_m=float(r), actual_r_m=float(r),
                                fd_kpa=float(p_fd), theis_kpa=p_ref,
                                absolute_error_kpa=abs(float(p_fd) - p_ref)))
    return pd.DataFrame(records), dict(grid_n=n, dx_m=dx, dt_h=dt_h,
                                       runtime_s=time.perf_counter() - start,
                                       max_balance_residual_m3_s=max_balance,
                                       actual_observation_radii_m=requested.tolist())


def comparison_rmse(a, b):
    keys = ['requested_r_m', 'time_h']
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not np.allclose(a[keys], b[keys], atol=1e-8, rtol=0):
        raise ValueError('Comparison locations or times differ')
    return float(np.sqrt(np.mean((a.fd_kpa.to_numpy() - b.fd_kpa.to_numpy())**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--quick', action='store_true', help='Run smaller grids for a quick diagnostic')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(domain_length_m=4000.0, K_m_s=1e-6, Ss_per_m=1e-6,
               thickness_m=5.0, rho_kg_m3=1000.0, gravity_m_s2=9.81,
               injection_rate_gpm=30.0, injection_hours=19.0, total_hours=96.0)
    radii = [100.0, 200.0, 300.0]
    cases = ([(101, 0.125), (201, 0.125), (301, 0.125),
              (201, 0.5), (201, 0.25), (201, 0.0625)]
             if not args.quick else
             [(51, 0.125), (101, 0.125), (151, 0.125),
              (101, 0.5), (101, 0.25), (101, 0.0625)])
    frames, summaries = {}, []
    for n, dt in cases:
        print(f'Running {n} x {n}, dt={dt:g} h', flush=True)
        frame, info = solve_fd(n, dt, cfg, radii)
        frames[(n, dt)] = frame
        # Identical comparison set in every experiment.
        valid = frame.theis_kpa.abs() >= 0.01
        sub = frame.loc[valid]
        diff = sub.fd_kpa - sub.theis_kpa
        info['n_comparison_points'] = len(sub)
        info['rmse_kpa'] = float(np.sqrt(np.mean(diff**2)))
        info['mean_abs_relative_error_pct'] = float(np.mean(np.abs(diff) / np.abs(sub.theis_kpa) * 100))
        summaries.append(info)
    data = pd.concat(frames.values(), ignore_index=True)
    summary = pd.DataFrame(summaries)
    data.to_csv(out / 'theis_comparison.csv', index=False)
    summary.to_csv(out / 'convergence_summary.csv', index=False)
    (out / 'benchmark_config.json').write_text(json.dumps(dict(cfg, radii_m=radii, cases=cases,
                         common_output_dt_h=0.5), indent=2), encoding='utf-8')

    center_grid = 201 if not args.quick else 101
    grids = [101, 201, 301] if not args.quick else [51, 101, 151]
    pairs = [((grids[0], 0.125), (grids[1], 0.125), 'spatial'),
             ((grids[1], 0.125), (grids[2], 0.125), 'spatial'),
             ((center_grid, 0.5), (center_grid, 0.25), 'temporal'),
             ((center_grid, 0.25), (center_grid, 0.125), 'temporal'),
             ((center_grid, 0.125), (center_grid, 0.0625), 'temporal')]
    rows = []
    for coarse, fine, kind in pairs:
        rows.append(dict(type=kind, coarse_grid=coarse[0], coarse_dt_h=coarse[1],
                         fine_grid=fine[0], fine_dt_h=fine[1],
                         solution_difference_rmse_kpa=comparison_rmse(frames[coarse], frames[fine])))
    self_conv = pd.DataFrame(rows)
    self_conv.to_csv(out / 'numerical_self_convergence.csv', index=False)

    finest = frames[(center_grid, 0.0625)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    for ax, r in zip(axes, radii):
        sub = finest[np.isclose(finest.requested_r_m, r)]
        ax.plot(sub.time_h, sub.theis_kpa, 'k-', label='Theis')
        ax.plot(sub.time_h, sub.fd_kpa, '--', label='Finite volume')
        ax.axvline(19, color='gray', ls=':', label='Shut-in')
        ax.set(title=f'Observation radius = {r:g} m', ylabel='Pressure change (kPa)')
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel('Time (hours)')
    fig.suptitle('Theis verification: injection and recovery')
    fig.savefig(out / 'theis_pressure_histories.png', dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for ax, r in zip(axes, radii):
        sub = finest[np.isclose(finest.requested_r_m, r)]
        ax.plot(sub.time_h, sub.fd_kpa - sub.theis_kpa)
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(19, color='gray', ls=':')
        ax.set(title=f'Observation radius = {r:g} m', ylabel='FV - Theis (kPa)')
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('Time (hours)')
    fig.suptitle('Pressure error versus time')
    fig.savefig(out / 'theis_pressure_errors.png', dpi=220)
    plt.close(fig)

    for kind, subset, xkey, xlabel, filename, title in [
        ('spatial', summary[np.isclose(summary.dt_h, 0.125)], 'dx_m', 'Grid spacing (m)',
         'spatial_convergence.png', 'Spatial refinement at dt = 0.125 h'),
        ('temporal', summary[summary.grid_n == center_grid], 'dt_h', 'Time step (hours)',
         'temporal_convergence.png', f'Temporal refinement on {center_grid} x {center_grid} grid')]:
        subset = subset.sort_values(xkey)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(subset[xkey], subset.rmse_kpa, 'o-', label='RMSE versus Theis')
        ax.set(xlabel=xlabel, ylabel='RMSE versus Theis (kPa)', title=title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, kind in zip(axes, ('spatial', 'temporal')):
        subset = self_conv[self_conv.type == kind].reset_index(drop=True)
        labels = ([f'{int(a)} to {int(b)}' for a, b in zip(subset.coarse_grid, subset.fine_grid)]
                  if kind == 'spatial' else
                  [f'{a:g} to {b:g}' for a, b in zip(subset.coarse_dt_h, subset.fine_dt_h)])
        ax.plot(np.arange(len(subset)), subset.solution_difference_rmse_kpa, 'o-')
        ax.set_xticks(np.arange(len(subset)), labels)
        ax.set(title=f'{kind.capitalize()} self-convergence', ylabel='Successive-solution RMSE (kPa)',
               xlabel='Grid size pair' if kind == 'spatial' else 'Timestep pair (h)')
        ax.grid(alpha=0.25)
    fig.savefig(out / 'numerical_self_convergence.png', dpi=220)
    plt.close(fig)
    print('\nCONVERGENCE SUMMARY\n' + summary.to_string(index=False))
    print('\nSELF-CONVERGENCE\n' + self_conv.to_string(index=False))
    print('\nSaved:', out)


if __name__ == '__main__':
    main()
