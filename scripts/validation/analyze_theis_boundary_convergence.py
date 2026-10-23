
"""Boundary-size and numerical convergence study for AVANT hydraulic FV.

Run from repository root.

Uses the existing verified-formulation FV benchmark. All cases
are synthetic and use fixed-head external boundaries.
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCRIPT = (
    ROOT / "scripts" / "validation"
    / "verify_injection_against_theis.py"
)

spec = importlib.util.spec_from_file_location("theis_fv", SCRIPT)
theis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "domain_length_m": 2000.0,
        "K_m_s": 1e-6,
        "Ss_per_m": 1e-6,
        "thickness_m": 5.0,
        "rho_kg_m3": 1000.0,
        "gravity_m_s2": 9.81,
        "injection_rate_gpm": 30.0,
        "injection_hours": 19.0,
        "total_hours": 96.0,
    }

    # Odd grid sizes with nearly identical spacing (~20 m).
    # The largest case may require substantial RAM.
    cases = [
        (2000.0, 101),
        (4000.0, 201),
        (6000.0, 301),
    ]

    radii = [100.0, 200.0, 300.0]
    dt_h = 0.125

    all_results = []
    summaries = []

    for domain, n in cases:
        local = dict(cfg)
        local["domain_length_m"] = domain

        print(
            f"Running domain={domain:g} m, "
            f"grid={n}x{n}, dt={dt_h} h",
            flush=True,
        )

        frame, info = theis.solve_fd(
            n, dt_h, local, radii
        )

        frame["domain_length_m"] = domain
        all_results.append(frame)

        # Use identical comparison times for all cases.
        # Exclude near-zero reference pressures.
        valid = frame["theis_kpa"].abs() >= 0.01
        selected = frame.loc[valid]

        error = (
            selected["fd_kpa"]
            - selected["theis_kpa"]
        )

        summaries.append({
            "domain_length_m": domain,
            "grid_n": n,
            "dx_m": info["dx_m"],
            "runtime_s": info["runtime_s"],
            "rmse_kpa": float(np.sqrt(np.mean(error**2))),
            "max_abs_error_kpa": float(np.max(np.abs(error))),
            "mean_abs_relative_error_pct": float(
                np.mean(
                    np.abs(error)
                    / selected["theis_kpa"].abs()
                ) * 100
            ),
            "max_balance_residual_m3_s":
                info["max_balance_residual_m3_s"],
        })

    data = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summaries)

    data.to_csv(
        out / "boundary_comparison.csv",
        index=False,
    )
    summary.to_csv(
        out / "boundary_summary.csv",
        index=False,
    )

    # Figure 1: pressure histories by domain size.
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 11), sharex=True
    )

    for ax, radius in zip(axes, radii):
        for domain, _ in cases:
            sub = data[
                np.isclose(data.domain_length_m, domain)
                & np.isclose(data.requested_r_m, radius)
            ]

            ax.plot(
                sub.time_h,
                sub.fd_kpa,
                label=f"FV domain {domain/1000:g} km",
            )

        reference = data[
            np.isclose(data.domain_length_m, cases[-1][0])
            & np.isclose(data.requested_r_m, radius)
        ]

        ax.plot(
            reference.time_h,
            reference.theis_kpa,
            "k--",
            linewidth=2,
            label="Theis",
        )

        ax.axvline(19.0, color="gray", linestyle=":")
        ax.set_ylabel("Pressure change (kPa)")
        ax.set_title(f"Observation radius ~ {radius:g} m")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle(
        "Finite-domain boundary effects versus Theis"
    )
    fig.tight_layout()
    fig.savefig(
        out / "boundary_pressure_histories.png",
        dpi=250,
    )
    plt.close(fig)

    # Figure 2: overall error by domain size.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        summary.domain_length_m / 1000,
        summary.rmse_kpa,
        "o-",
        linewidth=2,
    )

    ax.set_xlabel("Square domain side length (km)")
    ax.set_ylabel("RMSE against Theis (kPa)")
    ax.set_title("Boundary-size sensitivity")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        out / "boundary_error_convergence.png",
        dpi=250,
    )
    plt.close(fig)

    print("\nBOUNDARY SENSITIVITY SUMMARY")
    print(summary.to_string(index=False))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
