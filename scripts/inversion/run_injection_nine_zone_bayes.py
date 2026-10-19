
"""Nine-zone Bayesian pressure recovery from FD-generated strain.

Run from the AVANT repository root.
Fixed geometry and mechanics; no field observations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERM = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
ZCOLS = [
    f"zone_row{j}_col{i}_mpa"
    for j in range(3) for i in range(3)
]


def posterior(H, d, sigma_ns, prior_sd_pa):
    """Gaussian pressure posterior using stable linear solves."""
    n = H.shape[1]
    mu0 = np.zeros(n)
    C0 = np.eye(n) * prior_sd_pa**2
    R = np.eye(len(d)) * sigma_ns**2

    # Data-space formulation avoids forming H.T R^-1 H.
    S = H @ C0 @ H.T + R
    gain = np.linalg.solve(S, H @ C0).T

    mean = mu0 + gain @ (d - H @ mu0)
    cov = C0 - gain @ H @ C0
    cov = (cov + cov.T) / 2

    return mean, cov


def plot_map(values, title, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5.5, 6))
    im = ax.imshow(
        values.reshape(3, 3).T,
        origin="lower",
        extent=[-215, 215, -400, 400],
        vmin=vmin, vmax=vmax,
        cmap="viridis", aspect="equal"
    )
    for ix in range(3):
        for iy in range(3):
            ax.text(
                -215 + (ix + 0.5) * 430 / 3,
                -400 + (iy + 0.5) * 800 / 3,
                f"{values[ix*3+iy]:.3f}",
                ha="center", va="center",
                color="white", fontsize=9
            )
    ax.set_xlabel("Local x (m)")
    ax.set_ylabel("Local y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="MPa")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-sd-mpa", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(args.history)
    with np.load(args.kernel) as f:
        G = f["G_ns_per_pa"]
        B = f["basis"]
        training = f["training_channel_mask"].astype(bool)
        channels = f["channels"].astype(str)

    H = G @ B
    assert H.shape == (32, 9)
    assert np.allclose(B, np.eye(9))
    assert training.sum() == 24
    assert all(c in history for c in ZCOLS)

    time = history["time_h"].to_numpy(float)
    truth = history[ZCOLS].to_numpy(float)[:, PERM] * 1e6
    clean = truth @ H.T

    rng = np.random.default_rng(args.seed)
    prior_sd_pa = args.prior_sd_mpa * 1e6
    records = []
    saved = {}

    # Common time selection for both scenarios.
    targets = [5.0, 19.0, 19.5, 20.0, 24.0, 48.0, 96.0]

    for target in targets:
        idx = int(np.argmin(abs(time - target)))
        if abs(time[idx] - target) > 1e-8:
            raise ValueError(f"Missing requested time {target} h")

        ptrue = truth[idx]
        dtrue = clean[idx]

        for case, data_sd, assumed_sd in [
            ("noisefree", 0.0, 0.01),
            ("noisy", 2.0, 2.0),
        ]:
            obs = (
                dtrue.copy() if data_sd == 0
                else dtrue + rng.normal(0, data_sd, len(dtrue))
            )

            mean, cov = posterior(
                H[training], obs[training],
                assumed_sd, prior_sd_pa
            )

            sd = np.sqrt(np.maximum(np.diag(cov), 0))
            pred = H @ mean
            rmse_p = np.sqrt(np.mean((mean - ptrue)**2))
            train_rmse = np.sqrt(
                np.mean((pred[training]-obs[training])**2)
            )
            hold_rmse = np.sqrt(
                np.mean((pred[~training]-obs[~training])**2)
            )

            key = (target, case)
            saved[key] = (ptrue, mean, sd, cov, obs, pred)

            for zone in range(9):
                records.append({
                    "time_h": target,
                    "case": case,
                    "zone": zone,
                    "true_mpa": ptrue[zone]/1e6,
                    "mean_mpa": mean[zone]/1e6,
                    "sd_mpa": sd[zone]/1e6,
                    "lower95_mpa": (mean[zone]-1.96*sd[zone])/1e6,
                    "upper95_mpa": (mean[zone]+1.96*sd[zone])/1e6,
                    "pressure_rmse_mpa": rmse_p/1e6,
                    "training_rmse_ns": train_rmse,
                    "holdout_rmse_ns": hold_rmse,
                    "truth_in_95": bool(
                        abs(mean[zone]-ptrue[zone])
                        <= 1.96*sd[zone]
                    )
                })

    result = pd.DataFrame(records)
    result.to_csv(out / "pressure_posterior.csv", index=False)

    # Figures at the end of injection.
    for case in ("noisefree", "noisy"):
        ptrue, mean, sd, cov, obs, pred = saved[(19.0, case)]
        maximum = max(ptrue.max(), mean.max()) / 1e6

        plot_map(
            ptrue/1e6, "True pressure at 19 h",
            out / f"{case}_truth_19h.png",
            vmin=0, vmax=maximum
        )
        plot_map(
            mean/1e6, f"Recovered pressure: {case}",
            out / f"{case}_posterior_mean_19h.png",
            vmin=0, vmax=maximum
        )
        plot_map(
            sd/1e6, f"Posterior SD: {case}",
            out / f"{case}_posterior_sd_19h.png",
            vmin=0
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        z = np.arange(9)
        ax.errorbar(
            z, mean/1e6, yerr=1.96*sd/1e6,
            fmt="o", capsize=4, label="Posterior 95% interval"
        )
        ax.scatter(z, ptrue/1e6, marker="x",
                   s=80, label="Hydraulic truth")
        ax.set_xlabel("Analytical pressure zone")
        ax.set_ylabel("Pressure change (MPa)")
        ax.set_title(f"Nine-zone recovery at 19 h: {case}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"{case}_intervals_19h.png", dpi=250)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(obs[training], pred[training],
                   label="Training", alpha=0.8)
        ax.scatter(obs[~training], pred[~training],
                   label="Withheld", alpha=0.8)
        limits = [
            min(obs.min(), pred.min()),
            max(obs.max(), pred.max())
        ]
        ax.plot(limits, limits, "k--", label="1:1")
        ax.set_xlabel("Synthetic observed strain (ns)")
        ax.set_ylabel("Predicted strain (ns)")
        ax.set_title(f"Strain predictions: {case}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"{case}_strain_parity_19h.png", dpi=250)
        plt.close(fig)

        s = np.sqrt(np.diag(cov))
        corr = cov / np.outer(s, s)
        np.savetxt(
            out / f"{case}_pressure_correlation_19h.csv",
            corr, delimiter=","
        )

    # Compare center-zone pressure during injection and recovery.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time, truth[:, 4]/1e6,
            color="black", label="True central pressure")

    for case in ("noisefree", "noisy"):
        subset = result[
            (result["case"] == case) &
            (result["zone"] == 4)
        ].sort_values("time_h")
        ax.errorbar(
            subset.time_h, subset.mean_mpa,
            yerr=1.96*subset.sd_mpa,
            fmt="o-", capsize=3, label=case
        )

    ax.axvline(19, color="gray", linestyle="--")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Central-zone pressure (MPa)")
    ax.set_title("Central pressure recovery and uncertainty")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "central_pressure_recovery.png", dpi=250)
    plt.close(fig)

    summary = (
        result.groupby(["case", "time_h"])
        .agg(
            pressure_rmse_mpa=("pressure_rmse_mpa", "first"),
            training_rmse_ns=("training_rmse_ns", "first"),
            holdout_rmse_ns=("holdout_rmse_ns", "first"),
            zones_in_95=("truth_in_95", "sum")
        )
        .reset_index()
    )
    summary.to_csv(out / "recovery_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
