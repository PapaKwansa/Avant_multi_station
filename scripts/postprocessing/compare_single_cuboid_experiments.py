
"""Compare noise-free and noisy single-cuboid Bayesian pressure recovery.

Uses saved inversion results. Does not rerun the forward model or inversion.
Run from the AVANT repository root.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

ROOT = Path("results/pressure_superposition")

EXPERIMENTS = {
    "Noise-free": ROOT / "single_noisefree_004",
    "Noisy (2 ns)": ROOT / "single_noisy_001",
}

OUT = ROOT / "single_cuboid_comparison"
OUT.mkdir(parents=True, exist_ok=True)

records = []
datasets = {}

for label, folder in EXPERIMENTS.items():
    posterior = pd.read_csv(
        folder / "synthetic_pressure_posterior.csv"
    )

    with open(folder / "synthetic_summary.json") as f:
        summary = json.load(f)

    predictions = pd.read_csv(
        folder / "synthetic_channel_predictions.csv"
    )

    if len(posterior) != 1:
        raise ValueError(
            f"{label}: expected one pressure coefficient"
        )

    row = posterior.iloc[0]

    records.append({
        "experiment": label,
        "true_mpa": row["true_pa"] / 1e6,
        "posterior_mean_mpa": row["posterior_mean_pa"] / 1e6,
        "posterior_sd_mpa": row["posterior_sd_pa"] / 1e6,
        "lower95_mpa": row["lower95_pa"] / 1e6,
        "upper95_mpa": row["upper95_pa"] / 1e6,
        "absolute_error_pa": abs(
            row["posterior_mean_pa"] - row["true_pa"]
        ),
        "training_rmse_ns": summary["training_rmse_ns"],
        "holdout_rmse_ns": summary["holdout_rmse_ns"],
        "generated_noise_ns": summary["synthetic_noise_std_ns"],
        "assumed_error_ns": summary["assumed_error_std_ns"],
    })

    datasets[label] = predictions

comparison = pd.DataFrame(records)

comparison.to_csv(
    OUT / "single_cuboid_comparison.csv",
    index=False
)

truth = comparison["true_mpa"].iloc[0]

if not np.allclose(comparison["true_mpa"], truth):
    raise ValueError("Experiments have different pressure truths")

fig, axes = plt.subplots(
    1, 3, figsize=(17, 5.2)
)

# --------------------------------------------------
# Panel A: Posterior probability densities
# --------------------------------------------------

ax = axes[0]

# Common pressure axis covering both posteriors
xmin = comparison["lower95_mpa"].min() - 0.005
xmax = comparison["upper95_mpa"].max() + 0.005

x = np.linspace(xmin, xmax, 2000)

for _, row in comparison.iterrows():
    density = norm.pdf(
        x,
        loc=row["posterior_mean_mpa"],
        scale=row["posterior_sd_mpa"]
    )

    ax.plot(
        x, density,
        linewidth=2,
        label=row["experiment"]
    )

ax.axvline(
    truth,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="True pressure = 1 MPa"
)

ax.set_xlabel("Pressure change (MPa)")
ax.set_ylabel("Posterior probability density (1/MPa)")
ax.set_title("(a) Pressure posterior distributions")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# --------------------------------------------------
# Panel B: Posterior means and credible intervals
# --------------------------------------------------

ax = axes[1]

for i, row in comparison.iterrows():
    mean = row["posterior_mean_mpa"]

    lower = mean - row["lower95_mpa"]
    upper = row["upper95_mpa"] - mean

    ax.errorbar(
        mean,
        i,
        xerr=np.array([[lower], [upper]]),
        fmt="o",
        capsize=5,
        markersize=7
    )

ax.axvline(
    truth,
    color="black",
    linestyle="--",
    label="True pressure"
)

ax.set_yticks(range(len(comparison)))
ax.set_yticklabels(comparison["experiment"])
ax.set_xlabel("Pressure change (MPa)")
ax.set_title("(b) Posterior mean and 95% intervals")
ax.grid(axis="x", alpha=0.2)
ax.legend(fontsize=8)

# --------------------------------------------------
# Panel C: Training and withheld strain RMSE
# --------------------------------------------------

ax = axes[2]

positions = np.arange(len(comparison))
width = 0.35

ax.bar(
    positions - width / 2,
    comparison["training_rmse_ns"],
    width,
    label="Training"
)

ax.bar(
    positions + width / 2,
    comparison["holdout_rmse_ns"],
    width,
    label="Withheld"
)

ax.set_xticks(positions)
ax.set_xticklabels(comparison["experiment"])
ax.set_ylabel("Strain RMSE (nanostrain)")
ax.set_title("(c) Strain prediction residuals")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.suptitle(
    "Single-Cuboid Bayesian Pressure Recovery\n"
    "Noise-Free vs. Noisy Synthetic Observations",
    fontsize=15
)

fig.tight_layout()

output = OUT / "single_cuboid_comparison.png"
fig.savefig(output, dpi=300, bbox_inches="tight")

print("\nCOMPARISON RESULTS")
print(comparison.to_string(index=False))
print("\nSaved figure:", output)

plt.show()
