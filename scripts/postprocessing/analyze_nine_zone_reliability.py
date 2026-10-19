
"""Repeated-noise reliability analysis for nine pressure regions.

Run from the AVANT repository root.
Reuses saved analytical sensitivities; does not rerun mechanics.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(
    "results/pressure_superposition/nine_uniform_noisy_001"
)
OUT = ROOT / "reliability"
OUT.mkdir(parents=True, exist_ok=True)

N_REPEATS = 200
SEED = 2026
Z95 = 1.959963984540054

# Load the saved inversion
data = np.load(ROOT / "pressure_kernel_and_posterior.npz")
posterior = pd.read_csv(
    ROOT / "synthetic_pressure_posterior.csv"
)
summary = json.loads(
    (ROOT / "synthetic_summary.json").read_text()
)

G = np.asarray(data["G_ns_per_pa"], float)
B = np.asarray(data["basis"], float)
mask = np.asarray(
    data["training_channel_mask"], bool
)

H = G @ B
A = H[mask]

truth = posterior["true_pa"].to_numpy(float) / 1e6
prior_mean = (
    posterior["prior_mean_pa"].to_numpy(float) / 1e6
)
prior_sd = (
    posterior["prior_sd_pa"].to_numpy(float) / 1e6
)

noise_sd = float(summary["synthetic_noise_std_ns"])
assumed_sd = float(summary["assumed_error_std_ns"])

n = len(truth)
assert n == 9
assert A.shape[1] == n
assert np.isfinite(A).all()
assert noise_sd > 0 and assumed_sd > 0

# Convert sensitivity to nanostrain per MPa
A = A * 1e6

# Gaussian posterior covariance
precision = (
    np.diag(1.0 / prior_sd**2)
    + A.T @ A / assumed_sd**2
)

factor = cho_factor(precision, lower=True)
C = cho_solve(factor, np.eye(n))
sd = np.sqrt(np.diag(C))

# Repeated synthetic observations
rng = np.random.default_rng(SEED)
estimates = np.empty((N_REPEATS, n))

for r in range(N_REPEATS):
    observed = (
        A @ truth
        + rng.normal(0, noise_sd, A.shape[0])
    )

    rhs = (
        prior_mean / prior_sd**2
        + A.T @ observed / assumed_sd**2
    )

    estimates[r] = cho_solve(factor, rhs)

errors = estimates - truth

# Empirical credible interval coverage
lower = estimates - Z95 * sd
upper = estimates + Z95 * sd

covered = (
    (lower <= truth)
    & (truth <= upper)
)

results = pd.DataFrame({
    "zone": np.arange(n),
    "true_mpa": truth,
    "mean_estimate_mpa": estimates.mean(axis=0),
    "bias_mpa": errors.mean(axis=0),
    "rmse_mpa": np.sqrt(np.mean(errors**2, axis=0)),
    "posterior_sd_mpa": sd,
    "empirical_95_coverage": covered.mean(axis=0),
})

results.to_csv(
    OUT / "repeated_noise_summary.csv",
    index=False
)

np.savez_compressed(
    OUT / "repeated_noise_samples.npz",
    posterior_means_mpa=estimates,
    truth_mpa=truth,
    posterior_covariance_mpa2=C,
)

# Figure 1: Posterior correlation matrix
corr = C / np.outer(sd, sd)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(
    corr, cmap="coolwarm", vmin=-1, vmax=1
)

for i in range(n):
    for j in range(n):
        ax.text(
            j, i, f"{corr[i,j]:+.2f}",
            ha="center", va="center", fontsize=8
        )

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([f"P{i}" for i in range(n)])
ax.set_yticklabels([f"P{i}" for i in range(n)])
ax.set_title("Nine-Zone Posterior Pressure Correlations")
fig.colorbar(im, ax=ax, label="Correlation")
fig.tight_layout()
fig.savefig(OUT / "pressure_correlation.png", dpi=300)
plt.close(fig)

# Figure 2: Repeated-noise recovery
fig, ax = plt.subplots(figsize=(11, 5))

ax.boxplot(
    [estimates[:, j] for j in range(n)],
    tick_labels=[f"P{j}" for j in range(n)],
    showfliers=False
)

ax.axhline(
    1.0, color="black", linestyle="--",
    label="True pressure = 1 MPa"
)
ax.set_ylabel("Recovered pressure (MPa)")
ax.set_title(
    f"Nine-Zone Repeated-Noisy Recovery ({N_REPEATS} runs)"
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "repeated_noise_recovery.png", dpi=300)
plt.close(fig)

# Figure 3: Empirical coverage
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    np.arange(n),
    results["empirical_95_coverage"] * 100
)
ax.axhline(
    95, color="black", linestyle="--",
    label="Nominal 95%"
)
ax.set_xticks(np.arange(n))
ax.set_xticklabels([f"P{j}" for j in range(n)])
ax.set_ylabel("Empirical coverage (%)")
ax.set_ylim(0, 105)
ax.set_title("Marginal 95% Interval Coverage")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "empirical_coverage.png", dpi=300)
plt.close(fig)

# Figure 4: Prior-whitened sensitivity modes
weighted_A = (
    A * prior_sd[np.newaxis, :] / assumed_sd
)
singular_values = np.linalg.svd(
    weighted_A, compute_uv=False
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(
    np.arange(1, n+1),
    singular_values, "o-"
)
ax.axhline(
    1, color="black", linestyle="--",
    label="Prior-information threshold"
)
ax.set_xlabel("Singular mode")
ax.set_ylabel("Prior-whitened singular value")
ax.set_title("Nine-Zone Pressure Identifiability")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "singular_modes.png", dpi=300)
plt.close(fig)

print("\nNINE-ZONE REPEATED-NOISE RESULTS")
print(results.round(4).to_string(index=False))

print(
    "\nOverall pressure RMSE (MPa):",
    np.sqrt(np.mean(errors**2))
)
print(
    "Mean empirical 95% coverage:",
    covered.mean()
)
print(
    "Modes with singular value > 1:",
    np.sum(singular_values > 1),
    "/", n
)
print("\nSaved outputs to:", OUT)
