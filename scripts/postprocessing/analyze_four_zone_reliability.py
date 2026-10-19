
"""Four-zone pressure correlations, contrasts, and repeated-noise tests.

Run from AVANT repository root.
Uses saved four-zone sensitivity matrix and posterior results.
Does not modify the analytical forward model.
"""

from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(
    "results/pressure_superposition/four_uniform_noisy_001"
)
OUT = ROOT / "reliability"
OUT.mkdir(parents=True, exist_ok=True)

N_REPEATS = 200
SEED = 2026
Z95 = 1.959963984540054

data = np.load(ROOT / "pressure_kernel_and_posterior.npz")
table = pd.read_csv(ROOT / "synthetic_pressure_posterior.csv")
summary = json.loads(
    (ROOT / "synthetic_summary.json").read_text()
)

G = np.asarray(data["G_ns_per_pa"], dtype=float)
B = np.asarray(data["basis"], dtype=float)
H = G @ B

training = np.asarray(
    data["training_channel_mask"], dtype=bool
)
A = H[training]

truth = table["true_pa"].to_numpy(float)
prior_mean = table["prior_mean_pa"].to_numpy(float)
prior_sd = table["prior_sd_pa"].to_numpy(float)
mean = table["posterior_mean_pa"].to_numpy(float)
C = np.asarray(data["posterior_covariance_pa2"], float)

noise_sd = float(summary["synthetic_noise_std_ns"])
assumed_sd = float(summary["assumed_error_std_ns"])

assert noise_sd > 0 and assumed_sd > 0
assert len(truth) == 4
assert A.shape[1] == 4
assert np.isfinite(A).all()

# -------------------------------------------
# 1. Posterior correlation matrix
# -------------------------------------------

sd = np.sqrt(np.diag(C))
corr = C / np.outer(sd, sd)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

for i in range(4):
    for j in range(4):
        ax.text(
            j, i, f"{corr[i,j]:+.2f}",
            ha="center", va="center", fontsize=11
        )

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels([f"P{i}" for i in range(4)])
ax.set_yticklabels([f"P{i}" for i in range(4)])
ax.set_title("Posterior Pressure Correlations")
fig.colorbar(im, ax=ax, label="Correlation")
fig.tight_layout()
fig.savefig(OUT / "pressure_correlation.png", dpi=300)
plt.close(fig)

# -------------------------------------------
# 2. Pairwise pressure contrasts
# -------------------------------------------

rows = []
for i, j in combinations(range(4), 2):
    difference = mean[i] - mean[j]
    variance = C[i,i] + C[j,j] - 2*C[i,j]
    contrast_sd = np.sqrt(max(variance, 0))

    rows.append({
        "pair": f"P{i} - P{j}",
        "mean_mpa": difference / 1e6,
        "sd_mpa": contrast_sd / 1e6,
        "lower95_mpa": (difference-Z95*contrast_sd)/1e6,
        "upper95_mpa": (difference+Z95*contrast_sd)/1e6,
    })

contrasts = pd.DataFrame(rows)
contrasts.to_csv(OUT / "pressure_contrasts.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(contrasts))

ax.errorbar(
    contrasts["mean_mpa"], y,
    xerr=[
        contrasts["mean_mpa"]-contrasts["lower95_mpa"],
        contrasts["upper95_mpa"]-contrasts["mean_mpa"]
    ],
    fmt="o", capsize=4
)
ax.axvline(0, color="black", linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels(contrasts["pair"])
ax.invert_yaxis()
ax.set_xlabel("Pressure difference (MPa)")
ax.set_title("Four-Zone Pressure Contrasts: 95% Intervals")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "pressure_contrasts.png", dpi=300)
plt.close(fig)

# -------------------------------------------
# 3. Repeated-noise Bayesian inversion
# -------------------------------------------

# Work in MPa for better numerical conditioning.
A_mpa = A * 1e6
truth_mpa = truth / 1e6
m0 = prior_mean / 1e6
s0 = prior_sd / 1e6

precision = (
    np.diag(1 / s0**2)
    + A_mpa.T @ A_mpa / assumed_sd**2
)

factor = cho_factor(precision, lower=True)
C_mpa = cho_solve(factor, np.eye(4))
posterior_sd_mpa = np.sqrt(np.diag(C_mpa))

rng = np.random.default_rng(SEED)
estimates = np.zeros((N_REPEATS, 4))

for r in range(N_REPEATS):
    observed = (
        A_mpa @ truth_mpa
        + rng.normal(0, noise_sd, A_mpa.shape[0])
    )

    rhs = (
        m0 / s0**2
        + A_mpa.T @ observed / assumed_sd**2
    )
    estimates[r] = cho_solve(factor, rhs)

errors = estimates - truth_mpa

lower = estimates - Z95*posterior_sd_mpa
upper = estimates + Z95*posterior_sd_mpa
covered = (lower <= truth_mpa) & (truth_mpa <= upper)

results = pd.DataFrame({
    "zone": range(4),
    "true_mpa": truth_mpa,
    "mean_estimate_mpa": estimates.mean(axis=0),
    "bias_mpa": errors.mean(axis=0),
    "rmse_mpa": np.sqrt(np.mean(errors**2, axis=0)),
    "posterior_sd_mpa": posterior_sd_mpa,
    "empirical_95_coverage": covered.mean(axis=0),
})

results.to_csv(
    OUT / "repeated_noise_summary.csv", index=False
)

np.savez_compressed(
    OUT / "repeated_noise_samples.npz",
    posterior_means_mpa=estimates,
    truth_mpa=truth_mpa,
    posterior_covariance_mpa2=C_mpa,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [estimates[:, j] for j in range(4)],
    positions=np.arange(4),
    tick_labels=[f"P{j}" for j in range(4)],
    showfliers=False,
)
ax.axhline(1, color="black", linestyle="--",
           label="True pressure = 1 MPa")
ax.set_ylabel("Recovered pressure (MPa)")
ax.set_title(f"Repeated-Noisy Pressure Recovery ({N_REPEATS} runs)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "repeated_noise_recovery.png", dpi=300)
plt.close(fig)

print("\nPOSTERIOR CORRELATIONS")
print(np.round(corr, 3))
print("\nPRESSURE CONTRASTS")
print(contrasts.round(4).to_string(index=False))
print("\nREPEATED-NOISE RESULTS")
print(results.round(4).to_string(index=False))
print("\nOverall pressure RMSE (MPa):",
      np.sqrt(np.mean(errors**2)))
print("Saved outputs to:", OUT)
