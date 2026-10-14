
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path("results/pressure_superposition/larry_observed_fulltime_001")

data = np.load(
    ROOT / "four_zone_matrix_and_covariance.npz",
    allow_pickle=False,
)

cov = data["posterior_covariance_pa2"]

posterior = pd.read_csv(
    ROOT / "four_zone_pressure_posterior.csv"
)

rows = []

for time_index, group in posterior.groupby("time_index", sort=True):
    group = group.sort_values("zone")
    positions = np.where(data["time_indices"] == time_index)[0]

    if len(positions) != 1:
        raise ValueError(f"Invalid covariance mapping for time {time_index}")

    C = cov[positions[0]]
    means = group["mean_pa"].to_numpy()

    if len(means) != 4:
        raise ValueError("Expected four pressure zones")

    for i, j in itertools.combinations(range(4), 2):
        difference = means[i] - means[j]

        variance = C[i, i] + C[j, j] - 2 * C[i, j]
        sd = np.sqrt(max(variance, 0.0))

        rows.append({
            "time_index": time_index,
            "time_s": float(group["time_s"].iloc[0]),
            "zone_i": i,
            "zone_j": j,
            "difference_mpa": difference / 1e6,
            "sd_mpa": sd / 1e6,
            "lower95_mpa": (difference - 1.96 * sd) / 1e6,
            "upper95_mpa": (difference + 1.96 * sd) / 1e6,
        })

result = pd.DataFrame(rows)

output = ROOT / "pressure_contrast_summary.csv"
result.to_csv(output, index=False)

print("Saved:", output)
print("Number of contrasts:", len(result))
print("\nFinal-time pressure contrasts:")
print(
    result[result["time_index"] == result["time_index"].max()]
    .round(4)
    .to_string(index=False)
)
