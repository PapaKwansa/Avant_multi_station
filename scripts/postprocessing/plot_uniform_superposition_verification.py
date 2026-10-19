
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

root = Path(
    "results/pressure_superposition/larry_network_001"
)

with open(root / "verification_summary.json") as f:
    data = json.load(f)

cases = data["cases"]

labels = [
    " x ".join(map(str, case["splits"]))
    for case in cases
]

errors = [
    case["max_abs_error_ns"]
    for case in cases
]

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(labels, errors)

ax.set_yscale("log")
ax.set_xlabel("Cuboid subdivision")
ax.set_ylabel("Maximum absolute strain difference (nanostrain)")
ax.set_title(
    "Uniform-Pressure Superposition Verification\n"
    "Original Cuboid vs. Sum of Subdivided Cuboids"
)

for i, value in enumerate(errors):
    ax.annotate(
        f"{value:.2e}",
        (i, value),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

output = root / "uniform_superposition_verification.png"
fig.savefig(output, dpi=300)
print("Saved:", output)

plt.show()
