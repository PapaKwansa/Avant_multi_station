import matplotlib.pyplot as plt

analytical_time = 0.742   # seconds
numerical_time = 1800     # seconds

labels = ["Analytical model", "Numerical model (COMSOL)"]
times = [analytical_time, numerical_time]

# ===============================
# Global styling
# ===============================
plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ===============================
# Figure
# ===============================
plt.figure(figsize=(7, 7))

plt.bar(
    labels,
    times,
    color=["steelblue", "darkred"],
    edgecolor="black",
    linewidth=2
)

plt.ylabel("Runtime per forward model (seconds)")
plt.title("Analytical vs Numerical Model Runtime", pad=18)

plt.yscale("log")

# ===============================
# Clean, bold annotations
# ===============================
def format_time(v):
    if v >= 60:
        return f"{v/60:.1f} min"
    return f"{v:.2f} s"

for i, v in enumerate(times):
    plt.text(
        i,
        v * 1.8,                 # pushes text farther away from bar
        format_time(v),
        ha='center',
        fontsize=16,
        fontweight='bold',
        clip_on=False            # prevents cutoff
    )

# ===============================
# Layout improvements
# ===============================
plt.ylim(0.1, 6000)             # extra headroom
plt.tight_layout()
plt.subplots_adjust(top=0.88)   # more breathing room above title

plt.savefig("runtime_comparison.eps", format="eps", bbox_inches="tight")
plt.show()
