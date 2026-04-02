import time
import numpy as np
import matplotlib.pyplot as plt

import forward_model_multi_station
import multi_stations_input

# ===============================
# Load parameters
# ===============================

params = multi_stations_input.read_input()

pmax = params["pmax"]
tpeak = params["tpeak"]
d = params["d"]
time_arr = params["time"]
x_prime = params["x_prime"]
y_prime = params["y_prime"]
x0_prime = params["x0_prime"]
y0_prime = params["y0_prime"]
z = params["z"]
a_map = params["a_map"]
b_map = params["b_map"]
c_fixed = params["c_fixed"]
nu = params["nu"]
h = params["h"]
E_fixed = params["E_fixed"]
theta_map = params["theta_map"]
alpha = params["alpha"]
station_names = params["station_names"]

# ===============================
# Timing
# ===============================

n_runs = 100
times = []

for _ in range(n_runs):
    t0 = time.perf_counter()

    df = forward_model_multi_station(
        pmax=pmax, tpeak=tpeak, d=d, time=time_arr,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0_prime, y0_prime=y0_prime,
        z=z, a=a_map, b=b_map, c=c_fixed,
        nu=nu, h=h, E=E_fixed, theta_deg=theta_map,
        alpha=alpha, station_names=station_names,
    )

    times.append(time.perf_counter() - t0)

times = np.array(times)

mean_t = times.mean()
std_t = times.std()
median_t = np.median(times)

print(f"Mean runtime:   {mean_t:.6f} s")
print(f"Std runtime:    {std_t:.6f} s")
print(f"Median runtime: {median_t:.6f} s")

# ===============================
# Plot settings (nice)
# ===============================

plt.rcParams.update({
    "font.size": 16,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.dpi": 200
})

# ===============================
# Plot 1: Histogram
# ===============================

plt.figure(figsize=(8, 5))

plt.hist(times, bins=20, color="steelblue", edgecolor="black", alpha=0.8)

plt.axvline(mean_t, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_t:.4f}s")
plt.axvline(median_t, color="green", linestyle="-.", linewidth=2, label=f"Median = {median_t:.4f}s")

plt.xlabel("Runtime per forward model call (seconds)")
plt.ylabel("Frequency")
plt.title("Forward Model Runtime Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("runtime_histogram.png")
plt.close()

# ===============================
# Plot 2: Summary bar
# ===============================

plt.figure(figsize=(5, 5))

plt.bar(["Analytical model"], [mean_t], yerr=[std_t],
        color="steelblue", edgecolor="black", capsize=8)

plt.ylabel("Runtime (seconds)")
plt.title("Mean Runtime ± Std")

plt.tight_layout()
plt.savefig("runtime_summary.png")
plt.close()