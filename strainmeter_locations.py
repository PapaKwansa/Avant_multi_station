# ==========================================================
# Plan-view strain contours + ALL AVANT stations + inclusion
# ==========================================================
import os
os.environ["MPLBACKEND"] = "Agg"  # for headless environments
os.environ["OMP_NUM_THREADS"] = "1"  # limit threads for numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as mpi
import multi_station_strain as sf

# ---------------- Load parameters ----------------
params = mpi.read_input()

pmax = params['pmax']
tpeak = params['tpeak']
d = params['d']
time_vals = np.asarray(params['time'])

x0_prime = params['x0_prime']
y0_prime = params['y0_prime']
theta_deg_init = params['theta_deg']
z0 = params['z']          # depth of inclusion center (if used)
a = params['a']
b = params['b']
c = params['c']
nu = params['nu']
h = params['h']
E = params['E']
alpha = params.get('alpha', None)

# Set globals for strain functions (as you had before)
sf.nu = nu
sf.E = E
sf.alpha = alpha

# Characteristic strain
ec = sf.charac_strain(sf.linear_trans(alpha, nu, pmax, E), nu)

# ---------------- Load AVANT stations ----------------
stations_df = pd.read_csv('AVANT_stations.csv')

station_names = stations_df['station'].astype(str).str.strip().values
x_stations = stations_df['x_prime'].values
y_stations = stations_df['y_prime'].values
z_stations = stations_df['depth'].values

Ns = len(station_names)
print(f"[INFO] Loaded {Ns} AVANT stations: {station_names}")

# ==========================================================
# (1) SINGLE MAP: ALL STRAINMETERS + INCLUSION
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 7))

# Plot all strainmeter locations
ax.scatter(x_stations, y_stations, c='blue', s=60, label='Strainmeters')

# Annotate each station
for i, name in enumerate(station_names):
    ax.text(x_stations[i] + 5, y_stations[i] + 5, name, fontsize=9)

# Plot inclusion (source) location
ax.scatter(x0_prime, y0_prime, c='red', s=150, marker='*', label='Inclusion')

ax.set_xlabel("x' (m)")
ax.set_ylabel("y' (m)")
ax.set_title("Avant Field: Strainmeter Locations Relative to Inclusion")
ax.set_aspect('equal')
ax.legend()

plt.tight_layout()
plt.savefig("Avant_strainmeters_relative_to_inclusion.png", dpi=300)
plt.close()

print("[INFO] Saved: Avant_strainmeters_relative_to_inclusion.png")

# ==========================================================
