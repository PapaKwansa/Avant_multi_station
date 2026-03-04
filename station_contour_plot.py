# ==========================================================
# Full strain tensor mini-contours around each AVANT station
# at a SPECIFIED TIME during injection
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import multi_stations_input as mpi
import multi_station_strain as sf

# ==========================================================
# USER SETTINGS
# ==========================================================
TIME_DAY = 8.0      # <-- time of interest (days)
LOCAL_GRID = 200    # meters around each station
GRID_RES = 60       # contour resolution

# ==========================================================
# LOAD PARAMETERS
# ==========================================================
params = mpi.read_input()

pmax = params['pmax']
tpeak = params['tpeak']
d = params['d']
time_vals = params['time']

x_prime = params['x_prime']
y_prime = params['y_prime']
z = params['z']            # TRUE DEPTHS of each station
x0_prime = params['x0_prime']
y0_prime = params['y0_prime']
theta_deg = params['theta_deg']
a = params['a']
b = params['b']
c = params['c']
nu = params['nu']
h = params['h']
E = params['E']
alpha = params.get('alpha', None)

# Set globals for strain functions
sf.nu = nu
sf.E = E
sf.alpha = alpha

# Load stations
stations_df = pd.read_csv('AVANT_stations.csv')
station_names = stations_df['station'].astype(str).str.strip().values

print(f"[INFO] Loaded {len(station_names)} AVANT stations: {station_names}")

# ==========================================================
# PICK PRESSURE AT SELECTED TIME
# ==========================================================
# Interpolate pressure time series
def pressure_time_series(pmax, tpeak, d, time):
    t = np.asarray(time)
    p = np.zeros_like(t, dtype=float)
    rising = t <= tpeak
    falling = t > tpeak
    p[rising] = (pmax / tpeak) * t[rising]
    p[falling] = pmax * np.exp(-d * (t[falling] / tpeak - 1))
    return p

p_series = pressure_time_series(pmax, tpeak, d, time_vals)
p_at_time = np.interp(TIME_DAY, time_vals, p_series)

# Compute characteristic strain at this time
lt = sf.linear_trans(alpha, nu, p_at_time, E)
ec = sf.charac_strain(lt, nu)

print(f"[INFO] Using pressure = {p_at_time/1e6:.2f} MPa at t = {TIME_DAY} days")

# ==========================================================
# HELPER: coordinate transform (primed -> unprimed)
# ==========================================================
def compute_relative_xyz(xp, yp, x0p, y0p, theta_deg, z_station, h):
    theta = np.radians(theta_deg)
    dxp = xp - x0p
    dyp = yp - y0p

    x = dxp * np.cos(theta) + dyp * np.sin(theta)
    y = -dxp * np.sin(theta) + dyp * np.cos(theta)
    z = z_station - h   # depth relative to inclusion top

    return x, y, z

# ==========================================================
# SET UP 3x2 FIGURE FOR FULL TENSOR
# ==========================================================
fig, axes = plt.subplots(3, 2, figsize=(10, 11), dpi=150)

comp_labels = ["Epsilon_XX", "Epsilon_YY", "Epsilon_ZZ",
               "Epsilon_XY", "Epsilon_XZ", "Epsilon_YZ"]

comp_indices = [(0,0), (1,1), (2,2),
                (0,1), (0,2), (1,2)]

# Flatten axes for easy looping
axes = axes.ravel()

all_contours = []

# ==========================================================
# LOOP OVER TENSOR COMPONENTS
# ==========================================================
for ax, (label, idx) in zip(axes, zip(comp_labels, comp_indices)):

    # For each component, overlay all station mini-contours
    for xs, ys, zs, st_name in zip(x_prime, y_prime, z, station_names):

        # Local grid around this station
        X_vals = np.linspace(xs - LOCAL_GRID/2, xs + LOCAL_GRID/2, GRID_RES)
        Y_vals = np.linspace(ys - LOCAL_GRID/2, ys + LOCAL_GRID/2, GRID_RES)
        X, Y = np.meshgrid(X_vals, Y_vals)

        Z = np.zeros_like(X)

        # Compute strain locally
        for ii in range(GRID_RES):
            for jj in range(GRID_RES):
                xp = X[ii, jj]
                yp = Y[ii, jj]

                try:
                    x, y, z_eval = compute_relative_xyz(
                        xp, yp, x0_prime, y0_prime, theta_deg, zs, h)

                    tensor = sf.strain(
                        x=x, y=y, z=z_eval,
                        a=a, b=b, c=c,
                        ec=ec, h=h, nu=nu
                    )

                    Z[ii, jj] = tensor[idx] * 1e9  # nanostrain

                except Exception:
                    Z[ii, jj] = np.nan

        Z = np.nan_to_num(Z, nan=0.0)

        # Plot mini-contour for this station
        cs = ax.contourf(X, Y, Z, levels=25, alpha=0.6)
        all_contours.append(cs)

        # Mark station
        ax.scatter(xs, ys, color='black', s=15)
        ax.text(xs, ys, st_name, fontsize=7)

    # Plot inclusion outline on each panel
    theta = np.linspace(0, 2*np.pi, 200)
    x_ellipse = x0_prime + a * np.cos(theta)
    y_ellipse = y0_prime + b * np.sin(theta)
    ax.plot(x_ellipse, y_ellipse, 'r-', linewidth=2, label="Inclusion")

    ax.set_title(label)
    ax.set_xlabel("x' (m)")
    ax.set_ylabel("y' (m)")
    ax.set_aspect('equal')

# ==========================================================
# SINGLE COLORBAR BELOW
# ==========================================================
cbar = fig.colorbar(all_contours[0], ax=axes.tolist(),
                    orientation='horizontal', pad=0.08)
cbar.set_label("Strain (nanostrain)")

# Overall figure title
fig.suptitle(f"Full Strain Tensor around AVANT stations at t = {TIME_DAY} days",
             fontsize=14)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("full_tensor_mini_contours.png", dpi=300, bbox_inches="tight")
plt.show()

print("[DONE] Saved full_tensor_mini_contours.png")
