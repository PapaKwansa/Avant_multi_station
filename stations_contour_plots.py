# ==========================================================
# Strain contours: 6 subplots per component, per station, per theta
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as mpi
import multi_station_strain as sf

# ---------------- Load parameters ----------------
params = mpi.read_input()

pmax = params['pmax']
tpeak = params['tpeak']
d = params['d']
time_vals = params['time']

x_prime = params['x_prime']
y_prime = params['y_prime']
x0_prime = params['x0_prime']
y0_prime = params['y0_prime']
theta_deg_init = params['theta_deg']
z = params['z']
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

# Characteristic strain
ec = sf.charac_strain(sf.linear_trans(alpha, nu, pmax, E))

# Load stations
stations_df = pd.read_csv('FORGE_stations.csv')
num_stations = len(stations_df)

# Grid for plan view around each station
grid_size = 500
grid_points = 100

# Theta sweep
theta_range = np.linspace(-17.2, 180, 5)  # Reduced for testing

# Create output folder
os.makedirs('contour_plots_subplots', exist_ok=True)

# Component labels
comp_labels = ['Epsilon_XX', 'Epsilon_YY', 'Epsilon_ZZ', 'Epsilon_XY', 'Epsilon_XZ', 'Epsilon_YZ']

# Helper: compute relative coordinates
def compute_relative_xyz(xp, yp, x0p, y0p, theta_deg, zg, h):
    theta = np.radians(theta_deg)
    dxp = xp - x0p
    dyp = yp - y0p
    x = dxp * np.cos(theta) + dyp * np.sin(theta)
    y = -dxp * np.sin(theta) + dyp * np.cos(theta)
    z = zg - h
    return x, y, z

# ---------------- Loop over stations ----------------
for i_station in range(num_stations):
    x_station = x_prime[i_station]
    y_station = y_prime[i_station]

    for theta_deg in theta_range:
        # Grid around the station
        X_vals = np.linspace(x_station - grid_size/2, x_station + grid_size/2, grid_points)
        Y_vals = np.linspace(y_station - grid_size/2, y_station + grid_size/2, grid_points)
        X, Y = np.meshgrid(X_vals, Y_vals)

        # Initialize arrays for strain components
        eps_arrays = [np.zeros_like(X) for _ in range(6)]

        # Compute strain at each grid point
        for ii in range(X.shape[0]):
            for jj in range(X.shape[1]):
                xp = X[ii, jj]
                yp = Y[ii, jj]
                try:
                    x, y, z_eval = compute_relative_xyz(xp, yp, x0_prime, y0_prime, theta_deg, z, h)
                    tensor = sf.strain(x, y, z_eval, a, b, c, ec, h)

                    eps_arrays[0][ii,jj] = tensor[0,0]*1e9
                    eps_arrays[1][ii,jj] = tensor[1,1]*1e9
                    eps_arrays[2][ii,jj] = tensor[2,2]*1e9
                    eps_arrays[3][ii,jj] = tensor[0,1]*1e9
                    eps_arrays[4][ii,jj] = tensor[0,2]*1e9
                    eps_arrays[5][ii,jj] = tensor[1,2]*1e9

                except Exception:
                    for k in range(6):
                        eps_arrays[k][ii,jj] = np.nan

        # Plot 6 subplots, one per component
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, eps, label in zip(axes.flat, eps_arrays, comp_labels):
            eps = np.nan_to_num(eps, nan=0.0)
            max_val = np.max(np.abs(eps))
            if max_val > 0:
                levels = np.linspace(-max_val, max_val, 50)
                cs = ax.contourf(X, Y, eps, levels=levels, cmap='coolwarm', extend='both')
                cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
                cbar.set_label('Strain (nanostrain)', fontsize=10)
            else:
                ax.contourf(X, Y, eps, levels=50, cmap='coolwarm')
            
            # Mark station
            ax.plot(x_station, y_station, 'kx', markersize=10)
            
            ax.set_title(f'{label}', fontsize=12)
            ax.set_xlabel("x' (m)", fontsize=10)
            ax.set_ylabel("y' (m)", fontsize=10)
            ax.set_aspect('equal')

        # Overall title with orientation
        fig.suptitle(f'FS{i_station+1:02d} - Strain Components at θ={theta_deg:.1f}°', fontsize=16)

        plt.tight_layout()
        plt.savefig(f"contour_plots_subplots/FS{i_station+1:02d}_theta_{theta_deg:.1f}.png", dpi=300)
        plt.close()

print("Strain component subplots saved in 'contour_plots_subplots/' for each station and theta.")