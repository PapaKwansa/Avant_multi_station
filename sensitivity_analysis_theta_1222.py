# ==========================================================
# Sensitivity plots: Strain components vs. inclusion orientation (theta)
# High-quality, publication-ready formatting
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as mpi
import multi_station_strain as sf

# ---------------- Matplotlib global style ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

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

# Theta sweep (smooth curves)
theta_range = np.linspace(-17.2, 180, 360)

# Output directory
os.makedirs('sensitivity_plots', exist_ok=True)

# Labels and colors
comp_labels = [
    r'$\varepsilon_{xx}$',
    r'$\varepsilon_{yy}$',
    r'$\varepsilon_{zz}$',
    r'$\varepsilon_{xy}$',
    r'$\varepsilon_{xz}$',
    r'$\varepsilon_{yz}$'
]
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']

# ---------------- Helper function ----------------
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

    strain_vs_theta = np.zeros((len(theta_range), 6))

    for idx, theta_deg in enumerate(theta_range):
        try:
            x, y, z_eval = compute_relative_xyz(
                x_station, y_station,
                x0_prime, y0_prime,
                theta_deg, z, h
            )
            tensor = sf.strain(x, y, z_eval, a, b, c, ec, h)

            strain_vs_theta[idx, :] = np.array([
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[0, 1],
                tensor[0, 2],
                tensor[1, 2]
            ]) * 1e9  # nanostrain

        except Exception:
            strain_vs_theta[idx, :] = np.nan

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for k, (label, color) in enumerate(zip(comp_labels, colors)):
        ax.plot(theta_range, strain_vs_theta[:, k], label=label, color=color)

    ax.axvline(
        theta_deg_init,
        color='k',
        linestyle='--',
        linewidth=1.8,
        label=rf'Initial $\theta = {theta_deg_init}^\circ$'
    )

    ax.set_xlabel(r'Inclusion Orientation $\theta$ (degrees)')
    ax.set_ylabel(r'Strain (nanostrain)')
    ax.set_title(
        f'FS{i_station+1:02d}: Sensitivity of Strain Components to Inclusion Orientation'
    )

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, linestyle='--', alpha=0.3)

    # ---------------- Legend BELOW the plot ----------------
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(
        f"sensitivity_plots/FS{i_station+1:02d}_sensitivity.png",
        bbox_inches='tight'
    )
    plt.close()

print("High-quality sensitivity plots saved in 'sensitivity_plots/'.")
