# ==========================================================
# Sensitivity of strain components to inclusion orientation
# For each station: strain(θ) with baseline θ = -17.2°
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import multi_stations_input as mpi
import multi_station_strain as sf
import multi_station_coord_transform as mct

# ---------------- Load parameters ----------------
params = mpi.read_input()

pmax = params['pmax']
x_prime = np.asarray(params['x_prime'])
y_prime = np.asarray(params['y_prime'])
x0_prime = params['x0_prime']
y0_prime = params['y0_prime']
z = np.asarray(params['z'])

a = params['a']
b = params['b']
c = params['c']
nu = params['nu']
h = params['h']
E = params['E']
alpha = params.get('alpha', None)

theta_baseline = -17.2   # degrees

# Set strain globals
sf.nu = nu
sf.E = E
sf.alpha = alpha

# Characteristic strain using peak pressure
lt = sf.linear_trans(alpha, nu, pmax, E)
ec = sf.charac_strain(lt, nu)

# Theta sweep
theta_range = np.linspace(-30, 180, 360)

# Station names
stations_df = pd.read_csv('AVANT_stations.csv')
station_names = stations_df["station"].astype(str).str.strip().values
Ns = len(station_names)

os.makedirs("theta_sensitivity_plots", exist_ok=True)

# ---------------- Loop stations ----------------
for js in range(Ns):

    strain_vs_theta = np.zeros((len(theta_range), 6))

    for i, theta in enumerate(theta_range):

        # Rotate coordinates relative to inclusion orientation
        x_rot, y_rot = mct.primed_to_unprimed(
            x_prime[js],
            y_prime[js],
            x0_prime,
            y0_prime,
            theta
        )

        # Strain in GLOBAL coordinates (no tensor rotation)
        S = sf.strain(
            x=x_rot,
            y=y_rot,
            z=z[js],
            a=a,
            b=b,
            c=c,
            ec=ec,
            h=h,
            nu=nu
        )

        strain_vs_theta[i,:] = np.array([
            S[0,0], S[1,1], S[2,2],
            S[0,1], S[0,2], S[1,2]
        ]) * 1e9  # convert to nanostrain

    # ---------------- Plot ----------------
    plt.figure(figsize=(10,6))

    labels = [
        r'$\varepsilon_{xx}$',
        r'$\varepsilon_{yy}$',
        r'$\varepsilon_{zz}$',
        r'$\varepsilon_{xy}$',
        r'$\varepsilon_{xz}$',
        r'$\varepsilon_{yz}$'
    ]

    for k in range(6):
        plt.plot(theta_range, strain_vs_theta[:,k], label=labels[k])

    plt.axvline(theta_baseline, linestyle='--', color='k',
                label=rf'Baseline $\theta={theta_baseline}^\circ$')

    plt.xlabel(r'Inclusion Orientation $\theta$ (degrees)')
    plt.ylabel('Strain (nanostrain)')
    plt.title(f'{station_names[js]}: Strain vs Inclusion Orientation')

    plt.legend(bbox_to_anchor=(0.5,-0.2), loc='upper center', ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"theta_sensitivity_plots/{station_names[js]}_theta_sensitivity.png",
                bbox_inches='tight')
    plt.close()

print("Theta sensitivity plots completed.")
