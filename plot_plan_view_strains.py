# plot_plan_view_strains_vectorized.py
import numpy as np
import matplotlib.pyplot as plt

import multi_station_strain as sf
import multi_stations_input as inp
import multi_station_coord_transform as msc
import multi_station_rotation as rot

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
user_depth = float(input("Enter observation depth (m): "))

# ------------------------------------------------------------
# LOAD PARAMETERS
# ------------------------------------------------------------
params = inp.read_input()

# Material / physics
nu     = params["nu"]
alpha  = params["alpha"]
E      = params["E"]
pmax   = params["pmax"]
h      = params["h"]  # inclusion top depth

# Inclusion geometry
x0p = params["x0_prime"]
y0p = params["y0_prime"]
theta_deg = params["theta_deg"]
a, b, c = params["a"], params["b"], params["c"]

# ------------------------------------------------------------
# CREATE PLAN-VIEW GRID
# ------------------------------------------------------------
x_prime_vals = np.linspace(-600, 600, 120)
y_prime_vals = np.linspace(-600, 600, 120)
X_prime, Y_prime = np.meshgrid(x_prime_vals, y_prime_vals)

# ------------------------------------------------------------
# VECTORIZE COORDINATE TRANSFORMATION
# ------------------------------------------------------------
X_unprimed, Y_unprimed = msc.primed_to_unprimed(X_prime, Y_prime, x0p, y0p, theta_deg)

# Depth is scalar, broadcast to grid shape if needed
Z = np.full_like(X_prime, user_depth)

# ------------------------------------------------------------
# COMPUTE CHARACTERISTIC STRAIN
# ------------------------------------------------------------
linear = sf.linear_trans(alpha, nu, pmax, E)
ec     = sf.charac_strain(linear, nu=nu)

# ------------------------------------------------------------
# VECTORIZE STRAIN CALCULATION
# ------------------------------------------------------------
# Allocate arrays
shape = X_prime.shape
strain_xx = np.zeros(shape)
strain_yy = np.zeros(shape)
strain_zz = np.zeros(shape)
strain_xy = np.zeros(shape)
strain_xz = np.zeros(shape)
strain_yz = np.zeros(shape)

# Flatten grids for vectorized iteration
X_flat = X_unprimed.ravel()
Y_flat = Y_unprimed.ravel()
Z_flat = Z.ravel()

# Loop over flattened arrays (still much faster than 2D loops)
for idx in range(X_flat.size):
    x, y, z = X_flat[idx], Y_flat[idx], Z_flat[idx]

    try:
        # Compute strain tensor
        S = sf.strain(x, y, z, a, b, c, ec, h, nu)

        # Rotate tensor into primed frame
        exx_p, eyy_p, exy_p, exz_p, eyz_p, ezz_p = rot.rotate_strain_tensor(
            S[0,0], S[1,1], S[0,1], S[0,2], S[1,2], S[2,2], theta_deg
        )

        # Assign back to arrays
        strain_xx.ravel()[idx] = exx_p
        strain_yy.ravel()[idx] = eyy_p
        strain_zz.ravel()[idx] = ezz_p
        strain_xy.ravel()[idx] = exy_p
        strain_xz.ravel()[idx] = exz_p
        strain_yz.ravel()[idx] = eyz_p

    except:
        strain_xx.ravel()[idx] = np.nan
        strain_yy.ravel()[idx] = np.nan
        strain_zz.ravel()[idx] = np.nan
        strain_xy.ravel()[idx] = np.nan
        strain_xz.ravel()[idx] = np.nan
        strain_yz.ravel()[idx] = np.nan

# ------------------------------------------------------------
# CONVERT TO NANOSTRAIN
# ------------------------------------------------------------
strain_xx *= 1e9
strain_yy *= 1e9
strain_zz *= 1e9
strain_xy *= 1e9
strain_xz *= 1e9
strain_yz *= 1e9

# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 18), constrained_layout=True)

fields = [strain_xx, strain_yy, strain_zz, strain_xy, strain_xz, strain_yz]
titles = [
    f"XX strain at {user_depth} m",
    f"YY strain at {user_depth} m",
    f"ZZ strain at {user_depth} m",
    f"XY strain at {user_depth} m",
    f"XZ strain at {user_depth} m",
    f"YZ strain at {user_depth} m"
]

for ax, data, title in zip(axes.flat, fields, titles):
    cont = ax.contourf(X_prime, Y_prime, data, cmap="coolwarm", levels=100)
    cbar = plt.colorbar(cont, ax=ax)
    cbar.set_label("nanostrain", fontsize=14)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("x' (m)", fontsize=14)
    ax.set_ylabel("y' (m)", fontsize=14)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=12)

plt.show()
