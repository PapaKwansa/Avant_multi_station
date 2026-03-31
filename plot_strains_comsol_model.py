import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as inp

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
user_depth = float(input("Enter observation depth (m): "))

# ------------------------------------------------------------
# LOAD PARAMETERS
# ------------------------------------------------------------
params = inp.read_input()

nu     = params["nu"]
alpha  = params["alpha"]
E      = params["E"]
pmax   = params["pmax"]
tpeak  = params["tpeak"]
d      = params["d"]
h      = params["h"]

# Geometry (semi-axes)
a = params["a_fixed"]
b = params["b_fixed"]
c = params["c_fixed"]

# Inclusion center
x0 = params["x0_prime"]
y0 = params["y0_prime"]

theta_deg = params["theta_deg"]  # 0.0 in  COMSOL setup

# ------------------------------------------------------------
# CREATE PLAN-VIEW GRID
# ------------------------------------------------------------
x_vals = np.linspace(-600, 600, 150)
y_vals = np.linspace(-600, 600, 150)
X, Y = np.meshgrid(x_vals, y_vals)

# Flatten for forward model
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = np.full_like(x_flat, user_depth)

# ------------------------------------------------------------
# PRESSURE HISTORY (use peak pressure)
# ------------------------------------------------------------
time = np.array([tpeak])  # single time step at peak

# Treat each grid point as a "station"
station_names = [f"P{i}" for i in range(len(x_flat))]

df = forward_model_multi_station(
    pmax=pmax,
    tpeak=tpeak,
    d=d,
    time=time,
    x_prime=x_flat,
    y_prime=y_flat,
    x0_prime=x0,
    y0_prime=y0,
    z=z_flat,
    a=a,
    b=b,
    c=c,
    nu=nu,
    h=h,
    E=E,
    theta_deg=theta_deg,
    alpha=alpha,
    station_names=station_names,
    debug=False,
)

# ------------------------------------------------------------
# EXTRACT STRAIN COMPONENTS
# ------------------------------------------------------------
def reshape_component(prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return df[cols].values[0].reshape(X.shape)

eXX = reshape_component("eXX")
eYY = reshape_component("eYY")
eXY = reshape_component("eXY")
eZZ = reshape_component("eZZ")

# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
fig.subplots_adjust(top=0.90, hspace=0.35) 

# 👇 move ALL subplots downward
fig.subplots_adjust(top=0.90)

# 👇 place title safely above
fig.suptitle(
    f"Plan-view strain tensor components at depth {user_depth:.1f} m",
    fontsize=20,
    fontweight="bold"
)

fields = [eXX, eYY, eXY, eZZ]
titles = [
    rf"$\varepsilon_{{xx}}$ ",
    rf"$\varepsilon_{{yy}}$ ",
    rf"$\varepsilon_{{xy}}$ ",
    rf"$\varepsilon_{{zz}}$ ",
]

for ax, data, title in zip(axes.flat, fields, titles):

    # symmetric color scale for better visual comparison
    vmax = np.nanmax(np.abs(data))
    levels = np.linspace(-vmax, vmax, 40)

    cont = ax.contourf(X, Y, data, levels=levels, cmap="RdBu_r", extend="both")

    # colorbar
    cbar = plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Nanostrain", fontsize=15, fontweight="bold")
    cbar.ax.tick_params(labelsize=13)

    # titles and labels
    ax.set_title(title, fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("x (m)", fontsize=15, fontweight="bold")
    ax.set_ylabel("y (m)", fontsize=15, fontweight="bold")

    # axes styling
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="both", labelsize=13, width=1.5, length=5)
    ax.grid(True, alpha=0.25, linewidth=0.8)

# global title
fig.suptitle(
    f"Plan-view strain fields at depth {user_depth:.1f} m",
    fontsize=20,
    fontweight="bold",
    y=0.98
)

# save high-quality figure
plt.savefig("strain_plan_view.png", dpi=300)

plt.show()
