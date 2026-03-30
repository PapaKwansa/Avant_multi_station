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
# ------------------------------------------------------------
# HIGH‑QUALITY PLOTTING (presentation ready)
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "axes.linewidth": 2.2,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 200,
})

fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)

fields = [eXX, eYY, eXY, eZZ]
titles = [
    rf"$\varepsilon_{{xx}}$ at depth {user_depth} m",
    rf"$\varepsilon_{{yy}}$ at depth {user_depth} m",
    rf"$\varepsilon_{{xy}}$ at depth {user_depth} m",
    rf"$\varepsilon_{{zz}}$ at depth {user_depth} m",
]

for ax, data, title in zip(axes.flat, fields, titles):

    # High‑resolution contour
    cont = ax.contourf(
        X, Y, data,
        cmap="coolwarm",
        levels=200,
        antialiased=True
    )

    # Clean colorbar
    cbar = plt.colorbar(cont, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Strain (nanostrain)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Titles and labels
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("x (m)", fontweight="bold")
    ax.set_ylabel("y (m)", fontweight="bold")

    # Equal aspect ratio for physical correctness
    ax.set_aspect("equal")

    # Subtle grid
    ax.grid(True, alpha=0.25, linewidth=1.0)

# Save high‑quality figure
plt.savefig("plan_view_strain_fields.png", dpi=350, bbox_inches="tight")
plt.show()
