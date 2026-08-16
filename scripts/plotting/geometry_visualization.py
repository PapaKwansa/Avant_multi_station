import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------- Load station data ----------------------
stations_df = pd.read_csv("AVANT_stations.csv")
x = stations_df["x_prime"].values
y = stations_df["y_prime"].values
names = stations_df["station"].values

# ---------------------- Lens geometry ----------------------
a = 150.0   # semi-axis in x-direction
c = 290.0   # semi-axis in y-direction
x0 = -208.33333333333337
y0 = 83.33333333333326

# ---------------------- Distances ----------------------
dx = x - x0
dy = y - y0
r = np.sqrt(dx**2 + dy**2)

print("\nDistances from lens center:")
for name, dist in zip(names, r):
    print(f"{name}: {dist:.2f} m")

# ---------------------- Plot ----------------------
fig, ax = plt.subplots(figsize=(7, 6))

# Stations
ax.scatter(x, y, s=80, label="Stations", color="blue")
for xi, yi, name in zip(x, y, names):
    ax.text(xi + 10, yi + 10, name)

# Lens center
ax.scatter([x0], [y0], color="red", marker="x", s=120, label="Lens center")

# Lens footprint (ellipse)
theta = np.linspace(0, 2*np.pi, 400)
ellipse_x = x0 + a * np.cos(theta)
ellipse_y = y0 + c * np.sin(theta)
ax.plot(ellipse_x, ellipse_y, "r--", label="Lens footprint (a=150, c=290)")

ax.set_aspect("equal", "box")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Lens Geometry and Station Locations")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
