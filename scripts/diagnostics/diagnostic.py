import multi_station_strain as sf
import multi_stations_input as inp
import numpy as np

params = inp.read_input()

sf.nu = params["nu"]
sf.E = params["E"]
sf.alpha = params["alpha"]

a, b, c = params["a"], params["b"], params["c"]
h = params["h"]
alpha = params["alpha"]
nu = params["nu"]
E = params["E"]
pmax = params["pmax"]

ec = sf.charac_strain(sf.linear_trans(alpha, nu, pmax, E))

# Test at center of inclusion
x = 0
y = 0
z = h  # top of inclusion

S = sf.strain(x, y, z, a, b, c, ec, h)
print("Strain tensor at inclusion top:\n", S)
