from multi_stations_input import read_input
from forward_model_multi_station import forward_model_multi_station
import numpy as np

params = read_input()

# --- FIX: compute geometry properly ---
s0 = params.get("s", 100.0)  # default scale

a = s0 * params["a0"]
b = s0 * params["b0"]
c = s0 * params["c0"]

# --- FIX: define theta ---
theta_deg = params.get("theta_deg", params["theta_deg_start"])

# --- test exactly at peak time ---
test_time = np.array([params["tpeak"]])

df_test = forward_model_multi_station(
    pmax=params["pmax"],
    tpeak=params["tpeak"],
    d=params["d"],
    time=test_time,
    x_prime=params["x_prime"],
    y_prime=params["y_prime"],
    x0_prime=params["x0_prime"],
    y0_prime=params["y0_prime"],
    z=params["z"],
    a=a,
    b=b,
    c=c,
    nu=params["nu"],
    h=params["h"],
    E=params.get("E", 1e10),  # fallback
    theta_deg=theta_deg,
    alpha=params["alpha"],
    station_names=params["station_names"],
    debug=True
)

print("\n[RESULT] Forward model at tpeak:")
print(df_test)