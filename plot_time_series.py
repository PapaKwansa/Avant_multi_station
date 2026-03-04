import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forward_model_multi_station import forward_model_multi_station
import multi_stations_input as inp

# ------------------------------------------------------------
# LOAD PARAMETERS
# ------------------------------------------------------------
params = inp.read_input()

# ------------------------------------------------------------
# RUN FORWARD MODEL
# ------------------------------------------------------------
df = forward_model_multi_station(
    params['pmax'], params['tpeak'], params['d'], params['time'],
    params['x_prime'], params['y_prime'],
    params['x0_prime'], params['y0_prime'],
    params['z'], params['a'], params['b'], params['c'],
    params['nu'], params['h'], params['E'], params['theta_deg'],
    alpha=params.get('alpha'),
    station_names=params['station_names']
)

# ------------------------------------------------------------
# TIME ARRAY
# ------------------------------------------------------------
time = df['Time (days)']

# ------------------------------------------------------------
# STRAIN COMPONENTS
# ------------------------------------------------------------
components = [
    'Epsilon_XX_nanostrain', 'Epsilon_YY_nanostrain', 'Epsilon_ZZ_nanostrain',
    'Epsilon_XY_nanostrain', 'Epsilon_XZ_nanostrain', 'Epsilon_YZ_nanostrain'
]

# ------------------------------------------------------------
# PLOT TIME SERIES (ONE FIGURE PER STATION)
# ------------------------------------------------------------
for station in params['station_names']:

    fig, ax = plt.subplots(figsize=(10, 6))

    for comp in components:
        col = f"{comp}_{station}"
        label = comp.replace('Epsilon_', '').replace('_nanostrain', '')
        ax.plot(time, df[col], label=label)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Strain (nanostrain)")
    ax.set_title(f"Strain Time Series — Station {station}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"time_series_{station}.png")
    plt.show()
