# test_identifiability_pairwise_fixed.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from itertools import combinations

from pydream.core import run_dream
from pydream.parameters import SampledParam

import multi_stations_input
import bayesian_inversion_multi_station as bi
from forward_model_multi_station import forward_model_multi_station

# -------------------------
# Timestamp (Windows safe)
# -------------------------
TIMESTAMP = time.strftime("%Y_%m_%d_%H_%M_%S")

# -------------------------
# Load inputs
# -------------------------
params = multi_stations_input.read_input()

time_array = np.asarray(params['time'])
x_prime = np.asarray(params['x_prime'])
y_prime = np.asarray(params['y_prime'])
Ns = len(x_prime)

# -------------------------
# Build COMPONENT_COLS
# -------------------------
base_components = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

COMPONENT_COLS = []
for comp in base_components:
    for s in range(Ns):
        COMPONENT_COLS.append(f"{comp}_FS{s+1:02d}")

# -------------------------
# Generate synthetic observations
# -------------------------
df_clean = forward_model_multi_station(
    pmax=params['pmax'],
    tpeak=params['tpeak'],
    d=params['d'],
    time=time_array,
    x_prime=x_prime,
    y_prime=y_prime,
    x0_prime=params['x0_prime'],
    y0_prime=params['y0_prime'],
    z=params['z'],
    a=params['a'],
    b=params['b'],
    c=params['c'],
    nu=params['nu'],
    h=params['h'],
    E=params['E'],
    theta_deg=params['theta_deg'],
    alpha=params.get('alpha', None),
)

observed_vector = bi._flatten_df_to_vector(
    df_clean, cols=COMPONENT_COLS
)

# -------------------------
# Likelihood wrapper
# -------------------------
def likelihood_wrapper(theta):
    a_s, b_s, c_s, E_s, theta_deg_s, sigma_s = np.ravel(theta)

    return bi.likelihood(
        params=[a_s, b_s, c_s, E_s, theta_deg_s, sigma_s],
        observed_vector=observed_vector,
        time=time_array,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=params['x0_prime'],
        y0_prime=params['y0_prime'],
        z=params['z'],
        nu=params['nu'],
        pmax=params['pmax'],
        tpeak=params['tpeak'],
        d=params['d'],
        h=params['h'],
        alpha=params.get('alpha', None),
        component_cols=COMPONENT_COLS
    )

# -------------------------
# Parameters
# -------------------------
param_names = ['a', 'b', 'c', 'E', 'theta_deg', 'sigma']
baseline = [
    params['a'],
    params['b'],
    params['c'],
    params['E'],
    params['theta_deg'],
    0.05
]

# -------------------------
# Build priors
# -------------------------
def build_priors(free_indices):
    priors = []

    for i, name in enumerate(param_names):

        if i in free_indices:

            if name == 'sigma':
                priors.append(
                    SampledParam(uniform, loc=baseline[i], scale=1e-8)
                )

            elif name == 'theta_deg':
                width = 20.0
                priors.append(
                    SampledParam(
                        uniform,
                        loc=baseline[i] - width / 2,
                        scale=width
                    )
                )

            else:
                priors.append(
                    SampledParam(
                        uniform,
                        loc=0.8 * baseline[i],
                        scale=0.4 * abs(baseline[i])
                    )
                )

        else:
            # Fix parameter
            priors.append(
                SampledParam(uniform, loc=baseline[i], scale=1e-8)
            )

    return priors

# -------------------------
# Single / Pair identifiability
# -------------------------
def run_identifiability(free_indices, niterations=3000, nchains=3):

    priors = build_priors(free_indices)

    sampled_params, logps = run_dream(
        parameters=priors,
        likelihood=likelihood_wrapper,
        niterations=niterations,
        nchains=nchains,
        model_name=f"{TIMESTAMP}_DREAM"
    )

    posterior = np.stack([np.asarray(ch) for ch in sampled_params])

    # ----- Marginal plots -----
    for idx in free_indices:
        plt.figure()
        for ch in range(nchains):
            plt.hist(
                posterior[ch, :, idx],
                bins=40,
                density=True,
                alpha=0.5,
                label=f'Chain {ch+1}'
            )

        plt.axvline(baseline[idx], color='r', linestyle='--')
        plt.xlabel(param_names[idx])
        plt.ylabel("Density")
        plt.title(f"Posterior of {param_names[idx]}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----- Joint plot if pair -----
    if len(free_indices) == 2:
        i, j = free_indices

        samples_i = posterior[:, :, i].ravel()
        samples_j = posterior[:, :, j].ravel()

        plt.figure(figsize=(6, 6))
        plt.scatter(samples_i, samples_j, s=5, alpha=0.3)
        plt.axvline(baseline[i], color='r', linestyle='--')
        plt.axhline(baseline[j], color='r', linestyle='--')
        plt.xlabel(param_names[i])
        plt.ylabel(param_names[j])
        plt.title(f"Joint posterior: {param_names[i]} vs {param_names[j]}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return posterior

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    # ---- 1D identifiability ----
    for i, name in enumerate(param_names):
        print(f"\nTesting identifiability of {name} alone")
        run_identifiability([i], niterations=1500, nchains=3)

    # ---- Pairwise tests you requested ----
    pair_tests = [
        ('a', 'b'),
        ('b', 'E'),
        ('c', 'b'),
        ('a', 'E'),
        ('c', 'E'),
        ('a', 'c'),
        ('E', 'theta_deg'),
        ('a', 'theta_deg'),
        ('b', 'theta_deg'),
        ('c', 'theta_deg'),
        ('sigma', 'E'),
        ('sigma', 'a'),
        ('sigma', 'b'),
        ('sigma', 'c'),
        ('sigma', 'theta_deg'),
    ]

    for p1, p2 in pair_tests:
        i = param_names.index(p1)
        j = param_names.index(p2)
        print(f"\nTesting identifiability of {p1} and {p2}")
        run_identifiability([i, j], niterations=3000, nchains=3)
