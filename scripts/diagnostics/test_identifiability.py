# test_identifiability_fixed.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
from pydream.core import run_dream
from pydream.parameters import SampledParam
import time

import multi_stations_input
import bayesian_inversion_multi_station as bi
from forward_model_multi_station import forward_model_multi_station

# -------------------------
# Timestamp for safe filenames (Windows-friendly)
# -------------------------
TIMESTAMP = time.strftime("%Y_%m_%d_%H_%M_%S")

# -------------------------
# Load inputs
# -------------------------
params = multi_stations_input.read_input()
Ns = len(params['x_prime'])
time_array = np.array(params['time'])
x_prime = np.asarray(params['x_prime'])
y_prime = np.asarray(params['y_prime'])

# Build COMPONENT_COLS
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

# Generate synthetic "observed" dataset
df_clean = forward_model_multi_station(
    pmax=params['pmax'], tpeak=params['tpeak'], d=params['d'], time=time_array,
    x_prime=x_prime, y_prime=y_prime,
    x0_prime=params['x0_prime'], y0_prime=params['y0_prime'],
    z=params['z'], a=params['a'], b=params['b'], c=params['c'],
    nu=params['nu'], h=params['h'], E=params['E'],
    theta_deg=params['theta_deg'], alpha=params.get('alpha', None)
)
observed_vector = bi._flatten_df_to_vector(df_clean, cols=COMPONENT_COLS)

# -------------------------
# Likelihood wrapper
# -------------------------
def likelihood_wrapper(params_in):
    a_s, b_s, c_s, E_s, theta_deg_s, sigma_s = np.ravel(params_in)
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
# Parameter names and baseline values
# -------------------------
param_names = ['a', 'b', 'c', 'E', 'theta_deg', 'sigma']
baseline = [params['a'], params['b'], params['c'], params['E'], params['theta_deg'], 0.05]

# -------------------------
# Function to generate priors for identifiability tests
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
                # ---- FIX IS HERE ----
                theta_width = 20.0  # degrees (±10° around baseline)
                priors.append(
                    SampledParam(
                        uniform,
                        loc=baseline[i] - theta_width / 2,
                        scale=theta_width
                    )
                )

            else:
                # Other physical parameters (a, b, c, E)
                loc = baseline[i] * 0.8
                scale = abs(baseline[i] * 0.4)
                priors.append(
                    SampledParam(uniform, loc=loc, scale=scale)
                )

        else:
            # Fix non-free parameters
            priors.append(
                SampledParam(uniform, loc=baseline[i], scale=1e-8)
            )

    return priors


# -------------------------
# Test identifiability function
# -------------------------
def test_identifiability(free_indices, niterations=2000, nchains=3):
    priors = build_priors(free_indices)

    # Use Windows-safe model_name
    model_name = f"{TIMESTAMP}_DREAM"

    sampled_params, logps = run_dream(
        parameters=priors,
        likelihood=likelihood_wrapper,
        niterations=niterations,
        nchains=nchains,
        model_name=model_name
    )

    posterior = np.stack([np.asarray(ch) for ch in sampled_params], axis=0)

    # Plot posterior(s)
    for idx in free_indices:
        plt.figure()
        for ch in range(posterior.shape[0]):
            plt.hist(posterior[ch,:,idx], bins=30, density=True, alpha=0.5, label=f'Chain {ch+1}')
        # Show prior range and baseline
        low = baseline[idx]*0.8 if param_names[idx]!='sigma' else 1e-4
        high = baseline[idx]*1.2 if param_names[idx]!='sigma' else 0.1
        plt.axvline(baseline[idx], color='r', linestyle='--', label='Baseline')
        plt.title(f"Posterior of {param_names[idx]} (free)")
        plt.xlabel(param_names[idx])
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    return posterior

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # Test each parameter individually
    for i in range(len(param_names)):
        print(f"Testing identifiability of {param_names[i]} alone...")
        test_identifiability([i], niterations=1500, nchains=3)
    
    # Test a pair example (b and E)
    print("Testing identifiability of b and E together...")
    test_identifiability([1,3], niterations=2000, nchains=3)
