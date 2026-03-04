# ------------------------------------------------------------
# test_identifiability_pairwise_joint_heatmap_strain_volume.py
#
# Pairwise Bayesian identifiability analysis with:
# - Strain tensor time series
# - Volume time series constraint
# - Two free parameters at a time
# - All other parameters fixed at true values
# - Heatmap-only joint posterior plots
# ------------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import uniform
from pydream.core import run_dream
from pydream.parameters import SampledParam

import multi_stations_input
import bayesian_inversion_multi_station as bi
from forward_model_multi_station import forward_model_multi_station

# ------------------------------------------------------------
# Timestamp
# ------------------------------------------------------------
TIMESTAMP = time.strftime("%Y_%m_%d_%H_%M_%S")

# ------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------
params = multi_stations_input.read_input()

Ns = len(params['x_prime'])
time_array = np.asarray(params['time'])
x_prime = np.asarray(params['x_prime'])
y_prime = np.asarray(params['y_prime'])

# ------------------------------------------------------------
# Component columns
# ------------------------------------------------------------
base_components = [
    "Epsilon_XX_nanostrain",
    "Epsilon_YY_nanostrain",
    "Epsilon_ZZ_nanostrain",
    "Epsilon_XY_nanostrain",
    "Epsilon_XZ_nanostrain",
    "Epsilon_YZ_nanostrain",
]

COMPONENT_COLS = [
    f"{comp}_FS{s+1:02d}"
    for comp in base_components
    for s in range(Ns)
]

# ------------------------------------------------------------
# Synthetic strain observations (truth)
# ------------------------------------------------------------
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
    alpha=params.get('alpha', None)
)

observed_vector = bi._flatten_df_to_vector(df_clean, cols=COMPONENT_COLS)

# ------------------------------------------------------------
# Volume forward model + synthetic observations
# ------------------------------------------------------------
def bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))

def volume_forward(a, b, c, E, nu, delta_P):
    V0 = a * b * c
    K = bulk_modulus(E, nu)
    return V0 * (1.0 + delta_P / K)

delta_P = params['pmax'] * np.sin(np.pi * time_array / time_array.max())

V_true = volume_forward(
    params['a'], params['b'], params['c'],
    params['E'], params['nu'], delta_P
)

sigma_volume = 0.05 * np.std(V_true)
volume_obs = V_true + np.random.normal(
    0.0, sigma_volume, size=V_true.size
)

# ------------------------------------------------------------
# Likelihood wrapper (strain + volume)
# ------------------------------------------------------------
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
        component_cols=COMPONENT_COLS,
        volume_obs=volume_obs,
        delta_P=delta_P,
        sigma_volume=sigma_volume,
        volume_weight=1.0
    )

# ------------------------------------------------------------
# Parameter setup
# ------------------------------------------------------------
param_names = ['a', 'b', 'c', 'E', 'theta_deg', 'sigma']

baseline = [
    params['a'],
    params['b'],
    params['c'],
    params['E'],
    params['theta_deg'],
    0.05
]

# ------------------------------------------------------------
# Prior construction
# ------------------------------------------------------------
def build_priors(free_indices):
    priors = []
    for i, name in enumerate(param_names):
        if i in free_indices:
            if name == 'theta_deg':
                priors.append(
                    SampledParam(uniform, loc=baseline[i] - 15.0, scale=30.0)
                )
            else:
                priors.append(
                    SampledParam(uniform,
                                 loc=0.8 * baseline[i],
                                 scale=0.4 * baseline[i])
                )
        else:
            # Effectively fixed parameter
            priors.append(
                SampledParam(uniform, loc=baseline[i], scale=1e-8)
            )
    return priors

# ------------------------------------------------------------
# Heatmap-only joint posterior plot
# ------------------------------------------------------------
def plot_joint_heatmap(posterior, i, j, name_i, name_j, save_path):
    x = posterior[:, :, i].ravel()
    y = posterior[:, :, j].ravel()

    fig, ax = plt.subplots(figsize=(6, 5))

    h = ax.hist2d(
        x, y,
        bins=60,
        density=True,
        cmap="inferno"
    )

    ax.axvline(baseline[i], color='cyan', linestyle='--', linewidth=1)
    ax.axhline(baseline[j], color='cyan', linestyle='--', linewidth=1)

    ax.set_xlabel(name_i)
    ax.set_ylabel(name_j)

    corr = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"{name_i} vs {name_j} (corr = {corr:.2f})")

    plt.colorbar(h[3], ax=ax, label="Posterior density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Correlation({name_i}, {name_j}) = {corr:.3f}")

# ------------------------------------------------------------
# Pairwise inversion
# ------------------------------------------------------------
def pairwise_inversion(free_indices, niterations=3000, nchains=3):
    priors = build_priors(free_indices)
    pair_name = '_'.join([param_names[i] for i in free_indices])
    model_name = f"{TIMESTAMP}_{pair_name}"

    sampled_params, _ = run_dream(
        parameters=priors,
        likelihood=likelihood_wrapper,
        niterations=niterations,
        nchains=nchains,
        model_name=model_name
    )

    posterior = np.stack([np.asarray(chain) for chain in sampled_params])

    os.makedirs("posteriors", exist_ok=True)

    i, j = free_indices
    plot_joint_heatmap(
        posterior,
        i, j,
        param_names[i], param_names[j],
        save_path=f"posteriors/{model_name}_heatmap.png"
    )

# ------------------------------------------------------------
# MAIN: run all two-parameter combinations
# ------------------------------------------------------------
if __name__ == "__main__":

    all_pairs = list(combinations(range(len(param_names)), 2))

    for i, j in all_pairs:
        print(f"\nRunning inversion: {param_names[i]} vs {param_names[j]}")
        pairwise_inversion([i, j])
