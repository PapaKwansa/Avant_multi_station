# ------------------------------------------------------------
# test_identifiability_theta_geometry_pairwise_joint_marginal.py
#
# Pairwise Bayesian identifiability analysis:
# - Two-parameter inversion only
# - Joint posterior + marginals in one figure
# - KDE overlays and true-value markers
# ------------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, uniform
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
# Synthetic observations (truth)
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
# Likelihood wrapper
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
        component_cols=COMPONENT_COLS
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
                priors.append(SampledParam(uniform, loc=baseline[i] - 10, scale=20))
            else:
                priors.append(SampledParam(uniform, loc=0.8 * baseline[i], scale=0.4 * baseline[i]))
        else:
            priors.append(SampledParam(uniform, loc=baseline[i], scale=1e-8))
    return priors

# ------------------------------------------------------------
# Joint + marginal plotting (BEST PRACTICE)
# ------------------------------------------------------------
def plot_joint_and_marginals(posterior, i, j, name_i, name_j, save_path):
    x = posterior[:, :, i].ravel()
    y = posterior[:, :, j].ravel()

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4],
                          wspace=0.05, hspace=0.05)

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    # --- Joint histogram
    ax_joint.hist2d(x, y, bins=40, density=True, cmap='Greys', alpha=0.4)

    # --- KDE contours
    kde = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    ax_joint.contour(xi, yi, zi, colors='tab:blue', linewidths=1.5)

    # --- True values
    ax_joint.axvline(baseline[i], color='red', linestyle='--')
    ax_joint.axhline(baseline[j], color='red', linestyle='--')

    ax_joint.set_xlabel(name_i)
    ax_joint.set_ylabel(name_j)

    # --- Marginal X
    ax_x.hist(x, bins=30, density=True, color='gray', alpha=0.6)
    kde_x = gaussian_kde(x)
    xx = np.linspace(x.min(), x.max(), 300)
    ax_x.plot(xx, kde_x(xx), color='tab:blue')
    ax_x.axvline(baseline[i], color='red', linestyle='--')
    ax_x.axis('off')

    # --- Marginal Y
    ax_y.hist(y, bins=30, density=True, orientation='horizontal',
              color='gray', alpha=0.6)
    kde_y = gaussian_kde(y)
    yy = np.linspace(y.min(), y.max(), 300)
    ax_y.plot(kde_y(yy), yy, color='tab:blue')
    ax_y.axhline(baseline[j], color='red', linestyle='--')
    ax_y.axis('off')

    corr = np.corrcoef(x, y)[0, 1]
    ax_joint.set_title(f"{name_i} vs {name_j}  (corr = {corr:.2f})")

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
    plot_joint_and_marginals(
        posterior,
        i, j,
        param_names[i], param_names[j],
        save_path=f"posteriors/{model_name}_joint_marginal.png"
    )

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    theta_idx = param_names.index('theta_deg')
    geom_indices = [param_names.index(p) for p in ['a', 'b', 'c', 'E']]

    for gi in geom_indices:
        print(f"\nRunning inversion: theta_deg vs {param_names[gi]}")
        pairwise_inversion([theta_idx, gi])
