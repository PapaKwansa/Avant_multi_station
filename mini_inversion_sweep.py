# full_inversion_sweep.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from forward_model_multi_station import forward_model_multi_station

# -------------------- USER INPUT --------------------
df_obs = pd.read_csv("avant_cleaned_strain.csv")  # observed dataset
time_vals = df_obs["time_s"].values
strain_cols = [col for col in df_obs.columns if "strain_" in col]

stations_df = pd.read_csv("AVANT_stations.csv")
x_prime = stations_df["x_prime"].values
y_prime = stations_df["y_prime"].values
z = stations_df["depth"].values

# -------------------- PARAMETER BOUNDS --------------------
# Include all parameters that affect the forward model
param_names = [
    "s", "E", "theta_deg", "sigma", "h",
    "x0_prime", "y0_prime", "nu", "alpha", "pmax", "tpeak", "d"
]

param_bounds = [
    (50, 200),             # s
    (0.8e10, 1.2e10),      # E
    (-30, 0),              # theta_deg
    (1e-9, 1e-8),          # sigma (optional)
    (1000, 4000),          # h
    (-100, 0),             # x0_prime
    (-150, -50),           # y0_prime
    (0.2, 0.35),           # nu
    (0.5, 1.0),            # alpha
    (5e6, 15e6),           # pmax
    (5, 15),               # tpeak
    (1, 5)                 # d
]

# -------------------- GEOMETRY RATIOS --------------------
a0 = 1.0
b0 = 25.0 / 175.0
c0 = 125.0 / 175.0

# -------------------- RMSE FUNCTION --------------------
def compute_rmse(params):
    s_val, E_val, theta_val, sigma_val, h_val, x0_val, y0_val, nu_val, alpha_val, pmax_val, tpeak_val, d_val = params
    a, b, c = s_val*a0, s_val*b0, s_val*c0
    df_pred = forward_model_multi_station(
        pmax=pmax_val, tpeak=tpeak_val, d=d_val,
        time=time_vals,
        x_prime=x_prime, y_prime=y_prime,
        x0_prime=x0_val, y0_prime=y0_val, z=z,
        a=a, b=b, c=c, nu=nu_val, h=h_val,
        E=E_val, theta_deg=theta_val, alpha=alpha_val
    )
    pred = df_pred[strain_cols].values
    obs = df_obs[strain_cols].values
    rmse = np.sqrt(np.mean((pred - obs)**2))
    return rmse

# -------------------- PARAMETER SENSITIVITY SWEEP --------------------
def sweep_parameter(idx, n_points=15):
    sweep_vals = np.linspace(param_bounds[idx][0], param_bounds[idx][1], n_points)
    rmse_vals = []
    mid_vals = [(low+high)/2 for (low, high) in param_bounds]
    for val in sweep_vals:
        params = mid_vals.copy()
        params[idx] = val
        rmse_vals.append(compute_rmse(params))
    plt.figure()
    plt.plot(sweep_vals, rmse_vals, 'o-', lw=2)
    plt.xlabel(param_names[idx])
    plt.ylabel("RMSE")
    plt.title(f"Sensitivity sweep: {param_names[idx]}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"sensitivity_{param_names[idx]}.png", dpi=300)
    plt.close()
    print(f"Sensitivity plot saved: sensitivity_{param_names[idx]}.png")

# Run sweeps for all parameters
for i in range(len(param_bounds)):
    sweep_parameter(i)

# -------------------- OPTIMIZATION --------------------
result = differential_evolution(compute_rmse, param_bounds, maxiter=150, seed=42)
optimal_params = result.x
min_rmse = result.fun
print("\nOptimal parameters found:")
for name, val in zip(param_names, optimal_params):
    print(f"{name}: {val:.4g}")
print(f"Minimum RMSE: {min_rmse:.4f}")

# -------------------- VALIDATE OPTIMAL PARAMETERS --------------------
s_opt, E_opt, theta_opt, sigma_opt, h_opt, x0_opt, y0_opt, nu_opt, alpha_opt, pmax_opt, tpeak_opt, d_opt = optimal_params
a_opt, b_opt, c_opt = s_opt*a0, s_opt*b0, s_opt*c0
df_pred_opt = forward_model_multi_station(
    pmax=pmax_opt, tpeak=tpeak_opt, d=d_opt,
    time=time_vals, x_prime=x_prime, y_prime=y_prime,
    x0_prime=x0_opt, y0_prime=y0_opt, z=z,
    a=a_opt, b=b_opt, c=c_opt, nu=nu_opt, h=h_opt,
    E=E_opt, theta_deg=theta_opt, alpha=alpha_opt
)

# Plot predicted vs observed strains
plt.figure(figsize=(12,6))
for i in range(min(6, len(strain_cols))):
    plt.plot(time_vals, df_obs[strain_cols[i]], 'k.', label=f"Obs {strain_cols[i]}")
    plt.plot(time_vals, df_pred_opt[strain_cols[i]], '-', label=f"Pred {strain_cols[i]}")
plt.xlabel("Time (s)")
plt.ylabel("Strain (nε)")
plt.title("Predicted vs Observed Strains (First 6 components)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("pred_vs_obs_strains_full.png", dpi=300)
plt.close()
print("Predicted vs observed plot saved: pred_vs_obs_strains_full.png")