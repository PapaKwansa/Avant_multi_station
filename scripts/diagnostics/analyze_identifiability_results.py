# analyze_identifiability_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Settings
# -------------------------
RESULTS_FILE = 'multi_station_sampled_params.npy'  # replace with your output file from first test
PLOT_DIR = "identifiability_analysis"
os.makedirs(PLOT_DIR, exist_ok=True)

param_names = ['a', 'b', 'c', 'E', 'theta_deg', 'sigma']
baseline = [120.0, 20.0, 80.0, 1e10, -17.0, 0.05]  # replace with your baseline values

# -------------------------
# Load results
# -------------------------
chains_all = np.load(RESULTS_FILE)  # shape: (nchains, niterations, nparams)
nchains, niterations, nparams = chains_all.shape
print(f"Loaded chains: {chains_all.shape}")

# Flatten chains for analysis
flattened = chains_all.reshape((-1, nparams))  # shape: (nchains*niterations, nparams)

# -------------------------
# Posterior histograms
# -------------------------
for i, pname in enumerate(param_names):
    plt.figure(figsize=(6,4))
    plt.hist(flattened[:,i], bins=50, density=True, alpha=0.7, color='skyblue')
    plt.axvline(baseline[i], color='r', linestyle='--', label='Baseline')
    plt.title(f"Posterior distribution: {pname}")
    plt.xlabel(pname)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{pname}_posterior.png"))
    plt.close()

# -------------------------
# Pairwise correlation plots
# -------------------------
df_flat = pd.DataFrame(flattened, columns=param_names)
sns.pairplot(df_flat, kind='scatter', diag_kind='kde', plot_kws={'alpha':0.5})
plt.suptitle("Pairwise scatter and KDE plots of posterior samples", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pairwise_posteriors.png"))
plt.close()

# -------------------------
# Posterior summary statistics
# -------------------------
summary_list = []
for i, pname in enumerate(param_names):
    combined = flattened[:,i]
    summary_list.append({
        'parameter': pname,
        'baseline': baseline[i],
        'mean': np.mean(combined),
        'median': np.median(combined),
        'std': np.std(combined),
        'min': np.min(combined),
        'max': np.max(combined)
    })

df_summary = pd.DataFrame(summary_list)
summary_csv = os.path.join(PLOT_DIR, "posterior_summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"Saved summary CSV to {summary_csv}")
print("Posterior plots saved in folder:", PLOT_DIR)
