import pandas as pd
import matplotlib.pyplot as plt

# Load the observed dataset
df_obs = pd.read_csv("avant_cleaned_strain.csv")

# Extract time and strain columns
time_vals = df_obs["time_s"].values
strain_cols = [col for col in df_obs.columns if "strain_" in col]

# Plot all strains
plt.figure(figsize=(12,6))
for col in strain_cols:
    plt.plot(time_vals, df_obs[col], label=col)
plt.xlabel("Time (s)")
plt.ylabel("Strain (nε)")
plt.title("Observed Strain Time Series (All Components)")
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()