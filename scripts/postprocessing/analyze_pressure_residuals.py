
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "results/pressure_superposition/larry_observed_fulltime_001"
)

files = {
    "uniform": ROOT / "uniform_channel_predictions.csv",
    "four_zone": ROOT / "four_zone_channel_predictions.csv",
}

frames = []

for model, path in files.items():
    df = pd.read_csv(path)

    # Channel naming convention: eXX_S01, eYY_S02, etc.
    parts = df["channel"].str.rsplit("_", n=1)

    df["component"] = parts.str[0]
    df["station"] = parts.str[1]
    df["model"] = model

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

def summarize(group):
    residual = group["residual_ns"].to_numpy(dtype=float)

    return pd.Series({
        "n": len(residual),
        "rmse_ns": np.sqrt(np.mean(residual ** 2)),
        "bias_ns": np.mean(residual),
        "mae_ns": np.mean(np.abs(residual)),
        "max_abs_ns": np.max(np.abs(residual)),
    })

by_station = (
    data.groupby(["model", "set", "station"])
    .apply(summarize)
    .reset_index()
)

by_component = (
    data.groupby(["model", "set", "component"])
    .apply(summarize)
    .reset_index()
)

by_station_component = (
    data.groupby(["model", "set", "station", "component"])
    .apply(summarize)
    .reset_index()
)

by_station.to_csv(ROOT / "residuals_by_station.csv", index=False)
by_component.to_csv(ROOT / "residuals_by_component.csv", index=False)
by_station_component.to_csv(
    ROOT / "residuals_by_station_component.csv", index=False
)

print("\nRMSE BY STATION")
print(by_station.round(4).to_string(index=False))

print("\nRMSE BY COMPONENT")
print(by_component.round(4).to_string(index=False))

print("\nLargest four-zone station/component RMSE:")
print(
    by_station_component[
        by_station_component["model"] == "four_zone"
    ]
    .sort_values("rmse_ns", ascending=False)
    .head(12)
    .round(4)
    .to_string(index=False)
)
