import pandas as pd
import numpy as np
import os

BASELINE_PATH = os.getenv(
    "BASELINE_PATH",
    "/Users/rishwanthpb/MLOPS/MLOPS_PROJECT/data/processed/zone_demand_baseline.csv"
)

def load_baseline():
    """Load the 2019 baseline statistics saved during EDA."""
    baseline = pd.read_csv(BASELINE_PATH)
    return baseline.set_index("PULocationID")

def compute_kl_divergence(p, q):
    """
    Compute KL divergence between two distributions.
    p = live distribution
    q = baseline distribution
    Small value = similar, large value = drifted
    """
    # Add small epsilon to avoid log(0)
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10

    # Normalize to probabilities
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))

def compute_drift_score(live_demand_df):
    """
    Compare live demand distribution against 2019 baseline.
    live_demand_df must have columns: PULocationID, demand
    Returns a single drift score (float).
    """
    baseline = load_baseline()

    # Get mean demand per zone from live data
    live_mean = live_demand_df.groupby("PULocationID")["demand"].mean()

    # Find common zones
    common_zones = baseline.index.intersection(live_mean.index)

    if len(common_zones) < 10:
        print("Warning: fewer than 10 common zones — drift score unreliable")
        return 0.0

    baseline_values = baseline.loc[common_zones, "mean"].values
    live_values     = live_mean.loc[common_zones].values

    score = compute_kl_divergence(live_values, baseline_values)
    print(f"Drift score: {score:.4f} across {len(common_zones)} zones")
    return score

if __name__ == "__main__":
    # Quick test — simulate drift using 2020 April data
    features_dir = "/Users/rishwanthpb/MLOPS/MLOPS_PROJECT/data/features"

    # Load 2020-04 (COVID collapse — should show high drift)
    df_2020 = pd.read_parquet(f"{features_dir}/features_2020-04.parquet")
    score_2020 = compute_drift_score(df_2020)
    print(f"2020-04 drift score (expect HIGH): {score_2020:.4f}")

    # Load 2019-01 (baseline — should show low drift)
    df_2019 = pd.read_parquet(f"{features_dir}/features_2019-01.parquet")
    score_2019 = compute_drift_score(df_2019)
    print(f"2019-01 drift score (expect LOW):  {score_2019:.4f}")