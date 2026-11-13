import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

# ==========================================
# PEGAPLUS: CLUSTERING COMPARISON
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

BASELINE_PATH = PROJECT_DIR / "data/processed/cox2_clustered.csv"

CONSTRAINED_PATH = PROJECT_DIR / "data/processed/constrained_results.csv"

FINGERPRINT_PATH = PROJECT_DIR / "data/processed/cox2_fingerprints.csv"

print("=" * 60)
print("       PEGAPLUS: ALGORITHM COMPARISON")
print("=" * 60)

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

baseline = pd.read_csv(BASELINE_PATH)
constrained = pd.read_csv(CONSTRAINED_PATH)
fingerprints = pd.read_csv(FINGERPRINT_PATH)

X = fingerprints.iloc[:, :1024].values.astype(bool)

reference = baseline["reference_label"].values

baseline_labels = baseline["predicted_cluster"].values

constrained_labels = constrained["constrained_cluster"].values

print("\nDataset loaded successfully!")
print("Total molecules:", len(reference))

# ==========================================
# STEP 2: CALCULATE METRICS
# ==========================================

def evaluate(labels):

    ari = adjusted_rand_score(reference, labels)

    nmi = normalized_mutual_info_score(reference, labels)

    if len(np.unique(labels)) > 1:

        silhouette = silhouette_score(
            X,
            labels,
            metric="jaccard"
        )

    else:
        silhouette = np.nan

    return ari, nmi, silhouette


baseline_metrics = evaluate(baseline_labels)

constrained_metrics = evaluate(constrained_labels)

# ==========================================
# STEP 3: COMPARE CLUSTERING ASSIGNMENTS
# ==========================================

agreement = adjusted_rand_score(
    baseline_labels,
    constrained_labels
)

print("\nCLUSTERING AGREEMENT")
print("-" * 40)

print(
    "ARI between original and constrained clustering:",
    round(agreement, 4)
)

# ==========================================
# STEP 4: DISPLAY COMPARISON
# ==========================================

comparison = pd.DataFrame({

    "Metric": [
        "Adjusted Rand Index",
        "Normalized Mutual Information",
        "Silhouette Score (Jaccard)"
    ],

    "K-Means++": baseline_metrics,

    "Constrained K-Means": constrained_metrics

})

print("\nPERFORMANCE COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))

# ==========================================
# STEP 5: CLUSTER DISTRIBUTION
# ==========================================

print("\nCLUSTER DISTRIBUTION")
print("-" * 40)

print("\nOriginal K-Means++:")

print(
    pd.Series(baseline_labels)
    .value_counts()
    .sort_index()
)

print("\nConstrained K-Means:")

print(
    pd.Series(constrained_labels)
    .value_counts()
    .sort_index()
)

# ==========================================
# STEP 6: SAVE COMPARISON
# ==========================================

output_path = PROJECT_DIR / "results"

output_path.mkdir(exist_ok=True)

comparison.to_csv(
    output_path / "clustering_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")

print("=" * 60)
print("CLUSTERING COMPARISON COMPLETED!")
print("=" * 60)