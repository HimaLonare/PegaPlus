# ============================================================
# PEGAPLUS
# Interactive Machine Learning for Molecular Clustering
# Dataset: COX-2
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.cluster import KMeans

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)


# ============================================================
# STEP 1: PROJECT CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("                 PEGAPLUS PROJECT")
print("=" * 60)

BASE_DIR = Path(__file__).resolve().parent

SDF_PATH = (
    BASE_DIR / "reference" /
    "online_svm_evaluation" /
    "test_files" / "cox2.sdf"
)

LABEL_PATH = (
    BASE_DIR / "reference" /
    "online_svm_evaluation" /
    "test_files" / "cox2_correct.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 2: LOAD COX-2 DATASET
# ============================================================

print("\n[STEP 1] Loading COX-2 Dataset...")

if not SDF_PATH.exists():
    raise FileNotFoundError(f"SDF file not found: {SDF_PATH}")

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

supplier = Chem.SDMolSupplier(str(SDF_PATH))

molecules = [
    mol for mol in supplier
    if mol is not None
]

labels = pd.read_csv(
    LABEL_PATH,
    header=None
).iloc[:, 0].to_numpy()

if len(molecules) != len(labels):
    raise ValueError(
        f"Mismatch: {len(molecules)} molecules "
        f"but {len(labels)} labels."
    )

print("Total valid molecules:", len(molecules))
print("Total reference labels:", len(labels))

print("\nReference Label Distribution:")

unique_labels, label_counts = np.unique(
    labels, return_counts=True
)

for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} molecules")


# ============================================================
# STEP 3: GENERATE MORGAN FINGERPRINTS
# ============================================================

print("\n[STEP 2] Generating Morgan Fingerprints...")

RADIUS = 4
FP_SIZE = 1024

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=FP_SIZE
)

fingerprints = []

for mol in molecules:

    fp = generator.GetFingerprint(mol)

    arr = np.zeros(FP_SIZE, dtype=np.uint8)

    DataStructs.ConvertToNumpyArray(fp, arr)

    fingerprints.append(arr)

X = np.array(fingerprints, dtype=np.uint8)

y = labels

print("Fingerprint generation completed!")

print("Fingerprint matrix shape:", X.shape)
print("Fingerprint size:", FP_SIZE)
print("Active bits in first molecule:", np.sum(X[0]))


# ============================================================
# STEP 4: SAVE FINGERPRINT DATASET
# ============================================================

print("\n[STEP 3] Saving Fingerprint Dataset...")

columns = [
    f"FP_{i}" for i in range(FP_SIZE)
]

fingerprint_df = pd.DataFrame(
    X, columns=columns
)

fingerprint_df["reference_label"] = y

fingerprint_path = (
    OUTPUT_DIR / "cox2_fingerprints.csv"
)

fingerprint_df.to_csv(
    fingerprint_path,
    index=False
)

print("Fingerprint dataset saved:", fingerprint_path)


# ============================================================
# STEP 5: K-MEANS++ CLUSTERING
# ============================================================

print("\n[STEP 4] Performing K-Means++ Clustering...")

K = 8

kmeans = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=20,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X)

print("Clustering completed successfully!")
print("Number of clusters:", K)


# ============================================================
# STEP 6: CLUSTER DISTRIBUTION
# ============================================================

print("\n[STEP 5] Cluster Distribution:")

unique_clusters, cluster_counts = np.unique(
    cluster_labels,
    return_counts=True
)

for cluster, count in zip(
    unique_clusters, cluster_counts
):
    print(f"Cluster {cluster}: {count} molecules")


# ============================================================
# STEP 7: MODEL EVALUATION
# ============================================================

print("\n[STEP 6] Evaluating Clustering Performance...")

ari = adjusted_rand_score(
    y, cluster_labels
)

nmi = normalized_mutual_info_score(
    y, cluster_labels
)

silhouette = silhouette_score(
    X.astype(bool),
    cluster_labels,
    metric="jaccard"
)


# ============================================================
# STEP 8: DISPLAY PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 60)
print("             CLUSTERING PERFORMANCE")
print("=" * 60)

print(f"Adjusted Rand Index (ARI):      {ari:.4f}")

print(f"Normalized Mutual Information:  {nmi:.4f}")

print(f"Silhouette Score (Jaccard):      {silhouette:.4f}")

print("=" * 60)


# ============================================================
# STEP 9: SAVE CLUSTERING RESULTS
# ============================================================

print("\n[STEP 7] Saving Clustering Results...")

fingerprint_df["predicted_cluster"] = cluster_labels

clustered_path = (
    OUTPUT_DIR / "cox2_clustered.csv"
)

fingerprint_df.to_csv(
    clustered_path,
    index=False
)

print("Clustering results saved:", clustered_path)


# ============================================================
# STEP 10: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("                 PEGAPLUS SUMMARY")
print("=" * 60)

print("Dataset: COX-2")
print("Total molecules:", len(molecules))
print("Morgan fingerprint radius:", RADIUS)
print("Fingerprint dimensions:", FP_SIZE)
print("Number of clusters:", K)

print("\nPerformance:")
print("ARI:", round(ari, 4))
print("NMI:", round(nmi, 4))
print("Silhouette Score:", round(silhouette, 4))

print("\nGenerated Files:")
print("1. data/processed/cox2_fingerprints.csv")
print("2. data/processed/cox2_clustered.csv")

print("\nPEGAPLUS EXECUTION COMPLETED SUCCESSFULLY!")

print("=" * 60)


