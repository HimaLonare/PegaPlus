import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

fingerprint_path = "data/processed/cox2_fingerprints.csv"
cluster_path = "data/processed/cox2_clustered.csv"

fingerprints_df = pd.read_csv(fingerprint_path)
cluster_df = pd.read_csv(cluster_path)

# Extract 1024 molecular fingerprint features
X = fingerprints_df.iloc[:, :1024].values

# Correct column names
clusters = cluster_df["predicted_cluster"].values
labels = cluster_df["reference_label"].values

print("Dataset loaded successfully!")
print("Fingerprint shape:", X.shape)

# ==========================================
# STEP 2: PCA DIMENSIONALITY REDUCTION
# ==========================================

pca = PCA(n_components=2)

X_2d = pca.fit_transform(X)

explained_variance = sum(pca.explained_variance_ratio_) * 100

print("\nPCA completed successfully!")
print("Reduced dimensions:", X_2d.shape)

print(
    "Explained variance:",
    round(explained_variance, 2),
    "%"
)

# ==========================================
# STEP 3: PREDICTED CLUSTER VISUALIZATION
# ==========================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=clusters,
    cmap="tab10",
    s=60,
    alpha=0.8
)

plt.title("PegaPlus - Predicted Molecular Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(scatter, label="Predicted Cluster")

plt.tight_layout()

plt.savefig(
    "results/predicted_clusters.png",
    dpi=300
)

plt.show()

# ==========================================
# STEP 4: REFERENCE LABEL VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=labels,
    cmap="tab10",
    s=60,
    alpha=0.8
)

plt.title("PegaPlus - Reference Molecular Labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(scatter, label="Reference Label")

plt.tight_layout()

plt.savefig(
    "results/reference_labels.png",
    dpi=300
)

plt.show()

# ==========================================
# STEP 5: COMPLETION
# ==========================================

print("\n==========================================")
print("VISUALIZATION COMPLETED SUCCESSFULLY!")
print("==========================================")

print("Generated files:")
print("1. results/predicted_clusters.png")
print("2. results/reference_labels.png")