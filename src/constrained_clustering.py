import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

# ==========================================
# PEGAPLUS: CONSTRAINED K-MEANS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data/processed/cox2_clustered.csv"

MUST_LINK_PATH = PROJECT_DIR / "data/processed/must_link_constraints.csv"

CANNOT_LINK_PATH = PROJECT_DIR / "data/processed/cannot_link_constraints.csv"

OUTPUT_PATH = PROJECT_DIR / "data/processed/constrained_results.csv"

print("=" * 60)
print("       PEGAPLUS: CONSTRAINED CLUSTERING")
print("=" * 60)


# ==========================================
# STEP 1: LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

fingerprint_columns = [
    col for col in df.columns if col.startswith("FP_")
]

X = df[fingerprint_columns].values.astype(float)

reference_labels = df["reference_label"].values

n_samples = len(X)
n_clusters = 8

print("\n[STEP 1] Dataset loaded")
print("Molecules:", n_samples)
print("Fingerprint dimensions:", X.shape[1])


# ==========================================
# STEP 2: LOAD CONSTRAINTS
# ==========================================

must_df = pd.read_csv(MUST_LINK_PATH)
cannot_df = pd.read_csv(CANNOT_LINK_PATH)

must_link = list(
    zip(must_df["index_1"], must_df["index_2"])
)

cannot_link = list(
    zip(cannot_df["index_1"], cannot_df["index_2"])
)

print("\n[STEP 2] Constraints loaded")
print("Must-link:", len(must_link))
print("Cannot-link:", len(cannot_link))


# ==========================================
# STEP 3: INITIALIZE CLUSTER CENTERS
# ==========================================

initial_model = KMeans(
    n_clusters=n_clusters,
    init="k-means++",
    n_init=10,
    random_state=42
)

initial_model.fit(X)

centers = initial_model.cluster_centers_.copy()

print("\n[STEP 3] Initial cluster centers generated")


# ==========================================
# STEP 4: PREPARE CONSTRAINT GRAPH
# ==========================================

must_neighbors = [[] for _ in range(n_samples)]
cannot_neighbors = [[] for _ in range(n_samples)]

for i, j in must_link:
    must_neighbors[i].append(j)
    must_neighbors[j].append(i)

for i, j in cannot_link:
    cannot_neighbors[i].append(j)
    cannot_neighbors[j].append(i)

constraint_degree = [
    len(must_neighbors[i]) + len(cannot_neighbors[i])
    for i in range(n_samples)
]

# Assign highly constrained molecules first
order = sorted(
    range(n_samples),
    key=lambda i: constraint_degree[i],
    reverse=True
)


# ==========================================
# STEP 5: COP-KMEANS ALGORITHM
# ==========================================

def constrained_kmeans(X, centers, max_iter=100):

    labels = np.full(len(X), -1, dtype=int)

    for iteration in range(max_iter):

        previous_labels = labels.copy()

        labels = np.full(len(X), -1, dtype=int)

        for i in order:

            # Calculate distance to each cluster center
            distances = np.sum(
                (centers - X[i]) ** 2,
                axis=1
            )

            allowed_clusters = set(range(n_clusters))

            # Apply must-link constraints
            required_clusters = {
                labels[j]
                for j in must_neighbors[i]
                if labels[j] != -1
            }

            if len(required_clusters) > 1:
                raise ValueError(
                    "Incompatible must-link constraints."
                )

            if required_clusters:
                allowed_clusters &= required_clusters

            # Apply cannot-link constraints
            forbidden_clusters = {
                labels[j]
                for j in cannot_neighbors[i]
                if labels[j] != -1
            }

            allowed_clusters -= forbidden_clusters

            if not allowed_clusters:
                raise ValueError(
                    f"No valid cluster available for molecule {i + 1}. "
                    "The constraints may be infeasible."
                )

            # Select nearest allowed cluster
            best_cluster = min(
                allowed_clusters,
                key=lambda c: distances[c]
            )

            labels[i] = best_cluster

        # Update cluster centers
        for cluster in range(n_clusters):

            members = X[labels == cluster]

            if len(members) > 0:
                centers[cluster] = members.mean(axis=0)

        # Check convergence
        if np.array_equal(labels, previous_labels):
            print("Convergence reached at iteration:", iteration + 1)
            break

    return labels


print("\n[STEP 4] Running COP-KMeans...")

predicted_labels = constrained_kmeans(
    X,
    centers
)

print("Constrained clustering completed!")


# ==========================================
# STEP 6: VERIFY CONSTRAINTS
# ==========================================

must_satisfied = sum(
    predicted_labels[i] == predicted_labels[j]
    for i, j in must_link
)

cannot_satisfied = sum(
    predicted_labels[i] != predicted_labels[j]
    for i, j in cannot_link
)

print("\n[STEP 5] Constraint verification")

print(
    f"Must-link satisfied: {must_satisfied}/{len(must_link)}"
)

print(
    f"Cannot-link satisfied: {cannot_satisfied}/{len(cannot_link)}"
)


# ==========================================
# STEP 7: EVALUATE PERFORMANCE
# ==========================================

ari = adjusted_rand_score(
    reference_labels,
    predicted_labels
)

nmi = normalized_mutual_info_score(
    reference_labels,
    predicted_labels
)

if len(np.unique(predicted_labels)) > 1:

    silhouette = silhouette_score(
        X,
        predicted_labels,
        metric="jaccard"
    )

else:
    silhouette = float("nan")


print("\n" + "=" * 60)
print("          CONSTRAINED CLUSTERING RESULTS")
print("=" * 60)

print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"Silhouette Score (Jaccard): {silhouette:.4f}")

print("=" * 60)


# ==========================================
# STEP 8: SAVE RESULTS
# ==========================================

df["constrained_cluster"] = predicted_labels

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nPEGAPLUS CONSTRAINED CLUSTERING COMPLETED!")
print("=" * 60)