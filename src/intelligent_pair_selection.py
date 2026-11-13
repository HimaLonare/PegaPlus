import os
import pandas as pd
import numpy as np
from itertools import combinations


# ============================================================
# PEGAPLUS: INTELLIGENT MOLECULE PAIR SELECTION
# ============================================================

print("=" * 65)
print("       PEGAPLUS: INTELLIGENT MOLECULE PAIR SELECTION")
print("=" * 65)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/cox2_clustered.csv"
FEEDBACK_PATH = "data/processed/human_feedback.csv"
OUTPUT_PATH = "results/recommended_pairs.csv"

df = pd.read_csv(DATA_PATH)

print("\nTotal molecules:", len(df))


# ============================================================
# 2. EXTRACT MORGAN FINGERPRINTS
# ============================================================

fingerprint_columns = [
    col for col in df.columns if col.startswith("FP_")
]

if not fingerprint_columns:
    raise ValueError(
        "No fingerprint columns starting with 'FP_' found. "
        "Check cox2_clustered.csv."
    )

fingerprints = df[fingerprint_columns].to_numpy(dtype=np.uint8)

print("Fingerprint matrix shape:", fingerprints.shape)


# ============================================================
# 3. EXTRACT CLUSTER INFORMATION
# ============================================================

if "predicted_cluster" not in df.columns:
    raise ValueError(
        "Column 'predicted_cluster' not found in the dataset."
    )

clusters = df["predicted_cluster"].to_numpy()

# Molecule IDs are 1-based, matching feedback_app.py
molecule_ids = np.arange(1, len(df) + 1)


# ============================================================
# 4. LOAD EXISTING HUMAN FEEDBACK
# ============================================================

existing_pairs = set()

if os.path.exists(FEEDBACK_PATH):

    feedback_df = pd.read_csv(FEEDBACK_PATH)

    # Find the molecule ID columns
    column_lookup = {
        col.lower(): col for col in feedback_df.columns
    }

    first_col = next(
        (column_lookup[name] for name in
         ["molecule_1", "molecule1", "mol1"]
         if name in column_lookup),
        None
    )

    second_col = next(
        (column_lookup[name] for name in
         ["molecule_2", "molecule2", "mol2"]
         if name in column_lookup),
        None
    )

    if first_col and second_col:

        for _, row in feedback_df.iterrows():

            pair = tuple(sorted((
                int(row[first_col]),
                int(row[second_col])
            )))

            existing_pairs.add(pair)

    else:
        print("Warning: Could not identify feedback molecule columns.")

print("Previously reviewed pairs:", len(existing_pairs))


# ============================================================
# 5. CALCULATE TANIMOTO SIMILARITY
# ============================================================

def tanimoto_similarity(fp1, fp2):

    intersection = np.sum((fp1 == 1) & (fp2 == 1))

    union = np.sum((fp1 == 1) | (fp2 == 1))

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# 6. GENERATE ALL MOLECULE PAIRS
# ============================================================

all_pairs = []

for i, j in combinations(range(len(fingerprints)), 2):

    molecule_1 = int(molecule_ids[i])
    molecule_2 = int(molecule_ids[j])

    pair_key = tuple(sorted((molecule_1, molecule_2)))

    # Skip pairs that already have human feedback
    if pair_key in existing_pairs:
        continue

    similarity = tanimoto_similarity(
        fingerprints[i],
        fingerprints[j]
    )

    all_pairs.append({
        "molecule_1": molecule_1,
        "molecule_2": molecule_2,
        "cluster_1": int(clusters[i]),
        "cluster_2": int(clusters[j]),
        "tanimoto_similarity": similarity,
        "different_clusters": clusters[i] != clusters[j]
    })


pairs_df = pd.DataFrame(all_pairs)

if pairs_df.empty:
    print("\nNo new molecule pairs available.")
    raise SystemExit


print("\nTotal new candidate pairs:", len(pairs_df))


# ============================================================
# 7. SELECT HIGH-SIMILARITY PAIRS
# ============================================================

high_similarity = pairs_df.sort_values(
    by="tanimoto_similarity",
    ascending=False
).head(10).copy()

high_similarity["selection_strategy"] = "High Similarity"


# ============================================================
# 8. SELECT LOW-SIMILARITY PAIRS
# ============================================================

low_similarity = pairs_df.sort_values(
    by="tanimoto_similarity",
    ascending=True
).head(10).copy()

low_similarity["selection_strategy"] = "Low Similarity"


# ============================================================
# 9. SELECT CLUSTER-BOUNDARY PAIRS
# ============================================================

boundary_candidates = pairs_df[
    pairs_df["different_clusters"] == True
]

boundary_pairs = boundary_candidates.sort_values(
    by="tanimoto_similarity",
    ascending=False
).head(10).copy()

boundary_pairs["selection_strategy"] = "Cluster Boundary"


# ============================================================
# 10. COMBINE RECOMMENDED PAIRS
# ============================================================

recommended_pairs = pd.concat(
    [
        high_similarity,
        low_similarity,
        boundary_pairs
    ],
    ignore_index=True
)

# Remove duplicate molecule pairs across strategies
recommended_pairs["pair_key"] = recommended_pairs.apply(
    lambda row: tuple(sorted((
        int(row["molecule_1"]),
        int(row["molecule_2"])
    ))),
    axis=1
)

recommended_pairs = recommended_pairs.drop_duplicates(
    subset="pair_key",
    keep="first"
)

recommended_pairs = recommended_pairs.drop(
    columns=["pair_key", "different_clusters"]
)

# Sort for easier reading
recommended_pairs = recommended_pairs.sort_values(
    by=["selection_strategy", "tanimoto_similarity"],
    ascending=[True, False]
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

recommended_pairs.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("             RECOMMENDED MOLECULE PAIRS")
print("=" * 65)

for strategy in [
    "High Similarity",
    "Low Similarity",
    "Cluster Boundary"
]:

    subset = recommended_pairs[
        recommended_pairs["selection_strategy"] == strategy
    ]

    print(f"\n--- {strategy} ---")

    if subset.empty:
        print("No pairs found.")
    else:
        print(
            subset[
                [
                    "molecule_1",
                    "molecule_2",
                    "tanimoto_similarity",
                    "cluster_1",
                    "cluster_2"
                ]
            ].to_string(index=False)
        )


print("\n" + "=" * 65)
print("                 SUMMARY")
print("=" * 65)

print("Total candidate pairs:", len(pairs_df))
print("Recommended pairs:", len(recommended_pairs))

print(
    "High similarity pairs:",
    len(recommended_pairs[
        recommended_pairs["selection_strategy"] == "High Similarity"
    ])
)

print(
    "Low similarity pairs:",
    len(recommended_pairs[
        recommended_pairs["selection_strategy"] == "Low Similarity"
    ])
)

print(
    "Cluster boundary pairs:",
    len(recommended_pairs[
        recommended_pairs["selection_strategy"] == "Cluster Boundary"
    ])
)

print("\nResults saved to:", OUTPUT_PATH)

print("\nIntelligent pair selection completed successfully!")