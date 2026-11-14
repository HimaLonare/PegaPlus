import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================
# PEGAPLUS: HUMAN FEEDBACK VALIDATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEEDBACK_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "human_feedback.csv"
)

FINGERPRINT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "cox2_fingerprints.csv"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 65)
print("       PEGAPLUS: HUMAN FEEDBACK VALIDATION")
print("=" * 65)


# ============================================================
# STEP 1: LOAD DATASETS
# ============================================================

print("\n[STEP 1] Loading datasets...")

feedback_df = pd.read_csv(FEEDBACK_PATH)
fingerprint_df = pd.read_csv(FINGERPRINT_PATH)

# Identify fingerprint columns
fingerprint_columns = [
    col for col in fingerprint_df.columns
    if col.startswith("FP_")
]

if len(fingerprint_columns) == 0:
    raise ValueError("No fingerprint columns beginning with FP_ were found.")

fingerprints = fingerprint_df[fingerprint_columns].to_numpy()

# Convert fingerprints into binary values
fingerprints = (fingerprints > 0).astype(int)

print("Total molecules:", len(fingerprints))
print("Fingerprint dimensions:", fingerprints.shape[1])
print("Total feedback records:", len(feedback_df))


# ============================================================
# STEP 2: NORMALIZE HUMAN FEEDBACK
# ============================================================

print("\n[STEP 2] Processing human feedback...")

if "relationship" in feedback_df.columns:
    feedback_df["relationship"] = (
        feedback_df["relationship"]
        .fillna(feedback_df.get("feedback", pd.Series(dtype=str)))
    )
elif "feedback" in feedback_df.columns:
    feedback_df["relationship"] = feedback_df["feedback"]
else:
    raise ValueError("No relationship or feedback column found.")

feedback_df["relationship"] = (
    feedback_df["relationship"]
    .astype(str)
    .str.strip()
    .str.capitalize()
)

feedback_df = feedback_df[
    feedback_df["relationship"].isin(["Similar", "Dissimilar"])
].copy()

feedback_df["molecule_1"] = pd.to_numeric(
    feedback_df["molecule_1"], errors="coerce"
)

feedback_df["molecule_2"] = pd.to_numeric(
    feedback_df["molecule_2"], errors="coerce"
)

feedback_df = feedback_df.dropna(
    subset=["molecule_1", "molecule_2"]
)

feedback_df["molecule_1"] = feedback_df["molecule_1"].astype(int)
feedback_df["molecule_2"] = feedback_df["molecule_2"].astype(int)

# Remove invalid molecule IDs
valid_count = len(fingerprints)

feedback_df = feedback_df[
    feedback_df["molecule_1"].between(1, valid_count)
    & feedback_df["molecule_2"].between(1, valid_count)
]

print("Valid feedback records:", len(feedback_df))


# ============================================================
# STEP 3: CALCULATE TANIMOTO SIMILARITY
# ============================================================

print("\n[STEP 3] Calculating molecular similarity...")


def calculate_tanimoto(fp1, fp2):

    intersection = np.sum(fp1 * fp2)

    union = np.sum((fp1 + fp2) > 0)

    if union == 0:
        return 0.0

    return intersection / union


results = []

for _, row in feedback_df.iterrows():

    molecule_1 = int(row["molecule_1"])
    molecule_2 = int(row["molecule_2"])

    # Molecule IDs start at 1, Python indexing starts at 0
    fp1 = fingerprints[molecule_1 - 1]
    fp2 = fingerprints[molecule_2 - 1]

    similarity = calculate_tanimoto(fp1, fp2)

    results.append({
        "molecule_1": molecule_1,
        "molecule_2": molecule_2,
        "human_feedback": row["relationship"],
        "tanimoto_similarity": similarity
    })


validation_df = pd.DataFrame(results)

print("Tanimoto similarity calculated for all valid feedback pairs.")


# ============================================================
# STEP 4: SEPARATE FEEDBACK GROUPS
# ============================================================

print("\n[STEP 4] Comparing human feedback groups...")

similar_scores = validation_df.loc[
    validation_df["human_feedback"] == "Similar",
    "tanimoto_similarity"
].to_numpy()

dissimilar_scores = validation_df.loc[
    validation_df["human_feedback"] == "Dissimilar",
    "tanimoto_similarity"
].to_numpy()


def calculate_statistics(scores):

    if len(scores) == 0:
        return {
            "Count": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "Minimum": np.nan,
            "Maximum": np.nan
        }

    return {
        "Count": len(scores),
        "Mean": np.mean(scores),
        "Median": np.median(scores),
        "Minimum": np.min(scores),
        "Maximum": np.max(scores)
    }


similar_stats = calculate_statistics(similar_scores)
dissimilar_stats = calculate_statistics(dissimilar_scores)

summary_df = pd.DataFrame([
    {
        "Human Feedback": "Similar",
        **similar_stats
    },
    {
        "Human Feedback": "Dissimilar",
        **dissimilar_stats
    }
])

print("\nSIMILAR PAIRS")
print("------------------------------")
print("Count:", similar_stats["Count"])
print("Mean Tanimoto:", similar_stats["Mean"])
print("Median Tanimoto:", similar_stats["Median"])

print("\nDISSIMILAR PAIRS")
print("------------------------------")
print("Count:", dissimilar_stats["Count"])
print("Mean Tanimoto:", dissimilar_stats["Mean"])
print("Median Tanimoto:", dissimilar_stats["Median"])


# ============================================================
# STEP 5: STATISTICAL VALIDATION
# ============================================================

print("\n[STEP 5] Performing statistical evaluation...")

mann_whitney_p = np.nan
auc_score = np.nan
spearman_corr = np.nan
spearman_p = np.nan

if len(similar_scores) > 0 and len(dissimilar_scores) > 0:

    # Mann-Whitney U test
    test_result = mannwhitneyu(
        similar_scores,
        dissimilar_scores,
        alternative="greater"
    )

    mann_whitney_p = test_result.pvalue

    print("Mann-Whitney U statistic:", test_result.statistic)
    print("Mann-Whitney p-value:", mann_whitney_p)

    # AUC: how well Tanimoto similarity separates human labels
    binary_labels = (
        validation_df["human_feedback"] == "Similar"
    ).astype(int)

    auc_score = roc_auc_score(
        binary_labels,
        validation_df["tanimoto_similarity"]
    )

    print("ROC-AUC:", auc_score)

    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(
        binary_labels,
        validation_df["tanimoto_similarity"]
    )

    print("Spearman correlation:", spearman_corr)
    print("Spearman p-value:", spearman_p)

else:

    print(
        "Statistical comparison requires at least one "
        "Similar and one Dissimilar feedback record."
    )


# ============================================================
# STEP 6: GENERATE INTERPRETATION
# ============================================================

print("\n[STEP 6] Interpreting results...")

if len(similar_scores) > 0 and len(dissimilar_scores) > 0:

    similar_mean = np.mean(similar_scores)
    dissimilar_mean = np.mean(dissimilar_scores)

    difference = similar_mean - dissimilar_mean

    print("Mean similarity difference:", difference)

    if difference > 0:
        print(
            "Human Similar judgments have higher average "
            "fingerprint similarity."
        )

    elif difference < 0:
        print(
            "Human Similar judgments have lower average "
            "fingerprint similarity."
        )

    else:
        print(
            "Both groups have equal average fingerprint similarity."
        )

    if mann_whitney_p < 0.05:
        print(
            "The difference is statistically significant "
            "at the 0.05 level."
        )
    else:
        print(
            "The difference is not statistically significant "
            "at the 0.05 level."
        )

    print(
        "\nNote: Tanimoto similarity measures structural similarity, "
        "not biological activity."
    )


# ============================================================
# STEP 7: SAVE VALIDATION RESULTS
# ============================================================

print("\n[STEP 7] Saving validation results...")

pair_output = os.path.join(
    RESULTS_DIR,
    "human_feedback_validation.csv"
)

summary_output = os.path.join(
    RESULTS_DIR,
    "human_feedback_validation_summary.csv"
)

validation_df.to_csv(pair_output, index=False)
summary_df.to_csv(summary_output, index=False)

print("Saved:", pair_output)
print("Saved:", summary_output)


# ============================================================
# STEP 8: VISUALIZATION
# ============================================================

print("\n[STEP 8] Generating visualization...")

if len(similar_scores) > 0 and len(dissimilar_scores) > 0:

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        [similar_scores, dissimilar_scores],
        tick_labels=["Human: Similar", "Human: Dissimilar"]
    )

    plt.ylabel("Tanimoto Similarity")
    plt.title("Human Feedback vs Molecular Structural Similarity")

    plt.grid(axis="y", alpha=0.3)

    plot_path = os.path.join(
        RESULTS_DIR,
        "human_feedback_validation.png"
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Visualization saved:", plot_path)

else:

    print("Visualization skipped: both feedback groups are required.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("       HUMAN FEEDBACK VALIDATION COMPLETED")
print("=" * 65)

print("Total validated pairs:", len(validation_df))
print("Similar pairs:", len(similar_scores))
print("Dissimilar pairs:", len(dissimilar_scores))

print("\nGenerated files:")
print("1. human_feedback_validation.csv")
print("2. human_feedback_validation_summary.csv")
print("3. human_feedback_validation.png")

print("\nVALIDATION COMPLETED SUCCESSFULLY!")
print("=" * 65)