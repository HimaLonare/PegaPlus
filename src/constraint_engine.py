import pandas as pd
from pathlib import Path

# ==========================================
# PEGAPLUS: HUMAN FEEDBACK CONSTRAINT ENGINE
# ==========================================

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

FEEDBACK_PATH = (
    PROJECT_DIR / "data" / "processed" / "human_feedback.csv"
)

CLUSTER_PATH = (
    PROJECT_DIR / "data" / "processed" / "cox2_clustered.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

print("=" * 55)
print("       PEGAPLUS: CONSTRAINT ENGINE")
print("=" * 55)

if not FEEDBACK_PATH.exists():
    raise FileNotFoundError("Human feedback file not found!")

if not CLUSTER_PATH.exists():
    raise FileNotFoundError("Clustering dataset not found!")

feedback_df = pd.read_csv(FEEDBACK_PATH)
cluster_df = pd.read_csv(CLUSTER_PATH)

total_molecules = len(cluster_df)

print("\n[STEP 1] Loading datasets...")

print("Total molecules:", total_molecules)
print("Feedback records:", len(feedback_df))

# ==========================================
# STEP 2: INITIALIZE CONSTRAINTS
# ==========================================

must_link = []
cannot_link = []

invalid_records = []
pair_relationships = {}

print("\n[STEP 2] Processing human feedback...")

# ==========================================
# STEP 3: CONVERT FEEDBACK
# ==========================================

for _, row in feedback_df.iterrows():

    mol1 = int(row["molecule_1"])
    mol2 = int(row["molecule_2"])

    relationship = str(row["relationship"]).strip().lower()

    # Validate molecule IDs
    if not (1 <= mol1 <= total_molecules):
        invalid_records.append((mol1, mol2, "Invalid molecule ID"))
        continue

    if not (1 <= mol2 <= total_molecules):
        invalid_records.append((mol1, mol2, "Invalid molecule ID"))
        continue

    # A molecule cannot be compared with itself
    if mol1 == mol2:
        invalid_records.append((mol1, mol2, "Self-comparison"))
        continue

    # Normalize pair order
    pair = tuple(sorted((mol1, mol2)))

    if relationship == "similar":

        constraint = "must_link"

    elif relationship == "dissimilar":

        constraint = "cannot_link"

    else:

        invalid_records.append((mol1, mol2, "Unknown relationship"))
        continue

    pair_relationships.setdefault(pair, set()).add(constraint)


# ==========================================
# STEP 4: HANDLE DUPLICATES AND CONFLICTS
# ==========================================

conflicting_pairs = []

for pair, relationships in pair_relationships.items():

    if len(relationships) > 1:

        conflicting_pairs.append(pair)

    elif "must_link" in relationships:

        must_link.append(pair)

    elif "cannot_link" in relationships:

        cannot_link.append(pair)


# ==========================================
# STEP 5: DISPLAY RESULTS
# ==========================================

print("\n[STEP 3] Constraint generation completed!")

print("\nMUST-LINK CONSTRAINTS")
print("--------------------")

for pair in must_link:
    print(f"Molecule {pair[0]} <--> Molecule {pair[1]}")

print("\nCANNOT-LINK CONSTRAINTS")
print("----------------------")

for pair in cannot_link:
    print(f"Molecule {pair[0]} <--> Molecule {pair[1]}")

print("\nCONFLICTING PAIRS")
print("-----------------")

for pair in conflicting_pairs:
    print(pair)

print("\nINVALID RECORDS")
print("---------------")

for record in invalid_records:
    print(record)


# ==========================================
# STEP 6: SAVE CONSTRAINTS
# ==========================================

def save_constraints(pairs, filename):

    records = []

    for mol1, mol2 in pairs:

        records.append({
            "molecule_1": mol1,
            "molecule_2": mol2,

            # Convert IDs to zero-based array indices
            "index_1": mol1 - 1,
            "index_2": mol2 - 1
        })

    df = pd.DataFrame(
        records,
        columns=[
            "molecule_1",
            "molecule_2",
            "index_1",
            "index_2"
        ]
    )

    df.to_csv(OUTPUT_DIR / filename, index=False)


save_constraints(
    must_link,
    "must_link_constraints.csv"
)

save_constraints(
    cannot_link,
    "cannot_link_constraints.csv"
)


# ==========================================
# STEP 7: FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("             CONSTRAINT SUMMARY")
print("=" * 55)

print("Total feedback records:", len(feedback_df))
print("Must-link constraints:", len(must_link))
print("Cannot-link constraints:", len(cannot_link))
print("Conflicting pairs:", len(conflicting_pairs))
print("Invalid records:", len(invalid_records))

print("\nGenerated files:")

print("1. must_link_constraints.csv")
print("2. cannot_link_constraints.csv")

print("\nCONSTRAINT ENGINE COMPLETED SUCCESSFULLY!")
print("=" * 55)