import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

BASELINE_PATH = PROJECT_DIR / "data/processed/cox2_clustered.csv"
CONSTRAINED_PATH = PROJECT_DIR / "data/processed/constrained_results.csv"

print("=" * 60)
print("       PEGAPLUS: HUMAN FEEDBACK IMPACT ANALYSIS")
print("=" * 60)

# Load datasets
baseline = pd.read_csv(BASELINE_PATH)
constrained = pd.read_csv(CONSTRAINED_PATH)

# Compare cluster assignments
comparison = pd.DataFrame({
    "Molecule_ID": range(1, len(baseline) + 1),
    "Original_Cluster": baseline["predicted_cluster"],
    "Constrained_Cluster": constrained["constrained_cluster"]
})

# Identify changed molecules
changed = comparison[
    comparison["Original_Cluster"] != comparison["Constrained_Cluster"]
]

# Display results
print("\nTotal molecules:", len(comparison))
print("Molecules whose assignments changed:", len(changed))

percentage = len(changed) / len(comparison) * 100

print(f"Percentage of changed assignments: {percentage:.2f}%")

print("\nCHANGED MOLECULES")
print("-" * 60)

if len(changed) > 0:
    print(changed.to_string(index=False))
else:
    print("No molecule assignments changed.")

# Save results
output_path = PROJECT_DIR / "results"
output_path.mkdir(exist_ok=True)

comparison.to_csv(
    output_path / "feedback_impact_analysis.csv",
    index=False
)

changed.to_csv(
    output_path / "changed_molecules.csv",
    index=False
)

print("\nGenerated files:")
print("1. results/feedback_impact_analysis.csv")
print("2. results/changed_molecules.csv")

print("\nANALYSIS COMPLETED!")
print("=" * 60)