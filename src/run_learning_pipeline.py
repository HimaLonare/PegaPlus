import subprocess
import sys
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Scripts to execute, in the correct order
PIPELINE_STEPS = [
    ("Convert human feedback into constraints", "src/constraint_engine.py"),
    ("Run constrained clustering", "src/constrained_clustering.py"),
    ("Compare baseline and constrained clustering", "src/compare_clustering.py"),
    ("Analyze the impact of human feedback", "src/analyze_feedback_impact.py"),
    ("Generate new intelligent molecule pairs", "src/intelligent_pair_selection.py"),
]


def run_pipeline():

    print("\n" + "=" * 65)
    print("       PEGAPLUS: COMPLETE LEARNING PIPELINE")
    print("=" * 65)

    print("\nProject directory:", PROJECT_ROOT)

    # Check whether all required scripts exist
    for description, script in PIPELINE_STEPS:
        script_path = PROJECT_ROOT / script

        if not script_path.exists():
            print(f"\nERROR: Required script not found: {script}")
            return

    # Check whether human feedback exists
    feedback_file = PROJECT_ROOT / "data" / "processed" / "human_feedback.csv"

    if not feedback_file.exists():
        print("\nERROR: Human feedback file not found.")
        print("Please run the Streamlit feedback application first.")
        return

    # Execute each pipeline step
    for step_number, (description, script) in enumerate(PIPELINE_STEPS, start=1):

        print("\n" + "-" * 65)
        print(f"STEP {step_number}/{len(PIPELINE_STEPS)}: {description}")
        print("-" * 65)

        try:
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script)],
                cwd=PROJECT_ROOT,
                check=True
            )

            print(f"\nSUCCESS: {description}")

        except subprocess.CalledProcessError:
            print(f"\nERROR: Pipeline stopped during: {description}")
            print("Fix the error before running the pipeline again.")
            return

    print("\n" + "=" * 65)
    print("       PEGAPLUS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    print("\nUpdated results are available in:")
    print("  data/processed/")
    print("  results/")

    print("\nYou can now inspect the updated clustering results.")


if __name__ == "__main__":
    run_pipeline()