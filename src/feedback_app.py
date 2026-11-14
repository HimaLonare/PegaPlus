import streamlit as st
import pandas as pd
import os
import sys
import subprocess

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw


# ============================================================
# PEGAPLUS: INTELLIGENT HUMAN FEEDBACK APPLICATION
# ============================================================

st.set_page_config(
    page_title="PegaPlus - Intelligent Feedback",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 PegaPlus: Intelligent Molecule Feedback")

st.markdown("""
### Human-in-the-loop molecular clustering

Compare chemical structures, evaluate molecular similarity,
and provide feedback to improve clustering.
""")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "cox2_clustered.csv"
)

RECOMMENDATION_PATH = (
    PROJECT_ROOT / "results" /
    "recommended_pairs.csv"
)

FEEDBACK_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "human_feedback.csv"
)

SDF_PATH = (
    PROJECT_ROOT / "reference" /
    "online_svm_evaluation" /
    "test_files" / "cox2.sdf"
)


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

PIPELINE_STEPS = [
    (
        "Convert human feedback into constraints",
        "src/constraint_engine.py"
    ),
    (
        "Run constrained clustering",
        "src/constrained_clustering.py"
    ),
    (
        "Compare baseline and constrained clustering",
        "src/compare_clustering.py"
    ),
    (
        "Analyze the impact of human feedback",
        "src/analyze_feedback_impact.py"
    ),
    (
        "Generate new intelligent molecule pairs",
        "src/intelligent_pair_selection.py"
    )
]


# ============================================================
# LOAD CHEMICAL STRUCTURES
# ============================================================

@st.cache_resource
def load_molecules():

    if not SDF_PATH.exists():
        raise FileNotFoundError(
            f"COX-2 SDF file not found: {SDF_PATH}"
        )

    supplier = Chem.SDMolSupplier(str(SDF_PATH))

    molecules = [
        mol for mol in supplier
        if mol is not None
    ]

    return molecules


try:
    molecules = load_molecules()

except Exception as error:
    st.error(
        f"Unable to load chemical structures: {error}"
    )
    st.stop()


# ============================================================
# DISPLAY CHEMICAL STRUCTURE
# ============================================================

def display_molecule(molecule_id):

    # Molecule IDs start from 1.
    # Python list indexing starts from 0.

    index = int(molecule_id) - 1

    if index < 0 or index >= len(molecules):
        st.error(
            f"Invalid molecule ID: {molecule_id}"
        )
        return

    mol = molecules[index]

    # Generate 2D chemical structure image.
    image = Draw.MolToImage(
        mol,
        size=(450, 350),
        kekulize=True
    )

    st.image(
        image,
        use_container_width=True
    )


# ============================================================
# RUN COMPLETE LEARNING PIPELINE
# ============================================================

def run_learning_pipeline():

    logs = []

    logs.append("=" * 60)
    logs.append("PEGAPLUS: COMPLETE LEARNING PIPELINE")
    logs.append("=" * 60)

    logs.append(
        f"Project directory: {PROJECT_ROOT}"
    )

    # Verify scripts.
    for description, script in PIPELINE_STEPS:

        script_path = PROJECT_ROOT / script

        if not script_path.exists():
            logs.append(
                f"ERROR: Script not found: {script}"
            )
            return False, "\n".join(logs)

    # Verify feedback file.
    if not FEEDBACK_PATH.exists():
        logs.append(
            "ERROR: Human feedback file not found."
        )
        return False, "\n".join(logs)

    # Execute pipeline.
    for step_number, (description, script) in enumerate(
        PIPELINE_STEPS,
        start=1
    ):

        logs.append("\n" + "-" * 60)
        logs.append(
            f"STEP {step_number}/{len(PIPELINE_STEPS)}: {description}"
        )
        logs.append("-" * 60)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / script)
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )

            logs.append(result.stdout)

            if result.stderr:
                logs.append("\nWARNINGS / STDERR:")
                logs.append(result.stderr)

            logs.append(
                f"\nSUCCESS: {description}"
            )

        except subprocess.TimeoutExpired:
            logs.append(
                f"ERROR: {description} exceeded timeout."
            )
            return False, "\n".join(logs)

        except subprocess.CalledProcessError as error:
            logs.append(
                f"ERROR: Pipeline stopped during: {description}"
            )
            logs.append(error.stdout or "")
            logs.append(error.stderr or "")
            return False, "\n".join(logs)

        except Exception as error:
            logs.append(
                f"UNEXPECTED ERROR: {str(error)}"
            )
            return False, "\n".join(logs)

    logs.append("\n" + "=" * 60)
    logs.append(
        "PEGAPLUS PIPELINE COMPLETED SUCCESSFULLY!"
    )
    logs.append("=" * 60)

    return True, "\n".join(logs)


# ============================================================
# PIPELINE CONTROL CENTER
# ============================================================

st.divider()

st.header("⚙️ Learning Pipeline Control Center")

st.markdown("""
Run the complete human-in-the-loop learning process.

1. Convert human feedback into constraints.
2. Perform constrained clustering.
3. Compare clustering algorithms.
4. Analyze the impact of human feedback.
5. Generate new molecule recommendations.
""")

if st.button(
    "🚀 Run Complete Learning Pipeline",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Running the complete PegaPlus learning pipeline..."
    ):

        success, output = run_learning_pipeline()

    st.session_state["pipeline_output"] = output
    st.session_state["pipeline_success"] = success

    if success:
        st.cache_data.clear()

        st.success(
            "PegaPlus learning pipeline completed successfully!"
        )

        st.rerun()

    else:
        st.error(
            "Pipeline execution failed. Check the logs."
        )


# ============================================================
# PIPELINE EXECUTION LOGS
# ============================================================

if "pipeline_output" in st.session_state:

    with st.expander(
        "📋 View Pipeline Execution Logs",
        expanded=True
    ):

        st.code(
            st.session_state["pipeline_output"],
            language="text"
        )


# ============================================================
# LOAD DATASETS
# ============================================================

@st.cache_data
def load_data():

    molecules_df = pd.read_csv(DATA_PATH)

    if RECOMMENDATION_PATH.exists():
        recommendations_df = pd.read_csv(
            RECOMMENDATION_PATH
        )
    else:
        recommendations_df = pd.DataFrame()

    return molecules_df, recommendations_df


df, recommendations = load_data()


# ============================================================
# LOAD HUMAN FEEDBACK
# ============================================================

def load_feedback():

    if FEEDBACK_PATH.exists():

        feedback = pd.read_csv(FEEDBACK_PATH)

        if "relationship" not in feedback.columns:
            feedback["relationship"] = None

        if "feedback" not in feedback.columns:
            feedback["feedback"] = None

        feedback["relationship"] = (
            feedback["relationship"].fillna(
                feedback["feedback"]
            )
        )

        feedback["feedback"] = (
            feedback["feedback"].fillna(
                feedback["relationship"]
            )
        )

        return feedback

    return pd.DataFrame(
        columns=[
            "molecule_1",
            "molecule_2",
            "relationship",
            "cluster_1",
            "cluster_2",
            "timestamp",
            "feedback"
        ]
    )


feedback_df = load_feedback()


# ============================================================
# SAVE HUMAN FEEDBACK
# ============================================================

def save_feedback(
    molecule_1,
    molecule_2,
    feedback,
    cluster_1,
    cluster_2
):

    global feedback_df

    new_record = {
        "molecule_1": int(molecule_1),
        "molecule_2": int(molecule_2),
        "relationship": feedback,
        "feedback": feedback,
        "cluster_1": int(cluster_1),
        "cluster_2": int(cluster_2),
        "timestamp": pd.Timestamp.now().isoformat()
    }

    new_record_df = pd.DataFrame(
        [new_record]
    )

    feedback_df = pd.concat(
        [
            feedback_df,
            new_record_df
        ],
        ignore_index=True
    )

    feedback_df.to_csv(
        FEEDBACK_PATH,
        index=False
    )


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Molecules",
        len(df)
    )

with col2:
    st.metric(
        "Recommended Pairs",
        len(recommendations)
    )

with col3:
    st.metric(
        "Feedback Records",
        len(feedback_df)
    )


st.divider()


# ============================================================
# IDENTIFY REVIEWED PAIRS
# ============================================================

def get_pair_key(row):

    return tuple(sorted((
        int(row["molecule_1"]),
        int(row["molecule_2"])
    )))


reviewed_pairs = set()

for _, row in feedback_df.iterrows():
    reviewed_pairs.add(
        get_pair_key(row)
    )


# ============================================================
# FILTER UNREVIEWED RECOMMENDATIONS
# ============================================================

if not recommendations.empty:

    recommendations["pair_key"] = (
        recommendations.apply(
            get_pair_key,
            axis=1
        )
    )

    pending = recommendations[
        ~recommendations["pair_key"].isin(
            reviewed_pairs
        )
    ].copy()

else:
    pending = pd.DataFrame()


# ============================================================
# MOLECULE PAIR REVIEW
# ============================================================

st.header("🔬 Chemical Structure Comparison")

if pending.empty:

    st.success(
        "All currently recommended molecule pairs have been reviewed!"
    )

    st.info(
        "Run the learning pipeline to generate new recommendations."
    )

else:

    strategy = st.selectbox(
        "Select Recommendation Strategy",
        [
            "All Strategies",
            "High Similarity",
            "Low Similarity",
            "Cluster Boundary"
        ]
    )

    if strategy != "All Strategies":
        filtered = pending[
            pending["selection_strategy"] == strategy
        ]
    else:
        filtered = pending

    if filtered.empty:

        st.warning(
            "No unreviewed pairs available in this category."
        )

    else:

        st.write(
            f"**{len(filtered)} unreviewed pairs available.**"
        )

        pair_options = filtered.to_dict(
            "records"
        )

        selected_index = st.selectbox(
            "Choose a molecule pair",
            range(len(pair_options)),
            format_func=lambda i: (
                f"Molecule {pair_options[i]['molecule_1']} "
                f"↔ Molecule {pair_options[i]['molecule_2']} "
                f"({pair_options[i]['selection_strategy']})"
            )
        )

        selected_pair = pair_options[selected_index]

        molecule_1 = int(
            selected_pair["molecule_1"]
        )

        molecule_2 = int(
            selected_pair["molecule_2"]
        )

        cluster_1 = int(
            selected_pair["cluster_1"]
        )

        cluster_2 = int(
            selected_pair["cluster_2"]
        )

        # ====================================================
        # DISPLAY CHEMICAL STRUCTURES
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Molecule 1")

            st.write(
                f"**Molecule ID:** {molecule_1}"
            )

            st.write(
                f"**Predicted Cluster:** {cluster_1}"
            )

            display_molecule(molecule_1)

        with col2:

            st.subheader("Molecule 2")

            st.write(
                f"**Molecule ID:** {molecule_2}"
            )

            st.write(
                f"**Predicted Cluster:** {cluster_2}"
            )

            display_molecule(molecule_2)

        st.divider()

        # ====================================================
        # STRUCTURAL SIMILARITY
        # ====================================================

        st.subheader("📊 Structural Similarity")

        similarity = float(
            selected_pair["tanimoto_similarity"]
        )

        st.metric(
            "Tanimoto Similarity",
            f"{similarity:.4f}"
        )

        st.progress(
            max(0.0, min(1.0, similarity))
        )

        st.info(
            f"Recommendation Strategy: "
            f"{selected_pair['selection_strategy']}"
        )

        st.caption("""
        Tanimoto similarity measures the similarity between
        molecular fingerprints.

        A high structural similarity does not necessarily mean
        that two molecules have identical biological activity.
        """)

        # ====================================================
        # HUMAN FEEDBACK
        # ====================================================

        st.divider()

        st.subheader("🧑‍🔬 Human Feedback")

        st.markdown("""
        ### Evaluate Structural Similarity

        Examine the two chemical structures carefully.

        **Question:**

        Based ONLY on their chemical structures, would you
        consider these two molecules structurally similar?

        Please consider:
        - Molecular scaffolds and ring systems.
        - Functional groups.
        - Heteroatoms and their positions.
        - Substituents and their arrangement.

        Do NOT consider biological activity or pharmacological
        properties when providing your judgment.
        """)

        st.info("""
        **Important distinction:**

        Structural similarity refers to similarities in chemical
        structure.

        Biological activity similarity refers to similarities in
        biological effects.

        These are different properties. Our current feedback
        experiment focuses exclusively on structural similarity.
        """)

        feedback = st.radio(
            "Select your judgment:",
            [
                "Similar",
                "Dissimilar"
            ],
            horizontal=True
        )

        if st.button(
            "Submit Feedback",
            type="primary",
            use_container_width=True
        ):

            save_feedback(
                molecule_1,
                molecule_2,
                feedback,
                cluster_1,
                cluster_2
            )

            st.success(
                f"Feedback saved: Molecule {molecule_1} "
                f"and Molecule {molecule_2} → {feedback}"
            )

            st.rerun()


# ============================================================
# HUMAN FEEDBACK HISTORY
# ============================================================

st.divider()

st.header("📋 Human Feedback History")

feedback_df = load_feedback()

if not feedback_df.empty:

    # Hide old reference-label columns from the dashboard.
    # Historical records are preserved in the CSV.

    display_columns = [
        "molecule_1",
        "molecule_2",
        "relationship",
        "cluster_1",
        "cluster_2",
        "timestamp"
    ]

    available_columns = [
        col for col in display_columns
        if col in feedback_df.columns
    ]

    st.dataframe(
        feedback_df[available_columns],
        use_container_width=True
    )

    st.download_button(
        "Download Feedback CSV",
        data=feedback_df.to_csv(index=False),
        file_name="human_feedback.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No feedback submitted yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PegaPlus | Human-in-the-Loop Molecular Clustering "
    "| Machine Learning in Chemistry Project"
)