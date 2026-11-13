import streamlit as st
import pandas as pd
import os


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

Review intelligently selected molecule pairs and provide
feedback about their structural similarity.
""")


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/processed/cox2_clustered.csv"

RECOMMENDATION_PATH = "results/recommended_pairs.csv"

FEEDBACK_PATH = "data/processed/human_feedback.csv"


# ============================================================
# LOAD MOLECULE AND RECOMMENDATION DATA
# ============================================================

@st.cache_data
def load_data():

    molecules = pd.read_csv(DATA_PATH)

    if os.path.exists(RECOMMENDATION_PATH):
        recommendations = pd.read_csv(RECOMMENDATION_PATH)
    else:
        recommendations = pd.DataFrame()

    return molecules, recommendations


df, recommendations = load_data()


# ============================================================
# LOAD AND NORMALIZE HUMAN FEEDBACK
# ============================================================

def load_feedback():

    if os.path.exists(FEEDBACK_PATH):

        feedback = pd.read_csv(FEEDBACK_PATH)

        # Support both old and new feedback formats
        if "relationship" not in feedback.columns:
            feedback["relationship"] = None

        if "feedback" not in feedback.columns:
            feedback["feedback"] = None

        # Recover relationship from feedback where necessary
        feedback["relationship"] = (
            feedback["relationship"].fillna(feedback["feedback"])
        )

        # Recover feedback from relationship where necessary
        feedback["feedback"] = (
            feedback["feedback"].fillna(feedback["relationship"])
        )

        return feedback

    return pd.DataFrame(columns=[
        "molecule_1",
        "molecule_2",
        "relationship",
        "cluster_1",
        "cluster_2",
        "reference_label_1",
        "reference_label_2",
        "timestamp",
        "feedback"
    ])


feedback_df = load_feedback()


# ============================================================
# SAVE HUMAN FEEDBACK
# ============================================================

def save_feedback(molecule_1, molecule_2, feedback):

    global feedback_df

    # Get information about both molecules
    mol1 = df.iloc[molecule_1 - 1]
    mol2 = df.iloc[molecule_2 - 1]

    new_record = {

        "molecule_1": molecule_1,

        "molecule_2": molecule_2,

        # Used by constraint_engine.py
        "relationship": feedback,

        "cluster_1": int(mol1["predicted_cluster"]),

        "cluster_2": int(mol2["predicted_cluster"]),

        "reference_label_1": int(mol1["reference_label"]),

        "reference_label_2": int(mol2["reference_label"]),

        "timestamp": pd.Timestamp.now().isoformat(),

        # Kept for compatibility with the new feedback interface
        "feedback": feedback
    }

    new_record_df = pd.DataFrame([new_record])

    feedback_df = pd.concat(
        [feedback_df, new_record_df],
        ignore_index=True
    )

    # Save all existing and new records
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
# IDENTIFY ALREADY REVIEWED PAIRS
# ============================================================

def get_pair_key(row):

    return tuple(sorted((
        int(row["molecule_1"]),
        int(row["molecule_2"])
    )))


reviewed_pairs = set()

for _, row in feedback_df.iterrows():

    reviewed_pairs.add(get_pair_key(row))


# ============================================================
# FILTER UNREVIEWED RECOMMENDATIONS
# ============================================================

if not recommendations.empty:

    recommendations["pair_key"] = recommendations.apply(
        get_pair_key,
        axis=1
    )

    pending = recommendations[
        ~recommendations["pair_key"].isin(reviewed_pairs)
    ].copy()

else:

    pending = pd.DataFrame()


# ============================================================
# INTELLIGENT MOLECULE PAIR REVIEW
# ============================================================

st.header("🔍 Review Recommended Molecule Pairs")


if pending.empty:

    st.success(
        "All currently recommended molecule pairs have been reviewed!"
    )

    st.info(
        "Run intelligent_pair_selection.py again to generate "
        "new recommendations."
    )

else:

    # Strategy selection
    strategy = st.selectbox(
        "Select Recommendation Strategy",
        [
            "All Strategies",
            "High Similarity",
            "Low Similarity",
            "Cluster Boundary"
        ]
    )

    # Filter recommendations
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

        pair_options = filtered.to_dict("records")

        # Select molecule pair
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

        molecule_1 = int(selected_pair["molecule_1"])

        molecule_2 = int(selected_pair["molecule_2"])


        st.divider()

        st.subheader("Molecule Pair Details")


        # Display molecule information
        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Molecule 1")

            st.metric(
                "Molecule ID",
                molecule_1
            )

            st.metric(
                "Predicted Cluster",
                int(selected_pair["cluster_1"])
            )

        with col2:

            st.markdown("### Molecule 2")

            st.metric(
                "Molecule ID",
                molecule_2
            )

            st.metric(
                "Predicted Cluster",
                int(selected_pair["cluster_2"])
            )


        st.divider()

        st.subheader("Structural Similarity")

        st.metric(
            "Tanimoto Similarity",
            f"{selected_pair['tanimoto_similarity']:.4f}"
        )

        st.info(
            f"Recommendation Strategy: "
            f"{selected_pair['selection_strategy']}"
        )

        st.caption(
            "Tanimoto similarity measures fingerprint-based "
            "structural similarity. It does not directly establish "
            "similar biological activity."
        )


        # ====================================================
        # HUMAN FEEDBACK
        # ====================================================

        st.divider()

        st.subheader("Human Feedback")

        st.write(
            "Based on your chemical assessment, "
            "how would you classify these two molecules?"
        )

        feedback = st.radio(
            "Select your judgment:",
            ["Similar", "Dissimilar"],
            horizontal=True
        )


        if st.button(
            "Submit Feedback",
            type="primary"
        ):

            save_feedback(
                molecule_1,
                molecule_2,
                feedback
            )

            st.success(
                f"Feedback saved: Molecule {molecule_1} and "
                f"Molecule {molecule_2} → {feedback}"
            )

            st.rerun()


# ============================================================
# HUMAN FEEDBACK HISTORY
# ============================================================

st.divider()

st.header("📊 Human Feedback History")

feedback_df = load_feedback()


if not feedback_df.empty:

    st.dataframe(
        feedback_df,
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