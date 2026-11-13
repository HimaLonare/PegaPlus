import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PegaPlus - Human Feedback",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 PegaPlus: Human-in-the-Loop Learning")

st.markdown("""
### Molecular Similarity Feedback System

Select two molecules and tell the system whether they
should be considered chemically similar or dissimilar.

Your feedback will be saved and used in the next stage
to improve the clustering algorithm.
""")

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

DATA_PATH = "data/processed/cox2_clustered.csv"
FEEDBACK_PATH = "data/processed/human_feedback.csv"

if not os.path.exists(DATA_PATH):
    st.error("Clustering dataset not found. Run main.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Create molecule IDs
df["molecule_id"] = range(1, len(df) + 1)

st.success(f"Dataset loaded: {len(df)} molecules")

# ==========================================
# STEP 2: DISPLAY DATASET INFORMATION
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Molecules", len(df))

with col2:
    st.metric(
        "Number of Clusters",
        df["predicted_cluster"].nunique()
    )

with col3:
    st.metric(
        "Reference Labels",
        df["reference_label"].nunique()
    )

st.divider()

# ==========================================
# STEP 3: SELECT TWO MOLECULES
# ==========================================

st.subheader("1. Select Molecules")

molecule_ids = df["molecule_id"].tolist()

col1, col2 = st.columns(2)

with col1:
    molecule_1 = st.selectbox(
        "Select First Molecule",
        molecule_ids,
        index=0
    )

with col2:
    molecule_2 = st.selectbox(
        "Select Second Molecule",
        molecule_ids,
        index=1
    )

# Retrieve selected molecule records
mol1 = df.loc[df["molecule_id"] == molecule_1].iloc[0]
mol2 = df.loc[df["molecule_id"] == molecule_2].iloc[0]

# ==========================================
# STEP 4: DISPLAY MOLECULE INFORMATION
# ==========================================

st.subheader("2. Compare Selected Molecules")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### Molecule {molecule_1}")

    st.write("Predicted Cluster:", int(mol1["predicted_cluster"]))
    st.write("Reference Label:", int(mol1["reference_label"]))

with col2:
    st.markdown(f"### Molecule {molecule_2}")

    st.write("Predicted Cluster:", int(mol2["predicted_cluster"]))
    st.write("Reference Label:", int(mol2["reference_label"]))

if molecule_1 == molecule_2:
    st.warning("Please select two different molecules.")

st.divider()

# ==========================================
# STEP 5: COLLECT HUMAN FEEDBACK
# ==========================================

st.subheader("3. Provide Your Feedback")

st.info("""
Consider the molecular similarity based on your chemical
knowledge or structural information.
""")

relationship = st.radio(
    "What is the relationship between these molecules?",
    ["Similar", "Dissimilar"],
    horizontal=True
)

# ==========================================
# STEP 6: SAVE FEEDBACK
# ==========================================

if st.button("Submit Feedback", type="primary"):

    if molecule_1 == molecule_2:

        st.error("Please select two different molecules.")

    else:

        feedback_record = {
            "molecule_1": molecule_1,
            "molecule_2": molecule_2,
            "relationship": relationship,
            "cluster_1": int(mol1["predicted_cluster"]),
            "cluster_2": int(mol2["predicted_cluster"]),
            "reference_label_1": int(mol1["reference_label"]),
            "reference_label_2": int(mol2["reference_label"]),
            "timestamp": datetime.now().isoformat()
        }

        feedback_df = pd.DataFrame([feedback_record])

        if os.path.exists(FEEDBACK_PATH):

            feedback_df.to_csv(
                FEEDBACK_PATH,
                mode="a",
                header=False,
                index=False
            )

        else:

            feedback_df.to_csv(
                FEEDBACK_PATH,
                index=False
            )

        st.success("Human feedback saved successfully!")

# ==========================================
# STEP 7: DISPLAY PREVIOUS FEEDBACK
# ==========================================

st.divider()

st.subheader("4. Feedback History")

if os.path.exists(FEEDBACK_PATH):

    history_df = pd.read_csv(FEEDBACK_PATH)

    st.metric("Total Feedback Records", len(history_df))

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info("No feedback submitted yet.")

st.divider()

st.caption("PegaPlus | Human-in-the-Loop Molecular Clustering")