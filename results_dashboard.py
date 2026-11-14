import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PegaPlus Experimental Results",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 PegaPlus: Experimental Results Dashboard")

st.subheader("Human-in-the-Loop Molecular Clustering")

st.write(
    "An experimental dashboard for analyzing how human structural "
    "similarity feedback influences molecular clustering."
)

st.divider()


# ============================================================
# 2. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

FILES = {
    "baseline": DATA_DIR / "cox2_clustered.csv",
    "constrained": DATA_DIR / "constrained_results.csv",
    "feedback": DATA_DIR / "human_feedback.csv",
    "comparison": RESULTS_DIR / "clustering_comparison.csv",
    "impact": RESULTS_DIR / "feedback_impact_analysis.csv",
    "changed": RESULTS_DIR / "changed_molecules.csv",
    "must_link": DATA_DIR / "must_link_constraints.csv",
    "cannot_link": DATA_DIR / "cannot_link_constraints.csv",
    "validation": RESULTS_DIR / "human_feedback_validation.csv",
    "validation_summary": RESULTS_DIR / "human_feedback_validation_summary.csv"
}


# ============================================================
# 3. LOAD CSV FILES
# IMPORTANT: NO STREAMLIT CACHE
# ============================================================

def load_csv(path):

    if not path.exists():
        return None

    return pd.read_csv(path)


baseline_df = load_csv(FILES["baseline"])
constrained_df = load_csv(FILES["constrained"])
feedback_df = load_csv(FILES["feedback"])
comparison_df = load_csv(FILES["comparison"])
impact_df = load_csv(FILES["impact"])
changed_df = load_csv(FILES["changed"])
must_df = load_csv(FILES["must_link"])
cannot_df = load_csv(FILES["cannot_link"])
validation_df = load_csv(FILES["validation"])
validation_summary_df = load_csv(FILES["validation_summary"])


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Baseline clustering": baseline_df,
    "Constrained clustering": constrained_df,
    "Human feedback": feedback_df,
    "Clustering comparison": comparison_df,
    "Feedback impact analysis": impact_df,
    "Must-link constraints": must_df,
    "Cannot-link constraints": cannot_df
}

missing_files = [
    name for name, data in required_files.items()
    if data is None
]

if missing_files:

    st.error(
        "Missing required files:\n\n"
        + "\n".join(f"- {name}" for name in missing_files)
    )

    st.stop()


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

def check_columns(df, columns, filename):

    missing = [
        column for column in columns
        if column not in df.columns
    ]

    if missing:

        st.error(
            f"Missing columns in {filename}: "
            f"{', '.join(missing)}"
        )

        st.stop()


check_columns(
    baseline_df,
    ["predicted_cluster"],
    "cox2_clustered.csv"
)

check_columns(
    constrained_df,
    ["constrained_cluster"],
    "constrained_results.csv"
)

check_columns(
    comparison_df,
    ["Metric", "K-Means++", "Constrained K-Means"],
    "clustering_comparison.csv"
)

check_columns(
    impact_df,
    ["Molecule_ID", "Original_Cluster", "Constrained_Cluster"],
    "feedback_impact_analysis.csv"
)

check_columns(
    must_df,
    ["index_1", "index_2"],
    "must_link_constraints.csv"
)

check_columns(
    cannot_df,
    ["index_1", "index_2"],
    "cannot_link_constraints.csv"
)


# ============================================================
# 6. PREPARE CLUSTERING DATA
# ============================================================

if len(baseline_df) != len(constrained_df):

    st.error(
        "Baseline and constrained clustering files have "
        "different numbers of molecules. Rerun the pipeline."
    )

    st.stop()


baseline_labels = baseline_df[
    "predicted_cluster"
].to_numpy()

constrained_labels = constrained_df[
    "constrained_cluster"
].to_numpy()

total_molecules = len(baseline_labels)

if total_molecules == 0:
    st.error("No molecules found in the clustering dataset.")
    st.stop()


# Calculate actual assignment changes

changed_mask = baseline_labels != constrained_labels

changed_count = int(np.sum(changed_mask))

unchanged_count = total_molecules - changed_count

changed_percentage = (
    changed_count / total_molecules
) * 100


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def get_metric(metric_name):

    rows = comparison_df[
        comparison_df["Metric"].astype(str).str.contains(
            metric_name,
            case=False,
            regex=False
        )
    ]

    if rows.empty:
        return None, None

    baseline_value = pd.to_numeric(
        rows.iloc[0]["K-Means++"],
        errors="coerce"
    )

    constrained_value = pd.to_numeric(
        rows.iloc[0]["Constrained K-Means"],
        errors="coerce"
    )

    return baseline_value, constrained_value


def show_metric(title, baseline_value, constrained_value):

    st.markdown(f"**{title}**")

    col1, col2 = st.columns(2)

    if baseline_value is None or constrained_value is None:
        col1.metric("Baseline", "N/A")
        col2.metric("Constrained", "N/A")
        return

    if pd.isna(baseline_value) or pd.isna(constrained_value):
        col1.metric("Baseline", "N/A")
        col2.metric("Constrained", "N/A")
        return

    difference = constrained_value - baseline_value

    col1.metric(
        "Baseline",
        f"{baseline_value:.4f}"
    )

    col2.metric(
        "Constrained",
        f"{constrained_value:.4f}",
        delta=f"{difference:+.4f}"
    )


# ============================================================
# 8. CONSTRAINT EVALUATION
# ============================================================

def evaluate_constraints(constraint_df, must_link=True):

    """
    Evaluate human constraints using zero-based molecule indexes.

    Must-Link:
        Two molecules must belong to the same cluster.

    Cannot-Link:
        Two molecules must belong to different clusters.
    """

    results = []

    for _, row in constraint_df.iterrows():

        i = int(row["index_1"])
        j = int(row["index_2"])

        # Check whether indexes are valid

        if not (
            0 <= i < len(constrained_labels)
            and 0 <= j < len(constrained_labels)
        ):
            continue

        cluster_1 = constrained_labels[i]
        cluster_2 = constrained_labels[j]

        same_cluster = cluster_1 == cluster_2

        # Correct logic for each constraint type

        if must_link:
            is_satisfied = same_cluster
            constraint_type = "Must-Link"

        else:
            is_satisfied = not same_cluster
            constraint_type = "Cannot-Link"

        results.append({
            "Constraint Type": constraint_type,
            "Molecule 1": row["molecule_1"],
            "Molecule 2": row["molecule_2"],
            "Index 1": i,
            "Index 2": j,
            "Cluster 1": cluster_1,
            "Cluster 2": cluster_2,
            "Same Cluster": same_cluster,
            "Satisfied": is_satisfied
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return 0, 0, 0, result_df

    total = len(result_df)

    satisfied = int(result_df["Satisfied"].sum())

    failed = total - satisfied

    return satisfied, total, failed, result_df


# Evaluate both constraint types independently

must_satisfied, must_total, must_failed, must_details = (
    evaluate_constraints(must_df, must_link=True)
)

cannot_satisfied, cannot_total, cannot_failed, cannot_details = (
    evaluate_constraints(cannot_df, must_link=False)
)


# ============================================================
# 9. DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")

feedback_count = len(feedback_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Molecules",
    total_molecules
)

col2.metric(
    "Human Feedback Records",
    feedback_count
)

col3.metric(
    "Changed Assignments",
    changed_count
)

col4.metric(
    "Molecules Changed",
    f"{changed_percentage:.2f}%"
)

st.divider()


# ============================================================
# 10. BASELINE VS CONSTRAINED CLUSTERING
# ============================================================

st.header("2. Baseline vs Constrained Clustering")

st.caption(
    "Metrics are loaded from the actual clustering comparison CSV. "
    "ARI and NMI use reference labels for external evaluation. "
    "Silhouette uses Jaccard distance on binary fingerprints."
)

ari_baseline, ari_constrained = get_metric(
    "Adjusted Rand Index"
)

nmi_baseline, nmi_constrained = get_metric(
    "Normalized Mutual Information"
)

sil_baseline, sil_constrained = get_metric(
    "Silhouette Score"
)

col1, col2, col3 = st.columns(3)

with col1:

    show_metric(
        "Adjusted Rand Index",
        ari_baseline,
        ari_constrained
    )

with col2:

    show_metric(
        "Normalized Mutual Information",
        nmi_baseline,
        nmi_constrained
    )

with col3:

    show_metric(
        "Silhouette Score",
        sil_baseline,
        sil_constrained
    )


st.subheader("Clustering Performance Comparison")

chart_df = comparison_df.copy()

chart_df["K-Means++"] = pd.to_numeric(
    chart_df["K-Means++"],
    errors="coerce"
)

chart_df["Constrained K-Means"] = pd.to_numeric(
    chart_df["Constrained K-Means"],
    errors="coerce"
)

chart_df = chart_df.melt(
    id_vars="Metric",
    value_vars=[
        "K-Means++",
        "Constrained K-Means"
    ],
    var_name="Clustering Method",
    value_name="Score"
)

fig = px.bar(
    chart_df,
    x="Metric",
    y="Score",
    color="Clustering Method",
    barmode="group",
    text_auto=".4f",
    title="Baseline vs Constrained Clustering Metrics"
)

fig.update_layout(
    xaxis_title="Evaluation Metric",
    yaxis_title="Score"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ============================================================
# 11. CLUSTER DISTRIBUTION
# ============================================================

st.header("3. Cluster Distribution")

baseline_distribution = (
    baseline_df["predicted_cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

baseline_distribution.columns = [
    "Cluster",
    "Molecule Count"
]

baseline_distribution["Method"] = "K-Means++"


constrained_distribution = (
    constrained_df["constrained_cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

constrained_distribution.columns = [
    "Cluster",
    "Molecule Count"
]

constrained_distribution["Method"] = "Constrained K-Means"


distribution_df = pd.concat(
    [
        baseline_distribution,
        constrained_distribution
    ],
    ignore_index=True
)

fig = px.bar(
    distribution_df,
    x="Cluster",
    y="Molecule Count",
    color="Method",
    barmode="group",
    title="Molecule Distribution Across Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ============================================================
# 12. CLUSTER TRANSITION MATRIX
# ============================================================

st.header("4. Cluster Transition Matrix")

st.write(
    "This matrix shows how molecules moved between clusters."
)

st.caption(
    "Rows represent original K-Means++ clusters. "
    "Columns represent constrained clustering assignments."
)

transition_matrix = pd.crosstab(
    baseline_df["predicted_cluster"],
    constrained_df["constrained_cluster"]
)

fig = px.imshow(
    transition_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    labels={
        "x": "Constrained Cluster",
        "y": "Original Cluster",
        "color": "Molecule Count"
    },
    title="Cluster Transition Matrix"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    f"Total molecules represented: "
    f"{int(transition_matrix.to_numpy().sum())}"
)

st.divider()


# ============================================================
# 13. HUMAN FEEDBACK IMPACT
# ============================================================

st.header("5. Human Feedback Impact")

col1, col2 = st.columns(2)

col1.metric(
    "Assignments Changed",
    changed_count
)

col2.metric(
    "Assignments Unchanged",
    unchanged_count
)


impact_chart_df = pd.DataFrame({
    "Assignment Status": [
        "Changed",
        "Unchanged"
    ],
    "Molecules": [
        changed_count,
        unchanged_count
    ]
})

fig = px.pie(
    impact_chart_df,
    names="Assignment Status",
    values="Molecules",
    title="Impact of Human Feedback on Molecular Clustering",
    hole=0.4
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.subheader("Molecules with Changed Assignments")

actual_changes = impact_df[
    impact_df["Original_Cluster"]
    != impact_df["Constrained_Cluster"]
].copy()

st.dataframe(
    actual_changes,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="Download Changed Molecules CSV",
    data=actual_changes.to_csv(index=False),
    file_name="changed_molecules.csv",
    mime="text/csv"
)

st.divider()


# ============================================================
# 14. HUMAN CONSTRAINT SATISFACTION
# ============================================================

st.header("6. Human Constraint Satisfaction")

st.write(
    "Constraint satisfaction measures whether the clustering "
    "algorithm follows human-provided structural similarity judgments."
)

col1, col2 = st.columns(2)


with col1:

    st.subheader("Must-Link Constraints")

    st.metric(
        "Satisfied",
        f"{must_satisfied}/{must_total}"
    )

    st.progress(
        must_satisfied / must_total if must_total else 0
    )

    st.caption(
        "Must-link molecules should belong to the same cluster."
    )

    if must_total > 0 and must_failed == 0:
        st.success("All Must-Link constraints satisfied.")

    elif must_total > 0:
        st.warning(
            f"{must_failed} Must-Link constraints failed."
        )


with col2:

    st.subheader("Cannot-Link Constraints")

    st.metric(
        "Satisfied",
        f"{cannot_satisfied}/{cannot_total}"
    )

    st.progress(
        cannot_satisfied / cannot_total if cannot_total else 0
    )

    st.caption(
        "Cannot-link molecules should belong to different clusters."
    )

    if cannot_total > 0 and cannot_failed == 0:
        st.success("All Cannot-Link constraints satisfied.")

    elif cannot_total > 0:
        st.warning(
            f"{cannot_failed} Cannot-Link constraints failed."
        )


# Show detailed pair-wise verification

with st.expander("View individual constraint verification"):

    st.subheader("Must-Link Verification")

    st.dataframe(
        must_details,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Cannot-Link Verification")

    st.dataframe(
        cannot_details,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Each pair is checked directly against the constrained_cluster "
        "column in constrained_results.csv."
    )

st.divider()


# ============================================================
# 15. HUMAN FEEDBACK SUMMARY
# ============================================================

st.header("7. Human Feedback Summary")

if "relationship" in feedback_df.columns:

    feedback_column = "relationship"

elif "feedback" in feedback_df.columns:

    feedback_column = "feedback"

else:

    feedback_column = None


if feedback_column:

    feedback_counts = (
        feedback_df[feedback_column]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
        .reset_index()
    )

    feedback_counts.columns = [
        "Feedback Type",
        "Count"
    ]

    if not feedback_counts.empty:

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                feedback_counts,
                names="Feedback Type",
                values="Count",
                title="Distribution of Human Feedback",
                hole=0.4
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                feedback_counts,
                x="Feedback Type",
                y="Count",
                text="Count",
                title="Human Feedback Counts"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    st.subheader("Feedback Records")

    # Exclude legacy reference-label metadata.

    display_feedback = feedback_df.drop(
        columns=[
            "reference_label_1",
            "reference_label_2"
        ],
        errors="ignore"
    )

    st.dataframe(
        display_feedback,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Human Feedback CSV",
        data=display_feedback.to_csv(index=False),
        file_name="human_feedback.csv",
        mime="text/csv"
    )

else:

    st.warning(
        "Could not find a relationship or feedback column."
    )

st.divider()


# ============================================================
# 16. STRUCTURAL SIMILARITY VALIDATION
# ============================================================

st.header("8. Structural Similarity Validation")

st.write(
    "This section compares human similarity judgments with "
    "Tanimoto similarity calculated from molecular fingerprints."
)

if validation_df is not None and not validation_df.empty:

    check_columns(
        validation_df,
        ["tanimoto_similarity"],
        "human_feedback_validation.csv"
    )

    if "human_feedback" in validation_df.columns:

        validation_feedback_column = "human_feedback"

    elif "feedback" in validation_df.columns:

        validation_feedback_column = "feedback"

    elif "relationship" in validation_df.columns:

        validation_feedback_column = "relationship"

    else:

        validation_feedback_column = None


    if validation_feedback_column:

        validation_display = validation_df.copy()

        validation_display["tanimoto_similarity"] = pd.to_numeric(
            validation_display["tanimoto_similarity"],
            errors="coerce"
        )

        validation_display = validation_display.dropna(
            subset=["tanimoto_similarity"]
        )

        fig = px.box(
            validation_display,
            x=validation_feedback_column,
            y="tanimoto_similarity",
            color=validation_feedback_column,
            points="all",
            title="Tanimoto Similarity Across Human Feedback Categories"
        )

        fig.update_layout(
            xaxis_title="Human Feedback",
            yaxis_title="Tanimoto Similarity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Validation Results")

        st.dataframe(
            validation_display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="Download Validation Results CSV",
            data=validation_display.to_csv(index=False),
            file_name="human_feedback_validation.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Could not identify the feedback column in validation data."
        )


    if validation_summary_df is not None:

        st.subheader("Statistical Validation Summary")

        st.dataframe(
            validation_summary_df,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "Structural similarity validation results are not available."
    )

st.divider()


# ============================================================
# 17. EXPERIMENTAL INTERPRETATION
# ============================================================

st.header("9. Experimental Interpretation")

st.markdown(
    """
    The constrained clustering algorithm incorporates human
    must-link and cannot-link judgments.

    Constraint satisfaction measures whether the algorithm
    respects those judgments.

    ARI and NMI measure agreement with the dataset's reference
    labels, while Silhouette measures internal cluster separation
    using Jaccard distance.

    These metrics measure different properties.

    A successful constraint-based clustering run does not
    necessarily imply improved reference-label agreement.
    """
)


if (
    ari_baseline is not None
    and ari_constrained is not None
    and not pd.isna(ari_baseline)
    and not pd.isna(ari_constrained)
):

    st.subheader("Observed Results")

    ari_difference = ari_constrained - ari_baseline

    if ari_difference > 0:

        st.success(
            f"ARI increased by {ari_difference:.4f} "
            "after applying human constraints."
        )

    elif ari_difference < 0:

        st.warning(
            f"ARI decreased by {abs(ari_difference):.4f} "
            "after applying human constraints."
        )

    else:

        st.info(
            "ARI remained unchanged after applying human constraints."
        )


st.info(
    "These results represent an experiment on a dataset of "
    "158 molecules with human-provided feedback. The number "
    "of feedback records is limited, so conclusions should "
    "be treated as preliminary."
)


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        <p>
            <b>PegaPlus</b> |
            Machine Learning in Chemistry |
            Human-in-the-Loop Molecular Clustering
        </p>
    </div>
    """,
    unsafe_allow_html=True
)