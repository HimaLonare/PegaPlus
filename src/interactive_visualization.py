import os
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA

# ==========================================
# STEP 1: LOAD CLUSTERING DATA
# ==========================================

data_path = "data/processed/cox2_clustered.csv"

df = pd.read_csv(data_path)

# Extract fingerprint columns
fingerprint_columns = [
    col for col in df.columns if col.startswith("FP_")
]

X = df[fingerprint_columns].values

# Extract labels
clusters = df["predicted_cluster"]
reference_labels = df["reference_label"]

print("Dataset loaded successfully!")
print("Total molecules:", len(df))
print("Fingerprint dimensions:", X.shape[1])

# ==========================================
# STEP 2: PCA DIMENSIONALITY REDUCTION
# ==========================================

pca = PCA(n_components=2)

X_2d = pca.fit_transform(X)

df["PC1"] = X_2d[:, 0]
df["PC2"] = X_2d[:, 1]

df["Molecule_ID"] = range(1, len(df) + 1)

explained_variance = (
    sum(pca.explained_variance_ratio_) * 100
)

print("\nPCA completed!")
print(
    "Explained variance:",
    round(explained_variance, 2),
    "%"
)

# ==========================================
# STEP 3: CREATE RESULTS DIRECTORY
# ==========================================

os.makedirs("results", exist_ok=True)

# ==========================================
# STEP 4: INTERACTIVE PREDICTED CLUSTERS
# ==========================================

fig1 = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="predicted_cluster",
    hover_data={
        "Molecule_ID": True,
        "predicted_cluster": True,
        "reference_label": True,
        "PC1": False,
        "PC2": False
    },
    color_continuous_scale="Turbo",
    title="PegaPlus - Interactive Molecular Clusters"
)

fig1.update_traces(
    marker=dict(size=11, opacity=0.8)
)

fig1.update_layout(
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2",
    dragmode="lasso",
    template="plotly_white"
)

fig1.write_html(
    "results/interactive_clusters.html"
)

# ==========================================
# STEP 5: INTERACTIVE REFERENCE LABELS
# ==========================================

fig2 = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="reference_label",
    hover_data={
        "Molecule_ID": True,
        "predicted_cluster": True,
        "reference_label": True,
        "PC1": False,
        "PC2": False
    },
    color_continuous_scale="Turbo",
    title="PegaPlus - Interactive Reference Labels"
)

fig2.update_traces(
    marker=dict(size=11, opacity=0.8)
)

fig2.update_layout(
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2",
    dragmode="lasso",
    template="plotly_white"
)

fig2.write_html(
    "results/interactive_reference_labels.html"
)

# ==========================================
# STEP 6: COMPLETION
# ==========================================

print("\n========================================")
print("INTERACTIVE VISUALIZATION COMPLETED!")
print("========================================")

print("\nGenerated files:")

print("1. results/interactive_clusters.html")
print("2. results/interactive_reference_labels.html")

print("\nOpen these HTML files in your browser.")