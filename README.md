# PegaPlus: Human-in-the-Loop Molecular Clustering

PegaPlus is an experimental machine-learning project that explores how human judgments of molecular structural similarity can influence molecular clustering.

The project uses molecular fingerprints to represent molecules, applies baseline K-Means++ clustering, collects human pairwise similarity feedback, converts that feedback into must-link and cannot-link constraints, and runs a constraint-aware clustering workflow. It also analyzes assignment changes, recommends additional molecule pairs for review, and provides an interactive Streamlit dashboard.

> **Project scope:** This is an educational experimental implementation inspired by human-in-the-loop and constraint-based clustering ideas. It should not be interpreted as a validated production cheminformatics system or as a reproduction of every detail of a published PegaPlus method.

## Current experiment

The current run uses **158 molecules** and **15 human feedback records**.

| Experiment result | Value |
|---|---:|
| Molecules | 158 |
| Human feedback records | 15 |
| Must-link constraints | 9 |
| Cannot-link constraints | 6 |
| Must-link constraints satisfied | 9 / 9 |
| Cannot-link constraints satisfied | 6 / 6 |
| Molecules whose assignments changed | 9 |
| Percentage of molecules changed | 5.70% |

### Baseline and constrained clustering

| Metric | Baseline K-Means++ | Constrained clustering |
|---|---:|---:|
| Adjusted Rand Index (ARI) | 0.073954 | 0.065682 |
| Normalized Mutual Information (NMI) | 0.344067 | 0.328182 |
| Silhouette score (Jaccard distance) | 0.123142 | 0.097275 |

The constrained run satisfied all 15 generated constraints. In this experiment, ARI, NMI, and silhouette were slightly lower than the baseline values. Constraint satisfaction and agreement with reference labels measure different properties; satisfying human constraints does not guarantee improvement in every clustering metric.

### Human-feedback structural validation

The separate validation script compared the human labels with Tanimoto similarity calculated from Morgan fingerprints.

| Validation result | Value |
|---|---:|
| Similar judgments | 9 |
| Dissimilar judgments | 6 |
| Mean Tanimoto — Similar group | 0.4491 |
| Mean Tanimoto — Dissimilar group | 0.5502 |
| Mann–Whitney U statistic | 22.0 |
| Mann–Whitney p-value | 0.7418 |
| ROC-AUC | 0.4074 |
| Spearman correlation | -0.1576 |
| Spearman p-value | 0.5748 |

The current sample does not show a statistically significant positive relationship between the human judgments and Morgan-fingerprint Tanimoto similarity. The sample is small, so these findings are preliminary. Tanimoto similarity is a fingerprint-based structural-overlap measure; it is not a measure of biological activity and does not capture every aspect of chemical similarity.

## Workflow

1. **Molecular representation:** Load the molecular dataset and generate 1024-bit Morgan fingerprints using RDKit.
2. **Baseline clustering:** Run K-Means++ to obtain initial cluster assignments.
3. **Human feedback:** Collect pairwise judgments (`Similar` or `Dissimilar`) in a Streamlit application.
4. **Constraint generation:** Convert `Similar` judgments into must-link constraints and `Dissimilar` judgments into cannot-link constraints.
5. **Constrained clustering:** Run the project's COP-KMeans-style constraint-aware clustering implementation.
6. **Evaluation and comparison:** Compare baseline and constrained assignments using ARI, NMI, and Jaccard-based silhouette score.
7. **Feedback impact:** Identify molecules whose cluster assignments changed.
8. **Pair recommendation:** Recommend additional pairs using fingerprint similarity and cluster-boundary information.
9. **Feedback validation:** Compare human judgments with fingerprint Tanimoto similarity and calculate statistical summaries.
10. **Dashboard:** Explore the experiment through a Streamlit dashboard.

## Repository structure

```text
PegaPlus/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reference/
│   └── online_svm_evaluation/
│       └── test_files/
├── results/
├── src/
│   ├── feedback_app.py
│   ├── constraint_engine.py
│   ├── constrained_clustering.py
│   ├── compare_clustering.py
│   ├── analyze_feedback_impact.py
│   ├── intelligent_pair_selection.py
│   ├── run_learning_pipeline.py
│   ├── validate_human_feedback.py
│   ├── visualization.py
│   └── interactive_visualization.py
├── main.py
├── results_dashboard.py
├── README.md
└── .gitignore
```

The exact contents of `data/`, `notebooks/`, and `results/` may change as the experiment evolves.

## Requirements

- Python 3.11 (the project was developed with Python 3.11.9)
- RDKit
- pandas
- NumPy
- scikit-learn
- SciPy
- Streamlit
- Plotly
- Matplotlib
- Seaborn

Install the dependencies in your virtual environment. If the repository contains a `requirements.txt`, use it:

```bash
pip install -r requirements.txt
```

If there is no dependency file yet, install the packages used by the project in your environment. RDKit installation can depend on the operating system and Python environment; consult the official RDKit installation instructions if needed.

## Setup

Clone the repository:

```bash
git clone https://github.com/HimaLonare/PegaPlus.git
cd PegaPlus
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install project dependencies as described above.

## Running the project

Run the molecular fingerprint generation and baseline workflow:

```bash
python main.py
```

Launch the human-feedback collection application:

```bash
streamlit run src/feedback_app.py
```

Run the complete five-step learning pipeline after collecting or changing feedback:

```bash
python src/run_learning_pipeline.py
```

Run the separate human-feedback validation:

```bash
python src/validate_human_feedback.py
```

Launch the experimental results dashboard:

```bash
streamlit run results_dashboard.py
```

Run the pipeline again whenever the feedback records change so that the generated constraints, constrained assignments, comparison metrics, impact analysis, and recommendations are based on the latest feedback. Run the validation script separately to refresh the structural-similarity validation outputs.

## Generated outputs

Depending on the scripts run, the project writes files such as:

- `data/processed/cox2_fingerprints.csv` — molecular fingerprint features.
- `data/processed/cox2_clustered.csv` — baseline cluster assignments.
- `data/processed/human_feedback.csv` — pairwise human feedback.
- `data/processed/must_link_constraints.csv` — generated must-link constraints.
- `data/processed/cannot_link_constraints.csv` — generated cannot-link constraints.
- `data/processed/constrained_results.csv` — constrained clustering assignments.
- `results/clustering_comparison.csv` — baseline and constrained metric comparison.
- `results/feedback_impact_analysis.csv` — summary of assignment changes.
- `results/changed_molecules.csv` — molecules whose assignments changed.
- `results/recommended_pairs.csv` — candidate pairs recommended for further review.
- `results/human_feedback_validation.csv` — pair-level validation results.
- `results/human_feedback_validation_summary.csv` — validation statistics.
- `results/human_feedback_validation.png` — validation visualization.

## Interpretation and limitations

- **Small feedback sample:** The reported experiment uses 15 pairwise judgments. Results should be treated as preliminary and may change as more feedback is collected.
- **Fingerprint representation:** Morgan fingerprints and Tanimoto similarity represent one computational view of molecular structural overlap. They do not capture all chemical context or establish biological activity.
- **Constraint satisfaction vs. clustering quality:** Satisfying must-link and cannot-link constraints does not necessarily increase ARI, NMI, or silhouette score.
- **Experimental implementation:** The constrained clustering code is a project-level COP-KMeans-style implementation. Its behavior and limitations should be assessed from the source code; it should not be described as a formal or exact reproduction of a particular published algorithm without further verification.
- **Dataset and reference labels:** Evaluation against reference labels depends on the labels and dataset supplied with the project. It does not by itself establish chemical or biological validity.

## Reproducibility

For a comparable run:

1. Use the same molecular dataset and preprocessing.
2. Keep a copy of the human feedback CSV used for the experiment.
3. Run `python src/run_learning_pipeline.py`.
4. Run `python src/validate_human_feedback.py`.
5. Launch `streamlit run results_dashboard.py`.

Record the feedback file and code version alongside any results you report.

## Acknowledgements

Developed as a Machine Learning in Chemistry course project at IIT Hyderabad.

## Author

**Himanshu Lonare**  
GitHub: [HimaLonare](https://github.com/HimaLonare)
