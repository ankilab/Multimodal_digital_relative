# HANCOCK Multimodal Patient Analysis System

Patient similarity analysis using multimodal features: clinical, pathological, blood, ICD codes, and TMA data.

## Quick Start

```bash
# Activate environment
conda activate hancock_multimodal

# Run dashboard
python run_app.py  # http://localhost:8044

# Run analysis notebook
jupyter notebook patient_analysis_notebook.ipynb
```

## Files & Notebooks

**Main Notebooks:**
- `notebooks/tma_feature_comparison.ipynb` - TMA feature impact on new patients
- `notebooks/similarity_analysis.ipynb` - Training cohort similarity patterns

**App:**
- `run_app.py` - Dashboard entry point
- `app/` -  Dash application

## Project Structure

```
multimodal2/
├── app/                              # Dashboard application
│   ├── main.py, layout.py, callbacks.py, patient_encoding.py, utils.py
├── notebooks/                        # Analysis notebooks
│   ├── tma_feature_comparison.ipynb
│   └── similarity_analysis..ipynb
├── data/                             # Raw patient data
├── features/                         # Feature matrices & embeddings
├── models/                           # Pre-trained models
├── feature_extraction/               # Feature processing
├── data_exploration/                 # UMAP embedding
├── requirements.txt
└── README.md
```

## Core Modules

- **preprocessor.pkl**: Feature standardization & encoding
- **umap_model.pkl**: 2D patient embedding
- **get_umap_embedding()**: Load full training dataset
- **load_and_encode_patient()**: Process new patient data
- **cosine_similarity**: Find similar patients

## Usage

**Dashboard:**
1. `python run_app.py`
2. Select patient, adjust visualization
3. Click on UMAP points to find similar patients

**Analysis Notebooks:**
- Extract training data and patient vectors
- Compute cosine similarity scores
- Identify top 5 most similar patients
- Compare WITH/WITHOUT TMA features
- Visualize similarity patterns

## TMA Features

System analyzes 4 tissue microarray measurements:
- `cd3_z`, `cd3_inv`: CD3+ T cell density
- `cd8_z`, `cd8_inv`: CD8+ T cell density

Notebooks compare rankings and scores with/without these features.


## Acknowledgments

- Built with Dash, Plotly, and scikit-learn
- UMAP by McInnes, Healy & Melville (2018)
- Feature engineering pipeline adapted from [relevant sources]

