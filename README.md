# HANCOCK Multimodal Patient Analysis System

Patient similarity analysis using multimodal features: clinical, pathological, blood, ICD codes, and TMA data.

builds up on hancock repo 

## Quick Start

```bash
# Create environment
conda create --name hancock_multimodal 
conda activate hancock_multimodal
pip install -r requirements.txt

# Run dashboard
python run_app.py  # http://localhost:8044

# Run analysis notebook
jupyter notebook notebooks/tma_feature_comparison.ipynb
jupyter notebook notebooks/similarity_analysis.ipynb
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


## Usage

**Dashboard:**
1. `python run_app.py`
2. Select patient, adjust visualization
3. Click on UMAP points to find similar patients

<video controls src="app/tutorial/Screen Recording 2026-04-16 at 10.31.09.mov" title="Title"></video>

**Analysis Notebooks:**
- Extract training data and patient vectors
- Compute cosine similarity scores
- Identify top 5 most similar patients
- Compare WITH/WITHOUT TMA features
- Visualize similarity patterns

## Data

Data Sources:
- Data is retrieved from Medona
- Each feature definition (including allowed values) can be found in the corresponding documentation

Data Extraction:
- Extract patient-level data from Medona
- Store extracted data in a temporary Excel template

Data Formatting:
- The Excel template is transformed into a structured format for downstream analysis
- Alternatively, data can be directly exported into structured formats (json)

Required Data Modalities:
- Structured data
    - Blood data
    - Clinical data
    - Pathological data
- Text-based data
    - ICD codes
- TMA (Tissue Microarray) measurements

## Acknowledgments

- Built with Dash, Plotly, and scikit-learn
- UMAP by McInnes, Healy & Melville (2018)
- Feature engineering pipeline adapted from HANCOCK

