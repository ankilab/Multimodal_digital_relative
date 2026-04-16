"""Patient encoding and loading logic for Multimodal Patient Visualization.

This module handles the data loading and encoding pipeline for new patients,
transforming raw patient data into feature vectors and UMAP embeddings.
"""

import pandas as pd
import joblib
import json
from pathlib import Path

from feature_extraction.extract_text_features import get_icd_vectors
from feature_extraction.extract_tabular_features import (
    get_tabular_features,
    get_blood_features
)
from feature_extraction.extract_tma_features import get_tma_features

root_path = Path(__file__).parent
# Load models and metadata
preprocessor = joblib.load(root_path.parent /'models/preprocessor.pkl')
umap_model = joblib.load(root_path.parent /'models/umap_model.pkl')
with open(root_path.parent /'models/feature_order.json') as f:
    feature_order = json.load(f)


def load_and_encode_patient(base_path_str, patient_id="Unknown"):
    """Load raw data from a directory and encode it.

    Processes patient data through the feature extraction pipeline,
    applies preprocessing and UMAP transformation.

    Parameters
    ----------
    base_path_str : str
        Path to the raw patient data directory
    patient_id : str, optional
        Patient identifier. Defaults to "Unknown".

    Returns
    -------
    pd.DataFrame
        DataFrame row with encoded patient data and UMAP coordinates
    """
    base_path = Path(base_path_str)
    dest_dir = base_path.parent / "features"
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Extract ICD code features
    icd_vectors, icd_df, _ = get_icd_vectors(
        base_path / "text_data/icd_codes",
        save=False
    )
    icd_df.to_csv(dest_dir / "icd_codes.csv", index=False)

    # Extract clinical features
    clinical_vectors, clinical_df = get_tabular_features(
        base_path / "structured_data/clinical_data.json",
        save=False
    )
    clinical_df.to_csv(dest_dir / "clinical.csv", index=False)

    # Extract pathological features
    patho_vectors, patho_df = get_tabular_features(
        base_path / "structured_data/pathological_data.json",
        save=False
    )
    patho_df.to_csv(dest_dir / "pathological.csv", index=False)

    # Extract blood parameters
    blood_vectors, blood_df = get_blood_features(
        file_path_blood=base_path / "structured_data/blood_data.json",
        file_path_normal=root_path.parent / "models/blood_data_reference_ranges.json",
        file_path_clinical=base_path / "structured_data/clinical_data.json",
        save=False
    )
    blood_df.to_csv(dest_dir / "blood.csv", index=False)

    # Extract TMA cell densities
    tma_vectors, tma_df = get_tma_features(
        base_path / (
            "tma_celldensity_measurements/TMA_celldensity_measurements.csv"
        )
    )
    tma_df.to_csv(dest_dir / "tma_cell_density.csv", index=False)

    # Load and merge encoded data
    clinical = pd.read_csv(
        dest_dir / "clinical.csv",
        dtype={"patient_id": str}
    )
    patho = pd.read_csv(
        dest_dir / "pathological.csv",
        dtype={"patient_id": str}
    )
    blood = pd.read_csv(
        dest_dir / "blood.csv",
        dtype={"patient_id": str}
    )
    icd = pd.read_csv(
        dest_dir / "icd_codes.csv",
        dtype={"patient_id": str}
    )
    cell_density = pd.read_csv(
        dest_dir / "tma_cell_density.csv",
        dtype={"patient_id": str}
    )

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    # Prepare data for UMAP
    df_for_umap = df.drop(["patient_id"], axis=1)
    embeddings = preprocessor.transform(df_for_umap)
    umap_embedding = umap_model.transform(embeddings)

    # Add embedding coordinates to dataframe
    df_new = df.copy()
    df_new['patient_id'] = f"New_{patient_id}"
    df_new["Dim 1"] = umap_embedding[:, 0]
    df_new["Dim 2"] = umap_embedding[:, 1]
    df_new['method'] = 'umap'
    df_new['dataset'] = 'New Patient'

    return df_new
