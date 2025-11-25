
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from umap import UMAP
from data_exploration.umap_embedding import setup_preprocessing_pipeline, SEED
from feature_extraction.extract_tabular_features import (
    NOMINAL_FEATURES, ORDINAL_FEATURES, DISCRETE_FEATURES, BLOOD_FEATURES
)
from sklearn.preprocessing import LabelEncoder




def fit_label_encoders_from_csv(features_directory):
    """
    Reconstruct LabelEncoders from the CSV files.
    Since we don't have the original string values, we create placeholder mappings.
    
    Parameters
    ----------
    features_directory : str
        Directory containing feature CSV files
        
    Returns
    -------
    dict
        Dictionary mapping feature names to fitted LabelEncoder objects
    """
    fdir = Path(features_directory)
    
    # Load CSV files
    clinical = pd.read_csv(fdir/"clinical.csv", dtype={"patient_id": str})
    pathological = pd.read_csv(fdir/"pathological.csv", dtype={"patient_id": str})
    
    # Merge
    df = clinical.merge(pathological, on="patient_id", how="inner")
    
    label_encoders = {}
    
    # For each categorical feature, create a LabelEncoder with placeholder classes
    categorical_features = NOMINAL_FEATURES + ORDINAL_FEATURES
    
    for feature in categorical_features:
        if feature in df.columns:
            # Get unique integer values (excluding NaN)
            unique_values = df[feature].dropna().unique()
            unique_values = sorted([int(v) for v in unique_values if not pd.isna(v)])
            
            # Create placeholder string classes
            # The actual string values don't matter for transformation,
            # as long as they map to the correct integers
            placeholder_classes = [f"{feature}_class_{i}" for i in unique_values]
            
            # Create and fit LabelEncoder
            le = LabelEncoder()
            le.fit(placeholder_classes)
            
            # Verify the mapping is correct (should be 0, 1, 2, ...)
            # But the CSV might have gaps or different numbering
            # So we need to store the actual mapping
            le.classes_ = np.array(placeholder_classes)
            
            label_encoders[feature] = le
            print(f"Created LabelEncoder for {feature}: {len(unique_values)} classes (values: {unique_values})")
    
    return label_encoders

def save_fitted_models(features_directory, output_directory, 
                       umap_min_dist=0.1, umap_n_neighbors=15):
    """
    Loads multimodal features, fits the preprocessor and UMAP model, and saves them.
    Also reconstructs and saves LabelEncoders from CSV data.
    """
    fdir = Path(features_directory)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reconstructing LabelEncoders from CSV data...")
    label_encoders = fit_label_encoders_from_csv(features_directory)
    
    # Save label encoders
    joblib.dump(label_encoders, out_dir / "label_encoders.pkl")
    print(f"Saved {len(label_encoders)} label encoders to label_encoders.pkl\n")

    print("Loading preprocessed data...")
    # Load encoded data
    clinical = pd.read_csv(fdir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(fdir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(fdir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(fdir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density= pd.read_csv(fdir/"tma_cell_density.csv", dtype={"patient_id": str})

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    print(f"Data loaded. Shape: {df.shape}")

    # Prepare features
    # Drop patient_id to get feature matrix
    X = df.drop("patient_id", axis=1)
    
    # Save feature order
    feature_order = list(X.columns)
    with open(out_dir / "feature_order.json", "w") as f:
        json.dump(feature_order, f)
    print("Saved feature_order.json")

    # Setup and fit preprocessor
    print("Fitting preprocessor...")
    preprocessor = setup_preprocessing_pipeline(X.columns)
    preprocessor.fit(X)
    
    # Save preprocessor
    joblib.dump(preprocessor, out_dir / "preprocessor.pkl")
    print("Saved preprocessor.pkl")
    
    # Transform data for UMAP
    embeddings = preprocessor.transform(X)

    # Fit UMAP model
    print("Fitting UMAP model...")
    umap_model = UMAP(random_state=SEED, min_dist=umap_min_dist, n_neighbors=umap_n_neighbors)
    umap_model.fit(embeddings)
    
    # Get embeddings for normalization parameters
    umap_embeddings = umap_model.transform(embeddings)
    
    # Calculate normalization parameters from training data
    umap_min_x = np.min(umap_embeddings[:, 0])
    umap_max_x = np.max(umap_embeddings[:, 0])
    umap_min_y = np.min(umap_embeddings[:, 1])
    umap_max_y = np.max(umap_embeddings[:, 1])
    
    normalization_params = {
        "min_x": float(umap_min_x),
        "max_x": float(umap_max_x),
        "min_y": float(umap_min_y),
        "max_y": float(umap_max_y)
    }
    
    # Save normalization parameters
    with open(out_dir / "umap_normalization.json", "w") as f:
        json.dump(normalization_params, f)
    print(f"Saved normalization parameters: {normalization_params}")
    
    # Save UMAP model
    joblib.dump(umap_model, out_dir / "umap_model.pkl")
    print("Saved umap_model.pkl")
    
    print("Done.")

if __name__ == "__main__":
    # Assuming running from project root
    FEATURES_DIR = "features"
    RESULTS_DIR = "results"
    
    save_fitted_models(FEATURES_DIR, RESULTS_DIR)

