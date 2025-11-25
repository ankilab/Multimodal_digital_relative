
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_models(results_directory="results"):
    """
    Load the fitted preprocessor, UMAP model, and normalization parameters.
    
    Parameters
    ----------
    results_directory : str
        Directory containing the saved models
        
    Returns
    -------
    tuple
        (preprocessor, umap_model, feature_order, normalization_params)
    """
    results_dir = Path(results_directory)
    
    preprocessor = joblib.load(results_dir / "preprocessor.pkl")
    umap_model = joblib.load(results_dir / "umap_model.pkl")
    
    with open(results_dir / "feature_order.json", "r") as f:
        feature_order = json.load(f)
    
    with open(results_dir / "umap_normalization.json", "r") as f:
        normalization_params = json.load(f)
    
    return preprocessor, umap_model, feature_order, normalization_params


def preprocess_patient_data_from_dict(patient_data, feature_order, preprocessor, return_df=False):
    """
    Preprocess new patient data using the fitted preprocessor and saved label encoders.
    
    Parameters
    ----------
    patient_data : dict
        Dictionary containing patient features with raw string values
    feature_order : list
        List of feature names in the correct order
    preprocessor : sklearn ColumnTransformer
        Fitted preprocessing pipeline
    return_df : bool
        If True, return tuple (transformed, df_encoded) for debugging/verification
        
    Returns
    -------
    numpy.ndarray or tuple
        Preprocessed feature vector, or (vector, dataframe) if return_df=True
    """
    # Load saved label encoders
    try:
        label_encoders = joblib.load('results/label_encoders.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("Label encoders not found. Please run data_exploration/create_label_encoders.py first.")
    
    # Convert dict to DataFrame
    df_encoded = pd.DataFrame([patient_data])
    
    # 1. Clean categorical values (map synonyms to canonical training labels)
    # This logic handles discrepancies between new data terms and training data terms
    
    # smoking_status
    if 'smoking_status' in df_encoded.columns:
        df_encoded['smoking_status'] = df_encoded['smoking_status'].replace({
            'former': 'ex_smoker',
            'never': 'never_smoker'
        })
        
    # pT_stage
    if 'pT_stage' in df_encoded.columns:
        df_encoded['pT_stage'] = df_encoded['pT_stage'].replace({
            'pTis': 'T0is'
        })
        
    # resection_status_carcinoma_in_situ
    if 'resection_status_carcinoma_in_situ' in df_encoded.columns:
        df_encoded['resection_status_carcinoma_in_situ'] = df_encoded['resection_status_carcinoma_in_situ'].replace({
            'R0': 'CIS Absent'
        })
        
    # Binary features - map to 0/1 integers matching extract_tabular_features.py
    # Note: extract_tabular_features.py uses specific string replacements
    binary_mappings = {
        "alive": 0, "dead": 1,
        "no": 0, "yes": 1,
        "Absent": 0, "CIS": 1,
        "male": 0, "female": 1
    }
    
    binary_features = [
        'primarily_metastasis', 'perinodal_invasion', 'lymphovascular_invasion_L',
        'vascular_invasion_V', 'perineural_invasion_Pn', 'carcinoma_in_situ', 'sex'
    ]
    
    for feature in binary_features:
        if feature in df_encoded.columns:
            # Map values using the dictionary, keeping existing values if not found (e.g. if already 0/1)
            # We iterate to handle mixed types or partial matches safely
            df_encoded[feature] = df_encoded[feature].map(lambda x: binary_mappings.get(x, x))
            # Ensure numeric type
            df_encoded[feature] = pd.to_numeric(df_encoded[feature], errors='coerce')

    # 2. Apply Label Encoding for Nominal and Ordinal features
    # We iterate over the loaded encoders (which only contain Nominal and Ordinal)
    for feature, le in label_encoders.items():
        if feature in df_encoded.columns:
            try:
                # Handle unknown labels
                # If the value is not in classes, we might have an issue.
                # For pT_stage, we already cleaned pTis -> T0is.
                
                # Helper to transform safely
                def transform_safe(val):
                    if val in le.classes_:
                        return le.transform([val])[0]
                    else:
                        # Fallback: if 'unknown' or similar exists, use it.
                        # Or if there's a generic class.
                        # For now, print warning and try to map to first class or NaN?
                        # But we need an integer.
                        print(f"Warning: Unknown label '{val}' in {feature}. Known classes: {le.classes_}")
                        # Try to find a 'unknown' class
                        for cls in le.classes_:
                            if 'unknown' in str(cls).lower():
                                return le.transform([cls])[0]
                        # Extreme fallback: 0
                        return 0
                
                df_encoded[feature] = df_encoded[feature].apply(transform_safe)
                
            except Exception as e:
                print(f"Error encoding {feature}: {e}")
                # Leave as is? It will likely fail in preprocessor if it expects int.
                pass
    print("df_encoded")
    print(df_encoded)
    # Ensure all fe"atures from feature_order are present
    for feature in feature_order:
        if feature not in df_encoded.columns:
            df_encoded[feature] = np.nan
    
    # Reorder columns to match training data
    df_encoded = df_encoded[feature_order]
    print("df_encoded")
    print(df_encoded)
    
    # Transform using fitted preprocessor
    transformed = preprocessor.transform(df_encoded)
    print("transformed")
    print(transformed)
    
    if return_df:
        return transformed, df_encoded
        
    return transformed



def plot_new_patient_in_umap(new_patient_embedding, training_embeddings_file="results/umap_embeddings.csv", 
                              output_file="results/new_patient_umap.png"):
    """
    Plot the new patient's position in the UMAP space along with training data.
    
    Parameters
    ----------
    new_patient_embedding : numpy.ndarray
        2D UMAP coordinates for the new patient
    training_embeddings_file : str
        Path to CSV file containing training UMAP embeddings
    output_file : str
        Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Try to load training embeddings if available
    training_file = Path(training_embeddings_file)
    if training_file.exists():
        training_df = pd.read_csv(training_file)
        if "UMAP 1" in training_df.columns and "UMAP 2" in training_df.columns:
            ax.scatter(training_df["UMAP 1"], training_df["UMAP 2"], 
                      c='lightgray', s=10, alpha=0.5, label='Training patients')
    
    # Plot new patient
    ax.scatter(new_patient_embedding[0, 0], new_patient_embedding[0, 1], 
              c='red', s=100, marker='*', label='New patient', zorder=10)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('New Patient in UMAP Space')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()




def process_tma_data(tma_csv_path, patient_id):
    """
    Process TMA cell density measurements for a single patient.
    
    Parameters
    ----------
    tma_csv_path : Path
        Path to TMA measurements CSV file
    patient_id : str
        Patient ID to extract
        
    Returns
    -------
    dict
        Dictionary with TMA features (cd3_z, cd3_inv, cd8_z, cd8_inv)
    """
    df = pd.read_csv(tma_csv_path, dtype={"Case ID": str})
    
    # Filter for this patient
    df = df[df["Case ID"] == patient_id]
    
    if len(df) == 0:
        return {}
    
    # Extract location and marker from filename
    df["location"] = df["Image"].str.extract(r"(TumorCenter|InvasionFront)")
    df["marker"] = df["Image"].str.extract(r"(CD3|CD8)")
    
    cd3 = df[df.marker == "CD3"]
    cd8 = df[df.marker == "CD8"]
    
    # Calculate means by location
    tma_features = {}
    
    # CD3 tumor center
    cd3_z = cd3[cd3["location"] == "TumorCenter"]["Num Positive per mm^2"].mean()
    if not pd.isna(cd3_z):
        tma_features["cd3_z"] = cd3_z
    
    # CD3 invasion front
    cd3_inv = cd3[cd3["location"] == "InvasionFront"]["Num Positive per mm^2"].mean()
    if not pd.isna(cd3_inv):
        tma_features["cd3_inv"] = cd3_inv
    
    # CD8 tumor center
    cd8_z = cd8[cd8["location"] == "TumorCenter"]["Num Positive per mm^2"].mean()
    if not pd.isna(cd8_z):
        tma_features["cd8_z"] = cd8_z
    
    # CD8 invasion front
    cd8_inv = cd8[cd8["location"] == "InvasionFront"]["Num Positive per mm^2"].mean()
    if not pd.isna(cd8_inv):
        tma_features["cd8_inv"] = cd8_inv
    
    return tma_features


def process_blood_data(blood_json_path):
    """
    Process blood data from JSON file.
    Currently just extracts the values as-is since we only have one measurement.
    
    Parameters
    ----------
    blood_json_path : Path
        Path to blood data JSON file
        
    Returns
    -------
    dict
        Dictionary with blood features
    """
    with open(blood_json_path, "r") as f:
        blood_data = json.load(f)
    
    if isinstance(blood_data, list):
        # Convert list of measurements to dict
        blood_dict = {}
        for measurement in blood_data:
            loinc_name = measurement.get("LOINC_name")
            value = measurement.get("value")
            if loinc_name and value is not None:
                blood_dict[loinc_name] = value
        return blood_dict
    
    return blood_data




def load_patient_from_training_csvs(patient_id, features_dir="features"):
    """
    Load a patient's features directly from the training CSV files.
    
    Parameters
    ----------
    patient_id : str
        Patient ID to load
    features_dir : str
        Directory containing feature CSV files
        
    Returns
    -------
    dict or None
        Dictionary with all features, or None if patient not found
    """
    features_path = Path(features_dir)
    
    # Try to load from each CSV
    patient_features = {}
    found = False
    
    csv_files = {
        'clinical.csv': 'clinical',
        'pathological.csv': 'pathological',
        'blood.csv': 'blood',
        'icd_codes.csv': 'icd',
        'tma_cell_density.csv': 'tma'
    }
    
    patient_in_icd = False
    
    for csv_file, data_type in csv_files.items():
        csv_path = features_path / csv_file
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path, dtype={'patient_id': str})
        
        if patient_id in df['patient_id'].values:
            found = True
            row = df[df['patient_id'] == patient_id].iloc[0]
            
            # Add all columns except patient_id
            for col in df.columns:
                if col != 'patient_id':
                    patient_features[col] = row[col]
            
            print(f"Loaded {len(df.columns) - 1} features from {csv_file}")
            
            if csv_file == 'icd_codes.csv':
                patient_in_icd = True
    
    # If patient not in ICD codes, add ICD features with NaN
    if found and not patient_in_icd:
        # Load ICD codes CSV to get column names
        icd_path = features_path / 'icd_codes.csv'
        if icd_path.exists():
            icd_df = pd.read_csv(icd_path, dtype={'patient_id': str})
            icd_columns = [col for col in icd_df.columns if col != 'patient_id']
            
            # Add all ICD features as NaN
            for col in icd_columns:
                patient_features[col] = np.nan
            
            print(f"Added {len(icd_columns)} ICD features as NaN (patient not in icd_codes.csv)")
    
    if found:
        return patient_features
    return None


def main():
    """
    Main function to encode a new patient and project into UMAP space.
    """
    # Load the test patient data
    test_patient_dir = Path("test_patient_001/raw/structured_data")
    test_patient_raw_dir = Path("test_patient_001/raw")
    
    # Load clinical data to get patient ID
    with open(test_patient_dir / "clinical_data.json", "r") as f:
        clinical_data = json.load(f)
    
    patient_id = clinical_data.get("patient_id", "001")
    
    print(f"Processing patient: {patient_id}")
    
    # Try to load from training CSVs first (if patient exists there)
    patient_data = load_patient_from_training_csvs(patient_id)
    loaded_from_csv = patient_data is not None
    
    if loaded_from_csv:
        print(f"\n✓ Loaded patient {patient_id} from training CSVs")
        print(f"Total features from CSVs: {len(patient_data)}")
    else:
        print(f"\n✗ Patient {patient_id} not in training data, processing raw files...")
        
        # Load pathological data
        with open(test_patient_dir / "pathological_data.json", "r") as f:
            pathological_data = json.load(f)
            if isinstance(pathological_data, list):
                pathological_data = pathological_data[0]
        
        # Process blood data if available
        blood_data = {}
        blood_file = test_patient_dir / "blood_data.json"
        if blood_file.exists():
            blood_data = process_blood_data(blood_file)
            print(f"Loaded {len(blood_data)} blood features")
        
        # Process TMA data if available
        tma_data = {}
        tma_file = test_patient_raw_dir / "tma_celldensity_measurements" / "TMA_celldensity_measurements.csv"
        if tma_file.exists():
            tma_data = process_tma_data(tma_file, patient_id)
            print(f"Loaded {len(tma_data)} TMA features: {list(tma_data.keys())}")
        
        # ICD codes are empty for this patient
        icd_data = {}
        
        # Merge all data
        patient_data = {**clinical_data, **pathological_data, **blood_data, **tma_data, **icd_data}
        print(f"Total patient features: {len(patient_data)}")
    
    print("\nLoading models...")
    preprocessor, umap_model, feature_order, normalization_params = load_models()
    
    print(f"Feature order contains {len(feature_order)} features")
    print(f"Patient data contains {len(patient_data)} features")
    print(f"Normalization params: {normalization_params}")
    
    print("\nPreprocessing patient data...")
    
    # If loaded from CSVs, features are already encoded - just create DataFrame and fill missing
    if loaded_from_csv:
        # Features from CSV are already encoded, just need to ensure all features are present
        df = pd.DataFrame([patient_data])
        
        # Ensure all features from feature_order are present
        for feature in feature_order:
            if feature not in df.columns:
                df[feature] = np.nan
        
        # Reorder columns to match training data
        df = df[feature_order]
        
        # Transform using fitted preprocessor
        preprocessed = preprocessor.transform(df)
    else:
        # Raw data needs full encoding
        preprocessed = preprocess_patient_data_from_dict(patient_data, feature_order, preprocessor)
    
    print(f"Preprocessed shape: {preprocessed.shape}")
    
    print("\nProjecting into UMAP space...")
    umap_embedding = umap_model.transform(preprocessed)
    
    # Apply normalization using training data min/max
    normalized_x = (umap_embedding[0, 0] - normalization_params["min_x"]) / (normalization_params["max_x"] - normalization_params["min_x"])
    normalized_y = (umap_embedding[0, 1] - normalization_params["min_y"]) / (normalization_params["max_y"] - normalization_params["min_y"])
    
    print(f"Raw UMAP embedding: {umap_embedding}")
    print(f"Normalized UMAP coordinates: ({normalized_x:.6f}, {normalized_y:.6f})")
    
    print("\nPlotting...")
    plot_new_patient_in_umap(np.array([[normalized_x, normalized_y]]))
    
    print("\nDone!")
    return np.array([[normalized_x, normalized_y]])


if __name__ == "__main__":
    main()
