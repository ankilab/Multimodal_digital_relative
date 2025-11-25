import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_exploration.umap_embedding import get_umap_embedding
from feature_extraction.encode_new_patient import preprocess_patient_data_from_dict

print("=" * 80)
print("PATIENT 001 ENCODING COMPARISON")
print("=" * 80)

# ============================================================================
# EXPECTED: Use get_umap_embedding to get patient 001 from training data
# ============================================================================
print("\n1. Getting EXPECTED encoding using get_umap_embedding()...")

# Get UMAP embedding for all training data
df_umap = get_umap_embedding("features", umap_min_dist=0.1, umap_n_neighbors=15)

# Extract patient 001
if '001' in df_umap['patient_id'].values:
    patient_001_expected = df_umap[df_umap['patient_id'] == '001'].iloc[0]
    expected_umap_x = patient_001_expected['UMAP 1']
    expected_umap_y = patient_001_expected['UMAP 2']
    
    print(f"   Found patient 001 in training data")
    print(f"   Expected UMAP: ({expected_umap_x:.6f}, {expected_umap_y:.6f})")
else:
    print("   ERROR: Patient 001 not found in training data!")
    sys.exit(1)

# ============================================================================
# NEW: Load ALL raw data for patient 001 and encode using fitted models
# ============================================================================
print("\n2. Encoding patient 001 from raw data using fitted models...")

# Load fitted models
preprocessor = joblib.load('results/preprocessor.pkl')
umap_model = joblib.load('results/umap_model.pkl')

with open('results/feature_order.json') as f:
    feature_order = json.load(f)

with open('results/umap_normalization.json') as f:
    normalization_params = json.load(f)

# Load ALL raw data for patient 001
test_patient_dir = Path("test_patient_001/raw/structured_data")

# Clinical data
with open(test_patient_dir / "clinical_data.json") as f:
    clinical_raw = json.load(f)

# Pathological data
with open(test_patient_dir / "pathological_data.json") as f:
    patho_raw = json.load(f)
    if isinstance(patho_raw, list):
        patho_raw = patho_raw[0]

# Blood data - load and fill missing values using saved modes
blood_raw = {}
blood_file = test_patient_dir / "blood_data.json"

# Load blood modes
with open('results/blood_modes.json') as f:
    blood_modes_data = json.load(f)
    modes_male = blood_modes_data['modes_male']
    modes_female = blood_modes_data['modes_female']
    blood_features_list = blood_modes_data['blood_features']

if blood_file.exists():
    try:
        # Load blood data
        blood_df = pd.read_json(blood_file, dtype={"patient_id": str})
        
        # Pivot to get LOINC_name as columns
        blood_pivot = blood_df.pivot_table(
            index="patient_id", 
            columns="LOINC_name", 
            values="value", 
            aggfunc="first"
        ).reset_index()
        
        # Determine patient sex for mode selection
        patient_sex = clinical_raw.get('sex', 'male')
        modes_to_use = modes_male if patient_sex == 'male' else modes_female
        
        # Fill all blood features - use measured values if available, otherwise use mode
        for feature in blood_features_list:
            if feature in blood_pivot.columns:
                val = blood_pivot[feature].values[0]
                if not pd.isna(val):
                    blood_raw[feature] = val
                else:
                    # Use mode for missing value
                    blood_raw[feature] = modes_to_use.get(feature, 0)
            else:
                # Feature not measured, use mode
                blood_raw[feature] = modes_to_use.get(feature, 0)
        
        measured_count = sum(1 for f in blood_features_list if f in blood_pivot.columns and not pd.isna(blood_pivot[f].values[0]))
        print(f"   Loaded {measured_count} measured blood features, filled {len(blood_features_list) - measured_count} with modes")
    except Exception as e:
        print(f"   WARNING: Error processing blood data: {e}")
        # Fill all with modes
        patient_sex = clinical_raw.get('sex', 'male')
        modes_to_use = modes_male if patient_sex == 'male' else modes_female
        for feature in blood_features_list:
            blood_raw[feature] = modes_to_use.get(feature, 0)
else:
    print("   WARNING: No blood_data.json found, using modes for all blood features")
    # Fill all with modes
    patient_sex = clinical_raw.get('sex', 'male')
    modes_to_use = modes_male if patient_sex == 'male' else modes_female
    for feature in blood_features_list:
        blood_raw[feature] = modes_to_use.get(feature, 0)

# TMA data
tma_raw = {}
tma_file = Path("test_patient_001/raw/tma_celldensity_measurements/TMA_celldensity_measurements.csv")
if tma_file.exists():
    df_tma = pd.read_csv(tma_file, dtype={"Case ID": str})
    df_tma = df_tma[df_tma["Case ID"] == "001"]
    
    if len(df_tma) > 0:
        df_tma["location"] = df_tma["Image"].str.extract(r"(TumorCenter|InvasionFront)")
        df_tma["marker"] = df_tma["Image"].str.extract(r"(CD3|CD8)")
        
        cd3 = df_tma[df_tma.marker == "CD3"]
        cd8 = df_tma[df_tma.marker == "CD8"]
        
        cd3_z = cd3[cd3["location"] == "TumorCenter"]["Num Positive per mm^2"].mean()
        if not pd.isna(cd3_z):
            tma_raw["cd3_z"] = cd3_z
        
        cd3_inv = cd3[cd3["location"] == "InvasionFront"]["Num Positive per mm^2"].mean()
        if not pd.isna(cd3_inv):
            tma_raw["cd3_inv"] = cd3_inv
        
        cd8_z = cd8[cd8["location"] == "TumorCenter"]["Num Positive per mm^2"].mean()
        if not pd.isna(cd8_z):
            tma_raw["cd8_z"] = cd8_z
        
        cd8_inv = cd8[cd8["location"] == "InvasionFront"]["Num Positive per mm^2"].mean()
        if not pd.isna(cd8_inv):
            tma_raw["cd8_inv"] = cd8_inv
        
        print(f"   Loaded {len(tma_raw)} TMA features")
else:
    print("   WARNING: No TMA data found")

# ICD codes - process from text file
icd_raw = {}
icd_file = Path("test_patient_001/raw/text_data/icd_codes")
if icd_file.exists():
    import re
    
    # Get all ICD code text files for this patient
    icd_files = list(icd_file.glob("*_001.txt"))
    
    if icd_files:
        # Get all ICD code columns from the training data
        icd_csv = pd.read_csv('features/icd_codes.csv', dtype={'patient_id': str})
        icd_columns = [col for col in icd_csv.columns if col != 'patient_id']
        
        # Initialize all ICD codes to 0
        for col in icd_columns:
            icd_raw[col] = 0
        
        # Read all ICD code files and extract codes
        for icd_txt_file in icd_files:
            with open(icd_txt_file, 'r') as f:
                content = f.read()
            
            # Extract ICD codes using regex pattern [C##.#] or [D##.#] etc.
            # Pattern matches codes like [C13.9], [C32.9], [C77.0] inside brackets
            icd_codes = re.findall(r'\[([A-Z]\d{2,3}\.?\d*)', content)
            
            # Convert to feature names (e.g., C13.9 -> c139, C77.0 -> c770)
            for code in icd_codes:
                # Remove dots and convert to lowercase (C13.9 -> c139)
                feature_name = code.replace('.', '').lower()
                
                if feature_name in icd_raw:
                    icd_raw[feature_name] += 1
        
        print(f"   Loaded {sum(1 for v in icd_raw.values() if v > 0)} ICD codes from text files")
    else:
        print("   WARNING: No ICD code text files found")
else:
    print("   WARNING: No ICD data directory found")

# Merge all raw data
raw_data = {**clinical_raw, **patho_raw, **blood_raw, **tma_raw, **icd_raw}

print(f"   Total raw features loaded: {len(raw_data)}")

# Encode using fitted preprocessor
X_new = preprocess_patient_data_from_dict(raw_data, feature_order, preprocessor)

# Transform using fitted UMAP model
umap_new = umap_model.transform(X_new)

# Apply normalization
new_umap_x = (umap_new[0, 0] - normalization_params['min_x']) / (normalization_params['max_x'] - normalization_params['min_x'])
new_umap_y = (umap_new[0, 1] - normalization_params['min_y']) / (normalization_params['max_y'] - normalization_params['min_y'])

print(f"   New UMAP: ({new_umap_x:.6f}, {new_umap_y:.6f})")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

print(f"\nUMAP Coordinates:")
print(f"  Expected (from get_umap_embedding): ({expected_umap_x:.6f}, {expected_umap_y:.6f})")
print(f"  New (from raw data):                ({new_umap_x:.6f}, {new_umap_y:.6f})")
print(f"  User's target:                      (0.528702, 0.988135)")

diff_x = abs(expected_umap_x - new_umap_x)
diff_y = abs(expected_umap_y - new_umap_y)

print(f"\nDifference:")
print(f"  X: {diff_x:.6f}")
print(f"  Y: {diff_y:.6f}")
print(f"  Match: {'✓ YES' if diff_x < 0.001 and diff_y < 0.001 else '✗ NO'}")

# Feature comparison
print(f"\n" + "=" * 80)
print("FEATURE COMPARISON")
print("=" * 80)

# Get expected features (drop UMAP columns and patient_id)
expected_features = patient_001_expected.drop(['patient_id', 'UMAP 1', 'UMAP 2'])

# Create encoded DataFrame from raw data
NOMINAL_MAPPINGS = {
    'smoking_status': {'ex_smoker': 0, 'former': 0, 'never_smoker': 1, 'never': 1, 'smoker': 2, 'unknown': 3},
    'primary_tumor_site': {'CUP': 0, 'Hypopharynx': 1, 'Larynx': 2, 'Oral_cavity': 3, 'Oropharynx': 4},
    'histologic_type': {
        'Adenocarcinoma': 0, 'Adenosquamous_carcinoma': 1, 'Carcinoma_NOS': 2,
        'SCC_Basaloid': 3, 'SCC_Keratinizing': 4, 'SCC_NOS': 5,
        'SCC_Non_keratinizing': 6, 'SCC_Papillary': 7, 'SCC_Spindle_cell': 8
    },
    'hpv_association_p16': {'negative': 0, 'not_tested': 1, 'positive': 2}
}

ORDINAL_MAPPINGS = {
    'grading': {'G1': 0, 'G2': 1, 'G3': 2, 'GX': 3},
    'resection_status': {'R0': 0, 'R1': 1, 'R2': 2, 'RX': 3},
    'resection_status_carcinoma_in_situ': {'R0': 0, 'CIS Absent': 0, 'R1': 1, 'RX': 2},
    'pT_stage': {
        'T0is': 0, 'pT0': 1, 'pT1': 2, 'pT2': 3, 'pT3': 4, 'pT4': 5, 'pT4(NOS)': 6, 'pT4a': 7, 'pT4b': 8, 'pTX': 9, 'ypT0': 10
    },
    'pN_stage': {
        'pN0': 0, 'pN0i+': 1, 'pN1': 2, 'pN1mi': 3, 'pN2': 4, 'pN2a': 5,
        'pN2b': 6, 'pN2c': 7, 'pN3': 8, 'pN3a': 9, 'pN3b': 10, 'pNX': 11
    }
}

BINARY_MAPPINGS = {
    'sex': {'male': 0, 'female': 1},
    'primarily_metastasis': {'no': 0, 'yes': 1},
    'perinodal_invasion': {'no': 0, 'yes': 1},
    'lymphovascular_invasion_L': {0: 0, 1: 1, 'no': 0, 'yes': 1},
    'vascular_invasion_V': {0: 0, 1: 1, 'no': 0, 'yes': 1},
    'perineural_invasion_Pn': {0: 0, 1: 1, 'no': 0, 'yes': 1},
    'carcinoma_in_situ': {0: 0, 1: 1, 'no': 0, 'yes': 1}
}

# Encode raw data
df_new_encoded = pd.DataFrame([raw_data])

for feature, mapping in NOMINAL_MAPPINGS.items():
    if feature in df_new_encoded.columns:
        df_new_encoded[feature] = df_new_encoded[feature].map(mapping)

for feature, mapping in ORDINAL_MAPPINGS.items():
    if feature in df_new_encoded.columns:
        if feature == 'pT_stage':
            df_new_encoded[feature] = df_new_encoded[feature].replace({'pTis': 'T0is'})
        df_new_encoded[feature] = df_new_encoded[feature].map(mapping)

for feature, mapping in BINARY_MAPPINGS.items():
    if feature in df_new_encoded.columns:
        df_new_encoded[feature] = df_new_encoded[feature].map(mapping)

# Add missing features as NaN
for feature in feature_order:
    if feature not in df_new_encoded.columns:
        df_new_encoded[feature] = np.nan

df_new_encoded = df_new_encoded[feature_order]

df_new_encoded = X_new 

# Compare features
print(f"\n{'Feature':<50} {'Expected':<15} {'New':<15} {'Match':<10}")
print("-" * 90)

differences = []
for feature in feature_order:
    val_expected = expected_features[feature] if feature in expected_features.index else np.nan
    val_new = df_new_encoded[feature].values[0]
    
    both_nan = pd.isna(val_expected) and pd.isna(val_new)
    
    if both_nan:
        match = '✓'
    elif pd.isna(val_expected) or pd.isna(val_new):
        match = '✗'
        differences.append(feature)
    elif abs(float(val_expected) - float(val_new)) < 0.0001:  # Handle floating point comparison
        match = '✓'
    else:
        match = '✗'
        differences.append(feature)
    
    # Only print differences
   # if match == '✗':
    print(f"{feature:<50} {str(val_expected):<15} {str(val_new):<15} {match:<10}")

if not differences:
    print("\n✓ All features match!")
else:
    print(f"\n✗ Found {len(differences)} differences")

print("\n" + "=" * 80)
