import pandas as pd
import json
import re
import os

# ===== CONFIG =====
INPUT_FILE = "/Users/macjanine/Local/AIBE/Multimodal/new_patient_data.xlsx"
OUTPUT_DIR = "data"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== HELPERS =====
def clean_name(name):
    if pd.isna(name):
        return None
    name = str(name)
    name = name.replace("\n", " ")
    name = re.sub(r"\s+", "_", name.strip())
    return name.lower()

def is_number(x):
    return isinstance(x, (int, float)) and not pd.isna(x)

# ===== LOAD EXCEL =====
df = pd.read_excel(INPUT_FILE, header=None)
df = df.where(pd.notna(df), None)

section_row = df.iloc[1]
column_row = df.iloc[2]

original_columns = [str(c).strip() if not pd.isna(c) else None for c in column_row]
clean_columns = [clean_name(c) for c in column_row]

data = df.iloc[5:].copy()
data.columns = clean_columns
col_map = dict(zip(clean_columns, original_columns))

# Detect patient ID column
id_col = [c for c in clean_columns if c and "id" in c][0]

# ===== FIELD DEFINITIONS =====
clinical_fields = [
    "year_of_initial_diagnosis",
    "age_at_initial_diagnosis",
    "sex",
    "smoking_status",
    "primarily_metastasis",
    "survival_status",
    "survival_status_with_cause",
    "days_to_last_information",
    "first_treatment_intent",
    "first_treatment_modality",
    "days_to_first_treatment",
    "adjuvant_treatment_intent",
    "adjuvant_radiotherapy",
    "adjuvant_radiotherapy_modality",
    "adjuvant_systemic_therapy",
    "adjuvant_systemic_therapy_modality",
    "adjuvant_radiochemotherapy",
    "recurrence",
    "days_to_recurrence",
    "days_to_metastasis_1",
    "days_to_progress_1"
]

pathological_fields = [
    "primary_tumor_site",
    "pT_stage",
    "pN_stage",
    "grading",
    "hpv_association_p16",
    "number_of_positive_lymph_nodes",
    "number_of_resected_lymph_nodes",
    "perinodal_invasion",
    "lymphovascular_invasion_L",
    "vascular_invasion_V",
    "perineural_invasion_Pn",
    "resection_status",
    "resection_status_carcinoma_in_situ",
    "carcinoma_in_situ",
    "closest_resection_margin_in_cm",
    "histologic_type",
    "infiltration_depth_in_mm"
]

# ===== PROCESS PATIENTS =====
for _, row in data.iterrows():
    patient_id = str(row[id_col]).zfill(3)

    # ===== CREATE FOLDER STRUCTURE =====
    base_dir = os.path.join(OUTPUT_DIR, patient_id)
    features_dir = os.path.join(base_dir, "features")
    structured_dir = os.path.join(base_dir, "raw", "structured_data")
    text_dir = os.path.join(base_dir, "raw", "text_data", "icd_codes")
    tma_dir = os.path.join(base_dir, "raw", "tma_celldensity_measurements")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(structured_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(tma_dir, exist_ok=True)

    # Build lowercase row once
    row_dict = {str(k).lower(): v for k, v in row.items()}

    # ===== CLINICAL =====
    clinical = {
        k: row_dict.get(k.lower(), None)
        for k in clinical_fields
    }
    clinical["patient_id"] = patient_id

    with open(f"{structured_dir}/clinical_data.json", "w") as f:
        json.dump([clinical], f, indent=4)


    # ===== PATHOLOGICAL =====
    pathological = {
        k: row_dict.get(k.lower(), None)
        for k in pathological_fields
    }
    pathological["patient_id"] = patient_id
    

    with open(f"{structured_dir}/pathological_data.json", "w") as f:
        json.dump([pathological], f, indent=4)

    # ===== LOAD BLOOD REFERENCE =====
    with open("features/blood_data_reference_ranges.json") as f:
        ref_data = json.load(f)

    def normalize_loinc(s):
        if pd.isna(s):
            return ""
        s = str(s)
        s = s.replace("\n", " ")              # fix line breaks
        s = re.sub(r"\s+", " ", s)            # collapse spaces
        s = s.replace("\\/", "/")             # fix escaped slashes
        return s.strip().lower()

    # Build lookup by LOINC_name
    ref_lookup = {
        normalize_loinc(item["LOINC_name"]): item
        for item in ref_data
    }
    # ===== BLOOD =====
    # ===== BLOOD =====
    blood = []
    excluded_cols = set([id_col] + clinical_fields + pathological_fields)

    for col in data.columns:
       
        if col in excluded_cols:
            continue

        value = row[col]
        if not is_number(value):
            continue
        
        original_name = col_map.get(col, col)
        col_norm = normalize_loinc(original_name)

        if col_norm in ref_lookup:
            ref = ref_lookup[col_norm]

            blood.append({
                "patient_id": patient_id,
                "value": float(value),
                "unit": ref["unit"],
                "analyte_name": ref["analyte_name"],   # ✅ clean name
                "LOINC_code": None,                   # add later if needed
                "LOINC_name": ref["LOINC_name"],      # ✅ exact format
                "group": ref["group"],
                "days_before_first_treatment": 0
            })
        else:
            print(f"⚠️ No match for column: {col}")
        
    with open(f"{structured_dir}/blood_data.json", "w") as f:
        json.dump(blood, f, indent=4)

    # ===== ICD TEXT =====
    icd_text = ""
    for col in data.columns[::-1]:
        val = row[col]
        if isinstance(val, str) and val.strip():
            icd_text = val
            break

    with open(f"{text_dir}/icd_codes_{patient_id}.txt", "w") as f:
        f.write(icd_text)

    # ===== TMA =====
    tma_cols = ["Image", "Name", "Missing", "Centroid X µm", "Centroid Y µm", "Case ID", "Num Detections", "Num Negative", "Num Positive", "Positive %", "Num Positive per mm^2"]

    # Create empty row
    tma_row = {col: None for col in tma_cols}

    # Set patient_id as Case ID
    tma_row["Case ID"] = patient_id
    tma_row["Missing"] = False

    # Create DataFrame with one row
    tma_df = pd.DataFrame([tma_row], columns=tma_cols)


    tma_df.to_csv(f"{tma_dir}/TMA_celldensity_measurements.csv", index=False)

print("✅ Done! Clean hierarchical dataset created.")