"""
Save blood feature modes (imputation values) from training data.
These modes will be used to fill missing blood values for new patients.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load training blood data
blood_df = pd.read_csv('features/blood.csv', dtype={'patient_id': str})
clinical_df = pd.read_csv('features/clinical.csv', dtype={'patient_id': str})

# Get male and female patient IDs
ids_male = clinical_df[clinical_df['sex'] == 0]['patient_id'].tolist()
ids_female = clinical_df[clinical_df['sex'] == 1]['patient_id'].tolist()

# Blood feature columns (exclude patient_id)
blood_features = [col for col in blood_df.columns if col != 'patient_id']

print(f"Calculating modes for {len(blood_features)} blood features...")
print(f"  Male patients: {len(ids_male)}")
print(f"  Female patients: {len(ids_female)}")

# Calculate modes for males and females separately
modes_male = {}
modes_female = {}

for feature in blood_features:
    # Get values for males
    male_values = blood_df[blood_df['patient_id'].isin(ids_male)][feature].dropna()
    if len(male_values) > 0:
        # Calculate mode using histogram
        hist, bins = np.histogram(male_values, bins=30)
        idx_max = np.argmax(hist)
        mode_male = (bins[idx_max] + bins[idx_max + 1]) / 2
        modes_male[feature] = float(mode_male)
    
    # Get values for females
    female_values = blood_df[blood_df['patient_id'].isin(ids_female)][feature].dropna()
    if len(female_values) > 0:
        # Calculate mode using histogram
        hist, bins = np.histogram(female_values, bins=30)
        idx_max = np.argmax(hist)
        mode_female = (bins[idx_max] + bins[idx_max + 1]) / 2
        modes_female[feature] = float(mode_female)

# Save modes
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

blood_modes = {
    'modes_male': modes_male,
    'modes_female': modes_female,
    'blood_features': blood_features
}

with open(output_dir / 'blood_modes.json', 'w') as f:
    json.dump(blood_modes, f, indent=2)

print(f"\nSaved blood modes to results/blood_modes.json")
print(f"  Male modes: {len(modes_male)} features")
print(f"  Female modes: {len(modes_female)} features")
