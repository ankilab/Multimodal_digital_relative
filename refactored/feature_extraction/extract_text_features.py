import os
import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

root_path = Path(__file__).parent

def get_icd_vectors(icd_directory, save=False):
    """

    Parameters
    ----------
    icd_directory : string
        Path to the directory containing ICD codes (.txt files)

    Returns
    -------
    bow: numpy array
        One-hot-encoded ICD codes (bag of words)
    df: pandas DataFrame
        Dataframe with patient IDs and ICD codes

    """
    files = os.listdir(icd_directory)
    codes = []
    ids = []
    print(files)
    for file in files:
        file_path = os.path.join(icd_directory, file)
        with open(file_path, "r", encoding="utf-8") as text:
            # Get id from filename
            patient_id = re.search(r"([0-9]{3}).txt", file).group(1)
            file_has_lines = False
            for line in text:
                file_has_lines = True
                # Search for ICD codes
                patient_codes = re.findall(r"\[([A-Z0-9\.\s]+)\]", line)
                patient_codes = [code[:5].strip().replace(".", "") for code in patient_codes]
                patient_codes = " ".join(patient_codes)

                # Get first 4 characters of each ICD code
                codes.append(patient_codes)
                ids.append(patient_id)
            if not file_has_lines:
                ids.append(patient_id)
                codes.append("")

    # Create dataframe
    #print(len(ids))
    #print(len(codes))
    df = pd.DataFrame({"patient_id": pd.Series(ids, dtype=str), "icd_code": pd.Series(codes, dtype=str)})
    print(df)
    if save:
        cv = CountVectorizer(ngram_range=(1, 1), min_df=3)
        cv = cv.fit(df.icd_code)
        joblib.dump(cv, root_path.parent / "models/icd_vectorizer.joblib")
    else:
        cv = joblib.load(root_path.parent / "models/icd_vectorizer.joblib")

    bow = cv.transform(df.icd_code)

    bow_df = pd.DataFrame(data=bow.toarray(), columns=cv.get_feature_names_out().tolist())
    bow_df.insert(0, "patient_id", df.patient_id)

    return bow, bow_df, df
