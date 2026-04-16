"""
Utility functions for Multimodal Patient Visualization.

This module contains helper functions for feature decoding, feature grouping,
and attribute table generation for the HANCOCK dashboard.
"""

import pandas as pd
import joblib
from pathlib import Path
import json
from dash import html, dash_table


def get_feature_groups(features_dir="./features"):
    """
    Load feature groups from CSV files in the features directory.

    Parameters
    ----------
    features_dir : str
        Path to the features directory

    Returns
    -------
    dict
        Mapping of feature group names to lists of feature columns
    """
    groups = {}
    files = {
        'clinical.csv': 'Clinical Data',
        'pathological.csv': 'Pathological Data',
        'blood.csv': 'Blood Data',
        'tma_cell_density.csv': 'TMA Data',
        'icd_codes.csv': 'ICD Codes',
        'targets.csv': 'Survival and Treatment Data'
    }

    for filename, group_name in files.items():
        path = Path(features_dir) / filename
        if path.exists():
            try:
                df = pd.read_csv(path, nrows=0)
                features = [c for c in df.columns if c != 'patient_id']
                groups[group_name] = features
            except Exception:
                pass

    return groups


def decode_features(df):
    """
    Decode features from numerical values back to strings for display.

    Handles binary features, nominal features, and ordinal features by loading
    label encoders and performing inverse transforms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with encoded features

    Returns
    -------
    pd.DataFrame
        DataFrame with decoded features for display
    """
    df_decoded = df.copy()

    # Binary feature mappings
    binary_mappings = {
        'sex': {0: 'male', 1: 'female'},
        'primarily_metastasis': {0: 'no', 1: 'yes'},
        'lymphovascular_invasion_L': {0: 'no', 1: 'yes'},
        'vascular_invasion_V': {0: 'no', 1: 'yes'},
        'perineural_invasion_Pn': {0: 'no', 1: 'yes'},
        'perinodal_invasion': {0: 'no', 1: 'yes'},
        'carcinoma_in_situ': {0: 'Absent', 1: 'CIS'},
    }

    for col, mapping in binary_mappings.items():
        if col in df_decoded.columns:
            df_decoded[col] = df_decoded[col].map(mapping).fillna(df_decoded[col])

    # Load nominal/ordinal label encoders
    root_dir = Path(__file__).resolve().parents[1]

    for col in df_decoded.columns:
        nominal_path = root_dir / f"models/{col}_nominal_labelencoder.joblib"
        ordinal_path = root_dir / f"models/{col}_ordinal_labelencoder.joblib"

        encoder_path = None
        if nominal_path.exists():
            encoder_path = nominal_path
        elif ordinal_path.exists():
            encoder_path = ordinal_path

        if encoder_path:
            try:
                le = joblib.load(encoder_path)
                valid_mask = df_decoded[col].notna()
                unique_vals = df_decoded.loc[valid_mask, col].unique()

                mapping = {}
                for val in unique_vals:
                    try:
                        val_int = int(round(val))
                        decoded_val = le.inverse_transform([val_int])[0]
                        mapping[val] = decoded_val
                    except Exception:
                        pass

                df_decoded[col] = df_decoded[col].map(mapping).fillna(
                    df_decoded[col]
                )

            except Exception as e:
                print(f"Failed to decode {col}: {e}")

    return df_decoded


def _load_blood_reference_ranges():
    """
    Load blood test reference ranges from JSON file.

    Returns
    -------
    tuple
        (blood_ref_ranges, blood_ref_limits) dictionaries
    """
    blood_ref_ranges = {}
    blood_ref_limits = {}

    try:
        ref_path = Path("./features/blood_data_reference_ranges.json")
        if ref_path.exists():
            with open(ref_path, 'r') as f:
                ref_data = json.load(f)
                for item in ref_data:
                    loinc = item.get('LOINC_name')
                    unit = item.get('unit', '')
                    min_male = item.get('normal_male_min')
                    max_male = item.get('normal_male_max')
                    min_female = item.get('normal_female_min')
                    max_female = item.get('normal_female_max')

                    blood_ref_limits[loinc] = {
                        'male': (min_male, max_male),
                        'female': (min_female, max_female)
                    }

                    range_str = ""
                    if min_male is not None or max_male is not None:
                        range_str += f"Male: {min_male}-{max_male}"
                    if min_female is not None or max_female is not None:
                        if range_str:
                            range_str += "; "
                        range_str += f"Female: {min_female}-{max_female}"

                    if unit:
                        range_str += f" {unit}"

                    if loinc and range_str:
                        blood_ref_ranges[loinc] = range_str
    except Exception as e:
        print(f"Error loading reference ranges: {e}")

    return blood_ref_ranges, blood_ref_limits


def _load_icd_dictionary():
    """
    Load ICD code to description mapping.

    Returns
    -------
    dict
        Mapping of ICD codes to descriptions
    """
    icd_dict = {}
    try:
        icd_path = Path("./features/icd_codes_dictionary.csv")
        if icd_path.exists():
            df_icd = pd.read_csv(icd_path)
            icd_dict = pd.Series(
                df_icd.Description.values, index=df_icd.Code
            ).to_dict()
    except Exception as e:
        print(f"Error loading ICD dictionary: {e}")

    return icd_dict


def generate_attribute_table(df_full, target_patient, comparison_patients=None):
    """
    Generate an interactive attribute comparison table for patient(s).

    Creates a detailed HTML table showing patient attributes organized by
    feature groups with special formatting for blood data ranges and
    ICD code descriptions.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full patient dataframe with all features
    target_patient : str
        Patient ID to use as the primary comparison target
    comparison_patients : list, optional
        List of patient IDs to compare against target. Defaults to None.

    Returns
    -------
    list
        List of HTML/Dash components for display
    """
    if comparison_patients is None:
        comparison_patients = []

    patient_ids = [target_patient] + comparison_patients
    df_subset = df_full[df_full['patient_id'].isin(patient_ids)].copy()
    df_decoded = decode_features(df_subset)

    df_decoded = df_decoded.set_index('patient_id')
    df_decoded = df_decoded.reindex(patient_ids)
    df_transposed = df_decoded.T.reset_index().rename(
        columns={'index': 'Attribute'}
    )

    columns = df_transposed.columns.tolist()
    target_col = target_patient

    # Style definitions
    style_data_conditional = [
        {
            'if': {'column_id': target_col},
            'backgroundColor': '#dbeafe',
            'fontWeight': 'bold'
        }
    ]

    if comparison_patients:
        for col in comparison_patients:
            style_data_conditional.append({
                'if': {
                    'filter_query': f'{{{col}}} != {{{target_col}}}',
                    'column_id': col
                },
                'backgroundColor': '#ffcccc',
                'color': 'black'
            })

    feature_groups = get_feature_groups()
    children = []

    # Summary section
    summary_features = ['dataset', 'UMAP 1', 'UMAP 2']
    if 'Similarity' in df_transposed['Attribute'].values:
        summary_features.insert(0, 'Similarity')

    df_summary = df_transposed[
        df_transposed['Attribute'].isin(summary_features)
    ]

    if not df_summary.empty:
        children.append(html.H4("Summary"))
        children.append(html.Div(dash_table.DataTable(
            data=df_summary.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in columns],
            style_table={'overflowX': 'auto', 'marginBottom': '20px'},
            style_cell={
                'textAlign': 'left',
                'minWidth': '150px',
                'width': '150px',
                'maxWidth': '150px',
                'whiteSpace': 'normal',
                'hyphens': 'auto',
                'wordBreak': 'break-word'
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Attribute'},
                    'width': '200px',
                    'minWidth': '200px',
                    'maxWidth': '200px'
                }
            ],
            style_data_conditional=style_data_conditional
        )))

    # Load reference data
    blood_ref_ranges, blood_ref_limits = _load_blood_reference_ranges()
    icd_dict = _load_icd_dictionary()

    # Get patient sex for blood data validation
    patient_sex = {}
    if 'sex' in df_transposed['Attribute'].values:
        sex_row = df_transposed[df_transposed['Attribute'] == 'sex'].iloc[0]
        for pid in patient_ids:
            if pid in sex_row:
                patient_sex[pid] = sex_row[pid]

    group_order = [
        'Clinical Data',
        'Pathological Data',
        'Blood Data',
        'TMA Data',
        'ICD Codes',
        'Survival and Treatment Data'
    ]

    for group_name in group_order:
        if group_name not in feature_groups:
            continue

        features = feature_groups[group_name]
        df_group = df_transposed[df_transposed['Attribute'].isin(features)]

        if df_group.empty:
            continue

        tooltip_data = None
        current_style = style_data_conditional.copy()

        if group_name == 'Blood Data':
            tooltip_data = _build_blood_tooltips(
                df_group, columns, blood_ref_ranges, blood_ref_limits,
                patient_ids, patient_sex, current_style
            )

        elif group_name == 'ICD Codes':
            tooltip_data = _build_icd_tooltips(
                df_group, columns, icd_dict
            )

        table_component = html.Div(dash_table.DataTable(
            data=df_group.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'minWidth': '150px',
                'width': '150px',
                'maxWidth': '150px',
                'whiteSpace': 'normal',
                'hyphens': 'auto',
                'wordBreak': 'break-word'
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Attribute'},
                    'width': '200px',
                    'minWidth': '200px',
                    'maxWidth': '200px'
                }
            ],
            style_data_conditional=current_style,
            page_size=20 if group_name != 'ICD Codes' else 10,
            tooltip_data=tooltip_data,
            tooltip_duration=None,
        ))

        details = html.Details([
            html.Summary(
                html.B(f"{group_name} ({len(df_group)} attributes)"),
                style={
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'padding': '10px',
                    'backgroundColor': '#f0f0f0',
                    'color': 'black',
                    'position': 'relative',
                    'zIndex': '10'
                }
            ),
            html.Div(
                table_component,
                style={
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'borderTop': 'none'
                }
            )
        ], open=True, style={
            'marginBottom': '10px',
            'border': '1px solid #ccc',
            'borderRadius': '5px'
        })

        children.append(details)

    # Other attributes
    all_grouped_features = sum(feature_groups.values(), []) + summary_features
    df_other = df_transposed[
        ~df_transposed['Attribute'].isin(all_grouped_features)
    ]

    if not df_other.empty:
        children.append(html.Details([
            html.Summary(
                html.B(f"Other Attributes ({len(df_other)})"),
                style={
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'padding': '10px',
                    'backgroundColor': '#f0f0f0'
                }
            ),
            html.Div(dash_table.DataTable(
                data=df_other.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '150px',
                    'width': '150px',
                    'maxWidth': '150px',
                    'whiteSpace': 'normal'
                },
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Attribute'},
                        'width': '200px',
                        'minWidth': '200px',
                        'maxWidth': '200px'
                    }
                ],
                style_data_conditional=style_data_conditional
            ))
        ]))

    return children


def _build_blood_tooltips(df_group, columns, blood_ref_ranges, blood_ref_limits,
                          patient_ids, patient_sex, style_data_conditional):
    """
    Build tooltip data and conditional styles for blood data.

    Parameters
    ----------
    df_group : pd.DataFrame
        DataFrame subset for blood group
    columns : list
        Column names for the table
    blood_ref_ranges : dict
        Blood reference range strings
    blood_ref_limits : dict
        Blood reference limit values
    patient_ids : list
        Patient IDs to validate
    patient_sex : dict
        Mapping of patient ID to sex
    style_data_conditional : list
        List to append conditional styles to

    Returns
    -------
    list
        Tooltip data for each row
    """
    tooltip_data = []
    for i, row in enumerate(df_group.to_dict('records')):
        attr = row['Attribute']

        if attr in blood_ref_ranges:
            row_tooltip = {
                col: {
                    'value': f"Reference Range: {blood_ref_ranges[attr]}",
                    'type': 'markdown'
                }
                for col in columns
            }
        else:
            row_tooltip = {col: None for col in columns}

        tooltip_data.append(row_tooltip)

        if attr in blood_ref_limits:
            limits = blood_ref_limits[attr]
            for pid in patient_ids:
                if pid not in row or pid not in patient_sex:
                    continue

                sex = patient_sex[pid]
                val = row[pid]

                try:
                    val_float = float(val)
                    sex_limits = limits.get(sex)
                    if not sex_limits:
                        continue

                    min_val, max_val = sex_limits
                    is_out = False

                    if min_val is not None and val_float < min_val:
                        is_out = True
                    if max_val is not None and val_float > max_val:
                        is_out = True

                    if is_out:
                        style_data_conditional.append({
                            'if': {
                                'row_index': i,
                                'column_id': pid
                            },
                            'backgroundColor': '#8B0000',
                            'color': 'white'
                        })
                except Exception:
                    pass

    return tooltip_data


def _build_icd_tooltips(df_group, columns, icd_dict):
    """
    Build tooltip data for ICD codes with descriptions.

    Parameters
    ----------
    df_group : pd.DataFrame
        DataFrame subset for ICD codes
    columns : list
        Column names for the table
    icd_dict : dict
        Mapping of ICD codes to descriptions

    Returns
    -------
    list
        Tooltip data for each row
    """
    tooltip_data = []
    for row in df_group.to_dict('records'):
        attr = row['Attribute']
        desc = icd_dict.get(attr)

        if desc:
            row_tooltip = {
                col: {'value': f"**{attr}**: {desc}", 'type': 'markdown'}
                for col in columns
            }
        else:
            row_tooltip = {col: None for col in columns}

        tooltip_data.append(row_tooltip)

    return tooltip_data
