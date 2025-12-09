import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from dash import Dash, dcc, html, Output, Input, State, callback_context, dash_table, no_update
from pathlib import Path
import sys
import os

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import our encoding functions
from feature_extraction.extract_text_features import get_icd_vectors
from feature_extraction.extract_tabular_features import get_tabular_features, get_blood_features, get_target_classes
from feature_extraction.extract_tma_features import get_tma_features

# from feature_extraction.encode_new_patient import preprocess_patient_data_from_dict
from data_exploration.umap_embedding import get_umap_embedding

# -----------------------------------------------------------------------------
# 1. Load Training Data and UMAP Embeddings
# -----------------------------------------------------------------------------
print("Loading training data and UMAP embeddings...")
# This gets the expected embeddings for all training patients
df_train = get_umap_embedding("./features", umap_min_dist=0.1, umap_n_neighbors=15)
df_train['dataset'] = 'Training'

# -----------------------------------------------------------------------------
# 2. Encode New Patient (001)
# -----------------------------------------------------------------------------
print("Encoding new patient 001...")

# Load models and metadata
preprocessor = joblib.load('results/preprocessor.pkl')
umap_model = joblib.load('results/umap_model.pkl')
with open('results/feature_order.json') as f:
    feature_order = json.load(f)
with open('results/umap_normalization.json') as f:
    norm_params = json.load(f)

def load_and_encode_patient(base_path_str, patient_id="Unknown"):
    """
    Load raw data from a directory and encode it.
    Returns a DataFrame row with the encoded patient data and UMAP coordinates.
    """
    base_path = Path(base_path_str)
    dest_dir = base_path.parent / "features"
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
    # Save text features (ICD codes as bag of words)
    icd_vectors, icd_df, _ = get_icd_vectors(base_path / "text_data/icd_codes", save=False)
    icd_df.to_csv(dest_dir/"icd_codes.csv", index=False)

    # Save clinical features
    clinical_vectors, clinical_df = get_tabular_features(
        base_path / "structured_data/clinical_data.json", save=False)
    clinical_df.to_csv(dest_dir/"clinical.csv", index=False)

    # Save pathological features
    patho_vectors, patho_df = get_tabular_features(
        base_path / "structured_data/pathological_data.json", save=False)
    patho_df.to_csv(dest_dir/"pathological.csv", index=False)

    # Save blood parameters
    blood_vectors, blood_df = get_blood_features(
        file_path_blood=base_path / "structured_data/blood_data.json",
        file_path_normal=base_path / "structured_data/blood_data_reference_ranges.json",
        file_path_clinical=base_path / "structured_data/clinical_data.json",
        save=False
    )
    blood_df.to_csv(dest_dir / "blood.csv", index=False)

    # Save cell densities from CD3 and CD8 TMAs
    tma_vectors, tma_df = get_tma_features(base_path / "tma_celldensity_measurements/TMA_celldensity_measurements.csv")
    tma_df.to_csv(dest_dir/"tma_cell_density.csv", index=False)

    # Save target classes (recurrence, survival, treatment)
    target_df = get_target_classes(base_path / "structured_data/clinical_data.json")
    target_df.to_csv(dest_dir/"targets.csv", index=False)

    # encode data 
    # Load encoded data
    clinical = pd.read_csv(dest_dir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(dest_dir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(dest_dir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(dest_dir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density= pd.read_csv(dest_dir/"tma_cell_density.csv", dtype={"patient_id": str})
    targets = pd.read_csv(dest_dir/"targets.csv", dtype={"patient_id": str})

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.merge(targets, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    # feature order 
    # with open(results_dir / "feature_order.json", "r") as f:
    #     feature_order = json.load(f)

    # df = df[feature_order]

    # project into UMAP space
    # Preprocess embeddings
    results_dir = "./"
    preprocessor = joblib.load(results_dir + "preprocessor.pkl")
    
    # Exclude target columns from UMAP input
    target_cols = [c for c in targets.columns if c != "patient_id"]
    df_for_umap = df.drop(["patient_id"] + target_cols, axis=1)
    
    print(df_for_umap)
    embeddings = preprocessor.transform(df_for_umap)

    # Reduce to 2D
    umap_model = joblib.load(results_dir + "umap_model.pkl")
    umap_embedding = umap_model.transform(embeddings)
        

    # Add UMAP to the dataframe

    # Create DataFrame row
    df_new = df.copy()
    df_new['patient_id'] = f"New_{patient_id}"
    df_new["UMAP 1"] = umap_embedding[:, 0]
    df_new["UMAP 2"] = umap_embedding[:, 1]
    df_new['dataset'] = 'New Patient'
            
    return df_new

# Initial load of patient 001
print("Encoding initial patient 002...")
df_new_001 = load_and_encode_patient(f"test_patient_040/raw", "040")

# Concatenate
df_combined = pd.concat([df_train, df_new_001], ignore_index=True)

# -----------------------------------------------------------------------------
# 3. Dash App
# -----------------------------------------------------------------------------
app = Dash(__name__)

# Features to show in dropdown
hue_options = ['dataset'] + [col for col in df_combined.columns if col not in ['patient_id', 'UMAP 1', 'UMAP 2', 'dataset']]

app.layout = html.Div([
    html.H1("Multimodal Digital Relative ", style={'textAlign': 'center'}),
    html.H3("UMAP Visualization Dashboard for HANCOCK", style={'textAlign': 'center'}),
    
    # Control Panel Section (Combined)
    html.Div([
        # 1. Color By Dropdown
        html.Div([
            html.Label("Color by:"),
            dcc.Dropdown(
                id='hue-dropdown',
                options=[{'label': c, 'value': c} for c in hue_options],
                value='dataset',
                clearable=False
            )
        ], className='control-item', style={'flex': '1'}),
        
        # 2. Highlight Patient Dropdown
        html.Div([
            html.Label("Highlight Patient:"),
            dcc.Dropdown(
                id='patient-dropdown',
                # Options will be updated via callback
                options=[{'label': p, 'value': p} for p in ['None'] + sorted(df_combined['patient_id'].unique())],
                value="New_001",
                clearable=False
            )
        ], className='control-item', style={'flex': '1'}),
        
        # 3. Find Similar Button
        html.Div([
            html.Button('Find Similar Patients', id='find-similar-btn', n_clicks=0, className='custom-btn'),
        ], className='control-item', style={'minWidth': 'auto', 'justifyContent': 'flex-end'}),

        # 4. Load Patient Input
        html.Div([
            html.Label("Load New Patient Data (Path):"),
            dcc.Input(
                id='path-input',
                type='text',
                placeholder='/path/to/patient/raw',
                className='custom-input',
                style={'width': '100%'}
            ),
        ], className='control-item', style={'flex': '2'}), # Give it more space

        # 5. Load Button
        html.Div([
            html.Button('Load Patient', id='load-btn', n_clicks=0, className='custom-btn'),
        ], className='control-item', style={'minWidth': 'auto', 'justifyContent': 'flex-end'}),

        # Status Message (can be next to load button or at end)
        html.Div(id='status-msg', className='status-msg', style={'color': 'blue'}),

    ], className='control-panel', style={'zIndex': '3000', 'position': 'relative'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id='umap-graph', style={'height': '80vh', 'width': '100%'}),
        ], style={'width': '50%', 'height': '80vh', 'zIndex': '1', 'position': 'relative'}),
        
        html.Div([
            html.H3("Top 5 Similar Patients", id='side-panel-title'),
            html.Div(id='similarity-table-container')
        ], style={
            'width': '50%',
            'height': '80vh',
            'overflowY': 'auto',
            'backgroundColor': '#ffffff',
            'padding': '20px',
            # Removed border, borderRadius, boxShadow as requested
            'zIndex': 2000,
            'position': 'relative'
        })
    ], style={'display': 'flex', 'flexDirection': 'row', 'height': '80vh'}), # Flex container
    
    # Store for new patients
    dcc.Store(id='new-patients-store', data=[]),  # List of records
    # Store for similar patients IDs
    dcc.Store(id='similar-patients-store', data=[])
], style={'fontFamily': 'Arial, sans-serif'})

# Callback to load new patient
@app.callback(
    [Output('new-patients-store', 'data'),
     Output('status-msg', 'children'),
     Output('patient-dropdown', 'value', allow_duplicate=True)],
    [Input('load-btn', 'n_clicks')],
    [State('path-input', 'value'),
     State('new-patients-store', 'data')],
    prevent_initial_call=True
)
def load_patient(n_clicks, path, current_data):
    if n_clicks == 0 or not path:
        return current_data, "", no_update
    
    try:
        # Extract patient ID from path if possible, or generate one
        # Assuming path ends in .../patient_ID/raw or similar
        # Simple heuristic: take parent folder name if it looks like an ID, else 'Unknown'
        p = Path(path)
        # Try to find ID in path parts
        # e.g. .../test_patient_001/raw -> 001
        pt_id = "Unknown"
        for part in p.parts:
            if "patient_" in part:
                pt_id = part.split("patient_")[-1]
                break
        
        # Load and encode
        df_new_pt = load_and_encode_patient(path, pt_id)
        print(df_new_pt)
        
        # Convert to records for storage
        new_records = df_new_pt.to_dict('records')
        
        # Append to current data
        updated_data = current_data + new_records
        
        return updated_data, f"Successfully loaded patient {df_new_pt['patient_id'].iloc[0]}", df_new_pt['patient_id'].iloc[0]
    except Exception as e:
        return current_data, f"Error loading patient: {str(e)}", no_update

# Callback to update graph and dropdowns
@app.callback(
    [Output('umap-graph', 'figure'),
     Output('patient-dropdown', 'options')],
    [Input('hue-dropdown', 'value'),
     Input('patient-dropdown', 'value'),
     Input('new-patients-store', 'data'),
     Input('similar-patients-store', 'data')]
)
def update_graph(hue, highlight_patient, new_patients_data, similar_patients_ids):
    # Combine initial data with new patients
    if new_patients_data:
        df_new_pts = pd.DataFrame(new_patients_data)
        # Ensure columns match for concatenation (fill missing with NaN)
        df_plot = pd.concat([df_combined, df_new_pts], ignore_index=True)
    else:
        df_plot = df_combined.copy()

    # Decode features for display in plot (color by)
    df_plot = decode_features(df_plot)

    # Update dropdown options
    patient_options = [{'label': p, 'value': p} for p in ['None'] + sorted(df_plot['patient_id'].unique())]

    # Create scatter plot
    fig = px.scatter(
        df_plot, 
        x='UMAP 1', 
        y='UMAP 2', 
        color=hue,
        hover_data=['patient_id'],
        # title=f"UMAP Colored by {hue}", # Removed title
        color_discrete_sequence=px.colors.qualitative.Bold if hue == 'dataset' else px.colors.qualitative.Safe,
        # width=700, # Removed fixed width
        # height=700, # Removed fixed height
        # Increase right margin to accommodate legend without shrinking plot
        #margin=dict(l=50, r=400, t=50, b=50),
        #autosize=False
            )
    # Highlight similar patients
    if similar_patients_ids:
        sim_data = df_plot[df_plot['patient_id'].isin(similar_patients_ids)]
        if not sim_data.empty:
             fig.add_trace(
                go.Scatter(
                    x=sim_data['UMAP 1'],
                    y=sim_data['UMAP 2'],
                    mode='markers',
                    marker=dict(size=12, color='orange', symbol='diamond', line=dict(width=0)),
                    name="Similar Patients",
                    hoverinfo='skip' # Or show ID
                )
            )

    # Highlight selected patient
    if highlight_patient != 'None' and highlight_patient in df_plot['patient_id'].values:
        pt_data = df_plot[df_plot['patient_id'] == highlight_patient]
        if not pt_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=pt_data['UMAP 1'],
                    y=pt_data['UMAP 2'],
                    mode='markers',
                    marker=dict(size=15, color='#04316A', symbol='circle', line=dict(width=0)),
                    name=f"Selected: {highlight_patient}",
                    hoverinfo='skip'
                )
            )
            
    # Calculate fixed axis limits with some padding
    x_min, x_max = df_plot['UMAP 1'].min(), df_plot['UMAP 1'].max()
    y_min, y_max = df_plot['UMAP 2'].min(), df_plot['UMAP 2'].max()
    padding_x = (x_max - x_min) * 0.05
    padding_y = (y_max - y_min) * 0.05
    
    fig.update_layout(
        clickmode='event+select',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(
            scaleanchor="y", 
            scaleratio=1, 
            fixedrange=True, 
            constrain='domain',
            showticklabels=False, # Hide numbers
            range=[x_min - padding_x, x_max + padding_x], # Fixed range
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=False # Only bottom line
        ),
        yaxis=dict(
            scaleanchor="x", 
            scaleratio=1, 
            fixedrange=True, 
            constrain='domain',
            showticklabels=False, # Hide numbers
            range=[y_min - padding_y, y_max + padding_y], # Fixed range
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=False # Only left line
        ),
        #width=800, # Reduced width
        #height=650, # Reduced height
        # Adjusted margins: reduced right (since legend is bottom), increased bottom (for legend/colorbar)
        # Reduced left margin as requested
        margin=dict(l=0, r=50, t=20, b=200),
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        coloraxis_colorbar=dict(
            orientation="h",
            yanchor="top",
            y=-0.2, # Slightly lower to avoid overlap with axis titles if any, or same as legend
            xanchor="center",
            x=0.5,
            title=dict(side="top")
        )
    )
    return fig, patient_options

# Helper to get feature groups
def get_feature_groups(features_dir="./features"):
    groups = {}
    # Map filename to display name
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
            # Read only headers
            try:
                df = pd.read_csv(path, nrows=0)
                # Exclude patient_id
                features = [c for c in df.columns if c != 'patient_id']
                groups[group_name] = features
            except:
                pass
            
    return groups

def decode_features(df):
    """
    Decode features from numerical values back to strings for display.
    """
    df_decoded = df.copy()
    
    # 1. Binary Features (Hardcoded based on extract_tabular_features.py)
    # "sex": {0: 'male', 1: 'female'}
    if 'sex' in df_decoded.columns:
        df_decoded['sex'] = df_decoded['sex'].map({0: 'male', 1: 'female'}).fillna(df_decoded['sex'])
        
    # "primarily_metastasis": {0: 'no', 1: 'yes'}
    if 'primarily_metastasis' in df_decoded.columns:
        df_decoded['primarily_metastasis'] = df_decoded['primarily_metastasis'].map({0: 'no', 1: 'yes'}).fillna(df_decoded['primarily_metastasis'])
        
    # "lymphovascular_invasion_L": {0: 'no', 1: 'yes'}
    if 'lymphovascular_invasion_L' in df_decoded.columns:
        df_decoded['lymphovascular_invasion_L'] = df_decoded['lymphovascular_invasion_L'].map({0: 'no', 1: 'yes'}).fillna(df_decoded['lymphovascular_invasion_L'])

    # "vascular_invasion_V": {0: 'no', 1: 'yes'}
    if 'vascular_invasion_V' in df_decoded.columns:
        df_decoded['vascular_invasion_V'] = df_decoded['vascular_invasion_V'].map({0: 'no', 1: 'yes'}).fillna(df_decoded['vascular_invasion_V'])

    # "perineural_invasion_Pn": {0: 'no', 1: 'yes'}
    if 'perineural_invasion_Pn' in df_decoded.columns:
        df_decoded['perineural_invasion_Pn'] = df_decoded['perineural_invasion_Pn'].map({0: 'no', 1: 'yes'}).fillna(df_decoded['perineural_invasion_Pn'])

    # "perinodal_invasion": {0: 'no', 1: 'yes'}
    if 'perinodal_invasion' in df_decoded.columns:
        df_decoded['perinodal_invasion'] = df_decoded['perinodal_invasion'].map({0: 'no', 1: 'yes'}).fillna(df_decoded['perinodal_invasion'])
        
    # "carcinoma_in_situ": {0: 'Absent', 1: 'CIS'}
    if 'carcinoma_in_situ' in df_decoded.columns:
        df_decoded['carcinoma_in_situ'] = df_decoded['carcinoma_in_situ'].map({0: 'Absent', 1: 'CIS'}).fillna(df_decoded['carcinoma_in_situ'])

    # 2. Nominal/Ordinal Features (Load LabelEncoders)
    # Path to label encoders (project root)
    root_dir = Path(__file__).resolve().parents[1]
    
    # List of features that might have label encoders
    # We'll try to find any matching file
    for col in df_decoded.columns:
        # Check for nominal or ordinal encoder
        nominal_path = root_dir / f"{col}_nominal_labelencoder.joblib"
        ordinal_path = root_dir / f"{col}_ordinal_labelencoder.joblib"
        
        encoder_path = None
        if nominal_path.exists():
            encoder_path = nominal_path
        elif ordinal_path.exists():
            encoder_path = ordinal_path
            
        if encoder_path:
            try:
                le = joblib.load(encoder_path)
                # Inverse transform
                # Handle NaNs or unknown values gracefully
                # We need to ensure values are integers for inverse_transform
                # But some might be NaN
                
                # Create a mask for valid values
                valid_mask = df_decoded[col].notna()
                
                # Get unique valid values to map
                unique_vals = df_decoded.loc[valid_mask, col].unique()
                
                # Create a mapping dictionary
                mapping = {}
                for val in unique_vals:
                    try:
                        # Round to nearest int if it's a float (common in pandas with NaNs)
                        val_int = int(round(val))
                        decoded_val = le.inverse_transform([val_int])[0]
                        mapping[val] = decoded_val
                    except:
                        # If decoding fails (e.g. value not in encoder), keep original
                        pass
                
                # Apply mapping
                df_decoded[col] = df_decoded[col].map(mapping).fillna(df_decoded[col])
                
            except Exception as e:
                print(f"Failed to decode {col}: {e}")
                
    return df_decoded

# Helper to generate attribute table
def generate_attribute_table(df_full, target_patient, comparison_patients=[]):
    # Prepare data
    patient_ids = [target_patient] + comparison_patients
    
    # Filter for these patients
    df_subset = df_full[df_full['patient_id'].isin(patient_ids)].copy()
    
    # Decode features for display
    df_decoded = decode_features(df_subset)
    
    # Ensure order matches input list
    df_decoded = df_decoded.set_index('patient_id')
    df_decoded = df_decoded.reindex(patient_ids)
    
    # Transpose
    df_transposed = df_decoded.T.reset_index().rename(columns={'index': 'Attribute'})
    
    # Define columns
    columns = df_transposed.columns.tolist()
    target_col = target_patient
    
    # Define style
    style_data_conditional = []
    
    # Highlight Target Column
    style_data_conditional.append({
        'if': {'column_id': target_col},
        'backgroundColor': '#dbeafe', # Light blue matching theme
        'fontWeight': 'bold'
    })
    
    # Highlight differences if comparison patients exist
    if comparison_patients:
        for col in comparison_patients:
            style_data_conditional.append({
                'if': {
                    'filter_query': f'{{{col}}} != {{{target_col}}}',
                    'column_id': col
                },
                'backgroundColor': '#ffcccc', # Light red
                'color': 'black'
            })

    # Get Feature Groups
    feature_groups = get_feature_groups()
    
    # Create HTML components
    children = []
    
    # 1. Summary Section
    summary_features = ['dataset', 'UMAP 1', 'UMAP 2']
    if 'Similarity' in df_transposed['Attribute'].values:
        summary_features.insert(0, 'Similarity')
        
    df_summary = df_transposed[df_transposed['Attribute'].isin(summary_features)]
    
    if not df_summary.empty:
        children.append(html.H4("Summary"))
        children.append(html.Div(dash_table.DataTable(
            data=df_summary.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in columns],
            style_table={'overflowX': 'auto', 'marginBottom': '20px'},
            style_cell={'textAlign': 'left', 'minWidth': '150px', 'width': '150px', 'maxWidth': '150px', 'whiteSpace': 'normal', 'hyphens': 'auto', 'wordBreak': 'break-word'},
            style_cell_conditional=[
                {'if': {'column_id': 'Attribute'},
                 'width': '200px', 'minWidth': '200px', 'maxWidth': '200px'}
            ],
            style_data_conditional=style_data_conditional
        ), className='sync-table')) # Added class for sync scroll
        

        
    # 2. Grouped Sections
    group_order = ['Clinical Data', 'Pathological Data', 'Blood Data', 'TMA Data', 'ICD Codes']
    
    # 2. Grouped Sections
    group_order = ['Clinical Data', 'Pathological Data', 'Blood Data', 'TMA Data', 'ICD Codes', 'Survival and Treatment Data']
    
    # Load reference ranges for Blood Data
    blood_ref_ranges = {}
    blood_ref_limits = {} # Store min/max for logic
    try:
        # Assuming path relative to script or fixed path
        # We'll try to find the file in one of the patient directories or a common location
        # For now, let's look in test_patient_001 as a fallback/default
        ref_path = Path("./test_patient_001/raw/structured_data/blood_data_reference_ranges.json")
        if ref_path.exists():
            with open(ref_path, 'r') as f:
                ref_data = json.load(f)
                for item in ref_data:
                    # Key: LOINC_name (matches column name in blood.csv)
                    # Value: Range string
                    loinc = item.get('LOINC_name')
                    unit = item.get('unit', '')
                    min_male = item.get('normal_male_min')
                    max_male = item.get('normal_male_max')
                    min_female = item.get('normal_female_min')
                    max_female = item.get('normal_female_max')
                    
                    # Store limits
                    blood_ref_limits[loinc] = {
                        'male': (min_male, max_male),
                        'female': (min_female, max_female)
                    }

                    # Construct range string
                    # Simplified: showing both male/female if different, or just one if same/unspecified
                    # For tooltip, let's show both
                    
                    range_str = ""
                    if min_male is not None or max_male is not None:
                         range_str += f"Male: {min_male}-{max_male}"
                    if min_female is not None or max_female is not None:
                        if range_str: range_str += "; "
                        range_str += f"Female: {min_female}-{max_female}"
                    
                    if unit:
                        range_str += f" {unit}"
                        
                    if loinc and range_str:
                        blood_ref_ranges[loinc] = range_str
    except Exception as e:
        print(f"Error loading reference ranges: {e}")

    # Get patient sex mapping for highlighting
    # We need to use the decoded 'sex' values ('male', 'female')
    # df_subset has the raw data, but we need decoded sex.
    # We can get it from df_decoded which we used to create df_transposed
    # But df_decoded was local to generate_attribute_table and not returned.
    # Wait, generate_attribute_table calls decode_features internally.
    # So df_subset is raw, but we need decoded sex.
    # Let's decode df_subset again just to get sex mapping, or modify flow.
    # Actually, df_transposed comes from df_decoded.
    # But df_transposed is transposed, so 'sex' is a row now.
    
    patient_sex = {}
    if 'sex' in df_transposed['Attribute'].values:
        sex_row = df_transposed[df_transposed['Attribute'] == 'sex'].iloc[0]
        for pid in patient_ids:
            if pid in sex_row:
                patient_sex[pid] = sex_row[pid] # Should be 'male' or 'female'

    # Load ICD Dictionary
    icd_dict = {}
    try:
        icd_path = Path("./features/icd_codes_dictionary.csv")
        if icd_path.exists():
            df_icd = pd.read_csv(icd_path)
            # Create dict: Code -> Description
            icd_dict = pd.Series(df_icd.Description.values, index=df_icd.Code).to_dict()
    except Exception as e:
        print(f"Error loading ICD dictionary: {e}")

    for group_name in group_order:
        if group_name in feature_groups:
            features = feature_groups[group_name]
            df_group = df_transposed[df_transposed['Attribute'].isin(features)]
            
            if not df_group.empty:
                # Tooltips and Highlighting for Blood Data
                tooltip_data = None
                current_style_data_conditional = style_data_conditional.copy()
                
                if group_name == 'Blood Data':
                    tooltip_data = []
                    # Iterate over rows to build tooltips and conditional styles
                    for i, row in enumerate(df_group.to_dict('records')):
                        attr = row['Attribute']
                        
                        # Tooltips
                        if attr in blood_ref_ranges:
                            row_tooltip = {col: {'value': f"Reference Range: {blood_ref_ranges[attr]}", 'type': 'markdown'} for col in columns}
                        else:
                            row_tooltip = {col: None for col in columns}
                        tooltip_data.append(row_tooltip)
                        
                        # Highlighting
                        if attr in blood_ref_limits:
                            limits = blood_ref_limits[attr]
                            for pid in patient_ids:
                                if pid in row and pid in patient_sex:
                                    sex = patient_sex[pid]
                                    val = row[pid]
                                    
                                    # Check range
                                    try:
                                        # Ensure val is numeric
                                        val_float = float(val)
                                        
                                        # Get limits for sex
                                        sex_limits = limits.get(sex)
                                        if sex_limits:
                                            min_val, max_val = sex_limits
                                            
                                            is_out = False
                                            if min_val is not None and val_float < min_val:
                                                is_out = True
                                            if max_val is not None and val_float > max_val:
                                                is_out = True
                                                
                                            if is_out:
                                                current_style_data_conditional.append({
                                                    'if': {
                                                        'row_index': i,
                                                        'column_id': pid
                                                    },
                                                    'backgroundColor': '#8B0000', # Dark Red
                                                    'color': 'white'
                                                })
                                    except:
                                        # Not numeric or other error
                                        pass
                
                elif group_name == 'ICD Codes':
                    tooltip_data = []
                    for row in df_group.to_dict('records'):
                        attr = row['Attribute'] # This is the ICD code (e.g. C34.1)
                        
                        # Check if attribute is in dictionary
                        # Note: The attribute might have extra chars or be clean. 
                        # Assuming it matches the 'Code' in dictionary.
                        desc = icd_dict.get(attr)
                        
                        if desc:
                             row_tooltip = {col: {'value': f"**{attr}**: {desc}", 'type': 'markdown'} for col in columns}
                        else:
                             row_tooltip = {col: None for col in columns}
                        tooltip_data.append(row_tooltip)

                table_component = html.Div(dash_table.DataTable(
                    data=df_group.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in columns],
                    style_table={'overflowX': 'auto'},
                    # Removed style_header to show headers
                    style_cell={'textAlign': 'left', 'minWidth': '150px', 'width': '150px', 'maxWidth': '150px', 'whiteSpace': 'normal', 'hyphens': 'auto', 'wordBreak': 'break-word'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'Attribute'},
                         'width': '200px', 'minWidth': '200px', 'maxWidth': '200px'}
                    ],
                    style_data_conditional=current_style_data_conditional,
                    page_size=20 if group_name != 'ICD Codes' else 10,
                    tooltip_data=tooltip_data,
                    tooltip_duration=None,
                    # Removed css that forced hidden header
                ), style={'padding': '0px', 'border': '1px solid #ddd', 'borderTop': 'none'}, className='sync-table') # Added class for sync scroll
                
                details = html.Details([
                    html.Summary(html.B(f"{group_name} ({len(df_group)} attributes)"), style={'cursor': 'pointer', 'fontSize': '16px', 'padding': '10px', 'backgroundColor': '#f0f0f0', 'color': 'black', 'position': 'relative', 'zIndex': '10'}),
                    html.Div(table_component, style={'padding': '10px', 'border': '1px solid #ddd', 'borderTop': 'none'})
                ], open=True, style={'marginBottom': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px'})
                
                children.append(details)
                
    # 3. Other Attributes
    all_grouped_features = sum(feature_groups.values(), []) + summary_features
    df_other = df_transposed[~df_transposed['Attribute'].isin(all_grouped_features)]
    
    if not df_other.empty:
        children.append(html.Details([
            html.Summary(html.B(f"Other Attributes ({len(df_other)})"), style={'cursor': 'pointer', 'fontSize': '16px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),
            html.Div(dash_table.DataTable(
                data=df_other.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '150px', 'width': '150px', 'maxWidth': '150px', 'whiteSpace': 'normal'},
                style_cell_conditional=[
                    {'if': {'column_id': 'Attribute'},
                     'width': '200px', 'minWidth': '200px', 'maxWidth': '200px'}
                ],
                style_data_conditional=style_data_conditional
            ), style={'padding': '10px'})
        ], style={'marginBottom': '10px', 'border': '1px solid #ccc'}))
        
    return children

# Callback for Similarity Search AND Click Interaction
@app.callback(
    [Output('similarity-table-container', 'children'),
     Output('similar-patients-store', 'data'),
     Output('side-panel-title', 'children')],
    [Input('find-similar-btn', 'n_clicks'),
     Input('umap-graph', 'clickData'),
     Input('patient-dropdown', 'value')],
    [State('new-patients-store', 'data')]
)
def update_side_panel(n_clicks, clickData, highlight_patient, new_patients_data):
    ctx = callback_context
    if not ctx.triggered:
        return "", [], ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Reconstruct full dataframe
    if new_patients_data:
        df_new_pts = pd.DataFrame(new_patients_data)
        df_full = pd.concat([df_combined, df_new_pts], ignore_index=True)
    else:
        df_full = df_combined

    # Handle Dropdown Change
    if trigger_id == 'patient-dropdown':
        if not highlight_patient or highlight_patient == 'None':
             return "", [], ""
        
        # Show details for the highlighted patient
        children = generate_attribute_table(df_full, highlight_patient, [])
        return children, [], "Patient Details"

    # Handle "Find Similar" Button
    if trigger_id == 'find-similar-btn':
        if n_clicks == 0 or highlight_patient == 'None':
            return "", [], ""
            
        target_row = df_full[df_full['patient_id'] == highlight_patient]
        if target_row.empty:
            return "Selected patient not found.", [], ""
        
        # ... (Similarity Calculation Logic) ...
        # Prepare data for similarity calculation
        X_raw = df_full.copy()
        for col in feature_order:
            if col not in X_raw.columns:
                X_raw[col] = np.nan
        X_raw = X_raw[feature_order]
        
        try:
            X_encoded = preprocessor.transform(X_raw)
        except Exception as e:
            return f"Error in preprocessing for similarity: {e}", [], ""
        
        target_idx = df_full[df_full['patient_id'] == highlight_patient].index[0]
        target_vector = X_encoded[target_idx].reshape(1, -1)
        
        sim_scores = cosine_similarity(target_vector, X_encoded)[0]
        #sim_scores = euclidean_distances(target_vector, X_encoded)[0]
        
        sorted_indices = sim_scores.argsort()[::-1]
        top_indices = [i for i in sorted_indices if i != target_idx][:5]
        
        # Add similarity score to df_full temporarily for display
        df_full['Similarity'] = sim_scores
        df_full['Similarity'] = df_full['Similarity'].apply(lambda x: f"{x:.4f}")
        
        comparison_patients = df_full.iloc[top_indices]['patient_id'].tolist()
        
        children = generate_attribute_table(df_full, highlight_patient, comparison_patients)
        return children, comparison_patients, "Top 5 Similar Patients"

    # Handle Graph Click
    elif trigger_id == 'umap-graph':
        if not clickData:
            return "", [], "Patient Details"
            
        try:
            pt_id = clickData['points'][0]['customdata'][0]
        except:
            return "Could not retrieve patient ID.", [], "Patient Details"
            
        children = generate_attribute_table(df_full, pt_id, [])
        return children, [], "Patient Details"
        
    return "", [], ""

@app.callback(
    Output('patient-dropdown', 'value'),
    [Input('umap-graph', 'clickData')],
    prevent_initial_call=True
)
def update_dropdown_on_click(clickData):
    if clickData:
        # Extract patient_id from clickData
        # The structure depends on how the plot was created.
        # With px.scatter and hover_data=['patient_id'], it's usually in customdata
        try:
            point = clickData['points'][0]
            if 'customdata' in point:
                # customdata is a list, usually [patient_id] if hover_data=['patient_id']
                patient_id = point['customdata'][0]
                return patient_id
        except Exception as e:
            print(f"Error extracting patient_id from clickData: {e}")
            return dash.no_update
    return dash.no_update

if __name__ == '__main__':
    print("Starting Dash app...")
    app.run(debug=False, port=8044)
