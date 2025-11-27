import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input, State, callback_context
from pathlib import Path
import sys
import os

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import our encoding functions
from feature_extraction.extract_text_features import get_icd_vectors
from feature_extraction.extract_tabular_features import get_tabular_features, get_blood_features, get_target_classes
from feature_extraction.extract_tma_features import get_tma_features

from feature_extraction.encode_new_patient import preprocess_patient_data_from_dict
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

    # encode data 
    # Load encoded data
    clinical = pd.read_csv(dest_dir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(dest_dir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(dest_dir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(dest_dir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density= pd.read_csv(dest_dir/"tma_cell_density.csv", dtype={"patient_id": str})

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    # feature order 
    # with open(results_dir / "feature_order.json", "r") as f:
    #     feature_order = json.load(f)

    # df = df[feature_order]

    # project into UMAP space
    # Preprocess embeddings
    results_dir = "./"
    preprocessor = joblib.load(results_dir + "preprocessor.pkl")
    embeddings = preprocessor.transform(df.drop("patient_id", axis=1))

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
df_new_001 = load_and_encode_patient(f"test_patient_002/raw", "002")

# Concatenate
df_combined = pd.concat([df_train, df_new_001], ignore_index=True)

# -----------------------------------------------------------------------------
# 3. Dash App
# -----------------------------------------------------------------------------
app = Dash(__name__)

# Features to show in dropdown
hue_options = ['dataset'] + [col for col in df_combined.columns if col not in ['patient_id', 'UMAP 1', 'UMAP 2', 'dataset']]

app.layout = html.Div([
    html.H1("Multimodal UMAP Visualization"),
    
    # Data Loading Section
    html.Div([
        html.Label("Load New Patient Data (Path to 'raw' folder):"),
        html.Br(),
        dcc.Input(
            id='path-input',
            type='text',
            placeholder='/path/to/patient/raw',
            style={'width': '400px', 'marginRight': '10px'}
        ),
        html.Button('Load Patient', id='load-btn', n_clicks=0),
        html.Div(id='status-msg', style={'marginTop': '10px', 'color': 'blue'}),
    ], style={'padding': '20px', 'borderBottom': '1px solid #ccc', 'marginBottom': '20px'}),

    html.Div([
        html.Div([
            html.Label("Color by:"),
            dcc.Dropdown(
                id='hue-dropdown',
                options=[{'label': c, 'value': c} for c in hue_options],
                value='dataset',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Highlight Patient:"),
            dcc.Dropdown(
                id='patient-dropdown',
                # Options will be updated via callback
                options=[{'label': p, 'value': p} for p in ['None'] + sorted(df_combined['patient_id'].unique())],
                value="New_001",
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),
    ], style={'padding': '20px'}),
    
    html.Div([
        dcc.Graph(id='umap-graph', style={'height': '70vh', 'width': '70%', 'display': 'inline-block'}),
        html.Div([
            html.H3("Patient Details"),
            html.Div(id='click-data')
        ], style={'height': '70vh', 'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'overflowY': 'scroll', 'border': '1px solid #ccc'})
    ]),
    
    # Store for new patients
    dcc.Store(id='new-patients-store', data=[])  # List of records
], style={'fontFamily': 'Arial, sans-serif'})

# Callback to load new patient
@app.callback(
    [Output('new-patients-store', 'data'),
     Output('status-msg', 'children')],
    [Input('load-btn', 'n_clicks')],
    [State('path-input', 'value'),
     State('new-patients-store', 'data')]
)
def load_patient(n_clicks, path, current_data):
    if n_clicks == 0 or not path:
        return current_data, ""
    
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
        
        return updated_data, f"Successfully loaded patient {df_new_pt['patient_id'].iloc[0]}"
    except Exception as e:
        return current_data, f"Error loading patient: {str(e)}"

# Callback to update graph and dropdowns
@app.callback(
    [Output('umap-graph', 'figure'),
     Output('patient-dropdown', 'options')],
    [Input('hue-dropdown', 'value'),
     Input('patient-dropdown', 'value'),
     Input('new-patients-store', 'data')]
)
def update_graph(hue, highlight_patient, new_patients_data):
    # Combine initial data with new patients
    if new_patients_data:
        df_new_pts = pd.DataFrame(new_patients_data)
        # Ensure columns match for concatenation (fill missing with NaN)
        df_plot = pd.concat([df_combined, df_new_pts], ignore_index=True)
    else:
        df_plot = df_combined.copy()

    # Update dropdown options
    patient_options = [{'label': p, 'value': p} for p in ['None'] + sorted(df_plot['patient_id'].unique())]

    # Create scatter plot
    fig = px.scatter(
        df_plot, 
        x='UMAP 1', 
        y='UMAP 2', 
        color=hue,
        hover_data=['patient_id'],
        title=f"UMAP Colored by {hue}",
        color_discrete_sequence=px.colors.qualitative.Bold if hue == 'dataset' else px.colors.qualitative.Safe,
        width=700,
        height=700,
        # Increase right margin to accommodate legend without shrinking plot
        #margin=dict(l=50, r=400, t=50, b=50),
        #autosize=False
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
                    marker=dict(size=15, color='red', symbol='circle', line=dict(width=2, color='black')),
                    name=f"Selected: {highlight_patient}",
                    hoverinfo='skip'
                )
            )
            
    fig.update_layout(
        clickmode='event+select',
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(scaleanchor="y", scaleratio=1, fixedrange=True, constrain='domain'),
        yaxis=dict(scaleanchor="x", scaleratio=1, fixedrange=True, constrain='domain'),
        width=1000,
        height=700,
        # Increase right margin to accommodate legend without shrinking plot
        margin=dict(l=50, r=400, t=50, b=50),
        autosize=False,
        legend=dict(
            x=1.02,
            y=1.1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    return fig, patient_options

@app.callback(
    Output('click-data', 'children'),
    [Input('umap-graph', 'clickData'),
     Input('new-patients-store', 'data')]
)
def display_click_data(clickData, new_patients_data):
    if not clickData:
        return "Click a point to see details."
    
    # Reconstruct full dataframe to get details
    if new_patients_data:
        df_new_pts = pd.DataFrame(new_patients_data)
        df_full = pd.concat([df_combined, df_new_pts], ignore_index=True)
    else:
        df_full = df_combined
    
    # Get patient_id from hover data (customdata)
    try:
        pt_id = clickData['points'][0]['customdata'][0]
    except:
        return "Could not retrieve patient ID."
        
    row = df_full[df_full['patient_id'] == pt_id].iloc[0]
    
    # Format details
    details = []
    details.append(html.H4(f"ID: {pt_id}"))
    
    # Key features to show first
    key_features = ['dataset', 'sex','smoking_status', 'age_at_initial_diagnosis']
    
    for feat in key_features:
        if feat in row:
            details.append(html.P([html.B(f"{feat}: "), str(row[feat])]))
            
    details.append(html.Hr())
    details.append(html.H5("All Features:"))
    
    # List all other features
    for col in sorted(row.index):
        if col not in key_features and col not in ['patient_id', 'UMAP 1', 'UMAP 2']:
            val = row[col]
            # Skip 0 values for sparse features (like ICD codes) to reduce clutter
            if isinstance(val, (int, float)) and val == 0 and (col.startswith('c') or col.startswith('d')):
                continue
            details.append(html.Div([html.B(f"{col}: "), str(val)], style={'fontSize': '12px'}))
            
    return details

if __name__ == '__main__':
    print("Starting Dash app...")
    app.run(debug=True, port=8040)
