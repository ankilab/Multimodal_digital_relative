import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from dash import Dash, dcc, html, Output, Input, State, callback_context, dash_table
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
    print(df.drop("patient_id", axis=1))
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
        
        html.Div([
            html.Button('Find Similar Patients', id='find-similar-btn', n_clicks=0),
        ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '20px', 'verticalAlign': 'top'}),

    ], style={'padding': '20px'}),
    
    html.Div([
        dcc.Graph(id='umap-graph', style={'height': '70vh', 'width': '50%', 'display': 'inline-block'}),
        html.Div([
            html.H3("Top 5 Similar Patients"),
            html.Div(id='similarity-table-container')
        ], style={'height': '70vh', 'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'overflowY': 'scroll', 'border': '1px solid #ccc', 'marginLeft': '20px'})
    ]),
    
    # Store for new patients
    dcc.Store(id='new-patients-store', data=[]),  # List of records
    # Store for similar patients IDs
    dcc.Store(id='similar-patients-store', data=[])
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
    # Highlight similar patients
    if similar_patients_ids:
        sim_data = df_plot[df_plot['patient_id'].isin(similar_patients_ids)]
        if not sim_data.empty:
             fig.add_trace(
                go.Scatter(
                    x=sim_data['UMAP 1'],
                    y=sim_data['UMAP 2'],
                    mode='markers',
                    marker=dict(size=12, color='orange', symbol='diamond', line=dict(width=1, color='black')),
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
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)"
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
        'icd_codes.csv': 'ICD Codes'
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

# Helper to generate attribute table
def generate_attribute_table(df_full, target_patient, comparison_patients=[]):
    # Prepare data
    patient_ids = [target_patient] + comparison_patients
    
    # Filter for these patients
    df_subset = df_full[df_full['patient_id'].isin(patient_ids)].copy()
    
    # Ensure order matches input list
    df_subset = df_subset.set_index('patient_id')
    df_subset = df_subset.reindex(patient_ids)
    
    # Transpose
    df_transposed = df_subset.T.reset_index().rename(columns={'index': 'Attribute'})
    
    # Define columns
    columns = df_transposed.columns.tolist()
    target_col = target_patient
    
    # Define style
    style_data_conditional = []
    
    # Highlight Target Column
    style_data_conditional.append({
        'if': {'column_id': target_col},
        'backgroundColor': '#e6f2ff', # Light blue
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
        children.append(dash_table.DataTable(
            data=df_summary.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in columns],
            style_table={'overflowX': 'auto', 'marginBottom': '20px'},
            style_cell={'textAlign': 'left', 'minWidth': '150px', 'width': '150px', 'maxWidth': '150px', 'whiteSpace': 'normal'},
            style_cell_conditional=[
                {'if': {'column_id': 'Attribute'},
                 'width': '200px', 'minWidth': '200px', 'maxWidth': '200px'}
            ],
            style_data_conditional=style_data_conditional
        ))
        
    # 2. Grouped Sections
    group_order = ['Clinical Data', 'Pathological Data', 'Blood Data', 'TMA Data', 'ICD Codes']
    
    for group_name in group_order:
        if group_name in feature_groups:
            features = feature_groups[group_name]
            df_group = df_transposed[df_transposed['Attribute'].isin(features)]
            
            if not df_group.empty:
                table = dash_table.DataTable(
                    data=df_group.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'minWidth': '150px', 'width': '150px', 'maxWidth': '150px', 'whiteSpace': 'normal'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'Attribute'},
                         'width': '200px', 'minWidth': '200px', 'maxWidth': '200px'}
                    ],
                    style_data_conditional=style_data_conditional,
                    page_size=20 if group_name != 'ICD Codes' else 10
                )
                
                details = html.Details([
                    html.Summary(html.B(f"{group_name} ({len(df_group)} attributes)"), style={'cursor': 'pointer', 'fontSize': '16px', 'padding': '10px', 'backgroundColor': '#f0f0f0', 'color': 'black', 'position': 'relative', 'zIndex': '10'}),
                    html.Div(table, style={'padding': '10px', 'border': '1px solid #ddd', 'borderTop': 'none'})
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
     Output('similar-patients-store', 'data')],
    [Input('find-similar-btn', 'n_clicks'),
     Input('umap-graph', 'clickData')],
    [State('patient-dropdown', 'value'),
     State('new-patients-store', 'data')]
)
def update_side_panel(n_clicks, clickData, highlight_patient, new_patients_data):
    ctx = callback_context
    if not ctx.triggered:
        return "", []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Reconstruct full dataframe
    if new_patients_data:
        df_new_pts = pd.DataFrame(new_patients_data)
        df_full = pd.concat([df_combined, df_new_pts], ignore_index=True)
    else:
        df_full = df_combined.copy()
        
    # Handle "Find Similar" Button
    if trigger_id == 'find-similar-btn':
        if n_clicks == 0 or highlight_patient == 'None':
            return "", []
            
        target_row = df_full[df_full['patient_id'] == highlight_patient]
        if target_row.empty:
            return "Selected patient not found.", []
        
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
            return f"Error in preprocessing for similarity: {e}", []
        
        target_idx = df_full[df_full['patient_id'] == highlight_patient].index[0]
        target_vector = X_encoded[target_idx].reshape(1, -1)
        
        sim_scores = cosine_similarity(target_vector, X_encoded)[0]
        
        sorted_indices = sim_scores.argsort()[::-1]
        top_indices = [i for i in sorted_indices if i != target_idx][:5]
        
        # Add similarity score to df_full temporarily for display
        df_full['Similarity'] = sim_scores
        df_full['Similarity'] = df_full['Similarity'].apply(lambda x: f"{x:.4f}")
        
        comparison_patients = df_full.iloc[top_indices]['patient_id'].tolist()
        
        children = generate_attribute_table(df_full, highlight_patient, comparison_patients)
        return children, comparison_patients

    # Handle Graph Click
    elif trigger_id == 'umap-graph':
        if not clickData:
            return "", []
            
        try:
            pt_id = clickData['points'][0]['customdata'][0]
        except:
            return "Could not retrieve patient ID.", []
            
        children = generate_attribute_table(df_full, pt_id, [])
        return children, []
        
    return "", []

if __name__ == '__main__':
    print("Starting Dash app...")
    app.run(debug=True, port=8044)
