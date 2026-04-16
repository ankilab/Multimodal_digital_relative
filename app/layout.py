"""
Dash layout for Multimodal Patient Visualization.

This module contains the layout definition for the HANCOCK dashboard,
including the control panel, graph, and side panel components.
"""

from dash import dcc, html
from pathlib import Path

from data_exploration.umap_embedding import get_umap_embedding

# Load training data and UMAP embeddings
print("Loading training data and UMAP embeddings...")
df_train = get_umap_embedding("./features", umap_min_dist=0.1, umap_n_neighbors=15)
df_train['dataset'] = 'Training'


def create_app_layout(df_combined=None):
    """
    Create the main layout for the Dash application.

    Parameters
    ----------
    df_combined : pd.DataFrame, optional
        Combined training data with UMAP embeddings.
        If None, uses df_train loaded above.

    Returns
    -------
    dash.html.Div
        The main layout component
    """
    if df_combined is None:
        df_combined = df_train

    hue_options = [
        'dataset'
    ] + [col for col in df_combined.columns
         if col not in ['patient_id', 'UMAP 1', 'UMAP 2', 'dataset']]

    return html.Div([
        html.H1(
            "Multimodal Digital Relative",
            style={'textAlign': 'center'}
        ),
        html.H3(
            "UMAP Visualization Dashboard for HANCOCK",
            style={'textAlign': 'center'}
        ),

        # Control Panel Section
        html.Div([
            # Color By Dropdown
            html.Div([
                html.Label("Color by:"),
                dcc.Dropdown(
                    id='hue-dropdown',
                    options=[{'label': c, 'value': c} for c in hue_options],
                    value='dataset',
                    clearable=False
                )
            ], className='control-item', style={'flex': '1', 'marginBottom': '20px'}),

            # Highlight Patient Dropdown
            html.Div([
                html.Label("Highlight Patient:"),
                dcc.Dropdown(
                    id='patient-dropdown',
                    options=[{'label': p, 'value': p}
                             for p in ['None'] + sorted(
                                 df_combined['patient_id'].unique())],
                    value="None",
                    clearable=False
                )
            ], className='control-item', style={'flex': '1', 'marginBottom': '20px'}),

            # Load Patient Input and Button
            html.Div([
                html.Label("Load New Patient Data (Path):"),
                html.Div([
                    dcc.Input(
                        id='path-input',
                        type='text',
                        placeholder='/path/to/patient/raw',
                        className='custom-input',
                        n_submit=0,
                        style={'flex': '1', 'marginRight': '10px'}
                    ),
                    html.Button(
                        'Load Patient',
                        id='load-btn',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#9b59b6',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'fontSize': '16px',
                            'fontWeight': 'bold',
                            'flex': '0 0 auto'
                        }
                    ),
                ], style={'display': 'flex', 'width': '100%', 'gap': '10px'})
            ], className='control-item', style={'flex': '2', 'marginBottom': '20px'}),

            # Find Similar Button
            html.Div([
                html.Button(
                    'Find Similar Patients',
                    id='find-similar-btn',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#9b59b6',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontSize': '16px',
                        'fontWeight': 'bold',
                        'width': '100%'
                    }
                ),
            ], className='control-item',
            style={'minWidth': 'auto', 'justifyContent': 'flex-end', 'marginTop': '20px'}),

            # Status Message
            html.Div(
                id='status-msg',
                className='status-msg',
                style={'color': 'blue'}
            ),

        ], className='control-panel',
        style={'zIndex': '3000', 'position': 'relative'}),

        # Main Content Area (Graph + Right Panel)
        html.Div([
            # UMAP Graph
            html.Div([
                dcc.Graph(
                    id='umap-graph',
                    style={'height': '80vh', 'width': '100%'}
                ),
            ], style={
                'width': '50%',
                'height': '80vh',
                'zIndex': '1',
                'position': 'relative'
            }),

            # Right Side Panel
            html.Div([
                html.H3("Top 5 Similar Patients", id='side-panel-title'),
                html.Div(id='similarity-table-container')
            ], style={
                'width': '50%',
                'height': '80vh',
                'overflowY': 'auto',
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'zIndex': 2000,
                'position': 'relative'
            })
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'height': '80vh'
        }),

        # Data Stores
        dcc.Store(id='new-patients-store', data=[]),
        dcc.Store(id='similar-patients-store', data=[])
    ], style={'fontFamily': 'Arial, sans-serif'})
