"""
Dash callbacks for Multimodal Patient Visualization.

This module contains all Dash callbacks that handle user interactions,
data loading, similarity searches, and graph updates.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
from dash import Output, Input, State, callback_context, no_update

from .patient_encoding import load_and_encode_patient
from .utils import generate_attribute_table, decode_features

# Load models and metadata
preprocessor = joblib.load('models/preprocessor.pkl')
umap_model = joblib.load('models/umap_model.pkl')
with open('models/feature_order.json') as f:
    feature_order = json.load(f)


def register_callbacks(app, df_combined):
    """
    Register all Dash callbacks for the application.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance
    df_combined : pd.DataFrame
        Combined training data with UMAP embeddings
    """

    @app.callback(
        [Output('new-patients-store', 'data'),
         Output('status-msg', 'children'),
         Output('patient-dropdown', 'value', allow_duplicate=True)],
        [Input('load-btn', 'n_clicks'), Input('path-input', 'n_submit')],
        [State('path-input', 'value'),
         State('new-patients-store', 'data')],
        prevent_initial_call=True
    )
    def load_patient(n_clicks, n_submit, path, current_data):
        """Load and encode a new patient from the provided path."""
        # Trigger if button clicked or Enter pressed
        if (not n_clicks and not n_submit) or not path:
            return current_data, "", no_update

        try:
            p = Path(path)
            pt_id = "Unknown"
            for part in p.parts:
                match = re.search(r"(\d+)", part)
                if match:
                    pt_id = match.group(1)
                    break

            print(f"Loading and encoding patient from {path} with ID {pt_id}...")
            df_new_pt = load_and_encode_patient(path, pt_id)
            new_records = df_new_pt.to_dict('records')
            updated_data = current_data + new_records

            message = f"Successfully loaded patient {df_new_pt['patient_id'].iloc[0]}"
            return updated_data, message, df_new_pt['patient_id'].iloc[0]
        except Exception as e:
            return current_data, f"Error loading patient: {str(e)}", no_update

    @app.callback(
        [Output('umap-graph', 'figure'),
         Output('patient-dropdown', 'options')],
        [Input('hue-dropdown', 'value'),
         Input('patient-dropdown', 'value'),
         Input('new-patients-store', 'data'),
         Input('similar-patients-store', 'data')]
    )
    def update_graph(hue, highlight_patient, new_patients_data, similar_patients_ids):
        """Update UMAP graph based on user selections."""
        import plotly.graph_objects as go
        import plotly.express as px

        if new_patients_data:
            df_new_pts = pd.DataFrame(new_patients_data)
            df_plot = pd.concat([df_combined, df_new_pts], ignore_index=True)
        else:
            df_plot = df_combined.copy()

        df_plot = decode_features(df_plot)
        patient_options = [
            {'label': p, 'value': p}
            for p in ['None'] + sorted(df_plot['patient_id'].unique())
        ]

        fig = px.scatter(
            df_plot,
            x='UMAP 1',
            y='UMAP 2',
            color=hue,
            hover_data=['patient_id'],
            color_discrete_sequence=px.colors.qualitative.Bold
            if hue == 'dataset' else px.colors.qualitative.Safe,
        )

        if similar_patients_ids:
            sim_data = df_plot[df_plot['patient_id'].isin(similar_patients_ids)]
            if not sim_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sim_data['UMAP 1'],
                        y=sim_data['UMAP 2'],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='orange',
                            symbol='diamond',
                            line=dict(width=0)
                        ),
                        name="Similar Patients",
                        hoverinfo='skip'
                    )
                )

        if highlight_patient != 'None' and highlight_patient in df_plot['patient_id'].values:
            pt_data = df_plot[df_plot['patient_id'] == highlight_patient]
            if not pt_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pt_data['UMAP 1'],
                        y=pt_data['UMAP 2'],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='#04316A',
                            symbol='circle',
                            line=dict(width=0)
                        ),
                        name=f"Selected: {highlight_patient}",
                        hoverinfo='skip'
                    )
                )

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
                showticklabels=False,
                range=[x_min - padding_x, x_max + padding_x],
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=False
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
                constrain='domain',
                showticklabels=False,
                range=[y_min - padding_y, y_max + padding_y],
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=False
            ),
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
                y=-0.2,
                xanchor="center",
                x=0.5,
                title=dict(side="top")
            )
        )
        return fig, patient_options

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
        """Update side panel with patient details or similar patients."""
        ctx = callback_context
        if not ctx.triggered:
            return "", [], ""

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if new_patients_data:
            df_new_pts = pd.DataFrame(new_patients_data)
            df_full = pd.concat([df_combined, df_new_pts], ignore_index=True)
        else:
            df_full = df_combined

        if trigger_id == 'patient-dropdown':
            if not highlight_patient or highlight_patient == 'None':
                return "", [], ""
            children = generate_attribute_table(df_full, highlight_patient, [])
            return children, [], "Patient Details"

        if trigger_id == 'find-similar-btn':
            if n_clicks == 0 or highlight_patient == 'None':
                return "", [], ""

            target_row = df_full[df_full['patient_id'] == highlight_patient]
            if target_row.empty:
                return "Selected patient not found.", [], ""

            X_raw = df_full.copy()
            for col in feature_order:
                if col not in X_raw.columns:
                    X_raw[col] = np.nan
            X_raw = X_raw[feature_order]

            try:
                X_encoded = preprocessor.transform(X_raw)
            except Exception as e:
                msg = f"Error in preprocessing for similarity: {e}"
                return msg, [], ""

            target_idx = df_full[df_full['patient_id'] == highlight_patient].index[0]
            target_vector = X_encoded[target_idx].reshape(1, -1)
            sim_scores = cosine_similarity(target_vector, X_encoded)[0]

            sorted_indices = sim_scores.argsort()[::-1]
            top_indices = [i for i in sorted_indices if i != target_idx][:5]

            df_full['Similarity'] = sim_scores
            df_full['Similarity'] = df_full['Similarity'].apply(
                lambda x: f"{x:.4f}"
            )

            comparison_patients = df_full.iloc[top_indices]['patient_id'].tolist()
            children = generate_attribute_table(df_full, highlight_patient,
                                               comparison_patients)
            return children, comparison_patients, "Top 5 Similar Patients"

        elif trigger_id == 'umap-graph':
            if not clickData:
                return "", [], "Patient Details"

            try:
                pt_id = clickData['points'][0]['customdata'][0]
            except Exception:
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
        """Update dropdown when a point is clicked on the graph."""
        if clickData:
            try:
                point = clickData['points'][0]
                if 'customdata' in point:
                    patient_id = point['customdata'][0]
                    return patient_id
            except Exception as e:
                print(f"Error extracting patient_id from clickData: {e}")
                return no_update
        return no_update
