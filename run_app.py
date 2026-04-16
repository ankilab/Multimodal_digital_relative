"""
Refactored Entry Point for HANCOCK Multimodal Patient Visualization Dashboard.

This script replaces visualize_new_patient.py with a modular structure.

Usage:
    python -m app.main

    or

    python app/main.py

The application will start on port 8044 at http://localhost:8044
"""

from dash import Dash

from app.layout import create_app_layout, df_train
from app.callbacks import register_callbacks


def main():
    """Initialize and run the Dash application."""
    app = Dash(__name__)
    app.layout = create_app_layout(df_train)
    register_callbacks(app, df_train)

    print("Starting HANCOCK Dashboard...")
    print("Open your browser to http://localhost:8044")
    app.run(debug=False, port=8044)


if __name__ == "__main__":
    main()
