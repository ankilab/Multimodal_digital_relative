import os
import subprocess
import shutil
from pathlib import Path
import argparse
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# from feature_extraction.extract_tma_image_features import extract_image_features # Removed direct import

def run_qupath_pipeline(image_path, map_path, project_dir, output_measurements_path, qupath_executable="/Applications/QuPath-0.6.0-arm64.app/Contents/MacOS/QuPath-0.6.0-arm64"):
    """
    Runs the QuPath pipeline to process the TMA image.
    
    Args:
        image_path (str): Path to the TMA image.
        map_path (str): Path to the TMA map CSV.
        project_dir (str): Path to the QuPath project directory.
        output_measurements_path (str): Path to save the measurements CSV.
        qupath_executable (str): Path to the QuPath executable.
        
    Returns:
        str: Path to the directory containing exported tiles.
    """
    image_path = Path(image_path)
    map_path = Path(map_path)
    project_dir = Path(project_dir)
    
    # Ensure directories exist
    project_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = project_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    
    print(f"Processing image: {image_path}")
    print(f"Using TMA map: {map_path}")
    print(f"Output directory: {project_dir}")

    # 1. Import TMA Map AND Export Tiles (Combined Script)
    # We use a combined script to ensure the TMA grid persists in memory for the export step.
    
    combined_script = Path("qupath_scripts/process_tma_automated.groovy")
    
    if not combined_script.exists():
        raise FileNotFoundError("QuPath automated script not found: qupath_scripts/process_tma_automated.groovy")

    # Command to run combined script
    cmd_combined = [
        qupath_executable,
        "script",
        str(combined_script),
        "--image", str(image_path),
        "--args", f"tma_map_path={map_path}",
        "--args", f"output_dir={tiles_dir}",
        "--args", f"measurements_path={output_measurements_path}"
    ]
    
    print("Running QuPath TMA Pipeline (Import + Export)...")
    try:
        print("test")
        subprocess.run(cmd_combined, check=True, text=True) # Removed capture_output=True to see real-time output
        print("QuPath Pipeline Completed.")
    except subprocess.CalledProcessError:
        print("Error running QuPath pipeline. See output above for details.")
        raise

    return tiles_dir

def process_patient(image_path, map_path, patient_id, output_dir):
    """
    Full pipeline: QuPath -> Feature Extraction
    """
    # 1. QuPath Pipeline
    # Create a temporary project directory for this patient
    temp_project_dir = Path(output_dir) / f"temp_project_{patient_id}"
    
    try:
        # 1. QuPath Pipeline (Dearray, Import Map, Export Tiles, Cell Detection, Export Measurements)
        print("Running QuPath pipeline...")
        
        # Define path for measurements CSV
        measurements_out_dir = Path(output_dir) / "measurements"
        measurements_out_dir.mkdir(exist_ok=True)
        measurements_csv_path = measurements_out_dir / "tma_measurements.csv"
        
        tiles_dir = run_qupath_pipeline(image_path, map_path, temp_project_dir, measurements_csv_path)
        
        print(f"TMA measurements saved to {measurements_csv_path}")
        
        # 2. Feature Extraction
        print("Extracting features from tiles (using deeptexture_env)...")
        
        # Output file path
        features_out_dir = Path(output_dir) / "features"
        features_out_dir.mkdir(exist_ok=True)
        save_path = features_out_dir / "tma_tile_dtr_256_HE.csv"
        
        # Path to the extraction script
        extraction_script = Path("feature_extraction/extract_tma_image_features.py")
        
        # Command to run in conda env
        # conda run -n deeptexture_env python feature_extraction/extract_tma_image_features.py --tiles_dir ... --output_path ...
        
        cmd_extract = [
            "conda", "run", "-n", "deeptexture_env", "--no-capture-output",
            "python", str(extraction_script),
            "--tiles_dir", str(tiles_dir),
            "--output_path", str(save_path),
            "--dim", "256",
            "--backbone", "vgg",
            "--layer", "block3_conv3"
        ]
        
        try:
            subprocess.run(cmd_extract, check=True, text=True)
            print(f"Features saved to {save_path}")
            return save_path
        except subprocess.CalledProcessError as e:
            print(f"Error running feature extraction: {e}")
            raise
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return None
    finally:
        # Cleanup temp dir if needed
        # shutil.rmtree(temp_project_dir)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to TMA image")
    parser.add_argument("--map", required=True, help="Path to TMA map CSV")
    parser.add_argument("--id", required=True, help="Patient ID")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    process_patient(args.image, args.map, args.id, args.output)
