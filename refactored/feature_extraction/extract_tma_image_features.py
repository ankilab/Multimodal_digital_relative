from argparse import ArgumentParser
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
try:
    import deeptexture
except ImportError:
    deeptexture = None
    print("Warning: deeptexture module not found. Image feature extraction will fail.")


def extract_image_features(tile_dir, backbone, layer, dim, num_tiles_per_patient=2):
    """
    Get representations from tiles using the deeptexture package
    Reference: Komura, D. et al. "Universal encoding of pan-cancer histology by deep texture representations."
    Cell Reports 38, 110424,2022. https://doi.org/10.1016/j.celrep.2022.110424
    
    Parameters
    ----------
    tile_dir : str
        Path to directory containing tiles 
    backbone : str
        Backbone, for example "vgg"
    layer : str
        Name of the layer from which features are extracted
    dim : int
        Feature dimension
    num_tiles_per_patient : 
        Number of tiles to consider per patient
    Returns
    -------
    Dictionary that can be saved as compressed numpy file, with patient IDs as keys
    """
    # Get tile paths and patient IDs
    tiles = os.listdir(tile_dir)
    tile_paths = [tile_dir/t for t in tiles]
    patient_ids = [re.search(r"[0-9]{3}", t).group() for t in tiles]
    print(f"Found {len(tile_paths)} tiles from {len(np.unique(patient_ids))} cases.")

    # Create DTR object
    dtr_object = deeptexture.DTR(arch=backbone, layer=layer, dim=dim)

    # Get Deep Texture Representations (DTRs)
    # (Same as get_dtr_multifiles() but with progress bar)
    print("Extracting DTRs...")
    dtrs = np.vstack([dtr_object.get_dtr(str(t)) for t in tqdm(tile_paths)])

    # Concatenate DTRs of each patient
    savez_dict_concat = {}
    for pid in patient_ids:
        savez_dict_concat[pid] = []
    for i in range(len(patient_ids)):
        savez_dict_concat[patient_ids[i]].append(dtrs[i])

    # Convert concatenated DTRs to dictionary for compressed numpy file
    target_shape = num_tiles_per_patient * dim
    for key, val in savez_dict_concat.items():
        if len(val) == 1:
            patient_dtrs = val[0]
        else:
            patient_dtrs = np.concatenate(val)
        if patient_dtrs.shape[0] > target_shape:
            savez_dict_concat[key] = patient_dtrs[:target_shape]
        else:
            padding = target_shape - patient_dtrs.shape[0]
            padding_left = padding // 2
            padding_right = padding - padding_left
            savez_dict_concat[key] = np.pad(patient_dtrs, (padding_left, padding_right), mode="constant")

    return savez_dict_concat


if __name__ == "__main__":
    parser = ArgumentParser()
    # Batch mode arguments (optional)
    parser.add_argument("--root_dir", type=str, help="Path to directory containing QuPath projects (Batch Mode)")
    parser.add_argument("--dest_dir", type=str, help="Path to directory where the features are saved (Batch Mode)")
    
    # Single patient mode arguments
    parser.add_argument("--tiles_dir", type=str, help="Path to tiles directory (Single Patient Mode)")
    parser.add_argument("--output_path", type=str, help="Path to output .npz file (Single Patient Mode)")
    
    parser.add_argument("--dim", type=int, help="Feature dimension", default=256)
    parser.add_argument("--backbone", type=str, default="vgg", help="Backbone architecture")
    parser.add_argument("--layer", type=str, default="block3_conv3", help="Layer name")
    
    args = parser.parse_args()

    if args.tiles_dir and args.output_path:
        # Single Patient Mode
        print(f"Running in Single Patient Mode")
        print(f"Tiles Directory: {args.tiles_dir}")
        print(f"Output Path: {args.output_path}")
        
        savez_dict = extract_image_features(Path(args.tiles_dir), backbone=args.backbone, layer=args.layer, dim=args.dim)
        
        if args.output_path.endswith('.csv'):
            # Convert to DataFrame and save as CSV
            # Keys are Patient IDs, Values are feature arrays
            data = []
            for pid, features in savez_dict.items():
                row = {'Patient_ID': pid}
                for i, val in enumerate(features):
                    row[f'DTR_{i}'] = val
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(args.output_path, index=False)
        else:
            np.savez_compressed(args.output_path, **savez_dict)
            
        print(f"Features saved to {args.output_path}")

    elif args.root_dir and args.dest_dir:
        # Batch Mode
        root_dir = Path(args.root_dir)
        dest_dir = Path(args.dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Loop through QuPath projects in the specified root directory
        for sub_dir in os.listdir(root_dir):
            # Only continue if the QuPath project directory already contains the "tiles" directory
            if "tiles" in os.listdir(root_dir/sub_dir):
                marker = re.search(r"(CD[0-9]+|MHC1|PDL1|HE)", sub_dir)
                # The QuPath project name needs to contain one of the markers CD3, CD8, ..., PDL1, MHC1, HE
                if marker is not None:
                    marker = marker.group()

                    # logging
                    print(f"\n\nFolder: {sub_dir}")
                    print(f"Marker: {marker}")
                    print(f"Feature dimension: {args.dim}")

                    tile_dir = root_dir/sub_dir/"tiles"
                    savez_dict = extract_image_features(tile_dir, backbone=args.backbone, layer=args.layer, dim=args.dim)
                    filename = dest_dir/f"tma_tile_dtr_{args.dim}_{marker}.npz"
                    np.savez_compressed(filename, **savez_dict)
                    print("Done.\n\n")
                else:
                    warnings.warn(f"Warning: Found a folder without the marker specified in the folder name: {sub_dir}. Possible markers: CD3, CD8, ..., PDL1, MHC1")
            else:
                warnings.warn(f"Warning: Folder {sub_dir} does not contain a 'tiles' folder")
    else:
        print("Error: Please provide either (--tiles_dir and --output_path) for single mode OR (--root_dir and --dest_dir) for batch mode.")
