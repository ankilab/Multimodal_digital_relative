from feature_extraction.extract_tabular_features import NOMINAL_FEATURES, ORDINAL_FEATURES, DISCRETE_FEATURES
from feature_extraction.extract_tabular_features import BLOOD_FEATURES
from feature_extraction.extract_tma_features import TMA_FEATURES

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import joblib

SEED = 42


def setup_preprocessing_pipeline(
        columns: list[str], min_max_scaler: bool = False
) -> ColumnTransformer:
    """
    Sets up a sklearn pipeline for preprocessing the data before applying UMAP
    Parameters
    ----------
    columns : list of str
        List of columns to consider
    min_max_scaler: bool
        If True, MinMaxScaler is used instead of StandardScaler. Default: False

    Returns
    -------
    Scikit-learn ColumnTransformer

    """
    categorical_columns = NOMINAL_FEATURES
    numeric_columns = BLOOD_FEATURES + TMA_FEATURES + DISCRETE_FEATURES + ORDINAL_FEATURES

    categorical_columns = [col for col in categorical_columns if col in columns]
    numeric_columns = [col for col in numeric_columns if col in columns]

    remaining_columns = [col for col in columns if col not in categorical_columns and col not in numeric_columns]
    pipeline_categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    pipeline_numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler() if min_max_scaler else StandardScaler())
    ])
    pipeline_encoded = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", pipeline_categorical, categorical_columns),
            ("numeric", pipeline_numeric, numeric_columns),
            ("encoded", pipeline_encoded, remaining_columns)
        ],
        remainder="passthrough",
        verbose=False
    )
    return preprocessor


def get_umap_embedding(features_directory, umap_min_dist=0.1, umap_n_neighbors=15):
    """
    Loads multimodal features, preprocessed them and applies UMAP to reduce the dimensions.
    Parameters
    ----------
    features_directory : str
        Directory that contains the extracted features (.csv)
    umap_min_dist : flaot
        min_dist parameter for UMAP
    umap_n_neighbors : int
        n_neighors parameter for UMAP

    Returns
    -------
    Pandas DataFrame
        Dataframe containing the features and the resulting embedding in the "UMAP 1" and "UMAP 2" columns

    """
    fdir = Path(features_directory)

    # Load encoded data
    clinical = pd.read_csv(fdir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(fdir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(fdir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(fdir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density= pd.read_csv(fdir/"tma_cell_density.csv", dtype={"patient_id": str})
    targets = pd.read_csv(fdir/"targets.csv", dtype={"patient_id": str})

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.merge(targets, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    # Preprocess embeddings
    # Exclude target columns from UMAP
    target_cols = [c for c in targets.columns if c != "patient_id"]
    df_for_umap = df.drop(["patient_id"] + target_cols, axis=1)
    
    preprocessor = setup_preprocessing_pipeline(df_for_umap.columns)
    preprocessor = preprocessor.fit(df_for_umap)
    # Save preprocessor
    joblib.dump(preprocessor, fdir/"../models/preprocessor.pkl")

    embeddings = preprocessor.transform(df_for_umap)

    # Reduce to 2D
    umap_model = UMAP(random_state=SEED, min_dist=umap_min_dist, n_neighbors=umap_n_neighbors)
    umap_model.fit(embeddings)
    joblib.dump(umap_model, fdir/"../models/umap_model.pkl")
    umap = umap_model.transform(embeddings)
    #umap = np.array([umap_model.transform([embeddings[i]])[0] for i in range(embeddings.shape[0])])



    # Normalize axes
    #tx, ty = umap[:, 0], umap[:, 1]
    #tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    #ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    # Add UMAP to the dataframe
    df["UMAP 1"] = umap[:, 0]
    df["UMAP 2"] = umap[:, 1]

    


    # Check embeddings comparison
    # Original embedding
    # a_embedding=embeddings[0]

    # # New patient embedding
    # b = df.loc[0]
    # print(b)
    # b_embedding = preprocessor.transform(pd.DataFrame([b.drop("patient_id")]))

    # print("np.allclose(a_embedding,b_embedding):", np.allclose(a_embedding,b_embedding)) 
    # print("a_embedding == b_embedding:", [a_embedding] == b_embedding) 

    # print("np.allclose(umap_model.transform([a_embedding]), umap_model.transform(b_embedding))", np.allclose(umap_model.transform([a_embedding]), umap_model.transform(b_embedding)))
    # print("np.allclose(umap_model.transform(embeddings)[0], umap_model.transform(b_embedding))", np.allclose(umap_model.transform(embeddings)[0], umap_model.transform(b_embedding)))
    # print(umap_model.transform([a_embedding]) , umap_model.transform(b_embedding),umap_model.transform(embeddings)[0])

    # # Preprocess embeddings
    # preprocessor = joblib.load( "../preprocessor.pkl")
    # b_embeddings_loaded = preprocessor.transform(pd.DataFrame([b.drop("patient_id")]))

    # print("np.allclose(a_embedding,b_embedding):", np.allclose(embeddings[0],b_embeddings_loaded)) 


    # # Reduce to 2D
    # umap_model_loaded = joblib.load("../umap_model.pkl")
    # print("np.allclose(umap.transform(a), umap.transform(b))", np.allclose(umap_model_loaded.transform(embeddings[0]), umap_model_loaded.transform(b_embeddings_loaded)))  


    return df


def get_embedding(features_directory, method='umap', umap_min_dist=0.1, umap_n_neighbors=15,
                  pca_n_components=2, tsne_perplexity=30):
    """
    Loads multimodal features, preprocesses them and applies a dimensionality
    reduction method (UMAP, PCA, or t-SNE) to produce a 2-D embedding.

    Parameters
    ----------
    features_directory : str
        Directory that contains the extracted features (.csv)
    method : str
        Reduction method: 'umap', 'pca', or 'tsne'. Default: 'umap'
    umap_min_dist : float
        min_dist parameter for UMAP. Default: 0.1
    umap_n_neighbors : int
        n_neighbors parameter for UMAP. Default: 15
    pca_n_components : int
        Number of PCA components (first two are used for 2-D plot). Default: 2
    tsne_perplexity : float
        Perplexity for t-SNE. Default: 30

    Returns
    -------
    pd.DataFrame
        Dataframe containing the features and the 2-D embedding in
        "Dim 1" and "Dim 2" columns, plus a "method" column.
    """
    method = method.lower()
    fdir = Path(features_directory)

    # ---- Load and merge modalities (identical to get_umap_embedding) ----
    clinical = pd.read_csv(fdir / "clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(fdir / "pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(fdir / "blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(fdir / "icd_codes.csv", dtype={"patient_id": str})
    cell_density = pd.read_csv(fdir / "tma_cell_density.csv", dtype={"patient_id": str})
    targets = pd.read_csv(fdir / "targets.csv", dtype={"patient_id": str})

    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.merge(targets, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    target_cols = [c for c in targets.columns if c != "patient_id"]
    df_for_embedding = df.drop(["patient_id"] + target_cols, axis=1)

    # ---- Preprocess (identical to get_umap_embedding) ----
    preprocessor = setup_preprocessing_pipeline(df_for_embedding.columns)
    preprocessor = preprocessor.fit(df_for_embedding)
    joblib.dump(preprocessor, fdir / "../models/preprocessor.pkl")
    embeddings = preprocessor.transform(df_for_embedding)

    # ---- Fit reduction model ----
    if method == 'umap':
        model = UMAP(random_state=SEED, min_dist=umap_min_dist, n_neighbors=umap_n_neighbors)
        model.fit(embeddings)
        coords = model.transform(embeddings)
        joblib.dump(model, fdir / "../models/umap_model.pkl")
    elif method == 'pca':
        model = PCA(n_components=pca_n_components, random_state=SEED)
        coords = model.fit_transform(embeddings)
        joblib.dump(model, fdir / "../models/pca_model.pkl")
    elif method == 'tsne':
        model = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=SEED)
        coords = model.fit_transform(embeddings)
        joblib.dump(model, fdir / "../models/tsne_model.pkl")
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'umap', 'pca', or 'tsne'.")

    df["Dim 1"] = coords[:, 0]
    df["Dim 2"] = coords[:, 1]
    df["method"] = method
    return df


def plot_umap(dataframe, subplot_titles, subplot_features, numerical_features=[], marker_size=4, filename=None):
    """
    Shows 2D UMAP embeddings in a scatterplot where points are colored by distinct features.
    Parameters
    ----------
    dataframe : Pandas dataframe
        Dataframe that contains both the UMAP embeddings and features
    subplot_titles : list of str
        List of titles for suplots
    subplot_features : list of str
        List of features (columns in the dataframe)
    numerical_features : list of str
        List of features that are numerical. Default: []
    marker_size : int
        Marker size for scatter plot. Default: 4
    filename : str
        Plot is saved to a file if filename is specified. Default: None

    """
    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, len(subplot_titles), figsize=(7, 2.5))

    for i, feature in enumerate(subplot_features):

        if feature in numerical_features:
            df = dataframe.copy()
            palette = sns.color_palette("plasma", as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(df[feature].min().min(),
                                                                        df[feature].max().max()))
            hue_norm = sm.norm
            legend = False

        else:
            df = dataframe.copy().fillna("missing")
            palette = sns.color_palette("Set2", n_colors=len(df[feature].fillna("missing").unique()))
            hue_norm = None
            legend = True

        sns.scatterplot(df, x="UMAP 1", y="UMAP 2", hue=feature,
                        hue_norm=hue_norm, palette=palette, legend=legend, s=marker_size, ax=axes[i]);

        # Axes
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_aspect("equal")

        # Title
        axes[i].set_title(subplot_titles[i])

        plt.tight_layout(pad=2)
        if feature in numerical_features:
            plt.colorbar(sm, ax=axes[i], fraction=0.06)
        else:
            axes[i].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False, fontsize=6)

    sns.despine()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.show()