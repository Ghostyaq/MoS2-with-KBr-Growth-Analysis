"""
main.py

Main execution pipeline for MoS2 Raman analysis.

Workflow:

Training:
    spectra
        |
        v
    preprocessing
        |
        v
    peak detection
        |
        v
    gaussian fitting
        |
        v
    feature engineering
        |
        v
    model training


Prediction:
    LAS spectra
        |
        v
    feature extraction
        |
        v
    model prediction
        |
        v
    export results
"""


import os
import pandas as pd

import config
from pathlib import Path
from analysis import preprocessing
from analysis import peak_analysis
from analysis import fitting
from analysis import features
from analysis import models
from analysis import visualization

def status(message, status_callback):
    if status_callback is not None:
        status_callback(message)
    else:
        print(message)

def send_progress(value, progress_callback):
    if progress_callback is not None:
        progress_callback(value, 100)

# ============================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================

def generate_features(
        file_path, workers, status_callback=None, progress_callback=None
        ):
    """
    Complete Raman feature extraction pipeline.

    Returns:
        feature dataframe
    """

    status("\nLoading spectrum...", status_callback)
    data = preprocessing.preprocess(file_path)

    status("Finding Raman peaks...", status_callback)
    peak_locations = (peak_analysis.find_peak_locations(data))

    status("Fitting Gaussian peaks...", status_callback)
    fit_results = fitting.fit_all_spectra(
        data, peak_locations, workers, progress_callback
        )

    status("Combining features...", status_callback)
    peak_features = (features.calculate_peak_features(peak_locations))
    feature_table = (features.merge_fit_features(peak_features, fit_results))
    return feature_table

def generate_features_from_files(
        file_list, workers, status_callback=None, progress_callback=None
        ):
    """
    Process multiple Raman spectra files.

    Used for training datasets.
    """

    all_features = []

    for i, file in enumerate(file_list):
        status(f"\nProcessing {file}", status_callback)
        feature_table = generate_features(file, workers)
        feature_table["id"] = i
        all_features.append(feature_table)

    return pd.concat(
        all_features,
        ignore_index=True
    )


# ============================================================
# TRAINING
# ============================================================

def find_training_files(training_folder=config.TRAINING_DATA_DIR):
    """
    Recursively locate every training spectrum.

    Returns
    -------
    list[str]
        Sorted list of file paths.
    """

    training_folder = Path(training_folder)

    files = sorted(training_folder.rglob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No training files found in {training_folder}"
        )
    return [str(file) for file in files]

def create_training_labels(file_list):
    """
    Generate layer labels from folder names.
    """

    labels = []
    for i, file in enumerate(file_list):
        if "background" in file.lower():
            layer = "Background"
            thickness = 0
        elif "monolayer" in file.lower():
            layer = "Monolayer"
            thickness = 0.7

        elif "bilayer" in file.lower():
            layer = "Bilayer"
            thickness = 1.4

        else:
            raise ValueError(f"Cannot determine layer type from {file}")

        labels.append(
            {
                "id": i,
                "Layer": layer,
                "thickness": thickness
            }
        )
    return pd.DataFrame(labels)

def train_pipeline(
        training_files, status_callback=None, progress_callback=None
        ):
    """
    Train all models.

    Parameters:

    training_file:
        Raman spectra

    labels:
        dataframe containing:

            Layer
            thickness
    """

    feature_table = generate_features_from_files(
        training_files, config.TRAINING_WORKERS
        )
    labels = create_training_labels(training_files)
    feature_table = feature_table.merge(labels, on="id")    
    model_features = (features.create_model_features(feature_table))
    scaled_features, scaler = (features.scale_features(model_features))
    
    status("\nTraining models...", status_callback)
    models_dict = (
        models.train_all_models(
        pd.concat(
        [scaled_features, feature_table[["Layer", "thickness"]]],
        axis=1
        ),
        scaled_features.columns
        )
    )

    models.save_models(models_dict)
    status("\nModels saved.", status_callback)
    return models_dict, scaler

# ============================================================
# PREDICTION
# ============================================================


def predict_pipeline(
        spectrum_file, workers, status_callback=None, progress_callback=None
        ):
    """
    Apply trained models to new spectra.
    """

    feature_table = generate_features(
        spectrum_file, workers, status_callback, progress_callback
        )
    model_features = (features.create_model_features(feature_table))
    trained_models = (models.load_models())

    status("\nScaling features...", status_callback)
    scaled_features, _ = (features.scale_features(model_features))

    status("Predicting...", status_callback)
    predictions = (models.predict_models(trained_models, scaled_features))

    results = pd.concat([feature_table, predictions], axis=1)
    return results

# ============================================================
# EXPORT
# ============================================================


def save_results(results, status_callback, filename="analysis_results.csv"):
    """
    Save final analysis table.
    """

    os.makedirs("python_application/data/results", exist_ok=True)
    path = os.path.join("python_application/data/results", filename)

    results.to_csv(path, index=False)
    status(f"\nSaved: {path}", status_callback)