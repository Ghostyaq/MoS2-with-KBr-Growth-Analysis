"""
preprocessing.py

Functions for loading and preprocessing MoS2 Raman spectra.

Equivalent R functions:
- fread()
- normalize_data()
- Raman region filtering
"""

import numpy as np
import pandas as pd
import config

# ============================================================
# DATA LOADING
# ============================================================

def load_spectrum(file_path):
    """
    Load a WITec exported Raman spectrum file.

    Expected format:
        Column 0: Raman shift
        Columns 1+: spectra

    Parameters
        file_path : str
            Path to txt/csv file

    Returns
        dataframe : pandas.DataFrame
    """

    data = pd.read_csv(file_path, sep = None, engine = "python", header = None)
    return data

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_spectrum(data):
    """
    Basic checks that data is formatted correctly.
    """

    if data.shape[1] < 2:
        raise ValueError(
            "Spectrum file must contain Raman axis + spectra columns."
        )
        
    if data.iloc[:,0].isna().any():
        raise ValueError(
            "Raman axis contains missing values."
        )

    return True

# ============================================================
# SPLIT AXIS AND SPECTRA
# ============================================================

def separate_axis(data):
    """
    Separate Raman shift axis and intensity matrix.

    Returns
        raman_axis : numpy array
        spectra : numpy array
            rows = Raman shifts
            columns = spectra
    """

    raman_axis = data.iloc[:, config.RAMAN_AXIS_COLUMN].values

    spectra = (
        data
        .drop(columns=config.RAMAN_AXIS_COLUMN)
        .values
    )

    return raman_axis, spectra

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(data):
    """
    Normalize each Raman spectrum independently.

    Equivalent R:
        maxes <- apply(mat,2,max)
        mat <- sweep(mat,2,maxes,"/")*1000

    Returns
        normalized dataframe
    """

    raman_axis, spectra = separate_axis(data)
    max_values = np.nanmax(spectra, axis = 0)
    max_values[max_values == 0] = 1
    normalized = (spectra / max_values) * config.NORMALIZATION_SCALE
    
    output = pd.DataFrame(normalized)
    output.insert(0, "Raman Shift",raman_axis)
    return output

# ============================================================
# RAMAN REGION EXTRACTION
# ============================================================

def crop_raman_region(data):
    """
    Keep only Raman shifts used for MoS2 analysis.

    Equivalent R:
        raw[raw$V1 > 375 & raw$V1 < 420, ]
    """

    mask = (
        (data.iloc[:,0] > config.RAMAN_MIN)
        &
        (data.iloc[:,0] < config.RAMAN_MAX)
    )
    return data.loc[mask].reset_index(drop=True)

# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess(file_path):
    """
    Full preprocessing workflow.

    Steps:
    1. Load file
    2. Validate
    3. Normalize
    4. Crop Raman region

    Returns
        processed dataframe
    """

    raw = load_spectrum(file_path)
    validate_spectrum(raw)
    normalized = normalize_data(raw)
    cropped = crop_raman_region(normalized)
    return cropped
