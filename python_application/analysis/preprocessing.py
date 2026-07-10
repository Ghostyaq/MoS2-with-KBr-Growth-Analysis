"""
preprocessing.py

Functions for loading and preprocessing Raman spectra.

Equivalent R functions:
- fread()
- normalize_data()
"""

import numpy as np
import pandas as pd
import time
import config

# ============================================================
# LOADING
# ============================================================

def load_spectrum(file_path):
    """
    Load a WITec Raman spectrum export.

    Expected format:
        Column 0:
            Raman shift

        Columns 1+:
            Individual spectra

    Parameters
        file_path : str
            Path to txt/csv spectrum file

    Returns
        pd.DataFrame
    """

    #data = pd.read_csv(
    #    file_path, sep="\t", engine="python", header=None, dtype=np.float32
    #)

    raw = np.loadtxt(
        file_path,
        delimiter="\t",
        dtype=np.float32
    )

    data = pd.DataFrame(raw)
    return data

# ============================================================
# VALIDATION
# ============================================================

def validate_spectrum(data):
    """
    Check that the Raman file has the expected format.
    """

    if data.shape[1] < 2:
        raise ValueError(
            "Spectrum must contain Raman axis and at least one spectrum."
        )

    if data.iloc[:, 0].isna().any():
        raise ValueError(
            "Raman axis contains missing values."
        )
    return True

# ============================================================
# AXIS / SPECTRA SEPARATION
# ============================================================

def separate_axis(data):
    """
    Separate Raman shift axis from spectra matrix.

    Returns
        raman_axis : numpy.ndarray

        spectra : numpy.ndarray

            Shape:
                rows = Raman shifts
                columns = spectra
    """

    raman_axis = (data.iloc[:, config.RAMAN_AXIS_COLUMN].values)
    spectra = (data.drop(columns=config.RAMAN_AXIS_COLUMN).values)
    return raman_axis, spectra

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(data):
    """
    Normalize each spectrum independently.

    Equivalent R:
        maxes <- apply(mat,2,max)
        mat <- sweep(mat, 2, maxes, "/") * 1000

    Returns
        pd.DataFrame
    """

    raman_axis, spectra = separate_axis(data)
    # Maximum intensity per spectrum
    max_values = np.nanmax(spectra, axis=0)
    max_values[max_values == 0] = 1
    normalized = (spectra / max_values) * config.NORMALIZATION_SCALE
    output = pd.DataFrame(normalized)
    output.insert(0, "Raman Shift", raman_axis)
    return output

# ============================================================
# RAMAN WINDOW CROPPING
# ============================================================

def crop_raman_region(data):
    """
    Restrict Raman spectrum to the MoS2 peaks.

    Equivalent R:
        raw[
            raw$V1 > 375 &
            raw$V1 < 420,
        ]
    """
    raman_axis = data.iloc[:,0]
    mask = ((raman_axis > config.RAMAN_MIN) & (raman_axis < config.RAMAN_MAX))
    return (data.loc[mask].reset_index(drop=True))

# ============================================================
# FULL PIPELINE
# ============================================================

def preprocess(file_path):
    """
    Complete preprocessing pipeline.

    Steps:
        1. Load spectrum
        2. Validate
        3. Normalize
        4. Crop Raman window

    Parameters
        file_path : str

    Returns
        pd.DataFrame
    """
    start = time.time()
    raw = load_spectrum(file_path)
    print(f"File reading took {time.time()-start:.2f} seconds")
    validate_spectrum(raw)
    print(f"Validating took {time.time()-start:.2f} seconds")
    normalized = normalize_data(raw)
    print(f"Processing took {time.time()-start:.2f} seconds")
    cropped = crop_raman_region(normalized)
    print(f"Cropping took {time.time()-start:.2f} seconds")
    return cropped