"""
features.py

Feature engineering for MoS2 Raman spectra.

Equivalent R operations:
- mutate()
- left_join()
- select()
- scale()
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# PEAK FEATURES
# ============================================================

def calculate_peak_features(peak_summary):
    """
    Calculate features from peak locations.

    Equivalent R:
        diff_peak
        intensity_ratio

    Parameters
        peak_summary : dataframe

    Returns
        dataframe
    """

    features = peak_summary.copy()
    features["diff_peak"] = (abs(features["x_axis1"] - features["x_axis2"]))
    ratio = (features["intensity1"] / features["intensity2"])
    features["intensity_ratio"] = np.where(ratio > 1, ratio, 1 / ratio)
    return features

# ============================================================
# MERGING FIT FEATURES
# ============================================================

def merge_fit_features(peak_features, fit_results):
    """
    Combine peak and Gaussian fit information.

    Equivalent R:
        left_join(fit, by="id")
    """
    return peak_features.merge(fit_results, on="id", how="left")

# ============================================================
# MODEL FEATURE TABLE
# ============================================================

def create_model_features(feature_table):
    """
    Select features used by ML models.

    Equivalent R:
        select(...)
    """

    columns = [
        "diff_peak", "diff_fit",
        "A1", "A2", "area_ratio",
        "fwhm1", "fwhm2"
    ]

    return (feature_table[columns].copy())



# ============================================================
# SCALING
# ============================================================

def scale_features(features, scaler=None):
    """
    Standardize features.

    Equivalent R:
        scale()
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
    else:
        scaled = scaler.transform(features)

    scaled = pd.DataFrame(
        scaled,
        columns=features.columns,
        index=features.index
    )
    
    return scaled, scaler