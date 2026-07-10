"""
peak_analysis.py

Functions for locating MoS2 Raman peaks.

Equivalent R function:
    find_peak_locations()
"""
import numpy as np
import pandas as pd
import config

# ============================================================
# REGION EXTRACTION
# ============================================================

def get_peak_regions(data):
    """
    Split Raman spectra into E2g and A1g regions.

    Parameters
        data : pd.DataFrame
            Column 0 = Raman shift
            Columns 1+ = spectra

    Returns
        x_axis
        e2g_region
        a1g_region
    """
    x_axis = data.iloc[:, 0].values
    spectra = (data .iloc[:, 1:] .values)
    mask = ((x_axis > config.RAMAN_MIN) & (x_axis < config.RAMAN_MAX))
    x_axis = x_axis[mask]
    spectra = spectra[mask, :]

    e2g_mask = ((x_axis > config.E2G_MIN) & (x_axis < config.E2G_MAX))
    a1g_mask = ((x_axis > config.A1G_MIN) & (x_axis < config.A1G_MAX))
    e2g_region = spectra[e2g_mask, :]
    a1g_region = spectra[a1g_mask, :]

    return (x_axis, e2g_region, a1g_region)

# ============================================================
# PEAK FINDING
# ============================================================

def find_peak_indices(region):
    """
    Find maximum intensity row for every spectrum.

    Equivalent to:

        max.col(t(region))

    Parameters
        region : numpy array

    Returns
        indices : numpy array
    """
    return np.argmax(
        region,
        axis = 0
    )

# ============================================================
# INDEX -> RAMAN SHIFT
# ============================================================

def convert_peak_positions(indices, x_axis, region_min, region_max):
    """
    Convert local indices inside a peak window
    into Raman shift values.
    """

    mask = ((x_axis > region_min) & (x_axis < region_max))
    region_axis = x_axis[mask]
    return region_axis[indices]

# ============================================================
# MAIN FUNCTION
# ============================================================

def find_peak_locations(data):
    """
    Locate E2g and A1g peaks for every spectrum.

    Returns
        pandas.DataFrame

    Columns:
        id
        x_axis1
        intensity1
        x_axis2
        intensity2

    """
    x_axis, e2g, a1g = get_peak_regions(data)
    e2g_indices = find_peak_indices(e2g)
    a1g_indices = find_peak_indices(a1g)

    e2g_positions = convert_peak_positions(
        e2g_indices, x_axis,
        config.E2G_MIN, config.E2G_MAX
    )

    a1g_positions = convert_peak_positions(
        a1g_indices, x_axis,
        config.A1G_MIN, config.A1G_MAX
    )

    spectra = (data .iloc[:, 1:] .values)

    mask = (
        (data.iloc[:,0] > config.RAMAN_MIN) & 
        (data.iloc[:,0] < config.RAMAN_MAX)
    )
    spectra = spectra[mask,:]

    e2g_intensities = (e2g[e2g_indices, np.arange(e2g.shape[1])])
    a1g_intensities = (a1g[a1g_indices, np.arange(a1g.shape[1])])

    result = pd.DataFrame({
        "id": np.arange(spectra.shape[1]),
        "x_axis1": e2g_positions,
        "intensity1": e2g_intensities,
        "x_axis2": a1g_positions,
        "intensity2": a1g_intensities
    })

    return result