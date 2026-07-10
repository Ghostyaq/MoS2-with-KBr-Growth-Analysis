"""
fitting.py

Gaussian fitting routines for MoS2 Raman spectra.

Equivalent R functions:
- gaussian()
- double_gaussian()
- auto_gaussian_summary()
"""

import config
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# ============================================================
# GAUSSIAN FUNCTIONS
# ============================================================

def gaussian(x, A, mu, sigma):
    """
    Single Gaussian peak.
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    """
    Two Gaussian peaks + constant background.

    Represents:
        E2g + A1g + baseline
    """
    return (gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + C)

# ============================================================
# INITIAL PARAMETERS
# ============================================================

def get_initial_guess(
        peak_row,
        y
):
    """
    Generate starting parameters.

    Equivalent to:

    A1_guess <- peak_locations$intensity1[i]
    mu1_guess <- peak_locations$x_axis1[i]
    """

    return [
        peak_row["intensity1"],
        peak_row["x_axis1"],
        config.INITIAL_SIGMA,

        peak_row["intensity2"],
        peak_row["x_axis2"],
        config.INITIAL_SIGMA,

        np.min(y)
    ]

# ============================================================
# FIT STATISTICS
# ============================================================

def calculate_statistics(x, y, fitted_y):
    """
    Calculate RMSE, R², and SNR.
    """

    residuals = y - fitted_y
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot == 0:
        r_squared = 0

    else:
        r_squared = 1 - (ss_res / ss_tot)

    noise = np.std(residuals, ddof = 1)
    if noise == 0:
        snr = np.inf
    else:
        snr = np.max(y) / noise

    return (rmse, r_squared, snr)

# ============================================================
# FIT SINGLE SPECTRUM
# ============================================================

def fit_single_spectrum(x, y, peak_row):
    """
    Fit one Raman spectrum.

    Returns dictionary of extracted features.
    """

    failed_result = {
        "id": peak_row["id"],
        "mu1": 0, "mu2": 0,
        "fwhm1": 0, "fwhm2": 0,
        "A1": 0, "A2": 0,
        "area1": 0, "area2": 0, "area_ratio": 0,
        "snr": 0, "rmse": 0, "r_squared": 0,
        "diff_fit": 0, "status": "failed"
    }

    initial = get_initial_guess(peak_row, y)
    try:
        params, covariance = curve_fit(
            double_gaussian, x, y, p0=initial,
            bounds=(
                [0, 370, 0.5, 0, 390, 0.5, 0],
                [1100, 400, 20, 1100, 430, 20, 1100]
            ),
            maxfev=10000
        )

    except Exception:
        return failed_result

    (A1, mu1, sigma1, A2, mu2, sigma2, C) = params

    fitted_y = double_gaussian(x, *params)
    rmse, r_squared, snr = calculate_statistics(x, y, fitted_y)

    if r_squared < config.MIN_R2:
        failed_result["status"] = "r2 too low"
        failed_result["r_squared"] = r_squared
        return failed_result

    fwhm1 = 2.35482 * sigma1
    fwhm2 = 2.35482 * sigma2

    area1 = (A1 * sigma1 * np.sqrt(2*np.pi))
    area2 = (A2 * sigma2 * np.sqrt(2*np.pi))
    area_ratio = (area1 / area2 if area2 != 0 else np.inf)

    return {
        "id": peak_row["id"],
        "mu1": mu1, "mu2": mu2,
        "fwhm1": fwhm1, "fwhm2": fwhm2,
        "A1": A1, "A2": A2,
        "area1": area1, "area2": area2, "area_ratio": area_ratio,
        "snr": snr, "rmse": rmse, "r_squared": r_squared,
        "diff_fit": abs(mu2 - mu1), "status": "success"
    }

# ============================================================
# FIT WORKER
# ============================================================

def fit_worker(args):
    """
    Worker function for multiprocessing.

    Each process receives:
        x_fit
        y spectrum
        peak row
    """
    x_fit, y, peak_row = args
    return fit_single_spectrum(x_fit, y, peak_row)

# ============================================================
# FIT ALL SPECTRA
# ============================================================
def fit_all_spectra(data, peak_locations, workers=None, progress_callback=None):
    """
    Fit every Raman spectrum using multiprocessing.

    Parameters:
        data:
            Raman dataframe

        peak_locations:
            Output from find_peak_locations()

        workers:
            Number of CPU processes.
            Default:
                cpu_count()-1
    """

    start = time.time()
    x = data.iloc[:,0].values
    mask = ((x > config.RAMAN_MIN) & (x < config.RAMAN_MAX))
    x_fit = x[mask]
    spectra = (data.iloc[:,1:].values)

    jobs = []
    for _, peak_row in peak_locations.iterrows():
        spectrum_id = int(peak_row["id"])
        y = spectra[:, spectrum_id][mask]
        jobs.append((x_fit, y, peak_row))

    if workers is None:
        workers = max(cpu_count()-config.RESERVE_CORES, 1)

    print(
        f"Fitting {len(jobs)} spectra "
        f"using {workers} workers..."
    )

    if workers == 1:
        print("Running sequential fitting...")
        results = [fit_worker(job) for job in jobs]
    else:
        print(f"Running parallel fitting with {workers} workers...")

        with Pool(processes=workers) as pool:
            results = []
            total = len(jobs)
            for i, result in enumerate(pool.imap(fit_worker, jobs)):
                results.append(result)
                if progress_callback:
                    if i % 1000 == 0:
                        progress_callback(i + 1, total)

    print(
        f"Finished fitting in "
        f"{time.time()-start:.1f} seconds"
    )

    return pd.DataFrame(results)