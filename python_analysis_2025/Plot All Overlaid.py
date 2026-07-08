#Description: Plots all Raman Spectra in the Excel File, overlaid.
#How to Use: Change file_path to your data. Change xlim and ylim to your preferences.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import seaborn as sns
from scipy.signal import find_peaks

file_path = "example.xlsx"

df = pd.read_excel(file_path)

# Extract x-axis (Raman shift values) from the first column
wavenumbers = df.iloc[:, 0].values

# Extract spectral data (excluding the first column)
spectra_df = df.iloc[:, 1:]

# Generate a color for each column
colors = [
    (0.0, 0.0, 0.9),
    (0.80, 0.0, 0.0),
    (0.25, 0.69, 0.65)
    #(1.0, 0.75, 0.0),
    #(0.0, 0.0, 0.0)
]

# Plot all spectra overlaid
plt.figure(figsize=(8, 10))
plt.xlim(375, 430)
plt.ylim(700, 1800)

for i, col_name in enumerate(spectra_df.columns):
    plt.plot(wavenumbers, spectra_df.iloc[:, i].values, color=colors[i], label=col_name, linewidth=4)


#plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Overlaid Raman Spectra", fontsize=14)
plt.xlabel("Raman Shift (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.grid(True)

plt.tight_layout()
plt.show()


# Function to compute peak separations for each spectrum
def compute_peak_separations(wavenumbers, intensity, prominence=50):
    peaks, _ = find_peaks(intensity, prominence=prominence)
    peak_positions = wavenumbers[peaks]
    separations = np.diff(peak_positions)
    return peak_positions, separations

# Loop through each spectrum and compute peak separations
for i, col_name in enumerate(spectra_df.columns):
    intensity = spectra_df.iloc[:, i].values
    peak_positions, separations = compute_peak_separations(wavenumbers, intensity)
    
    print(f"\nSpectrum: {col_name}")
    print(f"  Peak Positions: {np.round(peak_positions, 2)} cm⁻¹")
    print(f"  Separation Distances: {np.round(separations, 2)} cm⁻¹")
