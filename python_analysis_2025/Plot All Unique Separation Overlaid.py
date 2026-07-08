#Description: Finds x number of unique separations, then plots those graphs.
#How to Use: Change target_regions and file_path as needed. num_wanted corresponds to the number of unique separation you would like. 6 is default, but arbitrary. Feel free to change xlim and ylim as necessary (at the bottom of the program).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

target_regions = [(385, 395), (405, 415)]  # around 390 and 410
file_path = "example.xlsx"
num_wanted = 6

# Step 1: Read Excel, skip 2nd row
df = pd.read_excel(file_path, header=0)  # first row = labels
df = df.drop(index=1)  # drop the useless second row

# Step 2: Extract x-axis and spectra
x = df.iloc[:, 0].astype(float).values  # Raman shift
spectra_df = df.iloc[:, 1:]  # all spectra columns
labels = spectra_df.columns.tolist()

# Step 3: Find peaks near 390 and 410
def find_peak_positions(y, x, target_regions):
    """Find peak positions in target regions given by [(min,max), ...]"""
    peaks, _ = find_peaks(y)
    peak_positions = x[peaks]
    peak_values = y[peaks]
    
    found_peaks = []
    for region in target_regions:
        mask = (peak_positions >= region[0]) & (peak_positions <= region[1])
        if np.any(mask):
            # Pick the highest peak in the region
            idx = np.argmax(peak_values[mask])
            region_peaks = peak_positions[mask]
            found_peaks.append(region_peaks[idx])
        else:
            found_peaks.append(np.nan)  # No peak found
    return found_peaks

peak_separations = []
for i, col in enumerate(spectra_df.columns):
    y = spectra_df[col].astype(float).values
    p1, p2 = find_peak_positions(y, x, target_regions)
    if not np.isnan(p1) and not np.isnan(p2):
        separation = abs(p2 - p1)
        peak_separations.append((i, p1, p2, separation))

# Step 4: Pick 6 spectra with unique separations
unique_spectra = []
used_separations = []
for entry in sorted(peak_separations, key=lambda e: e[3]):  # sort by separation
    if not any(abs(entry[3] - used) < 0.01 for used in used_separations):
        # 0.01 tolerance so close separations aren't "unique"
        unique_spectra.append(entry)
        used_separations.append(entry[3])
    if len(unique_spectra) == num_wanted:
        break

# Step 5: Plot
plt.figure(figsize=(10, 6))
for idx, p1, p2, sep in unique_spectra:
    label = labels[idx]
    plt.plot(x, spectra_df.iloc[:, idx], label=f"{label} | Δ = {sep:.2f} cm⁻¹")

plt.xlim(375, 430)
plt.ylim(700, 1700)
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Intensity (a.u.)")
plt.title("6 Raman Spectra with Unique Peak Separations")
plt.legend()
plt.tight_layout()
plt.show()
