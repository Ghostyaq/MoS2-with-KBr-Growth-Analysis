#Description: Plots Full-Width-Half-Max vs. KBr Volume
#How to Use: The current data is hardcoded in. You may need change the values. Specifically, the extract_day_label function may need to be changed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, medfilt
import re

file_path = "example.xlsx"

df = pd.read_excel(file_path)
wavenumbers = df.iloc[:, 0].values
spectra_df = df.iloc[:, 1:]
headers = df.columns[1:]

# This function is hardcoded - Feel free to delete/edit
# Function to map Day to concentration label
def extract_day_label(label):
    match = re.match(r"(\d+)", label)
    if not match:
        return None
    day = int(match.group(1))
    if day in [1, 2]:
        return "2mL KBr"
    elif day == 7:
        return "1mL KBr"
    elif day == 10:
        return "0.5mL KBr"
    elif day == 15:
        return "0.5mL KBr (diff. conditions)"
    else:
        return f"Day {day}"  # fallback in case of unexpected day

# Initialize storage
all_fwhm = []
group_labels = []

# Process each spectrum
for col_name in headers:
    spectrum = spectra_df[col_name].values
    spectrum = medfilt(spectrum, kernel_size=5)

    # Find peaks
    peaks, _ = find_peaks(spectrum, height=100, distance=5)
    if len(peaks) < 4:
        continue  # Skip if not enough peaks

    # Get top 4 peaks by intensity
    top_peak_indices = peaks[np.argsort(spectrum[peaks])[-4:]]
    selected = top_peak_indices[np.argsort(spectrum[top_peak_indices])[:2]]

    # FWHM for each selected peak
    for peak_idx in selected:
        results_half = peak_widths(spectrum, [peak_idx], rel_height=0.5)
        width_cm1 = results_half[0][0] * (wavenumbers[1] - wavenumbers[0])
        all_fwhm.append(width_cm1)
        group_labels.append(extract_day_label(col_name))

# Create DataFrame for plotting
result_df = pd.DataFrame({
    'Group': group_labels,
    'FWHM': all_fwhm
})

# Plot boxplot of FWHM grouped by concentration
plt.figure(figsize=(8, 6))
result_df.boxplot(column='FWHM', by='Group', grid=False)
plt.title("FWHM by KBr Concentration / Growth Condition")
plt.suptitle("")  # Remove automatic title
plt.xlabel("Group")
plt.ylabel("FWHM (cm$^{-1}$)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
