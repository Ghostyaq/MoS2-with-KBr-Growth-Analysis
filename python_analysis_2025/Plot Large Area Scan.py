#Description: Plots the Large Area Scan given its text file. Has options of plotting by Peak Separation Distance, Peak Height Ratios, or Peak FWHM Ratio.
#How To Use: Upload the csv file of the Large Area Scan from the WITec program (just change the extension name from .txt to .csv). Change square_size to be equal to the same resolution you took (300x300, 60x60, etc.) Change the variable graph to plot different features. Change target_regions to search for peaks in other areas. If the plot scale is incorrect/not representative of the data, change vmin and vmax to suit your needs (near the bottom of the program).
#Important Notes: Depending on the size of the file, this may take upwards of 5-10 minutes. Afterwards, all relevant data for plotting should be saved to an excel file. Then, you can set load_new_results to False, to skip the majority of loading time.

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

load_new_results = True
file_path = "LAS_example.csv"
square_size = 300 #Need to change this for different resolutions
graph = 'FWHM' #change to FWHM, Ratio, or Separation to change output graph
target_regions = [(380, 400), (400, 420)]  #change the peak locations if needed
save_to_excel = "../peak_separations.xlsx"

#If you need to change the scale on the graph, scroll to the bottom of this program and
#look for "#Adjust the scale in the graph"


# Step 2: helper to find peak positions
def find_peak_positions(y, x, target_regions):
    peaks, _ = find_peaks(y)
    peak_positions = x[peaks]
    peak_values = y[peaks]
    
    found_peaks = []
    found_indices = []
    for region in target_regions:
        mask = (peak_positions >= region[0]) & (peak_positions <= region[1])
        if np.any(mask):
            region_positions = peak_positions[mask]
            region_values = peak_values[mask]
            region_indices = peaks[mask]

            idx = np.argmax(region_values)
            peak_pos = region_positions[idx]
            peak_val = region_values[idx]
            peak_idx = region_indices[idx]
            
            if peak_val - 710 > 20: #assuming "20" is enough to distinguish from background
                found_peaks.append(peak_pos)
                found_peaks.append(peak_val)
                found_indices.append(peak_idx)
            else:
                found_peaks.append(np.nan)
                found_peaks.append(np.nan)
                found_indices.append(np.nan)
        else:
            found_peaks.append(np.nan)
            found_peaks.append(np.nan)
            found_indices.append(np.nan)
    #print(found_peaks)
    return found_peaks, found_indices

def calculate_fwhm(x, y, peak_index, baseline=None):
    peak_height = y[peak_index]
    
    if baseline is None:
        baseline = np.min(y)   # crude baseline estimate
    
    half_max = baseline + (peak_height - baseline) / 2.0
    
    # Left side
    left_idx = np.where(y[:peak_index] < half_max)[0]
    if len(left_idx) == 0:
        return np.nan
    left_idx = left_idx[-1]
    f_left = interp1d(y[left_idx:peak_index+1], x[left_idx:peak_index+1])
    x_left = f_left(half_max)
    
    # Right side
    right_idx = np.where(y[peak_index:] < half_max)[0]
    if len(right_idx) == 0:
        return np.nan
    right_idx = right_idx[0] + peak_index
    f_right = interp1d(y[peak_index:right_idx+1], x[peak_index:right_idx+1])
    x_right = f_right(half_max)
    
    return x_right - x_left

if load_new_results:
    df = pd.read_csv(file_path, delimiter="\t", engine='python', skip_blank_lines=True)

    # Strip whitespace from all string columns (helps with hidden spaces)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Convert first column to numeric (scientific notation handled)
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    # Convert remaining columns to numeric
    spectra_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    # Extract values
    x = df.iloc[:, 0].values
    labels = spectra_df.columns.tolist()
        
    # Step 3: Loop through spectra
    results = []
    for col in spectra_df.columns:
        y = spectra_df[col].astype(float).values
        peaks_info, peak_indices = find_peak_positions(y, x, target_regions)
        if not np.isnan(peaks_info[0]) and not np.isnan(peaks_info[2]):
            separation = abs(peaks_info[2] - peaks_info[0])
        else:
            separation = np.nan
        
        if not np.isnan(peak_indices[0]) and not np.isnan(peak_indices[1]):
            fwhm1 = calculate_fwhm(x, y, int(peak_indices[0]))
            fwhm2 = calculate_fwhm(x, y, int(peak_indices[1]))
            if not np.isnan(fwhm1) and not np.isnan(fwhm2):
                fwhm_ratio = fwhm2/fwhm1
            else:
                fwhm_ratio = np.nan
        else:
            fwhm_ratio = np.nan
            
            
        results.append((col, peaks_info[0], peaks_info[2], peaks_info[1], peaks_info[3], separation, peaks_info[3]/peaks_info[1] if (peaks_info[1] and peaks_info[3]) else np.nan, fwhm_ratio))
        
    results_df = pd.DataFrame(results, columns=["Spectrum", "Peak1 (~390)", "Peak2 (~410)", "Peak1 Value", "Peak2 Value", "Separation", "Ratio", "FWHM"])
    results_df.to_excel(save_to_excel, index=False)

results_df = pd.read_excel(save_to_excel)
scale = results_df[graph].to_numpy()

scale = np.where(pd.isna(scale), np.nan, scale)

# Reshape into square
if len(scale) != square_size**2:
    raise ValueError(f"Expected {square_size**2} data points, got {len(scale)}")

scale_square = scale.reshape((square_size, square_size))

cmap = plt.cm.viridis
cmap.set_bad(color='black')  # NaNs will appear black

# Create one figure
fig, ax = plt.subplots(figsize=(6,6))

# Show background image

# Overlay scale map
cax = ax.imshow(
    scale_square,
    cmap=cmap,
    interpolation='nearest',
    alpha=1,              # adjust transparency
    extent=[0, 300, 0, 300]
    ,vmin = 0.6, vmax = 1.5 #Adjust the scale in the graph
)

# Add colorbar
fig.colorbar(cax, ax=ax, label='Peak Height Ratio')

ax.set_title('Raman Peak Height Ratio Map')
ax.axis('off')
plt.show()
