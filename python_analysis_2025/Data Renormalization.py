#Description: In case the Raman Spectroscopy laser shuts off and messes with the magnitude of the laser, use this code to match up mins and maxs across days.
#How To Use: Change file_path to your desired excel file. Change location_to_save to where you want the renormalized data to be.

import pandas as pd
from scipy.signal import medfilt
import numpy as np

file_path = "example.xlsx"
location_to_save = "example_normalized.xlsx

# Step 1: Load data from Excel
df = pd.read_excel(file_path, skiprows=[1])

# Step 2: Clean column headers
df.columns = df.columns.str.strip()
df.columns.values[0] = "Raman Shift"

# Step 3: Extract date from headers
def extract_date(header):
    parts = str(header).split("|")
    return parts[0].strip() if len(parts) > 1 else "Unknown"

original_headers = df.columns[1:]
dates = [extract_date(col) for col in original_headers]

# Step 4: Prepare data
df_raman = df.copy()
df_raman.set_index("Raman Shift", inplace=True)

# Step 5: Group columns by date
date_groups = {}
for col, date in zip(original_headers, dates):
    date_groups.setdefault(date, []).append(col)

# Step 6: Get global min and max over ALL data
global_min = df_raman.min().min()
global_max = df_raman.max().max()
print(f"Global min: {global_min:.2f}, Global max: {global_max:.2f}")

# Step 7: Normalize each day's spectra using min-max scaling to global range
df_scaled = df_raman.copy()
for date, cols in date_groups.items():
    day_data = df_raman[cols]
    day_min = day_data.min().min()
    day_max = day_data.max().max()
    
    if day_max == day_min:
        print(f"Warning: Day {date} has flat spectra; skipping scaling.")
        continue

    # Apply min-max normalization to global range
    scaled = (day_data - day_min) / (day_max - day_min)  # scale to 0–1
    scaled = scaled * (global_max - global_min) + global_min  # scale to global min–max
    df_scaled[cols] = scaled

    print(f"Scaled {date}: local min {day_min:.2f}, max {day_max:.2f} → global min {global_min:.2f}, max {global_max:.2f}")

# Step 8: Save output
df_scaled.reset_index(inplace=True)
df_scaled.to_excel(location_to_save, index=False)
