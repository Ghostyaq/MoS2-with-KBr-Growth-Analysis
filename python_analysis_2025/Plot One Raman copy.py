#Description: Plots a singular Raman Spectrum given an Excel File and the Column.
#How to Use: Upload data in the same format as "Excel Format Template".

# %matplotlib qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import seaborn as sns

file_path = "example.xlsx"

df = pd.read_excel(file_path)
col_index = 0  # or use: np.random.randint(spectra_df.shape[1])


# Extract x-axis (Raman shift values) from the first column
wavenumbers = df.iloc[:, 0].values

# Extract spectral data (excluding the first column)
spectra_df = df.iloc[:, 1:]

# Randomly select one column (spectrum/sample)
col_name = spectra_df.columns[col_index]
spectrum = spectra_df.iloc[:, col_index].values

# Plot
plt.figure(figsize=(10, 5))
#plt.xlim(380,420)
#plt.ylim(750, 1500)
plt.plot(wavenumbers, spectrum, color='black', linewidth=2)
plt.title(f"Raman Spectrum\nSample: {col_name}", fontsize=14)
plt.xlabel("Raman Shift (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the sample name
print(f"Sample name (column header): {col_name}")
