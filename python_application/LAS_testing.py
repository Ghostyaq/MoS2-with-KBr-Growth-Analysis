import os
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px

def tidy_conversion(data):
    data2 = data.rename(columns = {0: "x_axis"})

    long_df = data2.melt(
        id_vars = "x_axis",
        var_name = "id",
        value_name = "intensity"
    )

    long_df["id"] = long_df["id"].astype(int)

    return long_df

def find_peak_locations(data):
    data = data[(data[0] > 375) & (data[0] < 420)]
    x_axis = data[0].values

    e2g_idx = np.where((x_axis > 375) & (x_axis < 395))[0]
    a1g_idx = np.where((x_axis > 395) & (x_axis < 420))[0]
    results = []

    for i in range(1, data.shape[1]):
        intensity = data.iloc[:, i].values

        peak1 = e2g_idx[np.argmax(intensity[e2g_idx])]
        peak2 = a1g_idx[np.argmax(intensity[a1g_idx])]

        results.append({
            "id": i,
            "x_axis1": x_axis[peak1],
            "intensity1": intensity[peak1],
            "x_axis2": x_axis[peak2],
            "intensity2": intensity[peak2]
        })
        print(f"Spectrum {i} processed.")

    return pd.DataFrame(results)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    return (gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + C)

def auto_gaussian_summary(data, peak_locations):
    data = data[(data[0] > 375) & (data[0] < 420)]
    x = data[0].values
    results = []

    for _, row in peak_locations.iterrows():
        spectrum_id = int(row["id"])
        y = data.iloc[:, spectrum_id].values
        p0 = [
            row["intensity1"], row["x_axis1"], 3,
            row["intensity2"], row["x_axis2"], 3,
            np.min(y)
        ]

        lower = [0, 375, 0.5, 0, 395, 0.5, 0]
        upper = [np.inf, 395, 15, np.inf, 420, 15, np.inf]

        try:
            popt, _ = curve_fit(
                double_gaussian, x, y,
                p0 = p0, bounds = (lower, upper), maxfev = 10000
            )
            print(f"Spectrum {spectrum_id} fitted.")

        except Exception as e:
            print(f"Spectrum {spectrum_id}: ", e)

            results.append({
                "id": spectrum_id, "mu1": 0, "mu2": 0, "fwhm1": 0, "fwhm2": 0,
                "A1": 0, "A2": 0, "area1": 0, "area2": 0, "area_ratio": 0,
                "snr": 0, "rmse": 0, "r_squared": 0, "diff": 0
            })
            continue

        A1, mu1, sigma1, A2, mu2, sigma2, C = popt
        fwhm1 = 2.35482 * sigma1
        fwhm2 = 2.35482 * sigma2

        area1 = A1 * sigma1 * np.sqrt(2*np.pi)
        area2 = A2 * sigma2 * np.sqrt(2*np.pi)
        area_ratio = area1 / area2
        
        fitted = double_gaussian(x, *popt)
        residuals = y - fitted
        rmse = np.sqrt(np.mean(residuals**2))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)

        r_squared = 1 - ss_res / ss_tot
        noise = np.std(residuals)
        snr = np.inf if noise == 0 else np.max(y) / noise

        results.append({
            "id": spectrum_id, "mu1": mu1, "mu2": mu2, 
            "fwhm1": fwhm1, "fwhm2": fwhm2, "A1": A1, "A2": A2,
            "area1": area1, "area2": area2, "area_ratio": area_ratio,
            "snr": snr, "rmse": rmse, "r_squared": r_squared, 
            "diff": abs(mu2 - mu1)
        })

    return pd.DataFrame(results)

size = 300
file_path = (
    f"data/default LAS/"
    f"{size}x{size}/"
    f"Large Area Scan.csv"
    )
compute_time = round(0.00588271 * size ** 2 + 2.21832, 2)

minutes = int(compute_time // 60)
seconds = int(compute_time % 60)
print(f"Time to Compute: {minutes}:{seconds:02d}")

data = pd.read_csv(
    file_path,
    header = None,
    sep = "\t"
)

data = data.apply(pd.to_numeric, errors = "coerce")
data = data.dropna()

peak_summary = find_peak_locations(data)
gaussian_results = auto_gaussian_summary(data, peak_summary)

print(data[0].max(), data[0].min())
print(data.shape)
print(data.head())
print(data.iloc[:10, :5])
print(peak_summary.head())
print(peak_summary.columns)

peak_summary["diff"] = np.abs(peak_summary["x_axis1"] - peak_summary["x_axis2"])
peak_summary["ratio"] = peak_summary["intensity1"] / peak_summary["intensity2"]

peak_summary["ratio"] = np.where(
    peak_summary["ratio"] > 1,
    peak_summary["ratio"],
    1 / peak_summary["ratio"]
)

heatmap_df = peak_summary.merge(gaussian_results, on = "id")

heatmap_df["x"] = ((heatmap_df["id"] - 1) % size) + 1
heatmap_df["y"] = ((heatmap_df["id"] - 1) // size) + 1
heatmap_df["curve"] = (
    (heatmap_df["intensity1"] > 730) & (heatmap_df["intensity2"] > 730)
)

pca_data = data[(data[0] > 375) & (data[0] < 420)]
spectra_matrix = pca_data.iloc[:, 1:].T
scaler = StandardScaler()
spectra_scaled = scaler.fit_transform(spectra_matrix)

pca = PCA(n_components = 5)
scores = pca.fit_transform(spectra_scaled)
pca_scores = pd.DataFrame(scores, columns = ["PC1", "PC2", "PC3", "PC4", "PC5"])

heatmap_df = pd.concat([heatmap_df, pca_scores], axis = 1)

kmeans_vars = heatmap_df[
    [
        "mu1", "mu2", "A1", "A2", "fwhm1", "fwhm2",
        "area1", "area2", "area_ratio", "snr", "rmse", "r_squared",
        "PC1", "PC2", "PC3", "PC4", "PC5"
    ]
]

kmeans_vars = StandardScaler().fit_transform(kmeans_vars)
cluster_num = 4
kmeans = KMeans(n_clusters = cluster_num, random_state = 0, n_init = 20)

clusters = kmeans.fit_predict(kmeans_vars)
heatmap_df["cluster"] = clusters + 1

ordering = np.argsort(kmeans.cluster_centers_[:, 0])
mapping = {
    old: new + 1
    for new, old in enumerate(ordering)
}

heatmap_df["cluster"] = [
    mapping[c] * 4
    for c in heatmap_df["cluster"] - 1
]

fig = px.imshow(
    heatmap_df.pivot(index = "y", columns = "x", values = "cluster"),
    origin = "upper", aspect = "equal",
    color_continuous_scale = ["red", "yellow", "green", "blue"]
)

fig.show()

heatmap_df.to_csv(
    "../paraview_data/analysis_results.csv",
    index = False
)
