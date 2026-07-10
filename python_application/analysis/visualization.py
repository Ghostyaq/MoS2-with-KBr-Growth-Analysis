"""
visualization.py

Visualization and export functions for MoS2 Raman analysis.

Equivalent R operations:
- ggplot heatmaps
- plotly 3D visualization
- heatmap_df generation
- write.csv() for ParaView
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CREATE MAP DATAFRAME
# ============================================================

def create_map_dataframe(metadata, predictions):
    """
    Combine spatial information with model predictions.

    Expected inputs:

    metadata:
        x 
        y

    predictions:
        lda_class
        thickness predictions

    Returns:
        dataframe ready for plotting/export

    Equivalent R:
        left_join()
    """

    result = metadata.copy()
    prediction_columns = predictions.columns
    for column in prediction_columns:
        result[column] = predictions[column].values
    return result

# ============================================================
# THICKNESS MAP
# ============================================================

def plot_thickness_map(dataframe, thickness_column="rf_thickness"):
    """
    Plot predicted thickness as a 2D map.

    Equivalent R:
        ggplot +
        geom_tile()

    Requires:
        x
        y
        thickness
    """

    if thickness_column not in dataframe.columns:
        raise ValueError(
            f"{thickness_column} not found."
        )

    pivot = dataframe.pivot(
        index="y",
        columns="x",
        values=thickness_column
    )


    plt.figure(figsize=(8,6))

    plt.imshow(
        pivot,
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(
        label="Thickness (nm)"
    )

    plt.xlabel("X position")
    plt.ylabel("Y position")

    plt.title(
        "MoS2 Thickness Map"
    )

    plt.show()



# ============================================================
# CLASSIFICATION MAP
# ============================================================

def plot_classification_map(
        dataframe,
        column="lda_class"
):
    """
    Plot LDA classification.

    Classes:
        Background
        Monolayer
        Bilayer
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"{column} not found."
        )


    labels = pd.factorize(
        dataframe[column]
    )[0]


    plot_data = dataframe.copy()

    plot_data["class_numeric"] = labels


    pivot = plot_data.pivot(
        index="y",
        columns="x",
        values="class_numeric"
    )


    plt.figure(figsize=(8,6))


    plt.imshow(
        pivot,
        origin="lower",
        aspect="auto"
    )


    plt.colorbar(
        label="Class"
    )


    plt.xlabel("X position")
    plt.ylabel("Y position")

    plt.title(
        "MoS2 Layer Classification"
    )


    plt.show()



# ============================================================
# MODEL COMPARISON
# ============================================================

def plot_model_comparison(
        predictions
):
    """
    Compare regression model outputs.

    Useful for:
        Linear
        Ridge
        Random Forest
    """

    thickness_columns = [
        col for col in predictions.columns
        if "thickness" in col
    ]


    plt.figure(figsize=(8,5))


    for column in thickness_columns:

        plt.hist(
            predictions[column],
            bins=50,
            alpha=0.5,
            label=column
        )


    plt.xlabel(
        "Predicted Thickness"
    )

    plt.ylabel(
        "Count"
    )

    plt.legend()

    plt.title(
        "Thickness Model Comparison"
    )

    plt.show()



# ============================================================
# PARAVIEW EXPORT
# ============================================================

def export_paraview_csv(
        dataframe,
        output_file,
        thickness_column="rf_thickness"
):
    """
    Export structured data for ParaView.

    Expected output:

        x,y,thickness

    Equivalent R:

        write.csv()
    """


    required = [
        "x",
        "y",
        thickness_column
    ]


    for column in required:

        if column not in dataframe.columns:
            raise ValueError(
                f"Missing column: {column}"
            )


    output = dataframe[
        [
            "x",
            "y",
            thickness_column
        ]
    ].copy()


    output = output.rename(
        columns={
            thickness_column:
            "thickness"
        }
    )


    output.to_csv(
        output_file,
        index=False
    )


# ============================================================
# QUICK SUMMARY
# ============================================================

def summarize_predictions(
        predictions
):
    """
    Print prediction statistics.
    """

    print("\nPrediction Summary")
    print("------------------")

    print(
        predictions.describe()
    )

    if "lda_class" in predictions.columns:

        print("\nLayer Counts")

        print(
            predictions["lda_class"]
            .value_counts()
        )