"""
models.py

Machine learning models for MoS2 Raman thickness prediction.

Equivalent R:
- MASS::lda
- lm()
- glmnet::cv.glmnet()
- randomForest::randomForest()
"""
import numpy as np
import pandas as pd
import joblib
import config
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from pathlib import Path

# ============================================================
# TRAINING DATA PREPARATION
# ============================================================

def prepare_training_data(feature_table, feature_columns):
    """
    Separate X and y.

    Equivalent R:
        X <- scaled_features[1:7]
        y <- scaled_features$thickness
    """


    X = feature_table[feature_columns]
    y_class = feature_table["Layer"]
    y_thickness = feature_table["thickness"]
    return (X, y_class, y_thickness)

# ============================================================
# LDA CLASSIFIER
# ============================================================

def train_lda(X, y):
    """
    Train Linear Discriminant Analysis.

    Equivalent:
        lda(
            Layer ~ features
        )
    """

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)
    return model

# ============================================================
# LINEAR REGRESSION
# ============================================================

def train_linear_regression(X, y):
    """
    Equivalent:
        lm(
            thickness ~ features
        )
    """

    model = LinearRegression()
    model.fit(X, y)
    return model



# ============================================================
# RIDGE REGRESSION
# ============================================================

def train_ridge(X, y):
    """
    Equivalent:
        cv.glmnet(
            alpha=0
        )

    Uses cross-validation to select lambda.
    """

    model = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
    model.fit(X, y)
    return model

# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(X, y):
    """
    Equivalent:
        randomForest(
            ntree=500,
            mtry=2
        )
    """

    model = RandomForestRegressor(
        n_estimators=config.RANDOM_FOREST_TREES,
        max_features=config.RANDOM_FOREST_FEATURES,
        random_state=42
    )

    model.fit(X, y)
    return model

# ============================================================
# TRAIN EVERYTHING
# ============================================================

def train_all_models(feature_table, feature_columns):
    """
    Train all models.

    Returns dictionary.
    """

    X, y_class, y_thickness = prepare_training_data(
        feature_table,
        feature_columns
    )

    models = {}
    print("Training LDA...")
    models["lda"] = train_lda(X, y_class)

    print("Training Linear Regression...")
    models["linear"] = train_linear_regression(X, y_thickness)

    print("Training Ridge...")
    models["ridge"] = train_ridge(X, y_thickness)

    print("Training Random Forest...")
    models["random_forest"] = train_random_forest(X, y_thickness)
    return models

# ============================================================
# PREDICTION
# ============================================================

def predict_models(models, features, status_callback, status):
    """
    Apply models to new spectra.

    Equivalent R:
        predict(model, newdata)
    """
    status("Start probability prediction", status_callback)
    probs = models["lda"].predict_proba(features)
    status("Ended probability prediction", status_callback)

    predictions = pd.DataFrame()
    print(features)
    status("Start linear prediction", status_callback)
    predictions["linear_thickness"] = (models["linear"].predict(features))
    status("Start ridge prediction", status_callback)
    predictions["ridge_thickness"] = (models["ridge"].predict(features))
    status("Start rf prediction", status_callback)
    predictions["rf_thickness"] = (models["random_forest"].predict(features))
    status("Start lda class", status_callback)
    predictions["lda_class"] = (models["lda"].predict(features))
    status("Start enumerating", status_callback)
    for i, label in enumerate(models["lda"].classes_):
        predictions[f"lda_{label}_prob"] = probs[:, i]

    status("Start lda confidence", status_callback)
    predictions["lda_confidence"] = probs.max(axis=1)
    status("Finished model predictions", status_callback)
    return predictions

# ============================================================
# SAVE / LOAD
# ============================================================

def save_models(models, folder=config.MODEL_DIR):
    """
    Save trained models.
    """
    os.makedirs(folder, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"{folder}/{name}.pkl")

def load_models(folder=config.MODEL_DIR):
    """
    Load saved models.
    """
    models = {}

    folder = Path(folder)

    for file in folder.iterdir():
        if file.suffix == ".pkl":
            name = file.stem
            models[name] = joblib.load(file)

    return models