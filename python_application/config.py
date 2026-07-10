"""
config.py

Configuration parameters for MoS2 Raman thickness analysis.
"""

# ============================================================
# FILE STRUCTURE
# ============================================================

TRAINING_DATA_DIR = "python_application/data/training_data/"
LAS_DATA_DIR = "python_application/data/default LAS/"
MODEL_DIR = "python_application/models"
PARAVIEW_OUTPUT = "python_application/data/results/analysis_results.csv"

# ============================================================
# LARGE AREA SCAN SETTINGS
# ============================================================

LAS_SIZE = 300
LAS_ROWS = LAS_SIZE
LAS_COLS = LAS_SIZE
TOTAL_SPECTRA = LAS_SIZE ** 2

# ============================================================
# NORMALIZATION
# ============================================================

NORMALIZATION_METHOD = "max"
NORMALIZATION_SCALE = 1000

# ============================================================
# RAMAN WINDOWS
# ============================================================

RAMAN_AXIS_COLUMN = 0

# Full fitting region
RAMAN_MIN = 375
RAMAN_MAX = 420

# Individual peak search windows
E2G_MIN = 375
E2G_MAX = 395

A1G_MIN = 395
A1G_MAX = 420

# ============================================================
# EXPECTED MoS2 PEAK VALUES
# ============================================================

E2G_EXPECTED = 390
A1G_EXPECTED = 410

# ============================================================
# GAUSSIAN FITTING
# ============================================================

INITIAL_SIGMA = 3
A_MIN = 0
A_MAX = 1100

SIGMA_MIN = 0.5
SIGMA_MAX = 20

MU1_MIN = 370
MU1_MAX = 400

MU2_MIN = 390
MU2_MAX = 430

MIN_R2 = 0.85

# ============================================================
# FEATURE ENGINEERING
# ============================================================

MODEL_FEATURES = [
    "diff_peak", "diff_fit",
    "A1", "A2", "area_ratio", "fwhm1", "fwhm2"
]

# ============================================================
# TRAINING LABELS
# ============================================================

LAYER_NAMES = [
    "background",
    "monolayer",
    "bilayer"
]

THICKNESS_MAP = {
    "background": 0,
    "monolayer": 0.7,
    "bilayer": 1.5
}

# ============================================================
# MACHINE LEARNING
# ============================================================

RANDOM_FOREST_TREES = 500
RANDOM_FOREST_FEATURES = 2
RIDGE_ALPHA = 0

# ============================================================
# PARALLEL PROCESSING
# ============================================================

RESERVE_CORES = 1
TRAINING_WORKERS = 1
PREDICTION_WORKERS = 6

# ============================================================
# OUTPUT / DEBUG
# ============================================================

VISUALIZATION_SCALE = 5
SAVE_INTERMEDIATE = True
VERBOSE = True
