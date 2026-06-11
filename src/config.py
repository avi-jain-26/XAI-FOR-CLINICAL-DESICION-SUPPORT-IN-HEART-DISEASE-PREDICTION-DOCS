"""
Central configuration: paths, feature definitions and run constants.

The feature set mirrors Table I of the paper ("Heart Disease Features"), which
is the CDC 2020 BRFSS "Personal Key Indicators of Heart Disease" schema used on
Kaggle (file: heart_2020_cleaned.csv).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Place the real Kaggle CSV here to run on real data instead of synthetic data.
# Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
CSV_PATH = DATA_DIR / "heart_2020_cleaned.csv"

# ---------------------------------------------------------------------------
# Run constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5            # cross-validation folds used by the stacking ensemble

# The real dataset has ~320k rows. SVM (RBF) scales poorly with row count, so we
# subsample for a tractable, runnable demo. Raise this if you have time/compute.
MAX_SAMPLES = 6000

# Number of test rows used when computing SHAP values (kept small for speed).
SHAP_SAMPLE_SIZE = 400

# ---------------------------------------------------------------------------
# Column / feature definitions  (see Table I of the paper)
# ---------------------------------------------------------------------------
TARGET = "HeartDisease"

# Yes/No -> 1/0
BINARY_COLS = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "PhysicalActivity",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

# Continuous numeric features (standardized before modelling)
NUMERIC_COLS = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# Ordinal categorical features (encoded to ordered integers, then standardized)
# Order goes from healthiest -> least healthy is encoded high -> low here.
GEN_HEALTH_ORDER = ["Poor", "Fair", "Good", "Very good", "Excellent"]  # 0..4

AGE_ORDER = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older",
]  # 0..12

# Nominal categorical feature (one-hot encoded)
RACE_CATEGORIES = [
    "White", "Black", "Asian", "Hispanic",
    "American Indian/Alaskan Native", "Other",
]

# Columns standardized with StandardScaler (continuous + ordinal).
SCALE_COLS = NUMERIC_COLS + ["AgeCategory", "GenHealth"]
