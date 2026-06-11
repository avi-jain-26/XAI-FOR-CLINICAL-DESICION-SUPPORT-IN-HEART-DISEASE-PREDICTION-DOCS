"""
Data loading, synthetic-data generation and preprocessing.

`load_data()` returns a raw DataFrame using the real Kaggle CSV if present,
otherwise a realistic synthetic dataset with the SAME schema (Table I of the
paper) so the project runs end-to-end with zero downloads.

`preprocess()` performs the steps described in the paper's Methodology
("Missing values are filled, categorical variables are transformed into numeric
representation, numerical values normalized, then split into train/test").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the real CSV if available, else generate synthetic data."""
    if cfg.CSV_PATH.exists():
        print(f"[data] Loading real dataset: {cfg.CSV_PATH}")
        df = pd.read_csv(cfg.CSV_PATH)
    else:
        print("[data] Real CSV not found -> generating synthetic dataset "
              f"({cfg.MAX_SAMPLES} rows). Drop the Kaggle file at\n"
              f"        {cfg.CSV_PATH}\n        to train on real data.")
        df = make_synthetic(cfg.MAX_SAMPLES)

    # Subsample large datasets so the SVM stays tractable for a demo.
    if len(df) > cfg.MAX_SAMPLES:
        df = df.sample(cfg.MAX_SAMPLES, random_state=cfg.RANDOM_STATE)
        print(f"[data] Subsampled to {len(df)} rows (see MAX_SAMPLES in config).")

    return df.reset_index(drop=True)


def make_synthetic(n: int) -> pd.DataFrame:
    """Create a synthetic dataset matching the Kaggle schema.

    The target is generated from a latent risk score so that clinically
    plausible features (age, smoking, stroke, diabetes, ...) actually drive the
    label -- this makes the trained models and the SHAP explanations meaningful.
    """
    rng = np.random.default_rng(cfg.RANDOM_STATE)

    age_idx = rng.integers(0, len(cfg.AGE_ORDER), n)            # 0..12
    bmi = np.clip(rng.normal(28, 6, n), 14, 60)
    smoking = rng.binomial(1, 0.41, n)
    alcohol = rng.binomial(1, 0.07, n)
    stroke = rng.binomial(1, 0.04, n)
    phys_health = np.round(rng.beta(0.6, 3.0, n) * 30).astype(int)
    ment_health = np.round(rng.beta(0.6, 3.0, n) * 30).astype(int)
    diff_walking = rng.binomial(1, 0.14, n)
    sex_male = rng.binomial(1, 0.50, n)
    race = rng.choice(cfg.RACE_CATEGORIES, n, p=[0.70, 0.10, 0.06, 0.09, 0.02, 0.03])
    diabetic = rng.binomial(1, 0.14, n)
    phys_activity = rng.binomial(1, 0.77, n)
    gen_health_ord = rng.choice(np.arange(5), n, p=[0.05, 0.12, 0.30, 0.35, 0.18])  # 0=Poor..4=Excellent
    sleep = np.clip(np.round(rng.normal(7.1, 1.4, n)), 2, 16).astype(int)
    asthma = rng.binomial(1, 0.13, n)
    kidney = rng.binomial(1, 0.04, n)
    skin_cancer = rng.binomial(1, 0.09, n)

    # Latent risk -> probability of heart disease
    z = (
        -4.7
        + 0.20 * age_idx
        + 0.045 * (bmi - 28)
        + 0.70 * smoking
        + 1.30 * stroke
        + 0.95 * diabetic
        + 0.85 * diff_walking
        + 0.55 * kidney
        + 0.020 * phys_health
        + 0.45 * sex_male
        - 0.30 * phys_activity
        - 0.28 * gen_health_ord          # healthier (higher) lowers risk
        + rng.normal(0, 1.0, n)          # noise
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    heart = (rng.random(n) < prob).astype(int)

    yn = np.array(["No", "Yes"])
    df = pd.DataFrame({
        "HeartDisease": yn[heart],
        "BMI": np.round(bmi, 2),
        "Smoking": yn[smoking],
        "AlcoholDrinking": yn[alcohol],
        "Stroke": yn[stroke],
        "PhysicalHealth": phys_health,
        "MentalHealth": ment_health,
        "DiffWalking": yn[diff_walking],
        "Sex": np.where(sex_male == 1, "Male", "Female"),
        "AgeCategory": np.array(cfg.AGE_ORDER)[age_idx],
        "Race": race,
        "Diabetic": yn[diabetic],
        "PhysicalActivity": yn[phys_activity],
        "GenHealth": np.array(cfg.GEN_HEALTH_ORDER)[gen_health_ord],
        "SleepTime": sleep,
        "Asthma": yn[asthma],
        "KidneyDisease": yn[kidney],
        "SkinCancer": yn[skin_cancer],
    })
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _yes_no_to_int(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.lower().str.startswith("yes")).astype(int)


def preprocess(df: pd.DataFrame):
    """Encode + scale + split. Returns DataFrames so SHAP keeps feature names.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    feature_names   : list[str]
    """
    df = df.copy()

    # Fill any missing values (numeric -> median, categorical -> mode).
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype.kind in "biufc":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Target: Yes/No -> 1/0
    y = _yes_no_to_int(df[cfg.TARGET])

    # Binary Yes/No features
    feat = pd.DataFrame(index=df.index)
    for col in cfg.BINARY_COLS:
        feat[col] = _yes_no_to_int(df[col])

    # Sex and Diabetic (Diabetic may have extra categories in real data)
    feat["Sex"] = (df["Sex"].astype(str).str.strip().str.lower() == "male").astype(int)
    feat["Diabetic"] = _yes_no_to_int(df["Diabetic"])

    # Ordinal encodings
    gen_map = {name: i for i, name in enumerate(cfg.GEN_HEALTH_ORDER)}
    age_map = {name: i for i, name in enumerate(cfg.AGE_ORDER)}
    feat["GenHealth"] = df["GenHealth"].map(gen_map).fillna(2).astype(int)   # default ~ "Good"
    feat["AgeCategory"] = df["AgeCategory"].map(age_map).fillna(6).astype(int)

    # Continuous numerics
    for col in cfg.NUMERIC_COLS:
        feat[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median()
                                                                   if df[col].dtype.kind in "biufc" else 0)

    # Nominal: one-hot encode Race
    race = pd.get_dummies(df["Race"].astype(str), prefix="Race", drop_first=True).astype(int)
    feat = pd.concat([feat, race], axis=1)

    feature_names = list(feat.columns)

    # Train/test split (stratified, like the paper's train/test methodology)
    X_train, X_test, y_train, y_test = train_test_split(
        feat, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )

    # Standardize continuous + ordinal columns (fit on TRAIN only -> no leakage).
    scale_cols = [c for c in cfg.SCALE_COLS if c in X_train.columns]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print(f"[data] Train={len(X_train)}  Test={len(X_test)}  Features={len(feature_names)}  "
          f"Positive rate={y.mean():.1%}")
    return X_train, X_test, y_train, y_test, feature_names
