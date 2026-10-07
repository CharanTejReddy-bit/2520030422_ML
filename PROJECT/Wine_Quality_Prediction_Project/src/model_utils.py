"""Core utilities for the Wine Quality Prediction project.

The project deliberately stays centered on the three algorithms from the
original proposal: Decision Tree, Random Forest and Gradient Boosting.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

QUALITY_THRESHOLD = 6.5
RANDOM_STATE = 42


def load_and_prepare(csv_path):
    df_raw = pd.read_csv(csv_path)
    raw_rows = len(df_raw)
    duplicate_count = int(df_raw.duplicated().sum())
    df = df_raw.drop_duplicates().reset_index(drop=True)
    missing_values = int(df.isna().sum().sum())
    X = df[FEATURES].copy()
    y = (df["quality"] >= QUALITY_THRESHOLD).astype(int)
    return df, X, y, {
        "raw_rows": raw_rows,
        "rows_after_duplicate_removal": len(df),
        "duplicates_removed": duplicate_count,
        "missing_values": missing_values,
    }


def build_baseline_models():
    """Exact-style baseline models retained for academic comparison."""
    return {
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(
                criterion="gini", max_depth=6, min_samples_split=5,
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=100, max_features="sqrt", oob_score=True,
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, subsample=0.8,
                random_state=RANDOM_STATE
            ))
        ])
    }


def build_tuning_spaces():
    """Search spaces keep the model family unchanged while allowing tuning."""
    return {
        "Decision Tree": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 4, 5, 6, 8, 10, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [None, 6, 10, 14, 18],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.7],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 150, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [1, 3, 5, 8],
            "model__subsample": [0.7, 0.8, 1.0],
        },
    }


def build_tuning_base_models():
    """Base estimators used by RandomizedSearchCV; still only the 3 core models."""
    return {
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
    }


def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
