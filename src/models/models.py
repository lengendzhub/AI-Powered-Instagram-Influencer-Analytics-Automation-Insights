"""
models.py — ML model training for engagement prediction and automation detection.

Model 1: Engagement rate regression
  - Compares Linear Regression, Random Forest, and Gradient Boosting
  - 5-fold cross-validation
  - Reports MAE, RMSE, R²
  - Saves feature importance plot

Model 2: Automation adoption classification
  - Random Forest and Logistic Regression (class_weight='balanced')
  - Reports precision, recall, F1 (not accuracy — imbalanced classes)
  - Saves confusion matrix

All models saved as .joblib, all metrics saved as .json.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/features.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model 1: Engagement Rate Prediction (Regression)
# ---------------------------------------------------------------------------

ENGAGEMENT_FEATURES = [
    "log_followers",
    "posts_per_month",
    "hashtag_diversity",
    "hashtag_entropy",
    "avg_caption_length",
    "avg_emoji_count",
    "has_cta",
    "has_link",
    "ff_ratio",
    "is_nano",
    "is_micro",
    "is_mid",
    "automation_flag",
]


def train_engagement_model(df: pd.DataFrame, config: dict) -> dict:
    """
    Train and compare engagement rate prediction models.

    Returns:
        Dict with best model name, metrics, and feature importances.
    """
    seed = config.get("random_seed", 42)
    test_size = config.get("models", {}).get("test_size", 0.2)
    n_estimators = config.get("models", {}).get("n_estimators", 200)
    learning_rate = config.get("models", {}).get("learning_rate", 0.1)
    cv_folds = config.get("models", {}).get("cv_folds", 5)

    logger.info("=" * 60)
    logger.info("  MODEL 1: Engagement Rate Prediction")
    logger.info("=" * 60)

    # --- Prepare data ---
    available_features = [f for f in ENGAGEMENT_FEATURES if f in df.columns]
    target = "engagement_rate"

    # Drop rows with NaN target or features
    df_model = df[available_features + [target]].dropna().copy()
    logger.info("Training data: %d rows, %d features", len(df_model), len(available_features))

    if len(df_model) < 50:
        logger.warning("Too few rows for engagement model (%d). Skipping.", len(df_model))
        return {}

    X = df_model[available_features].values
    y = df_model[target].values

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
    )

    # --- Scale features (fit on train only!) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Define models ---
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed,
        ),
    }

    # --- Train and evaluate ---
    results = {}
    best_model = None
    best_name = ""
    best_r2 = -np.inf

    for name, model in models.items():
        logger.info("  Training %s...", name)

        # Use scaled features for LinearRegression, raw for tree-based
        X_tr = X_train_scaled if name == "LinearRegression" else X_train
        X_te = X_test_scaled if name == "LinearRegression" else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Cross-validation R² on training set
        cv_scores = cross_val_score(
            model, X_tr, y_train, cv=cv_folds, scoring="r2",
        )
        cv_r2 = cv_scores.mean()

        results[name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "cv_r2_mean": round(cv_r2, 4),
            "cv_r2_std": round(cv_scores.std(), 4),
        }

        logger.info("    MAE=%.4f  RMSE=%.4f  R²=%.4f  CV-R²=%.4f±%.4f",
                     mae, rmse, r2, cv_r2, cv_scores.std())

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    logger.info("  Best model: %s (R² = %.4f)", best_name, best_r2)

    # --- Save best model and scaler ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "engagement_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "engagement_scaler.joblib")

    metrics_output = {
        "best_model": best_name,
        "features": available_features,
        "results": results,
    }
    with open(MODELS_DIR / "engagement_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    # --- Feature importance (for tree-based models) ---
    if hasattr(best_model, "feature_importances_"):
        save_feature_importance(
            best_model.feature_importances_,
            available_features,
            best_name,
            "engagement",
        )

    return metrics_output


# ---------------------------------------------------------------------------
# Model 2: Automation Adoption Prediction (Classification)
# ---------------------------------------------------------------------------

AUTOMATION_FEATURES = [
    "posting_hour_std",
    "hashtag_diversity",
    "hashtag_entropy",
    "has_cta",
    "avg_caption_length",
    "avg_emoji_count",
    "posts_per_month",
    "ff_ratio",
    "log_followers",
]


def train_automation_model(df: pd.DataFrame, config: dict) -> dict:
    """
    Train automation adoption classifier.

    IMPORTANT: The automation_flag label is a HEURISTIC, not ground truth.
    The model learns patterns in the heuristic signal.
    """
    seed = config.get("random_seed", 42)
    test_size = config.get("models", {}).get("test_size", 0.2)
    n_estimators = config.get("models", {}).get("n_estimators", 200)

    logger.info("=" * 60)
    logger.info("  MODEL 2: Automation Adoption Prediction")
    logger.info("=" * 60)
    logger.info("  ⚠️  NOTE: Labels are HEURISTIC, not ground truth.")

    # --- Prepare data ---
    available_features = [f for f in AUTOMATION_FEATURES if f in df.columns]
    target = "automation_flag"

    df_model = df[available_features + [target]].dropna().copy()
    logger.info("Training data: %d rows, %d features", len(df_model), len(available_features))

    if len(df_model) < 50:
        logger.warning("Too few rows for automation model (%d). Skipping.", len(df_model))
        return {}

    X = df_model[available_features].values
    y = df_model[target].astype(int).values

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info("  Class distribution: %s", dict(zip(unique, counts)))

    if len(unique) < 2:
        logger.warning("Only one class present. Cannot train classifier. Skipping.")
        return {}

    # --- Train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    # --- Define models (class_weight='balanced' for imbalance) ---
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
    }

    # --- Train and evaluate ---
    results = {}
    best_model = None
    best_name = ""
    best_f1 = -1.0

    for name, model in models.items():
        logger.info("  Training %s...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)

        results[name] = {
            "accuracy": round(report.get("accuracy", 0.0), 4),
            "macro_f1": round(macro_f1, 4),
            "precision_1": round(report.get("1", {}).get("precision", 0.0), 4),
            "recall_1": round(report.get("1", {}).get("recall", 0.0), 4),
            "f1_1": round(report.get("1", {}).get("f1-score", 0.0), 4),
        }

        logger.info("    Macro F1=%.4f  P(1)=%.4f  R(1)=%.4f  F1(1)=%.4f",
                     macro_f1,
                     results[name]["precision_1"],
                     results[name]["recall_1"],
                     results[name]["f1_1"])

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = model
            best_name = name

    logger.info("  Best model: %s (Macro F1 = %.4f)", best_name, best_f1)

    # --- Save ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "automation_model.joblib")

    metrics_output = {
        "best_model": best_name,
        "features": available_features,
        "results": results,
        "note": "Labels are heuristic-based, not ground truth.",
    }
    with open(MODELS_DIR / "automation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    # --- Confusion matrix ---
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    save_automation_confusion_matrix(cm, best_name)

    # --- Feature importance ---
    if hasattr(best_model, "feature_importances_"):
        save_feature_importance(
            best_model.feature_importances_,
            available_features,
            best_name,
            "automation",
        )

    return metrics_output


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    model_name: str,
    prefix: str,
) -> None:
    """Save feature importance bar chart as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_vals = importances[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(sorted_names)), sorted_vals[::-1], color="#4C72B0")
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {model_name} ({prefix})", fontsize=14)
        plt.tight_layout()

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = REPORTS_DIR / f"{prefix}_feature_importance.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Feature importance saved to %s", filepath)

    except ImportError:
        logger.warning("matplotlib not installed — skipping feature importance plot")


def save_automation_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Save automation classifier confusion matrix as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["No Automation (0)", "Automation (1)"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Oranges")
        ax.set_title(f"Automation Detection — {model_name}", fontsize=14)
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=16)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = REPORTS_DIR / "automation_confusion_matrix.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Automation confusion matrix saved to %s", filepath)

    except ImportError:
        logger.warning("matplotlib not installed — skipping confusion matrix plot")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_models_pipeline(config_path: str = "config.yaml") -> None:
    """Train both models and save all artifacts."""
    config = load_config(config_path)

    if not INPUT_PATH.exists():
        logger.error("Input file not found: %s", INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %d rows from %s", len(df), INPUT_PATH)

    # --- Model 1: Engagement prediction ---
    engagement_metrics = train_engagement_model(df, config)

    # --- Model 2: Automation prediction ---
    automation_metrics = train_automation_model(df, config)

    logger.info("=" * 60)
    logger.info("  ALL MODELS TRAINED SUCCESSFULLY")
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ML model training pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_models_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
