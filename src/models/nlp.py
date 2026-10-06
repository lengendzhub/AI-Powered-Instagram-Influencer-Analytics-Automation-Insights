"""
nlp.py — NLP pipeline for caption analysis and niche classification.

Preprocesses bio + caption text, builds TF-IDF features, and trains a
niche classifier. Reports classification metrics and saves a confusion matrix.

Outputs:
  - models/niche_classifier.joblib
  - models/tfidf_vectorizer.joblib
  - models/niche_classification_report.json
  - reports/confusion_matrix.png
  - Updated features.csv with predicted_niche column
"""

import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/features.csv")
OUTPUT_PATH = Path("data/processed/features.csv")  # Updated in-place
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for NLP.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove @mentions
      4. Remove special characters (keep alphanumeric and spaces)
      5. Collapse whitespace
    """
    if not text or pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+", "", text)          # Remove URLs
    text = re.sub(r"@\w+", "", text)                   # Remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)              # Remove # but keep the word
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)       # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()           # Collapse whitespace

    return text


def build_document(row: pd.Series) -> str:
    """
    Combine bio and hashtag text into a single document per influencer.

    In a full implementation, this would include all captions from posts.
    Here we use bio + top_hashtags as the primary text signal.
    """
    parts = []

    bio = row.get("bio", "")
    if bio and not pd.isna(bio):
        parts.append(str(bio))

    hashtags = row.get("top_hashtags", "")
    if hashtags and not pd.isna(hashtags):
        # Convert comma-separated hashtags to space-separated words
        parts.append(str(hashtags).replace(",", " "))

    # Add business category if available
    biz_cat = row.get("business_category", "")
    if biz_cat and not pd.isna(biz_cat):
        parts.append(str(biz_cat))

    return preprocess_text(" ".join(parts))


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_niche_classifier(
    df: pd.DataFrame,
    config: dict,
) -> tuple:
    """
    Train a niche classifier using TF-IDF + LogisticRegression/LinearSVC.

    Steps:
      1. Build text documents from bio + hashtags
      2. Split data (train/test, stratified)
      3. Fit TF-IDF on training split ONLY (no data leakage!)
      4. Train Logistic Regression and Linear SVM
      5. Pick the better model based on macro F1
      6. Save models, vectorizer, and metrics

    Returns:
        (best_model, vectorizer, classification_report_dict)
    """
    seed = config.get("random_seed", 42)
    test_size = config.get("models", {}).get("test_size", 0.2)
    max_features = config.get("features", {}).get("tfidf_max_features", 5000)
    ngram_range = tuple(config.get("features", {}).get("tfidf_ngram_range", [1, 2]))

    # --- Build text documents ---
    logger.info("Building text documents...")
    df["text_document"] = df.apply(build_document, axis=1)

    # Filter out rows with empty documents or missing niche labels
    mask = (df["text_document"].str.len() > 0) & (df["niche_hint"].notna())
    df_nlp = df[mask].copy()
    logger.info("NLP dataset: %d rows (excluded %d with empty text or missing niche)",
                len(df_nlp), len(df) - len(df_nlp))

    X_text = df_nlp["text_document"].values
    y = df_nlp["niche_hint"].values

    # --- Train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # --- TF-IDF vectorization (fit on TRAIN only!) ---
    logger.info("Fitting TF-IDF vectorizer (max_features=%d, ngram_range=%s)...",
                max_features, ngram_range)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- Train models ---
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed, multi_class="multinomial",
        ),
        "LinearSVC": LinearSVC(
            C=1.0, max_iter=2000, random_state=seed,
        ),
    }

    best_model = None
    best_name = ""
    best_f1 = -1.0
    best_report = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
        logger.info("  %s — Macro F1: %.4f, Accuracy: %.4f",
                     name, macro_f1, report.get("accuracy", 0.0))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = model
            best_name = name
            best_report = report

    logger.info("Best model: %s (Macro F1 = %.4f)", best_name, best_f1)

    # --- Save confusion matrix ---
    y_pred_best = best_model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred_best)
    save_confusion_matrix(cm, sorted(set(y)), best_name)

    return best_model, vectorizer, best_report, best_name


def save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    model_name: str,
) -> None:
    """Save confusion matrix as a PNG image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = REPORTS_DIR / "confusion_matrix.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Confusion matrix saved to %s", filepath)

    except ImportError:
        logger.warning("matplotlib not installed — skipping confusion matrix plot")


def run_nlp_pipeline(config_path: str = "config.yaml") -> None:
    """
    Main NLP pipeline entry point.

    Loads features.csv, trains niche classifier, saves models and metrics,
    adds predicted_niche column back to features.csv.
    """
    config = load_config(config_path)

    if not INPUT_PATH.exists():
        logger.error("Input file not found: %s", INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %d rows from %s", len(df), INPUT_PATH)

    # --- Train classifier ---
    best_model, vectorizer, report, model_name = train_niche_classifier(df, config)

    # --- Save models ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "niche_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    report_path = MODELS_DIR / "niche_classification_report.json"

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved vectorizer to %s", vectorizer_path)
    logger.info("Saved report to %s", report_path)

    # --- Add predictions to full dataset ---
    logger.info("Predicting niches for full dataset...")
    df["text_document"] = df.apply(build_document, axis=1)
    X_all_tfidf = vectorizer.transform(df["text_document"].values)
    df["predicted_niche"] = best_model.predict(X_all_tfidf)

    # Drop the temporary text_document column
    df = df.drop(columns=["text_document"])

    # Save updated features.csv
    df.to_csv(OUTPUT_PATH, index=False)

    logger.info("=" * 60)
    logger.info("  NLP PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("  Best model:     %s", model_name)
    logger.info("  Macro F1:       %.4f", report.get("macro avg", {}).get("f1-score", 0))
    logger.info("  Accuracy:       %.4f", report.get("accuracy", 0))
    logger.info("")
    logger.info("  Per-class F1:")
    for label in sorted(set(df["niche_hint"].dropna())):
        f1 = report.get(label, {}).get("f1-score", 0)
        logger.info("    %s: %.4f", label, f1)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="NLP pipeline — niche classification")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_nlp_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
