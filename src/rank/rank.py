"""
rank.py — Composite influencer ranking.

Computes a weighted influence score and ranks influencers
both overall and within each niche category.

Formula:
  score = w1 × norm(actual_engagement_rate)
        + w2 × norm(engagement_residual)
        + w3 × norm(log_followers)
        + w4 × norm(posting_consistency)
        + w5 × automation_adoption_probability

Weights are read from config.yaml.

Also performs a sensitivity analysis by perturbing weights.

Output: data/processed/ranked_influencers.csv
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/features.csv")
OUTPUT_PATH = Path("data/processed/ranked_influencers.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1], handling NaN gracefully."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_engagement_residual(df: pd.DataFrame) -> pd.Series:
    """
    Compute engagement residual = actual - predicted.

    Positive residual = influencer outperforms what their follower count
    would predict. This rewards genuine engagement over sheer size.

    Falls back to 0.5 if the engagement model is not available.
    """
    model_path = MODELS_DIR / "engagement_model.joblib"
    scaler_path = MODELS_DIR / "engagement_scaler.joblib"

    if not model_path.exists():
        logger.warning("Engagement model not found. Using 0.5 for all residuals.")
        return pd.Series(0.5, index=df.index)

    try:
        model = joblib.load(model_path)

        # Read feature list from metrics
        metrics_path = MODELS_DIR / "engagement_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            features = metrics.get("features", [])
        else:
            # Fallback feature list
            features = [
                "log_followers", "posts_per_month", "hashtag_diversity",
                "hashtag_entropy", "avg_caption_length", "avg_emoji_count",
                "has_cta", "has_link", "ff_ratio", "is_nano", "is_micro",
                "is_mid", "automation_flag",
            ]

        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0).values

        predicted = model.predict(X)
        actual = df["engagement_rate"].fillna(0).values
        residual = actual - predicted

        return pd.Series(residual, index=df.index)

    except Exception as e:
        logger.warning("Error computing engagement residual: %s. Using 0.5.", e)
        return pd.Series(0.5, index=df.index)


def compute_automation_probability(df: pd.DataFrame) -> pd.Series:
    """
    Get automation adoption probability from the trained classifier.

    Falls back to the binary automation_flag if model not available.
    """
    model_path = MODELS_DIR / "automation_model.joblib"

    if not model_path.exists():
        logger.warning("Automation model not found. Using automation_flag as probability.")
        return df.get("automation_flag", pd.Series(0.0, index=df.index)).astype(float)

    try:
        model = joblib.load(model_path)

        features = [
            "posting_hour_std", "hashtag_diversity", "hashtag_entropy",
            "has_cta", "avg_caption_length", "avg_emoji_count",
            "posts_per_month", "ff_ratio", "log_followers",
        ]
        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0).values

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X).astype(float)

        return pd.Series(proba, index=df.index)

    except Exception as e:
        logger.warning("Error computing automation probability: %s", e)
        return df.get("automation_flag", pd.Series(0.0, index=df.index)).astype(float)


def compute_posting_consistency(df: pd.DataFrame) -> pd.Series:
    """
    Compute posting consistency score.

    consistency = 1 / (1 + posting_hour_std)
    Higher = more consistent posting schedule.
    """
    if "posting_hour_std" not in df.columns:
        return pd.Series(0.5, index=df.index)

    std = df["posting_hour_std"].fillna(df["posting_hour_std"].median())
    return 1.0 / (1.0 + std)


def compute_influence_score(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute the composite influence score.

    Args:
        df: DataFrame with all required columns.
        weights: Dict with keys matching ranking_weights in config.yaml.

    Returns:
        Series of influence scores in [0, 1].
    """
    w_eng = weights.get("engagement", 0.40)
    w_res = weights.get("engagement_residual", 0.20)
    w_fol = weights.get("log_followers", 0.15)
    w_con = weights.get("posting_consistency", 0.15)
    w_aut = weights.get("automation_probability", 0.10)

    # Compute components
    engagement_norm = min_max_normalize(df["engagement_rate"].fillna(0))
    residual_norm = min_max_normalize(compute_engagement_residual(df))
    followers_norm = min_max_normalize(df.get("log_followers", np.log1p(df["follower_count"])))
    consistency_norm = min_max_normalize(compute_posting_consistency(df))
    automation_prob = compute_automation_probability(df)

    score = (
        w_eng * engagement_norm
        + w_res * residual_norm
        + w_fol * followers_norm
        + w_con * consistency_norm
        + w_aut * automation_prob
    )

    return score


def rank_influencers(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank_overall and rank_in_category columns."""
    # Overall ranking (higher score = better rank = lower number)
    df["rank_overall"] = df["influence_score"].rank(ascending=False, method="min").astype(int)

    # Per-category ranking
    niche_col = "predicted_niche" if "predicted_niche" in df.columns else "niche_hint"
    df["rank_in_category"] = (
        df.groupby(niche_col)["influence_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return df


def sensitivity_analysis(df: pd.DataFrame, base_weights: dict) -> str:
    """
    Perform sensitivity analysis by perturbing weights.

    Returns a markdown string with the analysis results.
    """
    perturbations = [
        {"engagement": 0.35, "engagement_residual": 0.25, "log_followers": 0.15,
         "posting_consistency": 0.15, "automation_probability": 0.10},
        {"engagement": 0.40, "engagement_residual": 0.20, "log_followers": 0.10,
         "posting_consistency": 0.20, "automation_probability": 0.10},
        {"engagement": 0.45, "engagement_residual": 0.15, "log_followers": 0.15,
         "posting_consistency": 0.15, "automation_probability": 0.10},
        {"engagement": 0.40, "engagement_residual": 0.20, "log_followers": 0.15,
         "posting_consistency": 0.15, "automation_probability": 0.10},
    ]

    # Get base top 10
    base_score = compute_influence_score(df, base_weights)
    base_top10 = set(df.nlargest(10, "influence_score" if "influence_score" in df.columns else df.columns[0])["handle"].values)

    # If influence_score already exists, use it
    if "influence_score" in df.columns:
        base_top10 = set(df.nlargest(10, "influence_score")["handle"].values)

    md = "# Sensitivity Analysis — Ranking Stability\n\n"
    md += "## Base Weights\n"
    md += f"```\n{json.dumps(base_weights, indent=2)}\n```\n\n"
    md += "## Base Top 10\n"
    md += "| Rank | Handle |\n|------|--------|\n"
    for i, handle in enumerate(base_top10, 1):
        md += f"| {i} | @{handle} |\n"
    md += "\n"

    md += "## Perturbation Results\n\n"
    md += "| Perturbation | Overlap with Base Top 10 | Stability |\n"
    md += "|-------------|--------------------------|----------|\n"

    for i, weights in enumerate(perturbations, 1):
        pert_score = compute_influence_score(df, weights)
        df_temp = df.copy()
        df_temp["_pert_score"] = pert_score
        pert_top10 = set(df_temp.nlargest(10, "_pert_score")["handle"].values)
        overlap = len(base_top10 & pert_top10)
        stability = "✅ Stable" if overlap >= 7 else "⚠️ Sensitive"
        md += f"| {i} | {overlap}/10 | {stability} |\n"

    md += "\n"
    md += "> **Interpretation:** If 7+ of 10 handles remain in the top 10 across\n"
    md += "> perturbations, the ranking is considered stable.\n"

    return md


def run_ranking(config_path: str = "config.yaml") -> None:
    """
    Main ranking pipeline.
    """
    config = load_config(config_path)
    weights = config.get("ranking_weights", {})

    if not INPUT_PATH.exists():
        logger.error("Input file not found: %s", INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %d rows from %s", len(df), INPUT_PATH)

    # --- Compute influence score ---
    logger.info("Computing influence scores...")
    df["influence_score"] = compute_influence_score(df, weights)

    # --- Rank ---
    logger.info("Ranking influencers...")
    df = rank_influencers(df)

    # --- Save ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # --- Sensitivity analysis ---
    logger.info("Running sensitivity analysis...")
    sensitivity_md = sensitivity_analysis(df, weights)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sensitivity_path = REPORTS_DIR / "sensitivity_analysis.md"
    with open(sensitivity_path, "w", encoding="utf-8") as f:
        f.write(sensitivity_md)

    # --- Summary ---
    niche_col = "predicted_niche" if "predicted_niche" in df.columns else "niche_hint"
    logger.info("=" * 60)
    logger.info("  RANKING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Total influencers ranked: %d", len(df))
    logger.info("  Output: %s", OUTPUT_PATH)
    logger.info("  Sensitivity: %s", sensitivity_path)
    logger.info("")
    logger.info("  Top 10 Overall:")
    top10 = df.nsmallest(10, "rank_overall")
    for _, row in top10.iterrows():
        logger.info("    #%d @%s (score=%.4f, %s, %d followers)",
                     row["rank_overall"], row["handle"],
                     row["influence_score"], row.get(niche_col, "?"),
                     row["follower_count"])
    logger.info("")
    logger.info("  Top 3 per category:")
    for niche in sorted(df[niche_col].dropna().unique()):
        niche_df = df[df[niche_col] == niche].nsmallest(3, "rank_in_category")
        handles = ", ".join(f"@{r['handle']}" for _, r in niche_df.iterrows())
        logger.info("    %s: %s", niche, handles)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Composite influencer ranking")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_ranking(config_path=args.config)


if __name__ == "__main__":
    main()
