"""
features.py — Phase 2 feature engineering pipeline.

Reads data/processed/influencers_clean.csv and computes advanced features
for ML modeling.

Output: data/processed/features.csv
"""

import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/influencers_clean.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature computation functions
# ---------------------------------------------------------------------------

def compute_hashtag_diversity(top_hashtags: str) -> float:
    """
    Compute hashtag diversity as unique_hashtags / total_hashtags.

    A value of 1.0 means all hashtags are unique (diverse);
    lower values indicate repetitive hashtag usage.
    """
    if not top_hashtags or pd.isna(top_hashtags):
        return 0.0
    tags = [t.strip() for t in str(top_hashtags).split(",") if t.strip()]
    if not tags:
        return 0.0
    unique = len(set(tags))
    total = len(tags)
    return unique / total


def compute_hashtag_entropy(top_hashtags: str) -> float:
    """
    Compute Shannon entropy of hashtag distribution.

    Higher entropy = more diverse hashtag usage.
    """
    if not top_hashtags or pd.isna(top_hashtags):
        return 0.0
    tags = [t.strip() for t in str(top_hashtags).split(",") if t.strip()]
    if not tags:
        return 0.0

    counter = Counter(tags)
    total = sum(counter.values())
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def count_emojis(text: str) -> int:
    """
    Count emoji characters in text.

    Uses a broad Unicode range pattern to match common emojis.
    """
    if not text or pd.isna(text):
        return 0
    # Broad emoji Unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "]+",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(str(text)))


def detect_cta(text: str) -> bool:
    """
    Detect call-to-action phrases in text.

    Checks for common CTA patterns like "link in bio", "swipe up",
    "DM for", "check out", etc.
    """
    if not text or pd.isna(text):
        return False
    text_lower = str(text).lower()
    cta_phrases = [
        "link in bio", "swipe up", "tap the link", "dm me", "dm for",
        "check out", "click the link", "shop now", "buy now",
        "limited time", "use code", "sign up", "subscribe",
        "comment below", "tag a friend", "share this",
    ]
    return any(phrase in text_lower for phrase in cta_phrases)


def detect_link(text: str) -> bool:
    """Detect URL patterns in text."""
    if not text or pd.isna(text):
        return False
    url_pattern = r"https?://\S+|www\.\S+"
    return bool(re.search(url_pattern, str(text)))


def compute_ff_ratio(followers: int, following: int) -> float:
    """
    Compute follower-to-following ratio.

    High ratio = celebrity-like (many followers, few following).
    Returns 0 if following is 0 to avoid division by zero.
    """
    if following <= 0:
        return 0.0
    return followers / following


def run_feature_engineering(config_path: str = "config.yaml") -> None:
    """
    Main feature engineering pipeline.

    Reads clean CSV, computes features, outputs features.csv.
    """
    config = load_config(config_path)

    if not INPUT_PATH.exists():
        logger.error("Input file not found: %s", INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %d rows from %s", len(df), INPUT_PATH)

    # --- Hashtag features ---
    logger.info("Computing hashtag features...")
    df["hashtag_diversity"] = df["top_hashtags"].apply(compute_hashtag_diversity)
    df["hashtag_entropy"] = df["top_hashtags"].apply(compute_hashtag_entropy)

    # --- Content type mix ---
    # These columns are computed in features_v1.py from the raw JSONL post data.
    # Here we just validate they exist and fill any gaps.
    logger.info("Validating content type features...")
    for col in ["pct_image", "pct_video", "pct_carousel"]:
        if col not in df.columns:
            logger.warning("  Column '%s' not found — setting to 0.0", col)
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # --- Caption statistics ---
    # avg_caption_length comes from features_v1 (computed from actual post captions).
    # Here we add emoji count, CTA, and link detection from the bio text.
    logger.info("Computing caption statistics...")
    if "avg_caption_length" not in df.columns:
        df["avg_caption_length"] = df["bio"].apply(
            lambda x: len(str(x)) if not pd.isna(x) else 0
        )
    df["avg_emoji_count"] = df["bio"].apply(count_emojis)
    df["has_cta"] = df["bio"].apply(detect_cta).astype(int)
    df["has_link"] = df["bio"].apply(detect_link).astype(int)

    # --- Follower-following ratio ---
    logger.info("Computing follower-following ratio...")
    df["ff_ratio"] = df.apply(
        lambda row: compute_ff_ratio(
            row.get("follower_count", 0),
            row.get("following_count", 0),
        ),
        axis=1,
    )

    # --- One-hot encode follower tiers ---
    logger.info("One-hot encoding follower tiers...")
    df["is_nano"] = (df["follower_tier"] == "nano").astype(int)
    df["is_micro"] = (df["follower_tier"] == "micro").astype(int)
    df["is_mid"] = (df["follower_tier"] == "mid").astype(int)

    # --- Log-transform followers (for modeling) ---
    df["log_followers"] = np.log1p(df["follower_count"])

    # --- Save ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Rows:    %d", len(df))
    logger.info("  Columns: %d", len(df.columns))
    logger.info("  Output:  %s", OUTPUT_PATH)
    logger.info("")
    logger.info("  New features added:")
    new_features = [
        "hashtag_diversity", "hashtag_entropy", "pct_image", "pct_video",
        "pct_carousel", "avg_caption_length", "avg_emoji_count",
        "has_cta", "has_link", "ff_ratio", "is_nano", "is_micro",
        "is_mid", "log_followers",
    ]
    for feat in new_features:
        if feat in df.columns:
            logger.info("    %s: mean=%.3f, std=%.3f",
                        feat, df[feat].mean(), df[feat].std())
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_feature_engineering(config_path=args.config)


if __name__ == "__main__":
    main()
