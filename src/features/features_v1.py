"""
features_v1.py — Phase 1 derived feature computation.

Reads profiles.jsonl and computes:
  - Engagement rate (mean & median)
  - Posts per month
  - Contact email (regex from bio)
  - Top 5 hashtags
  - Automation flag + evidence
  - Follower tier

Output: data/interim/influencers_v1.csv
"""

import argparse
import json
import logging
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.collect.parse import (
    detect_automation_signals,
    extract_email,
    compute_posting_regularity,
)

logger = logging.getLogger(__name__)

PROFILES_PATH = Path("data/raw/profiles.jsonl")
OUTPUT_PATH = Path("data/interim/influencers_v1.csv")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_profiles(profiles_path: Path) -> list[dict]:
    """Load all profile records from JSONL."""
    records = []
    with open(profiles_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", line_num, e)
    logger.info("Loaded %d profile records from %s", len(records), profiles_path)
    return records


def compute_engagement_rate(
    posts: list[dict], follower_count: int
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute mean and median engagement rate across posts.

    engagement_rate_per_post = (likes + comments) / followers × 100

    Posts with null likes are skipped (not imputed).

    Returns:
        (mean_engagement_rate, median_engagement_rate) — both can be None
        if no valid posts.
    """
    if follower_count <= 0 or not posts:
        return None, None

    rates = []
    for post in posts:
        likes = post.get("likes")
        comments = post.get("comments", 0)
        if likes is None:
            continue  # Hidden likes — skip, don't impute
        rate = (likes + comments) / follower_count * 100
        rates.append(rate)

    if not rates:
        return None, None

    return statistics.mean(rates), statistics.median(rates)


def compute_posts_per_month(posts: list[dict]) -> Optional[float]:
    """
    Estimate posting frequency from the timestamps of the last N posts.

    posts_per_month = num_posts / (date_range_in_days / 30)
    """
    if not posts or len(posts) < 2:
        return None

    timestamps = []
    for post in posts:
        ts = post.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(dt)
            except (ValueError, AttributeError):
                continue

    if len(timestamps) < 2:
        return None

    timestamps.sort()
    date_range_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400

    if date_range_days <= 0:
        return None

    return len(timestamps) / (date_range_days / 30.0)


def compute_top_hashtags(posts: list[dict], top_n: int = 5) -> str:
    """
    Get the top N most frequent hashtags across all posts.

    Returns a comma-separated string.
    """
    all_tags: list[str] = []
    for post in posts:
        tags = post.get("hashtags", [])
        if isinstance(tags, list):
            all_tags.extend(t.lower().strip() for t in tags if t)

    if not all_tags:
        return ""

    counter = Counter(all_tags)
    top = counter.most_common(top_n)
    return ", ".join(tag for tag, _ in top)


def compute_content_type_mix(posts: list[dict]) -> tuple[float, float, float]:
    """
    Compute the share of image, video, and carousel posts.

    Returns:
        (pct_image, pct_video, pct_carousel) — each in [0, 1].
    """
    if not posts:
        return 0.0, 0.0, 0.0

    counts = {"image": 0, "video": 0, "carousel": 0}
    for post in posts:
        media_type = post.get("media_type", "image").lower()
        if "carousel" in media_type or "sidecar" in media_type:
            counts["carousel"] += 1
        elif "video" in media_type:
            counts["video"] += 1
        else:
            counts["image"] += 1

    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0, 0.0

    return counts["image"] / total, counts["video"] / total, counts["carousel"] / total


def compute_avg_caption_length(posts: list[dict]) -> float:
    """
    Compute the average caption length (character count) across posts.
    """
    lengths = [len(p.get("caption", "")) for p in posts if p.get("caption")]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def assign_follower_tier(follower_count: int) -> str:
    """
    Assign follower tier label.

    - nano:  < 10,000
    - micro: 10,000 – 99,999
    - mid:   100,000 – 999,999
    """
    if follower_count < 10_000:
        return "nano"
    elif follower_count < 100_000:
        return "micro"
    else:
        return "mid"


def process_single_profile(record: dict) -> dict:
    """
    Compute all derived features for a single profile record.

    Args:
        record: Raw profile dict from profiles.jsonl.

    Returns:
        Flat dict with all original + derived fields (ready for CSV row).
    """
    posts = record.get("posts", [])
    follower_count = record.get("follower_count", 0)
    bio = record.get("bio", "")

    # --- Engagement rate ---
    er_mean, er_median = compute_engagement_rate(posts, follower_count)

    # --- Posts per month ---
    ppm = compute_posts_per_month(posts)

    # --- Contact email ---
    email = extract_email(bio) or ""

    # --- Top hashtags ---
    top_hashtags = compute_top_hashtags(posts)

    # --- Automation detection ---
    captions = [p.get("caption", "") for p in posts if p.get("caption")]
    auto_flag, auto_evidence = detect_automation_signals(bio, captions)

    # --- Posting regularity ---
    timestamps = [p.get("timestamp", "") for p in posts if p.get("timestamp")]
    posting_hour_std = compute_posting_regularity(timestamps)

    # --- Content type mix ---
    pct_image, pct_video, pct_carousel = compute_content_type_mix(posts)

    # --- Average caption length ---
    avg_caption_len = compute_avg_caption_length(posts)

    # --- Follower tier ---
    tier = assign_follower_tier(follower_count)

    return {
        # Profile fields
        "handle": record.get("handle", ""),
        "display_name": record.get("display_name", ""),
        "follower_count": follower_count,
        "following_count": record.get("following_count", 0),
        "post_count": record.get("post_count", 0),
        "bio": bio,
        "external_url": record.get("external_url", ""),
        "business_category": record.get("business_category", ""),
        "is_private": record.get("is_private", False),
        "is_verified": record.get("is_verified", False),
        "niche_hint": record.get("niche_hint", "Unknown"),
        "scraped_at": record.get("scraped_at", ""),
        "posts_fetched": record.get("posts_fetched", 0),
        # Derived fields
        "engagement_rate": er_mean,
        "engagement_rate_median": er_median,
        "posts_per_month": ppm,
        "contact_email": email,
        "top_hashtags": top_hashtags,
        "automation_flag": int(auto_flag),
        "automation_evidence": "; ".join(auto_evidence) if auto_evidence else "",
        "posting_hour_std": posting_hour_std,
        "pct_image": pct_image,
        "pct_video": pct_video,
        "pct_carousel": pct_carousel,
        "avg_caption_length": avg_caption_len,
        "follower_tier": tier,
    }


def run_features_v1(config_path: str = "config.yaml") -> None:
    """
    Main entry point: load profiles.jsonl → compute features → save CSV.
    """
    config = load_config(config_path)

    if not PROFILES_PATH.exists():
        logger.error("Profiles file not found: %s", PROFILES_PATH)
        return

    records = load_profiles(PROFILES_PATH)
    if not records:
        logger.error("No records loaded. Aborting.")
        return

    logger.info("Computing derived features for %d profiles...", len(records))

    rows = []
    for i, record in enumerate(records):
        row = process_single_profile(record)
        rows.append(row)
        if (i + 1) % 500 == 0:
            logger.info("  Processed %d / %d", i + 1, len(records))

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    logger.info("=" * 60)
    logger.info("  FEATURE COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Rows:      %d", len(df))
    logger.info("  Columns:   %s", list(df.columns))
    logger.info("  Output:    %s", OUTPUT_PATH)
    logger.info("")
    logger.info("  Engagement rate coverage: %.1f%%",
                (1 - df["engagement_rate"].isna().mean()) * 100)
    logger.info("  Contact email found:      %d (%.1f%%)",
                (df["contact_email"] != "").sum(),
                (df["contact_email"] != "").mean() * 100)
    logger.info("  Automation flag set:       %d (%.1f%%)",
                df["automation_flag"].sum(),
                df["automation_flag"].mean() * 100)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute Phase 1 derived features")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_features_v1(config_path=args.config)


if __name__ == "__main__":
    main()
