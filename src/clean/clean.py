"""
clean.py — Data cleaning and preprocessing pipeline.

Reads data/interim/influencers_v1.csv (or influencers_v1_validated.csv)
and applies a documented cleaning pipeline.

Every row dropped is logged to data/processed/cleaning_log.csv with the reason.

Output: data/processed/influencers_clean.csv
"""

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/interim/influencers_v1.csv")
INPUT_PATH_VALIDATED = Path("data/interim/influencers_v1_validated.csv")
OUTPUT_PATH = Path("data/processed/influencers_clean.csv")
CLEANING_LOG_PATH = Path("data/processed/cleaning_log.csv")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CleaningLog:
    """Tracks all cleaning actions for auditability."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.entries: list[dict] = []

    def add(self, handle: str, action: str, reason: str) -> None:
        """Record a cleaning action."""
        self.entries.append({
            "handle": handle,
            "action": action,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_bulk(self, handles: list[str], action: str, reason: str) -> None:
        """Record a cleaning action for multiple handles."""
        for handle in handles:
            self.add(handle, action, reason)

    def save(self) -> None:
        """Write the cleaning log to CSV."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.entries)
        df.to_csv(self.log_path, index=False)
        logger.info("Cleaning log saved to %s (%d entries)", self.log_path, len(self.entries))

    @property
    def count(self) -> int:
        return len(self.entries)


def step_deduplicate(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """STEP 1: Remove duplicate handles, keeping the first occurrence."""
    before = len(df)
    dupes = df[df.duplicated(subset="handle", keep="first")]
    if not dupes.empty:
        log.add_bulk(dupes["handle"].tolist(), "DROPPED", "duplicate handle")
    df = df.drop_duplicates(subset="handle", keep="first").reset_index(drop=True)
    dropped = before - len(df)
    logger.info("Step 1 — Deduplicate: dropped %d rows (had %d, now %d)", dropped, before, len(df))
    return df


def step_remove_bots_spam(df: pd.DataFrame, log: CleaningLog, config: dict) -> pd.DataFrame:
    """
    STEP 2: Remove bot and spam accounts based on engagement anomalies.

    Criteria:
      - High followers + very low engagement → likely bought followers
      - Very low followers + very high following → likely spam/follow-bots
      - Extremely high engagement → likely fake engagement
    """
    cleaning_cfg = config.get("cleaning", {})
    bot_min_followers = cleaning_cfg.get("bot_min_followers", 10000)
    bot_max_engagement = cleaning_cfg.get("bot_max_engagement", 0.1)
    spam_max_followers = cleaning_cfg.get("spam_max_followers", 100)
    spam_min_following = cleaning_cfg.get("spam_min_following", 5000)
    suspicious_engagement = cleaning_cfg.get("suspicious_engagement", 20.0)

    before = len(df)

    # Bot detection: high followers, very low engagement
    bot_mask = (
        (df["follower_count"] > bot_min_followers) &
        (df["engagement_rate"].notna()) &
        (df["engagement_rate"] < bot_max_engagement)
    )
    bot_handles = df.loc[bot_mask, "handle"].tolist()
    if bot_handles:
        log.add_bulk(bot_handles, "DROPPED",
                     f"suspected bot: followers>{bot_min_followers} and ER<{bot_max_engagement}%")

    # Spam detection: low followers, high following
    spam_mask = (
        (df["follower_count"] < spam_max_followers) &
        (df["following_count"] > spam_min_following)
    )
    spam_handles = df.loc[spam_mask, "handle"].tolist()
    if spam_handles:
        log.add_bulk(spam_handles, "DROPPED",
                     f"suspected spam: followers<{spam_max_followers} and following>{spam_min_following}")

    # Suspicious engagement
    suspicious_mask = (
        (df["engagement_rate"].notna()) &
        (df["engagement_rate"] > suspicious_engagement)
    )
    suspicious_handles = df.loc[suspicious_mask, "handle"].tolist()
    if suspicious_handles:
        log.add_bulk(suspicious_handles, "DROPPED",
                     f"suspicious engagement: ER>{suspicious_engagement}% (likely fake)")

    combined_mask = bot_mask | spam_mask | suspicious_mask
    df = df[~combined_mask].reset_index(drop=True)
    dropped = before - len(df)
    logger.info("Step 2 — Remove bots/spam: dropped %d rows (bots=%d, spam=%d, suspicious=%d)",
                dropped, len(bot_handles), len(spam_handles), len(suspicious_handles))
    return df


def step_normalize_followers(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """STEP 3: Ensure follower counts are clean integers."""
    # Convert to numeric, coercing errors
    df["follower_count"] = pd.to_numeric(df["follower_count"], errors="coerce")
    df["following_count"] = pd.to_numeric(df["following_count"], errors="coerce")

    # Remove any with negative or NaN follower counts
    bad_mask = df["follower_count"].isna() | (df["follower_count"] < 0)
    bad_handles = df.loc[bad_mask, "handle"].tolist()
    if bad_handles:
        log.add_bulk(bad_handles, "DROPPED", "invalid follower count (negative or NaN)")

    df = df[~bad_mask].reset_index(drop=True)
    df["follower_count"] = df["follower_count"].astype(int)
    df["following_count"] = df["following_count"].fillna(0).astype(int)

    logger.info("Step 3 — Normalize followers: dropped %d rows with invalid counts", len(bad_handles))
    return df


def step_winsorize_engagement(df: pd.DataFrame, log: CleaningLog, config: dict) -> pd.DataFrame:
    """STEP 4: Cap engagement rate outliers at the Nth percentile."""
    percentile = config.get("cleaning", {}).get("winsorize_percentile", 99)

    er_col = "engagement_rate"
    valid = df[er_col].dropna()
    if valid.empty:
        logger.info("Step 4 — Winsorize: no valid engagement rates to cap")
        return df

    cap_value = np.percentile(valid, percentile)
    affected = (df[er_col] > cap_value) & df[er_col].notna()
    n_affected = affected.sum()

    if n_affected > 0:
        log.add_bulk(
            df.loc[affected, "handle"].tolist(),
            "CAPPED",
            f"engagement_rate capped from >{cap_value:.2f}% to {cap_value:.2f}% (p{percentile})",
        )
        df.loc[affected, er_col] = cap_value

    logger.info("Step 4 — Winsorize: capped %d rows at %.2f%% (p%d)", n_affected, cap_value, percentile)
    return df


def step_normalize_hashtags(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """STEP 5: Normalize hashtags to lowercase, strip # prefix."""

    def clean_hashtags(val):
        if pd.isna(val) or val == "":
            return ""
        tags = [t.strip().lower().lstrip("#") for t in str(val).split(",")]
        tags = [t for t in tags if t]  # remove empties
        return ", ".join(tags)

    df["top_hashtags"] = df["top_hashtags"].apply(clean_hashtags)
    logger.info("Step 5 — Normalize hashtags: done")
    return df


def step_handle_missing(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    STEP 6: Handle missing values consistently.

    - contact_email: leave blank (empty string)
    - bio: replace NaN with empty string
    - engagement_rate: leave NaN (don't impute)
    - external_url: leave blank
    - NEVER fabricate data
    """
    str_cols = ["contact_email", "bio", "external_url", "display_name",
                "business_category", "automation_evidence", "top_hashtags"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    logger.info("Step 6 — Handle missing values: filled string NaNs with empty strings")
    return df


def step_assign_tiers(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """STEP 7: Re-assign follower tiers (in case data changed during cleaning)."""

    def tier(fc):
        if fc < 10_000:
            return "nano"
        elif fc < 100_000:
            return "micro"
        else:
            return "mid"

    df["follower_tier"] = df["follower_count"].apply(tier)
    logger.info("Step 7 — Assign tiers: %s", df["follower_tier"].value_counts().to_dict())
    return df


def run_cleaning(config_path: str = "config.yaml") -> None:
    """
    Execute the full cleaning pipeline.
    """
    config = load_config(config_path)

    # Use validated file if it exists, otherwise the raw features file
    input_path = INPUT_PATH_VALIDATED if INPUT_PATH_VALIDATED.exists() else INPUT_PATH
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return

    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)
    initial_count = len(df)

    log = CleaningLog(CLEANING_LOG_PATH)

    # Execute cleaning steps in order
    df = step_deduplicate(df, log)
    df = step_remove_bots_spam(df, log, config)
    df = step_normalize_followers(df, log)
    df = step_winsorize_engagement(df, log, config)
    df = step_normalize_hashtags(df, log)
    df = step_handle_missing(df, log)
    df = step_assign_tiers(df, log)

    # Save outputs
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.save()

    final_count = len(df)
    logger.info("=" * 60)
    logger.info("  CLEANING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Input rows:   %d", initial_count)
    logger.info("  Output rows:  %d", final_count)
    logger.info("  Dropped:      %d (%.1f%%)", initial_count - final_count,
                (initial_count - final_count) / initial_count * 100 if initial_count else 0)
    logger.info("  Log entries:  %d", log.count)
    logger.info("  Output:       %s", OUTPUT_PATH)
    logger.info("  Log:          %s", CLEANING_LOG_PATH)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Data cleaning pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_cleaning(config_path=args.config)


if __name__ == "__main__":
    main()
