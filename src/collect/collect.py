"""
collect.py — Resumable, rate-limited Instagram public-profile collector.

Reads handles from data/raw/handles.csv, scrapes public profile metadata
and the last N posts via Instaloader, and appends results to
data/raw/profiles.jsonl (one JSON object per line).

Features:
  - Resumable: skips handles already in profiles.jsonl
  - Rate-limited: random delay between requests (configurable)
  - Exponential backoff on HTTP 429
  - Hard stop after N consecutive failures
  - Skips private accounts and accounts >= max_followers
  - Logs all errors to data/raw/errors.csv
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import instaloader
except ImportError:
    instaloader = None  # type: ignore[assignment]
    print("WARNING: instaloader not installed. Install with: pip install instaloader")

from src.collect.parse import parse_profile, parse_post

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
HANDLES_PATH = Path("data/raw/handles.csv")
PROFILES_PATH = Path("data/raw/profiles.jsonl")
ERRORS_PATH = Path("data/raw/errors.csv")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_already_scraped(profiles_path: Path) -> set[str]:
    """Read profiles.jsonl and return a set of already-scraped handles."""
    scraped: set[str] = set()
    if not profiles_path.exists():
        return scraped
    with open(profiles_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    scraped.add(record.get("handle", "").lower())
                except json.JSONDecodeError:
                    continue
    return scraped


def append_profile(record: dict, profiles_path: Path) -> None:
    """Append a single profile record as one JSON line."""
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profiles_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def log_error(handle: str, error_type: str, detail: str, errors_path: Path) -> None:
    """Append a failure to errors.csv."""
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = errors_path.exists()
    with open(errors_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["handle", "error_type", "detail", "timestamp"])
        writer.writerow([handle, error_type, detail, datetime.now(timezone.utc).isoformat()])


def log_skip(handle: str, reason: str, errors_path: Path) -> None:
    """Log a skipped handle."""
    log_error(handle, "SKIPPED", reason, errors_path)
    logger.info("  ⏭️  Skipped @%s — %s", handle, reason)


def fetch_profile_data(
    loader: Any,
    handle: str,
    max_followers: int,
    posts_to_fetch: int,
) -> Optional[dict]:
    """
    Fetch profile metadata and last N posts for a single handle.

    Returns a dict ready for JSONL output, or None if the profile should be skipped.
    Raises exceptions on network/rate-limit errors.
    """
    profile = instaloader.Profile.from_username(loader.context, handle)

    # --- Skip rules ---
    if profile.is_private:
        return None  # caller handles the skip log
    if profile.followers >= max_followers:
        return None

    # --- Parse profile metadata ---
    record = parse_profile(profile)

    # --- Fetch last N posts ---
    posts_data = []
    try:
        post_iter = profile.get_posts()
        for i, post in enumerate(post_iter):
            if i >= posts_to_fetch:
                break
            posts_data.append(parse_post(post))
    except Exception as e:
        logger.warning("  ⚠️  Post fetch partial failure for @%s: %s", handle, e)

    record["posts"] = posts_data
    record["posts_fetched"] = len(posts_data)
    record["scraped_at"] = datetime.now(timezone.utc).isoformat()

    return record


def run_collector(config_path: str = "config.yaml") -> None:
    """
    Main collection loop.

    Reads handles, skips already-done, scrapes each profile with rate limiting,
    and handles errors with exponential backoff.
    """
    if instaloader is None:
        logger.error("instaloader is not installed. Aborting.")
        sys.exit(1)

    config = load_config(config_path)
    scraping_cfg = config.get("scraping", {})

    min_delay = scraping_cfg.get("min_delay_seconds", 4)
    max_delay = scraping_cfg.get("max_delay_seconds", 9)
    max_consec_fail = scraping_cfg.get("max_consecutive_failures", 5)
    max_followers = scraping_cfg.get("max_followers", 1_000_000)
    posts_to_fetch = scraping_cfg.get("posts_to_fetch", 10)
    backoff_base = scraping_cfg.get("backoff_base_seconds", 60)
    backoff_max = scraping_cfg.get("backoff_max_seconds", 600)

    # --- Load handles ---
    if not HANDLES_PATH.exists():
        logger.error("Handles file not found: %s", HANDLES_PATH)
        sys.exit(1)

    import pandas as pd
    handles_df = pd.read_csv(HANDLES_PATH, dtype=str)
    handles_df["handle"] = handles_df["handle"].str.strip().str.lower().str.lstrip("@")
    total_handles = len(handles_df)

    # --- Load already scraped ---
    already_done = get_already_scraped(PROFILES_PATH)
    remaining = handles_df[~handles_df["handle"].isin(already_done)]
    logger.info(
        "📊 Total: %d | Already scraped: %d | Remaining: %d",
        total_handles, len(already_done), len(remaining),
    )

    if remaining.empty:
        logger.info("✅ All handles already scraped. Nothing to do.")
        return

    # --- Initialize Instaloader ---
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        quiet=True,
    )

    # --- Main scraping loop ---
    consecutive_failures = 0
    success_count = 0
    skip_count = 0
    fail_count = 0
    current_backoff = backoff_base

    for idx, row in remaining.iterrows():
        handle = row["handle"]
        niche = row.get("niche_hint", "Unknown")
        progress = f"[{success_count + skip_count + fail_count + 1}/{len(remaining)}]"

        logger.info("%s Scraping @%s (niche: %s)...", progress, handle, niche)

        try:
            result = fetch_profile_data(loader, handle, max_followers, posts_to_fetch)

            if result is None:
                # Check the reason for skip
                try:
                    profile = instaloader.Profile.from_username(loader.context, handle)
                    if profile.is_private:
                        log_skip(handle, "private account", ERRORS_PATH)
                    elif profile.followers >= max_followers:
                        log_skip(handle, f"followers={profile.followers} >= {max_followers}", ERRORS_PATH)
                    else:
                        log_skip(handle, "unknown reason", ERRORS_PATH)
                except Exception:
                    log_skip(handle, "could not determine reason", ERRORS_PATH)
                skip_count += 1
                consecutive_failures = 0
            else:
                result["niche_hint"] = niche
                append_profile(result, PROFILES_PATH)
                success_count += 1
                consecutive_failures = 0
                current_backoff = backoff_base  # reset backoff
                logger.info(
                    "  ✅ @%s — %d followers, %d posts fetched",
                    handle, result.get("follower_count", 0), result.get("posts_fetched", 0),
                )

        except instaloader.exceptions.QueryReturnedNotFoundException:
            log_error(handle, "NOT_FOUND", "Profile does not exist", ERRORS_PATH)
            fail_count += 1
            consecutive_failures += 1
            logger.warning("  ❌ @%s — Profile not found", handle)

        except instaloader.exceptions.ConnectionException as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                log_error(handle, "RATE_LIMIT", error_str[:200], ERRORS_PATH)
                logger.warning(
                    "  🛑 Rate limited! Backing off for %ds...", current_backoff
                )
                time.sleep(current_backoff)
                current_backoff = min(current_backoff * 2, backoff_max)
            else:
                log_error(handle, "CONNECTION_ERROR", error_str[:200], ERRORS_PATH)
                logger.warning("  ❌ @%s — Connection error: %s", handle, error_str[:100])
            fail_count += 1
            consecutive_failures += 1

        except instaloader.exceptions.ProfileNotExistsException:
            log_error(handle, "NOT_FOUND", "Profile does not exist", ERRORS_PATH)
            fail_count += 1
            consecutive_failures += 1
            logger.warning("  ❌ @%s — Profile does not exist", handle)

        except Exception as e:
            log_error(handle, "UNKNOWN", str(e)[:200], ERRORS_PATH)
            fail_count += 1
            consecutive_failures += 1
            logger.warning("  ❌ @%s — Unexpected error: %s", handle, str(e)[:100])

        # --- Hard stop check ---
        if consecutive_failures >= max_consec_fail:
            logger.error(
                "🛑 HARD STOP: %d consecutive failures. Stopping to prevent ban.",
                consecutive_failures,
            )
            break

        # --- Rate-limit delay ---
        delay = random.uniform(min_delay, max_delay)
        logger.debug("  💤 Sleeping %.1fs...", delay)
        time.sleep(delay)

    # --- Final summary ---
    logger.info("=" * 60)
    logger.info("  COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info("  ✅ Success:  %d", success_count)
    logger.info("  ⏭️  Skipped:  %d", skip_count)
    logger.info("  ❌ Failed:   %d", fail_count)
    logger.info("  📁 Output:   %s", PROFILES_PATH)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Instagram profile collector")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_collector(config_path=args.config)


if __name__ == "__main__":
    main()
