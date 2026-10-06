"""
discover.py — Handle discovery helpers.

Provides utility functions to build and validate the handles.csv file
from multiple sources (hashtag pages, Google, snowball sampling, directories).
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed handles for snowball sampling — 5 per niche
# Replace or extend these with real handles you've verified.
# ---------------------------------------------------------------------------
SEED_HANDLES: dict[str, list[str]] = {
    "Tech": [
        "techburner", "traabornindia", "igabornindia",
        "geeksforgeeks", "techbar",
    ],
    "Fashion": [
        "stylebyami", "thebowtieguide", "fashiondreamby",
        "indianfashionblogger", "stylecraze",
    ],
    "Fitness": [
        "fitness_guru_in", "sahilkhan", "yasabornindia",
        "crossfitindia", "yogawithadriene",
    ],
    "Lifestyle": [
        "mostlysane", "filtercopy", "beingsalmankhan",
        "dollysingh", "koabornindia",
    ],
    "Travel": [
        "travelandleisureindia", "tripotocommunity", "soulabornindia",
        "goibibo", "wanderlustindia",
    ],
}

# Hashtag search terms per niche (for manual or automated hashtag crawling)
NICHE_HASHTAGS: dict[str, list[str]] = {
    "Tech": [
        "techreviewer", "techindia", "gadgetreview",
        "techyoutuber", "techblogger",
    ],
    "Fashion": [
        "fashionblogger", "indianfashion", "ootdindia",
        "styleinfluencer", "fashionista",
    ],
    "Fitness": [
        "fitnessindia", "fitnesscoach", "gymlife",
        "yogaindia", "fitfam",
    ],
    "Lifestyle": [
        "lifestyleblogger", "indianblogger", "dailyvlog",
        "contentcreator", "lifestyleindia",
    ],
    "Travel": [
        "travelindia", "travelblogger", "wanderlust",
        "exploreindia", "incredibleindia",
    ],
}

# Google dork templates for handle discovery
GOOGLE_QUERIES: dict[str, list[str]] = {
    "Tech": [
        'site:instagram.com "tech reviewer" "collab"',
        'site:instagram.com "gadget" "review" india',
    ],
    "Fashion": [
        'site:instagram.com "fashion blogger" "DM for collabs"',
        'site:instagram.com "style" "OOTD" india',
    ],
    "Fitness": [
        'site:instagram.com "fitness coach" "collab"',
        'site:instagram.com "personal trainer" india',
    ],
    "Lifestyle": [
        'site:instagram.com "lifestyle blogger" india',
        'site:instagram.com "content creator" "collab"',
    ],
    "Travel": [
        'site:instagram.com "travel blogger" india',
        'site:instagram.com "wanderlust" "DM" collab',
    ],
}


def load_handles(filepath: str | Path) -> pd.DataFrame:
    """Load handles CSV, returning a DataFrame with 'handle' and 'niche_hint' columns."""
    path = Path(filepath)
    if not path.exists():
        logger.warning("Handles file not found: %s — returning empty DataFrame", path)
        return pd.DataFrame(columns=["handle", "niche_hint"])

    df = pd.read_csv(path, dtype=str)
    required = {"handle", "niche_hint"}
    if not required.issubset(df.columns):
        raise ValueError(f"handles.csv must have columns {required}, got {set(df.columns)}")

    df["handle"] = df["handle"].str.strip().str.lower().str.lstrip("@")
    df = df.drop_duplicates(subset="handle", keep="first").reset_index(drop=True)
    logger.info("Loaded %d unique handles from %s", len(df), path)
    return df


def save_handles(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save handles DataFrame to CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d handles to %s", len(df), path)


def validate_distribution(df: pd.DataFrame, min_per_niche: int = 400) -> dict[str, int]:
    """
    Check niche distribution and warn if any niche is under-represented.

    Returns a dict of {niche: count}.
    """
    dist = df["niche_hint"].value_counts().to_dict()
    total = len(df)
    logger.info("Handle distribution (total=%d):", total)
    for niche, count in sorted(dist.items()):
        status = "✅" if count >= min_per_niche else "⚠️ LOW"
        logger.info("  %s: %d %s", niche, count, status)
    return dist


def merge_handle_sources(
    *sources: list[tuple[str, str]],
    existing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge handles from multiple sources, deduplicate, and return a DataFrame.

    Each source is a list of (handle, niche_hint) tuples.
    If existing_df is provided, new handles are appended to it.
    """
    all_rows = []
    for source in sources:
        all_rows.extend(source)

    new_df = pd.DataFrame(all_rows, columns=["handle", "niche_hint"])
    new_df["handle"] = new_df["handle"].str.strip().str.lower().str.lstrip("@")

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset="handle", keep="first").reset_index(drop=True)
    logger.info("Merged sources: %d unique handles", len(combined))
    return combined


def print_discovery_guide() -> None:
    """Print a quick-reference guide for manual handle discovery."""
    print("=" * 70)
    print("  HANDLE DISCOVERY GUIDE")
    print("=" * 70)
    print()
    print("METHOD A — Hashtag Exploration (on Instagram)")
    for niche, tags in NICHE_HASHTAGS.items():
        print(f"  {niche}: {', '.join('#' + t for t in tags)}")
    print()
    print("METHOD B — Google Dork Searches")
    for niche, queries in GOOGLE_QUERIES.items():
        print(f"  {niche}:")
        for q in queries:
            print(f"    → {q}")
    print()
    print("METHOD C — Snowball from Seed Accounts")
    for niche, seeds in SEED_HANDLES.items():
        print(f"  {niche}: {', '.join('@' + s for s in seeds)}")
    print()
    print("METHOD D — Public Directories")
    print("  • HypeAuditor free listings")
    print("  • Social Blade")
    print("  • Blog posts: 'Top Indian [niche] influencers'")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print_discovery_guide()
