"""
run_pipeline.py — Single entry point for the Phase 2 analysis pipeline.

Usage:
    python run_pipeline.py [--config config.yaml] [--skip-clean] [--skip-nlp]

Runs the following modules in order:
    1. src/clean/clean.py       — Data cleaning
    2. src/features/features.py — Feature engineering
    3. src/models/nlp.py        — NLP niche classification
    4. src/models/models.py     — ML model training
    5. src/rank/rank.py         — Composite ranking

Each step reads from the previous step's output.
To launch the dashboard after: streamlit run app/dashboard.py
"""

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config.yaml", skip_clean: bool = False, skip_nlp: bool = False) -> None:
    """Run the full Phase 2 pipeline."""

    steps = []

    if not skip_clean:
        steps.append(("Step 1/5: Data Cleaning", "src.clean.clean", "run_cleaning"))
    steps.append(("Step 2/5: Feature Engineering", "src.features.features", "run_feature_engineering"))
    if not skip_nlp:
        steps.append(("Step 3/5: NLP Pipeline", "src.models.nlp", "run_nlp_pipeline"))
    steps.append(("Step 4/5: ML Models", "src.models.models", "run_models_pipeline"))
    steps.append(("Step 5/5: Ranking", "src.rank.rank", "run_ranking"))

    total = len(steps)
    logger.info("=" * 70)
    logger.info("  INSTAGRAM INFLUENCER ANALYTICS — FULL PIPELINE")
    logger.info("  Config: %s", config_path)
    logger.info("  Steps:  %d", total)
    logger.info("=" * 70)

    for i, (name, module_path, func_name) in enumerate(steps, 1):
        logger.info("")
        logger.info("━" * 70)
        logger.info("  %s", name)
        logger.info("━" * 70)

        start = time.time()
        try:
            # Dynamically import and run
            module = __import__(module_path, fromlist=[func_name])
            func = getattr(module, func_name)
            func(config_path=config_path)
            elapsed = time.time() - start
            logger.info("  ✅ %s completed in %.1fs", name, elapsed)
        except Exception as e:
            elapsed = time.time() - start
            logger.error("  ❌ %s failed after %.1fs: %s", name, elapsed, e)
            logger.error("  Stopping pipeline.")
            sys.exit(1)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  ✅ PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Outputs:")
    logger.info("    data/processed/influencers_clean.csv")
    logger.info("    data/processed/features.csv")
    logger.info("    data/processed/ranked_influencers.csv")
    logger.info("    models/*.joblib")
    logger.info("    reports/confusion_matrix.png")
    logger.info("    reports/sensitivity_analysis.md")
    logger.info("")
    logger.info("  To launch the dashboard:")
    logger.info("    streamlit run app/dashboard.py")
    logger.info("=" * 70)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the full Instagram Influencer Analytics pipeline",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--skip-clean", action="store_true",
        help="Skip the cleaning step (use existing clean data)",
    )
    parser.add_argument(
        "--skip-nlp", action="store_true",
        help="Skip the NLP step (use existing niche predictions)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        config_path=args.config,
        skip_clean=args.skip_clean,
        skip_nlp=args.skip_nlp,
    )


if __name__ == "__main__":
    main()
