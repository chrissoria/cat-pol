#!/usr/bin/env python3
"""Incremental updater: fetch new San Diego ordinances/resolutions and append to HuggingFace.

Designed to run daily via launchd. On each run it:
1. Loads the current dataset from HuggingFace (cached locally)
2. Scrapes page 0, 1, ... until it hits a doc_num already in the dataset
3. Extracts PDF text for any new entries
4. Pushes the updated dataset back to HuggingFace

Usage:
    python scripts/update_san_diego.py              # Live run
    python scripts/update_san_diego.py --dry-run    # Scrape but don't push

Setup:
    1. pip install -r scripts/requirements.txt
    2. Set HUGGINGFACE_TOKEN in cat-pol/.env (needs write access)
    3. Install launchd plist (see scripts/install_launchd.sh)

Dependencies:
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load .env from cat-pol root (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Import scraper utilities from the build script
sys.path.insert(0, str(Path(__file__).parent))
from build_san_diego_dataset import (
    scrape_page,
    extract_pdf_text,
    _jittered_sleep,
    CHECKPOINT_DIR,
)

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent
LOG_DIR = SCRIPTS_DIR / "logs"
LAST_SCRAPE_PATH = SCRIPTS_DIR / ".last_scrape"
CACHE_PATH = SCRIPTS_DIR / ".dataset_cache.parquet"
REPO_ID = os.getenv("SD_DATASET_REPO", "chrissoria/san-diego-ordinances")
HF_TOKEN = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

PAGE_DELAY = 2.0
PDF_DELAY = 1.0
MAX_NEW_PAGES = 20  # safety cap — new docs rarely span more than a few pages

DATASET_COLUMNS = [
    "date", "doc_num", "doc_num_alt", "doc_type",
    "title", "text", "pdf_url", "year",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("sd_updater")
    log.setLevel(logging.DEBUG)

    # File handler — full debug
    fh = logging.FileHandler(LOG_DIR / "update.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    # Console handler — info+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(ch)

    return log


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_existing_dataset(log: logging.Logger) -> pd.DataFrame:
    """Load the current dataset from HuggingFace, with local caching."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets package not installed: pip install datasets")
        sys.exit(1)

    # Try local cache first (avoid hitting HF on every run during testing)
    if CACHE_PATH.exists():
        log.info(f"Loading dataset from local cache: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    log.info(f"Downloading dataset from HuggingFace: {REPO_ID}")
    try:
        ds = load_dataset(REPO_ID, split="train")
        df = ds.to_pandas()
        # Cache locally
        df.to_parquet(CACHE_PATH, index=False)
        log.info(f"Cached {len(df)} rows locally")
        return df
    except Exception as e:
        log.error(f"Failed to load dataset from HuggingFace: {e}")
        # If we have checkpoint data from the build script, use that
        final_cp = CHECKPOINT_DIR / "final_dataset.parquet"
        if final_cp.exists():
            log.info(f"Falling back to build checkpoint: {final_cp}")
            return pd.read_parquet(final_cp)
        log.error("No existing dataset found. Run build_san_diego_dataset.py first.")
        sys.exit(1)


def push_dataset(df: pd.DataFrame, log: logging.Logger) -> None:
    """Push updated dataset to HuggingFace."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        log.error("datasets/huggingface_hub not installed")
        sys.exit(1)

    if not HF_TOKEN:
        log.error("No HuggingFace token found. Set CATLLM_HUGGINGFACE_TOKEN or HUGGINGFACE_TOKEN in .env")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(REPO_ID, token=HF_TOKEN, private=False)
    log.info(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{REPO_ID}")

    # Update local cache
    df.to_parquet(CACHE_PATH, index=False)


# ---------------------------------------------------------------------------
# Incremental scraping
# ---------------------------------------------------------------------------


def fetch_new_entries(existing_doc_nums: set[str], log: logging.Logger) -> list[dict]:
    """Scrape pages starting from 0 until we hit known doc_nums.

    Returns a list of new row dicts with PDF text already extracted.
    """
    session = requests.Session()
    new_rows: list[dict] = []
    overlap_count = 0

    for page_num in range(MAX_NEW_PAGES):
        log.info(f"Scraping page {page_num}...")
        try:
            rows = scrape_page(page_num, session)
        except Exception as e:
            log.error(f"Page {page_num} failed: {e}")
            _jittered_sleep(PAGE_DELAY * 3)
            continue

        if not rows:
            log.info(f"Page {page_num} returned no rows — stopping")
            break

        page_new = []
        page_overlap = 0
        for row in rows:
            if row["doc_num"] in existing_doc_nums:
                page_overlap += 1
            else:
                page_new.append(row)

        log.info(f"  Page {page_num}: {len(page_new)} new, {page_overlap} already known")

        # Extract PDF text for new rows
        for i, row in enumerate(page_new):
            if row["pdf_url"]:
                log.debug(f"  Extracting PDF {i + 1}/{len(page_new)}: {row['doc_num']}")
                row["text"] = extract_pdf_text(row["pdf_url"], session)
                _jittered_sleep(PDF_DELAY)

        new_rows.extend(page_new)
        overlap_count += page_overlap

        # If the entire page is already known, we've caught up
        if page_overlap == len(rows):
            log.info("Full page overlap — caught up with existing dataset")
            break

        # If majority of page is known, likely caught up (handles boundary overlap)
        if page_overlap > len(rows) * 0.5 and page_num > 0:
            log.info("Majority overlap — caught up with existing dataset")
            break

        _jittered_sleep(PAGE_DELAY)

    return new_rows


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def save_last_scrape() -> None:
    """Record the current time as last successful scrape."""
    LAST_SCRAPE_PATH.write_text(datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_update(dry_run: bool = False) -> int:
    log = setup_logging()
    log.info("=" * 50)
    log.info(f"San Diego dataset updater starting ({'DRY RUN' if dry_run else 'LIVE'})")

    # Load existing dataset
    existing_df = load_existing_dataset(log)
    existing_doc_nums = set(existing_df["doc_num"].tolist())
    log.info(f"Existing dataset: {len(existing_df)} rows, {len(existing_doc_nums)} unique doc_nums")

    # Fetch new entries
    new_rows = fetch_new_entries(existing_doc_nums, log)

    if not new_rows:
        log.info("No new entries found. Dataset is up to date.")
        save_last_scrape()
        return 0

    new_df = pd.DataFrame(new_rows, columns=DATASET_COLUMNS)
    log.info(f"Found {len(new_df)} new entries:")
    log.info(f"  Ordinances: {(new_df['doc_type'] == 'ordinance').sum()}")
    log.info(f"  Resolutions: {(new_df['doc_type'] == 'resolution').sum()}")
    log.info(f"  Date range: {new_df['date'].min()} to {new_df['date'].max()}")
    log.info(f"  With text: {(new_df['text'].str.len() > 0).sum()}")

    if dry_run:
        log.info("DRY RUN — new entries found but not pushed:")
        for _, row in new_df.iterrows():
            log.info(f"  {row['date']} | {row['doc_num']} | {row['title'][:60]}")
        return 0

    # Merge and push
    updated_df = pd.concat([new_df, existing_df], ignore_index=True)
    updated_df = updated_df.drop_duplicates(subset=["doc_num"], keep="first")
    updated_df = updated_df.sort_values("date", ascending=False).reset_index(drop=True)

    log.info(f"Merged dataset: {len(updated_df)} total rows (was {len(existing_df)})")

    push_dataset(updated_df, log)
    save_last_scrape()

    log.info(f"Update complete. Added {len(new_df)} new entries.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Incremental San Diego dataset updater")
    parser.add_argument("--dry-run", action="store_true", help="Scrape but don't push to HuggingFace")
    args = parser.parse_args()
    sys.exit(run_update(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
