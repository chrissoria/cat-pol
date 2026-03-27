#!/usr/bin/env python3
"""Summarize Trump Truth Social images using alt-text format.

Processes all posts with images since 2024-11-05 (president-elect onwards)
and merges the summaries back into the dataset as an `image_alt_text` column.

Uses checkpointing to resume if interrupted.

Usage:
    python scripts/summarize_ts_images.py                # Run all
    python scripts/summarize_ts_images.py --batch-size 50 # Custom batch size
    python scripts/summarize_ts_images.py --dry-run       # Preview without running
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).parent
CHECKPOINT_PATH = SCRIPTS_DIR / "checkpoints" / "ts_image_alt_text.parquet"

load_dotenv(PROJECT_ROOT / ".env", override=True)

# Ensure fresh imports
for mod in list(sys.modules):
    if "cat_stack" in mod or "cat_pol" in mod:
        del sys.modules[mod]

import cat_pol as pol
from cat_pol.summarize import _download_urls, _is_url

TS_CACHE = SCRIPTS_DIR / ".ts_cache.parquet"
HF_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL = "qwen/qwen2.5-vl-72b-instruct:novita"


def load_checkpoint() -> dict:
    """Load existing checkpoint as {post_id: alt_text}."""
    if CHECKPOINT_PATH.exists():
        df = pd.read_parquet(CHECKPOINT_PATH)
        return dict(zip(df["post_id"].astype(str), df["image_alt_text"]))
    return {}


def save_checkpoint(results: dict):
    """Save checkpoint as parquet."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"post_id": pid, "image_alt_text": txt}
        for pid, txt in results.items()
    ])
    df.to_parquet(CHECKPOINT_PATH, index=False)


def summarize_batch(urls: list[str], batch_num: int, total_batches: int) -> list[str]:
    """Summarize a batch of image URLs. Returns list of alt-text strings."""
    print(f"\n[Batch {batch_num}/{total_batches}] Summarizing {len(urls)} images...")

    result = pol.summarize(
        input_data=urls,
        format="alt-text",
        tone=None,
        user_model=MODEL,
        model_source="huggingface",
        api_key=HF_KEY,
        description="Images posted by Donald Trump on Truth Social",
        thinking_budget=0,
        max_retries=2,
        batch_retries=0,
    )

    summaries = []
    for _, row in result.iterrows():
        s = row.get("summary", "")
        summaries.append(s if row.get("processing_status") == "success" else "")

    success = sum(1 for s in summaries if s)
    print(f"  -> {success}/{len(urls)} succeeded")
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Summarize Trump Truth Social images")
    parser.add_argument("--batch-size", type=int, default=25, help="Images per batch (default 25)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")
    args = parser.parse_args()

    if not HF_KEY:
        print("Error: HUGGINGFACE_API_KEY not found in .env")
        sys.exit(1)

    # Load dataset
    df = pd.read_parquet(TS_CACHE)
    since_elect = df[
        (df["date"] >= "2024-11-05") &
        (df["media_urls"].str.len() > 0) &
        (~df["media_urls"].str.lower().str.endswith(".mp4"))
    ].copy()
    print(f"Total images to process: {len(since_elect)}")

    # Load checkpoint
    completed = load_checkpoint()
    print(f"Already completed: {len(completed)}")

    # Filter to remaining
    remaining = since_elect[~since_elect["post_id"].astype(str).isin(completed)]
    print(f"Remaining: {len(remaining)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process these URLs:")
        for _, row in remaining.head(5).iterrows():
            print(f"  {row['date']} | {row['media_urls'][:80]}...")
        return

    if remaining.empty:
        print("All images already processed!")
    else:
        # Process in batches
        # For multi-image posts (semicolon-separated), take the first image URL (skip .mp4s)
        def _pick_image_url(media_str):
            for url in media_str.split(";"):
                url = url.strip()
                if url and not url.lower().endswith(".mp4"):
                    return url
            return media_str.split(";")[0].strip()  # fallback to first URL

        urls_list = [_pick_image_url(u) for u in remaining["media_urls"].tolist()]
        post_ids = remaining["post_id"].astype(str).tolist()
        total_batches = (len(urls_list) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(urls_list))
            batch_urls = urls_list[start:end]
            batch_ids = post_ids[start:end]

            summaries = summarize_batch(batch_urls, batch_idx + 1, total_batches)

            # Update checkpoint
            for pid, summary in zip(batch_ids, summaries):
                completed[pid] = summary
            save_checkpoint(completed)
            print(f"  Checkpoint saved ({len(completed)} total)")

    # Merge into dataset
    print(f"\nMerging {len(completed)} alt-text descriptions into dataset...")
    df["image_alt_text"] = df["post_id"].astype(str).map(completed).fillna("")

    # Update cache and push
    df.to_parquet(TS_CACHE, index=False)
    print(f"Updated local cache: {TS_CACHE}")

    from update_datasets import push_hf_dataset, setup_logging, TS_REPO
    log = setup_logging()
    push_hf_dataset(df, TS_REPO, TS_CACHE, log)
    print("Done! Dataset updated with image_alt_text column.")


if __name__ == "__main__":
    main()
