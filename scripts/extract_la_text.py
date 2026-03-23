#!/usr/bin/env python3
"""Extract text from LA ordinance PDFs and update the dataset.

Downloads each PDF, extracts text via pdfplumber, and updates the parquet.
Checkpoints every 100 rows. Skips rows that already have text.

Usage:
    python scripts/extract_la_text.py
    python scripts/extract_la_text.py --push
"""

from __future__ import annotations

import argparse
import io
import os
import random
import time
import warnings
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "la"
DATASET_PATH = CHECKPOINT_DIR / "la_final_dataset.parquet"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}


def _jittered_sleep(base: float):
    time.sleep(base * (0.7 + 0.6 * random.random()))


def extract_pdf_text(url: str, session: requests.Session) -> str:
    if not url:
        return ""
    try:
        resp = session.get(url, headers=HEADERS, timeout=60, verify=False)
        if resp.status_code != 200:
            return ""
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
            return "\n\n".join(pages)
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Extract text from LA ordinance PDFs")
    parser.add_argument("--push", action="store_true", help="Push updated dataset to HF")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print("Dataset not found. Run build_la_dataset.py first.")
        return

    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df)} rows")

    # Find rows needing text extraction
    needs_text = df[
        (df["pdf_url"].str.len() > 0) &
        ((df["text"].isna()) | (df["text"].str.len() == 0))
    ]
    print(f"Rows needing text: {needs_text.shape[0]}")
    already_done = (df["text"].str.len() > 0).sum()
    print(f"Already have text: {already_done}")

    if needs_text.empty:
        print("All rows already have text.")
        if args.push:
            _push(df)
        return

    session = requests.Session()
    extracted = 0
    failed = 0

    for i, (idx, row) in enumerate(needs_text.iterrows()):
        text = extract_pdf_text(row["pdf_url"], session)
        if text:
            df.at[idx, "text"] = text
            extracted += 1
        else:
            failed += 1

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(needs_text)} ({extracted} extracted, {failed} failed)")

        if (i + 1) % 100 == 0:
            df.to_parquet(DATASET_PATH, index=False)
            print(f"  [checkpoint] Saved")

        _jittered_sleep(args.delay)

    # Final save
    df.to_parquet(DATASET_PATH, index=False)

    total_with_text = (df["text"].str.len() > 0).sum()
    print(f"\nDone. Extracted: {extracted}, Failed: {failed}")
    print(f"Total with text: {total_with_text}/{len(df)}")

    if args.push:
        _push(df)


def _push(df: pd.DataFrame):
    from datasets import Dataset
    from huggingface_hub import HfApi

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    token = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("No HF token found.")
        return

    api = HfApi(token=token)
    api.whoami()
    ds = Dataset.from_pandas(df)
    ds.push_to_hub("chrissoria/la-ordinances", token=token, private=False)
    print(f"Pushed {len(df)} rows to HF")


if __name__ == "__main__":
    main()
