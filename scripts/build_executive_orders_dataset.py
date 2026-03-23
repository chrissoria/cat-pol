#!/usr/bin/env python3
"""Scraper: Presidential Executive Orders from Federal Register API → HuggingFace.

The Federal Register API requires no authentication and provides full text
of all executive orders from 1994 to present.

Usage:
    python scripts/build_executive_orders_dataset.py
    python scripts/build_executive_orders_dataset.py --push-only

Dependencies:
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
import os
import re
import time
import random
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "executive_orders"
API_BASE = "https://www.federalregister.gov/api/v1"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

DATASET_COLUMNS = [
    "date",
    "title",
    "executive_order_number",
    "president",
    "document_number",
    "text",
    "html_url",
    "pdf_url",
    "signing_date",
]


def _jittered_sleep(base: float):
    time.sleep(base * (0.7 + 0.6 * random.random()))


def _strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_all_executive_orders() -> list[dict]:
    """Fetch all executive orders from Federal Register API."""
    all_results = []
    page = 1

    while True:
        print(f"  Fetching page {page}...", end=" ", flush=True)

        resp = requests.get(
            f"{API_BASE}/documents.json",
            params={
                "conditions[type]": "PRESDOCU",
                "conditions[presidential_document_type]": "executive_order",
                "per_page": 1000,
                "page": page,
                "fields[]": [
                    "title", "document_number", "html_url", "pdf_url",
                    "publication_date", "signing_date", "executive_order_number",
                    "president", "full_text_xml_url", "body_html_url",
                    "abstract", "executive_order_notes",
                ],
            },
            headers=HEADERS,
            timeout=60,
        )

        if resp.status_code == 429:
            print("rate limited, waiting 60s...")
            time.sleep(60)
            continue

        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        print(f"{len(results)} results")

        if not results:
            break

        all_results.extend(results)

        if not data.get("next_page_url"):
            break

        page += 1
        _jittered_sleep(1.0)

    return all_results


def fetch_full_text(doc: dict) -> str:
    """Fetch full text for an executive order."""
    # Try body_html_url first
    body_url = doc.get("body_html_url", "")
    if body_url:
        try:
            resp = requests.get(body_url, headers=HEADERS, timeout=60)
            if resp.status_code == 200:
                return _strip_html(resp.text)
        except Exception:
            pass

    # Fall back to full_text_xml_url
    xml_url = doc.get("full_text_xml_url", "")
    if xml_url:
        try:
            resp = requests.get(xml_url, headers=HEADERS, timeout=60)
            if resp.status_code == 200:
                return _strip_html(resp.text)
        except Exception:
            pass

    return ""


def build_dataset(extract_text: bool = True, text_delay: float = 0.5) -> pd.DataFrame:
    """Build the executive orders dataset."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching executive order listings from Federal Register...")
    docs = fetch_all_executive_orders()
    print(f"Total: {len(docs)} executive orders")

    rows = []
    for i, doc in enumerate(docs):
        president = doc.get("president", {})
        president_name = president.get("name", "") if isinstance(president, dict) else str(president)

        text = ""
        if extract_text:
            text = fetch_full_text(doc)
            _jittered_sleep(text_delay)

        rows.append({
            "date": doc.get("publication_date", ""),
            "title": doc.get("title", ""),
            "executive_order_number": str(doc.get("executive_order_number", "")),
            "president": president_name,
            "document_number": doc.get("document_number", ""),
            "text": text,
            "html_url": doc.get("html_url", ""),
            "pdf_url": doc.get("pdf_url", ""),
            "signing_date": doc.get("signing_date", ""),
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(docs)}")
            # Checkpoint
            cp_path = CHECKPOINT_DIR / f"eo_batch_{i + 1:05d}.parquet"
            pd.DataFrame(rows[-50:]).to_parquet(cp_path, index=False)

    # Final checkpoint
    if len(rows) % 50 != 0:
        cp_path = CHECKPOINT_DIR / f"eo_batch_{len(rows):05d}.parquet"
        pd.DataFrame(rows[-(len(rows) % 50):]).to_parquet(cp_path, index=False)

    df = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    df = df.drop_duplicates(subset=["document_number"], keep="first")
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


def push_to_huggingface(df: pd.DataFrame, repo_id: str) -> None:
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("datasets and huggingface_hub required")

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    token = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("No HuggingFace token found in .env")

    api = HfApi(token=token)
    api.whoami()

    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Build executive orders dataset")
    parser.add_argument("--no-text", action="store_true", help="Skip full text extraction")
    parser.add_argument("--text-delay", type=float, default=0.5)
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--repo-id", default="chrissoria/executive-orders")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.push_only:
        final = CHECKPOINT_DIR / "eo_final_dataset.parquet"
        if final.exists():
            df = pd.read_parquet(final)
        else:
            print("No dataset found.")
            return
    else:
        df = build_dataset(extract_text=not args.no_text, text_delay=args.text_delay)

    print(f"\n=== Executive Orders ===")
    print(f"Total: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Presidents: {df['president'].nunique()}")
    print(f"With text: {(df['text'].str.len() > 0).sum()}")

    final = CHECKPOINT_DIR / "eo_final_dataset.parquet"
    df.to_parquet(final, index=False)
    print(f"Saved to {final}")

    if args.push_only:
        push_to_huggingface(df, args.repo_id)
    else:
        push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
