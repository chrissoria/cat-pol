#!/usr/bin/env python3
"""One-time scraper: Los Angeles ordinances from LA City Clerk Connect → HuggingFace.

Queries the ColdFusion search at cityclerk.lacity.org with ordinance number
wildcard ranges (e.g., 152*, 153*, ..., 188*). Each query returns up to 500
results with ordinance number, council file, title, effective date, and PDF URL.

PDFs are scanned images — text extraction is skipped by default (use --extract-text
to attempt OCR, which requires additional setup).

Usage:
    python scripts/build_la_dataset.py
    python scripts/build_la_dataset.py --start 170 --end 189
    python scripts/build_la_dataset.py --push-only
    python scripts/build_la_dataset.py --merge-push

Dependencies:
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
import io
import os
import random
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "la"
SEARCH_URL = "https://cityclerk.lacity.org/lacityclerkconnect/index.cfm"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

DATASET_COLUMNS = [
    "date",
    "ordinance_number",
    "council_file",
    "doc_type",
    "title",
    "text",
    "pdf_url",
    "council_file_url",
]


def _jittered_sleep(base: float):
    time.sleep(base * (0.7 + 0.6 * random.random()))


def search_ordinances(query: str, session: requests.Session) -> list[dict]:
    """Search LA City Clerk for ordinances matching a wildcard query.

    Args:
        query: Ordinance number search (e.g., "188*" for all 188xxx)
        session: requests Session for connection reuse

    Returns:
        List of dicts with ordinance metadata
    """
    data = {
        "fa": "vord.doSearch",
        "s": "2",
        "DOCIDSearch": query,
        "StartDate": "",
        "EndDate": "",
    }

    for attempt in range(3):
        try:
            resp = session.post(
                f"{SEARCH_URL}?fa=vord.doSearch&s=2",
                data=data,
                headers=HEADERS,
                timeout=60,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                _jittered_sleep(5 * (attempt + 1))
            else:
                print(f"  [error] Search failed for {query}: {e}")
                return []

    soup = BeautifulSoup(resp.text, "html.parser")

    entries = []
    # Scan ALL tables for rows where the first cell starts with a digit
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            # Column 0: Ordinance number — must start with a digit
            ord_num = cells[0].get_text(strip=True)
            if not ord_num or not ord_num[0].isdigit():
                continue

            # Column 1: Council file number
            council_file = cells[1].get_text(strip=True)

            # Column 2: Title
            title = cells[2].get_text(strip=True)

            # Column 3: Effective date
            date_raw = cells[3].get_text(strip=True)

            # Parse date (M/D/YYYY → YYYY-MM-DD)
            date = ""
            try:
                from datetime import datetime
                dt = datetime.strptime(date_raw, "%m/%d/%Y")
                date = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                date = date_raw

            # Extract PDF URL
            pdf_url = ""
            pdf_link = cells[0].find("a")
            if pdf_link and pdf_link.get("href", "").endswith(".pdf"):
                pdf_url = pdf_link["href"]
                if not pdf_url.startswith("http"):
                    pdf_url = f"https://cityclerk.lacity.org{pdf_url}"

            # Extract council file URL
            cf_url = ""
            cf_link = cells[1].find("a")
            if cf_link and cf_link.get("href"):
                href = cf_link["href"]
                if not href.startswith("http"):
                    cf_url = f"https://cityclerk.lacity.org/lacityclerkconnect/{href}"
                else:
                    cf_url = href

            entries.append({
                "date": date,
                "ordinance_number": ord_num,
                "council_file": council_file,
                "doc_type": "ordinance",
                "title": title,
                "text": "",
                "pdf_url": pdf_url,
                "council_file_url": cf_url,
        })

    return entries


def extract_pdf_text(url: str, session: requests.Session) -> str:
    """Download a PDF and extract text (LA PDFs are often scanned images)."""
    if not url:
        return ""
    try:
        resp = session.get(url, headers=HEADERS, timeout=60, verify=False)
        resp.raise_for_status()
    except Exception:
        return ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
            return "\n\n".join(pages)
    except Exception:
        return ""


def _load_existing_ords() -> set:
    """Load ordinance numbers already scraped from checkpoints."""
    existing = set()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for cp_file in sorted(CHECKPOINT_DIR.glob("la_ord_*.parquet")):
        try:
            df = pd.read_parquet(cp_file, columns=["ordinance_number"])
            existing.update(df["ordinance_number"].astype(str).tolist())
        except Exception:
            pass
    return existing


def build_dataset(
    start_prefix: int = 1520,
    end_prefix: int = 1890,
    extract_text: bool = False,
    query_delay: float = 3.0,
    pdf_delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch all LA ordinances by querying wildcard ranges."""

    existing_ords = _load_existing_ords()
    if existing_ords:
        print(f"  Found {len(existing_ords)} existing ordinances in checkpoints, will skip")

    session = requests.Session()
    all_rows: list[dict] = []
    total_new = 0

    # Query by prefix: 152*, 153*, ..., 188*
    for prefix in range(start_prefix, end_prefix + 1):
        query = f"{prefix}*"
        print(f"Searching: {query}...", end=" ", flush=True)

        entries = search_ordinances(query, session)

        # Filter out already-scraped
        new_entries = [e for e in entries if e["ordinance_number"] not in existing_ords]

        if extract_text:
            for i, entry in enumerate(new_entries):
                if entry["pdf_url"]:
                    entry["text"] = extract_pdf_text(entry["pdf_url"], session)
                    _jittered_sleep(pdf_delay)

        all_rows.extend(new_entries)
        total_new += len(new_entries)
        print(f"{len(entries)} found, {len(new_entries)} new")

        # Checkpoint every 5 prefixes
        if (prefix - start_prefix + 1) % 5 == 0 and all_rows:
            cp_path = CHECKPOINT_DIR / f"la_ord_{prefix:05d}.parquet"
            pd.DataFrame(all_rows).to_parquet(cp_path, index=False)
            print(f"  [checkpoint] Saved {len(all_rows)} rows to {cp_path.name}")
            all_rows = []  # Reset for next batch

        _jittered_sleep(query_delay)

    # Final checkpoint
    if all_rows:
        cp_path = CHECKPOINT_DIR / f"la_ord_{end_prefix:05d}.parquet"
        pd.DataFrame(all_rows).to_parquet(cp_path, index=False)
        print(f"  [checkpoint] Saved {len(all_rows)} rows (final)")

    print(f"\nDone: {total_new} new ordinances scraped")

    # Merge all checkpoints
    return merge_all_checkpoints()


def merge_all_checkpoints() -> pd.DataFrame:
    """Merge all LA checkpoint files."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp_file in sorted(CHECKPOINT_DIR.glob("la_ord_*.parquet")):
        dfs.append(pd.read_parquet(cp_file))
    if not dfs:
        print("No LA checkpoint files found.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["ordinance_number"], keep="first").reset_index(drop=True)
    print(f"Merged {len(df)} unique rows from {len(dfs)} checkpoint files")
    return df


def push_to_huggingface(df: pd.DataFrame, repo_id: str) -> None:
    """Push DataFrame to HuggingFace."""
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
    parser = argparse.ArgumentParser(description="Build LA ordinances dataset")
    parser.add_argument("--start", type=int, default=1520, help="Start ordinance prefix (default: 1520 for ~1979)")
    parser.add_argument("--end", type=int, default=1890, help="End ordinance prefix (default: 1890)")
    parser.add_argument("--extract-text", action="store_true", help="Attempt PDF text extraction (most LA PDFs are scanned)")
    parser.add_argument("--query-delay", type=float, default=3.0)
    parser.add_argument("--pdf-delay", type=float, default=1.0)
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--merge-push", action="store_true")
    parser.add_argument("--repo-id", default="chrissoria/la-ordinances")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.push_only or args.merge_push:
        df = merge_all_checkpoints()
        if df.empty:
            print("Nothing to push.")
            return
    else:
        df = build_dataset(
            start_prefix=args.start,
            end_prefix=args.end,
            extract_text=args.extract_text,
            query_delay=args.query_delay,
            pdf_delay=args.pdf_delay,
        )

    print(f"\n=== Los Angeles Ordinances ===")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rows with text: {(df['text'].str.len() > 0).sum()}")
    print(f"Rows with PDF URL: {(df['pdf_url'].str.len() > 0).sum()}")

    final_path = CHECKPOINT_DIR / "la_final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Final dataset saved to {final_path}")

    if args.output:
        df.to_csv(args.output, index=False)

    if args.push_only or args.merge_push:
        push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
