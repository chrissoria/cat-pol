#!/usr/bin/env python3
"""One-time scraper: San Francisco ordinances → HuggingFace dataset.

Usage:
    # Scrape all years (1999-2026)
    python scripts/build_sf_dataset.py

    # Scrape specific years
    python scripts/build_sf_dataset.py --start-year 2020 --end-year 2026

    # Skip PDF text extraction
    python scripts/build_sf_dataset.py --no-text

    # Push existing checkpoint to HuggingFace
    python scripts/build_sf_dataset.py --push-only

Dependencies:
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
import io
import random
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BASE_URL = "https://www.sfbos.org"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

DATASET_COLUMNS = [
    "date", "file_number", "enactment_number", "doc_type",
    "title", "text", "pdf_url", "year",
]


def _jittered_sleep(base: float) -> None:
    """Sleep for base seconds ± 50% jitter."""
    time.sleep(base * random.uniform(0.5, 1.5))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_date(raw: str, year: int) -> str:
    """Convert M/D/YYYY or M/D/YY → YYYY-MM-DD."""
    raw = raw.strip()
    if not raw:
        return ""
    # Try various date formats
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            from datetime import datetime
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def scrape_year(year: int, session: requests.Session) -> list[dict]:
    """Scrape all ordinances for a given year from sfbos.org."""
    url = f"{BASE_URL}/ordinances-{year}"
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = []
    all_trs = table.find_all("tr")

    # Skip header row (first row contains column labels in <td>)
    for tr in all_trs[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        file_number = tds[0].get_text(strip=True)
        enactment_td = tds[1]
        date_raw = tds[2].get_text(strip=True)
        title = tds[3].get_text(strip=True)

        # Enactment number and PDF URL from <a> tag
        a_tag = enactment_td.find("a")
        if a_tag:
            enactment_number = a_tag.get_text(strip=True)
            pdf_path = a_tag.get("href", "")
            if pdf_path.startswith("/"):
                pdf_url = f"{BASE_URL}{pdf_path}"
            elif pdf_path.startswith("http"):
                pdf_url = pdf_path
            else:
                pdf_url = f"{BASE_URL}/{pdf_path}"
        else:
            enactment_number = enactment_td.get_text(strip=True)
            pdf_url = ""

        date = parse_date(date_raw, year)

        rows.append({
            "date": date,
            "file_number": file_number,
            "enactment_number": enactment_number,
            "doc_type": "ordinance",
            "title": title,
            "text": "",
            "pdf_url": pdf_url,
            "year": year,
        })

    return rows


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def extract_pdf_text(url: str, session: requests.Session) -> str:
    """Download a PDF and extract its text content."""
    if not url:
        return ""

    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required: pip install pdfplumber")

    try:
        resp = session.get(url, headers=HEADERS, timeout=60, verify=False)
        resp.raise_for_status()
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n\n".join(pages_text)
    except Exception as e:
        print(f"  [warn] PDF extraction failed for {url}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def scrape_all_years(
    start_year: int = 1999,
    end_year: int = 2026,
    extract_text: bool = True,
    year_delay: float = 2.0,
    pdf_delay: float = 1.0,
) -> pd.DataFrame:
    """Scrape ordinances for years [start_year, end_year]."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_rows: list[dict] = []

    for year in range(end_year, start_year - 1, -1):  # newest first
        print(f"Scraping {year}...")
        try:
            rows = scrape_year(year, session)
        except Exception as e:
            print(f"  [error] Year {year} failed: {e}")
            _jittered_sleep(year_delay * 2)
            continue

        if not rows:
            print(f"  No ordinances found for {year}")
            _jittered_sleep(year_delay)
            continue

        print(f"  Found {len(rows)} ordinances")

        # Extract PDF text
        if extract_text:
            for i, row in enumerate(rows):
                if row["pdf_url"]:
                    print(f"  Extracting PDF {i + 1}/{len(rows)}: {row['enactment_number']}")
                    row["text"] = extract_pdf_text(row["pdf_url"], session)
                    _jittered_sleep(pdf_delay)

        all_rows.extend(rows)

        # Checkpoint after each year
        cp_path = CHECKPOINT_DIR / f"sf_{year}.parquet"
        pd.DataFrame(rows).to_parquet(cp_path, index=False)
        print(f"  [checkpoint] Saved {len(rows)} rows for {year}")

        _jittered_sleep(year_delay)

    if not all_rows:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["enactment_number"], keep="first").reset_index(drop=True)
    return df


def merge_all_checkpoints() -> pd.DataFrame:
    """Merge all SF checkpoint files."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp_file in sorted(CHECKPOINT_DIR.glob("sf_*.parquet")):
        dfs.append(pd.read_parquet(cp_file))
    if not dfs:
        print("No SF checkpoint files found.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["enactment_number"], keep="first").reset_index(drop=True)
    print(f"Merged {len(df)} unique rows from {len(dfs)} checkpoint files")
    return df


def push_to_huggingface(df: pd.DataFrame, repo_id: str) -> None:
    """Push DataFrame to HuggingFace."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("datasets and huggingface_hub required: pip install datasets huggingface_hub")

    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    token = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("No HuggingFace token found in .env")

    api = HfApi(token=token)
    api.whoami()

    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Build San Francisco ordinances dataset")
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--no-text", action="store_true", help="Skip PDF text extraction")
    parser.add_argument("--year-delay", type=float, default=2.0)
    parser.add_argument("--pdf-delay", type=float, default=1.0)
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--repo-id", default="chrissoria/sf-ordinances")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.push_only:
        df = merge_all_checkpoints()
        if df.empty:
            print("Nothing to push.")
            return
    else:
        df = scrape_all_years(
            start_year=args.start_year,
            end_year=args.end_year,
            extract_text=not args.no_text,
            year_delay=args.year_delay,
            pdf_delay=args.pdf_delay,
        )

    print(f"\nTotal rows: {len(df)}")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")
    print(f"Rows with text: {(df['text'].str.len() > 0).sum()}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    final_path = CHECKPOINT_DIR / "sf_final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Final dataset saved to {final_path}")

    if args.push_only:
        push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
