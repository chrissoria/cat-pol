#!/usr/bin/env python3
"""One-time scraper: San Diego city ordinances & resolutions → HuggingFace dataset.

Usage:
    # Scrape first 5 pages (test run)
    python scripts/build_san_diego_dataset.py --end 5

    # Scrape all pages
    python scripts/build_san_diego_dataset.py

    # Resume from page 500 (picks up existing checkpoints)
    python scripts/build_san_diego_dataset.py --start 500

    # Push to HuggingFace after scraping
    python scripts/build_san_diego_dataset.py --push-only

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

# docs.sandiego.gov has SSL cert issues; suppress warnings for verify=False
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BASE_URL = "https://www.sandiego.gov/city-clerk/officialdocs/council-resolutions-ordinances"
PDF_BASE = "http://docs.sandiego.gov/council_reso_ordinance"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}


def _jittered_sleep(base: float) -> None:
    """Sleep for base seconds ± 50% jitter."""
    time.sleep(base * random.uniform(0.5, 1.5))

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_doc_num(raw: str) -> tuple[str, str | None, str]:
    """Parse raw doc-num cell into (doc_num, doc_num_alt, doc_type).

    Examples:
        "R-316678/ R-2026-341"  → ("R-316678", "R-2026-341", "resolution")
        "O-22072/ O-2026-93"    → ("O-22072", "O-2026-93", "ordinance")
        "R-316678"              → ("R-316678", None, "resolution")
    """
    raw = raw.strip()
    parts = [p.strip() for p in raw.split("/") if p.strip()]
    doc_num = parts[0]
    doc_num_alt = parts[1] if len(parts) > 1 else None

    if doc_num.startswith("O-"):
        doc_type = "ordinance"
    elif doc_num.startswith("R-"):
        doc_type = "resolution"
    else:
        doc_type = "unknown"

    return doc_num, doc_num_alt, doc_type


def parse_date(raw: str) -> str:
    """Convert MM/DD/YYYY → YYYY-MM-DD."""
    raw = raw.strip()
    match = re.match(r"(\d{2})/(\d{2})/(\d{4})", raw)
    if match:
        m, d, y = match.groups()
        return f"{y}-{m}-{d}"
    return raw


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def scrape_page(page_num: int, session: requests.Session) -> list[dict]:
    """Scrape one page of the San Diego ordinances/resolutions table."""
    resp = session.get(
        BASE_URL,
        params={"page": str(page_num)},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return _parse_table_html(resp.text)


def _parse_table_html(html: str) -> list[dict]:
    """Extract rows from the ordinance/resolution HTML table."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="views-table")
    if not table:
        return []

    rows = []
    for tr in table.find("tbody", recursive=False).find_all("tr", recursive=False):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        date_raw = tds[0].get_text(strip=True)
        doc_num_raw = tds[1].get_text(strip=True)
        title_td = tds[2]

        # Title and PDF URL from the <a> tag
        a_tag = title_td.find("a")
        if a_tag:
            title = a_tag.get_text(strip=True)
            pdf_url = a_tag.get("href", "")
        else:
            title = title_td.get_text(strip=True)
            pdf_url = ""

        date = parse_date(date_raw)
        doc_num, doc_num_alt, doc_type = parse_doc_num(doc_num_raw)

        # Extract year from date
        try:
            year = int(date[:4])
        except (ValueError, IndexError):
            year = 0

        rows.append(
            {
                "date": date,
                "doc_num": doc_num,
                "doc_num_alt": doc_num_alt,
                "doc_type": doc_type,
                "title": title,
                "text": "",  # filled later by extract_pdf_text
                "pdf_url": pdf_url,
                "year": year,
            }
        )

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
        # docs.sandiego.gov has SSL cert issues; fall back to verify=False
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


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame | None:
    """Load a checkpoint parquet file if it exists."""
    if checkpoint_path.exists():
        return pd.read_parquet(checkpoint_path)
    return None


def save_checkpoint(df: pd.DataFrame, checkpoint_path: Path) -> None:
    """Save a DataFrame as a parquet checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(checkpoint_path, index=False)
    print(f"  [checkpoint] Saved {len(df)} rows to {checkpoint_path}")


def scrape_all_pages(
    start: int = 0,
    end: int = 1998,
    extract_text: bool = True,
    page_delay: float = 1.5,
    pdf_delay: float = 0.5,
    checkpoint_interval: int = 100,
) -> pd.DataFrame:
    """Scrape pages [start, end) and return a DataFrame of all entries."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    all_rows: list[dict] = []

    # Load any existing checkpoints in range
    for cp_file in sorted(CHECKPOINT_DIR.glob("pages_*.parquet")):
        cp_df = pd.read_parquet(cp_file)
        all_rows.extend(cp_df.to_dict("records"))
    if all_rows:
        print(f"Loaded {len(all_rows)} rows from existing checkpoints")

    # Determine which pages have already been scraped
    scraped_pages_file = CHECKPOINT_DIR / "scraped_pages.txt"
    scraped_pages: set[int] = set()
    if scraped_pages_file.exists():
        scraped_pages = {
            int(line.strip())
            for line in scraped_pages_file.read_text().splitlines()
            if line.strip().isdigit()
        }

    batch_rows: list[dict] = []
    empty_page_streak = 0

    for page_num in range(start, end):
        if page_num in scraped_pages:
            continue

        print(f"Scraping page {page_num}/{end - 1}...")
        try:
            rows = scrape_page(page_num, session)
        except Exception as e:
            print(f"  [error] Page {page_num} failed: {e}")
            _jittered_sleep(page_delay * 3)
            continue

        if not rows:
            empty_page_streak += 1
            if empty_page_streak >= 5:
                print(f"  [stop] {empty_page_streak} consecutive empty pages — stopping.")
                break
            _jittered_sleep(page_delay)
            continue

        empty_page_streak = 0

        # Extract PDF text for each row
        if extract_text:
            for i, row in enumerate(rows):
                if row["pdf_url"]:
                    print(f"  Extracting text from PDF {i + 1}/{len(rows)}: {row['doc_num']}")
                    row["text"] = extract_pdf_text(row["pdf_url"], session)
                    _jittered_sleep(pdf_delay)

        batch_rows.extend(rows)
        scraped_pages.add(page_num)

        # Record scraped page
        with open(scraped_pages_file, "a") as f:
            f.write(f"{page_num}\n")

        print(f"  Got {len(rows)} rows (batch total: {len(batch_rows)})")

        # Checkpoint every N pages
        if len(scraped_pages) % checkpoint_interval == 0 and batch_rows:
            batch_df = pd.DataFrame(batch_rows)
            cp_name = f"pages_{start:05d}_batch_{len(scraped_pages):05d}.parquet"
            save_checkpoint(batch_df, CHECKPOINT_DIR / cp_name)
            all_rows.extend(batch_rows)
            batch_rows = []

        _jittered_sleep(page_delay)

    # Save any remaining rows
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        cp_name = f"pages_{start:05d}_final.parquet"
        save_checkpoint(batch_df, CHECKPOINT_DIR / cp_name)
        all_rows.extend(batch_rows)

    if not all_rows:
        return pd.DataFrame(
            columns=["date", "doc_num", "doc_num_alt", "doc_type", "title", "text", "pdf_url", "year"]
        )

    df = pd.DataFrame(all_rows)
    # Deduplicate by doc_num
    df = df.drop_duplicates(subset=["doc_num"], keep="first").reset_index(drop=True)
    return df


def push_to_huggingface(df: pd.DataFrame, repo_id: str) -> None:
    """Push a DataFrame to HuggingFace as a parquet dataset."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "datasets and huggingface_hub are required: "
            "pip install datasets huggingface_hub"
        )

    # Ensure the HF token is available
    api = HfApi()
    api.whoami()  # raises if not logged in

    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, private=False)
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def merge_all_checkpoints() -> pd.DataFrame:
    """Merge all checkpoint files into a single DataFrame."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp_file in sorted(CHECKPOINT_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(cp_file))
    if not dfs:
        print("No checkpoint files found.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["doc_num"], keep="first").reset_index(drop=True)
    print(f"Merged {len(df)} unique rows from {len(dfs)} checkpoint files")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build San Diego ordinances dataset")
    parser.add_argument("--start", type=int, default=0, help="Start page (default: 0)")
    parser.add_argument("--end", type=int, default=1998, help="End page exclusive (default: 1998)")
    parser.add_argument("--no-text", action="store_true", help="Skip PDF text extraction")
    parser.add_argument("--page-delay", type=float, default=2.0, help="Base delay between pages in seconds (±50%% jitter)")
    parser.add_argument("--pdf-delay", type=float, default=1.0, help="Base delay between PDF downloads in seconds (±50%% jitter)")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N pages")
    parser.add_argument("--push-only", action="store_true", help="Only merge checkpoints and push to HF")
    parser.add_argument("--repo-id", default="chrissoria/san-diego-ordinances", help="HuggingFace repo ID")
    parser.add_argument("--output", default=None, help="Save final CSV to this path (optional)")

    args = parser.parse_args()

    if args.push_only:
        df = merge_all_checkpoints()
        if df.empty:
            print("Nothing to push.")
            return
    else:
        df = scrape_all_pages(
            start=args.start,
            end=args.end,
            extract_text=not args.no_text,
            page_delay=args.page_delay,
            pdf_delay=args.pdf_delay,
            checkpoint_interval=args.checkpoint_interval,
        )

    print(f"\nTotal rows: {len(df)}")
    print(f"Ordinances: {(df['doc_type'] == 'ordinance').sum()}")
    print(f"Resolutions: {(df['doc_type'] == 'resolution').sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rows with text: {(df['text'].str.len() > 0).sum()}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    # Save final merged parquet
    final_path = CHECKPOINT_DIR / "final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Final dataset saved to {final_path}")

    # Push to HuggingFace if requested
    if args.push_only:
        push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
