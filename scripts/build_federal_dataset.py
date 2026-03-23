#!/usr/bin/env python3
"""One-time scraper: Federal enacted public laws from GovInfo API → HuggingFace.

Fetches all public laws from the 104th Congress (1995) to present using the
GovInfo API (api.govinfo.gov). Extracts full text from HTM endpoint and
enriches with metadata from the package summary.

Usage:
    python scripts/build_federal_dataset.py
    python scripts/build_federal_dataset.py --no-text
    python scripts/build_federal_dataset.py --push-only
    python scripts/build_federal_dataset.py --merge-push
    python scripts/build_federal_dataset.py --congress 118    # Only 118th Congress

Dependencies:
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
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

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "federal"
GOVINFO_BASE = "https://api.govinfo.gov"
COLLECTION = "PLAW"
START_DATE = "1995-01-01T00:00:00Z"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

DATASET_COLUMNS = [
    "date",
    "congress",
    "law_number",
    "package_id",
    "doc_type",
    "title",
    "short_title",
    "pages",
    "text",
    "url",
    "pdf_url",
]


# ---------------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """Load GovInfo API key from .env or environment."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    key = os.getenv("GOVINFO_API_KEY") or os.getenv("DATA_GOV_API_KEY")
    if not key:
        key = "DEMO_KEY"
        print("[WARNING] No GOVINFO_API_KEY found in .env — using DEMO_KEY (lower rate limits)")
    return key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jittered_sleep(base: float):
    """Sleep with ±30% jitter to avoid rate-limit patterns."""
    time.sleep(base * (0.7 + 0.6 * random.random()))


def _strip_html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving paragraph breaks."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# GovInfo API calls
# ---------------------------------------------------------------------------

def fetch_all_packages(api_key: str, start_date: str = START_DATE, congress: int = None) -> list[dict]:
    """Fetch all PLAW package listings from GovInfo."""
    all_packages = []
    offset = 0
    page_size = 100

    while True:
        url = f"{GOVINFO_BASE}/collections/{COLLECTION}/{start_date}"
        params = {
            "offset": offset,
            "pageSize": page_size,
            "api_key": api_key,
        }
        if congress is not None:
            params["congress"] = congress

        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < 4:
                    _jittered_sleep(5 * (attempt + 1))
                else:
                    print(f"  Failed to fetch packages at offset {offset}: {e}")
                    return all_packages

        data = resp.json()
        packages = data.get("packages", [])
        if not packages:
            break

        all_packages.extend(packages)
        offset += page_size

        if len(all_packages) % 500 == 0 or not data.get("nextPage"):
            print(f"  Fetched {len(all_packages)} package listings...")

        if not data.get("nextPage"):
            break

        _jittered_sleep(0.3)

    return all_packages


def fetch_package_summary(package_id: str, api_key: str) -> dict | None:
    """Fetch detailed summary for a single package."""
    url = f"{GOVINFO_BASE}/packages/{package_id}/summary"
    for attempt in range(3):
        try:
            resp = requests.get(url, params={"api_key": api_key}, headers=HEADERS, timeout=60)
            if resp.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt < 2:
                _jittered_sleep(3 * (attempt + 1))
    return None


def fetch_text(package_id: str, api_key: str) -> str:
    """Fetch full text of a public law (HTM format → plain text)."""
    url = f"{GOVINFO_BASE}/packages/{package_id}/htm"
    for attempt in range(3):
        try:
            resp = requests.get(url, params={"api_key": api_key}, headers=HEADERS, timeout=120)
            if resp.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            if resp.status_code == 404:
                return ""
            resp.raise_for_status()
            return _strip_html_to_text(resp.text)
        except requests.exceptions.RequestException:
            if attempt < 2:
                _jittered_sleep(3 * (attempt + 1))
    return ""


# ---------------------------------------------------------------------------
# Load existing checkpoint IDs
# ---------------------------------------------------------------------------

def _load_existing_package_ids() -> set:
    """Load package IDs already scraped from checkpoint files."""
    existing = set()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for cp_file in sorted(CHECKPOINT_DIR.glob("federal_plaw_*.parquet")):
        try:
            df = pd.read_parquet(cp_file, columns=["package_id"])
            existing.update(df["package_id"].astype(str).tolist())
        except Exception:
            pass
    return existing


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(
    api_key: str,
    extract_text: bool = True,
    api_delay: float = 1.0,
    congress: int = None,
) -> pd.DataFrame:
    """Fetch all public laws and build the dataset."""

    existing_ids = _load_existing_package_ids()
    if existing_ids:
        print(f"  Found {len(existing_ids)} existing laws in checkpoints, will skip")

    print("Fetching public law listings from GovInfo...")
    packages = fetch_all_packages(api_key, congress=congress)
    # Filter to PUBLIC laws only (exclude PRIVATE)
    packages = [p for p in packages if p.get("docClass", "").upper() == "PUBLIC"]
    print(f"  Found {len(packages)} public laws total")

    new_rows: list[dict] = []
    skipped = 0

    for i, pkg in enumerate(packages):
        package_id = pkg["packageId"]

        if package_id in existing_ids:
            skipped += 1
            continue

        # Parse congress and law number from packageId (e.g., "PLAW-119publ1")
        match = re.match(r"PLAW-(\d+)publ(\d+)", package_id)
        congress_num = match.group(1) if match else pkg.get("congress", "")
        law_num = match.group(2) if match else ""

        # Fetch summary for enrichment
        summary = fetch_package_summary(package_id, api_key)
        _jittered_sleep(api_delay * 0.5)

        short_title = ""
        pages = ""
        pdf_url = ""
        if summary:
            st_list = summary.get("shortTitle", [])
            if st_list and isinstance(st_list, list):
                short_title = st_list[0].get("title", "")
            pages = str(summary.get("pages", ""))
            dl = summary.get("download", {})
            pdf_url = dl.get("pdfLink", "")

        # Fetch full text
        text = ""
        if extract_text:
            text = fetch_text(package_id, api_key)
            _jittered_sleep(api_delay)

        row = {
            "date": pkg.get("dateIssued", "")[:10],
            "congress": str(congress_num),
            "law_number": f"Pub.L. {congress_num}-{law_num}" if law_num else "",
            "package_id": package_id,
            "doc_type": "public_law",
            "title": pkg.get("title", ""),
            "short_title": short_title,
            "pages": pages,
            "text": text,
            "url": f"https://www.govinfo.gov/app/details/{package_id}",
            "pdf_url": pdf_url,
        }

        new_rows.append(row)
        processed = len(new_rows)

        if processed % 25 == 0 or processed == 1:
            print(f"  Processed {i + 1}/{len(packages)} ({processed} new, {skipped} skipped)")

        # Incremental checkpoint every 50 new items
        if processed % 50 == 0:
            cp_path = CHECKPOINT_DIR / f"federal_plaw_{processed:05d}.parquet"
            pd.DataFrame(new_rows[-50:]).to_parquet(cp_path, index=False)
            print(f"  [checkpoint] Saved batch to {cp_path.name}")

    # Final checkpoint for remaining items
    remainder = len(new_rows) % 50
    if remainder > 0:
        cp_path = CHECKPOINT_DIR / f"federal_plaw_{len(new_rows):05d}.parquet"
        pd.DataFrame(new_rows[-remainder:]).to_parquet(cp_path, index=False)
        print(f"  [checkpoint] Saved {remainder} laws (final)")

    print(f"  Done: {len(new_rows)} new, {skipped} skipped")

    df = pd.DataFrame(new_rows, columns=DATASET_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Merge & push
# ---------------------------------------------------------------------------

def merge_all_checkpoints() -> pd.DataFrame:
    """Merge all federal checkpoint files."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp_file in sorted(CHECKPOINT_DIR.glob("federal_plaw_*.parquet")):
        dfs.append(pd.read_parquet(cp_file))
    if not dfs:
        print("No federal checkpoint files found.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["package_id"], keep="first").reset_index(drop=True)
    print(f"Merged {len(df)} unique rows from {len(dfs)} checkpoint files")
    return df


def push_to_huggingface(df: pd.DataFrame, repo_id: str) -> None:
    """Push DataFrame to HuggingFace."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("datasets and huggingface_hub required: pip install datasets huggingface_hub")

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build federal public laws dataset")
    parser.add_argument("--no-text", action="store_true", help="Skip text extraction")
    parser.add_argument("--api-delay", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--push-only", action="store_true",
                        help="Only push existing checkpoint to HF (no scraping)")
    parser.add_argument("--merge-push", action="store_true",
                        help="Merge checkpoints and push to HF")
    parser.add_argument("--congress", type=int, default=None,
                        help="Only fetch a specific Congress (e.g., 118)")
    parser.add_argument("--repo-id", default="chrissoria/federal-public-laws")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = _load_api_key()

    if args.push_only or args.merge_push:
        df = merge_all_checkpoints()
        if df.empty:
            print("Nothing to push.")
            return
    else:
        df = build_dataset(
            api_key=api_key,
            extract_text=not args.no_text,
            api_delay=args.api_delay,
            congress=args.congress,
        )

        # Merge with any prior checkpoints
        existing_df = merge_all_checkpoints()
        if not existing_df.empty:
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["package_id"], keep="first").reset_index(drop=True)

    # Stats
    print(f"\n=== Federal Public Laws ===")
    print(f"Total rows: {len(df)}")
    print(f"Congress range: {df['congress'].min()} to {df['congress'].max()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rows with text: {(df['text'].str.len() > 0).sum()}")
    print(f"Rows with PDF URL: {(df['pdf_url'].str.len() > 0).sum()}")

    # Save final
    final_path = CHECKPOINT_DIR / "federal_final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Final dataset saved to {final_path}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    if args.push_only or args.merge_push:
        push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
