#!/usr/bin/env python3
"""Scraper: Bakersfield ordinances from municipal.codes/enactments → HuggingFace.

Uses Playwright (visible browser) to bypass Cloudflare protection on municipal.codes.

Usage:
    python scripts/build_bakersfield_dataset.py
    python scripts/build_bakersfield_dataset.py --start-page 1 --end-page 10
    python scripts/build_bakersfield_dataset.py --push-only

Dependencies:
    pip install -r scripts/requirements.txt
    pip install playwright
    python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import os
import random
import time
import warnings
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BASE_URL = "https://bakersfield.municipal.codes"

DATASET_COLUMNS = [
    "date", "ordinance_number", "doc_type", "title",
    "text", "disposition", "effective_date", "url", "year",
]


def _jittered_sleep(base: float) -> None:
    time.sleep(base * random.uniform(0.5, 1.5))


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------


def _create_browser(pw):
    """Create a Playwright browser context that bypasses Cloudflare."""
    browser = pw.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 720},
    )
    return browser, context


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def scrape_enactments_page(page, page_num: int) -> list[dict]:
    """Scrape one page of the enactments table."""
    url = f"{BASE_URL}/enactments" if page_num == 1 else f"{BASE_URL}/enactments?page={page_num}"
    page.goto(url, timeout=30000)
    page.wait_for_timeout(2000)

    # The enactments table is table index 1 (index 0 is a header-only table)
    tables = page.query_selector_all("table")
    if len(tables) < 2:
        return []

    table = tables[1]
    rows = table.query_selector_all("tr")

    entries = []
    for row in rows:
        cells = row.query_selector_all("td")
        if len(cells) < 4:
            continue

        # Column 0: Number (with link)
        a_tag = cells[0].query_selector("a")
        if a_tag:
            ordinance_number = a_tag.inner_text().strip()
            detail_path = a_tag.get_attribute("href") or ""
            detail_url = f"{BASE_URL}{detail_path}" if detail_path.startswith("/") else detail_path
        else:
            ordinance_number = cells[0].inner_text().strip()
            detail_url = ""

        # Column 1: Description
        title = cells[1].inner_text().strip()

        # Column 2: Action Date
        date_raw = cells[2].inner_text().strip()

        # Column 3: Disposition
        disposition = cells[3].inner_text().strip()

        # Determine doc_type from ordinance number
        num_lower = ordinance_number.lower()
        if "ord" in num_lower:
            doc_type = "ordinance"
        elif "res" in num_lower:
            doc_type = "resolution"
        else:
            doc_type = "ordinance"

        # Parse date (M/D/YYYY → YYYY-MM-DD, or year-only → YYYY-01-01)
        date = ""
        try:
            from datetime import datetime
            dt = datetime.strptime(date_raw, "%m/%d/%Y")
            date = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            # Some older entries only have a year
            if date_raw.strip().isdigit() and len(date_raw.strip()) == 4:
                date = f"{date_raw.strip()}-01-01"
            else:
                date = date_raw

        try:
            year = int(date[:4]) if date else 0
        except ValueError:
            year = 0

        entries.append({
            "date": date,
            "ordinance_number": ordinance_number,
            "doc_type": doc_type,
            "title": title,
            "text": "",  # filled by detail page scraping
            "disposition": disposition,
            "effective_date": "",
            "url": detail_url,
            "year": year,
        })

    return entries


def scrape_detail_page(page, entry: dict) -> dict:
    """Scrape the detail page for an ordinance to get full text."""
    if not entry["url"]:
        return entry

    try:
        page.goto(entry["url"], timeout=30000)
        page.wait_for_timeout(1500)

        main = page.query_selector("#main-column, main, article")
        if main:
            text = main.inner_text().strip()
        else:
            text = page.inner_text("body").strip()

        entry["text"] = text

        # Try to extract effective date
        if "Effective From" in text:
            for line in text.split("\n"):
                line = line.strip()
                if line and line != "Effective From":
                    # Check if the previous line was "Effective From"
                    pass
            # More targeted extraction
            try:
                idx = text.index("Effective From")
                chunk = text[idx + len("Effective From"):idx + 60].strip()
                # First line after label
                eff_date = chunk.split("\n")[0].strip()
                from datetime import datetime
                dt = datetime.strptime(eff_date, "%m/%d/%Y")
                entry["effective_date"] = dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                pass

    except Exception as e:
        print(f"  [warn] Detail page failed for {entry['ordinance_number']}: {e}")

    return entry


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def scrape_all_pages(
    start_page: int = 1,
    end_page: int = 184,
    fetch_details: bool = True,
    page_delay: float = 2.0,
    detail_delay: float = 1.5,
    checkpoint_interval: int = 10,
) -> pd.DataFrame:
    """Scrape all enactment pages."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Track which pages are already done
    scraped_pages_file = CHECKPOINT_DIR / "bakersfield_scraped_pages.txt"
    scraped_pages: set[int] = set()
    if scraped_pages_file.exists():
        scraped_pages = {
            int(line.strip())
            for line in scraped_pages_file.read_text().splitlines()
            if line.strip().isdigit()
        }

    all_rows: list[dict] = []

    # Load existing checkpoints
    for cp_file in sorted(CHECKPOINT_DIR.glob("bakersfield_batch_*.parquet")):
        cp_df = pd.read_parquet(cp_file)
        all_rows.extend(cp_df.to_dict("records"))
    if all_rows:
        print(f"Loaded {len(all_rows)} rows from existing checkpoints")

    with sync_playwright() as pw:
        browser, context = _create_browser(pw)
        page = context.new_page()

        batch_rows: list[dict] = []

        for page_num in range(start_page, end_page + 1):
            if page_num in scraped_pages:
                continue

            print(f"Scraping page {page_num}/{end_page}...")
            try:
                entries = scrape_enactments_page(page, page_num)
            except Exception as e:
                print(f"  [error] Page {page_num} failed: {e}")
                _jittered_sleep(page_delay * 2)
                continue

            if not entries:
                print(f"  No entries on page {page_num}")
                break

            # Fetch detail pages for full text
            if fetch_details:
                for i, entry in enumerate(entries):
                    if entry["url"]:
                        if (i + 1) % 10 == 0 or i == 0:
                            print(f"  Detail {i + 1}/{len(entries)}: {entry['ordinance_number']}")
                        scrape_detail_page(page, entry)
                        _jittered_sleep(detail_delay)

            batch_rows.extend(entries)
            scraped_pages.add(page_num)

            with open(scraped_pages_file, "a") as f:
                f.write(f"{page_num}\n")

            print(f"  Got {len(entries)} entries (batch total: {len(batch_rows)})")

            # Checkpoint
            if len(scraped_pages) % checkpoint_interval == 0 and batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                cp_name = f"bakersfield_batch_{len(scraped_pages):05d}.parquet"
                batch_df.to_parquet(CHECKPOINT_DIR / cp_name, index=False)
                print(f"  [checkpoint] Saved {len(batch_df)} rows")
                all_rows.extend(batch_rows)
                batch_rows = []

            _jittered_sleep(page_delay)

        browser.close()

    # Save remaining
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        cp_name = "bakersfield_batch_final.parquet"
        batch_df.to_parquet(CHECKPOINT_DIR / cp_name, index=False)
        all_rows.extend(batch_rows)

    if not all_rows:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["ordinance_number"], keep="first").reset_index(drop=True)
    return df


def merge_all_checkpoints() -> pd.DataFrame:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp_file in sorted(CHECKPOINT_DIR.glob("bakersfield_batch_*.parquet")):
        dfs.append(pd.read_parquet(cp_file))
    if not dfs:
        print("No Bakersfield checkpoint files found.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["ordinance_number"], keep="first").reset_index(drop=True)
    print(f"Merged {len(df)} unique rows from {len(dfs)} checkpoint files")
    return df


def push_to_huggingface(df: pd.DataFrame, repo_id: str) -> None:
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("datasets and huggingface_hub required")

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
    parser = argparse.ArgumentParser(description="Build Bakersfield ordinances dataset")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=184)
    parser.add_argument("--no-details", action="store_true", help="Skip detail page scraping")
    parser.add_argument("--page-delay", type=float, default=4.0, help="Base delay between list pages (±50%% jitter)")
    parser.add_argument("--detail-delay", type=float, default=5.0, help="Base delay between detail pages (±50%% jitter)")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint every N pages")
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--repo-id", default="chrissoria/bakersfield-ordinances")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.push_only:
        df = merge_all_checkpoints()
        if df.empty:
            print("Nothing to push.")
            return
    else:
        df = scrape_all_pages(
            start_page=args.start_page,
            end_page=args.end_page,
            fetch_details=not args.no_details,
            page_delay=args.page_delay,
            detail_delay=args.detail_delay,
            checkpoint_interval=args.checkpoint_interval,
        )

    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rows with text: {(df['text'].str.len() > 0).sum()}")
    print(f"Dispositions: {df['disposition'].value_counts().to_dict()}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    final_path = CHECKPOINT_DIR / "bakersfield_final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Final dataset saved to {final_path}")

    if args.push_only:
        push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
