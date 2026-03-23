#!/usr/bin/env python3
"""Lightweight scraper: extract just ordinance numbers + dates from Berkeley listing pages.

Paginates through https://berkeley.municipal.codes/enactments and scrapes only
the table columns (ordinance number, date) — no detail page visits needed.
Output is merged into the existing Berkeley dataset to fill in missing dates.

Usage:
    python scripts/scrape_berkeley_dates.py
    python scripts/scrape_berkeley_dates.py --merge   # merge into existing dataset
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BASE_URL = "https://berkeley.municipal.codes"
OUTPUT_FILE = CHECKPOINT_DIR / "berkeley_dates.parquet"


def scrape_dates(page_delay: float = 3.0, max_pages: int = 200) -> pd.DataFrame:
    """Scrape ordinance number + date from listing pages."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError("playwright required: pip install playwright && python -m playwright install chromium")

    all_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # visible browser to bypass Cloudflare
        page = browser.new_page()

        # Load first page to get past Cloudflare
        print("Loading first page (Cloudflare check)...")
        page.goto(f"{BASE_URL}/enactments", timeout=60000)
        page.wait_for_timeout(5000)

        for page_num in range(1, max_pages + 1):
            url = f"{BASE_URL}/enactments?page={page_num}"
            print(f"Page {page_num}...", end=" ", flush=True)

            try:
                page.goto(url, timeout=30000)
                page.wait_for_timeout(2000)

                rows = page.query_selector_all("table tbody tr")
                if not rows:
                    print("no rows — end of pages")
                    break

                page_entries = 0
                for row in rows:
                    cells = row.query_selector_all("td")
                    if len(cells) < 3:
                        continue

                    ord_num = cells[0].inner_text().strip()
                    date_raw = cells[2].inner_text().strip()

                    # Parse date
                    date = ""
                    try:
                        dt = datetime.strptime(date_raw, "%m/%d/%Y")
                        date = dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        if date_raw.strip().isdigit() and len(date_raw.strip()) == 4:
                            date = date_raw.strip()
                        else:
                            date = date_raw  # keep as-is (could be "—")

                    all_rows.append({
                        "ordinance_number": ord_num,
                        "scraped_date": date,
                    })
                    page_entries += 1

                print(f"{page_entries} entries")

                # Checkpoint every 10 pages
                if page_num % 10 == 0:
                    tmp = pd.DataFrame(all_rows)
                    tmp.to_parquet(OUTPUT_FILE, index=False)
                    print(f"  [checkpoint] {len(all_rows)} total entries saved")

            except Exception as e:
                print(f"error: {e}")
                continue

            time.sleep(page_delay * (0.7 + 0.6 * __import__('random').random()))

        browser.close()

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(df)} entries saved to {OUTPUT_FILE}")
    return df


def merge_dates():
    """Merge scraped dates into the existing Berkeley dataset."""
    dataset_path = CHECKPOINT_DIR / "berkeley_final_dataset.parquet"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    if not OUTPUT_FILE.exists():
        print(f"Scraped dates not found: {OUTPUT_FILE}")
        return

    df = pd.read_parquet(dataset_path)
    dates = pd.read_parquet(OUTPUT_FILE)

    print(f"Dataset: {len(df)} rows")
    print(f"Scraped dates: {len(dates)} entries")

    # Before merge stats
    full_before = df['date'].str.match(r'^\d{4}-\d{2}-\d{2}$').sum()
    print(f"Full dates before merge: {full_before}")

    # Merge: update date where scraped_date is better (has month/day)
    date_map = {}
    for _, row in dates.iterrows():
        sd = row['scraped_date']
        if sd and re.match(r'^\d{4}-\d{2}-\d{2}$', sd):
            date_map[row['ordinance_number']] = sd

    print(f"Scraped entries with full YYYY-MM-DD dates: {len(date_map)}")

    updated = 0
    for idx, row in df.iterrows():
        ord_num = row['ordinance_number']
        if ord_num in date_map:
            current = row['date']
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(current)):
                df.at[idx, 'date'] = date_map[ord_num]
                # Also update year
                df.at[idx, 'year'] = int(date_map[ord_num][:4])
                updated += 1

    full_after = df['date'].str.match(r'^\d{4}-\d{2}-\d{2}$').sum()
    print(f"Full dates after merge: {full_after} (+{full_after - full_before})")
    print(f"Rows updated: {updated}")

    df.to_parquet(dataset_path, index=False)
    print(f"Saved updated dataset to {dataset_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Berkeley ordinance dates")
    parser.add_argument("--merge", action="store_true", help="Merge scraped dates into dataset")
    parser.add_argument("--page-delay", type=float, default=3.0)
    parser.add_argument("--max-pages", type=int, default=200)
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_dates()
    else:
        scrape_dates(page_delay=args.page_delay, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
