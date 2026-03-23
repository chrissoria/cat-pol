#!/usr/bin/env python3
"""Scrape all executive orders from the American Presidency Project (UCSB).

Covers 1791-present (~10,800 executive orders). Checkpoints every 50 documents.

Usage:
    python scripts/build_executive_orders_ucsb.py
    python scripts/build_executive_orders_ucsb.py --merge-push
"""

from __future__ import annotations

import argparse
import os
import re
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "executive_orders"
UCSB = "https://www.presidency.ucsb.edu"
HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

PRESIDENT_PARTIES = {
    "George Washington": "None", "John Adams": "Federalist",
    "Thomas Jefferson": "Democratic-Republican", "James Madison": "Democratic-Republican",
    "James Monroe": "Democratic-Republican", "John Quincy Adams": "Democratic-Republican",
    "Andrew Jackson": "Democratic", "Martin Van Buren": "Democratic",
    "William Henry Harrison": "Whig", "John Tyler": "Whig",
    "James K. Polk": "Democratic", "Zachary Taylor": "Whig",
    "Millard Fillmore": "Whig", "Franklin Pierce": "Democratic",
    "James Buchanan": "Democratic", "Abraham Lincoln": "Republican",
    "Andrew Johnson": "Democratic", "Ulysses S. Grant": "Republican",
    "Rutherford B. Hayes": "Republican", "James A. Garfield": "Republican",
    "Chester A. Arthur": "Republican", "Grover Cleveland": "Democratic",
    "Benjamin Harrison": "Republican", "William McKinley": "Republican",
    "Theodore Roosevelt": "Republican", "William Howard Taft": "Republican",
    "Woodrow Wilson": "Democratic", "Warren G. Harding": "Republican",
    "Calvin Coolidge": "Republican", "Herbert Hoover": "Republican",
    "Franklin D. Roosevelt": "Democratic", "Harry S. Truman": "Democratic",
    "Dwight D. Eisenhower": "Republican", "John F. Kennedy": "Democratic",
    "Lyndon B. Johnson": "Democratic", "Richard Nixon": "Republican",
    "Gerald R. Ford": "Republican", "Jimmy Carter": "Democratic",
    "Ronald Reagan": "Republican", "George H. W. Bush": "Republican",
    "Bill Clinton": "Democratic", "George W. Bush": "Republican",
    "Barack Obama": "Democratic", "Donald Trump": "Republican",
    "Donald J. Trump": "Republican", "Donald J. Trump (2nd Term)": "Republican",
    "Joe Biden": "Democratic", "Joseph R. Biden Jr.": "Democratic",
}


def _sleep(base: float):
    time.sleep(base * (0.7 + 0.6 * random.random()))


def _get(session, url, timeout=60, retries=3):
    """GET with retry logic."""
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  [retry] {e.__class__.__name__}, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def collect_urls(session) -> list[str]:
    """Collect all executive order document URLs from UCSB listing pages."""
    # Check for cached URL list
    url_cache = CHECKPOINT_DIR / "eo_urls.txt"
    if url_cache.exists():
        urls = url_cache.read_text().strip().split("\n")
        print(f"Loaded {len(urls)} URLs from cache")
        return urls

    print("Collecting executive order URLs from UCSB...")
    all_urls = []
    page = 0

    while True:
        try:
            resp = _get(
                session,
                f"{UCSB}/documents/app-categories/written-presidential-orders/presidential/executive-orders",
                timeout=60,
            )
            # Need to pass page param via URL
            resp = _get(
                session,
                f"{UCSB}/documents/app-categories/written-presidential-orders/presidential/executive-orders?items_per_page=60&page={page}",
                timeout=60,
            )
        except Exception as e:
            print(f"  [error] Page {page} failed: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        nodes = soup.find_all("div", class_="field-title")
        found = 0
        for node in nodes:
            a = node.find("a", href=True)
            if a:
                href = a["href"]
                full = href if href.startswith("http") else f"{UCSB}{href}"
                all_urls.append(full)
                found += 1

        if found == 0:
            break

        page += 1
        if page % 10 == 0:
            print(f"  Page {page}: {len(all_urls)} URLs...")

        _sleep(2.0)

    all_urls = list(dict.fromkeys(all_urls))
    print(f"Total: {len(all_urls)} unique URLs")

    # Cache the URL list
    url_cache.write_text("\n".join(all_urls))
    return all_urls


def scrape_document(session, url) -> dict | None:
    """Scrape a single executive order page."""
    try:
        resp = _get(session, url, timeout=60)
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    body = soup.find("div", class_="field-docs-content")
    text = body.get_text(separator="\n").strip() if body else ""
    if not text:
        return None

    date_el = soup.find("span", class_="date-display-single")
    date_str = date_el.get_text(strip=True) if date_el else ""
    date = ""
    for fmt in ["%B %d, %Y", "%b %d, %Y"]:
        try:
            date = datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            break
        except ValueError:
            continue

    speaker_el = soup.find("h3", class_="diet-title")
    speaker = speaker_el.get_text(strip=True) if speaker_el else ""
    speaker_clean = re.sub(r'\s*\(.*?\)\s*$', '', speaker)

    title_el = soup.find("div", class_="field-ds-doc-title")
    title = title_el.get_text(strip=True) if title_el else ""
    if not title:
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else ""

    eo_match = re.search(r'Executive Order (\d+)', title)
    eo_number = eo_match.group(1) if eo_match else ""

    party = PRESIDENT_PARTIES.get(speaker, PRESIDENT_PARTIES.get(speaker_clean, ""))

    return {
        "date": date,
        "title": title,
        "executive_order_number": eo_number,
        "president": speaker,
        "party": party,
        "text": text,
        "url": url,
        "year": int(date[:4]) if date else 0,
    }


def load_existing_urls() -> set:
    existing = set()
    for cp in sorted(CHECKPOINT_DIR.glob("eo_ucsb_*.parquet")):
        try:
            df = pd.read_parquet(cp, columns=["url"])
            existing.update(df["url"].tolist())
        except Exception:
            pass
    return existing


def build_dataset():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    all_urls = collect_urls(session)
    existing = load_existing_urls()
    if existing:
        print(f"Already scraped: {len(existing)}, will skip")

    rows = []
    skipped = 0

    for i, url in enumerate(all_urls):
        if url in existing:
            skipped += 1
            continue

        doc = scrape_document(session, url)
        if doc:
            rows.append(doc)

        processed = len(rows)
        if processed % 25 == 0 and processed > 0:
            print(f"  Scraped {i+1}/{len(all_urls)} ({processed} new, {skipped} skipped)")

        if processed % 50 == 0 and processed > 0:
            cp = CHECKPOINT_DIR / f"eo_ucsb_{processed:05d}.parquet"
            pd.DataFrame(rows[-50:]).to_parquet(cp, index=False)
            print(f"  [checkpoint] {cp.name}")

        _sleep(2.0)  # 2s delay to be respectful

    # Final checkpoint
    remainder = len(rows) % 50
    if remainder > 0:
        cp = CHECKPOINT_DIR / f"eo_ucsb_{len(rows):05d}.parquet"
        pd.DataFrame(rows[-remainder:]).to_parquet(cp, index=False)

    print(f"\nDone: {len(rows)} new, {skipped} skipped")


def merge_and_push(repo_id: str):
    dfs = []
    for cp in sorted(CHECKPOINT_DIR.glob("eo_ucsb_*.parquet")):
        dfs.append(pd.read_parquet(cp))
    if not dfs:
        print("No checkpoint files found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    print(f"\n=== Executive Orders (UCSB) ===")
    print(f"Total: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Presidents: {df['president'].nunique()}")
    print(f"With text: {(df['text'].str.len() > 0).sum()}")
    print(f"With EO number: {(df['executive_order_number'].str.len() > 0).sum()}")

    df.to_parquet(CHECKPOINT_DIR / "eo_final_dataset.parquet", index=False)

    from datasets import Dataset
    from huggingface_hub import HfApi

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    token = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("No HF token")

    api = HfApi(token=token)
    api.whoami()
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Build executive orders dataset from UCSB")
    parser.add_argument("--merge-push", action="store_true", help="Merge checkpoints and push")
    parser.add_argument("--repo-id", default="chrissoria/executive-orders")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge_push:
        merge_and_push(args.repo_id)
    else:
        build_dataset()
        merge_and_push(args.repo_id)


if __name__ == "__main__":
    main()
