#!/usr/bin/env python3
"""Generic scraper for Code Publishing municipal codes → HuggingFace.

Scrapes codified municipal code sections from codepublishing.com. Each section
includes the full text of the law plus ordinance references with dates.

Supports: Clovis, Newport Beach, Bakersfield, and any other Code Publishing city.

Usage:
    python scripts/build_codepublishing_dataset.py clovis
    python scripts/build_codepublishing_dataset.py newportbeach
    python scripts/build_codepublishing_dataset.py clovis --push-only

Dependencies:
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BASE_URL = "https://www.codepublishing.com"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

# City configurations: slug → (state, code_prefix, hf_repo, display_name)
CITY_CONFIG = {
    "clovis": {
        "state": "CA",
        "prefix": "Clovis",
        "repo": "chrissoria/clovis-ordinances",
        "name": "Clovis",
        "titles": list(range(1, 11)),  # Titles 1-10
    },
    "newportbeach": {
        "state": "CA",
        "prefix": "NewportBeach",
        "repo": "chrissoria/newport-beach-ordinances",
        "name": "Newport Beach",
        "titles": list(range(1, 22)),  # Titles 1-21
    },
}

DATASET_COLUMNS = [
    "section_id",
    "title_num",
    "chapter",
    "section_title",
    "text",
    "ordinance_refs",
    "url",
]


def _jittered_sleep(base: float):
    time.sleep(base * (0.7 + 0.6 * random.random()))


def discover_sections(state: str, prefix: str, titles: list, delay: float = 0.5) -> list[str]:
    """Discover all section HTML files for a city by scanning title/chapter patterns."""
    session = requests.Session()
    all_files = []

    for title_num in titles:
        title_dir = f"{prefix}{title_num:02d}"
        base_path = f"{BASE_URL}/{state}/{prefix}/html/{title_dir}"

        # Try chapter files: Clovis0101.html, Clovis0102.html, ...
        for chapter in range(1, 100):
            url = f"{base_path}/{title_dir}{chapter:02d}.html"
            try:
                resp = session.head(url, headers=HEADERS, timeout=10, allow_redirects=True)
                if resp.status_code == 200:
                    all_files.append(url)
                elif resp.status_code == 404:
                    if chapter > 5:
                        break
                else:
                    break
            except requests.exceptions.RequestException:
                break

            _jittered_sleep(delay * 0.3)

        # Also try direct section pattern: Clovis01.html (title-level content)
        title_url = f"{base_path}/{title_dir}.html"
        try:
            resp = session.head(title_url, headers=HEADERS, timeout=10)
            if resp.status_code == 200 and title_url not in all_files:
                all_files.append(title_url)
        except:
            pass

        print(f"  Title {title_num}: found {sum(1 for f in all_files if f'/{title_dir}/' in f)} sections")

    return all_files


def scrape_section(url: str, session: requests.Session) -> dict | None:
    """Scrape a single code section page."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
    except requests.exceptions.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("div", id="mainContent")
    if not content:
        return None

    # Extract title
    h1 = content.find("h1")
    section_title = h1.get_text(strip=True) if h1 else ""

    # Extract full text
    text = content.get_text(separator="\n").strip()
    if not text or len(text) < 20:
        return None

    # Extract ordinance references with dates
    ord_refs = re.findall(
        r'(?:Ord\.|Ordinance)\s*([\d]+[-\w]*)'
        r'(?:.*?(?:eff\.?|effective)\s*(\w+\.?\s*\d{1,2},?\s*\d{4}))?',
        text
    )
    ord_ref_strs = []
    for num, date in ord_refs:
        if date:
            ord_ref_strs.append(f"Ord. {num} (eff. {date})")
        else:
            ord_ref_strs.append(f"Ord. {num}")

    # Parse section ID from URL
    filename = url.split("/")[-1].replace(".html", "")

    # Parse title number
    title_match = re.search(r'(\d{2})\d{2}$', filename)
    title_num = ""
    chapter = ""
    if title_match:
        title_num = str(int(filename[-4:-2]))
        chapter = str(int(filename[-2:]))

    return {
        "section_id": filename,
        "title_num": title_num,
        "chapter": chapter,
        "section_title": section_title,
        "text": text,
        "ordinance_refs": "; ".join(ord_ref_strs),
        "url": url,
    }


def build_dataset(city_key: str, delay: float = 1.0) -> pd.DataFrame:
    """Scrape all code sections for a city."""
    cfg = CITY_CONFIG[city_key]
    cp_dir = CHECKPOINT_DIR / city_key
    cp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering sections for {cfg['name']}...")
    section_urls = discover_sections(cfg["state"], cfg["prefix"], cfg["titles"])
    print(f"Found {len(section_urls)} section files")

    # Load existing
    existing = set()
    final_path = cp_dir / f"{city_key}_final_dataset.parquet"
    if final_path.exists():
        edf = pd.read_parquet(final_path)
        existing = set(edf["url"].tolist())
        print(f"Already scraped: {len(existing)} sections")

    session = requests.Session()
    rows = []
    skipped = 0

    for i, url in enumerate(section_urls):
        if url in existing:
            skipped += 1
            continue

        section = scrape_section(url, session)
        if section:
            rows.append(section)

        if (len(rows)) % 25 == 0 and len(rows) > 0:
            print(f"  Scraped {i + 1}/{len(section_urls)} ({len(rows)} new, {skipped} skipped)")

        # Checkpoint every 100
        if len(rows) % 100 == 0 and rows:
            cp_path = cp_dir / f"{city_key}_batch_{len(rows):05d}.parquet"
            pd.DataFrame(rows[-100:]).to_parquet(cp_path, index=False)

        _jittered_sleep(delay)

    print(f"Done: {len(rows)} new sections, {skipped} skipped")

    df = pd.DataFrame(rows, columns=DATASET_COLUMNS)

    # Merge with existing if any
    if existing and final_path.exists():
        edf = pd.read_parquet(final_path)
        df = pd.concat([df, edf], ignore_index=True)
        df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

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
    parser = argparse.ArgumentParser(description="Build Code Publishing municipal code dataset")
    parser.add_argument("city", choices=list(CITY_CONFIG.keys()), help="City to scrape")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--merge-push", action="store_true")
    args = parser.parse_args()

    cfg = CITY_CONFIG[args.city]
    cp_dir = CHECKPOINT_DIR / args.city
    cp_dir.mkdir(parents=True, exist_ok=True)

    if args.push_only or args.merge_push:
        final_path = cp_dir / f"{args.city}_final_dataset.parquet"
        if final_path.exists():
            df = pd.read_parquet(final_path)
        else:
            print("No dataset found.")
            return
    else:
        df = build_dataset(args.city, delay=args.delay)

    print(f"\n=== {cfg['name']} Municipal Code ===")
    print(f"Total sections: {len(df)}")
    print(f"With text: {(df['text'].str.len() > 0).sum()}")
    print(f"With ordinance refs: {(df['ordinance_refs'].str.len() > 0).sum()}")

    final_path = cp_dir / f"{args.city}_final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Saved to {final_path}")

    if args.push_only or args.merge_push:
        push_to_huggingface(df, cfg["repo"])


if __name__ == "__main__":
    main()
