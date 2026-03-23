#!/usr/bin/env python3
"""Build unified presidential speeches dataset: SOTU + Inaugurals + Opposition Responses.

Sources:
- State of the Union: HuggingFace jsulz/state-of-the-union-addresses (1790-2024)
- Inaugurals: American Presidency Project (1789-present)
- Opposition SOTU Responses: American Presidency Project (2001-present)

Usage:
    python scripts/build_presidential_speeches_dataset.py
    python scripts/build_presidential_speeches_dataset.py --push-only

Dependencies:
    pip install -r scripts/requirements.txt
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

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "presidential_speeches"
UCSB_BASE = "https://www.presidency.ucsb.edu"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

DATASET_COLUMNS = [
    "date",
    "speaker",
    "party",
    "speech_type",
    "title",
    "text",
    "url",
    "year",
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


# Presidential party affiliations
PRESIDENT_PARTIES = {
    "George Washington": "None",
    "John Adams": "Federalist",
    "Thomas Jefferson": "Democratic-Republican",
    "James Madison": "Democratic-Republican",
    "James Monroe": "Democratic-Republican",
    "John Quincy Adams": "Democratic-Republican",
    "Andrew Jackson": "Democratic",
    "Martin Van Buren": "Democratic",
    "John Tyler": "Whig",
    "James K. Polk": "Democratic",
    "Zachary Taylor": "Whig",
    "Millard Fillmore": "Whig",
    "Franklin Pierce": "Democratic",
    "James Buchanan": "Democratic",
    "Abraham Lincoln": "Republican",
    "Andrew Johnson": "Democratic",
    "Ulysses S. Grant": "Republican",
    "Rutherford B. Hayes": "Republican",
    "Chester A. Arthur": "Republican",
    "Grover Cleveland": "Democratic",
    "Benjamin Harrison": "Republican",
    "William McKinley": "Republican",
    "Theodore Roosevelt": "Republican",
    "William Howard Taft": "Republican",
    "Woodrow Wilson": "Democratic",
    "Warren G. Harding": "Republican",
    "Calvin Coolidge": "Republican",
    "Herbert Hoover": "Republican",
    "Franklin D. Roosevelt": "Democratic",
    "Harry S. Truman": "Democratic",
    "Dwight D. Eisenhower": "Republican",
    "John F. Kennedy": "Democratic",
    "Lyndon B. Johnson": "Democratic",
    "Richard Nixon": "Republican",
    "Gerald R. Ford": "Republican",
    "Jimmy Carter": "Democratic",
    "Ronald Reagan": "Republican",
    "George H. W. Bush": "Republican",
    "Bill Clinton": "Democratic",
    "George W. Bush": "Republican",
    "Barack Obama": "Democratic",
    "Donald Trump": "Republican",
    "Joe Biden": "Democratic",
    "Donald J. Trump": "Republican",
}


# ---------------------------------------------------------------------------
# 1. State of the Union addresses (from HuggingFace)
# ---------------------------------------------------------------------------

def load_sotu() -> pd.DataFrame:
    """Load SOTU addresses from existing HuggingFace dataset."""
    print("Loading State of the Union addresses from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("jsulz/state-of-the-union-addresses", split="train")
    df = ds.to_pandas()

    rows = []
    for _, row in df.iterrows():
        speaker = row.get("potus", "")
        text = row.get("speech_html", "")
        # speech_html is already plain text in this dataset
        if not isinstance(text, str):
            text = str(text) if text else ""

        date = str(row.get("date", ""))[:10]

        rows.append({
            "date": date,
            "speaker": speaker,
            "party": PRESIDENT_PARTIES.get(speaker, ""),
            "speech_type": "State of the Union",
            "title": f"State of the Union Address ({date[:4]})",
            "text": text,
            "url": "",
            "year": int(date[:4]) if date and len(date) >= 4 else 0,
        })

    print(f"  Loaded {len(rows)} SOTU addresses")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Inaugural addresses (scrape from UCSB)
# ---------------------------------------------------------------------------

def scrape_inaugurals() -> pd.DataFrame:
    """Scrape inaugural addresses from the American Presidency Project."""
    print("Scraping inaugural addresses from UCSB...")

    session = requests.Session()
    # Get the listing page
    url = f"{UCSB_BASE}/documents/app-categories/spoken-addresses-and-remarks/presidential/inaugural-addresses"
    resp = session.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find document links
    doc_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/documents/" in href and href != url:
            title = a.get_text(strip=True)
            if title and ("inaugural" in title.lower() or "address" in title.lower()):
                full_url = href if href.startswith("http") else f"{UCSB_BASE}{href}"
                doc_links.append((title, full_url))

    # Deduplicate
    doc_links = list(dict(doc_links).items())
    print(f"  Found {len(doc_links)} inaugural address links")

    # If listing didn't work well, try paginated search
    if len(doc_links) < 20:
        print("  Trying search approach...")
        doc_links = []
        page = 0
        while True:
            search_url = f"{UCSB_BASE}/advanced-search"
            params = {
                "field-keywords": "",
                "field-keywords2": "",
                "field-keywords3": "",
                "from[date]": "",
                "to[date]": "",
                "category": "46",  # Inaugural addresses category
                "items_per_page": 100,
                "page": page,
            }
            resp = session.get(search_url, params=params, headers=HEADERS, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")

            found = 0
            for div in soup.find_all("div", class_="views-row"):
                a = div.find("a")
                if a and a.get("href"):
                    href = a["href"]
                    title = a.get_text(strip=True)
                    full_url = href if href.startswith("http") else f"{UCSB_BASE}{href}"
                    doc_links.append((title, full_url))
                    found += 1

            if found == 0:
                break
            page += 1
            _jittered_sleep(1.0)

        print(f"  Found {len(doc_links)} via search")

    rows = []
    for title, doc_url in doc_links:
        _jittered_sleep(1.5)

        try:
            resp = session.get(doc_url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract text from the document body
            body = soup.find("div", class_="field-docs-content")
            if not body:
                body = soup.find("div", class_="field--name-field-docs-content")
            text = body.get_text(separator="\n").strip() if body else ""

            # Extract date
            date_el = soup.find("span", class_="date-display-single")
            date_str = date_el.get_text(strip=True) if date_el else ""
            date = ""
            if date_str:
                for fmt in ["%B %d, %Y", "%b %d, %Y", "%B %d %Y"]:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        date = dt.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue

            # Extract speaker
            speaker_el = soup.find("h3", class_="diet-title")
            if not speaker_el:
                speaker_el = soup.find("div", class_="field-title")
            speaker = speaker_el.get_text(strip=True) if speaker_el else ""

            if text:
                rows.append({
                    "date": date,
                    "speaker": speaker,
                    "party": PRESIDENT_PARTIES.get(speaker, ""),
                    "speech_type": "Inaugural Address",
                    "title": title,
                    "text": text,
                    "url": doc_url,
                    "year": int(date[:4]) if date and len(date) >= 4 else 0,
                })

        except Exception as e:
            print(f"  [warn] Failed for {title}: {e}")
            continue

    print(f"  Scraped {len(rows)} inaugural addresses")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Opposition SOTU responses (scrape from UCSB)
# ---------------------------------------------------------------------------

def scrape_opposition_responses() -> pd.DataFrame:
    """Scrape opposition party SOTU responses from UCSB."""
    print("Scraping opposition SOTU responses from UCSB...")

    session = requests.Session()
    rows = []

    # Search UCSB for opposition/response speeches
    for search_term in [
        "response to the state of the union",
        "republican response state union",
        "democratic response state union",
        "opposition response state union",
        "response to the president address",
    ]:
        page = 0
        while page < 5:  # Limit pages per search
            resp = session.get(
                f"{UCSB_BASE}/advanced-search",
                params={
                    "field-keywords": search_term,
                    "items_per_page": 50,
                    "page": page,
                },
                headers=HEADERS,
                timeout=30,
            )
            soup = BeautifulSoup(resp.text, "html.parser")

            found = 0
            for div in soup.find_all("div", class_="views-row"):
                a = div.find("a")
                if not a or not a.get("href"):
                    continue

                title = a.get_text(strip=True)
                # Filter for actual response speeches
                title_lower = title.lower()
                if "response" not in title_lower and "reply" not in title_lower:
                    continue

                href = a["href"]
                full_url = href if href.startswith("http") else f"{UCSB_BASE}{href}"

                # Skip if already collected
                if any(r["url"] == full_url for r in rows):
                    continue

                _jittered_sleep(1.5)

                try:
                    doc_resp = session.get(full_url, headers=HEADERS, timeout=30)
                    if doc_resp.status_code != 200:
                        continue

                    doc_soup = BeautifulSoup(doc_resp.text, "html.parser")

                    body = doc_soup.find("div", class_="field-docs-content")
                    if not body:
                        body = doc_soup.find("div", class_="field--name-field-docs-content")
                    text = body.get_text(separator="\n").strip() if body else ""

                    date_el = doc_soup.find("span", class_="date-display-single")
                    date_str = date_el.get_text(strip=True) if date_el else ""
                    date = ""
                    if date_str:
                        for fmt in ["%B %d, %Y", "%b %d, %Y"]:
                            try:
                                dt = datetime.strptime(date_str, fmt)
                                date = dt.strftime("%Y-%m-%d")
                                break
                            except ValueError:
                                continue

                    speaker_el = doc_soup.find("h3", class_="diet-title")
                    if not speaker_el:
                        speaker_el = doc_soup.find("div", class_="field-title")
                    speaker = speaker_el.get_text(strip=True) if speaker_el else ""

                    # Determine party from context
                    party = ""
                    if "republican" in title_lower:
                        party = "Republican"
                    elif "democrat" in title_lower:
                        party = "Democratic"
                    elif speaker:
                        party = PRESIDENT_PARTIES.get(speaker, "")

                    if text:
                        rows.append({
                            "date": date,
                            "speaker": speaker,
                            "party": party,
                            "speech_type": "Opposition SOTU Response",
                            "title": title,
                            "text": text,
                            "url": full_url,
                            "year": int(date[:4]) if date and len(date) >= 4 else 0,
                        })
                        found += 1

                except Exception as e:
                    print(f"  [warn] Failed: {e}")

            if found == 0:
                break
            page += 1
            _jittered_sleep(1.0)

    print(f"  Scraped {len(rows)} opposition responses")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build and push
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """Build the unified presidential speeches dataset."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SOTU
    sotu_df = load_sotu()
    sotu_df.to_parquet(CHECKPOINT_DIR / "sotu.parquet", index=False)

    # 2. Inaugurals
    inaugural_df = scrape_inaugurals()
    inaugural_df.to_parquet(CHECKPOINT_DIR / "inaugurals.parquet", index=False)

    # 3. Opposition responses
    response_df = scrape_opposition_responses()
    response_df.to_parquet(CHECKPOINT_DIR / "opposition_responses.parquet", index=False)

    # Combine
    dfs = [sotu_df]
    if not inaugural_df.empty:
        dfs.append(inaugural_df)
    if not response_df.empty:
        dfs.append(response_df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "speech_type", "speaker"], keep="first")
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
        raise RuntimeError("No HuggingFace token found")

    api = HfApi(token=token)
    api.whoami()
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Build presidential speeches dataset")
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--repo-id", default="chrissoria/presidential-speeches")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.push_only:
        final = CHECKPOINT_DIR / "speeches_final_dataset.parquet"
        if final.exists():
            df = pd.read_parquet(final)
        else:
            print("No dataset found.")
            return
    else:
        df = build_dataset()

    print(f"\n=== Presidential Speeches ===")
    print(f"Total: {len(df)}")
    print(f"By type:")
    for t, c in df["speech_type"].value_counts().items():
        print(f"  {t}: {c}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Speakers: {df['speaker'].nunique()}")
    print(f"With text: {(df['text'].str.len() > 0).sum()}")

    final = CHECKPOINT_DIR / "speeches_final_dataset.parquet"
    df.to_parquet(final, index=False)
    print(f"Saved to {final}")

    push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
