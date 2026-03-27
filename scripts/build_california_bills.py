#!/usr/bin/env python3
"""Scrape California state bills from leginfo.legislature.ca.gov.

Scrapes bill text, title, author, status, and dates from the official
California Legislative Information website (public domain).

Usage:
    python scripts/build_california_bills.py                          # Current session
    python scripts/build_california_bills.py --session 20232024       # Specific session
    python scripts/build_california_bills.py --session 20252026 --push # Scrape and push to HF
    python scripts/build_california_bills.py --max-bills 50           # Limit for testing

Sessions available: 20252026, 20232024, 20212022, 20192020, 20172018,
                    20152016, 20132014, 20112012, 20092010, 20072008,
                    20052006, 20032004, 20012002, 19992000
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPTS_DIR / "checkpoints" / "ca_bills"
LOG_DIR = SCRIPTS_DIR / "logs"

load_dotenv(PROJECT_ROOT / ".env", override=True)

BASE_URL = "https://leginfo.legislature.ca.gov/faces"
SEARCH_URL = f"{BASE_URL}/billSearchClient.xhtml"
TEXT_URL = f"{BASE_URL}/billTextClient.xhtml"
STATUS_URL = f"{BASE_URL}/billStatusClient.xhtml"
HISTORY_URL = f"{BASE_URL}/billHistoryClient.xhtml"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests

HF_TOKEN = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
HF_REPO = "chrissoria/california-bills"

SESSIONS = [
    "20252026", "20232024", "20212022", "20192020", "20172018",
    "20152016", "20132014", "20112012", "20092010", "20072008",
    "20052006", "20032004", "20012002", "19992000",
]


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ca_bills_scraper")
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_DIR / "ca_bills.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(ch)

    return log


def get_bill_list(session: str, log: logging.Logger) -> list[dict]:
    """Get list of all bill IDs for a session by scraping search results."""
    log.info(f"Fetching bill list for session {session}...")

    session_obj = requests.Session()
    session_obj.headers.update({
        "User-Agent": "Mozilla/5.0 (research bot; UC Berkeley; chrissoria@berkeley.edu)"
    })

    # The search page uses JSF, so we need to scrape the rendered HTML
    # Try fetching the search results page
    params = {
        "session_year": session,
        "house": "Both",
        "lawCode": "All",
        "author": "All",
        "search_type": "search_all",
    }

    resp = session_obj.get(SEARCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract bill links from the results table
    bills = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "bill_id=" in href:
            bill_id_match = re.search(r'bill_id=(\d+[A-Z]+\d+)', href)
            if bill_id_match:
                bill_id = bill_id_match.group(1)
                bill_label = link.get_text(strip=True)
                if bill_id not in [b["bill_id"] for b in bills]:
                    bills.append({
                        "bill_id": bill_id,
                        "bill_label": bill_label,
                    })

    log.info(f"Found {len(bills)} bills for session {session}")
    return bills


def scrape_bill_text(bill_id: str, session_obj: requests.Session) -> dict:
    """Scrape a single bill's text and metadata."""
    url = f"{TEXT_URL}?bill_id={bill_id}"
    try:
        resp = session_obj.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")

    result = {
        "bill_id": bill_id,
        "url": url,
    }

    page_text = soup.get_text()

    # Extract title and bill number from h1: "AB-1 Residential property insurance: wildfire risk.(2025-2026)"
    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(strip=True)
        # Skip the generic "Bill Text" h1
        all_h1s = soup.find_all("h1")
        for h in all_h1s:
            t = h.get_text(strip=True)
            if t != "Bill Text" and re.match(r'[A-Z]{2,}', t):
                h1_text = t
                break

        # Parse: "AB-1 Residential property insurance: wildfire risk.(2025-2026)"
        h1_match = re.match(r'([A-Z]+-\d+)\s+(.*?)(?:\(\d{4}-\d{4}\))?$', h1_text)
        if h1_match:
            result["bill_number"] = h1_match.group(1)
            result["title"] = h1_match.group(2).strip().rstrip(".")
        else:
            result["bill_number"] = h1_text.split()[0] if h1_text else bill_id
            result["title"] = " ".join(h1_text.split()[1:]).rstrip(".") if h1_text else ""
    else:
        result["bill_number"] = bill_id
        result["title"] = ""

    # Extract author from bill text (usually "INTRODUCED BY" line)
    author_match = re.search(
        r'(?:Introduced by|INTRODUCED BY)\s+(?:Assembly Member|Senator|Committee on)s?\s+(.+?)(?:\n|$)',
        page_text
    )
    if author_match:
        result["author"] = author_match.group(1).strip().rstrip(".")
    else:
        result["author"] = ""

    # Extract bill text from the main content area
    bill_content = soup.find("div", {"id": "bill_all"})
    if bill_content:
        result["text"] = bill_content.get_text(separator="\n", strip=True)
    else:
        # Fallback: get everything between "Bill Start" and end
        result["text"] = ""
        in_bill = False
        parts = []
        for el in soup.find_all(["h2", "p", "span", "div"]):
            t = el.get_text(strip=True)
            if "Bill Start" in t:
                in_bill = True
                continue
            if in_bill and t:
                parts.append(t)
        result["text"] = "\n".join(parts) if parts else page_text[:5000]

    # Extract version/status from the version dropdown
    # Versions appear as: "10/09/25 - Chaptered", "09/15/25 - Enrolled", etc.
    version_match = re.search(r'(\d{2}/\d{2}/\d{2})\s*-\s*(Chaptered|Enrolled|Vetoed|Amended|Introduced)', page_text)
    if version_match:
        result["status"] = version_match.group(2)
        try:
            result["date"] = pd.to_datetime(version_match.group(1), format="%m/%d/%y").strftime("%Y-%m-%d")
        except Exception:
            pass
    else:
        if "Chaptered" in page_text or "CHAPTERED" in page_text:
            result["status"] = "Chaptered"
        elif "Vetoed" in page_text:
            result["status"] = "Vetoed"
        else:
            result["status"] = "In Progress"

    # Extract introduced date
    intro_match = re.search(r'(\d{2}/\d{2}/\d{2})\s*-\s*Introduced', page_text)
    if intro_match:
        try:
            result["date_introduced"] = pd.to_datetime(intro_match.group(1), format="%m/%d/%y").strftime("%Y-%m-%d")
        except Exception:
            pass

    return result


def scrape_session(session: str, log: logging.Logger, max_bills: int = None) -> pd.DataFrame:
    """Scrape all bills for a session."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{session}.parquet"

    # Load existing checkpoint
    existing = pd.DataFrame()
    if checkpoint_file.exists():
        existing = pd.read_parquet(checkpoint_file)
        log.info(f"Loaded {len(existing)} bills from checkpoint")
    existing_ids = set(existing["bill_id"].tolist()) if not existing.empty else set()

    # Get bill list
    bills = get_bill_list(session, log)
    if max_bills:
        bills = bills[:max_bills]

    # Filter to unscraped
    to_scrape = [b for b in bills if b["bill_id"] not in existing_ids]
    log.info(f"Need to scrape {len(to_scrape)} bills ({len(existing_ids)} already done)")

    if not to_scrape:
        return existing

    session_obj = requests.Session()
    session_obj.headers.update({
        "User-Agent": "Mozilla/5.0 (research bot; UC Berkeley; chrissoria@berkeley.edu)"
    })

    rows = existing.to_dict("records") if not existing.empty else []
    for i, bill in enumerate(to_scrape):
        log.info(f"[{i+1}/{len(to_scrape)}] Scraping {bill['bill_label']}...")
        data = scrape_bill_text(bill["bill_id"], session_obj)

        if "error" in data:
            log.warning(f"  Failed: {data['error']}")
        else:
            data["session"] = session
            data["session_label"] = f"{session[:4]}-{session[4:]}"
            rows.append(data)

        # Checkpoint every 50 bills
        if (i + 1) % 50 == 0:
            df = pd.DataFrame(rows)
            df.to_parquet(checkpoint_file, index=False)
            log.info(f"  Checkpoint: {len(df)} bills saved")

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    df.to_parquet(checkpoint_file, index=False)
    log.info(f"Session {session}: {len(df)} bills total")
    return df


def push_to_hf(df: pd.DataFrame, log: logging.Logger):
    """Push dataset to HuggingFace."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        log.error("datasets/huggingface_hub not installed")
        return

    if not HF_TOKEN:
        log.error("No HuggingFace token found")
        return

    ds = Dataset.from_pandas(df)
    ds.push_to_hub(HF_REPO, token=HF_TOKEN, private=False)
    log.info(f"Pushed {len(df)} bills to https://huggingface.co/datasets/{HF_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Scrape California state bills")
    parser.add_argument("--session", default="20252026", choices=SESSIONS,
                        help="Legislative session to scrape (default: 20252026)")
    parser.add_argument("--all-sessions", action="store_true",
                        help="Scrape all sessions from 1999 to present")
    parser.add_argument("--max-bills", type=int, default=None,
                        help="Max bills to scrape (for testing)")
    parser.add_argument("--push", action="store_true",
                        help="Push to HuggingFace after scraping")
    args = parser.parse_args()

    log = setup_logging()

    if args.all_sessions:
        all_dfs = []
        for session in SESSIONS:
            df = scrape_session(session, log, max_bills=args.max_bills)
            all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True)
        log.info(f"Total: {len(combined)} bills across {len(SESSIONS)} sessions")
    else:
        combined = scrape_session(args.session, log, max_bills=args.max_bills)

    if args.push:
        push_to_hf(combined, log)

    log.info("Done.")


if __name__ == "__main__":
    main()
