#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Build and maintain the federal-bills-active dataset on HuggingFace.

Fetches active bills from the 119th Congress (2025-2026) via the Congress.gov API,
mirrors the federal_laws schema where possible, and adds status tracking columns.

Usage:
    python build_federal_bills.py                # Full build
    python build_federal_bills.py --incremental  # Only fetch recently updated bills
    python build_federal_bills.py --dry-run      # Don't push to HuggingFace

Data source: https://api.congress.gov/v3/
HuggingFace repo: chrissoria/federal-bills-active
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
CACHE_FILE = SCRIPT_DIR / ".federal_bills_cache.parquet"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints" / "federal_bills"
LOG_DIR = SCRIPT_DIR / "logs"
HF_REPO = "chrissoria/federal-bills-active"
CONGRESS = 119  # 119th Congress (2025-2026), started Jan 3 2025 (Trump inauguration Jan 20 2025)
SINCE_DATE = "2025-01-20"  # Trump inauguration

API_BASE = "https://api.congress.gov/v3"
API_KEY = os.getenv("CONGRESS_API_KEY") or os.getenv("GOVINFO_API_KEY") or os.getenv("DATA_GOV_API_KEY") or "DEMO_KEY"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "federal_bills.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def api_get(endpoint, params=None, retries=3):
    """Make a Congress.gov API request with retry logic."""
    params = params or {}
    params["api_key"] = API_KEY
    params["format"] = "json"

    for attempt in range(retries):
        try:
            resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            log.error(f"API request failed: {endpoint} — {e}")
            return None
    return None


def fetch_bill_list(congress=CONGRESS, limit=250, offset=0, from_date=None):
    """Fetch a page of bills from Congress.gov."""
    params = {"limit": limit, "offset": offset}
    if from_date:
        params["fromDateTime"] = f"{from_date}T00:00:00Z"
    data = api_get(f"/bill/{congress}", params=params)
    if data and "bills" in data:
        return data["bills"], data.get("pagination", {}).get("count", 0)
    return [], 0


def fetch_bill_detail(congress, bill_type, bill_number):
    """Fetch detailed info for a single bill."""
    data = api_get(f"/bill/{congress}/{bill_type.lower()}/{bill_number}")
    if data and "bill" in data:
        return data["bill"]
    return None


def fetch_bill_text(congress, bill_type, bill_number):
    """Fetch the latest text version of a bill."""
    data = api_get(f"/bill/{congress}/{bill_type.lower()}/{bill_number}/text")
    if data and "textVersions" in data:
        versions = data["textVersions"]
        if versions:
            # Get the latest version
            latest = versions[0]
            for fmt in latest.get("formats", []):
                if fmt.get("type") == "Formatted Text":
                    try:
                        resp = requests.get(fmt["url"], timeout=30,
                                            params={"api_key": API_KEY})
                        if resp.status_code == 200:
                            import re
                            clean = re.sub(r'<html><body><pre>|</pre></body></html>', '', resp.text).strip()
                            return clean, latest.get("type", "")
                    except Exception:
                        pass
    return "", ""


def fetch_bill_subjects(congress, bill_type, bill_number):
    """Fetch subjects/topics for a bill."""
    data = api_get(f"/bill/{congress}/{bill_type.lower()}/{bill_number}/subjects")
    if data and "subjects" in data:
        subjects = data["subjects"]
        if "legislativeSubjects" in subjects:
            return [s["name"] for s in subjects["legislativeSubjects"]]
    return []


def fetch_bill_actions(congress, bill_type, bill_number):
    """Fetch action history for a bill."""
    data = api_get(f"/bill/{congress}/{bill_type.lower()}/{bill_number}/actions",
                   params={"limit": 100})
    if data and "actions" in data:
        return data["actions"]
    return []


def determine_status(actions):
    """Determine bill status from its action history."""
    if not actions:
        return "Introduced"

    action_texts = [a.get("text", "").lower() for a in actions]

    for text in action_texts:
        if "became public law" in text or "signed by president" in text:
            return "Signed into Law"
        if "passed/agreed to in senate" in text and "passed/agreed to in house" in text:
            return "Passed Both Chambers"
        if "presented to president" in text or "sent to president" in text:
            return "Sent to President"
        if "vetoed" in text:
            return "Vetoed"

    for text in action_texts:
        if "passed/agreed to in senate" in text:
            return "Passed Senate"
        if "passed/agreed to in house" in text or "passed house" in text:
            return "Passed House"
        if "cloture" in text:
            return "Senate Floor"
        if "placed on" in text and "calendar" in text:
            return "Calendared"
        if "reported by" in text or "ordered to be reported" in text:
            return "Reported from Committee"

    for text in action_texts:
        if "referred to" in text and "committee" in text:
            return "In Committee"

    return "Introduced"


def build_bill_row(bill_summary, detail, text_content, text_version, subjects, actions):
    """Build a row dict mirroring the federal_laws schema + status columns."""
    sponsor = detail.get("sponsors", [{}])[0] if detail.get("sponsors") else {}
    policy_area = detail.get("policyArea", {})
    latest_action = detail.get("latestAction", {})

    bill_type = bill_summary.get("type", "HR").lower()
    bill_number = bill_summary.get("number", "")

    return {
        # Dates
        "date_last_action": latest_action.get("actionDate", ""),
        "date_introduced": detail.get("introducedDate", ""),
        "date_updated": detail.get("updateDate", bill_summary.get("updateDate", "")),
        # Bill identity
        "bill_type": bill_summary.get("type", ""),
        "bill_number": bill_number,
        "congress": str(detail.get("congress", CONGRESS)),
        "title": detail.get("title", bill_summary.get("title", "")),
        "short_title": "",
        "status": determine_status(actions),
        "last_action_text": latest_action.get("text", ""),
        # Sponsor
        "sponsor_full_name": sponsor.get("fullName", ""),
        "sponsor_party": sponsor.get("party", ""),
        "sponsor_state": sponsor.get("state", ""),
        "num_cosponsors": detail.get("cosponsors", {}).get("count", 0),
        "chamber_of_origin": detail.get("originChamber", bill_summary.get("originChamber", "")),
        # Classification
        "policy_area": policy_area.get("name", "") if isinstance(policy_area, dict) else "",
        "subjects": "; ".join(subjects) if subjects else "",
        # Vote data (populated later)
        "republican_yeas": None,
        "democrat_yeas": None,
        "republican_nays": None,
        "democrat_nays": None,
        "total_yeas": None,
        "total_nays": None,
        "republican_support_pct": None,
        "democrat_support_pct": None,
        "is_bipartisan": None,
        # Text and metadata
        "text": text_content,
        "bill_text_version": text_version,
        "url": detail.get("legislationUrl", f"https://www.congress.gov/bill/{CONGRESS}th-congress/{bill_type}-bill/{bill_number}"),
        "pdf_url": "",
        "doc_type": "bill",
        "public_law_number": "",
        "package_id": "",
        "pages": "",
        "source": "federal_bills_active",
    }


def load_cache():
    """Load existing cached data."""
    if CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)
    return pd.DataFrame()


def save_cache(df):
    """Save data to local cache."""
    df.to_parquet(CACHE_FILE, index=False)


def push_to_hf(df):
    """Push dataset to HuggingFace."""
    from datasets import Dataset
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(HF_REPO, private=False)
    log.info(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{HF_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Build federal-bills-active dataset")
    parser.add_argument("--incremental", action="store_true",
                        help="Only fetch recently updated bills")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't push to HuggingFace")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max bills to fetch (for testing)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"Federal Bills builder starting (congress={CONGRESS})")

    existing = load_cache()
    existing_bills = set()
    if not existing.empty and "bill_number" in existing.columns and "bill_type" in existing.columns:
        existing_bills = set(zip(existing["bill_type"], existing["bill_number"]))
    log.info(f"Existing cache: {len(existing)} bills")

    # Fetch bill listing from Congress.gov
    all_bills = []
    offset = 0
    page_size = 250
    total = None

    while True:
        bills, count = fetch_bill_list(congress=CONGRESS, limit=page_size, offset=offset)
        if total is None:
            total = count
            log.info(f"Total bills in {CONGRESS}th Congress: {total}")

        if not bills:
            break

        all_bills.extend(bills)
        offset += page_size
        log.info(f"Fetched {len(all_bills)}/{total} bill listings...")

        if args.limit and len(all_bills) >= args.limit:
            all_bills = all_bills[:args.limit]
            break

        if offset >= total:
            break

        time.sleep(0.5)  # Rate limiting

    log.info(f"Total bill listings: {len(all_bills)}")

    # Filter to bills needing detail fetch
    if args.incremental and not existing.empty:
        # Only fetch bills updated since our last run
        last_update = existing["date_updated"].max() if "date_updated" in existing.columns else ""
        bills_to_fetch = [
            b for b in all_bills
            if b.get("updateDate", "") > last_update
               or (b.get("type", ""), str(b.get("number", ""))) not in existing_bills
        ]
        log.info(f"Incremental: {len(bills_to_fetch)} bills updated since {last_update}")
    else:
        bills_to_fetch = all_bills

    # Fetch details for each bill
    rows = []
    for i, bill in enumerate(bills_to_fetch):
        bill_type = bill.get("type", "HR").lower()
        bill_number = bill.get("number", "")

        if i > 0 and i % 50 == 0:
            log.info(f"Processing {i}/{len(bills_to_fetch)}...")

        detail = fetch_bill_detail(CONGRESS, bill_type, bill_number)
        if not detail:
            continue

        time.sleep(0.3)

        text_content, text_version = fetch_bill_text(CONGRESS, bill_type, bill_number)
        time.sleep(0.3)

        subjects = fetch_bill_subjects(CONGRESS, bill_type, bill_number)
        time.sleep(0.3)

        actions = fetch_bill_actions(CONGRESS, bill_type, bill_number)
        time.sleep(0.3)

        row = build_bill_row(bill, detail, text_content, text_version, subjects, actions)
        rows.append(row)

        # Checkpoint every 100 bills
        if len(rows) % 100 == 0:
            checkpoint_df = pd.DataFrame(rows)
            checkpoint_df.to_parquet(CHECKPOINT_DIR / f"checkpoint_{len(rows)}.parquet", index=False)
            log.info(f"Checkpoint: {len(rows)} bills processed")

    if not rows:
        log.info("No new bills to process.")
        return

    new_df = pd.DataFrame(rows)
    log.info(f"Fetched details for {len(new_df)} bills")

    # Merge with existing data
    if not existing.empty:
        # Update existing rows, add new ones
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["bill_type", "bill_number"], keep="last"
        )
    else:
        combined = new_df

    # Normalize column name: API may use either variant
    if "last_action_date" in combined.columns and "date_last_action" not in combined.columns:
        combined = combined.rename(columns={"last_action_date": "date_last_action"})
    combined = combined.sort_values("date_last_action", ascending=False).reset_index(drop=True)

    # Status summary
    if "status" in combined.columns:
        log.info("Status breakdown:")
        for status, count in combined["status"].value_counts().items():
            log.info(f"  {status}: {count}")

    # =========================================================================
    # Generate Threads summaries for new/updated bills missing them
    # =========================================================================
    hf_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY")
    if hf_key and not args.dry_run:
        if "summary_threads" not in combined.columns:
            combined["summary_threads"] = None
        needs_summary = combined[combined["summary_threads"].isna() & (combined["text"].str.len() > 50)]
        if len(needs_summary) > 0:
            log.info(f"Generating Threads summaries for {len(needs_summary)} bills...")
            try:
                import cat_pol
                result = cat_pol.summarize(
                    input_data=needs_summary["text"].tolist(),
                    format="threads",
                    tone="eli5",
                    description="U.S. federal bills",
                    api_key=hf_key,
                    user_model="qwen/qwen3-vl-235b-a22b-instruct:novita",
                    model_source="huggingface",
                    creativity=0,
                )
                for i, (orig_idx, _) in enumerate(needs_summary.iterrows()):
                    if i < len(result) and result.iloc[i].get("processing_status") == "success":
                        combined.at[orig_idx, "summary_threads"] = result.iloc[i]["summary"]
                summarized = (result["processing_status"] == "success").sum()
                log.info(f"Generated {summarized}/{len(needs_summary)} Threads summaries")
            except Exception as e:
                log.warning(f"Threads summarization failed: {e}")
        else:
            log.info("All bills have Threads summaries. Up to date.")

        # Bullet-point summaries
        if "summary_bullets" not in combined.columns:
            combined["summary_bullets"] = None
        needs_bullets = combined[combined["summary_bullets"].isna() & (combined["text"].str.len() > 50)]
        if len(needs_bullets) > 0:
            log.info(f"Generating bullet summaries for {len(needs_bullets)} bills...")
            try:
                import cat_pol
                result_b = cat_pol.summarize(
                    input_data=needs_bullets["text"].tolist(),
                    format="bullets",
                    tone="eli5",
                    description="U.S. federal bills in the 119th Congress",
                    api_key=hf_key,
                    user_model="qwen/qwen3-vl-235b-a22b-instruct:novita",
                    model_source="huggingface",
                    creativity=0,
                )
                for i, (orig_idx, _) in enumerate(needs_bullets.iterrows()):
                    if i < len(result_b) and result_b.iloc[i].get("processing_status") == "success":
                        combined.at[orig_idx, "summary_bullets"] = result_b.iloc[i]["summary"]
                summarized_b = (result_b["processing_status"] == "success").sum()
                log.info(f"Generated {summarized_b}/{len(needs_bullets)} bullet summaries")
            except Exception as e:
                log.warning(f"Bullet summarization failed: {e}")
        else:
            log.info("All bills have bullet summaries. Up to date.")
    elif not hf_key:
        log.info("No HuggingFace key — skipping summaries.")

    save_cache(combined)
    log.info(f"Cached {len(combined)} total bills")

    if not args.dry_run:
        push_to_hf(combined)
    else:
        log.info("Dry run — skipping HuggingFace push")

    log.info("Done.")


if __name__ == "__main__":
    main()
