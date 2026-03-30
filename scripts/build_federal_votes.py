#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Build the federal-votes dataset on HuggingFace.

Fetches individual roll call votes from clerk.house.gov and senate.gov
for bills in the federal-bills-active and federal-laws datasets.

One row per legislator per vote (alter-level data).

Usage:
    python build_federal_votes.py                # Build from bills cache
    python build_federal_votes.py --dry-run      # Don't push to HuggingFace
    python build_federal_votes.py --limit 50     # Only process first N bills with votes

Data sources:
    - House: https://clerk.house.gov/evs/{year}/roll{number}.xml
    - Senate: https://www.senate.gov/legislative/LIS/roll_call_votes/vote{congress}{session}/vote_{congress}_{session}_{number}.xml

HuggingFace repo: chrissoria/federal-votes
"""

import argparse
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
BILLS_CACHE = SCRIPT_DIR / ".federal_bills_cache.parquet"
LAWS_CACHE = SCRIPT_DIR / ".federal_laws_cache.parquet"
VOTES_CACHE = SCRIPT_DIR / ".federal_votes_cache.parquet"
LOG_DIR = SCRIPT_DIR / "logs"
HF_REPO = "chrissoria/federal-votes"

API_BASE = "https://api.congress.gov/v3"
API_KEY = os.getenv("CONGRESS_API_KEY") or os.getenv("GOVINFO_API_KEY") or os.getenv("DATA_GOV_API_KEY") or "DEMO_KEY"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "federal_votes.log", mode="a"),
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


def fetch_vote_actions(congress, bill_type, bill_number):
    """Get recorded vote URLs from a bill's action history."""
    data = api_get(f"/bill/{congress}/{bill_type.lower()}/{bill_number}/actions",
                   params={"limit": 100})
    if not data or "actions" not in data:
        return []

    votes = []
    for action in data["actions"]:
        recorded = action.get("recordedVotes", [])
        for rv in recorded:
            votes.append({
                "chamber": rv.get("chamber", ""),
                "roll_number": rv.get("rollNumber"),
                "session": rv.get("sessionNumber"),
                "date": rv.get("date", ""),
                "url": rv.get("url", ""),
                "question": action.get("text", ""),
            })
    return votes


def parse_house_roll_call(xml_url):
    """Parse a House roll call XML file into individual vote rows."""
    try:
        resp = requests.get(xml_url, timeout=30)
        if resp.status_code != 200:
            return [], {}
        root = ET.fromstring(resp.text)
    except Exception as e:
        log.warning(f"Failed to parse House XML {xml_url}: {e}")
        return [], {}

    # Metadata
    meta = {}
    q = root.find('.//vote-question')
    meta["vote_question"] = q.text.strip() if q is not None and q.text else ""
    r = root.find('.//vote-result')
    meta["vote_result"] = r.text.strip() if r is not None and r.text else ""

    # Totals
    for tag in root.findall('.//vote-totals/totals-by-vote/total-by-vote'):
        label = tag.find('total-by-vote-label')
        count = tag.find('total-by-vote-count')
        if label is not None and count is not None:
            key = label.text.strip().lower()
            if key in ("yea", "aye"):
                meta["yeas_total"] = int(count.text)
            elif key in ("nay", "no"):
                meta["nays_total"] = int(count.text)
            elif key == "present":
                meta["present_total"] = int(count.text)
            elif key == "not voting":
                meta["not_voting_total"] = int(count.text)

    # Individual votes
    rows = []
    for rv in root.findall('.//recorded-vote'):
        leg = rv.find('legislator')
        vote = rv.find('vote')
        if leg is not None and vote is not None:
            rows.append({
                "legislator_name": leg.text.strip() if leg.text else "",
                "legislator_id": leg.get("name-id", ""),
                "party": leg.get("party", ""),
                "state": leg.get("state", ""),
                "district": leg.get("district", ""),
                "vote": vote.text.strip() if vote.text else "",
            })

    return rows, meta


def parse_senate_roll_call(xml_url):
    """Parse a Senate roll call XML file into individual vote rows."""
    try:
        resp = requests.get(xml_url, timeout=30)
        if resp.status_code != 200:
            return [], {}
        root = ET.fromstring(resp.text)
    except Exception as e:
        log.warning(f"Failed to parse Senate XML {xml_url}: {e}")
        return [], {}

    meta = {}
    q = root.find('.//question')
    meta["vote_question"] = q.text.strip() if q is not None and q.text else ""
    r = root.find('.//result')
    meta["vote_result"] = r.text.strip() if r is not None and r.text else ""

    # Totals
    yeas = root.find('.//count/yeas')
    nays = root.find('.//count/nays')
    if yeas is not None:
        meta["yeas_total"] = int(yeas.text)
    if nays is not None:
        meta["nays_total"] = int(nays.text)

    # Individual votes
    rows = []
    for member in root.findall('.//members/member'):
        name_el = member.find('last_name')
        first_el = member.find('first_name')
        party_el = member.find('party')
        state_el = member.find('state')
        vote_el = member.find('vote_cast')
        lis_id = member.find('lis_member_id')

        last = name_el.text.strip() if name_el is not None and name_el.text else ""
        first = first_el.text.strip() if first_el is not None and first_el.text else ""

        rows.append({
            "legislator_name": f"{last}, {first}" if first else last,
            "legislator_id": lis_id.text.strip() if lis_id is not None and lis_id.text else "",
            "party": party_el.text.strip() if party_el is not None and party_el.text else "",
            "state": state_el.text.strip() if state_el is not None and state_el.text else "",
            "district": "",
            "vote": vote_el.text.strip() if vote_el is not None and vote_el.text else "",
        })

    return rows, meta


def process_bill_votes(congress, bill_type, bill_number):
    """Fetch and parse all recorded votes for a bill."""
    vote_actions = fetch_vote_actions(congress, bill_type, bill_number)
    if not vote_actions:
        return []

    all_rows = []
    for va in vote_actions:
        url = va.get("url", "")
        chamber = va.get("chamber", "")
        roll_number = va.get("roll_number")

        if not url:
            continue

        if chamber == "House":
            individual_votes, meta = parse_house_roll_call(url)
        elif chamber == "Senate":
            individual_votes, meta = parse_senate_roll_call(url)
        else:
            continue

        time.sleep(0.3)

        for iv in individual_votes:
            iv.update({
                "bill_type": bill_type,
                "bill_number": str(bill_number),
                "congress": str(congress),
                "chamber": chamber,
                "roll_number": roll_number,
                "vote_date": va.get("date", "")[:10],
                "vote_question": meta.get("vote_question", va.get("question", "")),
                "vote_result": meta.get("vote_result", ""),
                "yeas_total": meta.get("yeas_total", None),
                "nays_total": meta.get("nays_total", None),
            })
            all_rows.append(iv)

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Build federal-votes dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max bills to process")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("Federal Votes builder starting")

    # Load bills from cache
    bills_dfs = []
    if BILLS_CACHE.exists():
        bills_dfs.append(pd.read_parquet(BILLS_CACHE))
        log.info(f"Loaded {len(bills_dfs[-1])} bills from bills cache")

    if not bills_dfs:
        log.error("No bills cache found. Run build_federal_bills.py first.")
        sys.exit(1)

    bills = pd.concat(bills_dfs, ignore_index=True)

    # Check all bills beyond "Introduced" for recorded votes — the API returns
    # empty for bills without roll calls, so we cast a wide net to catch
    # procedural votes on bills still in progress (e.g., cloture, motions).
    skip_statuses = {"Introduced"}
    bills_with_votes = bills[~bills["status"].isin(skip_statuses)]
    log.info(f"Bills to check for votes: {len(bills_with_votes)}")

    if args.limit:
        bills_with_votes = bills_with_votes.head(args.limit)

    # Load existing votes to skip already processed bills
    existing_votes = pd.DataFrame()
    processed_bills = set()
    if VOTES_CACHE.exists():
        existing_votes = pd.read_parquet(VOTES_CACHE)
        processed_bills = set(zip(existing_votes["bill_type"], existing_votes["bill_number"]))
        log.info(f"Existing votes cache: {len(existing_votes)} rows, {len(processed_bills)} bills")

    # Process each bill
    all_new_rows = []
    for i, (_, bill) in enumerate(bills_with_votes.iterrows()):
        bt = bill.get("bill_type", "HR")
        bn = str(bill.get("bill_number", ""))

        if (bt, bn) in processed_bills:
            continue

        if i > 0 and i % 10 == 0:
            log.info(f"Processing bill {i}/{len(bills_with_votes)}: {bt} {bn}")

        rows = process_bill_votes(119, bt, bn)
        if rows:
            all_new_rows.extend(rows)
            log.info(f"  {bt} {bn}: {len(rows)} individual votes")

        time.sleep(0.5)

    if not all_new_rows:
        log.info("No new votes to add.")
        if not existing_votes.empty:
            log.info(f"Existing dataset has {len(existing_votes)} vote records.")
        return

    new_df = pd.DataFrame(all_new_rows)
    log.info(f"Fetched {len(new_df)} new vote records")

    # Combine with existing
    if not existing_votes.empty:
        combined = pd.concat([existing_votes, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["bill_type", "bill_number", "congress", "chamber",
                     "roll_number", "legislator_id"],
            keep="last"
        )
    else:
        combined = new_df

    combined = combined.sort_values(["vote_date", "bill_number"], ascending=[False, True]).reset_index(drop=True)

    # Summary
    log.info(f"Total vote records: {len(combined)}")
    if "party" in combined.columns:
        log.info("Party breakdown:")
        for party, count in combined["party"].value_counts().items():
            log.info(f"  {party}: {count}")
    if "chamber" in combined.columns:
        log.info("Chamber breakdown:")
        for chamber, count in combined["chamber"].value_counts().items():
            log.info(f"  {chamber}: {count}")

    combined.to_parquet(VOTES_CACHE, index=False)
    log.info(f"Cached {len(combined)} vote records")

    if not args.dry_run:
        from datasets import Dataset
        ds = Dataset.from_pandas(combined)
        ds.push_to_hub(HF_REPO, private=False)
        log.info(f"Pushed {len(combined)} rows to https://huggingface.co/datasets/{HF_REPO}")
    else:
        log.info("Dry run — skipping HuggingFace push")

    log.info("Done.")


if __name__ == "__main__":
    main()
