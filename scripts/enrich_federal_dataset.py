#!/usr/bin/env python3
"""Enrich federal public laws dataset with Congress.gov metadata.

Adds sponsor, cosponsors, party, policy area, bill type/number, and vote
information by cross-referencing the Congress.gov API. Designed to run as
a separate pass after build_federal_dataset.py finishes.

Usage:
    python scripts/enrich_federal_dataset.py
    python scripts/enrich_federal_dataset.py --congress 119    # Only one Congress
    python scripts/enrich_federal_dataset.py --merge-push      # Push enriched dataset to HF

Dependencies:
    pip install -r scripts/requirements.txt
    Requires GOVINFO_API_KEY in .env (same key works for Congress.gov API)
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
from dotenv import load_dotenv

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "federal"
CONGRESS_BASE = "https://api.congress.gov/v3"
ENRICHMENT_FILE = CHECKPOINT_DIR / "federal_enrichment.parquet"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}


def _load_api_key() -> str:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    key = os.getenv("GOVINFO_API_KEY") or os.getenv("DATA_GOV_API_KEY")
    if not key:
        key = "DEMO_KEY"
        print("[WARNING] No API key found — using DEMO_KEY")
    return key


def _jittered_sleep(base: float):
    time.sleep(base * (0.7 + 0.6 * random.random()))


def _api_get(url: str, api_key: str, max_retries: int = 3) -> dict | None:
    """GET with retry and rate-limit handling."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                params={"api_key": api_key, "format": "json"},
                headers=HEADERS,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                _jittered_sleep(3 * (attempt + 1))
            else:
                print(f"  API error: {e}")
    return None


# ---------------------------------------------------------------------------
# Fetch bill metadata for a single law
# ---------------------------------------------------------------------------

def fetch_bill_for_law(congress: int, law_number: int, api_key: str) -> dict | None:
    """Find the bill that became this public law via the /law endpoint."""
    data = _api_get(f"{CONGRESS_BASE}/law/{congress}", api_key)
    if not data:
        return None

    # Search through the law listing for our law number
    target = f"{congress}-{law_number}"
    bills = data.get("bills", [])

    for bill in bills:
        for law in bill.get("laws", []):
            if law.get("number") == target:
                return {
                    "bill_type": bill.get("type", ""),
                    "bill_number": bill.get("number", ""),
                    "bill_congress": bill.get("congress", ""),
                    "origin_chamber": bill.get("originChamber", ""),
                    "bill_url": bill.get("url", ""),
                }

    # Paginate if needed
    while data.get("pagination", {}).get("next"):
        _jittered_sleep(0.3)
        data = _api_get(data["pagination"]["next"], api_key)
        if not data:
            break
        for bill in data.get("bills", []):
            for law in bill.get("laws", []):
                if law.get("number") == target:
                    return {
                        "bill_type": bill.get("type", ""),
                        "bill_number": bill.get("number", ""),
                        "bill_congress": bill.get("congress", ""),
                        "origin_chamber": bill.get("originChamber", ""),
                        "bill_url": bill.get("url", ""),
                    }

    return None


def fetch_bill_details(bill_url: str, api_key: str) -> dict:
    """Fetch sponsor, cosponsor count, policy area from a bill URL."""
    result = {
        "sponsor_name": "",
        "sponsor_party": "",
        "sponsor_state": "",
        "cosponsor_count": 0,
        "policy_area": "",
        "introduced_date": "",
        "subjects": "",
    }

    data = _api_get(bill_url, api_key)
    if not data or "bill" not in data:
        return result

    bill = data["bill"]

    # Sponsor
    sponsors = bill.get("sponsors", [])
    if sponsors:
        sp = sponsors[0]
        result["sponsor_name"] = sp.get("fullName", "")
        result["sponsor_party"] = sp.get("party", "")
        result["sponsor_state"] = sp.get("state", "")

    # Cosponsors
    result["cosponsor_count"] = bill.get("cosponsors", {}).get("count", 0)

    # Policy area
    result["policy_area"] = bill.get("policyArea", {}).get("name", "")

    # Introduced date
    result["introduced_date"] = bill.get("introducedDate", "")

    # Subjects (fetch separately)
    subjects_url = bill.get("subjects", {}).get("url", "")
    if subjects_url:
        _jittered_sleep(0.5)
        subj_data = _api_get(subjects_url, api_key)
        if subj_data:
            subjects = subj_data.get("subjects", [])
            subj_names = [
                s.get("name", "") if isinstance(s, dict) else str(s)
                for s in subjects
                if (s.get("name") if isinstance(s, dict) else s)
            ]
            result["subjects"] = "; ".join(subj_names)

    return result


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

def enrich_dataset(api_key: str, congress_filter: int = None, api_delay: float = 1.0) -> pd.DataFrame:
    """Enrich the federal dataset with Congress.gov metadata."""
    # Load the base dataset
    base_path = CHECKPOINT_DIR / "federal_final_dataset.parquet"
    if not base_path.exists():
        print("No federal dataset found. Run build_federal_dataset.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(base_path)
    print(f"Loaded {len(df)} laws from base dataset")

    if congress_filter is not None:
        df = df[df["congress"].astype(str) == str(congress_filter)]
        print(f"Filtered to Congress {congress_filter}: {len(df)} laws")

    # Load existing enrichment data to skip already-done
    existing = {}
    if ENRICHMENT_FILE.exists():
        existing_df = pd.read_parquet(ENRICHMENT_FILE)
        existing = set(existing_df["package_id"].tolist())
        print(f"Already enriched: {len(existing)} laws")

    enrichment_rows = []
    # Also load existing rows to preserve
    if ENRICHMENT_FILE.exists():
        enrichment_rows = pd.read_parquet(ENRICHMENT_FILE).to_dict("records")

    new_count = 0
    for idx, row in df.iterrows():
        pid = row["package_id"]
        if pid in existing:
            continue

        # Parse congress and law number from package_id
        match = re.match(r"PLAW-(\d+)publ(\d+)", pid)
        if not match:
            continue

        congress = int(match.group(1))
        law_num = int(match.group(2))

        # Step 1: Find the bill
        bill_info = fetch_bill_for_law(congress, law_num, api_key)
        _jittered_sleep(api_delay)

        enrichment = {
            "package_id": pid,
            "bill_type": "",
            "bill_number": "",
            "origin_chamber": "",
            "sponsor_name": "",
            "sponsor_party": "",
            "sponsor_state": "",
            "cosponsor_count": 0,
            "policy_area": "",
            "introduced_date": "",
            "subjects": "",
        }

        if bill_info:
            enrichment.update({
                "bill_type": bill_info["bill_type"],
                "bill_number": bill_info["bill_number"],
                "origin_chamber": bill_info["origin_chamber"],
            })

            # Step 2: Get bill details (sponsor, policy area, etc.)
            if bill_info.get("bill_url"):
                details = fetch_bill_details(bill_info["bill_url"], api_key)
                _jittered_sleep(api_delay)
                enrichment.update(details)

        enrichment_rows.append(enrichment)
        new_count += 1

        if new_count % 25 == 0 or new_count == 1:
            print(f"  Enriched {new_count} new ({row['law_number']}): {enrichment['sponsor_name'] or '(no sponsor)'} — {enrichment['policy_area'] or '(no area)'}")

        # Checkpoint every 50
        if new_count % 50 == 0:
            edf = pd.DataFrame(enrichment_rows)
            edf.to_parquet(ENRICHMENT_FILE, index=False)
            print(f"  [checkpoint] Saved {len(edf)} enrichment rows")

    # Final save
    if new_count > 0:
        edf = pd.DataFrame(enrichment_rows)
        edf.to_parquet(ENRICHMENT_FILE, index=False)
        print(f"\nSaved {len(edf)} total enrichment rows ({new_count} new)")

    return pd.DataFrame(enrichment_rows)


def merge_and_push(repo_id: str):
    """Merge enrichment data into the base dataset and push to HF."""
    base_path = CHECKPOINT_DIR / "federal_final_dataset.parquet"
    if not base_path.exists() or not ENRICHMENT_FILE.exists():
        print("Missing base dataset or enrichment file.")
        return

    df = pd.read_parquet(base_path)
    enrich = pd.read_parquet(ENRICHMENT_FILE)

    print(f"Base: {len(df)} rows, Enrichment: {len(enrich)} rows")

    # Merge on package_id
    merged = df.merge(enrich, on="package_id", how="left")
    print(f"Merged: {len(merged)} rows")
    print(f"With sponsor: {(merged['sponsor_name'].str.len() > 0).sum()}")
    print(f"With policy area: {(merged['policy_area'].str.len() > 0).sum()}")

    # Save merged
    merged.to_parquet(base_path, index=False)
    print(f"Saved merged dataset to {base_path}")

    # Push to HF
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

    ds = Dataset.from_pandas(merged)
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"Pushed {len(merged)} rows to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Enrich federal dataset with Congress.gov metadata")
    parser.add_argument("--congress", type=int, default=None, help="Only enrich a specific Congress")
    parser.add_argument("--api-delay", type=float, default=1.0)
    parser.add_argument("--merge-push", action="store_true", help="Merge enrichment and push to HF")
    parser.add_argument("--repo-id", default="chrissoria/federal-public-laws")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = _load_api_key()

    if args.merge_push:
        merge_and_push(args.repo_id)
    else:
        enrich_dataset(
            api_key=api_key,
            congress_filter=args.congress,
            api_delay=args.api_delay,
        )


if __name__ == "__main__":
    main()
