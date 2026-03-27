#!/usr/bin/env python3
"""Weekly updater for California state bills dataset.

Downloads the Friday incremental dump from leginfo.legislature.ca.gov,
merges with existing data, and pushes to HuggingFace.

Designed to run via launchd on Sundays at 9 AM.

Usage:
    python scripts/update_california_bills.py
    python scripts/update_california_bills.py --dry-run
    python scripts/update_california_bills.py --full    # Re-download full dump
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).parent
LOG_DIR = SCRIPTS_DIR / "logs"
CACHE_DIR = SCRIPTS_DIR / "checkpoints" / "ca_bills"

load_dotenv(PROJECT_ROOT / ".env", override=True)

HF_TOKEN = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
HF_REPO = "chrissoria/california-bills"

FULL_DUMP_URL = "https://downloads.leginfo.legislature.ca.gov/pubinfo_2025.zip"
FRIDAY_DUMP_URL = "https://downloads.leginfo.legislature.ca.gov/pubinfo_Fri.zip"

CACHE_FILE = CACHE_DIR / "california_bills.parquet"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ca_bills_updater")
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


def clean(s):
    if isinstance(s, str):
        return s.strip().strip("`")
    return s


def download_and_extract(url: str, log: logging.Logger) -> str:
    """Download zip and extract to temp directory. Returns path."""
    log.info(f"Downloading {url}...")
    tmp_dir = tempfile.mkdtemp(prefix="ca_bills_")
    zip_path = os.path.join(tmp_dir, "dump.zip")

    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    log.info(f"Downloaded {os.path.getsize(zip_path) / 1e6:.0f} MB")

    # Extract key tables + LOB files
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name in ("BILL_TBL.dat", "BILL_VERSION_TBL.dat",
                        "BILL_VERSION_AUTHORS_TBL.dat", "BILL_HISTORY_TBL.dat") \
               or name.startswith("BILL_VERSION_TBL_") and name.endswith(".lob"):
                z.extract(name, tmp_dir)

    return tmp_dir


def parse_dump(data_dir: str, log: logging.Logger) -> pd.DataFrame:
    """Parse extracted database tables into a clean DataFrame."""
    # Parse BILL_TBL
    bills = pd.read_csv(f"{data_dir}/BILL_TBL.dat", sep="\t", header=None, dtype=str, on_bad_lines="skip")
    bills = bills.map(clean)
    bills.columns = ["bill_id", "session", "extra_sess", "measure_type", "measure_num",
                     "version_status", "c6", "c7", "c8", "c9", "current_version",
                     "active", "source", "last_update", "current_committee", "current_location",
                     "current_house", "current_status", "date_introduced"]
    bills = bills[bills["active"] == "Y"]
    log.info(f"Bills: {len(bills)}")

    # Parse BILL_VERSION_TBL
    versions = pd.read_csv(f"{data_dir}/BILL_VERSION_TBL.dat", sep="\t", header=None, dtype=str, on_bad_lines="skip")
    versions = versions.map(clean)
    versions.columns = ["version_id", "bill_id", "version_num", "date", "status", "c5",
                        "subject", "vote_req", "urgency", "appropriation", "fiscal_committee",
                        "c11", "local_program", "c13", "lob_file", "active", "source", "last_update"]
    versions = versions[versions["active"] == "Y"]

    # Get latest version per bill
    versions["version_num"] = pd.to_numeric(versions["version_num"], errors="coerce")
    latest = versions.sort_values("version_num", ascending=False).groupby("bill_id").first().reset_index()

    # Parse AUTHORS
    authors = pd.read_csv(f"{data_dir}/BILL_VERSION_AUTHORS_TBL.dat", sep="\t", header=None, dtype=str, on_bad_lines="skip")
    authors = authors.map(clean)
    authors.columns = ["version_id", "type", "house", "name", "role", "c5", "active", "source", "last_update", "primary"]
    authors = authors[authors["active"] == "Y"]
    leads = authors[authors["role"] == "LEAD_AUTHOR"].groupby("version_id")["name"].apply(", ".join).reset_index()
    leads.columns = ["version_id", "lead_authors"]

    # Join
    merged = bills.merge(latest[["bill_id", "version_id", "subject", "lob_file", "vote_req", "urgency", "fiscal_committee"]], on="bill_id", how="left")
    merged = merged.merge(leads, on="version_id", how="left")

    # Read bill text
    def read_lob(lob_file):
        if pd.isna(lob_file) or not lob_file:
            return ""
        path = f"{data_dir}/{lob_file}"
        if os.path.exists(path):
            with open(path, "r", errors="replace") as f:
                return f.read()
        return ""

    log.info("Reading bill text from LOB files...")
    merged["text"] = merged["lob_file"].apply(read_lob)

    # Structure output
    result = pd.DataFrame({
        "date": pd.to_datetime(merged["date_introduced"]).dt.strftime("%Y-%m-%d"),
        "session": merged["session"].apply(lambda s: f"{s[:4]}-{s[4:]}" if len(str(s)) == 8 else s),
        "bill_number": merged["measure_type"] + "-" + merged["measure_num"],
        "bill_id": merged["bill_id"],
        "doc_type": merged["measure_type"].map({
            "AB": "assembly_bill", "SB": "senate_bill",
            "ACA": "assembly_constitutional_amendment", "SCA": "senate_constitutional_amendment",
            "ACR": "assembly_concurrent_resolution", "SCR": "senate_concurrent_resolution",
            "AJR": "assembly_joint_resolution", "SJR": "senate_joint_resolution",
            "HR": "house_resolution", "SR": "senate_resolution",
        }).fillna("other"),
        "title": merged["subject"],
        "text": merged["text"],
        "url": merged["bill_id"].apply(lambda x: f"https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id={x}"),
        "origin_chamber": merged["measure_type"].apply(lambda t: "Assembly" if str(t).startswith("A") else "Senate"),
        "sponsor_name": merged["lead_authors"],
        "status": merged["current_status"],
        "current_house": merged["current_house"],
        "current_committee": merged["current_committee"],
        "vote_requirement": merged["vote_req"],
        "urgency": merged["urgency"],
        "fiscal_committee": merged["fiscal_committee"],
    })

    log.info(f"Parsed {len(result)} bills ({(result['text'].str.len() > 0).sum()} with text)")
    return result


def push_to_hf(df: pd.DataFrame, log: logging.Logger):
    """Push to HuggingFace."""
    try:
        from datasets import Dataset
    except ImportError:
        log.error("datasets package not installed")
        return

    if not HF_TOKEN:
        log.error("No HuggingFace token")
        return

    ds = Dataset.from_pandas(df)
    ds.push_to_hub(HF_REPO, token=HF_TOKEN, private=False)
    log.info(f"Pushed {len(df)} bills to https://huggingface.co/datasets/{HF_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Update California bills dataset")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't push")
    parser.add_argument("--full", action="store_true", help="Re-download full dump instead of Friday incremental")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("California bills updater starting")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    url = FULL_DUMP_URL if args.full else FRIDAY_DUMP_URL

    try:
        data_dir = download_and_extract(url, log)
        new_data = parse_dump(data_dir, log)

        # Merge with existing cache
        if CACHE_FILE.exists() and not args.full:
            existing = pd.read_parquet(CACHE_FILE)
            log.info(f"Existing cache: {len(existing)} bills")
            # Update existing bills and add new ones
            combined = pd.concat([existing, new_data]).drop_duplicates(subset=["bill_id"], keep="last")
            combined = combined.sort_values("date", ascending=False).reset_index(drop=True)
            log.info(f"After merge: {len(combined)} bills (+{len(combined) - len(existing)} new/updated)")
        else:
            combined = new_data

        combined.to_parquet(CACHE_FILE, index=False)
        log.info(f"Cache saved: {CACHE_FILE}")

        if not args.dry_run:
            push_to_hf(combined, log)

        # Cleanup
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)

    except Exception as e:
        log.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
