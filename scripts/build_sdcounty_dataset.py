#!/usr/bin/env python3
"""One-time scraper: San Diego County ordinances via Legistar API → HuggingFace.

SD County's Legistar uses subject-area MatterTypes (not Ordinance/Resolution),
so we filter by title keyword instead.

Usage:
    python scripts/build_sdcounty_dataset.py
    python scripts/build_sdcounty_dataset.py --merge-push

Dependencies:
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
import io
import os
import random
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "sdcounty"
API_BASE = "https://webapi.legistar.com/v1/sdcounty"
GRANICUS_BASE = "https://sdcounty.legistar1.com"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

DATASET_COLUMNS = [
    "date", "matter_id", "matter_file", "doc_type", "title", "text",
    "attachment_url", "body", "status", "year",
]


def _jittered_sleep(base: float):
    time.sleep(base * (0.7 + 0.6 * random.random()))


def fetch_all_matters(keyword: str) -> list[dict]:
    """Fetch all matters containing keyword in title."""
    all_items = []
    skip = 0
    while True:
        resp = requests.get(
            f"{API_BASE}/matters",
            params={
                "$filter": f"substringof('{keyword}',MatterTitle)",
                "$top": 1000,
                "$skip": skip,
                "$orderby": "MatterIntroDate desc",
            },
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json()
        if not items:
            break
        all_items.extend(items)
        skip += 1000
        if len(items) < 1000:
            break
        _jittered_sleep(0.5)
    return all_items


def fetch_attachments(matter_id: int) -> list[dict]:
    """Fetch attachments for a matter."""
    try:
        resp = requests.get(f"{API_BASE}/matters/{matter_id}/attachments", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def find_best_attachment(attachments: list[dict]) -> str:
    """Find the best document attachment URL."""
    # Prefer DOCX/PDF with ordinance-related names
    for att in attachments:
        name = (att.get("MatterAttachmentName") or "").lower()
        url = att.get("MatterAttachmentHyperlink", "")
        if ("ordinance" in name or "clean" in name) and (url.endswith(".docx") or url.endswith(".pdf")):
            return url
    # Fall back to any PDF/DOCX
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url.endswith(".pdf") or url.endswith(".docx"):
            return url
    # Fall back to any attachment
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url:
            return url
    return ""


def extract_text(url: str, session: requests.Session) -> str:
    """Download and extract text from PDF or DOCX."""
    if not url:
        return ""
    url_lower = url.lower()
    try:
        resp = session.get(url, headers=HEADERS, timeout=60, verify=False)
        resp.raise_for_status()
    except Exception:
        return ""

    if url_lower.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                return "\n\n".join(pages)
        except Exception:
            return ""
    elif url_lower.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(resp.content))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            return ""
    elif url_lower.endswith(".doc"):
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            os.unlink(tmp_path)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    return ""


def _load_existing_ids() -> set:
    existing = set()
    for cp in sorted(CHECKPOINT_DIR.glob("sdcounty_*.parquet")):
        try:
            df = pd.read_parquet(cp, columns=["matter_id"])
            existing.update(df["matter_id"].astype(str).tolist())
        except Exception:
            pass
    return existing


def build_dataset(extract_text_flag: bool = True, delay: float = 1.0) -> pd.DataFrame:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    existing_ids = _load_existing_ids()
    if existing_ids:
        print(f"  Found {len(existing_ids)} existing in checkpoints, will skip")

    # Fetch ordinances and resolutions by title keyword
    all_rows = []
    for keyword, doc_type in [("ORDINANCE", "ordinance"), ("RESOLUTION", "resolution")]:
        print(f"Fetching {doc_type}s from Legistar (keyword: {keyword})...")
        matters = fetch_all_matters(keyword)
        print(f"  Found {len(matters)} {doc_type}s")

        new_rows = []
        skipped = 0

        for i, matter in enumerate(matters):
            mid = str(matter["MatterId"])
            if mid in existing_ids:
                skipped += 1
                continue

            intro_date = matter.get("MatterIntroDate", "") or ""
            date = intro_date[:10] if intro_date else ""
            try:
                year = int(date[:4]) if date else 0
            except ValueError:
                year = 0

            attachments = fetch_attachments(matter["MatterId"])
            att_url = find_best_attachment(attachments)
            _jittered_sleep(0.3)

            text = ""
            if extract_text_flag and att_url:
                text = extract_text(att_url, session)
                _jittered_sleep(delay)

            row = {
                "date": date,
                "matter_id": mid,
                "matter_file": matter.get("MatterFile", ""),
                "doc_type": doc_type,
                "title": matter.get("MatterTitle", "").strip(),
                "text": text,
                "attachment_url": att_url,
                "body": matter.get("MatterBodyName", ""),
                "status": matter.get("MatterStatusName", ""),
                "year": year,
            }

            new_rows.append(row)
            all_rows.append(row)
            processed = len(new_rows)

            if processed % 25 == 0 or processed == 1:
                print(f"  Processed {i+1}/{len(matters)} ({processed} new, {skipped} skipped)")

            if processed % 50 == 0:
                cp = CHECKPOINT_DIR / f"sdcounty_{doc_type}s_{processed:05d}.parquet"
                pd.DataFrame(new_rows[-50:]).to_parquet(cp, index=False)
                print(f"  [checkpoint] {cp.name}")

        # Final checkpoint
        remainder = len(new_rows) % 50
        if remainder > 0:
            cp = CHECKPOINT_DIR / f"sdcounty_{doc_type}s_{len(new_rows):05d}.parquet"
            pd.DataFrame(new_rows[-remainder:]).to_parquet(cp, index=False)

        print(f"  Done: {len(new_rows)} new, {skipped} skipped")

    df = pd.DataFrame(all_rows, columns=DATASET_COLUMNS)
    df = df.drop_duplicates(subset=["matter_id"], keep="first").reset_index(drop=True)
    return df


def merge_all_checkpoints() -> pd.DataFrame:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp in sorted(CHECKPOINT_DIR.glob("sdcounty_*.parquet")):
        dfs.append(pd.read_parquet(cp))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["matter_id"], keep="first").reset_index(drop=True)
    print(f"Merged {len(df)} unique rows from {len(dfs)} checkpoint files")
    return df


def push_to_hf(df: pd.DataFrame, repo_id: str):
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
    parser = argparse.ArgumentParser(description="Build SD County ordinances dataset")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--merge-push", action="store_true")
    parser.add_argument("--repo-id", default="chrissoria/sd-county-ordinances")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge_push:
        df = merge_all_checkpoints()
        if df.empty:
            print("Nothing to push.")
            return
    else:
        df = build_dataset(extract_text_flag=not args.no_text, delay=args.delay)
        existing = merge_all_checkpoints()
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["matter_id"], keep="first").reset_index(drop=True)

    print(f"\n=== SD County ===")
    print(f"Total: {len(df)}")
    print(f"Ordinances: {(df['doc_type'] == 'ordinance').sum()}")
    print(f"Resolutions: {(df['doc_type'] == 'resolution').sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"With text: {(df['text'].str.len() > 0).sum()}")

    final = CHECKPOINT_DIR / "sdcounty_final_dataset.parquet"
    df.to_parquet(final, index=False)
    print(f"Saved to {final}")

    push_to_hf(df, args.repo_id)


if __name__ == "__main__":
    main()
