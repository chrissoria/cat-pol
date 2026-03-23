#!/usr/bin/env python3
"""Generic Legistar scraper: fetch ordinances & resolutions for any Legistar city → HuggingFace.

Supports: Oakland, Long Beach, Fresno, Salinas (and any other Legistar city).

Usage:
    python scripts/build_legistar_dataset.py oakland
    python scripts/build_legistar_dataset.py longbeach
    python scripts/build_legistar_dataset.py fresno
    python scripts/build_legistar_dataset.py oakland --no-text
    python scripts/build_legistar_dataset.py oakland --push-only

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

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
API_BASE = "https://webapi.legistar.com/v1"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}

# City configurations: slug → (display_name, hf_repo, matter_types)
CITY_CONFIG = {
    "oakland": {
        "name": "Oakland",
        "repo": "chrissoria/oakland-ordinances",
        "types": ["Ordinance"],  # Oakland has no resolutions in Legistar
    },
    "longbeach": {
        "name": "Long Beach",
        "repo": "chrissoria/long-beach-ordinances",
        "types": ["Ordinance", "Resolution"],
    },
    "fresno": {
        "name": "Fresno",
        "repo": "chrissoria/fresno-ordinances",
        "types": ["Ordinance", "Resolution"],
    },
    "salinas": {
        "name": "Salinas",
        "repo": "chrissoria/salinas-ordinances",
        "types": ["Ordinance", "Resolution"],
    },
}

DATASET_COLUMNS = [
    "date", "matter_id", "matter_file", "enactment_number", "doc_type",
    "title", "text", "attachment_url", "year",
]


def _jittered_sleep(base: float) -> None:
    time.sleep(base * random.uniform(0.5, 1.5))


# ---------------------------------------------------------------------------
# Legistar API
# ---------------------------------------------------------------------------


def fetch_all_matters(city_slug: str, matter_type: str) -> list[dict]:
    """Fetch all matters of a given type from the Legistar API."""
    all_items = []
    skip = 0
    while True:
        for attempt in range(5):
            try:
                resp = requests.get(
                    f"{API_BASE}/{city_slug}/matters",
                    params={
                        "$filter": f"MatterTypeName eq '{matter_type}'",
                        "$top": 1000,
                        "$skip": skip,
                        "$orderby": "MatterIntroDate desc",
                    },
                    headers=HEADERS,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                wait = (attempt + 1) * 10
                print(f"  [retry {attempt + 1}/5] API error: {e}, waiting {wait}s...")
                time.sleep(wait)
                data = None
        if data is None:
            print("  [error] Giving up after 5 retries")
            break
        if not data:
            break
        all_items.extend(data)
        skip += 1000
        if len(data) < 1000:
            break
        _jittered_sleep(2.0)
    return all_items


def fetch_attachments(city_slug: str, matter_id: int) -> list[dict]:
    """Fetch attachments for a specific matter (with retry)."""
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{API_BASE}/{city_slug}/matters/{matter_id}/attachments",
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < 2:
                time.sleep((attempt + 1) * 5)
    return []


def find_best_attachment(attachments: list[dict]) -> str:
    """Find the best PDF attachment URL."""
    # Prefer PDFs with ordinance/resolution in name
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink") or ""
        name = (att.get("MatterAttachmentName") or "").lower()
        if url.endswith(".pdf") and ("ordinance" in name or "resolution" in name):
            return url
    # Fall back to first PDF
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink") or ""
        if url.endswith(".pdf"):
            return url
    # Fall back to any attachment with a URL
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink") or ""
        if url:
            return url
    return ""


# ---------------------------------------------------------------------------
# Document text extraction
# ---------------------------------------------------------------------------


def extract_text(url: str, session: requests.Session) -> str:
    """Download a PDF or DOCX and extract text."""
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

    if url_lower.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(resp.content))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            return ""

    return ""


# Keep old name as alias for backwards compat
extract_pdf_text = extract_text


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_dataset(
    city_slug: str,
    extract_text: bool = True,
    pdf_delay: float = 1.0,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Fetch all ordinances/resolutions for a Legistar city with incremental checkpointing."""
    config = CITY_CONFIG[city_slug]
    session = requests.Session()
    all_rows: list[dict] = []

    # Load already-processed matter IDs from checkpoints
    processed_ids: set[str] = set()
    for cp_file in CHECKPOINT_DIR.glob(f"{city_slug}_*.parquet"):
        if "final" not in cp_file.name:
            cp_df = pd.read_parquet(cp_file)
            all_rows.extend(cp_df.to_dict("records"))
            processed_ids.update(cp_df["matter_id"].astype(str).tolist())
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} matters already processed")

    for matter_type in config["types"]:
        doc_type = matter_type.lower()
        print(f"Fetching {matter_type}s for {config['name']}...")
        matters = fetch_all_matters(city_slug, matter_type)
        print(f"  Found {len(matters)} {doc_type}s")

        batch_rows: list[dict] = []
        batch_count = 0

        for i, matter in enumerate(matters):
            matter_id = str(matter["MatterId"])

            # Skip already-processed
            if matter_id in processed_ids:
                continue

            intro_date = matter.get("MatterIntroDate", "") or ""
            date = intro_date[:10] if intro_date else ""

            try:
                year = int(date[:4]) if date else 0
            except ValueError:
                year = 0

            # Fetch attachment with rate limiting
            attachments = fetch_attachments(city_slug, int(matter_id))
            attachment_url = find_best_attachment(attachments)
            _jittered_sleep(1.0)

            row = {
                "date": date,
                "matter_id": matter_id,
                "matter_file": matter.get("MatterFile", ""),
                "enactment_number": matter.get("MatterEnactmentNumber", "") or "",
                "doc_type": doc_type,
                "title": matter.get("MatterTitle", ""),
                "text": "",
                "attachment_url": attachment_url,
                "year": year,
            }

            if extract_text and attachment_url:
                row["text"] = extract_pdf_text(attachment_url, session)
                _jittered_sleep(pdf_delay)

            batch_rows.append(row)
            processed_ids.add(matter_id)
            batch_count += 1

            if batch_count % 25 == 0:
                print(f"  Processed {i + 1}/{len(matters)} ({batch_count} new)")

            # Incremental checkpoint
            if batch_count % checkpoint_every == 0 and batch_rows:
                cp_path = CHECKPOINT_DIR / f"{city_slug}_{doc_type}s_{batch_count:05d}.parquet"
                pd.DataFrame(batch_rows).to_parquet(cp_path, index=False)
                print(f"  [checkpoint] Saved {len(batch_rows)} rows to {cp_path.name}")
                all_rows.extend(batch_rows)
                batch_rows = []

        # Save remaining batch
        if batch_rows:
            cp_path = CHECKPOINT_DIR / f"{city_slug}_{doc_type}s_final.parquet"
            pd.DataFrame(batch_rows).to_parquet(cp_path, index=False)
            print(f"  [checkpoint] Saved {len(batch_rows)} {doc_type}s (final)")
            all_rows.extend(batch_rows)

    df = pd.DataFrame(all_rows, columns=DATASET_COLUMNS)
    df = df.drop_duplicates(subset=["matter_id"], keep="first").reset_index(drop=True)
    return df


def merge_checkpoints(city_slug: str) -> pd.DataFrame:
    """Merge all checkpoint files for a city."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp_file in sorted(CHECKPOINT_DIR.glob(f"{city_slug}_*.parquet")):
        if "final" not in cp_file.name:
            dfs.append(pd.read_parquet(cp_file))
    if not dfs:
        print(f"No checkpoint files found for {city_slug}.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["matter_id"], keep="first").reset_index(drop=True)
    print(f"Merged {len(df)} unique rows from {len(dfs)} checkpoint files")
    return df


def push_to_huggingface(df: pd.DataFrame, repo_id: str) -> None:
    """Push DataFrame to HuggingFace."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("datasets and huggingface_hub required")

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    token = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("No HuggingFace token found in .env")

    api = HfApi(token=token)
    api.whoami()

    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Build Legistar city dataset")
    parser.add_argument("city", choices=list(CITY_CONFIG.keys()), help="City to scrape")
    parser.add_argument("--no-text", action="store_true", help="Skip PDF text extraction")
    parser.add_argument("--pdf-delay", type=float, default=1.0)
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    config = CITY_CONFIG[args.city]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.push_only:
        df = merge_checkpoints(args.city)
        if df.empty:
            print("Nothing to push.")
            return
    else:
        df = build_dataset(
            city_slug=args.city,
            extract_text=not args.no_text,
            pdf_delay=args.pdf_delay,
        )

    print(f"\n=== {config['name']} ===")
    print(f"Total rows: {len(df)}")
    for dt in df["doc_type"].unique():
        print(f"  {dt.title()}s: {(df['doc_type'] == dt).sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rows with text: {(df['text'].str.len() > 0).sum()}")
    print(f"Rows with attachment: {(df['attachment_url'].str.len() > 0).sum()}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    final_path = CHECKPOINT_DIR / f"{args.city}_final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Final dataset saved to {final_path}")

    if args.push_only:
        push_to_huggingface(df, config["repo"])


if __name__ == "__main__":
    main()
