#!/usr/bin/env python3
"""One-time scraper: Salinas ordinances & resolutions via Legistar API → HuggingFace.

Usage:
    python scripts/build_salinas_dataset.py
    python scripts/build_salinas_dataset.py --no-text
    python scripts/build_salinas_dataset.py --push-only

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
API_BASE = "https://webapi.legistar.com/v1/salinas"

HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
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


def fetch_all_matters(matter_type: str) -> list[dict]:
    """Fetch all matters of a given type from the Legistar API."""
    all_items = []
    skip = 0
    while True:
        resp = requests.get(
            f"{API_BASE}/matters",
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
        if not data:
            break
        all_items.extend(data)
        skip += 1000
        if len(data) < 1000:
            break
        _jittered_sleep(1.0)
    return all_items


def fetch_attachments(matter_id: int) -> list[dict]:
    """Fetch attachments for a specific matter."""
    try:
        resp = requests.get(
            f"{API_BASE}/matters/{matter_id}/attachments",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def find_best_attachment(attachments: list[dict]) -> str:
    """Find the best PDF attachment URL from a list of attachments."""
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        name = (att.get("MatterAttachmentName") or "").lower()
        # Prefer PDF files, especially those with "ordinance" or "resolution" in name
        if url.endswith(".pdf"):
            if "ordinance" in name or "resolution" in name:
                return url
    # Fall back to first PDF
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url.endswith(".pdf"):
            return url
    # Fall back to first attachment with a URL
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url:
            return url
    return ""


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def extract_pdf_text(url: str, session: requests.Session) -> str:
    """Download a PDF, DOCX, or DOC and extract text."""
    if not url:
        return ""

    url_lower = url.lower()
    if not (url_lower.endswith(".pdf") or url_lower.endswith(".docx") or url_lower.endswith(".doc")):
        return ""

    try:
        resp = session.get(url, headers=HEADERS, timeout=60, verify=False)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] Download failed for {url}: {e}")
        return ""

    if url_lower.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                return "\n\n".join(pages)
        except Exception as e:
            print(f"  [warn] PDF extraction failed for {url}: {e}")
            return ""

    if url_lower.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(resp.content))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"  [warn] DOCX extraction failed for {url}: {e}")
            return ""

    if url_lower.endswith(".doc"):
        # Old .doc format — try textract, fall back to antiword
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            try:
                result = subprocess.run(
                    ["textutil", "-convert", "txt", "-stdout", tmp_path],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            # Fallback: try antiword
            try:
                result = subprocess.run(
                    ["antiword", tmp_path],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            os.unlink(tmp_path)
            print(f"  [warn] .doc extraction failed for {url}: no converter available")
            return ""
        except Exception as e:
            print(f"  [warn] .doc extraction failed for {url}: {e}")
            return ""

    return ""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_existing_matter_ids(doc_type: str) -> set:
    """Load matter_ids already scraped for this doc_type from checkpoints."""
    existing = set()
    for cp_file in sorted(CHECKPOINT_DIR.glob(f"salinas_{doc_type}s*.parquet")):
        try:
            df = pd.read_parquet(cp_file, columns=["matter_id"])
            existing.update(df["matter_id"].astype(str).tolist())
        except Exception:
            pass
    return existing


def build_dataset(
    extract_text: bool = True,
    pdf_delay: float = 1.0,
    types: list = None,
) -> pd.DataFrame:
    """Fetch all Salinas ordinances and resolutions from Legistar."""
    session = requests.Session()

    all_rows: list[dict] = []

    if types is None:
        types = [("Ordinance", "ordinance"), ("Resolution", "resolution")]

    for matter_type, doc_type in types:
        # Load already-scraped matter IDs to skip
        existing_ids = _load_existing_matter_ids(doc_type)
        if existing_ids:
            print(f"  Found {len(existing_ids)} existing {doc_type}s in checkpoints, will skip")

        print(f"Fetching {matter_type}s from Legistar API...")
        matters = fetch_all_matters(matter_type)
        print(f"  Found {len(matters)} {matter_type.lower()}s")

        type_rows: list[dict] = []
        skipped = 0

        for i, matter in enumerate(matters):
            matter_id = str(matter["MatterId"])

            # Skip already-scraped
            if matter_id in existing_ids:
                skipped += 1
                continue

            intro_date = matter.get("MatterIntroDate", "") or ""
            date = intro_date[:10] if intro_date else ""

            try:
                year = int(date[:4]) if date else 0
            except ValueError:
                year = 0

            # Fetch attachment URL
            attachments = fetch_attachments(matter_id)
            attachment_url = find_best_attachment(attachments)

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

            # Extract PDF text
            if extract_text and attachment_url:
                row["text"] = extract_pdf_text(attachment_url, session)
                _jittered_sleep(pdf_delay)

            type_rows.append(row)
            all_rows.append(row)

            processed = len(type_rows)
            if processed % 25 == 0 or processed == 1:
                print(f"  Processed {i + 1}/{len(matters)} ({processed} new, {skipped} skipped)")

            # Incremental checkpoint every 50 new items
            if processed % 50 == 0:
                cp_path = CHECKPOINT_DIR / f"salinas_{doc_type}s_{processed:05d}.parquet"
                pd.DataFrame(type_rows[-50:]).to_parquet(cp_path, index=False)
                print(f"  [checkpoint] Saved batch to {cp_path.name}")

            # Rate limit on attachment API calls
            if not extract_text:
                _jittered_sleep(0.3)

        # Final checkpoint for remaining items
        remainder = len(type_rows) % 50
        if remainder > 0:
            cp_path = CHECKPOINT_DIR / f"salinas_{doc_type}s_{len(type_rows):05d}.parquet"
            pd.DataFrame(type_rows[-remainder:]).to_parquet(cp_path, index=False)
            print(f"  [checkpoint] Saved {remainder} {doc_type}s (final)")

        print(f"  Done: {len(type_rows)} new, {skipped} skipped")

    df = pd.DataFrame(all_rows, columns=DATASET_COLUMNS)
    df = df.drop_duplicates(subset=["matter_id"], keep="first").reset_index(drop=True)
    return df


def merge_all_checkpoints() -> pd.DataFrame:
    """Merge all Salinas checkpoint files."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for cp_file in sorted(CHECKPOINT_DIR.glob("salinas_*.parquet")):
        dfs.append(pd.read_parquet(cp_file))
    if not dfs:
        print("No Salinas checkpoint files found.")
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
    parser = argparse.ArgumentParser(description="Build Salinas ordinances dataset")
    parser.add_argument("--no-text", action="store_true", help="Skip PDF text extraction")
    parser.add_argument("--pdf-delay", type=float, default=1.0)
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--merge-push", action="store_true", help="Merge checkpoints and push to HF")
    parser.add_argument("--type", choices=["ordinance", "resolution", "both"], default="both",
                        help="Which doc type(s) to scrape")
    parser.add_argument("--repo-id", default="chrissoria/salinas-ordinances")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.push_only or args.merge_push:
        df = merge_all_checkpoints()
        if df.empty:
            print("Nothing to push.")
            return
    else:
        type_map = {
            "ordinance": [("Ordinance", "ordinance")],
            "resolution": [("Resolution", "resolution")],
            "both": None,
        }
        df = build_dataset(
            extract_text=not args.no_text,
            pdf_delay=args.pdf_delay,
            types=type_map[args.type],
        )

    print(f"\nTotal rows: {len(df)}")
    print(f"Ordinances: {(df['doc_type'] == 'ordinance').sum()}")
    print(f"Resolutions: {(df['doc_type'] == 'resolution').sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rows with text: {(df['text'].str.len() > 0).sum()}")
    print(f"Rows with attachment: {(df['attachment_url'].str.len() > 0).sum()}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    final_path = CHECKPOINT_DIR / "salinas_final_dataset.parquet"
    df.to_parquet(final_path, index=False)
    print(f"Final dataset saved to {final_path}")

    if args.push_only or args.merge_push:
        push_to_huggingface(df, args.repo_id)


if __name__ == "__main__":
    main()
