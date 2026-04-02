#!/usr/bin/env python3
"""Weekly incremental updater for all datasets.

Updates all city + federal datasets and pushes to HuggingFace.
Designed to run via launchd (Sundays at 9 AM).

Supported sources:
    sd   - San Diego (sandiego.gov)
    sf   - San Francisco (sfbos.org)
    sal  - Salinas (Legistar API)
    oak  - Oakland (Legistar API)
    lb   - Long Beach (Legistar API)
    fre  - Fresno (Legistar API)
    ber  - Berkeley (municipal.codes, Playwright)
    fed  - Federal public laws (GovInfo API)

Usage:
    python scripts/update_datasets.py                  # Update all
    python scripts/update_datasets.py --city sd         # San Diego only
    python scripts/update_datasets.py --city fed        # Federal only
    python scripts/update_datasets.py --exclude ts      # All except Truth Social
    python scripts/update_datasets.py --dry-run         # Scrape but don't push

Setup:
    1. pip install -r scripts/requirements.txt
    2. Set CATLLM_HUGGINGFACE_TOKEN in cat-pol/.env
    3. Set GOVINFO_API_KEY in cat-pol/.env (for federal)
    4. pip install playwright && python -m playwright install chromium (for Berkeley)
    5. bash scripts/install_launchd.sh
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).parent
LOG_DIR = SCRIPTS_DIR / "logs"
CHECKPOINT_DIR = SCRIPTS_DIR / "checkpoints"

load_dotenv(PROJECT_ROOT / ".env", override=True)

sys.path.insert(0, str(SCRIPTS_DIR))

HF_TOKEN = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

PAGE_DELAY = 2.0
PDF_DELAY = 1.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("dataset_updater")
    if log.handlers:
        return log  # already set up
    log.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_DIR / "update.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(ch)

    return log


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_hf_dataset(repo_id: str, cache_path: Path, log: logging.Logger) -> pd.DataFrame:
    """Load dataset from HuggingFace with local caching."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets package not installed: pip install datasets")
        sys.exit(1)

    if cache_path.exists():
        log.info(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    log.info(f"Downloading from HuggingFace: {repo_id}")
    try:
        ds = load_dataset(repo_id, split="train")
        df = ds.to_pandas()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log.info(f"Cached {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"Failed to load {repo_id}: {e}")
        return pd.DataFrame()


def push_hf_dataset(df: pd.DataFrame, repo_id: str, cache_path: Path, log: logging.Logger) -> None:
    """Push dataset to HuggingFace and update local cache."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        log.error("datasets/huggingface_hub not installed")
        return

    if not HF_TOKEN:
        log.error("No HuggingFace token. Set CATLLM_HUGGINGFACE_TOKEN in .env")
        return

    api = HfApi(token=HF_TOKEN)
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, token=HF_TOKEN, private=False)
    log.info(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")

    df.to_parquet(cache_path, index=False)


def extract_pdf_text(url: str, session: requests.Session) -> str:
    """Download a PDF, DOCX, or DOC and extract text."""
    if not url:
        return ""
    url_lower = url.lower()
    if not (url_lower.endswith(".pdf") or url_lower.endswith(".docx") or url_lower.endswith(".doc")):
        return ""
    try:
        resp = session.get(url, timeout=60, verify=False)
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
    if url_lower.endswith(".doc"):
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
                    os.unlink(tmp_path)
                    return result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            os.unlink(tmp_path)
        except Exception:
            pass
        return ""
    return ""


def _jittered_sleep(base: float) -> None:
    import random, time
    time.sleep(base * random.uniform(0.5, 1.5))


# ===========================================================================
# San Diego
# ===========================================================================

SD_REPO = os.getenv("SD_DATASET_REPO", "chrissoria/san-diego-ordinances")
SD_CACHE = SCRIPTS_DIR / ".sd_cache.parquet"
SD_MAX_PAGES = 20


def update_san_diego(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for San Diego ordinances/resolutions."""
    from build_san_diego_dataset import scrape_page as sd_scrape_page

    log.info("=" * 50)
    log.info(f"[San Diego] Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(SD_REPO, SD_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"[San Diego] Loaded {len(existing_df)} rows from build checkpoint")
        else:
            log.error("[San Diego] No existing dataset. Run build_san_diego_dataset.py first.")
            return 1

    existing_ids = set(existing_df["doc_num"].tolist())
    log.info(f"[San Diego] Existing: {len(existing_df)} rows, {len(existing_ids)} unique doc_nums")

    session = requests.Session()
    new_rows: list[dict] = []

    for page_num in range(SD_MAX_PAGES):
        log.info(f"[San Diego] Scraping page {page_num}...")
        try:
            rows = sd_scrape_page(page_num, session)
        except Exception as e:
            log.error(f"[San Diego] Page {page_num} failed: {e}")
            _jittered_sleep(PAGE_DELAY * 3)
            continue

        if not rows:
            break

        page_new = [r for r in rows if r["doc_num"] not in existing_ids]
        page_overlap = len(rows) - len(page_new)
        log.info(f"[San Diego]   {len(page_new)} new, {page_overlap} known")

        for i, row in enumerate(page_new):
            if row["pdf_url"]:
                row["text"] = extract_pdf_text(row["pdf_url"], session)
                _jittered_sleep(PDF_DELAY)

        new_rows.extend(page_new)

        if page_overlap == len(rows):
            log.info("[San Diego] Caught up — full page overlap")
            break
        if page_overlap > len(rows) * 0.5 and page_num > 0:
            log.info("[San Diego] Caught up — majority overlap")
            break

        _jittered_sleep(PAGE_DELAY)

    if not new_rows:
        log.info("[San Diego] No new entries. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"[San Diego] Found {len(new_df)} new entries ({new_df['date'].min()} to {new_df['date'].max()})")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['doc_num']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["doc_num"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, SD_REPO, SD_CACHE, log)
    log.info(f"[San Diego] Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# San Francisco
# ===========================================================================

SF_REPO = os.getenv("SF_DATASET_REPO", "chrissoria/sf-ordinances")
SF_CACHE = SCRIPTS_DIR / ".sf_cache.parquet"
SF_BASE_URL = "https://www.sfbos.org"
SF_HEADERS = {
    "User-Agent": "cat-pol-dataset-builder/1.0 (academic research; chrissoria@berkeley.edu)",
}


def _sf_scrape_year(year: int, session: requests.Session) -> list[dict]:
    """Scrape SF ordinances for a single year."""
    from bs4 import BeautifulSoup
    from datetime import datetime as dt

    resp = session.get(f"{SF_BASE_URL}/ordinances-{year}", headers=SF_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header row
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        file_number = tds[0].get_text(strip=True)
        a_tag = tds[1].find("a")
        if a_tag:
            enactment_number = a_tag.get_text(strip=True)
            pdf_path = a_tag.get("href", "")
            pdf_url = f"{SF_BASE_URL}{pdf_path}" if pdf_path.startswith("/") else pdf_path
        else:
            enactment_number = tds[1].get_text(strip=True)
            pdf_url = ""

        date_raw = tds[2].get_text(strip=True)
        title = tds[3].get_text(strip=True)

        # Parse date
        date = ""
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                date = dt.strptime(date_raw, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

        rows.append({
            "date": date,
            "file_number": file_number,
            "enactment_number": enactment_number,
            "doc_type": "ordinance",
            "title": title,
            "text": "",
            "pdf_url": pdf_url,
            "year": year,
        })

    return rows


def update_san_francisco(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for San Francisco ordinances.

    SF is organized by year, so we only need to scrape the current year
    (and possibly the previous year near January).
    """
    log.info("=" * 50)
    log.info(f"[San Francisco] Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(SF_REPO, SF_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "sf_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"[San Francisco] Loaded {len(existing_df)} rows from build checkpoint")
        else:
            log.error("[San Francisco] No existing dataset. Run build_sf_dataset.py first.")
            return 1

    existing_ids = set(existing_df["enactment_number"].tolist())
    log.info(f"[San Francisco] Existing: {len(existing_df)} rows, {len(existing_ids)} unique")

    session = requests.Session()
    now = datetime.now()
    years_to_check = [now.year]
    # Also check previous year in January (new ordinances may still be posted)
    if now.month == 1:
        years_to_check.append(now.year - 1)

    new_rows: list[dict] = []

    for year in years_to_check:
        log.info(f"[San Francisco] Scraping {year}...")
        try:
            rows = _sf_scrape_year(year, session)
        except Exception as e:
            log.error(f"[San Francisco] Year {year} failed: {e}")
            continue

        page_new = [r for r in rows if r["enactment_number"] not in existing_ids]
        log.info(f"[San Francisco]   {year}: {len(page_new)} new out of {len(rows)} total")

        for i, row in enumerate(page_new):
            if row["pdf_url"]:
                log.debug(f"[San Francisco]   PDF {i + 1}/{len(page_new)}: {row['enactment_number']}")
                row["text"] = extract_pdf_text(row["pdf_url"], session)
                _jittered_sleep(PDF_DELAY)

        new_rows.extend(page_new)
        _jittered_sleep(PAGE_DELAY)

    if not new_rows:
        log.info("[San Francisco] No new entries. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"[San Francisco] Found {len(new_df)} new ordinances ({new_df['date'].min()} to {new_df['date'].max()})")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['enactment_number']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["enactment_number"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, SF_REPO, SF_CACHE, log)
    log.info(f"[San Francisco] Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Salinas
# ===========================================================================

SAL_REPO = os.getenv("SAL_DATASET_REPO", "chrissoria/salinas-ordinances")
SAL_CACHE = SCRIPTS_DIR / ".sal_cache.parquet"
SAL_API_BASE = "https://webapi.legistar.com/v1/salinas"


def _sal_fetch_recent_matters(matter_type: str) -> list[dict]:
    """Fetch recent matters from the Salinas Legistar API."""
    resp = requests.get(
        f"{SAL_API_BASE}/matters",
        params={
            "$filter": f"MatterTypeName eq '{matter_type}'",
            "$top": 200,
            "$orderby": "MatterIntroDate desc",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _sal_get_attachment_url(matter_id: int) -> str:
    """Get the best PDF attachment URL for a matter."""
    try:
        resp = requests.get(f"{SAL_API_BASE}/matters/{matter_id}/attachments", timeout=30)
        resp.raise_for_status()
        attachments = resp.json()
    except Exception:
        return ""

    # Prefer PDFs, especially those with ordinance/resolution in name
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        name = att.get("MatterAttachmentName", "").lower()
        if url.endswith(".pdf") and ("ordinance" in name or "resolution" in name):
            return url
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url.endswith(".pdf"):
            return url
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url:
            return url
    return ""


def update_salinas(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for Salinas ordinances/resolutions."""
    log.info("=" * 50)
    log.info(f"[Salinas] Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(SAL_REPO, SAL_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "salinas_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"[Salinas] Loaded {len(existing_df)} rows from build checkpoint")
        else:
            log.error("[Salinas] No existing dataset. Run build_salinas_dataset.py first.")
            return 1

    existing_ids = set(existing_df["matter_id"].astype(str).tolist())
    log.info(f"[Salinas] Existing: {len(existing_df)} rows, {len(existing_ids)} unique matter IDs")

    session = requests.Session()
    new_rows: list[dict] = []

    for matter_type, doc_type in [("Ordinance", "ordinance"), ("Resolution", "resolution")]:
        log.info(f"[Salinas] Fetching recent {matter_type}s...")
        try:
            matters = _sal_fetch_recent_matters(matter_type)
        except Exception as e:
            log.error(f"[Salinas] API fetch failed for {matter_type}: {e}")
            continue

        type_new = []
        for matter in matters:
            mid = str(matter["MatterId"])
            if mid in existing_ids:
                continue

            intro_date = matter.get("MatterIntroDate", "") or ""
            date = intro_date[:10] if intro_date else ""
            try:
                year = int(date[:4]) if date else 0
            except ValueError:
                year = 0

            attachment_url = _sal_get_attachment_url(matter["MatterId"])
            _jittered_sleep(0.3)

            row = {
                "date": date,
                "matter_id": mid,
                "matter_file": matter.get("MatterFile", ""),
                "enactment_number": matter.get("MatterEnactmentNumber", "") or "",
                "doc_type": doc_type,
                "title": matter.get("MatterTitle", ""),
                "text": "",
                "attachment_url": attachment_url,
                "year": year,
            }

            if attachment_url:
                row["text"] = extract_pdf_text(attachment_url, session)
                _jittered_sleep(PDF_DELAY)

            type_new.append(row)

        log.info(f"[Salinas]   {len(type_new)} new {matter_type.lower()}s")
        new_rows.extend(type_new)

    if not new_rows:
        log.info("[Salinas] No new entries. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"[Salinas] Found {len(new_df)} new entries ({new_df['date'].min()} to {new_df['date'].max()})")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['matter_file']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["matter_id"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, SAL_REPO, SAL_CACHE, log)
    log.info(f"[Salinas] Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Generic Legistar updater (Oakland, Long Beach, Fresno)
# ===========================================================================

LEGISTAR_CITIES = {
    "oak": {
        "name": "Oakland",
        "slug": "oakland",
        "repo": "chrissoria/oakland-ordinances",
        "cache": SCRIPTS_DIR / ".oak_cache.parquet",
        "types": ["Ordinance"],
    },
    "lb": {
        "name": "Long Beach",
        "slug": "longbeach",
        "repo": "chrissoria/long-beach-ordinances",
        "cache": SCRIPTS_DIR / ".lb_cache.parquet",
        "types": ["Ordinance", "Resolution"],
    },
    "fre": {
        "name": "Fresno",
        "slug": "fresno",
        "repo": "chrissoria/fresno-ordinances",
        "cache": SCRIPTS_DIR / ".fre_cache.parquet",
        "types": ["Ordinance", "Resolution"],
    },
}

LEGISTAR_API = "https://webapi.legistar.com/v1"


def _legistar_fetch_recent(city_slug: str, matter_type: str) -> list[dict]:
    """Fetch recent matters from a Legistar city API."""
    resp = requests.get(
        f"{LEGISTAR_API}/{city_slug}/matters",
        params={
            "$filter": f"MatterTypeName eq '{matter_type}'",
            "$top": 200,
            "$orderby": "MatterIntroDate desc",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _legistar_get_attachment(city_slug: str, matter_id: int) -> str:
    """Get the best PDF attachment URL for a Legistar matter."""
    try:
        resp = requests.get(
            f"{LEGISTAR_API}/{city_slug}/matters/{matter_id}/attachments",
            timeout=30,
        )
        resp.raise_for_status()
        attachments = resp.json()
    except Exception:
        return ""

    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        name = att.get("MatterAttachmentName", "").lower()
        if url.endswith(".pdf") and ("ordinance" in name or "resolution" in name):
            return url
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url.endswith(".pdf"):
            return url
    for att in attachments:
        url = att.get("MatterAttachmentHyperlink", "")
        if url:
            return url
    return ""


def update_legistar_city(city_key: str, log: logging.Logger, dry_run: bool = False) -> int:
    """Generic incremental update for any Legistar city."""
    cfg = LEGISTAR_CITIES[city_key]
    tag = f"[{cfg['name']}]"

    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(cfg["repo"], cfg["cache"], log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / f"{cfg['slug']}_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"{tag} Loaded {len(existing_df)} rows from build checkpoint")
        else:
            log.error(f"{tag} No existing dataset. Run build_legistar_dataset.py {cfg['slug']} first.")
            return 1

    existing_ids = set(existing_df["matter_id"].astype(str).tolist())
    log.info(f"{tag} Existing: {len(existing_df)} rows, {len(existing_ids)} unique")

    session = requests.Session()
    new_rows: list[dict] = []

    for matter_type in cfg["types"]:
        doc_type = matter_type.lower()
        log.info(f"{tag} Fetching recent {matter_type}s...")
        try:
            matters = _legistar_fetch_recent(cfg["slug"], matter_type)
        except Exception as e:
            log.error(f"{tag} API failed for {matter_type}: {e}")
            continue

        type_new = []
        for matter in matters:
            mid = str(matter["MatterId"])
            if mid in existing_ids:
                continue

            intro_date = matter.get("MatterIntroDate", "") or ""
            date = intro_date[:10] if intro_date else ""
            try:
                year = int(date[:4]) if date else 0
            except ValueError:
                year = 0

            attachment_url = _legistar_get_attachment(cfg["slug"], matter["MatterId"])
            _jittered_sleep(0.3)

            row = {
                "date": date,
                "matter_id": mid,
                "matter_file": matter.get("MatterFile", ""),
                "enactment_number": matter.get("MatterEnactmentNumber", "") or "",
                "doc_type": doc_type,
                "title": matter.get("MatterTitle", ""),
                "text": "",
                "attachment_url": attachment_url,
                "year": year,
            }

            if attachment_url:
                row["text"] = extract_pdf_text(attachment_url, session)
                _jittered_sleep(PDF_DELAY)

            type_new.append(row)

        log.info(f"{tag}   {len(type_new)} new {doc_type}s")
        new_rows.extend(type_new)

    if not new_rows:
        log.info(f"{tag} No new entries. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"{tag} Found {len(new_df)} new entries ({new_df['date'].min()} to {new_df['date'].max()})")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['matter_file']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["matter_id"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, cfg["repo"], cfg["cache"], log)
    log.info(f"{tag} Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Berkeley (Playwright — Cloudflare-protected)
# ===========================================================================

BER_REPO = os.getenv("BER_DATASET_REPO", "chrissoria/berkeley-ordinances")
BER_CACHE = SCRIPTS_DIR / ".ber_cache.parquet"
BER_BASE_URL = "https://berkeley.municipal.codes"


def update_berkeley(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for Berkeley ordinances (requires Playwright)."""
    tag = "[Berkeley]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error(f"{tag} Playwright not installed. Skipping Berkeley.")
        return 1

    existing_df = load_hf_dataset(BER_REPO, BER_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "berkeley_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"{tag} Loaded {len(existing_df)} rows from build checkpoint")
        else:
            log.error(f"{tag} No existing dataset. Run build_berkeley_dataset.py first.")
            return 1

    existing_ords = set(existing_df["ordinance_number"].astype(str).tolist())
    log.info(f"{tag} Existing: {len(existing_df)} rows")

    # Scrape first few pages (most recent ordinances) to find new entries
    new_rows = []
    max_pages = 5  # Only check recent pages for weekly updates

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            log.info(f"{tag} Loading page (Cloudflare check)...")
            page.goto(f"{BER_BASE_URL}/enactments", timeout=60000)
            page.wait_for_timeout(5000)

            found_existing = False
            for page_num in range(1, max_pages + 1):
                url = f"{BER_BASE_URL}/enactments?page={page_num}"
                page.goto(url, timeout=30000)
                page.wait_for_timeout(2000)

                tables = page.query_selector_all("table")
                if len(tables) < 2:
                    break

                rows = tables[1].query_selector_all("tr")
                page_new = 0

                for row in rows:
                    cells = row.query_selector_all("td")
                    if len(cells) < 4:
                        continue

                    a_tag = cells[0].query_selector("a")
                    if a_tag:
                        ord_num = a_tag.inner_text().strip()
                        detail_path = a_tag.get_attribute("href") or ""
                        detail_url = f"{BER_BASE_URL}{detail_path}" if detail_path.startswith("/") else detail_path
                    else:
                        ord_num = cells[0].inner_text().strip()
                        detail_url = ""

                    if ord_num in existing_ords:
                        found_existing = True
                        continue

                    title = cells[1].inner_text().strip()
                    date_raw = cells[2].inner_text().strip()
                    disposition = cells[3].inner_text().strip()

                    # Parse date
                    date = ""
                    try:
                        dt = datetime.strptime(date_raw, "%m/%d/%Y")
                        date = dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        if date_raw.strip().isdigit() and len(date_raw.strip()) == 4:
                            date = date_raw.strip()
                        else:
                            date = date_raw

                    try:
                        year = int(date[:4]) if date else 0
                    except ValueError:
                        year = 0

                    num_lower = ord_num.lower()
                    doc_type = "resolution" if "res" in num_lower else "ordinance"

                    entry = {
                        "date": date,
                        "ordinance_number": ord_num,
                        "doc_type": doc_type,
                        "title": title,
                        "text": "",
                        "disposition": disposition,
                        "effective_date": "",
                        "url": detail_url,
                        "year": year,
                    }

                    # Scrape detail page for text
                    if detail_url:
                        try:
                            page.goto(detail_url, timeout=30000)
                            page.wait_for_timeout(1500)
                            main = page.query_selector("#main-column, main, article")
                            entry["text"] = main.inner_text().strip() if main else page.inner_text("body").strip()
                        except Exception as e:
                            log.warning(f"{tag} Detail page failed for {ord_num}: {e}")
                        _jittered_sleep(2.0)

                    new_rows.append(entry)
                    page_new += 1

                log.info(f"{tag} Page {page_num}: {page_new} new entries")

                # Stop early if we've hit existing entries (all caught up)
                if found_existing and page_new == 0:
                    break

                _jittered_sleep(3.0)

            browser.close()

    except Exception as e:
        log.error(f"{tag} Playwright scraping failed: {e}")
        if not new_rows:
            return 1

    if not new_rows:
        log.info(f"{tag} No new entries. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"{tag} Found {len(new_df)} new entries")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['ordinance_number']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["ordinance_number"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, BER_REPO, BER_CACHE, log)
    log.info(f"{tag} Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Federal public laws (GovInfo API)
# ===========================================================================

FED_REPO = os.getenv("FED_DATASET_REPO", "chrissoria/federal-public-laws")
FED_CACHE = SCRIPTS_DIR / ".fed_cache.parquet"
GOVINFO_BASE = "https://api.govinfo.gov"


def update_federal(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for federal public laws."""
    import re
    from bs4 import BeautifulSoup

    tag = "[Federal]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    api_key = os.getenv("GOVINFO_API_KEY") or os.getenv("DATA_GOV_API_KEY") or "DEMO_KEY"
    if api_key == "DEMO_KEY":
        log.warning(f"{tag} Using DEMO_KEY — rate limits apply")

    existing_df = load_hf_dataset(FED_REPO, FED_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "federal" / "federal_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"{tag} Loaded {len(existing_df)} rows from build checkpoint")
        else:
            log.error(f"{tag} No existing dataset. Run build_federal_dataset.py first.")
            return 1

    existing_ids = set(existing_df["package_id"].astype(str).tolist())
    log.info(f"{tag} Existing: {len(existing_df)} rows")

    # Fetch recent packages (last 90 days to catch any new laws)
    from datetime import timedelta
    since_dt = datetime.now(timezone.utc) - timedelta(days=90)
    since_str = since_dt.strftime("%Y-%m-%dT00:00:00Z")

    log.info(f"{tag} Checking for new public laws since {since_str[:10]}...")

    try:
        all_pkgs = []
        offset = 0
        while True:
            resp = requests.get(
                f"{GOVINFO_BASE}/collections/PLAW/{since_str}",
                params={"offset": offset, "pageSize": 100, "api_key": api_key},
                headers={"User-Agent": "cat-pol-updater/1.0"},
                timeout=60,
            )
            if resp.status_code == 429:
                import time
                log.warning(f"{tag} Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()
            pkgs = data.get("packages", [])
            if not pkgs:
                break
            all_pkgs.extend(pkgs)
            offset += 100
            if not data.get("nextPage"):
                break
            _jittered_sleep(0.5)
    except Exception as e:
        log.error(f"{tag} Failed to fetch package listings: {e}")
        return 1

    # Filter to public laws only, skip existing
    public_pkgs = [p for p in all_pkgs if p.get("docClass", "").upper() == "PUBLIC"]
    new_pkgs = [p for p in public_pkgs if p["packageId"] not in existing_ids]
    log.info(f"{tag} Found {len(public_pkgs)} recent public laws, {len(new_pkgs)} are new")

    if not new_pkgs:
        log.info(f"{tag} No new laws. Up to date.")
        return 0

    def _strip_html(html):
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style"]):
            t.decompose()
        text = soup.get_text(separator="\n")
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    new_rows = []
    for pkg in new_pkgs:
        pid = pkg["packageId"]
        match = re.match(r"PLAW-(\d+)publ(\d+)", pid)
        congress_num = match.group(1) if match else ""
        law_num = match.group(2) if match else ""

        # Fetch summary
        short_title, pages, pdf_url = "", "", ""
        try:
            sr = requests.get(
                f"{GOVINFO_BASE}/packages/{pid}/summary",
                params={"api_key": api_key}, timeout=60,
            )
            if sr.status_code == 200:
                sd = sr.json()
                st = sd.get("shortTitle", [])
                if st and isinstance(st, list):
                    short_title = st[0].get("title", "")
                pages = str(sd.get("pages", ""))
                pdf_url = sd.get("download", {}).get("pdfLink", "")
            _jittered_sleep(0.5)
        except Exception:
            pass

        # Fetch text
        text = ""
        try:
            tr = requests.get(
                f"{GOVINFO_BASE}/packages/{pid}/htm",
                params={"api_key": api_key}, timeout=120,
            )
            if tr.status_code == 200:
                text = _strip_html(tr.text)
            _jittered_sleep(1.0)
        except Exception:
            pass

        new_rows.append({
            "date": pkg.get("dateIssued", "")[:10],
            "congress": str(congress_num),
            "law_number": f"Pub.L. {congress_num}-{law_num}" if law_num else "",
            "package_id": pid,
            "doc_type": "public_law",
            "title": pkg.get("title", ""),
            "short_title": short_title,
            "pages": pages,
            "text": text,
            "url": f"https://www.govinfo.gov/app/details/{pid}",
            "pdf_url": pdf_url,
        })

    new_df = pd.DataFrame(new_rows)
    log.info(f"{tag} Fetched {len(new_df)} new laws")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['law_number']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["package_id"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, FED_REPO, FED_CACHE, log)
    log.info(f"{tag} Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Bakersfield (municipal.codes, Playwright — same as Berkeley)
# ===========================================================================

BAK_REPO = os.getenv("BAK_DATASET_REPO", "chrissoria/bakersfield-ordinances")
BAK_CACHE = SCRIPTS_DIR / ".bak_cache.parquet"
BAK_BASE_URL = "https://bakersfield.municipal.codes"


def update_bakersfield(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for Bakersfield ordinances (requires Playwright)."""
    tag = "[Bakersfield]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error(f"{tag} Playwright not installed. Skipping.")
        return 1

    existing_df = load_hf_dataset(BAK_REPO, BAK_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "bakersfield_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"{tag} Loaded {len(existing_df)} rows from checkpoint")
        else:
            log.error(f"{tag} No existing dataset. Run build_bakersfield_dataset.py first.")
            return 1

    existing_ords = set(existing_df["ordinance_number"].astype(str).tolist())
    log.info(f"{tag} Existing: {len(existing_df)} rows")

    new_rows = []
    max_pages = 3

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(f"{BAK_BASE_URL}/enactments", timeout=60000)
            page.wait_for_timeout(5000)

            found_existing = False
            for page_num in range(1, max_pages + 1):
                url = f"{BAK_BASE_URL}/enactments?page={page_num}"
                page.goto(url, timeout=30000)
                page.wait_for_timeout(2000)

                tables = page.query_selector_all("table")
                if len(tables) < 2:
                    break

                rows = tables[1].query_selector_all("tr")
                page_new = 0

                for row in rows:
                    cells = row.query_selector_all("td")
                    if len(cells) < 4:
                        continue

                    a_tag = cells[0].query_selector("a")
                    ord_num = a_tag.inner_text().strip() if a_tag else cells[0].inner_text().strip()
                    detail_url = ""
                    if a_tag:
                        detail_path = a_tag.get_attribute("href") or ""
                        detail_url = f"{BAK_BASE_URL}{detail_path}" if detail_path.startswith("/") else detail_path

                    if ord_num in existing_ords:
                        found_existing = True
                        continue

                    date_raw = cells[2].inner_text().strip()
                    date = ""
                    try:
                        from datetime import datetime as _dt
                        date = _dt.strptime(date_raw, "%m/%d/%Y").strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        if date_raw.strip().isdigit() and len(date_raw.strip()) == 4:
                            date = date_raw.strip()
                        else:
                            date = date_raw

                    entry = {
                        "date": date,
                        "ordinance_number": ord_num,
                        "doc_type": "ordinance",
                        "title": cells[1].inner_text().strip(),
                        "text": "",
                        "disposition": cells[3].inner_text().strip(),
                        "effective_date": "",
                        "url": detail_url,
                        "year": int(date[:4]) if date and date[:4].isdigit() else 0,
                    }

                    if detail_url:
                        try:
                            page.goto(detail_url, timeout=30000)
                            page.wait_for_timeout(1500)
                            main = page.query_selector("#main-column, main, article")
                            entry["text"] = main.inner_text().strip() if main else page.inner_text("body").strip()
                        except Exception:
                            pass
                        _jittered_sleep(2.0)

                    new_rows.append(entry)
                    page_new += 1

                log.info(f"{tag} Page {page_num}: {page_new} new entries")
                if found_existing and page_new == 0:
                    break
                _jittered_sleep(3.0)

            browser.close()
    except Exception as e:
        log.error(f"{tag} Playwright failed: {e}")
        if not new_rows:
            return 1

    if not new_rows:
        log.info(f"{tag} No new entries. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"{tag} Found {len(new_df)} new entries")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['ordinance_number']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["ordinance_number"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, BAK_REPO, BAK_CACHE, log)
    log.info(f"{tag} Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Executive Orders (Federal Register API + UCSB)
# ===========================================================================

EO_REPO = os.getenv("EO_DATASET_REPO", "chrissoria/executive-orders")
EO_CACHE = SCRIPTS_DIR / ".eo_cache.parquet"


def update_executive_orders(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for executive orders via Federal Register API."""
    tag = "[ExecOrders]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(EO_REPO, EO_CACHE, log)
    if existing_df.empty:
        log.error(f"{tag} No existing dataset.")
        return 1

    existing_ids = set()
    if "document_number" in existing_df.columns:
        existing_ids = set(existing_df["document_number"].astype(str).tolist())
    elif "url" in existing_df.columns:
        existing_ids = set(existing_df["url"].astype(str).tolist())
    log.info(f"{tag} Existing: {len(existing_df)} orders")

    # Fetch recent EOs from Federal Register (no auth needed)
    import re
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(
            "https://www.federalregister.gov/api/v1/documents.json",
            params={
                "conditions[type]": "PRESDOCU",
                "conditions[presidential_document_type]": "executive_order",
                "per_page": 100,
                "page": 1,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"{tag} API fetch failed: {e}")
        return 1

    new_rows = []
    for doc in data.get("results", []):
        doc_num = doc.get("document_number", "")
        if doc_num in existing_ids:
            continue

        president = doc.get("president", {})
        president_name = president.get("name", "") if isinstance(president, dict) else str(president)

        # Fetch text
        text = ""
        body_url = doc.get("body_html_url", "")
        if body_url:
            try:
                tr = requests.get(body_url, timeout=60)
                if tr.status_code == 200:
                    soup = BeautifulSoup(tr.text, "html.parser")
                    text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()
            except Exception:
                pass

        new_rows.append({
            "date": doc.get("publication_date", ""),
            "title": doc.get("title", ""),
            "executive_order_number": str(doc.get("executive_order_number", "")),
            "president": president_name,
            "document_number": doc_num,
            "text": text,
            "html_url": doc.get("html_url", ""),
            "pdf_url": doc.get("pdf_url", ""),
            "signing_date": doc.get("signing_date", ""),
        })

    if not new_rows:
        log.info(f"{tag} No new executive orders. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"{tag} Found {len(new_df)} new executive orders")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | EO {r['executive_order_number']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    id_col = "document_number" if "document_number" in updated.columns else "url"
    updated = updated.drop_duplicates(subset=[id_col], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, EO_REPO, EO_CACHE, log)
    log.info(f"{tag} Done. {len(updated)} total (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Presidential Speeches (SOTU + Inaugurals from UCSB)
# ===========================================================================

SPEECH_REPO = os.getenv("SPEECH_DATASET_REPO", "chrissoria/presidential-speeches")
SPEECH_CACHE = SCRIPTS_DIR / ".speech_cache.parquet"


def update_speeches(log: logging.Logger, dry_run: bool = False) -> int:
    """Check for new presidential speeches on UCSB."""
    tag = "[Speeches]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(SPEECH_REPO, SPEECH_CACHE, log)
    if existing_df.empty:
        log.error(f"{tag} No existing dataset.")
        return 1

    log.info(f"{tag} Existing: {len(existing_df)} speeches")
    # Speeches don't change frequently — just log and skip
    log.info(f"{tag} No automated SOTU/inaugural detection. Up to date.")
    return 0


# ===========================================================================
# Trump Truth Social (CNN/Stiles archive)
# ===========================================================================

TS_REPO = os.getenv("TS_DATASET_REPO", "chrissoria/trump-truth-social")
TS_CACHE = SCRIPTS_DIR / ".ts_cache.parquet"
TS_ARCHIVE_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"

# Trump presidential terms (inauguration dates)
_TRUMP_TERMS = [
    ("2017-01-20", "2021-01-20"),
    ("2025-01-20", None),  # current term, no end date
]

# Trump president-elect periods (election day to day before inauguration)
_TRUMP_ELECT_TERMS = [
    ("2016-11-08", "2017-01-20"),
    ("2024-11-05", "2025-01-20"),
]


def _is_president(date_str: str) -> bool:
    """Return True if the given date falls within a Trump presidential term."""
    if not date_str or len(date_str) < 10:
        return False
    d = date_str[:10]
    for start, end in _TRUMP_TERMS:
        if d >= start and (end is None or d < end):
            return True
    return False


def _is_president_elect(date_str: str) -> bool:
    """Return True if the given date falls within a Trump president-elect period."""
    if not date_str or len(date_str) < 10:
        return False
    d = date_str[:10]
    for start, end in _TRUMP_ELECT_TERMS:
        if d >= start and d < end:
            return True
    return False


def update_trump_truths(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for Trump Truth Social posts."""
    tag = "[TruthSocial]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    from html import unescape
    from bs4 import BeautifulSoup

    # Load existing dataset
    existing_df = load_hf_dataset(TS_REPO, TS_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "truthsocial" / "trump_truths.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"{tag} Loaded {len(existing_df)} posts from local checkpoint")

    existing_ids = set()
    if not existing_df.empty:
        existing_ids = set(existing_df["post_id"].astype(str).tolist())
    log.info(f"{tag} Existing: {len(existing_ids)} posts")

    # Fetch full archive from CNN
    log.info(f"{tag} Fetching archive from CNN...")
    try:
        resp = requests.get(TS_ARCHIVE_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"{tag} Failed to fetch archive: {e}")
        return 1

    log.info(f"{tag} Archive contains {len(data)} posts")

    def html_to_text(html):
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        return unescape(soup.get_text()).strip()

    # Find new posts
    new_rows = []
    for post in data:
        pid = str(post.get("id", ""))
        if pid in existing_ids:
            continue

        content_html = post.get("content", "")
        media = post.get("media", [])
        media_urls = [
            m.get("url", "") if isinstance(m, dict) else str(m)
            for m in media
            if m
        ]
        created = post.get("created_at", "")

        # Extract links
        links = []
        if content_html:
            try:
                soup = BeautifulSoup(content_html, "html.parser")
                links = [a["href"] for a in soup.find_all("a", href=True)]
            except Exception:
                pass

        new_rows.append({
            "date": created[:10] if created else "",
            "datetime": created,
            "text": html_to_text(content_html),
            "content_html": content_html,
            "url": post.get("url", ""),
            "post_id": pid,
            "replies_count": post.get("replies_count", 0),
            "reblogs_count": post.get("reblogs_count", 0),
            "favourites_count": post.get("favourites_count", 0),
            "media_urls": "; ".join(media_urls) if media_urls else "",
            "links": "; ".join(links) if links else "",
            "has_media": len(media_urls) > 0,
            "is_president": _is_president(created),
            "is_president_elect": _is_president_elect(created),
            "day_of_week": pd.Timestamp(created).day_name() if created else "",
            "time": created[11:19] if created and len(created) >= 19 else "",
        })

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    log.info(f"{tag} Found {len(new_df)} new posts")

    if dry_run:
        if not new_df.empty:
            for _, r in new_df.head(10).iterrows():
                log.info(f"  {r['date']} | {r['text'][:60]}")
        else:
            log.info(f"{tag} No new posts. Up to date.")
        return 0

    # Update engagement counts and media URLs for existing posts
    if not existing_df.empty:
        archive_map = {str(p["id"]): p for p in data}
        for idx, row in existing_df.iterrows():
            pid = str(row["post_id"])
            if pid in archive_map:
                p = archive_map[pid]
                existing_df.at[idx, "replies_count"] = p.get("replies_count", 0)
                existing_df.at[idx, "reblogs_count"] = p.get("reblogs_count", 0)
                existing_df.at[idx, "favourites_count"] = p.get("favourites_count", 0)
                media = p.get("media", [])
                urls = [
                    m.get("url", "") if isinstance(m, dict) else str(m)
                    for m in media if m
                ]
                existing_df.at[idx, "media_urls"] = "; ".join(urls) if urls else ""
                existing_df.at[idx, "has_media"] = len(urls) > 0

    updated = pd.concat([new_df, existing_df], ignore_index=True) if not new_df.empty else existing_df.copy()
    updated = updated.drop_duplicates(subset=["post_id"], keep="first")
    updated = updated.sort_values("datetime", ascending=False).reset_index(drop=True)

    # Recompute status columns for all rows (handles backfill)
    updated["is_president"] = updated["date"].apply(_is_president)
    updated["is_president_elect"] = updated["date"].apply(_is_president_elect)
    updated["day_of_week"] = pd.to_datetime(updated["date"], errors="coerce").dt.day_name()
    updated["time"] = pd.to_datetime(updated["datetime"], errors="coerce").dt.strftime("%H:%M:%S").fillna("")

    # Merge stock/ETF prices (daily + intraday)
    _TS_TICKERS = {
        "^GSPC": "sp500", "DIA": "dia", "QQQ": "qqq", "DJT": "djt",
        "LMT": "lmt", "WAR": "war", "CNRG": "cnrg", "XLV": "xlv",
        "XPH": "xph", "GLD": "gld", "USO": "uso", "XLI": "xli",
        "EWW": "eww", "VGK": "vgk", "IBIT": "ibit", "FXI": "fxi",
        "TLT": "tlt", "UUP": "uup",
    }
    try:
        import yfinance as yf
        import numpy as np

        post_dt = pd.to_datetime(updated["datetime"], utc=True, errors="coerce")
        min_date = updated["date"].min()
        all_dates = pd.date_range(min_date, pd.Timestamp.now()).strftime("%Y-%m-%d")

        for yf_ticker, col_prefix in _TS_TICKERS.items():
            try:
                # Daily open/close
                daily = yf.download(yf_ticker, start=min_date, end=pd.Timestamp.now().strftime("%Y-%m-%d"), progress=False)
                if not daily.empty:
                    daily = daily.droplevel("Ticker", axis=1) if isinstance(daily.columns, pd.MultiIndex) else daily
                    if isinstance(daily.columns, pd.MultiIndex):
                        daily.columns = [c[0] if isinstance(c, tuple) else c for c in daily.columns]
                    daily.columns.name = None  # clear 'Price' name that breaks rename
                    daily.index = daily.index.strftime("%Y-%m-%d")
                    daily_full = daily[["Open", "Close"]].reindex(all_dates).ffill()
                    updated[f"{col_prefix}_open"] = updated["date"].map(daily_full["Open"]).round(2)
                    updated[f"{col_prefix}_close"] = updated["date"].map(daily_full["Close"]).round(2)

                # Intraday: build from 1m + 5m + 1h, cache locally
                intraday_cache = CHECKPOINT_DIR / f"{col_prefix}_intraday.parquet"
                frames = []
                for interval, period in [("1m", "7d"), ("5m", "60d"), ("1h", "730d")]:
                    data = yf.download(yf_ticker, period=period, interval=interval, progress=False)
                    if not data.empty:
                        data = data.droplevel("Ticker", axis=1) if isinstance(data.columns, pd.MultiIndex) else data
                        # Flatten any remaining MultiIndex columns
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                        data.columns.name = None  # clear 'Price' name
                        if "Close" in data.columns:
                            frames.append(data[["Close"]].rename(columns={"Close": "price"}))

                if frames:
                    intraday = pd.concat(frames).sort_index()
                    intraday = intraday[~intraday.index.duplicated(keep="first")]

                    # Merge with any existing cache
                    if intraday_cache.exists():
                        old = pd.read_parquet(intraday_cache)
                        if "price" in old.columns:
                            intraday = pd.concat([old[["price"]], intraday]).sort_index()
                        else:
                            intraday = pd.concat([old, intraday]).sort_index()
                        intraday = intraday[~intraday.index.duplicated(keep="last")]

                    intraday.to_parquet(intraday_cache)

                    sp_sorted = intraday.reset_index()
                    sp_sorted.columns = ["sp_dt", "price"]
                    # Unify datetime resolution to avoid merge_asof dtype mismatch
                    sp_sorted["sp_dt"] = pd.to_datetime(sp_sorted["sp_dt"], utc=True)

                    for offset_min, suffix, direction in [
                        (-60, "1hr_before", "backward"), (-5, "5min_before", "backward"),
                        (0, "at_post", "backward"), (5, "5min_after", "forward"),
                        (60, "1hr_after", "forward"),
                    ]:
                        col_name = f"{col_prefix}_{suffix}"
                        offset_dt = post_dt + pd.Timedelta(minutes=offset_min)
                        posts = pd.DataFrame({
                            "post_dt": pd.to_datetime(offset_dt, utc=True),
                            "orig_idx": updated.index,
                        }).dropna().sort_values("post_dt")
                        merged = pd.merge_asof(posts, sp_sorted, left_on="post_dt", right_on="sp_dt", direction=direction)
                        updated[col_name] = np.nan
                        updated.loc[merged["orig_idx"].values, col_name] = merged["price"].round(2).values

                log.info(f"{tag} Merged {yf_ticker} ({col_prefix}) data")
            except Exception as e:
                log.warning(f"{tag} Could not merge {yf_ticker}: {e}")
    except ImportError:
        log.warning(f"{tag} yfinance not installed, skipping stock data")

    # Merge GDELT geopolitical event data
    try:
        import time as _time
        import zipfile
        import io

        gdelt_cache = CHECKPOINT_DIR / "gdelt_daily.parquet"
        gdelt_existing = pd.read_parquet(gdelt_cache) if gdelt_cache.exists() else pd.DataFrame()

        # Determine which dates we need
        all_post_dates = set(updated["date"].unique())
        existing_dates = set(gdelt_existing["date"].tolist()) if not gdelt_existing.empty else set()
        missing_dates = sorted(all_post_dates - existing_dates)

        # Only fetch recent missing dates (GDELT raw files cover individual 15-min windows)
        # Use the daily export file list for efficiency
        if missing_dates:
            log.info(f"{tag} Fetching GDELT data for {len(missing_dates)} missing dates...")
            new_gdelt_rows = []
            for date_str in missing_dates[-30:]:  # limit to last 30 missing dates per run
                ymd = date_str.replace("-", "")
                # GDELT daily export URL
                export_url = f"http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"
                try:
                    _time.sleep(1)
                    resp = requests.get(export_url, timeout=60)
                    if resp.status_code != 200:
                        continue
                    z = zipfile.ZipFile(io.BytesIO(resp.content))
                    fname = z.namelist()[0]
                    events = pd.read_csv(z.open(fname), sep="\t", header=None, low_memory=False)
                    # Filter US-involved events
                    us = events[(events[7] == "USA") | (events[17] == "USA")]
                    if us.empty:
                        continue
                    row = {
                        "date": date_str,
                        "gdelt_military": int(us[us[28].isin(["18", "19", "20"])].shape[0]),
                        "gdelt_sanctions": int(us[us[28] == "17"].shape[0]),
                        "gdelt_threat": int(us[us[28] == "13"].shape[0]),
                        "gdelt_protest": int(us[us[28] == "14"].shape[0]),
                        "gdelt_force_posture": int(us[us[28] == "15"].shape[0]),
                        "gdelt_diplomatic": int(us[us[28].isin(["01","02","03","04","05","06","07","08"])].shape[0]),
                        "gdelt_material_conflict": int(us[us[29] == 4].shape[0]),
                        "gdelt_verbal_conflict": int(us[us[29] == 3].shape[0]),
                        "gdelt_material_cooperation": int(us[us[29] == 2].shape[0]),
                        "gdelt_verbal_cooperation": int(us[us[29] == 1].shape[0]),
                        "gdelt_goldstein_avg": round(us[30].mean(), 2) if us[30].notna().any() else 0,
                        "gdelt_avg_tone": round(us[34].mean(), 2) if us[34].notna().any() else 0,
                        "gdelt_total_events": int(len(us)),
                    }
                    new_gdelt_rows.append(row)
                    log.info(f"{tag} GDELT {date_str}: {row['gdelt_total_events']} events")
                except Exception as e:
                    log.warning(f"{tag} GDELT fetch failed for {date_str}: {e}")

            if new_gdelt_rows:
                new_gdelt = pd.DataFrame(new_gdelt_rows)
                gdelt_existing = pd.concat([gdelt_existing, new_gdelt], ignore_index=True)
                gdelt_existing = gdelt_existing.drop_duplicates(subset=["date"], keep="last")
                gdelt_existing = gdelt_existing.sort_values("date").reset_index(drop=True)

        # Compute derived columns on full history
        if not gdelt_existing.empty:
            import numpy as np
            gd = gdelt_existing.sort_values("date").copy()
            # Percentages
            for col in ["gdelt_military", "gdelt_sanctions", "gdelt_threat", "gdelt_protest", "gdelt_force_posture", "gdelt_diplomatic"]:
                gd[col + "_pct"] = (gd[col] / gd["gdelt_total_events"] * 100).round(2)
            # Z-scores
            for col in ["gdelt_military", "gdelt_sanctions", "gdelt_threat", "gdelt_protest", "gdelt_material_conflict"]:
                mean, std = gd[col].mean(), gd[col].std()
                gd[col + "_zscore"] = ((gd[col] - mean) / std).round(2) if std > 0 else 0
            # Day-over-day deltas
            for col in ["gdelt_military", "gdelt_sanctions", "gdelt_threat", "gdelt_protest", "gdelt_material_conflict", "gdelt_goldstein_avg", "gdelt_avg_tone"]:
                gd[col + "_delta"] = gd[col].diff().round(2)

            gd.to_parquet(gdelt_cache, index=False)

            # Merge into dataset
            gd_indexed = gd.set_index("date")
            gdelt_cols = [c for c in gd.columns if c != "date"]
            for col in gdelt_cols:
                updated[col] = updated["date"].map(gd_indexed[col])
            log.info(f"{tag} Merged {len(gdelt_cols)} GDELT columns")
    except Exception as e:
        log.warning(f"{tag} Could not merge GDELT data: {e}")

    # Summarize new images with alt-text
    try:
        import cat_pol as pol
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            needs_summary = updated[
                (updated["media_urls"].str.len() > 0) &
                (~updated["media_urls"].str.lower().str.endswith(".mp4")) &
                (updated.get("image_alt_text", pd.Series([""] * len(updated))).fillna("").str.len() == 0) &
                (pd.to_datetime(updated["datetime"], errors="coerce") >= "2024-11-05")
            ]
            if len(needs_summary) > 0:
                log.info(f"{tag} Summarizing {len(needs_summary)} new images...")
                for batch_start in range(0, len(needs_summary), 25):
                    batch = needs_summary.iloc[batch_start:batch_start + 25]
                    # For multi-image posts, pick first non-video URL
                    def _pick_image(media_str):
                        for u in str(media_str).split(";"):
                            u = u.strip()
                            if u and not u.lower().endswith(".mp4"):
                                return u
                        return str(media_str).split(";")[0].strip()
                    batch_urls = [_pick_image(u) for u in batch["media_urls"].tolist()]
                    try:
                        result = pol.summarize(
                            input_data=batch_urls,
                            format="alt-text",
                            tone=None,
                            user_model="qwen/qwen2.5-vl-72b-instruct:novita",
                            model_source="huggingface",
                            api_key=hf_key,
                            description="Images posted by Donald Trump on Truth Social",
                            thinking_budget=0,
                        )
                        for i, (orig_idx, _) in enumerate(batch.iterrows()):
                            if i < len(result) and result.iloc[i].get("processing_status") == "success":
                                updated.at[orig_idx, "image_alt_text"] = result.iloc[i].get("summary", "")
                        log.info(f"{tag} Image batch {batch_start // 25 + 1}: done")
                    except Exception as e:
                        log.warning(f"{tag} Image batch failed: {e}")
            else:
                log.info(f"{tag} No new images to summarize")
    except Exception as e:
        log.warning(f"{tag} Could not summarize images: {e}")

    # Classify new text posts with 5-model ensemble
    _CAT_COLS = [
        "cat_attacking_individual", "cat_attacking_opposition", "cat_threatening_intl",
        "cat_enacting_aggressive", "cat_enacting_nonaggressive", "cat_deescalating",
        "cat_praising_endorsing", "cat_self_promotion", "cat_other",
    ]
    _CAT_NAMES = [
        "Attacking individual — targeting a specific person by name",
        "Attacking political opposition — targeting Democrats, a party, or political group broadly",
        "Threatening/escalating internationally — conditional threats, tariff warnings, military posturing",
        "Enacting aggressive policy — imposing tariffs, sanctions, bans, military action (already done)",
        "Enacting non-aggressive policy — signing bills, executive orders, domestic programs, appointments",
        "De-escalating conflict — toning down, announcing deals, peace talks, ceasefire",
        "Praising/endorsing — positive statements about a person, leader, ally",
        "Self-promotion — boasting about achievements, economy, polls, ratings",
        "Other — does not fit any above category",
    ]
    _CAT_SYSTEM_PROMPT = (
        'Classification guidance per category:\n\n'
        '- Attacking individual — targeting a specific person by name: Assign "Attacking individual" when a text explicitly criticizes or attacks a specific person by name, and not when it simply mentions or praises them, or shares a link discussing them without commentary. Do not assign this category to texts that broadly target a party, group, or ideology unless they specifically name and attack an individual.\n'
        '- Attacking political opposition: Assign this category when a text explicitly criticizes or attacks Democrats, a party, or a political group broadly, and not just a specific individual. Do not assign this category when a text is just sharing a link or when the criticism is directed at a specific person rather than a broader group or party.\n'
        '- Enacting aggressive policy: Assign when the text explicitly describes or reports a specific aggressive action that has been taken, such as imposing tariffs, enforcing sanctions, implementing bans, or conducting military operations. Do not assign to URLs or links alone.\n'
        '- Enacting non-aggressive policy: Assign when the text explicitly mentions the implementation of a domestic policy, program, or appointment through actions like signing bills or executive orders, and does not involve aggressive actions.\n'
        '- Praising/endorsing: Assign when a text contains explicit positive statements or endorsements about a specific person, leader, or ally. Do not assign when a text merely reports on someone\'s actions or shares a link without expressing clear positive sentiment.\n'
        '- URL/link only: Do not assign — filter these to Other instead.\n'
    )
    try:
        import cat_pol as pol
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")

        if hf_key and openai_key and anthropic_key and google_key:
            # Ensure cat columns exist
            for col in _CAT_COLS:
                if col not in updated.columns:
                    updated[col] = pd.NA

            # Find unclassified text posts (since election only)
            needs_classify = updated[
                (updated["text"].fillna("").str.strip().str.len() > 20) &
                (updated["cat_other"].isna()) &
                (pd.to_datetime(updated["datetime"], errors="coerce") >= "2024-11-05")
            ]
            if len(needs_classify) > 0:
                log.info(f"{tag} Classifying {len(needs_classify)} new text posts (5-model ensemble)...")
                try:
                    result = pol.classify(
                        input_data=needs_classify["text"].tolist(),
                        categories=_CAT_NAMES,
                        document_context="Trump Truth Social posts",
                        system_prompt=_CAT_SYSTEM_PROMPT,
                        models=[
                            ("meta-llama/llama-4-maverick-17b-128e-instruct-fp8:novita", "huggingface", hf_key, {"creativity": 0}),
                            ("qwen/qwen3-32b-fp8:novita", "huggingface", hf_key, {"creativity": 0}),
                            ("deepseek/deepseek-v3.2:novita", "huggingface", hf_key, {"creativity": 0}),
                        ],
                        add_other=False,
                        check_verbosity=False,
                        safety=True,
                        filename="truthsocial_classify_checkpoint",
                        save_directory=str(CHECKPOINT_DIR / "truthsocial"),
                        thinking_budget=0,
                    )
                    # Map consensus columns back
                    consensus_cols = [c for c in result.columns if "consensus" in c]
                    for i, (orig_idx, _) in enumerate(needs_classify.iterrows()):
                        if i < len(result):
                            for cat_col, cons_col in zip(_CAT_COLS, consensus_cols):
                                updated.at[orig_idx, cat_col] = result.iloc[i].get(cons_col, 0)
                    classified = result["processing_status"].eq("success").sum()
                    log.info(f"{tag} Classified {classified}/{len(needs_classify)} posts")
                except Exception as e:
                    log.warning(f"{tag} Classification failed: {e}")
            else:
                log.info(f"{tag} No new posts to classify")
        else:
            log.warning(f"{tag} Missing API keys for classification (need HF, OpenAI, Anthropic, Google)")
    except Exception as e:
        log.warning(f"{tag} Could not classify posts: {e}")

    # Reorder columns: date/time first, then content, then classifications, then market/GDELT
    cat_cols = sorted([c for c in updated.columns if c.startswith("cat_")])
    col_order = [
        "date", "time", "day_of_week", "datetime",
        "text", "content_html", "url",
    ] + cat_cols + [
        "post_id",
        "is_president", "is_president_elect",
        "replies_count", "reblogs_count", "favourites_count",
        "media_urls", "links", "has_media", "image_alt_text",
    ]
    # Include any extra columns not in the list
    col_order += [c for c in updated.columns if c not in col_order]
    # Only keep columns that exist
    col_order = [c for c in col_order if c in updated.columns]
    updated = updated[col_order]

    push_hf_dataset(updated, TS_REPO, TS_CACHE, log)
    log.info(f"{tag} Done. {len(updated)} total posts (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Los Angeles (City Clerk Connect)
# ===========================================================================

LA_REPO = os.getenv("LA_DATASET_REPO", "chrissoria/la-ordinances")
LA_CACHE = SCRIPTS_DIR / ".la_cache.parquet"


def update_la(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for LA ordinances."""
    from bs4 import BeautifulSoup

    tag = "[LA]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(LA_REPO, LA_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "la" / "la_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"{tag} Loaded {len(existing_df)} rows from build checkpoint")
        else:
            log.error(f"{tag} No existing dataset. Run build_la_dataset.py first.")
            return 1

    existing_ords = set(existing_df["ordinance_number"].astype(str).tolist())
    # Find the highest ordinance number to search from
    max_ord = max(int(o) for o in existing_ords if o.isdigit())
    log.info(f"{tag} Existing: {len(existing_df)} rows, max ord: {max_ord}")

    # Search for new ordinances starting from the highest known prefix
    session = requests.Session()
    search_url = "https://cityclerk.lacity.org/lacityclerkconnect/index.cfm"
    start_prefix = max_ord // 10  # e.g., 18888 -> 1888
    new_rows = []

    for prefix in range(start_prefix, start_prefix + 10):
        query = f"{prefix}*"
        try:
            resp = session.post(
                f"{search_url}?fa=vord.doSearch&s=2",
                data={"DOCIDSearch": query, "StartDate": "", "EndDate": ""},
                headers={"User-Agent": "cat-pol-updater/1.0"},
                timeout=60,
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"{tag} Search failed for {query}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                ord_num = cells[0].get_text(strip=True)
                if not ord_num or not ord_num[0].isdigit():
                    continue
                if ord_num in existing_ords:
                    continue

                date_raw = cells[3].get_text(strip=True)
                date = ""
                try:
                    from datetime import datetime as _dt
                    date = _dt.strptime(date_raw, "%m/%d/%Y").strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    date = date_raw

                pdf_url = ""
                pdf_link = cells[0].find("a")
                if pdf_link and pdf_link.get("href", "").endswith(".pdf"):
                    pdf_url = pdf_link["href"]
                    if not pdf_url.startswith("http"):
                        pdf_url = f"https://cityclerk.lacity.org{pdf_url}"

                cf_url = ""
                cf_link = cells[1].find("a")
                if cf_link and cf_link.get("href"):
                    href = cf_link["href"]
                    if not href.startswith("http"):
                        cf_url = f"https://cityclerk.lacity.org/lacityclerkconnect/{href}"
                    else:
                        cf_url = href

                new_rows.append({
                    "date": date,
                    "ordinance_number": ord_num,
                    "council_file": cells[1].get_text(strip=True),
                    "doc_type": "ordinance",
                    "title": cells[2].get_text(strip=True),
                    "text": "",
                    "pdf_url": pdf_url,
                    "council_file_url": cf_url,
                })

        _jittered_sleep(3.0)

    if not new_rows:
        log.info(f"{tag} No new ordinances. Up to date.")
        return 0

    # Extract text from PDFs
    for row in new_rows:
        if row["pdf_url"]:
            row["text"] = extract_pdf_text(row["pdf_url"], session)
            _jittered_sleep(0.3)

    new_df = pd.DataFrame(new_rows)
    text_count = (new_df["text"].str.len() > 0).sum()
    log.info(f"{tag} Found {len(new_df)} new ordinances ({text_count} with text)")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['ordinance_number']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["ordinance_number"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, LA_REPO, LA_CACHE, log)
    log.info(f"{tag} Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# San Diego County (Legistar — title-based filtering)
# ===========================================================================

SDCO_REPO = os.getenv("SDCO_DATASET_REPO", "chrissoria/sd-county-ordinances")
SDCO_CACHE = SCRIPTS_DIR / ".sdco_cache.parquet"


def update_sd_county(log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for SD County ordinances/resolutions."""
    tag = "[SD County]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(SDCO_REPO, SDCO_CACHE, log)
    if existing_df.empty:
        fallback = CHECKPOINT_DIR / "sdcounty" / "sdcounty_final_dataset.parquet"
        if fallback.exists():
            existing_df = pd.read_parquet(fallback)
            log.info(f"{tag} Loaded {len(existing_df)} rows from checkpoint")
        else:
            log.error(f"{tag} No existing dataset. Run build_sdcounty_dataset.py first.")
            return 1

    existing_ids = set(existing_df["matter_id"].astype(str).tolist())
    log.info(f"{tag} Existing: {len(existing_df)} rows")

    session = requests.Session()
    new_rows = []

    for keyword, doc_type in [("ORDINANCE", "ordinance"), ("RESOLUTION", "resolution")]:
        log.info(f"{tag} Fetching recent {doc_type}s...")
        try:
            resp = requests.get(
                "https://webapi.legistar.com/v1/sdcounty/matters",
                params={
                    "$filter": f"substringof('{keyword}',MatterTitle)",
                    "$top": 200,
                    "$orderby": "MatterIntroDate desc",
                },
                timeout=30,
            )
            resp.raise_for_status()
            matters = resp.json()
        except Exception as e:
            log.error(f"{tag} API fetch failed for {doc_type}: {e}")
            continue

        type_new = []
        for matter in matters:
            mid = str(matter["MatterId"])
            if mid in existing_ids:
                continue

            intro_date = matter.get("MatterIntroDate", "") or ""
            date = intro_date[:10] if intro_date else ""
            try:
                year = int(date[:4]) if date else 0
            except ValueError:
                year = 0

            # Get attachment
            att_url = ""
            try:
                att_resp = requests.get(
                    f"https://webapi.legistar.com/v1/sdcounty/matters/{matter['MatterId']}/attachments",
                    timeout=15,
                )
                atts = att_resp.json()
                for a in atts:
                    url = a.get("MatterAttachmentHyperlink", "")
                    if url.endswith(".pdf") or url.endswith(".docx"):
                        att_url = url
                        break
                if not att_url:
                    for a in atts:
                        url = a.get("MatterAttachmentHyperlink", "")
                        if url:
                            att_url = url
                            break
            except Exception:
                pass
            _jittered_sleep(0.3)

            text = ""
            if att_url:
                text = extract_pdf_text(att_url, session)
                _jittered_sleep(0.5)

            type_new.append({
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
            })

        log.info(f"{tag}   {len(type_new)} new {doc_type}s")
        new_rows.extend(type_new)

    if not new_rows:
        log.info(f"{tag} No new entries. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"{tag} Found {len(new_df)} new entries")

    if dry_run:
        for _, r in new_df.iterrows():
            log.info(f"  {r['date']} | {r['matter_file']} | {r['title'][:60]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["matter_id"], keep="first")
    updated = updated.sort_values("date", ascending=False).reset_index(drop=True)

    push_hf_dataset(updated, SDCO_REPO, SDCO_CACHE, log)
    log.info(f"{tag} Done. {len(updated)} total rows (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Code Publishing cities (Clovis, Newport Beach)
# ===========================================================================

CODEPUB_CITIES = {
    "clovis": {
        "name": "Clovis",
        "state": "CA",
        "prefix": "Clovis",
        "repo": "chrissoria/clovis-ordinances",
        "cache": SCRIPTS_DIR / ".clovis_cache.parquet",
        "titles": list(range(1, 11)),
    },
    "nb": {
        "name": "Newport Beach",
        "state": "CA",
        "prefix": "NewportBeach",
        "repo": "chrissoria/newport-beach-ordinances",
        "cache": SCRIPTS_DIR / ".nb_cache.parquet",
        "titles": list(range(1, 22)),
    },
}


def update_codepub_city(city_key: str, log: logging.Logger, dry_run: bool = False) -> int:
    """Incremental update for a Code Publishing city."""
    import re

    cfg = CODEPUB_CITIES[city_key]
    tag = f"[{cfg['name']}]"
    log.info("=" * 50)
    log.info(f"{tag} Starting update ({'DRY RUN' if dry_run else 'LIVE'})")

    existing_df = load_hf_dataset(cfg["repo"], cfg["cache"], log)
    if existing_df.empty:
        cp = CHECKPOINT_DIR / city_key / f"{city_key}_final_dataset.parquet"
        if cp.exists():
            existing_df = pd.read_parquet(cp)
            log.info(f"{tag} Loaded {len(existing_df)} sections from checkpoint")
        else:
            log.error(f"{tag} No existing dataset. Run build_codepublishing_dataset.py first.")
            return 1

    existing_urls = set(existing_df["url"].tolist())
    log.info(f"{tag} Existing: {len(existing_df)} sections")

    # Re-scrape all sections (Code Publishing pages are small, fast)
    session = requests.Session()
    base = f"https://www.codepublishing.com/{cfg['state']}/{cfg['prefix']}/html"
    new_rows = []

    for title_num in cfg["titles"]:
        title_dir = f"{cfg['prefix']}{title_num:02d}"
        for chapter in range(1, 100):
            url = f"{base}/{title_dir}/{title_dir}{chapter:02d}.html"
            if url in existing_urls:
                continue
            try:
                resp = session.get(url, headers={"User-Agent": "cat-pol-updater/1.0"}, timeout=15)
                if resp.status_code == 404:
                    if chapter > 5:
                        break
                    continue
                if resp.status_code != 200:
                    break
            except:
                break

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            content = soup.find("div", id="mainContent")
            if not content:
                continue

            text = content.get_text(separator="\n").strip()
            if len(text) < 20:
                continue

            h1 = content.find("h1")
            section_title = h1.get_text(strip=True) if h1 else ""
            filename = url.split("/")[-1].replace(".html", "")

            ord_refs = re.findall(
                r'(?:Ord\.|Ordinance)\s*([\d]+[-\w]*)'
                r'(?:.*?(?:eff\.?|effective)\s*(\w+\.?\s*\d{1,2},?\s*\d{4}))?',
                text
            )
            ord_ref_strs = [
                f"Ord. {num} (eff. {date})" if date else f"Ord. {num}"
                for num, date in ord_refs
            ]

            new_rows.append({
                "section_id": filename,
                "title_num": str(title_num),
                "chapter": str(chapter),
                "section_title": section_title,
                "text": text,
                "ordinance_refs": "; ".join(ord_ref_strs),
                "url": url,
            })

            _jittered_sleep(0.5)

    if not new_rows:
        log.info(f"{tag} No new sections. Up to date.")
        return 0

    new_df = pd.DataFrame(new_rows)
    log.info(f"{tag} Found {len(new_df)} new sections")

    if dry_run:
        for _, r in new_df.head(10).iterrows():
            log.info(f"  {r['section_id']} | {r['section_title'][:50]}")
        return 0

    updated = pd.concat([new_df, existing_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    push_hf_dataset(updated, cfg["repo"], cfg["cache"], log)
    log.info(f"{tag} Done. {len(updated)} total sections (+{len(new_df)} new)")
    return 0


# ===========================================================================
# Main
# ===========================================================================

ALL_CITIES = ["sd", "sf", "sal", "oak", "lb", "fre", "ber", "bak", "la", "clovis", "nb", "sdco", "fed", "eo", "speeches", "ts"]


def main():
    parser = argparse.ArgumentParser(description="Weekly dataset updater for all cities")
    parser.add_argument("--dry-run", action="store_true", help="Scrape but don't push")
    parser.add_argument("--city", choices=ALL_CITIES + ["all"], default="all", help="Which city to update")
    parser.add_argument("--exclude", nargs="+", choices=ALL_CITIES, default=[], help="Sources to exclude (e.g. --exclude ts)")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info(f"Dataset updater starting at {datetime.now(timezone.utc).isoformat()}")

    results = {}

    cities = ALL_CITIES if args.city == "all" else [args.city]
    cities = [c for c in cities if c not in args.exclude]

    for city in cities:
        if city == "sd":
            results["San Diego"] = update_san_diego(log, dry_run=args.dry_run)
        elif city == "sf":
            results["San Francisco"] = update_san_francisco(log, dry_run=args.dry_run)
        elif city == "sal":
            results["Salinas"] = update_salinas(log, dry_run=args.dry_run)
        elif city == "ber":
            results["Berkeley"] = update_berkeley(log, dry_run=args.dry_run)
        elif city == "bak":
            results["Bakersfield"] = update_bakersfield(log, dry_run=args.dry_run)
        elif city == "la":
            results["Los Angeles"] = update_la(log, dry_run=args.dry_run)
        elif city == "sdco":
            results["SD County"] = update_sd_county(log, dry_run=args.dry_run)
        elif city in CODEPUB_CITIES:
            cp_cfg = CODEPUB_CITIES[city]
            results[cp_cfg["name"]] = update_codepub_city(city, log, dry_run=args.dry_run)
        elif city == "fed":
            results["Federal"] = update_federal(log, dry_run=args.dry_run)
        elif city == "eo":
            results["Executive Orders"] = update_executive_orders(log, dry_run=args.dry_run)
        elif city == "speeches":
            results["Speeches"] = update_speeches(log, dry_run=args.dry_run)
        elif city == "ts":
            results["TruthSocial"] = update_trump_truths(log, dry_run=args.dry_run)
        elif city in LEGISTAR_CITIES:
            cfg = LEGISTAR_CITIES[city]
            results[cfg["name"]] = update_legistar_city(city, log, dry_run=args.dry_run)

    # Summary
    log.info("=" * 60)
    for city, rc in results.items():
        status = "OK" if rc == 0 else f"FAILED ({rc})"
        log.info(f"  {city}: {status}")

    log.info("All updates complete.")

    # =====================================================================
    # Post-update: classify any new unclassified SD/SF ordinances
    # (only on weekly Sunday runs, not daily TS-only runs)
    # =====================================================================
    classify_cities = {"sd", "sf", "sal"}
    if not args.dry_run and classify_cities & set(cities):
        _classify_new_ordinances(log)

    sys.exit(max(results.values()) if results else 0)


CLASSIFY_CATEGORIES = [
    "Community Plan Updates (e.g., comprehensive plans, neighborhood plans)",
    "Contract Amendments (e.g., modifications, revisions)",
    "Environmental Compliance (e.g., stormwater regulations, habitat protections)",
    "Infrastructure Projects (e.g., road improvements, water systems)",
    "Public Safety Communications (e.g., emergency response systems, communication networks)",
    "Construction Management Services (e.g., project oversight, coordination)",
    "Rezoning (e.g., residential to commercial, density adjustments)",
    "Historical Preservation (e.g., landmark designation, restoration projects)",
    "Health and Safety Applications (e.g., permits, inspections)",
    "Public Art Projects (e.g., murals, sculptures)",
    "Parking Management Services (e.g., lot management, enforcement)",
    "Housing Development (e.g., affordable housing, new construction)",
]


def _classify_new_ordinances(log: logging.Logger):
    """Classify any new unclassified ordinances in SD and SF datasets."""
    hf_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("CATLLM_HUGGINGFACE_TOKEN")
    if not hf_key:
        log.info("[Classify] No HuggingFace API key found, skipping classification.")
        return

    import cat_pol as pol

    for city, repo, cache, doc_filter in [
        ("SD", SD_REPO, SD_CACHE, lambda df: (df["doc_type"] == "ordinance") & (df["text"].str.len() > 0)),
        ("SF", SF_REPO, SF_CACHE, lambda df: df["text"].str.len() > 0),
        ("SAL", SAL_REPO, SAL_CACHE, lambda df: (df["doc_type"] == "ordinance") & (df["text"].str.len() > 0)),
    ]:
        tag = f"[Classify-{city}]"
        try:
            existing_df = load_hf_dataset(repo, cache, log)
            if existing_df.empty:
                continue

            # Initialize classification columns if they don't exist yet
            if "classification_status" not in existing_df.columns:
                log.info(f"{tag} Initializing classification columns (first run).")
                existing_df["classification_status"] = pd.NA

            # Find unclassified ordinances with text
            mask = doc_filter(existing_df) & (
                existing_df["classification_status"].isna() |
                (existing_df["classification_status"] == "")
            )
            unclassified = existing_df[mask]

            if len(unclassified) == 0:
                log.info(f"{tag} All ordinances classified. Up to date.")
                continue

            log.info(f"{tag} Found {len(unclassified)} unclassified ordinances.")

            # Classify with Qwen3-VL-235B via HuggingFace (novita router)
            results = pol.classify(
                input_data=unclassified["text"].tolist(),
                categories=CLASSIFY_CATEGORIES,
                document_context=f"{city} city ordinances",
                api_key=hf_key,
                user_model="qwen/qwen2.5-vl-72b-instruct:novita",
                model_source="huggingface",
                add_other=True,
                check_verbosity=False,
            )

            if results is None or results.empty:
                log.warning(f"{tag} Classification returned no results.")
                continue

            # Merge back
            cat_cols = [c for c in results.columns if c.startswith("category_")]
            for i, (orig_idx, _) in enumerate(unclassified.iterrows()):
                if i >= len(results):
                    break
                for col in cat_cols:
                    if col in results.columns:
                        existing_df.at[orig_idx, col] = results.iloc[i][col]
                existing_df.at[orig_idx, "classification_status"] = results.iloc[i]["processing_status"]

            classified = (results["processing_status"] == "success").sum()
            log.info(f"{tag} Classified {classified}/{len(unclassified)} new ordinances.")

            push_hf_dataset(existing_df, repo, cache, log)
            log.info(f"{tag} Pushed updated dataset.")

        except Exception as e:
            log.error(f"{tag} Classification failed: {e}")


if __name__ == "__main__":
    main()
