# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Source registry for cat-pol datasets.

Maps source identifiers to HuggingFace repos and standard column mappings.
All sources are accessed via the HuggingFace Datasets Server API
(no auth required, no downloads needed).

Source naming convention:
    {level}_{jurisdiction}     e.g., city_san_diego, federal_us

    level: city, county, state, federal
    jurisdiction: geographic name (snake_case)

Document types can be filtered via the doc_type parameter:
    ordinances, resolutions, public_laws, executive_orders, speeches, social_media
"""

from __future__ import annotations

import requests
import pandas as pd

HF_API = "https://datasets-server.huggingface.co/rows"
MAX_PER_PAGE = 100


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCES = {
    # ---- California cities (Democratic-leaning) ----
    "city_san_diego": {
        "repo": "chrissoria/san-diego-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance", "resolution"],
        "doc_type_col": "doc_type",  # column that holds the doc type
        "jurisdiction": "San Diego, CA",
        "level": "city",
    },
    "city_los_angeles": {
        "repo": "chrissoria/la-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance"],
        "doc_type_col": None,
        "jurisdiction": "Los Angeles, CA",
        "level": "city",
    },
    "city_san_francisco": {
        "repo": "chrissoria/sf-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance"],
        "doc_type_col": None,
        "jurisdiction": "San Francisco, CA",
        "level": "city",
    },
    "city_berkeley": {
        "repo": "chrissoria/berkeley-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance"],
        "doc_type_col": None,
        "jurisdiction": "Berkeley, CA",
        "level": "city",
    },
    "city_oakland": {
        "repo": "chrissoria/oakland-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance"],
        "doc_type_col": None,
        "jurisdiction": "Oakland, CA",
        "level": "city",
    },
    "city_long_beach": {
        "repo": "chrissoria/long-beach-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance", "resolution"],
        "doc_type_col": "doc_type",
        "jurisdiction": "Long Beach, CA",
        "level": "city",
    },
    "city_fresno": {
        "repo": "chrissoria/fresno-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance", "resolution"],
        "doc_type_col": "doc_type",
        "jurisdiction": "Fresno, CA",
        "level": "city",
    },
    "city_salinas": {
        "repo": "chrissoria/salinas-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance", "resolution"],
        "doc_type_col": "doc_type",
        "jurisdiction": "Salinas, CA",
        "level": "city",
    },
    # ---- California counties ----
    "county_san_diego": {
        "repo": "chrissoria/sd-county-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance", "resolution"],
        "doc_type_col": "doc_type",
        "jurisdiction": "San Diego County, CA",
        "level": "county",
    },
    # ---- California cities (Republican-leaning) ----
    "city_bakersfield": {
        "repo": "chrissoria/bakersfield-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance"],
        "doc_type_col": None,
        "jurisdiction": "Bakersfield, CA",
        "level": "city",
    },
    "city_clovis": {
        "repo": "chrissoria/clovis-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance"],
        "doc_type_col": None,
        "jurisdiction": "Clovis, CA",
        "level": "city",
    },
    "city_newport_beach": {
        "repo": "chrissoria/newport-beach-ordinances",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["ordinance"],
        "doc_type_col": None,
        "jurisdiction": "Newport Beach, CA",
        "level": "city",
    },
    # ---- Federal ----
    "federal_laws": {
        "repo": "chrissoria/federal-public-laws",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["public_law"],
        "doc_type_col": None,
        "jurisdiction": "United States",
        "level": "federal",
    },
    "federal_executive_orders": {
        "repo": "chrissoria/executive-orders",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["executive_order"],
        "doc_type_col": None,
        "jurisdiction": "United States",
        "level": "federal",
    },
    "federal_speeches": {
        "repo": "chrissoria/presidential-speeches",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["sotu", "inaugural", "opposition_response"],
        "doc_type_col": "speech_type",
        "jurisdiction": "United States",
        "level": "federal",
    },
    # ---- Social media ----
    "social_trump_truth": {
        "repo": "chrissoria/trump-truth-social",
        "text_col": "text",
        "date_col": "date",
        "doc_types": ["social_media_post"],
        "doc_type_col": None,
        "jurisdiction": "United States",
        "level": "federal",
    },
}


def list_sources(level: str = None) -> dict:
    """List available sources, optionally filtered by level.

    Args:
        level: Filter by level — "city", "county", "state", or "federal".
            None returns all sources.

    Returns:
        Dict of source_name → {repo, jurisdiction, level, doc_types}
    """
    result = {}
    for name, cfg in SOURCES.items():
        if level and cfg["level"] != level:
            continue
        result[name] = {
            "repo": cfg["repo"],
            "jurisdiction": cfg["jurisdiction"],
            "level": cfg["level"],
            "doc_types": cfg["doc_types"],
        }
    return result


def fetch_source(
    source: str,
    n: int = None,
    since: str = None,
    until: str = None,
    doc_type: str = None,
    text_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch data from a registered source on HuggingFace.

    Uses the `datasets` library for reliable access. Falls back to the
    HuggingFace Datasets Server REST API if `datasets` is not installed.

    Args:
        source: Source name (e.g., "city_san_diego", "federal_laws").
            Use list_sources() to see all available sources.
        n: Maximum number of rows to return.
        since: Only include rows on or after this date (YYYY-MM-DD).
        until: Only include rows on or before this date (YYYY-MM-DD).
        doc_type: Filter by document type (e.g., "ordinance", "resolution").
            Only applies to sources with multiple doc types.
        text_only: If True, drop rows with empty text. Default True.

    Returns:
        DataFrame with all original columns plus standardized `source` column.
    """
    if source not in SOURCES:
        available = ", ".join(sorted(SOURCES.keys()))
        raise ValueError(
            f"Unknown source '{source}'. Available sources: {available}"
        )

    cfg = SOURCES[source]
    repo = cfg["repo"]
    text_col = cfg["text_col"]
    date_col = cfg["date_col"]

    # Primary: use datasets library (no rate limits, cached locally)
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, split="train")
        df = ds.to_pandas()
    except ImportError:
        # Fallback: REST API with retry
        df = _fetch_via_api(repo)

    if df.empty:
        return df

    # Add source identifier
    df["source"] = source

    # Ensure standard text/date columns
    if text_col != "text" and text_col in df.columns:
        df["text"] = df[text_col]
    if date_col != "date" and date_col in df.columns:
        df["date"] = df[date_col]

    # Filter by doc_type
    if doc_type and cfg["doc_type_col"] and cfg["doc_type_col"] in df.columns:
        df = df[df[cfg["doc_type_col"]].str.lower().str.contains(doc_type.lower())]

    # Filter by date
    if since or until:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
        if since:
            df = df[df["_date"] >= pd.to_datetime(since)]
        if until:
            df = df[df["_date"] <= pd.to_datetime(until)]
        df = df.drop(columns=["_date"])

    # Filter empty text
    if text_only and "text" in df.columns:
        df = df[df["text"].notna() & (df["text"].str.strip() != "")]

    # Sort by date descending
    if "date" in df.columns:
        df = df.sort_values("date", ascending=False)

    # Limit
    if n is not None:
        df = df.head(n)

    return df.reset_index(drop=True)


def _fetch_via_api(repo: str) -> pd.DataFrame:
    """Fallback: fetch via HF Datasets Server REST API with retry."""
    import time

    all_rows = []
    offset = 0

    while True:
        for attempt in range(3):
            try:
                resp = requests.get(
                    HF_API,
                    params={
                        "dataset": repo,
                        "config": "default",
                        "split": "train",
                        "offset": offset,
                        "length": MAX_PER_PAGE,
                    },
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    return pd.DataFrame(all_rows)

        data = resp.json()
        rows = [r["row"] for r in data.get("rows", [])]
        if not rows:
            break

        all_rows.extend(rows)
        offset += MAX_PER_PAGE

        if not data.get("num_rows_total") or offset >= data["num_rows_total"]:
            break

    return pd.DataFrame(all_rows)
