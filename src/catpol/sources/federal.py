"""Federal public laws data loader.

Loads enacted federal public laws (1995-present) from HuggingFace,
sourced from the GovInfo API (Government Publishing Office).
"""

from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd


_DEFAULT_REPO = "chrissoria/federal-public-laws"


def fetch_federal_laws(
    n: int = None,
    since: str = None,
    until: str = None,
    congress: int = None,
    repo_id: str = _DEFAULT_REPO,
) -> pd.DataFrame:
    """
    Fetch enacted federal public laws from HuggingFace.

    Args:
        n: Maximum number of rows to return (most recent first).
        since: Only include laws on or after this date (YYYY-MM-DD).
        until: Only include laws on or before this date (YYYY-MM-DD).
        congress: Only include laws from this Congress number (e.g., 118).
        repo_id: HuggingFace dataset repository ID.

    Returns:
        DataFrame with columns: date, congress, law_number, package_id,
        doc_type, title, short_title, pages, text, url, pdf_url
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for data loading. "
            "Install it with: pip install 'cat-pol[sources]'"
        )

    ds = load_dataset(repo_id, split="train")
    df = ds.to_pandas()

    # Parse dates
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter by congress
    if congress is not None:
        df = df[df["congress"].astype(str) == str(congress)]

    # Filter by date range
    if since:
        since_dt = pd.to_datetime(since)
        df = df[df["_date"] >= since_dt]
    if until:
        until_dt = pd.to_datetime(until)
        df = df[df["_date"] <= until_dt]

    # Sort by date descending (most recent first)
    df = df.sort_values("_date", ascending=False).reset_index(drop=True)

    # Limit
    if n is not None:
        df = df.head(n)

    df = df.drop(columns=["_date"])
    return df
