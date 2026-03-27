"""San Diego County ordinances data loader.

Loads SD County ordinances and resolutions from HuggingFace,
sourced from the SD County Legistar API.
"""

from __future__ import annotations

import pandas as pd


_DEFAULT_REPO = "chrissoria/sd-county-ordinances"


def fetch_sd_county_ordinances(
    n: int = None,
    doc_type: str = None,
    since: str = None,
    until: str = None,
    repo_id: str = _DEFAULT_REPO,
) -> pd.DataFrame:
    """
    Fetch San Diego County ordinances/resolutions from HuggingFace.

    Args:
        n: Maximum number of rows to return (most recent first).
        doc_type: Filter by "ordinance" or "resolution".
        since: Only include rows on or after this date (YYYY-MM-DD).
        until: Only include rows on or before this date (YYYY-MM-DD).
        repo_id: HuggingFace dataset repository ID.

    Returns:
        DataFrame with columns: date, matter_id, matter_file, doc_type,
        title, text, attachment_url, body, status, year
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

    if doc_type:
        df = df[df["doc_type"].str.lower() == doc_type.lower()]

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")

    if since:
        df = df[df["_date"] >= pd.to_datetime(since)]
    if until:
        df = df[df["_date"] <= pd.to_datetime(until)]

    df = df.sort_values("_date", ascending=False).reset_index(drop=True)

    if n is not None:
        df = df.head(n)

    df = df.drop(columns=["_date"])
    return df
