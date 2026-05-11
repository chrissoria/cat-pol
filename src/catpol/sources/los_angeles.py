"""Los Angeles ordinances data loader.

Loads LA city ordinances from HuggingFace,
sourced from the LA City Clerk Connect system.
"""

from __future__ import annotations

import pandas as pd


_DEFAULT_REPO = "chrissoria/la-ordinances"


def fetch_la_ordinances(
    n: int = None,
    since: str = None,
    until: str = None,
    repo_id: str = _DEFAULT_REPO,
) -> pd.DataFrame:
    """
    Fetch Los Angeles ordinances from HuggingFace.

    Args:
        n: Maximum number of rows to return (most recent first).
        since: Only include ordinances on or after this date (YYYY-MM-DD).
        until: Only include ordinances on or before this date (YYYY-MM-DD).
        repo_id: HuggingFace dataset repository ID.

    Returns:
        DataFrame with columns: date, ordinance_number, council_file,
        doc_type, title, text, pdf_url, council_file_url
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
