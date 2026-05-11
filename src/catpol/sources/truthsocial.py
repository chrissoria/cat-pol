"""Trump Truth Social posts data loader.

Loads Trump's Truth Social posts from HuggingFace,
sourced from the CNN/Stiles public archive.
"""

from __future__ import annotations

import pandas as pd


_DEFAULT_REPO = "chrissoria/trump-truth-social"


def fetch_trump_truths(
    n: int = None,
    since: str = None,
    until: str = None,
    repo_id: str = _DEFAULT_REPO,
) -> pd.DataFrame:
    """
    Fetch Trump's Truth Social posts from HuggingFace.

    Args:
        n: Maximum number of posts to return (most recent first).
        since: Only include posts on or after this date (YYYY-MM-DD).
        until: Only include posts on or before this date (YYYY-MM-DD).
        repo_id: HuggingFace dataset repository ID.

    Returns:
        DataFrame with columns: date, datetime, text, content_html,
        post_id, url, replies_count, reblogs_count, favourites_count,
        media_urls, links, has_media
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

    # Filter by date range
    if since:
        since_dt = pd.to_datetime(since)
        df = df[df["_date"] >= since_dt]
    if until:
        until_dt = pd.to_datetime(until)
        df = df[df["_date"] <= until_dt]

    # Sort by datetime descending (most recent first)
    df = df.sort_values("datetime", ascending=False).reset_index(drop=True)

    # Limit
    if n is not None:
        df = df.head(n)

    df = df.drop(columns=["_date"])
    return df
