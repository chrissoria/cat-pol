"""Code Publishing municipal code data loaders.

Loads codified municipal code sections from HuggingFace for cities
that use Code Publishing (codepublishing.com).
"""

from __future__ import annotations

import pandas as pd


def _fetch_codepublishing_dataset(
    repo_id: str,
    n: int = None,
    title_num: int = None,
) -> pd.DataFrame:
    """Shared loader for Code Publishing datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for data loading. "
            "Install it with: pip install 'cat-pol[sources]'"
        )

    ds = load_dataset(repo_id, split="train")
    df = ds.to_pandas()

    if title_num is not None:
        df = df[df["title_num"].astype(str) == str(title_num)]

    df = df.reset_index(drop=True)

    if n is not None:
        df = df.head(n)

    return df


def fetch_clovis_code(
    n: int = None,
    title_num: int = None,
    repo_id: str = "chrissoria/clovis-ordinances",
) -> pd.DataFrame:
    """
    Fetch Clovis municipal code sections from HuggingFace.

    Args:
        n: Maximum number of sections to return.
        title_num: Only include sections from this title number.
        repo_id: HuggingFace dataset repository ID.
    """
    return _fetch_codepublishing_dataset(repo_id, n=n, title_num=title_num)


def fetch_newport_beach_code(
    n: int = None,
    title_num: int = None,
    repo_id: str = "chrissoria/newport-beach-ordinances",
) -> pd.DataFrame:
    """
    Fetch Newport Beach municipal code sections from HuggingFace.

    Args:
        n: Maximum number of sections to return.
        title_num: Only include sections from this title number.
        repo_id: HuggingFace dataset repository ID.
    """
    return _fetch_codepublishing_dataset(repo_id, n=n, title_num=title_num)
