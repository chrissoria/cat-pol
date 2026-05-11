# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fetch San Francisco ordinances from HuggingFace."""

from __future__ import annotations

import pandas as pd


def fetch_sf_ordinances(
    n: int | None = None,
    since: str | None = None,
    until: str | None = None,
    repo_id: str = "chrissoria/sf-ordinances",
) -> pd.DataFrame:
    """Load San Francisco ordinances from HuggingFace and return a filtered DataFrame.

    Parameters
    ----------
    n : int, optional
        Maximum number of rows to return.
    since : str, optional
        Start date (inclusive) in ``YYYY-MM-DD`` format.
    until : str, optional
        End date (inclusive) in ``YYYY-MM-DD`` format.
    repo_id : str
        HuggingFace dataset repository ID.

    Returns
    -------
    pd.DataFrame
        Sorted by date descending.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for data sources. "
            "Install it with: pip install cat-pol[sources]"
        )

    ds = load_dataset(repo_id, split="train")
    df = ds.to_pandas()

    if since is not None:
        df = df[df["date"] >= since]

    if until is not None:
        df = df[df["date"] <= until]

    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    if n is not None:
        df = df.head(n)

    return df
