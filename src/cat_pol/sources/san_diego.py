# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fetch San Diego city ordinances/resolutions from HuggingFace."""

from __future__ import annotations

import pandas as pd


def fetch_san_diego_ordinances(
    n: int | None = None,
    doc_type: str | None = None,
    since: str | None = None,
    until: str | None = None,
    repo_id: str = "chrissoria/san-diego-ordinances",
) -> pd.DataFrame:
    """Load San Diego ordinances/resolutions from HuggingFace and return a filtered DataFrame.

    Parameters
    ----------
    n : int, optional
        Maximum number of rows to return.
    doc_type : str, optional
        Filter by document type: ``"ordinance"`` or ``"resolution"``.
        ``None`` returns both.
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

    if doc_type is not None:
        doc_type_lower = doc_type.lower()
        if doc_type_lower not in ("ordinance", "resolution"):
            raise ValueError(
                f"doc_type must be 'ordinance' or 'resolution', got {doc_type!r}"
            )
        df = df[df["doc_type"] == doc_type_lower]

    if since is not None:
        df = df[df["date"] >= since]

    if until is not None:
        df = df[df["date"] <= until]

    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    if n is not None:
        df = df.head(n)

    return df
