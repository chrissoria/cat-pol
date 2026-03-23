# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fetch ordinances/resolutions from Legistar-based city datasets on HuggingFace."""

from __future__ import annotations

import pandas as pd


def _fetch_legistar_dataset(
    repo_id: str,
    n: int | None = None,
    doc_type: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> pd.DataFrame:
    """Shared loader for Legistar-sourced city datasets."""
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


def fetch_oakland_ordinances(
    n: int | None = None,
    since: str | None = None,
    until: str | None = None,
    repo_id: str = "chrissoria/oakland-ordinances",
) -> pd.DataFrame:
    """Load Oakland ordinances from HuggingFace.

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
    return _fetch_legistar_dataset(repo_id, n=n, since=since, until=until)


def fetch_long_beach_ordinances(
    n: int | None = None,
    doc_type: str | None = None,
    since: str | None = None,
    until: str | None = None,
    repo_id: str = "chrissoria/long-beach-ordinances",
) -> pd.DataFrame:
    """Load Long Beach ordinances/resolutions from HuggingFace.

    Parameters
    ----------
    n : int, optional
        Maximum number of rows to return.
    doc_type : str, optional
        Filter by ``"ordinance"`` or ``"resolution"``. ``None`` returns both.
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
    return _fetch_legistar_dataset(repo_id, n=n, doc_type=doc_type, since=since, until=until)


def fetch_fresno_ordinances(
    n: int | None = None,
    doc_type: str | None = None,
    since: str | None = None,
    until: str | None = None,
    repo_id: str = "chrissoria/fresno-ordinances",
) -> pd.DataFrame:
    """Load Fresno ordinances/resolutions from HuggingFace.

    Parameters
    ----------
    n : int, optional
        Maximum number of rows to return.
    doc_type : str, optional
        Filter by ``"ordinance"`` or ``"resolution"``. ``None`` returns both.
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
    return _fetch_legistar_dataset(repo_id, n=n, doc_type=doc_type, since=since, until=until)
