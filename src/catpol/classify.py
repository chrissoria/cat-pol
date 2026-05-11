# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import catstack

from ._utils import build_policy_description
from ._source_registry import fetch_source, list_sources


def classify(
    input_data=None,
    categories=None,
    source: str = None,
    doc_type: str = None,
    since: str = None,
    until: str = None,
    n: int = None,
    document_context="",
    description="",
    add_other="prompt",
    check_verbosity=True,
    **kwargs,
):
    """Classify policy documents into categories using LLMs.

    Can classify either raw text (via input_data) or pull directly from a
    registered political data source (via source parameter).

    Parameters
    ----------
    input_data : list[str], str, or pd.Series, optional
        Text to classify. If not provided, `source` must be set.
    categories : list[str]
        Category names for classification.
    source : str, optional
        Pull data from a registered source instead of providing input_data.
        Examples: "city_san_diego", "federal_laws", "federal_executive_orders",
        "social_trump_truth". Use cat_pol.list_sources() to see all options.
    doc_type : str, optional
        Filter source by document type (e.g., "ordinance", "resolution").
        Only applies to sources with multiple doc types.
    since : str, optional
        Only include source rows on or after this date (YYYY-MM-DD).
    until : str, optional
        Only include source rows on or before this date (YYYY-MM-DD).
    n : int, optional
        Maximum number of source rows to classify.
    document_context : str
        Context about the policy document being analyzed.
    description : str
        Additional context about the document or analysis task.
    add_other : str or bool
        Whether to add an "Other" category. Default "prompt".
    check_verbosity : bool
        Whether to check category verbosity. Default True.
    **kwargs
        All other arguments are passed through to catstack.classify().

    Returns
    -------
    pd.DataFrame
        Classification results. When using `source`, the result includes
        all original source columns plus classification columns.

    Examples
    --------
    >>> import cat_pol as pol
    >>>
    >>> # Classify from a source
    >>> results = pol.classify(
    ...     source="city_san_diego",
    ...     categories=["Housing", "Public Safety", "Finance"],
    ...     since="2020-01-01",
    ...     n=50,
    ...     api_key="sk-...",
    ... )
    >>>
    >>> # Classify raw text (same as before)
    >>> results = pol.classify(
    ...     input_data=["The city shall zone area X for residential use"],
    ...     categories=["Zoning", "Public Safety"],
    ...     api_key="sk-...",
    ... )
    >>>
    >>> # Classify Trump's Truth Social posts
    >>> results = pol.classify(
    ...     source="social_trump_truth",
    ...     categories=["Economy", "Immigration", "Foreign Policy"],
    ...     since="2024-01-01",
    ...     n=100,
    ...     api_key="sk-...",
    ... )
    """
    import pandas as pd

    source_df = None

    if source is not None:
        # Fetch from registered source
        source_df = fetch_source(
            source=source,
            n=n,
            since=since,
            until=until,
            doc_type=doc_type,
        )

        if source_df.empty:
            print(f"[cat-pol] No data found for source '{source}' with the given filters.")
            return pd.DataFrame()

        print(f"[cat-pol] Fetched {len(source_df)} rows from '{source}'")
        input_data = source_df["text"].tolist()

        # Auto-generate description from source metadata if not provided
        if not document_context and not description:
            from ._source_registry import SOURCES
            cfg = SOURCES[source]
            document_context = (
                f"{cfg['jurisdiction']} — {', '.join(cfg['doc_types'])}"
            )

    elif input_data is None:
        raise ValueError(
            "Either input_data or source must be provided. "
            "Use source='city_san_diego' to pull from a registered source, "
            "or pass input_data directly."
        )

    desc = build_policy_description(document_context, description)
    result = catstack.classify(
        input_data,
        categories,
        description=desc,
        add_other=add_other,
        check_verbosity=check_verbosity,
        **kwargs,
    )

    # If we pulled from a source, join classification results back to source metadata
    if source_df is not None and result is not None:
        # The result has input_data as first column — join source columns
        source_cols = [c for c in source_df.columns if c != "text"]
        for col in source_cols:
            if col not in result.columns:
                result[col] = source_df[col].values[:len(result)]

    return result
