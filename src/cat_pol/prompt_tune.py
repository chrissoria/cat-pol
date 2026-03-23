# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import cat_stack

from ._utils import build_policy_description
from ._source_registry import fetch_source, SOURCES


def prompt_tune(
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
    **kwargs,
):
    """Optimize classification prompts for policy documents using user feedback.

    Thin wrapper around cat_stack.prompt_tune() that supports pulling data
    directly from a registered political data source and injects
    policy-document-specific prompt framing.

    Parameters
    ----------
    input_data : list[str], str, or pd.Series, optional
        Text to tune on. If not provided, `source` must be set.
    categories : list[str]
        Category names for classification.
    source : str, optional
        Pull data from a registered source instead of providing input_data.
        Examples: "city_san_diego", "federal_executive_orders".
        Use cat_pol.list_sources() to see all options.
    doc_type : str, optional
        Filter source by document type (e.g., "ordinance", "resolution").
    since : str, optional
        Only include source rows on or after this date (YYYY-MM-DD).
    until : str, optional
        Only include source rows on or before this date (YYYY-MM-DD).
    n : int, optional
        Maximum number of source rows to use for tuning.
    document_context : str
        Context about the policy document being analyzed.
    description : str
        Additional context about the document or analysis task.
    add_other : str or bool
        Whether to add an "Other" category. Default "prompt".
    **kwargs
        All other arguments passed through to cat_stack.prompt_tune()
        (api_key, user_model, model_source, models, sample_size,
        max_iterations, multi_label, creativity, ui, optimize, etc.)

    Returns
    -------
    dict
        - "system_prompt": str — the optimized prompt
        - "iterations": list of dicts with metrics per iteration
        - "per_category_summary": dict — per-category metrics from best run

    Examples
    --------
    >>> import cat_pol as pol
    >>>
    >>> # Tune on San Diego ordinances
    >>> result = pol.prompt_tune(
    ...     source="city_san_diego",
    ...     categories=["Housing", "Public Safety", "Finance"],
    ...     doc_type="ordinance",
    ...     since="2020-01-01",
    ...     n=100,
    ...     api_key="sk-...",
    ...     sample_size=10,
    ... )
    >>> print(result["system_prompt"])
    >>>
    >>> # Then classify with the optimized prompt
    >>> results = pol.classify(
    ...     source="city_san_diego",
    ...     categories=["Housing", "Public Safety", "Finance"],
    ...     system_prompt=result["system_prompt"],
    ...     api_key="sk-...",
    ... )
    """
    import pandas as pd

    if source is not None:
        source_df = fetch_source(
            source=source,
            n=n,
            since=since,
            until=until,
            doc_type=doc_type,
        )

        if source_df.empty:
            print(f"[cat-pol] No data found for source '{source}' with the given filters.")
            return None

        print(f"[cat-pol] Fetched {len(source_df)} rows from '{source}'")
        input_data = source_df["text"].tolist()

        if not document_context and not description:
            cfg = SOURCES[source]
            document_context = (
                f"{cfg['jurisdiction']} — {', '.join(cfg['doc_types'])}"
            )

    elif input_data is None:
        raise ValueError(
            "Either input_data or source must be provided."
        )

    desc = build_policy_description(document_context, description)

    return cat_stack.prompt_tune(
        input_data,
        categories,
        description=desc,
        add_other=add_other,
        **kwargs,
    )
