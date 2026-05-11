# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import catstack

from ._utils import build_policy_description
from ._source_registry import fetch_source, SOURCES


def extract(
    input_data=None,
    api_key=None,
    source: str = None,
    doc_type: str = None,
    since: str = None,
    until: str = None,
    n: int = None,
    document_context="",
    description="",
    **kwargs,
):
    """Discover categories from policy documents using LLMs.

    Thin wrapper around catstack.extract() that supports pulling data
    from a registered political data source and injects policy-document-specific
    prompt framing.

    Parameters
    ----------
    input_data : list[str] or str, optional
        Policy document text to analyze. If not provided, `source` must be set.
    api_key : str
        API key for the LLM provider.
    source : str, optional
        Pull data from a registered source. Use cat_pol.list_sources() for options.
    doc_type : str, optional
        Filter source by document type.
    since : str, optional
        Only include rows on or after this date (YYYY-MM-DD).
    until : str, optional
        Only include rows on or before this date (YYYY-MM-DD).
    n : int, optional
        Maximum number of source rows to use.
    document_context : str
        Context about the policy document being analyzed.
    description : str
        Additional context about the document or analysis task.
    **kwargs
        All other arguments are passed through to catstack.extract().

    Returns
    -------
    dict
        Dictionary with 'counts_df', 'top_categories', and 'raw_top_text'.
    """
    if source is not None:
        source_df = fetch_source(source=source, n=n, since=since, until=until, doc_type=doc_type)
        if source_df.empty:
            print(f"[cat-pol] No data found for source '{source}'.")
            return None
        print(f"[cat-pol] Fetched {len(source_df)} rows from '{source}'")
        input_data = source_df["text"].tolist()
        if not document_context and not description:
            cfg = SOURCES[source]
            document_context = f"{cfg['jurisdiction']} — {', '.join(cfg['doc_types'])}"
    elif input_data is None:
        raise ValueError("Either input_data or source must be provided.")

    desc = build_policy_description(document_context, description)
    return catstack.extract(
        input_data,
        api_key,
        survey_question=desc,
        **kwargs,
    )
