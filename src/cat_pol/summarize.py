# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import cat_stack

from ._source_registry import fetch_source, SOURCES

# Policy-specific tone presets
_TONES = {
    "eli5": (
        "Use plain language that a non-expert member of the public could understand. "
        "Avoid legal jargon, technical terms, and bureaucratic language. Explain what "
        "this means in practical, everyday terms — who is affected and how it changes "
        "their daily life."
    ),
    "legal": (
        "Use legal precision. Preserve exact statutory references, section numbers, "
        "and legal terms of art. Note the specific code sections being amended, added, "
        "or repealed. Include effective dates, enforcement mechanisms, and any conditions "
        "or exceptions. Use formal legislative language."
    ),
}


def summarize(
    input_data=None,
    source: str = None,
    doc_type: str = None,
    since: str = None,
    until: str = None,
    n: int = None,
    format: str = "paragraph",
    tone: str = "eli5",
    **kwargs,
):
    """Summarize policy documents using LLMs.

    Thin wrapper around cat_stack.summarize() that supports pulling data
    from a registered political data source and adds a policy-specific
    tone parameter.

    Parameters
    ----------
    input_data : list[str], str, or PDF directory, optional
        Policy document text or documents to summarize.
        If not provided, `source` must be set.
    source : str, optional
        Pull data from a registered source. Use cat_pol.list_sources() for options.
    doc_type : str, optional
        Filter source by document type.
    since : str, optional
        Only include rows on or after this date (YYYY-MM-DD).
    until : str, optional
        Only include rows on or before this date (YYYY-MM-DD).
    n : int, optional
        Maximum number of source rows to summarize.
    format : str, optional
        Output structure. Default "paragraph".
        Options: "paragraph", "bullets", "one-liner", "structured", "report"
    tone : str, optional
        Writing style for the summary. Can be combined with any format.
        Default "eli5" (plain language).
            - "eli5" (default): Plain language, no jargon — explains what
              this means for everyday people in practical terms.
            - "legal": Formal language preserving legal precision — exact
              section references, effective dates, enforcement mechanisms.
            - None: Neutral/standard tone (no tone instruction added).
    **kwargs
        All other arguments are passed through to cat_stack.summarize().

    Returns
    -------
    pd.DataFrame
        Summarization results.

    Examples
    --------
    >>> # Legal-precision bullet points
    >>> pol.summarize(source="city_san_diego", n=5, format="bullets", tone="legal", api_key="sk-...")
    >>>
    >>> # Plain-language full report
    >>> pol.summarize(source="federal_laws", n=3, format="report", tone="eli5", api_key="sk-...")
    """
    if source is not None:
        source_df = fetch_source(source=source, n=n, since=since, until=until, doc_type=doc_type)
        if source_df.empty:
            print(f"[cat-pol] No data found for source '{source}'.")
            import pandas as pd
            return pd.DataFrame()
        print(f"[cat-pol] Fetched {len(source_df)} rows from '{source}'")
        input_data = source_df["text"].tolist()

        # Auto-set description from source metadata if not provided
        if "description" not in kwargs:
            cfg = SOURCES[source]
            kwargs["description"] = f"{cfg['jurisdiction']} — {', '.join(cfg['doc_types'])}"

    elif input_data is None:
        raise ValueError("Either input_data or source must be provided.")

    # Apply tone as additional instructions
    if tone is not None:
        tone_lower = tone.lower()
        if tone_lower not in _TONES:
            valid = ", ".join(f'"{k}"' for k in _TONES)
            raise ValueError(f"tone must be one of {valid}, got '{tone}'")

        tone_instruction = _TONES[tone_lower]
        user_instructions = kwargs.get("instructions", "")
        if user_instructions:
            kwargs["instructions"] = f"{tone_instruction}\n\n{user_instructions}"
        else:
            kwargs["instructions"] = tone_instruction

    return cat_stack.summarize(input_data, format=format, **kwargs)
