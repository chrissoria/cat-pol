# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import cat_stack

from ._utils import build_policy_description


def extract(
    input_data,
    api_key,
    document_context="",
    description="",
    **kwargs,
):
    """Discover categories from policy document excerpts using LLMs.

    Thin wrapper around cat_stack.extract() that injects policy-document-specific
    prompt framing into the description.

    Parameters
    ----------
    input_data : list[str] or str
        Policy document text to analyze.
    api_key : str
        API key for the LLM provider.
    document_context : str
        Context about the policy document being analyzed.
    description : str
        Additional context about the document or analysis task.
    **kwargs
        All other arguments are passed through to cat_stack.extract().

    Returns
    -------
    dict
        Dictionary with 'counts_df', 'top_categories', and 'raw_top_text'.
    """
    desc = build_policy_description(document_context, description)
    return cat_stack.extract(
        input_data,
        api_key,
        survey_question=desc,
        **kwargs,
    )
