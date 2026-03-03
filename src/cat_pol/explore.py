# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import cat_stack

from ._utils import build_policy_description


def explore(
    input_data,
    api_key,
    document_context="",
    description="",
    **kwargs,
):
    """Explore raw categories from policy document excerpts (no deduplication).

    Thin wrapper around cat_stack.explore() that injects policy-document-specific
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
        All other arguments are passed through to cat_stack.explore().

    Returns
    -------
    list[str]
        Raw list of categories (with duplicates across iterations).
    """
    desc = build_policy_description(document_context, description)
    return cat_stack.explore(
        input_data,
        api_key,
        description=desc,
        **kwargs,
    )
