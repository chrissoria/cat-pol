# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import cat_stack

from ._utils import build_policy_description


def classify(
    input_data,
    categories,
    document_context="",
    description="",
    add_other="prompt",
    check_verbosity=True,
    **kwargs,
):
    """Classify policy document excerpts into categories using LLMs.

    Thin wrapper around cat_stack.classify() that injects policy-document-specific
    prompt framing into the description.

    Parameters
    ----------
    input_data : list[str] or str
        Policy document text to classify (list of text, image dir, or PDF dir).
    categories : list[str]
        Category names for classification.
    document_context : str
        Context about the policy document being analyzed.
    description : str
        Additional context about the document or analysis task.
    add_other : str or bool
        Whether to add an "Other" category. Default "prompt".
    check_verbosity : bool
        Whether to check category verbosity. Default True.
    **kwargs
        All other arguments are passed through to cat_stack.classify().

    Returns
    -------
    pd.DataFrame
        Classification results.
    """
    desc = build_policy_description(document_context, description)
    return cat_stack.classify(
        input_data,
        categories,
        description=desc,
        add_other=add_other,
        check_verbosity=check_verbosity,
        **kwargs,
    )
