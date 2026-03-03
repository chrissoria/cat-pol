# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import cat_stack


def summarize(input_data, **kwargs):
    """Summarize policy documents using LLMs.

    Pure pass-through to cat_stack.summarize().

    Parameters
    ----------
    input_data : list[str], str, or PDF directory
        Policy document text or documents to summarize.
    **kwargs
        All arguments are passed through to cat_stack.summarize().

    Returns
    -------
    pd.DataFrame
        Summarization results.
    """
    return cat_stack.summarize(input_data, **kwargs)
