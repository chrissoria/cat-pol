# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def build_policy_description(document_context, description=""):
    """Build a policy-document-framed description string.

    Combines the document context and optional description into a single
    string with policy-document-specific framing.

    Parameters
    ----------
    document_context : str
        Context about the policy document being analyzed.
    description : str
        Additional context about the document or analysis task.

    Returns
    -------
    str
        Combined description with policy-document framing.
    """
    parts = []
    if document_context:
        parts.append(
            f"The following is an excerpt from a policy document. Context: {document_context}."
        )
    if description:
        parts.append(description)
    return " ".join(parts)
