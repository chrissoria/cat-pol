# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .__about__ import __version__
from .classify import classify
from .extract import extract
from .explore import explore
from .prompt_tune import prompt_tune
from .summarize import summarize
from ._source_registry import list_sources, fetch_source
from . import sources

# Semantic consolidation for explore() output, re-exported from the shared
# cat-stack engine so the discovery workflow (explore -> collapse_themes) is
# complete without a second import. Its prompts are self-contained (not
# domain-keyed), so no policy wrapping is needed.
from catstack import collapse_themes

__all__ = [
    "classify",
    "collapse_themes",
    "extract",
    "explore",
    "fetch_source",
    "list_sources",
    "prompt_tune",
    "sources",
    "summarize",
]
