# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import tempfile

import requests
import cat_stack

from ._source_registry import fetch_source, SOURCES


def _is_url(s: str) -> bool:
    """Check if a string is a URL."""
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def _guess_extension(url: str, content_type: str = "") -> str:
    """Guess file extension from URL path or Content-Type header."""
    # Try URL path first
    path = url.split("?")[0].split("#")[0]
    _, ext = os.path.splitext(path)
    if ext:
        return ext.lower()

    # Fall back to Content-Type
    ct = content_type.lower()
    _CT_MAP = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "image/svg+xml": ".svg", "image/tiff": ".tiff",
        "image/bmp": ".bmp", "application/pdf": ".pdf",
        "text/html": ".html", "text/plain": ".txt",
    }
    for mime, ext in _CT_MAP.items():
        if mime in ct:
            return ext
    return ""


def _download_urls(urls: list[str]) -> tuple[str, list[str]]:
    """Download URLs to a temp directory. Returns (temp_dir, file_paths)."""
    tmp_dir = tempfile.mkdtemp(prefix="catpol_dl_")
    paths = []
    for i, url in enumerate(urls):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            ext = _guess_extension(url, resp.headers.get("Content-Type", ""))
            dest = os.path.join(tmp_dir, f"file_{i:04d}{ext}")
            with open(dest, "wb") as f:
                f.write(resp.content)
            paths.append(dest)
        except Exception as e:
            print(f"[cat-pol] Failed to download {url}: {e}")
    return tmp_dir, paths


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
    input_data : list[str], str, PDF directory, or URLs, optional
        Policy document text, file paths, or URLs to summarize.
        URLs (http/https) are automatically downloaded and passed to
        the LLM. Use ``input_mode="visual"`` for image URLs or
        ``input_mode="text"`` for text extraction from images.
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
        Notable: input_mode ("visual" or "text"), input_type, user_model, api_key.

    Returns
    -------
    pd.DataFrame
        Summarization results.

    Examples
    --------
    >>> # Legal-precision bullet points
    >>> pol.summarize(source="city_san_diego", n=5, format="bullets", tone="legal", api_key="sk-...")
    >>>
    >>> # Summarize images from URLs
    >>> pol.summarize(["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
    ...              format="bullets", tone=None, api_key="sk-...")
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

    # Detect and download URLs
    _tmp_dir = None
    if isinstance(input_data, list) and input_data and _is_url(input_data[0]):
        urls = [u for u in input_data if _is_url(u)]
        print(f"[cat-pol] Downloading {len(urls)} files...")
        _tmp_dir, file_paths = _download_urls(urls)
        if not file_paths:
            raise ValueError("Failed to download any files from the provided URLs.")
        print(f"[cat-pol] Downloaded {len(file_paths)} files")
        input_data = file_paths
    elif isinstance(input_data, str) and _is_url(input_data):
        print("[cat-pol] Downloading 1 file...")
        _tmp_dir, file_paths = _download_urls([input_data])
        if not file_paths:
            raise ValueError("Failed to download the file from the provided URL.")
        input_data = file_paths[0]

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

    try:
        return cat_stack.summarize(input_data, format=format, **kwargs)
    finally:
        if _tmp_dir is not None:
            import shutil
            shutil.rmtree(_tmp_dir, ignore_errors=True)
