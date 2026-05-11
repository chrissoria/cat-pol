# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import tempfile

import requests
import catstack

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


# Structured bill analysis format
_BILL_ANALYSIS_FIELDS = [
    "policy_domain",
    "what_it_does",
    "who_benefits",
    "who_bears_cost",
    "dollar_amounts",
    "framing_gap",
]

_BILL_ANALYSIS_PROMPT = (
    "Do NOT write a prose summary. Instead, extract structured policy data and respond "
    "with ONLY the following JSON (no markdown, no extra text):\n\n"
    '{"summary": "{\\\"policy_domain\\\": \\\"FILL\\\", '
    '\\\"what_it_does\\\": \\\"FILL\\\", '
    '\\\"who_benefits\\\": \\\"FILL\\\", '
    '\\\"who_bears_cost\\\": \\\"FILL\\\", '
    '\\\"dollar_amounts\\\": null, '
    '\\\"framing_gap\\\": null}"}\n\n'
    "Replace each FILL with a value drawn strictly from the document text:\n"
    "  policy_domain — primary policy area (e.g. Healthcare, Defense, Tax, Immigration)\n"
    "  what_it_does  — one sentence: the concrete action this bill takes\n"
    "  who_benefits  — who gains from this bill\n"
    "  who_bears_cost — who pays, financially or regulatorily\n"
    "  dollar_amounts — explicit dollar figures or appropriations (null if none)\n"
    "  framing_gap    — gap between stated title/purpose and actual effect (null if none)\n"
    "Do not editorialize."
)


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

    Thin wrapper around catstack.summarize() that supports pulling data
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
        Options: "paragraph", "bullets", "one-liner", "structured", "report", "threads",
        "bill_analysis" (returns a DataFrame with 6 named columns: policy_domain,
        what_it_does, who_benefits, who_bears_cost, dollar_amounts, framing_gap).
    tone : str, optional
        Writing style for the summary. Can be combined with any format.
        Default "eli5" (plain language).
            - "eli5" (default): Plain language, no jargon — explains what
              this means for everyday people in practical terms.
            - "legal": Formal language preserving legal precision — exact
              section references, effective dates, enforcement mechanisms.
            - None: Neutral/standard tone (no tone instruction added).
    **kwargs
        All other arguments are passed through to catstack.summarize().
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

    # Intercept bill_analysis format before passing to cat-stack
    _bill_analysis_mode = isinstance(format, str) and format.lower() == "bill_analysis"
    if _bill_analysis_mode:
        # Use "raw" format so cat-stack adds NO preset instruction — our JSON schema runs solo
        format = "raw"
        tone = None   # JSON extraction conflicts with eli5/legal prose framing
        existing = kwargs.get("instructions", "")
        kwargs["instructions"] = (
            _BILL_ANALYSIS_PROMPT + ("\n\n" + existing if existing else "")
        )
        kwargs.pop("description", None)   # description would frame this as "context", not instruction
        kwargs.setdefault("creativity", 0)

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
        result = catstack.summarize(input_data, format=format, **kwargs)
    finally:
        if _tmp_dir is not None:
            import shutil
            shutil.rmtree(_tmp_dir, ignore_errors=True)

    if _bill_analysis_mode and result is not None:
        import json
        import re

        def _find_json_objects(s):
            """Scan string for top-level JSON objects using bracket matching."""
            objects = []
            i = 0
            while i < len(s):
                if s[i] != '{':
                    i += 1
                    continue
                depth, in_str, escape, start = 0, False, False, i
                for j in range(i, len(s)):
                    c = s[j]
                    if escape:
                        escape = False
                        continue
                    if c == '\\' and in_str:
                        escape = True
                        continue
                    if c == '"' and not escape:
                        in_str = not in_str
                        continue
                    if not in_str:
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                objects.append(s[start:j + 1])
                                i = j + 1
                                break
                else:
                    break
            return objects

        def _parse_bill_json(raw):
            if not isinstance(raw, str):
                return {f: None for f in _BILL_ANALYSIS_FIELDS}
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
            # Try direct parse first
            try:
                parsed = json.loads(cleaned)
                if (
                    isinstance(parsed, dict)
                    and "summary" in parsed
                    and isinstance(parsed["summary"], str)
                ):
                    nested = json.loads(parsed["summary"])
                    if isinstance(nested, dict) and any(f in nested for f in _BILL_ANALYSIS_FIELDS):
                        return {f: nested.get(f) for f in _BILL_ANALYSIS_FIELDS}
                if isinstance(parsed, dict) and any(f in parsed for f in _BILL_ANALYSIS_FIELDS):
                    return {f: parsed.get(f) for f in _BILL_ANALYSIS_FIELDS}
            except (json.JSONDecodeError, ValueError):
                pass
            # Scan for embedded JSON objects containing our keys (cat-stack style)
            for candidate in _find_json_objects(cleaned):
                try:
                    parsed = json.loads(candidate)
                    if (
                        isinstance(parsed, dict)
                        and "summary" in parsed
                        and isinstance(parsed["summary"], str)
                    ):
                        nested = json.loads(parsed["summary"])
                        if isinstance(nested, dict) and any(f in nested for f in _BILL_ANALYSIS_FIELDS):
                            return {f: nested.get(f) for f in _BILL_ANALYSIS_FIELDS}
                    if isinstance(parsed, dict) and any(f in parsed for f in _BILL_ANALYSIS_FIELDS):
                        return {f: parsed.get(f) for f in _BILL_ANALYSIS_FIELDS}
                except (json.JSONDecodeError, ValueError):
                    continue

            # Last resort: salvage values from malformed near-JSON output.
            recovered = {}
            for i, field in enumerate(_BILL_ANALYSIS_FIELDS):
                remaining = _BILL_ANALYSIS_FIELDS[i + 1 :]
                if remaining:
                    next_keys = "|".join(re.escape(name) for name in remaining)
                    boundary = rf',\s*"(?:{next_keys})"\s*:|\s*\}}|$'
                else:
                    boundary = r"\s*\}|$"

                pattern = rf'"{re.escape(field)}"\s*:\s*(.*?)(?={boundary})'
                match = re.search(pattern, cleaned, flags=re.DOTALL)
                if not match:
                    recovered[field] = None
                    continue

                value = match.group(1).strip().rstrip(",").strip()
                if value in {"null", "None"}:
                    recovered[field] = None
                elif len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                    try:
                        recovered[field] = json.loads(value)
                    except json.JSONDecodeError:
                        recovered[field] = value[1:-1]
                else:
                    recovered[field] = value

            if any(v is not None for v in recovered.values()):
                return recovered

            return {f: None for f in _BILL_ANALYSIS_FIELDS}

        parsed_rows = result["summary"].apply(_parse_bill_json)
        import pandas as pd
        expanded = pd.DataFrame(parsed_rows.tolist(), index=result.index)
        if expanded.isna().all(axis=1).any():
            expanded["summary_raw"] = result["summary"]
        result = pd.concat(
            [result.drop(columns=["summary"]), expanded], axis=1
        )

    return result
