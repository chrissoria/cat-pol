# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cat-pol is a Python package for political text classification and analysis powered by LLMs. It wraps [cat-stack](https://github.com/chrissoria/cat-stack) with policy-specific prompt framing and provides access to 15 public political datasets on HuggingFace (no auth required).

## Build & Install

```bash
# Install in development mode
pip install -e .

# With optional extras
pip install -e ".[sources]"      # datasets + huggingface_hub for source fetching
pip install -e ".[pdf]"          # PDF support via cat-stack
pip install -e ".[embeddings]"   # Embedding support via cat-stack
```

Build system: Hatchling (configured in `pyproject.toml`). Core deps: `cat-stack>=0.1.0`, `pandas`.

## Testing

Tests in `tests/` are **integration tests** that hit real HuggingFace data and LLM APIs — they are not unit tests and require network access + API keys.

```bash
python tests/run_sd_sf_classify_200.py
python tests/run_sd_sf_political_lean_200.py
```

## Architecture

### Core API (5 entry points in `src/cat_pol/`)

All functions accept either `input_data=` (raw text/DataFrame) or `source=` (HuggingFace dataset name). When using `source=`, jurisdiction and doc_type metadata auto-populate the LLM prompt description.

- **`classify()`** — Classify documents into user-defined categories. Wraps `cat_stack.classify()` with policy framing.
- **`extract()`** — Discover and normalize categories from documents (with deduplication). Returns dict with `top_categories`, `counts_df`, `raw_top_text`.
- **`explore()`** — Raw category extraction without deduplication.
- **`summarize()`** — Summarize documents with format (`paragraph`, `bullets`, `one-liner`, `structured`, `report`) and tone (`eli5`, `legal`, or None) control.
- **`prompt_tune()`** — Optimize classification prompts via iterative user feedback.

All functions pass `**kwargs` through to their cat-stack counterparts.

### Source Registry (`_source_registry.py`)

Central `SOURCES` dict maps source identifiers to HuggingFace repo metadata (repo, text_col, date_col, doc_types, jurisdiction, level). Naming convention: `{level}_{jurisdiction}` (e.g., `city_san_diego`, `federal_laws`).

Two public functions: `list_sources(level=None)` and `fetch_source(source, ...)`.

Fetching uses `datasets` library when available, falls back to HuggingFace REST API with retry logic.

### Source Modules (`src/cat_pol/sources/`)

Per-source fetch functions for direct access. Each module follows the same pattern: loads via `datasets.load_dataset()`, supports `n`, `since`, `until`, optional `doc_type`, sorts by date descending, returns a pandas DataFrame.

### Adding a New Source

1. Add entry to `SOURCES` dict in `_source_registry.py`
2. Create source fetch function in `src/cat_pol/sources/`
3. Export from `sources/__init__.py`

### Policy Prompt Framing (`_utils.py`)

`build_policy_description()` combines `document_context` and `description` into a policy-document-framed prompt string used by all core functions.

## Data Pipeline

`scripts/` contains scrapers that build and maintain the 15 HuggingFace datasets:
- `update_datasets.py` — Weekly incremental update runner (launchd, Sundays 9 AM)
- `build_*.py` — One-time full-build scripts per source
- Scrapers use caches (`.{source}_cache.parquet`), checkpoints (`scripts/checkpoints/`), and logs (`scripts/logs/`)

## License

GPL-3.0-or-later. All files carry SPDX headers:
```
# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
# SPDX-License-Identifier: GPL-3.0-or-later
```
