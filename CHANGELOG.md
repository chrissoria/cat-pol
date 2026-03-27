# Changelog

All notable changes to cat-pol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-27

### Added
- **San Diego County source** (`county_san_diego`): New county-level data source with ordinances and resolutions from `chrissoria/sd-county-ordinances`.
- **URL support in `summarize()`**: Pass image or document URLs directly — they are automatically downloaded, processed, and cleaned up. Works with `input_mode="visual"` for images.
- **California state bills scraper** (`update_california_bills.py`, `build_california_bills.py`): New data pipeline for California legislative bills.
- **SD County scraper** (`build_sdcounty_dataset.py`): One-time build script for San Diego County ordinances.
- **Ordinance classifier script** (`classify_all_ordinances.py`): Batch classification of ordinances across all city datasets.
- **Truth Social image summarizer** (`summarize_ts_images.py`): Standalone script for backfilling image alt-text on Truth Social posts.
- **Daily Truth Social launchd job**: Truth Social updates now run daily at 9 AM (separate from the weekly job for other sources).
- **California bills launchd job**: Weekly update on Sundays at 9:30 AM.
- **`--exclude` flag** on `update_datasets.py`: Skip specific sources (e.g., `--exclude ts`).

### Changed
- **Truth Social image model**: Switched from `qwen3-vl-235b-a22b-instruct:novita` to `qwen2.5-vl-72b-instruct:novita` — the 235B model was returning persistent 400 errors via the Novita router.
- **Truth Social ordinance classification model**: Same model swap as above.
- **Image/classification backfill scoped to post-election posts** (>= 2024-11-05) to avoid expensive backfills of the full 32k+ archive.
- **launchd installer** (`install_launchd.sh`): Now installs three jobs (weekly all-sources, daily Truth Social, weekly CA bills) instead of one.

### Fixed
- **datetime type mismatch** in stock ticker merges that caused all ticker data to fail on March 25-26 runs.

---

## [1.0.0] - 2026-03-22

### Notes
- First stable release. All core features tested and production-ready.

### Added
- **Source registry** (`_source_registry.py`): 15 political data sources on HuggingFace, accessible via `source=` parameter on all core functions. Sources include 11 California cities, federal public laws, executive orders, presidential speeches, and Trump Truth Social posts. All public, no auth required.
- **`source=` parameter** on `classify()`, `extract()`, `explore()`, `summarize()`, `prompt_tune()`: Pull data directly from registered HuggingFace datasets instead of providing `input_data`.
  - `doc_type=` filter (e.g., "ordinance", "resolution")
  - `since=` / `until=` date range filters
  - `n=` row limit
- **`list_sources()`**: List all available data sources, optionally filtered by `level=` ("city", "federal").
- **`fetch_source()`**: Fetch raw data from any registered source as a DataFrame.
- **`prompt_tune()`**: Thin wrapper around `cat_stack.prompt_tune()` with source integration and policy-document framing.
- **`tone=` parameter** on `summarize()`: Policy-specific writing style control.
  - `"eli5"` (default): Plain language, no jargon — explains in practical terms.
  - `"legal"`: Formal language preserving legal precision, section references, effective dates.
  - `None`: Neutral tone.
  - Combinable with any `format=` (e.g., `format="bullets", tone="legal"`).
- **Weekly automated scrapers** (`scripts/update_datasets.py`): All 15 sources updated every Sunday at 9 AM via launchd. Includes:
  - San Diego, San Francisco, Berkeley, Bakersfield (custom scrapers)
  - Oakland, Long Beach, Fresno, Salinas (Legistar API)
  - Los Angeles (City Clerk Connect + PDF text extraction)
  - Clovis, Newport Beach (Code Publishing)
  - Federal public laws (GovInfo API)
  - Executive orders (Federal Register API)
  - Presidential speeches (American Presidency Project)
  - Trump Truth Social (CNN/Stiles archive)
- **One-time build scripts** for each data source in `scripts/`:
  - `build_san_diego_dataset.py`, `build_sf_dataset.py`, `build_berkeley_dataset.py`, `build_bakersfield_dataset.py`
  - `build_legistar_dataset.py` (Oakland, Long Beach, Fresno)
  - `build_salinas_dataset.py`, `build_la_dataset.py`
  - `build_codepublishing_dataset.py` (Clovis, Newport Beach)
  - `build_federal_dataset.py`, `build_executive_orders_ucsb.py`
  - `build_presidential_speeches_dataset.py`
  - `enrich_federal_dataset.py` (adds sponsors, votes, policy areas)
- **Data loaders** (`src/cat_pol/sources/`): Per-source fetch functions for direct HuggingFace access.

### Data Sources

| Source | Rows | HF Repo |
|--------|------|---------|
| San Diego | 87,983 | chrissoria/san-diego-ordinances |
| Los Angeles | 34,427 | chrissoria/la-ordinances |
| Trump Truth Social | 32,000+ | chrissoria/trump-truth-social |
| Berkeley | 9,028 | chrissoria/berkeley-ordinances |
| Federal Laws | 5,915 | chrissoria/federal-public-laws |
| San Francisco | 4,048 | chrissoria/sf-ordinances |
| Long Beach | 3,898 | chrissoria/long-beach-ordinances |
| Newport Beach | 2,719 | chrissoria/newport-beach-ordinances |
| Bakersfield | 2,655 | chrissoria/bakersfield-ordinances |
| Salinas | 2,574 | chrissoria/salinas-ordinances |
| Clovis | 2,343 | chrissoria/clovis-ordinances |
| Oakland | 1,824 | chrissoria/oakland-ordinances |
| Executive Orders | 1,530+ | chrissoria/executive-orders |
| Fresno | 706 | chrissoria/fresno-ordinances |
| Presidential Speeches | 305 | chrissoria/presidential-speeches |

---

## [0.1.0] - 2026-03-20

### Added
- Initial release: thin wrapper around cat-stack with policy-document framing.
- `classify()`, `extract()`, `explore()`, `summarize()` with `document_context=` parameter.

---

[1.1.0]: https://github.com/chrissoria/cat-pol/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/chrissoria/cat-pol/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/chrissoria/cat-pol/releases/tag/v0.1.0
