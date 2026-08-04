# cat-pol

**Political text classification and analysis powered by LLMs.** A policy-specific wrapper around [cat-stack](https://github.com/chrissoria/cat-stack) with built-in access to 15 political data sources on HuggingFace.

## Installation

```bash
pip install cat-pol
```

With optional extras:

```bash
pip install "cat-pol[pdf]"         # PDF document processing
pip install "cat-pol[embeddings]"  # Embedding-based similarity scoring
pip install "cat-pol[sources]"     # Data source loading (datasets, huggingface_hub)
pip install "cat-pol[agent]"       # Claude-subscription backend (model_source="claude-agent")
pip install "cat-pol[codex-agent]" # ChatGPT-subscription backend (model_source="codex-agent")
```

The subscription backends authenticate through your Claude or ChatGPT plan instead of a metered API key — pass `model_source="claude-agent"` (or `"codex-agent"`) and leave `api_key` unset. `model_source="claude-code"` (the Claude Code CLI, if installed) needs no extra at all.

## Quick Start

### Classify ordinances from a built-in source

```python
import catpol as pol

results = pol.classify(
    source="city_san_diego",
    categories=["Housing", "Public Safety", "Infrastructure", "Finance"],
    doc_type="ordinance",
    since="2022-01-01",
    n=50,
    api_key="sk-...",
)
```

### Classify raw text

```python
results = pol.classify(
    input_data=[
        "The committee voted to approve the rezoning request for parcel 42.",
        "Motion to table the budget amendment until the next session.",
    ],
    categories=["Approval", "Rejection", "Deferral", "Amendment"],
    document_context="City council meeting minutes",
    api_key="sk-...",
)
```

### Optimize prompts with user feedback

```python
result = pol.prompt_tune(
    source="city_san_diego",
    categories=["Pro-Business", "Pro-Regulation", "Tax Increase", "Tax Decrease"],
    doc_type="ordinance",
    since="2020-01-01",
    n=100,
    api_key="sk-...",
    sample_size=15,
)

# Use the optimized prompt for full classification
results = pol.classify(
    source="city_san_diego",
    categories=["Pro-Business", "Pro-Regulation", "Tax Increase", "Tax Decrease"],
    system_prompt=result["system_prompt"],
    api_key="sk-...",
)
```

### Summarize with different formats

```python
# Bullet points
pol.summarize(source="federal_executive_orders", n=10, format="bullets", api_key="sk-...")

# Full report
pol.summarize(source="federal_laws", n=5, format="report", api_key="sk-...")

# One-liner
pol.summarize(source="social_trump_truth", since="2024-01-01", n=20, format="one-liner", api_key="sk-...")
```

### Discover categories

```python
result = pol.extract(
    source="city_berkeley",
    n=200,
    api_key="sk-...",
)
print(result["top_categories"])
```

### Fetch raw data

```python
# List all sources
pol.list_sources()
pol.list_sources(level="city")
pol.list_sources(level="federal")

# Fetch data
df = pol.fetch_source("city_san_diego", n=100, since="2020-01-01", doc_type="ordinance")
df = pol.fetch_source("federal_executive_orders", n=50)
df = pol.fetch_source("social_trump_truth", since="2024-01-01")
```

## Data Sources

All datasets are public on HuggingFace — no authentication required.

### California Cities

| Source | Rows | Types | Repo |
|--------|------|-------|------|
| `city_san_diego` | 87,983 | ordinances, resolutions | [chrissoria/san-diego-ordinances](https://huggingface.co/datasets/chrissoria/san-diego-ordinances) |
| `city_los_angeles` | 34,427 | ordinances | [chrissoria/la-ordinances](https://huggingface.co/datasets/chrissoria/la-ordinances) |
| `city_berkeley` | 9,028 | ordinances | [chrissoria/berkeley-ordinances](https://huggingface.co/datasets/chrissoria/berkeley-ordinances) |
| `city_san_francisco` | 4,033 | ordinances | [chrissoria/sf-ordinances](https://huggingface.co/datasets/chrissoria/sf-ordinances) |
| `city_long_beach` | 3,898 | ordinances, resolutions | [chrissoria/long-beach-ordinances](https://huggingface.co/datasets/chrissoria/long-beach-ordinances) |
| `city_bakersfield` | 2,655 | ordinances | [chrissoria/bakersfield-ordinances](https://huggingface.co/datasets/chrissoria/bakersfield-ordinances) |
| `city_newport_beach` | 2,719 | ordinances | [chrissoria/newport-beach-ordinances](https://huggingface.co/datasets/chrissoria/newport-beach-ordinances) |
| `city_salinas` | 2,574 | ordinances, resolutions | [chrissoria/salinas-ordinances](https://huggingface.co/datasets/chrissoria/salinas-ordinances) |
| `city_clovis` | 2,343 | ordinances | [chrissoria/clovis-ordinances](https://huggingface.co/datasets/chrissoria/clovis-ordinances) |
| `city_oakland` | 1,824 | ordinances | [chrissoria/oakland-ordinances](https://huggingface.co/datasets/chrissoria/oakland-ordinances) |
| `city_fresno` | 706 | ordinances, resolutions | [chrissoria/fresno-ordinances](https://huggingface.co/datasets/chrissoria/fresno-ordinances) |

### Federal

| Source | Rows | Types | Repo |
|--------|------|-------|------|
| `federal_laws` | 5,915 | public laws (1995–present) | [chrissoria/federal-public-laws](https://huggingface.co/datasets/chrissoria/federal-public-laws) |
| `federal_executive_orders` | 1,530+ | executive orders | [chrissoria/executive-orders](https://huggingface.co/datasets/chrissoria/executive-orders) |
| `federal_speeches` | 305 | SOTU, inaugurals | [chrissoria/presidential-speeches](https://huggingface.co/datasets/chrissoria/presidential-speeches) |

### Social Media

| Source | Rows | Types | Repo |
|--------|------|-------|------|
| `social_trump_truth` | 32,000+ | Truth Social posts | [chrissoria/trump-truth-social](https://huggingface.co/datasets/chrissoria/trump-truth-social) |

All sources are updated weekly (Sundays at 9 AM) via automated scrapers. Truth Social is updated **daily** at 9 AM.

### Trump Truth Social Dataset Columns

The `social_trump_truth` dataset is enriched with metadata, market data, and image descriptions:

**Post metadata:**

| Column | Description |
|--------|-------------|
| `date` | Post date (YYYY-MM-DD) |
| `time` | Post time in UTC (HH:MM:SS) |
| `day_of_week` | Day name (Monday, Tuesday, etc.) |
| `datetime` | Full ISO timestamp |
| `text` | Post text content |
| `url` | Truth Social post URL |
| `post_id` | Unique post identifier |
| `is_president` | Whether Trump was president at time of post |
| `is_president_elect` | Whether Trump was president-elect at time of post |
| `replies_count` | Number of replies |
| `reblogs_count` | Number of reblogs |
| `favourites_count` | Number of favourites |
| `media_urls` | Image/video URLs attached to the post |
| `has_media` | Whether the post has media attachments |
| `image_alt_text` | AI-generated factual image description (alt-text format) |

**Market data (18 tickers):**

Each ticker has 7 columns following the convention `{ticker}_{metric}`:

| Metric | Description |
|--------|-------------|
| `{ticker}_open` | Daily open price |
| `{ticker}_close` | Daily close price |
| `{ticker}_1hr_before` | Price 1 hour before the post |
| `{ticker}_5min_before` | Price 5 minutes before the post |
| `{ticker}_at_post` | Price at time of post |
| `{ticker}_5min_after` | Price 5 minutes after the post |
| `{ticker}_1hr_after` | Price 1 hour after the post |

Tickers included:

| Ticker | Name | Category |
|--------|------|----------|
| `sp500` | S&P 500 (^GSPC) | Broad market |
| `dia` | SPDR Dow Jones Industrial Average ETF | Broad market |
| `qqq` | Invesco QQQ (Nasdaq-100) | Tech/growth |
| `djt` | Trump Media & Technology Group | Trump-linked |
| `lmt` | Lockheed Martin | Defense |
| `war` | Themes US Military Academy ETF | Defense |
| `xli` | Industrial Select Sector SPDR | Industrials |
| `xlv` | Health Care Select Sector SPDR | Healthcare |
| `xph` | SPDR S&P Pharmaceuticals ETF | Pharma |
| `cnrg` | SPDR S&P Kensho Clean Power ETF | Clean energy |
| `gld` | SPDR Gold Shares | Gold/commodities |
| `uso` | United States Oil Fund | Oil/energy |
| `fxi` | iShares China Large-Cap ETF | China/trade |
| `eww` | iShares MSCI Mexico ETF | Mexico/trade |
| `vgk` | Vanguard FTSE Europe ETF | Europe |
| `ibit` | iShares Bitcoin ETF | Crypto |
| `tlt` | iShares 20+ Year Treasury Bond ETF | Bonds/rates |
| `uup` | Invesco DB US Dollar Index | USD strength |

Intraday prices use the highest available resolution: 1-minute (last ~7 days), 5-minute (last ~60 days), or hourly (last ~2 years). Weekend/holiday posts use the most recent trading day's values. The `sp500_resolution` column indicates the data resolution used.

## API

| Function | Description |
|----------|-------------|
| `classify()` | Classify text into predefined categories |
| `prompt_tune()` | Optimize classification prompts via user feedback |
| `extract()` | Discover and normalize categories from text |
| `explore()` | Raw category extraction (no deduplication) |
| `collapse_themes()` | Consolidate an explore() inventory into a deduplicated taxonomy (re-export from cat-stack) |
| `summarize()` | Summarize text, PDFs, or image URLs with format options (paragraph, bullets, one-liner, structured, report, alt-text) |
| `list_sources()` | List available data sources |
| `fetch_source()` | Fetch raw data from a source |

All functions accept either `input_data=` (raw text, files, directories) or `source=` (pull from HuggingFace). All `cat-stack` parameters (multi-model ensemble, batch mode, chain-of-thought, etc.) pass through via `**kwargs`.

## Ecosystem

| Package | Role |
|---------|------|
| [cat-stack](https://github.com/chrissoria/cat-stack) | Domain-agnostic LLM classification engine |
| **cat-pol** | Political text classification (this package) |
| [cat-vader](https://github.com/chrissoria/catvader) | Social media text (Reddit, Twitter/X) |
| [cat-ademic](https://github.com/chrissoria/cat-ademic) | Academic papers and citations |

## License

GPL-3.0-or-later
