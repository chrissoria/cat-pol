# cat-pol

Policy document classification powered by LLMs. A thin, policy-specific wrapper around [cat-stack](https://github.com/chrissoria/cat-stack).

`cat-pol` adds policy-document-specific prompt framing ("The following is an excerpt from a policy document...") on top of the domain-agnostic `cat-stack` engine — giving LLMs the context that text comes from policy documents.

## Installation

```bash
pip install cat-pol
```

With optional extras:

```bash
pip install "cat-pol[pdf]"         # PDF document processing
pip install "cat-pol[embeddings]"  # Embedding-based similarity scoring
```

## Quick Start

### Classify policy document excerpts

```python
import cat_pol

results = cat_pol.classify(
    input_data=[
        "The committee voted to approve the rezoning request for parcel 42.",
        "Motion to table the budget amendment until the next session.",
    ],
    categories=["Approval", "Rejection", "Deferral", "Amendment"],
    document_context="City council meeting minutes from March 2026",
    api_key="sk-...",
)
```

### Discover categories from policy text

```python
result = cat_pol.extract(
    input_data=excerpts,
    api_key="sk-...",
    document_context="FOIA request responses regarding environmental compliance",
)
print(result["top_categories"])
```

### Summarize policy documents

```python
summaries = cat_pol.summarize(
    input_data=documents,
    api_key="sk-...",
    description="Legislative bill summaries from the 118th Congress",
)
```

## How It Works

`cat-pol` is a thin wrapper that:

1. Takes your `document_context` parameter
2. Injects policy-specific framing: *"The following is an excerpt from a policy document. Context: {document_context}."*
3. Delegates to `cat-stack` for all LLM communication, classification logic, batch processing, and ensemble methods

All `cat-stack` parameters (multi-model ensemble, batch mode, chain-of-thought, etc.) are passed through via `**kwargs`.

## API

| Function | Description |
|----------|-------------|
| `classify()` | Classify excerpts into predefined categories |
| `extract()` | Discover and normalize categories from document text |
| `explore()` | Raw category extraction (no deduplication) |
| `summarize()` | Summarize documents (pass-through to cat-stack) |

## Ecosystem

| Package | Role |
|---------|------|
| [cat-stack](https://github.com/chrissoria/cat-stack) | Domain-agnostic LLM classification engine |
| [cat-survey](https://github.com/chrissoria/cat-survey) | Survey response classification |
| **cat-pol** | Policy document classification (this package) |
| [cat-cog](https://github.com/chrissoria/cat-cog) | Cognitive assessment scoring (CERAD) |

## License

GPL-3.0-or-later
