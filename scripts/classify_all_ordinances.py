#!/usr/bin/env python3
"""Classify all remaining ordinances for SD and SF, merge with existing, push to HF."""

from dotenv import load_dotenv
import os, sys, pandas as pd
import cat_pol as pol
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

load_dotenv("/Users/chrissoria/Documents/Research/cat-llm/.env", override=True)
api_key = os.getenv("OPENAI_API_KEY")

CATEGORIES = [
    "Community Plan Updates (e.g., comprehensive plans, neighborhood plans)",
    "Contract Amendments (e.g., modifications, revisions)",
    "Environmental Compliance (e.g., stormwater regulations, habitat protections)",
    "Infrastructure Projects (e.g., road improvements, water systems)",
    "Public Safety Communications (e.g., emergency response systems, communication networks)",
    "Construction Management Services (e.g., project oversight, coordination)",
    "Rezoning (e.g., residential to commercial, density adjustments)",
    "Historical Preservation (e.g., landmark designation, restoration projects)",
    "Health and Safety Applications (e.g., permits, inspections)",
    "Public Art Projects (e.g., murals, sculptures)",
    "Parking Management Services (e.g., lot management, enforcement)",
    "Housing Development (e.g., affordable housing, new construction)",
]

CAT_COLS = [f"category_{i}" for i in range(1, 14)]  # 12 + Other
CAT_NAMES = [c.split("(")[0].strip() for c in CATEGORIES] + ["Other"]

CITY = sys.argv[1] if len(sys.argv) > 1 else "sd"

# Load source
if CITY == "sd":
    repo = "chrissoria/san-diego-ordinances"
    existing_csv = "/Users/chrissoria/Documents/Research/cat-pol/tests/sd_classified_1000.csv"
    context = "San Diego city ordinances"
    print("Loading San Diego...")
    ds = load_dataset(repo, split="train")
    df = ds.to_pandas()
    ords = df[(df["doc_type"] == "ordinance") & (df["text"].str.len() > 0)].sort_values("date", ascending=False).reset_index(drop=True)
elif CITY == "sf":
    repo = "chrissoria/sf-ordinances"
    existing_csv = "/Users/chrissoria/Documents/Research/cat-pol/tests/sf_classified_1000.csv"
    context = "San Francisco city ordinances"
    print("Loading San Francisco...")
    ds = load_dataset(repo, split="train")
    df = ds.to_pandas()
    ords = df[df["text"].str.len() > 0].reset_index(drop=True)
else:
    print(f"Usage: {sys.argv[0]} [sd|sf]")
    sys.exit(1)

print(f"Total ordinances with text: {len(ords)}")

# Load existing classifications (first 1000)
existing = pd.read_csv(existing_csv)
already_done = len(existing)
print(f"Already classified: {already_done}")

# Get remaining
remaining = ords.iloc[already_done:]
print(f"Remaining: {len(remaining)}")

if len(remaining) == 0:
    print("All done!")
    sys.exit(0)

# Classify remaining in batch mode (50% cost savings)
print(f"\nClassifying {len(remaining)} remaining ordinances (batch mode)...")
new_results = pol.classify(
    input_data=remaining["text"].tolist(),
    categories=CATEGORIES,
    document_context=context,
    api_key=api_key,
    add_other=True,
    check_verbosity=False,
    batch_mode=True,
    filename=f"{CITY}_remaining_{len(remaining)}.csv",
    save_directory="/Users/chrissoria/Documents/Research/cat-pol/tests/",
)

print(f"New results: {new_results.shape}")
print(f"Status: {new_results['processing_status'].value_counts().to_dict()}")

# Merge: build classification columns for the full ordinance set
print("\nMerging classifications onto full dataset...")

# Initialize classification columns as NA
for col in CAT_COLS:
    ords[col] = pd.NA

ords["classification_status"] = ""

# Fill in first 1000 from existing CSV
for i in range(min(already_done, len(ords))):
    for col in CAT_COLS:
        if col in existing.columns:
            ords.at[i, col] = existing.at[i, col]
    ords.at[i, "classification_status"] = existing.at[i, "processing_status"]

# Fill in remaining from new results
for i, (_, row) in enumerate(new_results.iterrows()):
    orig_idx = already_done + i
    if orig_idx >= len(ords):
        break
    for col in CAT_COLS:
        if col in new_results.columns:
            ords.at[orig_idx, col] = row[col]
    ords.at[orig_idx, "classification_status"] = row["processing_status"]

classified_count = (ords["classification_status"] == "success").sum()
print(f"Total classified: {classified_count}/{len(ords)}")

# Merge back into full dataset (including non-ordinances and no-text rows)
for col in CAT_COLS + ["classification_status"]:
    df[col] = pd.NA

# Map back using the original index positions of ordinances with text
if CITY == "sd":
    text_mask = (df["doc_type"] == "ordinance") & (df["text"].str.len() > 0)
else:
    text_mask = df["text"].str.len() > 0

text_indices = df[text_mask].sort_values("date", ascending=False).index.tolist()

for ords_idx in range(len(ords)):
    if ords_idx < len(text_indices):
        df_idx = text_indices[ords_idx]
        for col in CAT_COLS + ["classification_status"]:
            df.at[df_idx, col] = ords.at[ords_idx, col]

# Rename category columns to meaningful names
rename_map = {f"category_{i+1}": name for i, name in enumerate(CAT_NAMES)}
df = df.rename(columns=rename_map)

print(f"\nFull dataset with classifications: {df.shape}")
classified_in_full = (df["classification_status"] == "success").sum()
print(f"Classified rows in full dataset: {classified_in_full}")

# Push to HF
load_dotenv("/Users/chrissoria/Documents/Research/cat-pol/.env", override=True)
token = os.getenv("CATLLM_HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
api = HfApi(token=token)
api.whoami()

ds_out = Dataset.from_pandas(df)
ds_out.push_to_hub(repo, token=token, private=False)
print(f"Pushed {len(df)} rows to {repo}")
