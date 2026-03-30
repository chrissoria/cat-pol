# Federal Bills & Votes Datasets

Two linked datasets tracking U.S. federal legislation in the 119th Congress (2025–2026):

- **[chrissoria/federal-bills-active](https://huggingface.co/datasets/chrissoria/federal-bills-active)** — 2,531 bills with full text, status, sponsors, and vote breakdowns
- **[chrissoria/federal-votes](https://huggingface.co/datasets/chrissoria/federal-votes)** — 56,569 individual legislator votes (alter-level)

Both datasets update daily at 9:15 AM via automated pipeline.

---

## federal-bills-active — Codebook

### Date Columns

| Column | Type | Description |
|--------|------|-------------|
| `date_last_action` | string | Date of the most recent action on the bill (YYYY-MM-DD). Dataset is sorted by this column, newest first. |
| `date_introduced` | string | Date the bill was first introduced in its chamber of origin. |
| `date_updated` | string | Date the record was last updated in the Congress.gov system. May reflect metadata changes, not substantive action. |

### Bill Identity

| Column | Type | Description |
|--------|------|-------------|
| `bill_type` | string | Type of legislation. See values below. |
| `bill_number` | string | Numeric identifier within the bill type (e.g., "7147"). |
| `congress` | string | Congressional session number. "119" = 119th Congress (Jan 2025 – Jan 2027). |
| `title` | string | Official long title of the bill. |
| `short_title` | string | Popular/short title if available (often empty for bills in early stages). |
| `status` | string | Current stage in the legislative process. See values below. |
| `last_action_text` | string | Full text of the most recent action (e.g., "Referred to the Committee on the Judiciary."). |

#### `bill_type` Values

| Value | Meaning | Count |
|-------|---------|-------|
| `HR` | House Bill — standard legislation originating in the House | 1,519 |
| `S` | Senate Bill — standard legislation originating in the Senate | 665 |
| `HRES` | House Resolution — non-binding, House-only (rules, opinions) | 154 |
| `SRES` | Senate Resolution — non-binding, Senate-only | 89 |
| `HJRES` | House Joint Resolution — binding, requires both chambers + President (often constitutional amendments) | 60 |
| `SJRES` | Senate Joint Resolution — same as HJRES but originates in Senate | 23 |
| `HCONRES` | House Concurrent Resolution — non-binding, both chambers (budget, adjournment) | 14 |
| `SCONRES` | Senate Concurrent Resolution — same as HCONRES but originates in Senate | 7 |

#### `status` Values (Legislative Pipeline)

Listed in order of the legislative process:

| Value | Meaning | Count |
|-------|---------|-------|
| `Introduced` | Filed but no committee assignment yet | 11 |
| `In Committee` | Referred to one or more committees for review | 2,051 |
| `Reported from Committee` | Committee approved the bill (voted to report it) | 62 |
| `Calendared` | Placed on the chamber's legislative calendar (scheduled for floor action) | 100 |
| `Passed House` | Approved by the House of Representatives | 173 |
| `Passed Senate` | Approved by the Senate | 88 |
| `Senate Floor` | Active on the Senate floor (cloture votes, debate) but not yet passed | 4 |
| `Signed into Law` | Passed both chambers and signed by the President | 40 |
| `Vetoed` | Passed both chambers but vetoed by the President | 2 |

### Sponsor Information

| Column | Type | Description |
|--------|------|-------------|
| `sponsor_full_name` | string | Full name with title, party, and state (e.g., "Rep. Cole, Tom [R-OK-4]"). |
| `sponsor_party` | string | Party of the bill's primary sponsor: `R` (Republican), `D` (Democrat), `I` (Independent). |
| `sponsor_state` | string | Two-letter state abbreviation of the sponsor. |
| `num_cosponsors` | int | Number of cosponsors who signed onto the bill. |
| `chamber_of_origin` | string | `House` or `Senate` — where the bill was introduced. |

### Classification

| Column | Type | Description |
|--------|------|-------------|
| `policy_area` | string | Primary policy area assigned by CRS (Congressional Research Service). Top areas: Taxation, Government Operations, Health, Armed Forces, Crime, International Affairs. |
| `subjects` | string | Semicolon-separated list of subject terms (e.g., "Immigration; Border security; Department of Homeland Security"). |

### Vote Data (Passage Votes)

These columns are populated only for bills that have had recorded floor votes. Null for bills still in committee.

| Column | Type | Description |
|--------|------|-------------|
| `republican_yeas` | float | Number of Republican Yea/Aye votes on passage. |
| `democrat_yeas` | float | Number of Democrat Yea/Aye votes on passage. |
| `republican_nays` | float | Number of Republican Nay/No votes on passage. |
| `democrat_nays` | float | Number of Democrat Nay/No votes on passage. |
| `total_yeas` | float | Total Yea/Aye votes across all parties. |
| `total_nays` | float | Total Nay/No votes across all parties. |
| `republican_support_pct` | float | Percentage of voting Republicans who voted Yea (0–100). |
| `democrat_support_pct` | float | Percentage of voting Democrats who voted Yea (0–100). |
| `is_bipartisan` | bool | `True` if both parties had >10% support. |

### Text and Metadata

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Full text of the bill (latest version). Plain text extracted from Congress.gov. |
| `bill_text_version` | string | Version of the text (e.g., "Introduced in House", "Engrossed in House", "Enrolled Bill"). |
| `url` | string | Link to the bill page on congress.gov. |
| `pdf_url` | string | Link to PDF version (when available). |
| `doc_type` | string | Always "bill" in this dataset. |
| `source` | string | Always "federal_bills_active". |

---

## federal-votes — Codebook

One row per legislator per roll call vote (alter-level data). Join to bills on `bill_type` + `bill_number` + `congress`.

| Column | Type | Description |
|--------|------|-------------|
| `date_of_vote` | string | Date the roll call vote occurred (YYYY-MM-DD). Dataset sorted newest first. |
| `bill_type` | string | Bill type (HR, S, HJRES, etc.). Foreign key to bills dataset. |
| `bill_number` | string | Bill number. Foreign key to bills dataset. |
| `congress` | string | Congressional session ("119"). |
| `chamber` | string | `House` or `Senate` — where this vote took place. |
| `roll_call_number` | int | Official roll call number for the session. |
| `vote_type` | string | What was being voted on. See values below. |
| `vote_outcome` | string | Result of the vote: `Passed`, `Failed`, `Cloture on the Motion to Proceed Rejected`, or empty. |
| `legislator_full_name` | string | Legislator's name (House: last name only; Senate: "Last, First"). |
| `legislator_id` | string | Official ID (House: name-id attribute; Senate: LIS member ID). |
| `party` | string | `R` (Republican), `D` (Democrat), or `I` (Independent). |
| `state` | string | Two-letter state abbreviation. |
| `district` | string | Congressional district number (House only; empty for Senate). |
| `vote` | string | Individual vote cast. See values below. |
| `total_yeas` | int | Total Yea votes on this roll call (all legislators). |
| `total_nays` | int | Total Nay votes on this roll call (all legislators). |

#### `vote` Values

| Value | Meaning | Count |
|-------|---------|-------|
| `Yea` | Voted in favor (Senate and some House votes) | 37,250 |
| `Nay` | Voted against (Senate and some House votes) | 14,679 |
| `Aye` | Voted in favor (House suspension votes) | 1,288 |
| `No` | Voted against (House suspension votes) | 1,239 |
| `Not Voting` | Did not vote (absent, abstained, or recused) | 2,100 |
| `Present` | Voted "present" (neither for nor against) | 13 |

#### `vote_type` Values (Top 10)

| Value | Meaning | Count |
|-------|---------|-------|
| `On Passage` | Final vote on whether to pass the bill | 17,302 |
| `On Motion to Suspend the Rules and Pass` | Expedited passage (2/3 majority required, no amendments) | 15,141 |
| `On Motion to Suspend the Rules and Pass, as Amended` | Same as above, with amendments | 6,045 |
| `On Motion to Recommit` | Vote to send bill back to committee (usually fails) | 5,626 |
| `On the Motion` | General procedural motion | 1,800 |
| `On Ordering the Previous Question` | House procedural vote to end debate | 1,733 |
| `On Agreeing to the Resolution` | Vote on a resolution | 1,733 |
| `On the Motion to Proceed` | Senate vote to begin debate on a bill | 1,498 |
| `On the Joint Resolution` | Vote on a joint resolution | 1,200 |
| `On Cloture on the Motion to Proceed` | Senate vote to overcome filibuster (60 votes needed) | 1,199 |

---

## Joining the Datasets

```python
import pandas as pd
from datasets import load_dataset

bills = load_dataset("chrissoria/federal-bills-active", split="train").to_pandas()
votes = load_dataset("chrissoria/federal-votes", split="train").to_pandas()

# Join: get all individual votes for a specific bill
hr7147_votes = votes[(votes["bill_type"] == "HR") & (votes["bill_number"] == "7147")]

# Voting pattern by party
hr7147_votes.groupby(["party", "vote"]).size().unstack(fill_value=0)

# Join bills with vote summary
merged = bills.merge(
    votes.groupby(["bill_type", "bill_number"]).size().reset_index(name="num_votes"),
    on=["bill_type", "bill_number"],
    how="left"
)
```

---

## Data Source

- **API**: [Congress.gov API](https://api.congress.gov/) (bill metadata, actions, subjects, text)
- **House Roll Calls**: [clerk.house.gov](https://clerk.house.gov/) (XML)
- **Senate Roll Calls**: [senate.gov](https://www.senate.gov/legislative/votes.htm) (XML)
- **Update frequency**: Daily at 9:15 AM PT (incremental)
- **Coverage**: 119th Congress (January 2025 – present)

## License

GPL-3.0-or-later
