# System Ticket Data Summary

Streamlit app + notebook that turn raw system ticket data into AI-powered
storytelling summaries per customer per product, using LangChain / LangGraph
with Google Gemini / Gemma models.

Full documentation (implementation steps, code reference, user guide, business
analysis): [docs/Ticket_Data_Summary_Documentation.pdf](docs/Ticket_Data_Summary_Documentation.pdf)

## Project layout

| File | Purpose |
|---|---|
| `Ticket Data (2).txt` | Raw ticket data (comma-separated, mixed English/German) |
| `notebooks/ticket_summary.ipynb` | Step-by-step prototype: every pipeline step explored line by line, then as a function |
| `pipeline.py` | Reusable pipeline module + LangGraph graph (shared logic) |
| `app.py` | Streamlit app (upload, preprocessing report, AI summaries, insights) |
| `requirements.txt` / `pyproject.toml` | Dependencies (managed with uv) |
| `.env.example` | Template for the required secrets - copy to `.env` locally |
| `docs/` | Standalone documentation (HTML source + rendered PDF) |
| `data/` | Generated `tickets_converted.csv` / `.xlsx` + small test files |

## Setup (local)

```bash
uv sync                       # or: pip install -r requirements.txt
cp .env.example .env          # then paste your key into .env
streamlit run app.py
```

`.env` contents (never committed - the file is git-ignored):

```
GOOGLE_API_KEY=your-key-here   # https://aistudio.google.com/apikey
GEMINI_MODEL=gemma-4-31b-it
```

Model strategy: `gemma-4-31b-it` for development (very high free-tier quota),
switch the line to `gemini-3.5-flash` for final runs where narrative polish
matters most. Only `.env` changes - no code edits needed.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub. `.env` is git-ignored; `.env.example` documents
   the required variables.
2. On [share.streamlit.io](https://share.streamlit.io): New app, pick the
   repo/branch, main file `app.py`.
3. In the app's Settings, add Secrets:

   ```toml
   GOOGLE_API_KEY = "your-key-here"
   GEMINI_MODEL = "gemma-4-31b-it"
   ```

   Streamlit exposes top-level secrets as environment variables, so the app
   picks them up with no code changes.

## The pipeline

Every step is a LangGraph node; one `invoke` runs a file end to end:

```
START -> load -> repair -> convert -> clean -> filter -> map_products -> summarize -> END
```

1. **Load & validate** - `csv.reader` with `utf-8-sig` (strips the BOM).
   Accepts files with or without the header line: a valid header is used
   as-is; a headerless file whose line 1 matches the order-number pattern
   gets the standard 38-column header applied (reported in the UI). Anything
   else raises a ValueError with exact line numbers - validate, never repair
   silently.
2. **Repair** - 19 sample rows were written with one missing `N/A`
   placeholder, sliding every value from ADDITIONAL_ORDER_DESCRIPTION onward
   one column left (fingerprint: a team code TSCW2/TSCKT/TSCS2 sitting in
   NOTE_MAXIMUM). The repair re-inserts the placeholder (`list.insert`),
   drops the padding at the row end (`list.pop`), and prints every repaired
   order number. Values only move back under their correct headers.
3. **Convert** - saves `data/tickets_converted.csv` / `.xlsx` after repair.
4. **Clean** - strip whitespace, literal `N/A` to real missing values, parse
   timestamps (bad dates raise), derive `RESOLUTION_MINUTES`, drop
   completely-empty columns with a printed report.
5. **Filter** - keep only HDW, NET, KAI, KAV, GIGA, VOD, KAD; sort
   chronologically.
6. **Map products** - KAI, NET -> Broadband; KAV -> Voice; KAD -> TV;
   GIGA -> GIGA; VOD -> VOD; HDW kept as its own Hardware group.
7. **Summarize** - per customer per product (18 summaries for the sample),
   five-section storytelling format (Initial Issue, Follow-ups, Developments,
   Later Incidents, Recent Events), German tickets translated, no invented
   facts (audited: zero invented or omitted ticket numbers), each LLM call
   retried up to 3 times.

## The app

- **Upload** (sidebar): raw `.txt` with or without header; clear errors for
  invalid files; a data-quality panel lists repaired tickets.
- **AI Summaries**: generate once, then browse by customer and product;
  download all summaries as Markdown.
- **Data**: converted and cleaned tables, CSV download.
- **Insights**: an executive scorecard plus three business questions, every
  finding computed live from the filtered data:
  - **Scorecard**: repeat-contact rate, first-contact resolution, escalation
    rate, average resolution time, contacts per issue.
  - **Q1. What breaks the most?** Pareto per device (German terms
    normalized) + tickets per product.
  - **Q2. Do fixes hold?** Repeat-chain timeline - one row per chain, one
    dot per contact, colored remote fix vs escalation - with the chain table
    in an expander.
  - **Q3. Which customers need attention?** Tickets per customer by product
    + first vs repeat contacts - the churn-risk view.
  - **More views**: most common fixes, first-contact resolution by product,
    workload per support team, top root-cause codes, daily volume, weekday x
    hour heatmap, customer timelines (labeled as needing more data).

## Key findings in the sample data

- **44% of tickets are one device** (the cable router), across all three
  customers in two weeks - prevention (firmware push, batch swap,
  self-service reset) attacks half the queue at once.
- **Repeated remote fixes never held**: hold times of 2-25 hours, and 100%
  of repeat chains ended in a device replacement or technician anyway -
  supporting an escalate-on-second-contact rule.
- **First-contact resolution is 67% overall but 0% for Hardware** - four
  products are perfect, one device destroys the KPI.
- **Every customer is a repeat caller** (10-11 contacts each in two weeks) -
  a churn risk, not just a support cost.

## Notebook

`notebooks/ticket_summary.ipynb` documents every step three ways: a markdown
explanation, line-by-line exploration cells with visible output, then the
assembled function. Steps 0-6 run without an API key; Step 10 generates the
18 summaries; Step 11 reproduces the analysis charts.
