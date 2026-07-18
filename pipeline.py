"""End-to-end LangGraph pipeline for the system ticket summary task.

Mirrors ticket_summary.ipynb step by step:

    load -> repair -> convert -> clean -> filter -> map_products -> summarize

Shared by the Streamlit app (app.py). Rules baked in:
- validate, never repair silently: wrong field count raises with line numbers
- the known column shift (missing N/A placeholder) is repaired with a printed
  report of every affected order number
- only completely empty columns are dropped, and the drop is reported
"""
import csv
import io
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

DATA_DIR = Path("data")

# Only these service categories are analyzed (per task description)
ALLOWED_CATEGORIES = ["HDW", "NET", "KAI", "KAV", "GIGA", "VOD", "KAD"]

# Category -> Product mapping (per task description). HDW passes the category
# filter but has no product in the doc, so it is kept as its own group.
CATEGORY_TO_PRODUCT = {
    "KAI": "Broadband",
    "NET": "Broadband",
    "KAV": "Voice",
    "KAD": "TV",
    "GIGA": "GIGA",
    "VOD": "VOD",
    "HDW": "Hardware",
}

# Support team codes: only ever legal in PLANNING_GROUP_KB. One of these in
# NOTE_MAXIMUM is the fingerprint of a shifted row.
TEAM_CODES = {"TSCW2", "TSCKT", "TSCS2"}

# Line 1 of the ticket system's export, verbatim. This single string is the
# whole file contract: uploaded headers are compared against it, and files
# uploaded without a header line get exactly this header applied.
EXPECTED_HEADER = (
    "ORDER_NUMBER,ORDER_ID,ORDER_UNIT_ID,ACCEPTANCE_TIME,COMPLETION_TIME,"
    "CUSTOMER_NUMBER,CUSTOMER_COUNT,ORDER_TYPE,ORDER_CLASS,PROCESSING_STATUS,"
    "SERVICE_CATEGORY,ORDER_DESCRIPTION_1,ORDER_DESCRIPTION_2,"
    "ORDER_DESCRIPTION_3_MAXIMUM,ADDITIONAL_ORDER_DESCRIPTION_MAXIMUM,"
    "NOTE_MAXIMUM,PLANNING_GROUP_KB,COMPLETION_RESULT_KB,"
    "REFERENCE_COMPLETION_RESULT,COMPLETION_NOTE_MAXIMUM,NETWORK_LEVEL,CAUSE,"
    "REFERENCE_ERROR_CAUSE,SERVICE_PROVIDER,REFERENCE_SERVICE_PROVIDER,"
    "SUBUNIT_NAME,CUSTOMER_COMPLETION_TIME,PROCESSING_END_TIME_MAXIMUM,"
    "PROCESSING_END_TIME_MINIMUM,ACCEPTANCE_TIME_MINIMUM,ASSIGNMENT_TIME_MINIMUM,"
    "IMIL_TIME_MINIMUM,CUSTOMER_TIME_MINIMUM,START_TIME_MINIMUM,ASSIGNMENT_TIME,"
    "ASSIGNED_BY_NAME,ASSIGNMENT_PROCESSING_STATUS,ASSIGNMENT_ADDITIONAL_INFO"
)
EXPECTED_COLUMNS = EXPECTED_HEADER.split(",")

# A real ticket row starts with an order number like 001-0671177/24
ORDER_NUMBER_PATTERN = re.compile(r"\d{3}-\d+/\d{2}")

STORY_PROMPT = ChatPromptTemplate.from_template("""\
You are a telecom customer-service analyst. Write a storytelling summary of the
support-ticket history of customer **{customer}** for the product **{product}**
(service categories: {categories}).

Structure the summary into exactly these five sections:
1. **Initial Issue**
2. **Follow-ups**
3. **Developments**
4. **Later Incidents**
5. **Recent Events**

For every section provide:
- **Timeframe:** the period covered
- **Ticket Numbers:** the relevant order numbers
- **Narrative:** what happened - the nature of the issues, the customer's feedback, actions taken by support, and outcomes.

Rules:
- Use ONLY the ticket data below; never invent facts.
- Keep events chronological and split the timeline sensibly across the five sections.
- If there are few tickets (even a single one), still produce all five sections but keep them very brief.
- Some tickets are in German - translate everything and write the whole summary in English.

Tickets (chronological):
{tickets}
""")


# --------------------------------------------------------------------------- #
# Step 1: load and validate                                                    #
# --------------------------------------------------------------------------- #
def load_raw_tickets(source):
    """Parse raw tickets from a path, file-like object, raw text or bytes.

    Accepts files with or without the header line:
    - line 1 equals the standard header  -> header present
    - line 1 looks like a ticket row     -> headerless, standard header applied
    - anything else                      -> raise (wrong file / wrong scheme)

    Returns (header, rows, had_header). Raises ValueError on any malformed
    input (validate, never repair silently).
    """
    if isinstance(source, bytes):
        text = source.decode("utf-8-sig")          # uploaded files arrive as bytes
    elif hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else str(raw).lstrip("﻿")
    elif isinstance(source, (str, Path)) and Path(str(source)).exists():
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = str(source).lstrip("﻿")

    parsed = [r for r in csv.reader(io.StringIO(text)) if r]
    if not parsed:
        raise ValueError("The file is empty.")

    first = parsed[0]
    if first == EXPECTED_COLUMNS:                  # line 1 is exactly our header
        header, data, had_header = first, parsed[1:], True
        if not data:
            raise ValueError(
                "No ticket rows found - the file contains only the header line."
            )
    else:                                          # headerless file
        if len(first) != len(EXPECTED_COLUMNS):
            raise ValueError(
                f"This file has {len(first)} fields per row, expected "
                f"{len(EXPECTED_COLUMNS)} - different ticket scheme?"
            )
        if not ORDER_NUMBER_PATTERN.fullmatch(first[0]):
            raise ValueError(
                "Line 1 is neither the ticket header nor a ticket row "
                f"(first field: {first[0]!r}) - wrong file?"
            )
        header, data, had_header = list(EXPECTED_COLUMNS), parsed, False

    n_cols = len(header)
    bad = [(i + (2 if had_header else 1), len(r))
           for i, r in enumerate(data) if len(r) != n_cols]
    if bad:
        raise ValueError(
            f"Incomplete tickets - expected {n_cols} fields per row, "
            f"but got (line, fields): {bad}"
        )

    return header, data, had_header


# --------------------------------------------------------------------------- #
# Step 1.5: repair the shifted rows                                            #
# --------------------------------------------------------------------------- #
def find_shifted_orders(rows, header):
    """Order numbers of rows carrying a team code where the note belongs."""
    note_idx = header.index("NOTE_MAXIMUM")
    return [r[0] for r in rows if r[note_idx] in TEAM_CODES]


def repair_shifted_rows(rows: list, header: list) -> list:
    """Fix rows written with one missing N/A: insert it back, everything slides right."""
    add_idx = header.index("ADDITIONAL_ORDER_DESCRIPTION_MAXIMUM")
    note_idx = header.index("NOTE_MAXIMUM")

    repaired = []
    for r in rows:
        if r[note_idx] in TEAM_CODES:   # team code where the note belongs = shifted row
            r.insert(add_idx, "N/A")    # insert the forgotten N/A -> rest slides right
            r.pop()                     # drop the padding beyond the last column
            repaired.append(r[0])

    print(f"Repaired {len(repaired)} shifted tickets: {repaired}")
    return rows


# --------------------------------------------------------------------------- #
# Step 2: convert and save                                                     #
# --------------------------------------------------------------------------- #
def convert_and_save(df: pd.DataFrame, out_dir: Path = DATA_DIR) -> dict:
    """Persist the converted data as CSV and Excel; return the file paths."""
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "tickets_converted.csv"
    xlsx_path = out_dir / "tickets_converted.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return {"csv": csv_path, "xlsx": xlsx_path}


# --------------------------------------------------------------------------- #
# Step 3: clean                                                                #
# --------------------------------------------------------------------------- #
def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing values, parse timestamps, drop all-empty columns."""
    out = df.copy()
    out = out.apply(lambda s: s.str.strip())
    out = out.replace({"N/A": pd.NA, "": pd.NA})

    out["ACCEPTANCE_TIME"] = pd.to_datetime(out["ACCEPTANCE_TIME"], format="%m/%d/%Y %H:%M")
    out["COMPLETION_TIME"] = pd.to_datetime(out["COMPLETION_TIME"], format="%m/%d/%Y %H:%M")

    out["RESOLUTION_MINUTES"] = (
        (out["COMPLETION_TIME"] - out["ACCEPTANCE_TIME"]).dt.total_seconds() / 60
    )

    empty_cols = out.columns[out.isna().all()]
    if len(empty_cols):
        out = out.drop(columns=empty_cols)
        print(f"Dropped {len(empty_cols)} all-empty columns: {list(empty_cols)}")
    return out


# --------------------------------------------------------------------------- #
# Step 4: filter                                                               #
# --------------------------------------------------------------------------- #
def filter_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only tickets in ALLOWED_CATEGORIES, sorted chronologically."""
    out = df[df["SERVICE_CATEGORY"].isin(ALLOWED_CATEGORIES)].copy()
    return out.sort_values("ACCEPTANCE_TIME").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 5: map products                                                         #
# --------------------------------------------------------------------------- #
def map_products(df: pd.DataFrame) -> pd.DataFrame:
    """Add a PRODUCT column derived from SERVICE_CATEGORY."""
    out = df.copy()
    out["PRODUCT"] = out["SERVICE_CATEGORY"].map(CATEGORY_TO_PRODUCT)
    return out


# --------------------------------------------------------------------------- #
# Step 6: summarize with Gemini                                                #
# --------------------------------------------------------------------------- #
def format_tickets(df: pd.DataFrame) -> str:
    """Render tickets as compact chronological lines for the LLM."""
    lines = []
    for _, r in df.iterrows():
        parts = [
            str(r["ORDER_NUMBER"]),
            r["ACCEPTANCE_TIME"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["ACCEPTANCE_TIME"]) else "?",
            f"cat={r['SERVICE_CATEGORY']}",
            "issue: " + " / ".join(str(r[c]) for c in [
                "ORDER_DESCRIPTION_1", "ORDER_DESCRIPTION_2", "ORDER_DESCRIPTION_3_MAXIMUM"
            ] if c in r.index and pd.notna(r[c])),
        ]
        if pd.notna(r["NOTE_MAXIMUM"]):
            parts.append(f"note: {r['NOTE_MAXIMUM']}")
        resolution = " / ".join(str(r[c]) for c in [
            "COMPLETION_RESULT_KB", "COMPLETION_NOTE_MAXIMUM"
        ] if c in r.index and pd.notna(r[c]))
        if resolution:
            parts.append(f"resolution: {resolution}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def get_llm() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is empty - add it to .env "
            "(get one at https://aistudio.google.com/apikey)."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    return ChatGoogleGenerativeAI(model=model, temperature=0.3)


def summarize_products(df: pd.DataFrame, llm=None, on_progress=None) -> dict:
    """Generate the 5-section storytelling summary per customer per product.

    Returns {customer: {product: summary_text}}.
    """
    # with_retry: the Gemini endpoint intermittently drops connections; retry
    # each summary call up to 3 times instead of failing the whole run.
    chain = (STORY_PROMPT | (llm or get_llm()) | StrOutputParser()).with_retry(
        stop_after_attempt=3)
    summaries = {}
    for (cust, product), group in df.groupby(["CUSTOMER_NUMBER", "PRODUCT"]):
        if on_progress:
            on_progress(f"customer {cust} - {product}", len(group))
        summaries.setdefault(cust, {})[product] = chain.invoke({
            "customer": cust,
            "product": product,
            "categories": ", ".join(sorted(group["SERVICE_CATEGORY"].unique())),
            "tickets": format_tickets(group),
        })
    return summaries


# --------------------------------------------------------------------------- #
# LangGraph assembly                                                           #
# --------------------------------------------------------------------------- #
class PipelineState(TypedDict, total=False):
    raw_source: object            # path, file-like object, or raw text content
    header: list                  # column names from the raw file
    rows: list                    # ticket rows as plain lists (repaired in place)
    df: Optional[pd.DataFrame]    # working DataFrame, transformed node by node
    summaries: dict               # product -> storytelling summary text


def build_pipeline(llm=None, on_progress=None):
    """Compile the full load -> summarize LangGraph pipeline."""

    def node_load(state: PipelineState) -> PipelineState:
        header, rows, _ = load_raw_tickets(state["raw_source"])
        return {"header": header, "rows": rows}

    def node_repair(state: PipelineState) -> PipelineState:
        rows = repair_shifted_rows(state["rows"], state["header"])
        return {"df": pd.DataFrame(rows, columns=state["header"])}

    def node_convert(state: PipelineState) -> PipelineState:
        convert_and_save(state["df"])
        return {}

    def node_clean(state: PipelineState) -> PipelineState:
        return {"df": clean_tickets(state["df"])}

    def node_filter(state: PipelineState) -> PipelineState:
        return {"df": filter_categories(state["df"])}

    def node_map(state: PipelineState) -> PipelineState:
        return {"df": map_products(state["df"])}

    def node_summarize(state: PipelineState) -> PipelineState:
        return {"summaries": summarize_products(state["df"], llm=llm, on_progress=on_progress)}

    builder = StateGraph(PipelineState)

    builder.add_node("load", node_load)
    builder.add_node("repair", node_repair)
    builder.add_node("convert", node_convert)
    builder.add_node("clean", node_clean)
    builder.add_node("filter", node_filter)
    builder.add_node("map_products", node_map)
    builder.add_node("summarize", node_summarize)

    builder.add_edge(START, "load")
    builder.add_edge("load", "repair")
    builder.add_edge("repair", "convert")
    builder.add_edge("convert", "clean")
    builder.add_edge("clean", "filter")
    builder.add_edge("filter", "map_products")
    builder.add_edge("map_products", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()
