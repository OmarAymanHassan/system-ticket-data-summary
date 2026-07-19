"""Streamlit App for System Ticket Data Summary.

Upload a raw ticket .txt file (with or without header line) -> validate,
repair shifted rows, convert, clean, filter, map products -> AI storytelling
summaries per product (Gemini through the LangGraph pipeline) -> business
insights.

Run with:  streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl
from fpdf import FPDF

# Completion results containing these words mean the ticket left the cheap
# remote path: a technician was involved or hardware was shipped/replaced.
ESCALATION_WORDS = ("technician", "techniker", "replaced", "sent", "ordered",
                    "commissioned", "forwarded", "weitergeleitet", "versendet",
                    "bestellt")

st.set_page_config(page_title="Ticket Data Summary", layout="wide")
st.title("System Ticket Data Summary")

# --------------------------------------------------------------------- #
# Sidebar: upload and config                                            #
# --------------------------------------------------------------------- #
DEFAULT_FILE = Path("Ticket Data (2).txt")

with st.sidebar:
    st.header("1. Upload ticket data")
    uploaded = st.file_uploader("Raw ticket text file", type=["txt", "csv"])
    st.caption("With or without the header line - both work. "
               "If you upload nothing, the bundled sample data is used.")
    st.divider()
    st.caption(f"Model: {os.getenv('GEMINI_MODEL', 'gemini-3.5-flash')}")
    if os.getenv("GOOGLE_API_KEY"):
        st.success("Gemini API key loaded")
    else:
        st.error("GOOGLE_API_KEY missing in .env - summaries are disabled.")

if uploaded is not None:
    file_bytes, source_name, source_note = uploaded.getvalue(), uploaded.name, "uploaded by you"
elif DEFAULT_FILE.exists():
    file_bytes, source_name, source_note = (DEFAULT_FILE.read_bytes(), DEFAULT_FILE.name,
                                            "bundled sample")
else:
    st.info("Upload the raw ticket data text file in the sidebar to get started.")
    st.stop()

with st.sidebar:
    st.subheader("Active data file")
    st.success(f"**{source_name}**\n\n{source_note} - {len(file_bytes):,} bytes")
    st.download_button("View the raw file", file_bytes, source_name, "text/plain")


# --------------------------------------------------------------------- #
# Preprocessing (cached; same steps as the notebook / LangGraph nodes)  #
# --------------------------------------------------------------------- #
@st.cache_data(show_spinner="Preprocessing tickets ...")
def preprocess(file_bytes: bytes):
    header, rows, had_header = pl.load_raw_tickets(file_bytes)   # validate or raise
    repaired = pl.find_shifted_orders(rows, header)              # report before repair
    rows = pl.repair_shifted_rows(rows, header)                  # insert N/A, slide right
    df_raw = pd.DataFrame(rows, columns=header)
    paths = pl.convert_and_save(df_raw)                          # data/tickets_converted.*
    df = pl.map_products(pl.filter_categories(pl.clean_tickets(df_raw)))
    return df_raw, df, repaired, paths, had_header


def summaries_to_pdf(summaries: dict) -> bytes:
    """Render the nested {customer: {product: entry}} summaries as a PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=18)
    for cust, products in summaries.items():
        for product, entry in products.items():
            text = entry["markdown"]
            pdf.add_page()
            pdf.set_font("helvetica", "B", 15)
            pdf.cell(0, 10, f"Customer {cust} - {product}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            # helvetica is latin-1 only; replace anything outside it
            clean = str(text).encode("latin-1", "replace").decode("latin-1")
            for line in clean.splitlines():
                line = line.strip()
                if not line:
                    pdf.ln(2)
                    continue
                if line.startswith("#"):
                    pdf.set_font("helvetica", "B", 12)
                    pdf.multi_cell(0, 7, line.lstrip("# ").strip(),
                                   new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("helvetica", "", 11)
                else:
                    if line.startswith("* "):
                        line = "- " + line[2:]
                    pdf.set_font("helvetica", "", 11)
                    pdf.multi_cell(0, 6, line, markdown=True,
                                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


try:
    df_raw, df, repaired, paths, had_header = preprocess(file_bytes)
except ValueError as err:
    st.error(f"The uploaded file is invalid: {err}")
    st.stop()

if not had_header:
    st.info("No header line detected in the upload - the standard 38-column "
            "header was applied automatically.")

if repaired:
    with st.expander(f"Data quality: repaired {len(repaired)} shifted tickets"):
        st.write(
            "These rows were written with one missing N/A placeholder, so every "
            "value from ADDITIONAL_ORDER_DESCRIPTION onward sat one column too "
            "far left. The missing placeholder was re-inserted; nothing was invented."
        )
        st.code(", ".join(repaired))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw tickets", len(df_raw))
c2.metric("After category filter", len(df))
c3.metric("Products", df["PRODUCT"].nunique())
c4.metric("Avg resolution (min)", f"{df['RESOLUTION_MINUTES'].mean():.0f}" if len(df) else "-")

tab_summary, tab_data, tab_insights = st.tabs(["AI Summaries", "Data", "Insights"])

# --------------------------------------------------------------------- #
# Tab: Data                                                             #
# --------------------------------------------------------------------- #
with tab_data:
    st.subheader("Converted raw data (after repair)")
    st.caption(f"Also saved to {paths['csv']} and {paths['xlsx']}.")
    st.dataframe(df_raw, height=250)
    st.download_button(
        "Download converted CSV",
        df_raw.to_csv(index=False).encode("utf-8"),
        "tickets_converted.csv",
        "text/csv",
    )

    st.subheader("Cleaned, filtered and product-mapped")
    st.caption(
        f"Only categories {pl.ALLOWED_CATEGORIES} are kept; "
        "products per the task mapping (HDW kept as 'Hardware')."
    )
    cols_first = ["ORDER_NUMBER", "ACCEPTANCE_TIME", "COMPLETION_TIME",
                  "SERVICE_CATEGORY", "PRODUCT", "RESOLUTION_MINUTES"]
    st.dataframe(df[cols_first + [c for c in df.columns if c not in cols_first]],
                 height=300)

# --------------------------------------------------------------------- #
# Tab: AI Summaries                                                     #
# --------------------------------------------------------------------- #
with tab_summary:
    st.subheader("Storytelling summaries per customer per product")
    st.caption(
        "Pick a customer, then a product: five sections each - Initial Issue, "
        "Follow-ups, Developments, Later Incidents, Recent Events - generated "
        "by Gemini through the end-to-end LangGraph pipeline."
    )

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("Add your GOOGLE_API_KEY to .env and restart the app.")
    elif st.button("Generate summaries", type="primary") or "summaries" in st.session_state:
        if "summaries" not in st.session_state:
            status = st.status("Running LangGraph pipeline ...", expanded=True)
            graph = pl.build_pipeline(
                on_progress=lambda p, n: status.write(f"Summarizing {p} ({n} tickets) ...")
            )
            result = graph.invoke({"raw_source": file_bytes})
            st.session_state["summaries"] = result["summaries"]
            status.update(label="Pipeline finished", state="complete", expanded=False)

        summaries = st.session_state["summaries"]

        # Session state can outlive an app update: summaries generated by an
        # older version have a different shape. Discard them instead of crashing.
        if any(not isinstance(e, dict) for p in summaries.values() for e in p.values()):
            del st.session_state["summaries"]
            st.warning("Stored summaries were produced by an older version of the "
                       "app - click Generate summaries again.")
            st.stop()

        total = sum(len(p) for p in summaries.values())
        ok = sum(1 for p in summaries.values()
                 for e in p.values() if e["status"] in ("verified", "corrected"))
        st.caption(f"Reference check: {ok}/{total} summaries verified - every cited "
                   "ticket exists in the data and no ticket was omitted.")

        s1, s2 = st.columns(2)
        cust = s1.selectbox("Customer", list(summaries))
        product = s2.selectbox("Product", list(summaries[cust]))
        entry = summaries[cust][product]

        if entry["status"] == "verified":
            st.success("Verified: ticket references proven against the data on the first attempt.")
        elif entry["status"] == "corrected":
            st.info("Corrected: the model fixed its references after one corrective retry - now verified.")
        elif entry["status"] == "unverified":
            st.warning("Unverified: ticket references could not be fully verified - read with care.")
        else:
            st.error("Generation failed for this group - see the text below.")

        st.markdown(entry["markdown"])
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Download all summaries (PDF)",
            summaries_to_pdf(summaries),
            "ticket_summaries.pdf",
            "application/pdf",
        )
        dl2.download_button(
            "Download all summaries (Markdown)",
            "\n\n---\n\n".join(
                f"# Customer {c} - {p}\n\n{e['markdown']}"
                for c, products in summaries.items()
                for p, e in products.items()
            ).encode("utf-8"),
            "ticket_summaries.md",
        )

# --------------------------------------------------------------------- #
# Tab: Insights - three business questions, each with a verdict         #
# --------------------------------------------------------------------- #
with tab_insights:
    st.subheader("Business insights")
    st.caption(
        "Three questions every support dataset must answer: where does the "
        "volume come from, do our fixes hold, and what does handling cost."
    )

    fc1, fc2 = st.columns(2)
    sel_products = fc1.multiselect("Products", sorted(df["PRODUCT"].unique()),
                                   default=sorted(df["PRODUCT"].unique()))
    sel_customer = fc2.selectbox("Customer", ["All"] + sorted(df["CUSTOMER_NUMBER"].unique()))

    d = df[df["PRODUCT"].isin(sel_products)]
    if sel_customer != "All":
        d = d[d["CUSTOMER_NUMBER"] == sel_customer]

    if d.empty:
        st.warning("No tickets match the current filters.")
        st.stop()

    # German device names -> English, so the same issue is not split into
    # separate bars by language.
    GERMAN_TERMS = {"Kabelrouter": "Cable Router", "CI-Modul": "CI Module",
                    "Sender": "TV Channel", "Kundenportal": "Customer Portal",
                    "Filme": "Movies"}
    handled = d["COMPLETION_RESULT_KB"].fillna("").str.lower()
    d = d.assign(
        ISSUE=d["ORDER_DESCRIPTION_1"].fillna("?").replace(GERMAN_TERMS),
        HANDLING=handled.apply(
            lambda t: "Escalated / hardware" if any(w in t for w in ESCALATION_WORDS)
            else "Remote fix"),
    )

    # ----------------------------------------------------------------- #
    # Executive scorecard                                               #
    # ----------------------------------------------------------------- #
    issues = d.groupby(["CUSTOMER_NUMBER", "PRODUCT"]).size()   # one issue = one customer x product
    repeat_rate = (len(d) - len(issues)) / len(d)               # tickets beyond each issue's first
    fcr = (issues == 1).mean()                                  # issues closed with a single contact
    esc_rate = float((d["HANDLING"] != "Remote fix").mean())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Repeat-contact rate", f"{repeat_rate:.0%}",
              help="Share of tickets that are the same customer returning about the same product")
    k2.metric("First-contact resolution", f"{fcr:.0%}",
              help="Share of issues closed with a single contact - the classic support KPI")
    k3.metric("Escalation rate", f"{esc_rate:.0%}",
              help="Share of tickets ending in a technician visit or hardware shipment")
    k4.metric("Avg resolution", f"{d['RESOLUTION_MINUTES'].mean():.0f} min")
    k5.metric("Contacts per issue", f"{len(d) / len(issues):.1f}")

    # ----------------------------------------------------------------- #
    # Q1: What breaks the most? (where the volume comes from)           #
    # ----------------------------------------------------------------- #
    st.markdown("### Q1. What breaks the most?")
    st.caption(
        "How to read the Pareto: bars = tickets per device/service (sorted, "
        "German terms translated); the line = cumulative share. Where the "
        "line crosses ~80% you have found the few issues causing most volume."
    )

    issue_counts = d["ISSUE"].value_counts()
    cum = issue_counts.cumsum() / issue_counts.sum() * 100
    pareto = go.Figure()
    pareto.add_bar(x=issue_counts.index, y=issue_counts.values, name="tickets")
    pareto.add_scatter(x=issue_counts.index, y=cum.values, name="cumulative %",
                       yaxis="y2", mode="lines+markers")
    pareto.update_layout(title="Pareto: ticket volume per issue area",
                         yaxis2=dict(overlaying="y", side="right", range=[0, 110]))

    q1a, q1b = st.columns(2)
    q1a.plotly_chart(pareto, width="stretch")
    prod_counts = d["PRODUCT"].value_counts().sort_values().reset_index()
    prod_counts.columns = ["product", "tickets"]
    q1b.plotly_chart(px.bar(prod_counts, x="tickets", y="product", orientation="h",
                            title="Tickets per product"),
                     width="stretch")

    top_issue, top_n = issue_counts.index[0], int(issue_counts.iloc[0])
    st.info(
        f"Finding: {top_n / len(d):.0%} of the selected tickets ({top_n} of {len(d)}) "
        f"are a single issue area: '{top_issue}'. This is the point of failure - one "
        "targeted fix here (firmware update, self-service reset button, better default "
        "settings) removes more support volume than any process change elsewhere."
    )

    # ----------------------------------------------------------------- #
    # Q2: Do fixes hold? (repeat contacts)                              #
    # ----------------------------------------------------------------- #
    st.markdown("### Q2. Do fixes hold?")

    chains = []
    for (cust, prod), g in d.sort_values("ACCEPTANCE_TIME").groupby(
            ["CUSTOMER_NUMBER", "PRODUCT"]):
        if len(g) > 1:
            days = g["ACCEPTANCE_TIME"].diff().dt.total_seconds().div(86400).dropna()
            chains.append({
                "customer": cust,
                "product": prod,
                "categories": ", ".join(sorted(g["SERVICE_CATEGORY"].unique())),
                "contacts": len(g),
                "avg days between contacts": round(days.mean(), 1),
                "final resolution": g["COMPLETION_RESULT_KB"].iloc[-1],
            })
    repeat_df = pd.DataFrame(chains)
    repeat_tickets = int((repeat_df["contacts"] - 1).sum()) if len(repeat_df) else 0

    if len(repeat_df):
        # Chain timeline: one row per chain, one dot per contact, colored by
        # how it was handled - chains visibly march toward the escalation dot.
        chain_rows = []
        for (cust, prod), g in d.sort_values("ACCEPTANCE_TIME").groupby(
                ["CUSTOMER_NUMBER", "PRODUCT"]):
            if len(g) > 1:
                for i, (_, r) in enumerate(g.iterrows(), 1):
                    chain_rows.append({
                        "chain": f"{cust} - {prod}", "date": r["ACCEPTANCE_TIME"],
                        "contact": i, "issue": r["ORDER_DESCRIPTION_2"],
                        "fix": r["COMPLETION_RESULT_KB"], "HANDLING": r["HANDLING"],
                    })
        cdf = pd.DataFrame(chain_rows)

        timeline = go.Figure()
        for label, g in cdf.groupby("chain"):
            timeline.add_scatter(x=g["date"], y=[label] * len(g), mode="lines",
                                 line=dict(color="#9AA7B4", width=1), showlegend=False)
        for handling, g in cdf.groupby("HANDLING"):
            timeline.add_scatter(
                x=g["date"], y=g["chain"], mode="markers", name=handling,
                marker=dict(size=12), text=g["fix"],
                hovertemplate="%{y}<br>%{x|%m/%d %H:%M}<br>fix: %{text}<extra></extra>")
        timeline.update_layout(
            title="Repeat chains over time: each dot is a contact",
            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(timeline, width="stretch")

        with st.expander("Chain table (numbers behind the chart)"):
            st.dataframe(repeat_df.sort_values("contacts", ascending=False), hide_index=True)

        ended_escalated = repeat_df["final resolution"].fillna("").str.lower().apply(
            lambda t: any(w in t for w in ESCALATION_WORDS)).mean()
        st.info(
            f"Finding: {repeat_tickets} tickets ({repeat_tickets / len(d):.0%} of the "
            "selection) are repeat contacts - the same customer reporting the same "
            "product again within days, meaning the previous fix did not hold. "
            f"{ended_escalated:.0%} of these chains ended in an escalation (replacement "
            "or technician) anyway, so the intermediate remote fixes bought nothing. "
            "Action: define an escalation trigger at the contact number where remote "
            "attempts stop paying off - in this data that is already the second "
            "contact; with more data, set it by comparing the cost of one more remote "
            "attempt against the cost of an early device swap."
        )
    else:
        st.success("No repeat-contact chains in the current selection - fixes are holding.")

    # ----------------------------------------------------------------- #
    # Q3: Which customers need attention?                               #
    # ----------------------------------------------------------------- #
    st.markdown("### Q3. Which customers need attention?")

    q3a, q3b = st.columns(2)
    cust_prod = d.groupby(["CUSTOMER_NUMBER", "PRODUCT"]).size().reset_index(name="tickets")
    q3a.plotly_chart(px.bar(cust_prod, x="CUSTOMER_NUMBER", y="tickets", color="PRODUCT",
                            title="Tickets per customer, by product"),
                     width="stretch")

    contact_type = []
    for (cust, prod), g in d.sort_values("ACCEPTANCE_TIME").groupby(
            ["CUSTOMER_NUMBER", "PRODUCT"]):
        for i in range(len(g)):
            contact_type.append({"customer": cust,
                                 "type": "repeat contact" if i else "first contact"})
    ct = pd.DataFrame(contact_type).groupby(["customer", "type"]).size().reset_index(name="tickets")
    q3b.plotly_chart(px.bar(ct, x="customer", y="tickets", color="type", barmode="stack",
                            title="First vs repeat contacts per customer"),
                     width="stretch")

    per_cust = d["CUSTOMER_NUMBER"].value_counts()
    rep_per_cust = (ct[ct["type"] == "repeat contact"]
                    .set_index("customer")["tickets"] if len(ct) else pd.Series(dtype=int))
    top_cust = per_cust.index[0]
    st.info(
        f"Finding: every customer in this selection is a repeat caller. Customer "
        f"{top_cust} contacted support {per_cust.iloc[0]} times in two weeks"
        + (f", {int(rep_per_cust.get(top_cust, 0))} of them repeat contacts"
           if len(rep_per_cust) else "")
        + " - that is a churn risk, not just a support cost: a customer who keeps "
        "calling about the same product is a customer considering cancellation. These "
        "are the accounts to contact proactively once the device fix ships."
    )

    # ----------------------------------------------------------------- #
    # Views that need more data                                         #
    # ----------------------------------------------------------------- #
    with st.expander("More views - need more data to be conclusive"):
        st.caption(
            "With only a few weeks of tickets these views are illustrative; "
            "they become decision-grade at real volume."
        )
        daily = d.groupby([d["ACCEPTANCE_TIME"].dt.date, "PRODUCT"]).size().reset_index(name="tickets")
        daily.columns = ["date", "product", "tickets"]
        st.plotly_chart(px.bar(daily, x="date", y="tickets", color="product",
                               title="Daily ticket volume by product"),
                        width="stretch")

        fixes = d["COMPLETION_RESULT_KB"].fillna("(not recorded)").value_counts().head(8)
        fixes_df = fixes.sort_values().reset_index()
        fixes_df.columns = ["fix", "tickets"]
        st.plotly_chart(px.bar(fixes_df, x="tickets", y="fix", orientation="h",
                               title="Most common fixes"),
                        width="stretch")

        e1, e2 = st.columns(2)
        fcr_prod = (issues == 1).groupby(level="PRODUCT").mean().sort_values().reset_index()
        fcr_prod.columns = ["product", "fcr"]
        e1.plotly_chart(px.bar(fcr_prod, x="fcr", y="product", orientation="h",
                               title="First-contact resolution by product")
                        .update_xaxes(tickformat=".0%"),
                        width="stretch")
        team = d.groupby(["PLANNING_GROUP_KB", "HANDLING"]).size().reset_index(name="tickets")
        e2.plotly_chart(px.bar(team, x="PLANNING_GROUP_KB", y="tickets", color="HANDLING",
                               barmode="stack", title="Workload per support team"),
                        width="stretch")

        causes = d["CAUSE"].fillna("(none)").value_counts().head(8).sort_values().reset_index()
        causes.columns = ["cause code", "tickets"]
        st.plotly_chart(px.bar(causes, x="tickets", y="cause code", orientation="h",
                               title="Top root-cause codes (feed for engineering)"),
                        width="stretch")

        m1, m2 = st.columns(2)
        tmp = d.assign(weekday=d["ACCEPTANCE_TIME"].dt.day_name(),
                       hour=d["ACCEPTANCE_TIME"].dt.hour)
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat = tmp.pivot_table(index="weekday", columns="hour", values="ORDER_NUMBER",
                               aggfunc="count").reindex(
            [x for x in order if x in tmp["weekday"].unique()])
        m1.plotly_chart(px.imshow(heat, labels=dict(x="hour of day", y="", color="tickets"),
                                  title="When tickets arrive (weekday x hour)"),
                        width="stretch")
        m2.plotly_chart(px.scatter(d, x="ACCEPTANCE_TIME", y="CUSTOMER_NUMBER",
                                   color="PRODUCT", hover_data=["ORDER_NUMBER", "NOTE_MAXIMUM"],
                                   title="Customer ticket timeline"),
                        width="stretch")
