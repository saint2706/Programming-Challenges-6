"""Streamlit app: type a query, see its real execution plan, and get told what is wrong with it.

Run with:  uv run streamlit run src/sql_profiler/app.py
"""

from __future__ import annotations

import threading

import duckdb
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

from sql_profiler import compare as comparison
from sql_profiler import db, lab, render, rules
from sql_profiler.plan import profile

# one measurement at a time: two sessions timing queries together would corrupt each other's numbers
RUN_LOCK = threading.Lock()
STARTER = """SELECT i.country, round(sum(l.quantity * l.price), 2) AS revenue
FROM invoice_lines l
JOIN invoices i USING (invoice)
JOIN products p USING (stock_code)
WHERE strftime(i.invoice_date, '%Y') = '2011' AND p.description LIKE '%HEART%'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10"""
SEVERITY_ICON = {"critical": "🔴", "warn": "🟠", "info": "🔵"}
CASES = {c.id: c for c in lab.LAB}


@st.cache_resource(show_spinner="Loading the retail tables...")
def database():
    return db.connect()


def show_plan(plan, findings, height: int | None = None) -> None:
    svg = render.plan_svg(plan, findings)
    page = f"<style>{render.STYLE}</style><div class=plan style='border:0'>{svg}</div>"
    depth = max(n.depth for n in plan.nodes) + 1
    components.html(page, height=height or min(1100, 60 + depth * 135), scrolling=True)


def show_findings(findings) -> None:
    if not findings:
        st.success("No findings: nothing in this plan crosses a threshold.")
    for f in findings:
        with st.container(border=True):
            st.markdown(f"{SEVERITY_ICON[f.severity]} **{f.title}**  \n`{f.rule}`")
            st.write(f.detail)
            st.markdown(f"**Try:** {f.suggestion}")


def show_profile(plan) -> list:
    findings = rules.diagnose(plan)
    c = st.columns(4)
    c[0].metric("Wall clock", f"{plan.latency_s * 1000:,.1f} ms")
    c[1].metric(
        "CPU time",
        f"{plan.cpu_s * 1000:,.1f} ms",
        f"{plan.parallelism:.1f}x parallel",
        delta_color="off",
    )
    c[2].metric("Rows returned", f"{plan.rows_returned:,}")
    c[3].metric("Operators", len(plan.nodes))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Plan")
        st.caption(
            "Result at the top. Box colour: share of work. Line thickness: rows. Outline: a finding."
        )
        show_plan(plan, findings)
    with right:
        st.subheader(f"Findings ({len(findings)})")
        show_findings(findings)
    st.subheader("Operators by self time")
    st.dataframe(
        pl.DataFrame(render.operator_rows(plan)),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Plan as text"):
        st.code(render.text_tree(plan), language=None)
    return findings


def profile_tab() -> None:
    sql = st.text_area("SQL (one SELECT)", STARTER, height=190)
    if st.button("Profile", type="primary"):
        with RUN_LOCK:
            try:
                plan = profile(database().cursor(), sql, st.session_state["timeout"])
            except (ValueError, TimeoutError) as exc:
                st.error(str(exc))
                return
            except duckdb.Error as exc:  # a DuckDB binder/catalog error should be shown, not crash the page
                st.error(f"{type(exc).__name__}: {exc}")
                return
        st.session_state["last_plan"] = plan
    if "last_plan" in st.session_state:
        show_profile(st.session_state["last_plan"])


def compare_tab() -> None:
    case_id = st.selectbox(
        "Start from an experiment",
        ["(my own queries)", *CASES],
        format_func=lambda k: CASES[k].title if k in CASES else k,
    )
    case = CASES.get(case_id)
    if case:
        st.caption(case.idea + (f"  \n{case.note}" if case.note else ""))
    a_sql = st.text_area(
        "Query A (the original)",
        case.a_sql if case else STARTER,
        height=150,
        key=f"a_{case_id}",
    )
    b_sql = st.text_area(
        "Query B (the rewrite)",
        case.b_sql if case else STARTER,
        height=150,
        key=f"b_{case_id}",
    )
    repeats = st.slider("Timed runs each", 3, 15, 9)
    if st.button("Compare", type="primary"):
        with RUN_LOCK, st.spinner("Checking results, timing both, profiling both..."):
            try:
                st.session_state["comparison"] = comparison.compare(
                    database().cursor(),
                    a_sql,
                    b_sql,
                    repeats,
                    st.session_state["timeout"],
                )
            except (ValueError, TimeoutError) as exc:
                st.error(str(exc))
                return
            except duckdb.Error as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                return
    result = st.session_state.get("comparison")
    if not result:
        return
    if result.check.same:
        st.success(f"Same rows: {result.check.rows_a:,} identical rows.")
    else:
        st.warning(
            f"Not the same result: {result.check.reason} (A {result.check.rows_a:,} rows, B {result.check.rows_b:,}). A speedup means nothing unless the answers agree."
        )
    c = st.columns(3)
    c[0].metric("A median", f"{result.a.median_s * 1000:,.2f} ms")
    c[1].metric(
        "B median",
        f"{result.b.median_s * 1000:,.2f} ms",
        f"{result.speedup:.2f}x vs A",
        delta_color="off",
    )
    verdict = {
        "b_faster": "B is faster",
        "a_faster": "A is faster",
        "no_clear_difference": "No clear difference",
    }[result.verdict]
    c[2].metric(
        "Verdict",
        verdict,
        help="Claimed only when every run of one beats every run of the other.",
    )
    runs = pl.DataFrame(
        [
            {"query": "A", "run": i + 1, "ms": t * 1000}
            for i, t in enumerate(result.a.runs_s)
        ]
        + [
            {"query": "B", "run": i + 1, "ms": t * 1000}
            for i, t in enumerate(result.b.runs_s)
        ]
    )
    st.scatter_chart(runs, x="run", y="ms", color="query")
    st.write(
        "**Plan:** "
        + (
            "same operators on the same tables"
            if result.same_plan
            else "; ".join(comparison.plan_diff(result.plan_a, result.plan_b))
        )
    )
    st.write(
        f"**Rows read from tables:** A {result.plan_a.rows_scanned:,}, B {result.plan_b.rows_scanned:,}"
    )
    left, right = st.columns(2)
    for col, label, plan in ((left, "A", result.plan_a), (right, "B", result.plan_b)):
        with col:
            st.subheader(f"Plan {label}")
            findings = rules.diagnose(plan)
            show_plan(plan, findings)
            show_findings(findings)


def lab_tab() -> None:
    st.write(
        "Each row runs a query and its rewrite: same rows required (except where noted), "
        "then timed several times. Many come out as no difference, because DuckDB already fixes them."
    )
    if st.button("Run every experiment (about 30 seconds)"):
        with RUN_LOCK, st.spinner("Running..."):
            st.session_state["lab"] = lab.run_lab(database().cursor(), 7)
    for case, c in st.session_state.get("lab", []):
        verdict = {
            "b_faster": "B faster",
            "a_faster": "A faster",
            "no_clear_difference": "no clear difference",
        }[c.verdict]
        st.markdown(
            f"**{case.title}** · A {c.a.median_s * 1000:,.2f} ms · B {c.b.median_s * 1000:,.2f} ms · "
            f"{verdict} ({c.speedup:.2f}x) · same rows: {c.check.same} · plans {'same' if c.same_plan else 'differ'}"
        )


def schema_tab() -> None:
    con = database().cursor()
    for (name,) in con.execute("SHOW TABLES").fetchall():
        rows = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        st.markdown(f"**{name}** ({rows:,} rows)")
        st.dataframe(
            pl.from_arrow(con.execute(f"DESCRIBE {name}").to_arrow_table()).select(
                "column_name", "column_type"
            ),
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="SQL Query Profiler", layout="wide")
    st.title("SQL query performance profiler")
    with st.sidebar:
        st.caption(
            f"Tables loaded from `{db.data_dir().name}/` into in-memory DuckDB; file access is switched off."
        )
        st.slider("Stop a query after (seconds)", 5, 120, 30, key="timeout")
    tabs = st.tabs(["Profile a query", "Compare a rewrite", "Experiments", "Tables"])
    with tabs[0]:
        profile_tab()
    with tabs[1]:
        compare_tab()
    with tabs[2]:
        lab_tab()
    with tabs[3]:
        schema_tab()


main()
