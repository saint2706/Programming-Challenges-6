"""A self-contained HTML report: the latest runs, per-task attempts, the DAG as an inline SVG and the
row counts of every content table.

Nothing is loaded from elsewhere and there is no script. Every value taken from the database or the
task graph is HTML-escaped, and a Content-Security-Policy meta tag forbids scripts and remote loads,
so a task name or an error message cannot run code in the browser.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from html import escape
from pathlib import Path

import duckdb

from star_warehouse.dag import Task
from star_warehouse.warehouse import CONTENT_TABLES

CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"

STATUS_FILL = {
    "succeeded": "#d7f0d2",
    "ok": "#d7f0d2",
    "failed": "#f6c9c4",
    "error": "#f6c9c4",
    "skipped": "#e6e6e6",
    "interrupted": "#fbe7b5",
}

CSS = """
body { font-family: system-ui, sans-serif; margin: 16px; color: #1d1d1f; background: #fff; }
h1 { font-size: 1.4em; } h2 { font-size: 1.1em; margin-top: 1.6em; }
table { border-collapse: collapse; font-size: 0.9em; margin-top: 0.4em; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { padding: 1px 6px; border-radius: 4px; }
.scroll { overflow-x: auto; }
svg text { font-size: 12px; }
"""


def _pill(status: str) -> str:
    fill = STATUS_FILL.get(status, "#ffffff")
    return f'<span class="pill" style="background:{fill}">{escape(status)}</span>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def dag_svg(tasks: Sequence[Task], status: dict[str, str]) -> str:
    """Layered layout: a task sits one column right of its deepest upstream."""
    by_name = {t.name: t for t in tasks}
    depth: dict[str, int] = {}

    def level(name: str) -> int:
        if name not in depth:
            deps = by_name[name].deps
            depth[name] = 0 if not deps else 1 + max(level(d) for d in deps)
        return depth[name]

    for t in tasks:
        level(t.name)
    columns: dict[int, list[str]] = {}
    for t in tasks:  # declaration order inside a column
        columns.setdefault(depth[t.name], []).append(t.name)
    width = 170 * (max(columns) + 1) + 20
    height = 46 * max(len(c) for c in columns.values()) + 20
    pos: dict[str, tuple[int, int]] = {}
    for col, names in columns.items():
        for row, name in enumerate(names):
            pos[name] = (20 + col * 170, 20 + row * 46)
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Task dependency graph">'
    )
    marker = (
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#666"/></marker></defs>'
    )
    parts = [head, marker]
    for t in tasks:
        x2, y2 = pos[t.name]
        for dep in t.deps:
            x1, y1 = pos[dep]
            parts.append(
                f'<line x1="{x1 + 150}" y1="{y1 + 16}" x2="{x2}" y2="{y2 + 16}" '
                'stroke="#666" marker-end="url(#arrow)"/>'
            )
    for t in tasks:
        x, y = pos[t.name]
        fill = STATUS_FILL.get(status.get(t.name, ""), "#ffffff")
        parts.append(
            f'<rect x="{x}" y="{y}" width="150" height="32" rx="5" fill="{fill}" stroke="#444"/>'
            f'<text x="{x + 8}" y="{y + 20}">{escape(t.name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render(con: duckdb.DuckDBPyConnection, tasks: Sequence[Task], title: str) -> str:
    runs = con.execute(
        "SELECT run_id, interval_start, interval_end, trigger, status, started_at, finished_at "
        "FROM ops.run ORDER BY started_at DESC LIMIT 20"
    ).fetchall()
    latest_status: dict[str, str] = {}
    task_rows: list[list[str]] = []
    if runs:
        latest = runs[0][0]
        for task, attempt, status, started, finished, error in con.execute(
            "SELECT task, attempt, status, started_at, finished_at, error FROM ops.task_run "
            "WHERE run_id = ? ORDER BY started_at, task, attempt",
            [latest],
        ).fetchall():
            latest_status[task] = status
            task_rows.append(
                [
                    escape(task),
                    escape(str(attempt)),
                    _pill(status),
                    escape(_fmt(started)),
                    escape(_fmt(finished)),
                    escape(error or ""),
                ]
            )
    run_rows = [
        [
            escape(run_id[:8]),
            escape(f"{start} to {end}"),
            escape(trigger),
            _pill(status),
            escape(_fmt(started)),
            escape(_fmt(finished)),
        ]
        for run_id, start, end, trigger, status, started, finished in runs
    ]
    counts = [
        [escape(table), f'<span class="num">{escape(_count(con, table))}</span>']
        for table in CONTENT_TABLES
    ]
    sections = [
        f"<h1>{escape(title)}</h1>",
        "<h2>Runs (latest 20)</h2>",
        _table(
            ["Run", "Interval", "Trigger", "Status", "Started", "Finished"], run_rows
        )
        if run_rows
        else "<p>No runs yet.</p>",
        "<h2>Lineage: task graph (latest run's status)</h2>",
        dag_svg(tasks, latest_status) if tasks else "<p>No tasks.</p>",
        "<h2>Attempts in the latest run</h2>",
        _table(["Task", "Attempt", "Status", "Started", "Finished", "Error"], task_rows)
        if task_rows
        else "<p>No task attempts yet.</p>",
        "<h2>Row counts</h2>",
        _table(["Table", "Rows"], counts),
    ]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{escape(CSP)}">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head><body>"
        + "".join(sections)
        + "</body></html>\n"
    )


def _count(con: duckdb.DuckDBPyConnection, table: str) -> str:
    # Table names come from CONTENT_TABLES, never from input.
    return str(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _fmt(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return "" if value is None else str(value)


def write(
    con: duckdb.DuckDBPyConnection, tasks: Sequence[Task], out: Path, title: str
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(con, tasks, title), encoding="utf-8", newline="\n")
    return out
