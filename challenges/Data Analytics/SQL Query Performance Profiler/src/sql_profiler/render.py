"""Draw a plan: an SVG tree, an HTML report around it, and a plain-text tree for the terminal.

The tree is drawn the way plans are usually read: the result at the top, tables at the bottom, data
flowing upward. Box colour is the operator's share of the work, edge thickness is how many rows flow
along it (log scale), and a box's outline turns amber or red when a finding points at it.

Everything that comes from the query (SQL, table names, filter text) is HTML-escaped: a profile of an
untrusted query must not be able to inject markup into the report.
"""

from __future__ import annotations

import math
from html import escape

from sql_profiler.plan import Plan, PlanNode
from sql_profiler.rules import Finding

BOX_W, BOX_H = 200, 86
GAP_X, GAP_Y = 24, 46
MARGIN = 20

STYLE = """
:root{--bg:#ffffff;--fg:#1c2430;--muted:#5b6678;--card:#f5f7fa;--line:#c9d1dc;--edge:#8a96a8;
--hot:#e8590c;--warn:#d97706;--crit:#c92a2a;--info:#3b82c4}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#10151c;--fg:#e6ebf2;
--muted:#9aa6b8;--card:#19212b;--line:#334052;--edge:#6b7a90;--hot:#ff8a4c;--warn:#f0a43a;
--crit:#ff6b6b;--info:#6cb0f0}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;
max-width:1180px;padding:16px}
h1{font-size:1.3rem;margin:.2em 0}h2{font-size:1.05rem;margin:1.6em 0 .5em}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;
overflow:auto;white-space:pre-wrap}
.stats{display:flex;flex-wrap:wrap;gap:10px}.stat{background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:6px 12px}.stat b{display:block;font-size:1.15rem}
.stat span{color:var(--muted);font-size:.8rem}
.finding{border-left:4px solid var(--info);background:var(--card);padding:8px 12px;margin:8px 0;
border-radius:0 6px 6px 0}.finding.warn{border-color:var(--warn)}.finding.critical{border-color:var(--crit)}
.finding h3{margin:0;font-size:.98rem}.finding p{margin:.3em 0}.tag{font-size:.72rem;
text-transform:uppercase;color:var(--muted)}
.plan{overflow:auto;border:1px solid var(--line);border-radius:6px;background:var(--card)}
table{border-collapse:collapse;width:100%}td,th{padding:4px 8px;border-bottom:1px solid var(--line);
text-align:right;font-variant-numeric:tabular-nums}td:first-child,th:first-child{text-align:left}
svg text{fill:var(--fg);font:12px system-ui,sans-serif}svg .muted{fill:var(--muted)}
svg text.name{font-weight:600}
"""


def _fmt_rows(n: int | None) -> str:
    return "-" if n is None else f"{n:,}"


def _fmt_time(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:,.1f} ms" if ms >= 0.1 else f"{ms * 1000:,.0f} us"


def _detail(node: PlanNode) -> str:
    """The one line under an operator's name that says what it works on."""
    if node.name == "TABLE_SCAN":
        return node.filters or "no static filter"
    if node.conditions:
        return f"{node.join_type or ''} {node.conditions}".strip()
    for key in ("Order By", "Groups", "Top", "Projections"):
        value = node.info.get(key)
        if value:
            return ", ".join(value) if isinstance(value, list) else str(value)
    return ""


def _clip(text: str, width: int = 30) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _layout(root: PlanNode) -> tuple[dict[int, float], float]:
    """x centre per node id and total width, children left to right under their parent."""
    widths: dict[int, float] = {}

    def measure(n: PlanNode) -> float:
        own = BOX_W + GAP_X
        widths[n.id] = max(own, sum(measure(c) for c in n.children))
        return widths[n.id]

    total = measure(root)
    centres: dict[int, float] = {}

    def place(n: PlanNode, left: float) -> None:
        centres[n.id] = left + widths[n.id] / 2
        x = left
        for c in n.children:
            place(c, x)
            x += widths[c.id]

    place(root, 0)
    return centres, total


def _heat(share: float) -> str:
    """Pale to hot orange by share of work; opacity keeps the text readable in both themes."""
    return f"color-mix(in srgb, var(--hot) {min(share, 1.0) * 70:.0f}%, var(--card))"


def plan_svg(plan: Plan, findings: list[Finding] | tuple[Finding, ...] = ()) -> str:
    centres, total_w = _layout(plan.root)
    depth = max(n.depth for n in plan.nodes)
    height = MARGIN * 2 + (depth + 1) * BOX_H + depth * GAP_Y
    width = total_w + MARGIN * 2
    worst: dict[int, str] = {}
    for f in findings:  # findings arrive most severe first, keep the first per node
        if f.node_id is not None and f.severity != "info":
            worst.setdefault(f.node_id, f.severity)
    max_rows = max((n.rows for n in plan.nodes), default=1) or 1

    def pos(n: PlanNode) -> tuple[float, float]:
        return centres[n.id] + MARGIN - BOX_W / 2, MARGIN + n.depth * (BOX_H + GAP_Y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" role="img" aria-label="Query plan">'
    ]
    for n in plan.nodes:  # edges first so boxes sit on top
        px, py = pos(n)
        for c in n.children:
            cx, cy = pos(c)
            thick = 1 + 7 * math.log10(c.rows + 1) / math.log10(max_rows + 1)
            x1, y1 = cx + BOX_W / 2, cy
            x2, y2 = px + BOX_W / 2, py + BOX_H
            mid = (y1 + y2) / 2
            parts.append(
                f'<path d="M{x1:.1f},{y1:.1f} C{x1:.1f},{mid:.1f} {x2:.1f},{mid:.1f} {x2:.1f},{y2:.1f}" '
                f'fill="none" stroke="var(--edge)" stroke-opacity="0.55" stroke-width="{thick:.1f}">'
                f"<title>{escape(_fmt_rows(c.rows))} rows</title></path>"
            )
    for n in plan.nodes:
        x, y = pos(n)
        outline = {"critical": "var(--crit)", "warn": "var(--warn)"}.get(
            worst.get(n.id, ""), "var(--line)"
        )
        stroke_w = 3 if n.id in worst else 1
        name = f"{n.label} {n.table}" if n.table else n.label
        q = n.q_error
        est = f" (est {_fmt_rows(n.est_rows)})" if n.est_rows is not None else ""
        tip = f"{name}\nrows {_fmt_rows(n.rows)}{est}\nself time {_fmt_time(n.self_time_s)} ({n.share:.0%})"
        if n.rows_scanned:
            tip += f"\nrows scanned {_fmt_rows(n.rows_scanned)}"
        for key, value in n.info.items():
            if key not in {"Estimated Cardinality"}:
                tip += f"\n{key}: {value if not isinstance(value, list) else ', '.join(map(str, value))}"
        bar_w = max(1.5, (BOX_W - 16) * min(n.share, 1.0))
        flag = (
            f"  {q:,.0f}x off"
            if q is not None
            and n.estimate_is_comparable
            and q >= 10
            and max(n.est_rows or 0, n.rows) >= 1000
            else ""
        )
        parts.append(
            f'<g transform="translate({x:.1f},{y:.1f})"><title>{escape(tip)}</title>'
            f'<rect width="{BOX_W}" height="{BOX_H}" rx="7" fill="{_heat(n.share)}" '
            f'stroke="{outline}" stroke-width="{stroke_w}"/>'
            f'<text x="10" y="19" class="name">{escape(_clip(name, 26))}</text>'
            f'<text x="10" y="36" class="muted">{escape(_clip(_detail(n)))}</text>'
            f'<text x="10" y="53">rows {escape(_fmt_rows(n.rows))}{escape(est and _clip(est, 24))}</text>'
            f'<text x="10" y="69">{escape(_fmt_time(n.self_time_s))} · {n.share:.0%}{escape(flag)}</text>'
            f'<rect x="8" y="{BOX_H - 8}" width="{bar_w:.1f}" height="4" rx="2" fill="var(--hot)"/></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def text_tree(plan: Plan) -> str:
    """The plan as indented text: operator, rows (estimate), self time and share."""
    lines = []
    for n in plan.nodes:
        name = f"{n.label} {n.table}" if n.table else n.label
        est = f" (est {_fmt_rows(n.est_rows)})" if n.est_rows is not None else ""
        lines.append(
            f"{'  ' * n.depth}{name}  rows={_fmt_rows(n.rows)}{est}  {_fmt_time(n.self_time_s)}  {n.share:.0%}"
        )
    return "\n".join(lines)


def operator_rows(plan: Plan) -> list[dict]:
    """Operators sorted by self time, as plain dicts for a table."""
    return [
        {
            "operator": n.label,
            "on": n.table or "",
            "rows": n.rows,
            "est_rows": n.est_rows,
            "q_error": None if n.q_error is None else round(n.q_error, 1),
            "rows_scanned": n.rows_scanned,
            "self_ms": round(n.self_time_s * 1000, 3),
            "share_pct": round(n.share * 100, 1),
        }
        for n in sorted(plan.nodes, key=lambda n: -n.self_time_s)
    ]


def report_html(
    plan: Plan, findings: list[Finding], title: str = "Query profile"
) -> str:
    """A single self-contained HTML page: stats, findings, the plan drawing and an operator table."""
    stats = [
        (f"{plan.latency_s * 1000:,.1f} ms", "wall clock"),
        (f"{plan.cpu_s * 1000:,.1f} ms", f"CPU ({plan.parallelism:.1f}x parallel)"),
        (f"{plan.rows_returned:,}", "rows returned"),
        (f"{plan.peak_memory_bytes / 1e6:,.0f} MB", "peak buffer memory"),
        (str(len(plan.nodes)), "operators"),
    ]
    stat_html = "".join(
        f'<div class="stat"><b>{escape(v)}</b><span>{escape(k)}</span></div>'
        for v, k in stats
    )
    if findings:
        find_html = "".join(
            f'<div class="finding {escape(f.severity)}"><span class="tag">{escape(f.severity)} · {escape(f.rule)}</span>'
            f"<h3>{escape(f.title)}</h3><p>{escape(f.detail)}</p><p><b>Try:</b> {escape(f.suggestion)}</p></div>"
            for f in findings
        )
    else:
        find_html = "<p>No findings: nothing in this plan crosses a threshold.</p>"
    rows_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape('' if v is None else (f'{v:,}' if isinstance(v, int) else str(v)))}</td>"
            for v in row.values()
        )
        + "</tr>"
        for row in operator_rows(plan)
    )
    head = "".join(f"<th>{escape(k)}</th>" for k in operator_rows(plan)[0])
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{STYLE}</style></head><body>"
        f"<h1>{escape(title)}</h1><pre>{escape(plan.sql)}</pre><div class=stats>{stat_html}</div>"
        f"<h2>Findings</h2>{find_html}<h2>Plan</h2><p class=tag>result at the top; box colour = share of work; "
        f"line thickness = rows; outline = finding</p><div class=plan>{plan_svg(plan, findings)}</div>"
        f"<h2>Operators by self time</h2><table><tr>{head}</tr>{rows_html}</table>"
        f"<p class=tag>Self time is CPU time summed over threads, so it can exceed wall clock.</p></body></html>"
    )
