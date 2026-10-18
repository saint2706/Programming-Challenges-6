"""Turning an engine result into something a person can read: a grid of text, CSV, Markdown, or Rich."""

from __future__ import annotations

import csv
import io
from typing import Any

from rich.table import Table as RichTable
from rich.text import Text


def fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return ""
    if percent:
        return f"{value * 100:.1f}%"
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e15:
        return f"{round(value):,}"
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"


def _row_label(header: dict, depth_fields: int) -> list[str]:
    """The text of a row header, one entry per row field."""
    labels = [l or "" for l in header["labels"]] + [""] * (
        depth_fields - len(header["labels"])
    )
    if header["kind"] == "total":
        labels = ["Grand Total"] + [""] * (depth_fields - 1)
    elif header["kind"] == "subtotal":
        d = header["depth"]
        labels[d - 1] = f"{labels[d - 1]} Total"
        labels[d:] = [""] * (depth_fields - d)
    return labels[:depth_fields]


def _col_label(header: dict) -> list[str]:
    if header["kind"] == "total":
        return ["Grand Total"]
    labels = [l or "" for l in header["labels"]]
    if header["kind"] == "subtotal":
        d = header["depth"]
        labels[d - 1] = f"{labels[d - 1]} Total"
        labels = labels[:d]
    return labels


def to_grid(result: dict, collapse_repeats: bool = False) -> list[list[str]]:
    """The pivot as rows of text cells: header rows first, then one row per row item."""
    row_depth = max(len(result["row_fields"]), 1)
    measures = result["measures"]
    multi = len(measures) > 1
    col_headers = [_col_label(c) for c in result["cols"]]
    col_levels = max(
        [len(h) for h in col_headers]
        + [len(result["col_fields"]), 1 if not multi else 0]
    )
    cells: list[tuple[list[str], int]] = []  # (column header path, measure index)
    for ci, header in enumerate(col_headers):
        for mi in range(len(measures) if multi else 1):
            cells.append((header, mi, ci))  # type: ignore[arg-type]

    out: list[list[str]] = []
    top_levels = col_levels + (1 if multi else 0)
    for level in range(top_levels):
        line = [""] * row_depth
        if level == 0:
            line = [
                " / ".join(result["row_fields"]) if result["row_fields"] else ""
            ] + [""] * (row_depth - 1)
        for header, mi, _ in cells:
            if multi and level == top_levels - 1:
                line.append(measures[mi]["label"])
            else:
                line.append(header[level] if level < len(header) else "")
        out.append(line)
    previous: list[str] = [""] * row_depth
    for ri, header in enumerate(result["rows"]):
        labels = _row_label(header, row_depth)
        shown = list(labels)
        if collapse_repeats and header["kind"] == "leaf":
            for i in range(row_depth):
                if labels[: i + 1] == previous[: i + 1]:
                    shown[i] = ""
                else:
                    break
        if header["kind"] == "leaf":
            previous = labels
        line = shown
        for _, mi, ci in cells:
            m = measures[mi]
            line.append(fmt(result["cells"][ri][ci][mi], m["percent"]))
        out.append(line)
    return out


def header_height(result: dict) -> int:
    multi = len(result["measures"]) > 1
    levels = max(
        [len(_col_label(c)) for c in result["cols"]]
        + [len(result["col_fields"]), 1 if not multi else 0]
    )
    return levels + (1 if multi else 0)


def to_csv(result: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(to_grid(result))
    return buf.getvalue()


def to_markdown(result: dict) -> str:
    grid = to_grid(result, collapse_repeats=True)
    h = header_height(result)
    width = len(grid[0])
    header = grid[:h]
    # a multi-line header is flattened with " / " so every column has one title
    titles = [
        " / ".join(p for p in (row[i] for row in header) if p) for i in range(width)
    ]
    body = grid[h:]
    widths = [
        max(len(titles[i]), *(len(r[i]) for r in body), 3)
        if body
        else max(len(titles[i]), 3)
        for i in range(width)
    ]
    numeric = [i >= max(len(result["row_fields"]), 1) for i in range(width)]

    def line(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                c.rjust(w) if numeric[i] else c.ljust(w)
                for i, (c, w) in enumerate(zip(cells, widths))
            )
            + " |"
        )

    sep = (
        "| "
        + " | ".join(
            ("-" * (w - 1) + ":") if numeric[i] else "-" * w
            for i, w in enumerate(widths)
        )
        + " |"
    )
    return "\n".join([line(titles), sep, *(line(r) for r in body)]) + "\n"


def to_rich(result: dict, title: str | None = None) -> RichTable:
    grid = to_grid(result, collapse_repeats=True)
    h = header_height(result)
    width = len(grid[0])
    row_depth = max(len(result["row_fields"]), 1)
    table = RichTable(title=title, show_lines=False, header_style="bold")
    titles = [
        " / ".join(p for p in (row[i] for row in grid[:h]) if p) for i in range(width)
    ]
    for i, t in enumerate(titles):
        table.add_column(
            t, justify="left" if i < row_depth else "right", overflow="fold"
        )
    for ri, line in enumerate(grid[h:]):
        kind = result["rows"][ri]["kind"] if result["rows"] else "leaf"
        style = "bold" if kind != "leaf" else None
        table.add_row(*[Text(c) for c in line], style=style)
    return table


def crosstab_tables(ct: dict) -> list[Any]:
    """Rich renderables for a cross-tab: counts with percentages, then the statistics."""
    out: list[Any] = []
    counts = RichTable(
        title=f"{ct['row_field']} x {ct['col_field']}  (n = {ct['n']:,.0f})",
        header_style="bold",
    )
    counts.add_column(ct["row_field"])
    for label in ct["col_labels"]:
        counts.add_column(label, justify="right")
    counts.add_column("Total", justify="right")
    for i, label in enumerate(ct["row_labels"]):
        total = ct["row_totals"][i]
        cells = [
            f"{c:,.0f} ({c / total * 100:.1f}%)" if total else f"{c:,.0f}"
            for c in ct["counts"][i]
        ]
        counts.add_row(label, *cells, f"{total:,.0f}")
    counts.add_row(
        "Total",
        *[f"{t:,.0f}" for t in ct["col_totals"]],
        f"{ct['n']:,.0f}",
        style="bold",
    )
    out.append(counts)
    s = ct.get("stats")
    if s:
        stats = RichTable(title="Association", header_style="bold")
        stats.add_column("statistic")
        stats.add_column("value", justify="right")

        def p(v: float) -> str:
            return "< 1e-300" if v < 1e-300 else (f"{v:.3g}")

        stats.add_row("chi-square (dof)", f"{s['chi2']:,.2f} ({s['dof']})")
        stats.add_row("p-value", p(s["p_value"]))
        if s.get("chi2_yates") is not None:
            stats.add_row(
                "chi-square, Yates", f"{s['chi2_yates']:,.2f}  p = {p(s['p_yates'])}"
            )
        stats.add_row("G (likelihood ratio)", f"{s['g']:,.2f}  p = {p(s['p_g'])}")
        stats.add_row(
            "Cramer's V",
            f"{s['cramers_v']:.3f}  (bias-corrected {s['cramers_v_corrected']:.3f})",
        )
        if s.get("phi") is not None:
            stats.add_row("phi", f"{s['phi']:.3f}")
        if s.get("odds_ratio") is not None:
            lo, hi = s["odds_ratio_ci"]
            stats.add_row(
                "odds ratio", f"{s['odds_ratio']:.3f}  (95% CI {lo:.3f} to {hi:.3f})"
            )
        if s.get("fisher_p") is not None:
            stats.add_row("Fisher exact p", p(s["fisher_p"]))
        stats.add_row("smallest expected count", f"{s['min_expected']:.2f}")
        out.append(stats)
    for w in ct.get("warnings", []):
        out.append(Text(f"warning: {w}", style="yellow"))
    sp = ct.get("simpson")
    if sp:
        layers = RichTable(title=f"By {ct['layer_field']}", header_style="bold")
        layers.add_column(ct["layer_field"])
        layers.add_column("n", justify="right")
        layers.add_column(
            f"{ct['row_labels'][0]}: share of {ct['col_labels'][0]}", justify="right"
        )
        layers.add_column(
            f"{ct['row_labels'][1]}: share of {ct['col_labels'][0]}", justify="right"
        )
        layers.add_column("odds ratio", justify="right")
        for layer in ct["layers"]:
            r = layer["rates"] or [None, None]
            layers.add_row(
                layer["label"],
                f"{layer['n']:,.0f}",
                "" if r[0] is None else f"{r[0] * 100:.1f}%",
                "" if r[1] is None else f"{r[1] * 100:.1f}%",
                "" if layer["odds_ratio"] is None else f"{layer['odds_ratio']:.2f}",
            )
        pr = sp["pooled_rates"]
        layers.add_row(
            "pooled",
            f"{ct['n']:,.0f}",
            "" if pr[0] is None else f"{pr[0] * 100:.1f}%",
            "" if pr[1] is None else f"{pr[1] * 100:.1f}%",
            "" if sp["pooled_odds_ratio"] is None else f"{sp['pooled_odds_ratio']:.2f}",
            style="bold",
        )
        out.append(layers)
        out.append(Text(sp["verdict"], style="bold red" if sp["reversal"] else "green"))
    return out
