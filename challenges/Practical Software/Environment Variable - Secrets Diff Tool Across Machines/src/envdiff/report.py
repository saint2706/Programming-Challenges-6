"""Render a :class:`Report` as a terminal table, JSON, or Markdown. No renderer sees a value.

The only value-derived text any of them can emit is an equality group letter, a short keyed
fingerprint when asked for, a coarse shape tag, and anything the caller listed in ``reveal``.
"""

import io
import json
import string

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from envdiff.compare import KeyResult, Report

LETTERS = string.ascii_uppercase + string.ascii_lowercase
ABSENT, PASSTHROUGH = "—", "·"


def group_name(group: int) -> str:
    if group < 0:
        return PASSTHROUGH
    return LETTERS[group] if group < len(LETTERS) else f"#{group + 1}"


def cell(result: KeyResult, label: str, show_fingerprints: bool) -> str:
    if label in result.absent:
        return ABSENT
    name = group_name(result.groups[label])
    if show_fingerprints and label in result.fingerprints:
        return f"{name} {result.fingerprints[label]}"
    return name


RELATION_TEXT = {
    "whitespace": "only whitespace differs",
    "quotes": "only quoting differs",
    "case": "only letter case differs",
    "empty": "one is empty",
    "pass-through": "one comes from the runtime environment",
}


def verdict(result: KeyResult) -> str:
    parts = []
    if result.status == "differs":
        why = "; ".join(
            f"{group_name(a)}/{group_name(b)}: {RELATION_TEXT[r]}"
            for a, b, r in result.relations
        )
        parts.append("differs" + (f" ({why})" if why else ""))
    if result.absent:
        parts.append(f"missing in {', '.join(result.absent)}")
    return "; ".join(parts) or "same"


def summary_line(report: Report) -> str:
    bits = [
        f"{len(report.sources)} sources",
        f"{len(report.keys)} variables",
        f"{report.count('same')} same",
        f"{report.count('differs')} differ",
        f"{report.count('missing')} missing somewhere",
    ]
    if report.ignored:
        bits.append(f"{report.ignored} ignored")
    return " · ".join(bits)


def _selected(report: Report, show_all: bool) -> list[KeyResult]:
    return [k for k in report.keys if show_all or k.status != "same"]


def render_text(
    report: Report, show_all: bool = False, show_fingerprints: bool = False,
    revealed: dict[str, dict[str, str]] | None = None, width: int | None = None,
) -> str:  # fmt: skip
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=width or 200,
        highlight=False,
        force_terminal=False,
        color_system=None,
    )
    console.print("[bold]Sources[/bold]")
    for source in report.sources:
        count = len(source.entries)
        where = "" if source.origin == source.label else f"  {escape(source.origin)}"
        console.print(
            f"  {escape(source.label)}{where}  ({source.fmt}, {count} variables)",
            soft_wrap=True,
        )
    console.print(f"\n{summary_line(report)}")
    rows = _selected(report, show_all)
    if rows:
        table = Table(show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("VARIABLE", no_wrap=True)
        for source in report.sources:
            table.add_column(escape(source.label), justify="center", no_wrap=True)
        table.add_column("VERDICT")
        for result in rows:
            cells = [cell(result, s.label, show_fingerprints) for s in report.sources]
            table.add_row(result.key, *cells, verdict(result))
        console.print()
        console.print(table)
        console.print(
            "\nCells with the same letter hold the same value. Values are never shown.",
            style="dim",
        )
    elif report.keys:
        console.print("\nNo differences.")
    if revealed:
        console.print("\n[bold]Revealed values (requested with --reveal)[/bold]")
        for key, per_source in revealed.items():
            for label, value in per_source.items():
                console.print(
                    f"  {key} [{escape(label)}] = {escape(repr(value))}", soft_wrap=True
                )
    if report.findings:
        console.print("\n[bold]Findings[/bold]")
        for finding in report.findings:
            where = f"  [{', '.join(finding.sources)}]" if finding.sources else ""
            console.print(
                f"  {finding.severity:<4}  {escape(finding.message)}{escape(where)}",
                soft_wrap=True,
            )
    return buffer.getvalue()


def render_json(
    report: Report, show_all: bool = True, show_fingerprints: bool = False,
    revealed: dict[str, dict[str, str]] | None = None,
) -> str:  # fmt: skip
    keys = []
    for result in _selected(report, show_all):
        item = {
            "key": result.key,
            "status": result.status,
            "relations": [
                {"groups": [a, b], "why": why} for a, b, why in result.relations
            ],
            "groups": {
                label: (None if g < 0 else g) for label, g in result.groups.items()
            },
            "absent": result.absent,
            "shapes": result.shapes,
        }
        if show_fingerprints:
            item["fingerprints"] = result.fingerprints
        keys.append(item)
    document = {
        "sources": [
            {
                "label": s.label,
                "origin": s.origin,
                "format": s.fmt,
                "variables": len(s.entries),
            }
            for s in report.sources
        ],
        "summary": {
            "variables": len(report.keys),
            "same": report.count("same"),
            "differs": report.count("differs"),
            "missing": report.count("missing"),
            "ignored": report.ignored,
        },
        "keys": keys,
        "findings": [
            {
                "code": f.code,
                "severity": f.severity,
                "key": f.key,
                "sources": f.sources,
                "line": f.line or None,
                "message": f.message,
            }
            for f in report.findings
        ],
    }
    if revealed:
        document["revealed"] = revealed
    return json.dumps(document, indent=2) + "\n"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    report: Report, show_all: bool = False, show_fingerprints: bool = False,
    revealed: dict[str, dict[str, str]] | None = None,
) -> str:  # fmt: skip
    lines = ["## Environment comparison", "", summary_line(report), ""]
    rows = _selected(report, show_all)
    if rows:
        header = ["Variable", *(_md(s.label) for s in report.sources), "Verdict"]
        lines += [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for result in rows:
            cells = [cell(result, s.label, show_fingerprints) for s in report.sources]
            lines.append(
                "| "
                + " | ".join([f"`{result.key}`", *cells, _md(verdict(result))])
                + " |"
            )
        lines += [
            "",
            "Cells with the same letter hold the same value. Values are never shown.",
        ]
    else:
        lines.append("No differences.")
    if revealed:
        lines += ["", "### Revealed values", ""]
        lines += [
            f"- `{key}` `{label}` = `{json.dumps(value)}`"
            for key, per in revealed.items()
            for label, value in per.items()
        ]
    if report.findings:
        lines += ["", "### Findings", ""]
        lines += [
            f"- **{f.severity}** {_md(f.message)} [{_md(', '.join(f.sources))}]"
            for f in report.findings
        ]
    return "\n".join(lines) + "\n"
