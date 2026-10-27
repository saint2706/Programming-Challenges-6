"""Human and machine outputs: Markdown checklist, JSON, and a plain table.

Every value from a transcript passes through ``one_line`` and ``safe_text`` first, so a
terminal control sequence or a newline inside a task cannot reshape the output.
"""

import json

from actionitems.extract import Extraction
from actionitems.models import ActionItem
from actionitems.text import one_line, safe_text


def _cell(value: object) -> str:
    if value is None:
        return "-"
    return safe_text(one_line(str(value))).replace("|", "\\|")


# Characters that would start Markdown, HTML, links, mentions or table syntax. A backslash
# before ASCII punctuation is a literal in CommonMark, so this neutralises them in place.
_MD_SPECIAL = frozenset("\\`*_[]()<>#|@&")


def _md(value: object) -> str:
    """Escape a value for Markdown body text: no raw HTML, links, images, mentions or headings."""
    text = safe_text(one_line(str(value)))
    return "".join(f"\\{c}" if c in _MD_SPECIAL else c for c in text)


def _line_label(item: ActionItem) -> str:
    return ", ".join(f"line {s.line}" for s in item.sources[:3])


def to_markdown(extractions: list[Extraction]) -> str:
    out: list[str] = []
    for extraction in extractions:
        out.append(f"## {_md(extraction.name)} ({extraction.meeting_date.isoformat()})")
        out.append("")
        if not extraction.items:
            out.append("_No action items found._")
            out.append("")
            continue
        for item in extraction.items:
            box = "[ ]"
            owner = f" @{_md(item.owner)}" if item.owner else ""
            due = f" (due {_md(item.due)})" if item.due else ""
            task = _md(item.task)
            tag = ", LLM-added" if item.item_type == "llm" else ""
            out.append(
                f"- {box} {task}{owner}{due} <!-- {_line_label(item)}, conf {item.confidence:.2f}{tag} -->"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_json(extractions: list[Extraction]) -> str:
    payload = {
        "documents": [
            {
                "name": e.name,
                "meeting_date": e.meeting_date.isoformat(),
                "timezone": e.timezone,
                "backend": e.backend,
                "items": [item.to_dict() for item in e.items],
            }
            for e in extractions
        ]
    }
    # ensure_ascii keeps control, bidi and surrogate characters out of the file as raw bytes.
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def to_table(extractions: list[Extraction]) -> str:
    headers = ["document", "task", "owner", "due", "type", "conf"]
    rows: list[list[str]] = []
    for extraction in extractions:
        for item in extraction.items:
            rows.append(
                [
                    _cell(extraction.name),
                    _cell(item.task),
                    _cell(item.owner),
                    _cell(item.due),
                    _cell(item.item_type),
                    f"{item.confidence:.2f}",
                ]
            )
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()
    ]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(
            "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        )
    return "\n".join(lines) + "\n"
