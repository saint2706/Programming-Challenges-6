"""iCalendar (RFC 5545) export: one VTODO per action item with a due date.

Text values are escaped (backslash, semicolon, comma, newline), so a CR/LF in a task cannot
inject a property (CRLF injection). Lines are folded at 75 octets without splitting a UTF-8
character. UID is derived from the content, so re-exporting the same items gives the same file.
"""

import hashlib
from datetime import UTC, date, datetime

from actionitems.extract import Extraction
from actionitems.models import ActionItem
from actionitems.text import one_line, safe_text

PRODID = "-//actionitems//Meeting Notes to Action Items//EN"
FOLD_OCTETS = 75


def escape_text(value: str) -> str:
    cleaned = safe_text(one_line(value))
    return (
        cleaned.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
    )


def fold(line: str) -> list[str]:
    """Fold to 75-octet lines; continuation lines start with a space and carry 74 octets."""
    pieces: list[str] = []
    current = ""
    size = 0
    limit = FOLD_OCTETS
    for char in line:
        width = len(char.encode("utf-8"))
        if size + width > limit:
            pieces.append(current)
            current, size, limit = "", 0, FOLD_OCTETS - 1
        current += char
        size += width
    pieces.append(current)
    return [pieces[0], *(" " + p for p in pieces[1:])]


def _uid(item: ActionItem, meeting: date, name: str) -> str:
    key = "|".join(
        [name, meeting.isoformat(), item.task, item.owner or "", item.due or ""]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24] + "@actionitems"


def _todo(item: ActionItem, extraction: Extraction, stamp: str) -> list[str]:
    lines = [
        "BEGIN:VTODO",
        f"UID:{_uid(item, extraction.meeting_date, extraction.name)}",
        f"DTSTAMP:{stamp}",
        f"SUMMARY:{escape_text(item.task)}",
        "STATUS:NEEDS-ACTION",
        "PRIORITY:5",
    ]
    if item.due:
        lines.append(f"DUE;VALUE=DATE:{item.due.replace('-', '')}")
    owner = item.owner or "unassigned"
    where = ", ".join(f"line {s.line}" for s in item.sources[:3])
    llm = " Added by the LLM pass, not the rules." if item.item_type == "llm" else ""
    lines.append(
        f"DESCRIPTION:{escape_text(f'Owner: {owner}. Source: {extraction.name}, {where}. Confidence {item.confidence:.2f}.{llm}')}"
    )
    if item.owner:
        lines.append(f"CATEGORIES:{escape_text(item.owner)}")
    lines.append("END:VTODO")
    return lines


def to_ics(extractions: list[Extraction], stamp: datetime | None = None) -> str:
    """Build the calendar. ``stamp`` is injectable so tests (and reruns) are deterministic."""
    moment = (stamp or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", f"PRODID:{PRODID}", "CALSCALE:GREGORIAN"]
    for extraction in extractions:
        for item in extraction.items:
            if item.due:
                lines.extend(_todo(item, extraction, moment))
    lines.append("END:VCALENDAR")
    folded = [part for line in lines for part in fold(line)]
    return "\r\n".join(folded) + "\r\n"
