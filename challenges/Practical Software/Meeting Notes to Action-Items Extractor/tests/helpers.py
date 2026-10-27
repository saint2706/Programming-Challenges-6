"""Shared helpers: run the rules-only extractor on a string without touching the filesystem."""

import json

from actionitems.extract import Extraction, Options, extract_document
from actionitems.ingest import Document

MEETING = "2026-11-02"  # a Monday
ATTENDEES = ("Priya Shah", "Marco Rossi", "Dana Lee", "Eli Park")


def run(
    text: str,
    *,
    meeting: str | None = MEETING,
    tz: str = "UTC",
    attendees: tuple[str, ...] = ATTENDEES,
    min_confidence: float = 0.4,
    name: str = "notes.txt",
) -> Extraction:
    options = Options(
        attendees=attendees,
        meeting_date=meeting,
        timezone=tz,
        min_confidence=min_confidence,
        use_nlp=False,
    )
    return extract_document(Document(name, text, None), options)


def reply(state: dict, content: str) -> None:
    """Make the fake server answer with one chat completion whose content is ``content``."""
    state["body"] = json.dumps({"choices": [{"message": {"content": content}}]}).encode(
        "utf-8"
    )
