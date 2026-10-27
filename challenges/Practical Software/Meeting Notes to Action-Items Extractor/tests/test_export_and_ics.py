import json
from datetime import UTC, date, datetime

import pytest
from actionitems.export import to_json, to_markdown, to_table
from actionitems.extract import Extraction
from actionitems.ics import escape_text, fold, to_ics
from actionitems.models import ActionItem, Source
from helpers import run
from icalendar import Calendar

STAMP = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


def make_item(
    task="Send the deck", owner="Marco Rossi", due="2026-11-06", line=3
) -> ActionItem:
    return ActionItem(
        task=task,
        owner=owner,
        due=due,
        due_phrase="by Friday" if due else None,
        due_confidence=0.9 if due else 0.0,
        item_type="commitment",
        confidence=0.9,
        reasons=[],
        sources=[Source(line, None, owner, "excerpt")],
    )


def make_extraction(*items: ActionItem, name="notes.txt") -> Extraction:
    return Extraction(name, date(2026, 11, 2), "UTC", "rules-only", list(items))


# ---- markdown / json / table ----------------------------------------------------------------


def test_markdown_is_a_checklist_with_owner_due_and_source_line():
    text = to_markdown([make_extraction(make_item())])
    assert "- [ ] Send the deck" in text
    assert "Marco Rossi" in text
    assert "2026-11-06" in text
    assert "line 3" in text


def test_markdown_for_no_items_says_so():
    assert "No action items" in to_markdown([make_extraction()])


def test_json_round_trips_items_and_ends_with_newline():
    out = to_json([make_extraction(make_item())])
    assert out.endswith("\n")
    data = json.loads(out)
    doc = data["documents"][0]
    assert doc["name"] == "notes.txt"
    assert doc["items"][0]["task"] == "Send the deck"
    assert doc["items"][0]["sources"][0]["line"] == 3


def test_table_has_header_and_one_row_per_item():
    rows = to_table(
        [
            make_extraction(
                make_item(), make_item(task="Book the room", owner=None, due=None)
            )
        ]
    )
    lines = [line for line in rows.splitlines() if line.strip()]
    assert len(lines) == 2 + 2  # header, separator, two rows
    assert "Book the room" in rows


def test_markdown_escapes_pipes_in_tasks():
    text = to_table([make_extraction(make_item(task="Fix a | b"))])
    assert "a | b" not in text or "\\|" in text


# ---- ICS ------------------------------------------------------------------------------------


def test_escape_text_protects_structural_characters():
    assert escape_text("a;b,c\\d") == "a\\;b\\,c\\\\d"


def test_escape_text_removes_line_breaks():
    assert "\n" not in escape_text("one\r\nTWO")
    assert "\r" not in escape_text("one\rTWO")


def test_fold_limits_lines_to_75_octets():
    pieces = fold("X" * 200)
    assert all(len(p.encode("utf-8")) <= 75 for p in pieces)
    assert pieces[1].startswith(" ")
    assert "".join(p[1:] if i else p for i, p in enumerate(pieces)) == "X" * 200


def test_fold_never_splits_a_multibyte_character():
    line = "é" * 80  # two octets each
    pieces = fold(line)
    rebuilt = "".join(p[1:] if i else p for i, p in enumerate(pieces))
    assert rebuilt == line
    assert all(len(p.encode("utf-8")) <= 75 for p in pieces)


def test_short_line_is_not_folded():
    assert fold("SUMMARY:short") == ["SUMMARY:short"]


def test_ics_uses_crlf_and_has_one_vtodo_per_dated_item():
    out = to_ics(
        [make_extraction(make_item(), make_item(task="No date", due=None))], stamp=STAMP
    )
    assert "\r\n" in out
    assert "\n" not in out.replace("\r\n", "")
    assert out.count("BEGIN:VTODO") == 1


def test_ics_uid_is_stable_for_the_same_content():
    ext = make_extraction(make_item())
    first = to_ics([ext], stamp=STAMP)
    second = to_ics([ext], stamp=STAMP)
    assert first == second


def test_ics_crlf_injection_in_a_task_cannot_add_a_component():
    evil = make_item(task="ok\r\nEND:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:pwned")
    out = to_ics([make_extraction(evil)], stamp=STAMP)
    raw_lines = [line for line in out.split("\r\n") if line]
    assert raw_lines.count("END:VCALENDAR") == 1
    assert "BEGIN:VEVENT" not in raw_lines


def test_ics_round_trips_through_icalendar():
    out = to_ics(
        [make_extraction(make_item(task="Send the deck; with, commas"))], stamp=STAMP
    )
    cal = Calendar.from_ical(out)
    (todo,) = [c for c in cal.walk() if c.name == "VTODO"]
    assert str(todo["SUMMARY"]) == "Send the deck; with, commas"
    assert todo["DUE"].dt == date(2026, 11, 6)
    assert "Marco Rossi" in str(todo["CATEGORIES"])


def test_ics_with_no_dated_items_is_an_empty_calendar():
    out = to_ics([make_extraction(make_item(due=None))], stamp=STAMP)
    cal = Calendar.from_ical(out)
    assert [c for c in cal.walk() if c.name == "VTODO"] == []


def test_ics_from_a_real_extraction_round_trips():
    text = "Marco Rossi: I'll send the deck by Friday.\nDana Lee: Okay, I'll book the room tomorrow."
    out = to_ics([run(text)], stamp=STAMP)
    cal = Calendar.from_ical(out)
    dues = sorted(c["DUE"].dt for c in cal.walk() if c.name == "VTODO")
    assert dues == [date(2026, 11, 3), date(2026, 11, 6)]


@pytest.mark.parametrize("task", ["plain", "semi;colon", "comma,here", "back\\slash"])
def test_ics_summary_survives_round_trip(task):
    out = to_ics([make_extraction(make_item(task=task))], stamp=STAMP)
    (todo,) = [c for c in Calendar.from_ical(out).walk() if c.name == "VTODO"]
    assert str(todo["SUMMARY"]) == task
