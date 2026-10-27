from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest
from actionitems.dates import (
    date_from_header,
    date_from_mtime,
    find_due_phrase,
    parse_meeting_date,
    resolve_due,
    validate_timezone,
    week_monday,
)
from actionitems.errors import InputError
from actionitems.formats import parse_document

MON = date(2026, 11, 2)
FRI = date(2026, 11, 6)


# ---- format parsing -----------------------------------------------------------------------


def test_speaker_turns_are_parsed():
    (first, *_) = parse_document("Priya Shah: Let's start.\nMarco Rossi: I'll send it.")
    assert first.speaker == "Priya Shah"
    assert first.text == "Let's start."
    assert first.line == 1


def test_zoom_timestamp_and_speaker():
    (utt,) = parse_document("[00:00:05] Dana Lee: Okay, I will book it.")
    assert utt.timestamp == "00:00:05"
    assert utt.speaker == "Dana Lee"
    assert utt.text == "Okay, I will book it."


def test_teams_header_applies_to_following_lines():
    text = "Dana Lee 00:00:52\nSure, I will book it.\n\nMarco Rossi 00:01:20\nDone."
    utts = [u for u in parse_document(text) if u.text]
    speakers = {u.text: u.speaker for u in utts}
    assert speakers["Sure, I will book it."] == "Dana Lee"
    assert speakers["Done."] == "Marco Rossi"


def test_webvtt_voice_tag_and_plain_cue():
    text = (
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:05.000\n<v Priya Shah>Let's start.</v>\n\n"
        "2\n00:00:05.500 --> 00:00:09.000\nMarco Rossi: I'll fix it.\n"
    )
    utts = [u for u in parse_document(text) if u.kind == "text"]
    assert [(u.speaker, u.text) for u in utts] == [
        ("Priya Shah", "Let's start."),
        ("Marco Rossi", "I'll fix it."),
    ]
    assert utts[0].timestamp == "00:00:01"


def test_srt_drops_index_and_timing_lines():
    text = "1\n00:00:00,000 --> 00:00:04,000\nDana Lee: Hello there.\n"
    utts = [u for u in parse_document(text) if u.kind == "text"]
    assert [(u.speaker, u.text) for u in utts] == [("Dana Lee", "Hello there.")]


def test_notes_bullets_and_action_section():
    text = (
        "Action items:\n- Marco to send the deck by Friday.\nDecisions:\n- We agreed."
    )
    utts = parse_document(text)
    bullet = next(u for u in utts if "send the deck" in u.text)
    assert bullet.kind == "action_section"
    assert bullet.section == "action items"
    decided = next(u for u in utts if "We agreed" in u.text)
    assert decided.section == "decisions"


def test_checkbox_states_are_distinct():
    text = "- [ ] Send the deck (Marco Rossi)\n- [x] Book the room (Dana Lee)"
    kinds = [u.kind for u in parse_document(text)]
    assert kinds == ["checkbox_open", "checkbox_done"]


def test_checkbox_does_not_inherit_an_earlier_speaker():
    text = "Priya Shah: Thanks all.\n- [ ] Send the deck (Marco Rossi)"
    box = next(u for u in parse_document(text) if u.kind == "checkbox_open")
    assert box.speaker is None


def test_asr_text_without_punctuation_is_plain_text():
    (utt,) = parse_document("can you book the room Dana")
    assert utt.kind == "text"
    assert utt.speaker is None


def test_labels_such_as_date_are_not_speakers():
    utts = parse_document("Date: 2026-11-02\nAttendees: Priya Shah, Dana Lee")
    assert all(u.speaker is None for u in utts)


# ---- due phrases ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("I'll send it by Friday.", "by Friday"),
        ("I will book it tomorrow.", "tomorrow"),
        ("Done in 3 days.", "in 3 days"),
        ("Ship it next Friday", "next Friday"),
    ],
)
def test_find_due_phrase(text, phrase):
    found = find_due_phrase(text)
    assert found is not None
    assert found.text.lower() == phrase.lower()


def test_find_due_phrase_none_when_absent():
    assert find_due_phrase("We will talk about it.") is None


# ---- resolve_due (meeting on Monday 2026-11-02 unless stated) -------------------------------


@pytest.mark.parametrize(
    ("phrase", "meeting", "expected", "confidence"),
    [
        ("tomorrow", MON, date(2026, 11, 3), 0.95),
        ("today", MON, MON, None),
        ("EOD", MON, MON, None),
        ("by Friday", MON, FRI, None),
        ("on Monday", MON, MON, None),
        ("by Friday", FRI, FRI, None),
        ("next Friday", MON, date(2026, 11, 13), None),
        ("next Friday", FRI, date(2026, 11, 13), None),
        ("this week", MON, FRI, 0.5),
        ("next week", MON, date(2026, 11, 9), 0.5),
        ("by end of week", MON, FRI, None),
        ("in 3 days", MON, date(2026, 11, 5), 0.9),
        ("in 2 weeks", MON, date(2026, 11, 16), 0.9),
        ("2026-11-20", MON, date(2026, 11, 20), None),
    ],
)
def test_resolve_due_phrases(phrase, meeting, expected, confidence):
    resolved = resolve_due(phrase, meeting)
    assert resolved is not None, phrase
    assert resolved.due == expected
    if confidence is not None:
        assert resolved.confidence == confidence


def test_resolve_due_unknown_phrase_is_none():
    assert resolve_due("soonish", MON) is None


def test_week_monday_is_iso_monday():
    assert week_monday(FRI) == MON
    assert week_monday(MON) == MON


# ---- meeting dates and timezones -------------------------------------------------------------


def test_validate_timezone_accepts_iana_names():
    assert validate_timezone("Europe/London") == ZoneInfo("Europe/London")


def test_validate_timezone_rejects_unknown_names():
    with pytest.raises(InputError):
        validate_timezone("Mars/Olympus")


def test_parse_meeting_date_plain_and_iso_datetime():
    tz = ZoneInfo("UTC")
    assert parse_meeting_date("2026-11-02", tz) == MON
    assert parse_meeting_date("2026-11-02T23:30:00+00:00", tz) == MON


def test_parse_meeting_date_rejects_garbage():
    with pytest.raises(InputError):
        parse_meeting_date("next tuesday-ish", ZoneInfo("UTC"))


def test_date_from_header_reads_a_date_line():
    assert date_from_header("Meeting\nDate: 2026-11-02\n", ZoneInfo("UTC")) == MON


def test_date_from_header_none_without_a_date():
    assert date_from_header("Priya: no date here", ZoneInfo("UTC")) is None


def test_date_from_mtime_uses_the_meeting_timezone_across_dst():
    # 23:30 UTC on 29 March 2026 is 00:30 BST on 30 March in London (clocks went forward).
    instant = datetime(2026, 3, 29, 23, 30, tzinfo=UTC)
    assert date_from_mtime(instant, ZoneInfo("Europe/London")) == date(2026, 3, 30)
    assert date_from_mtime(instant, ZoneInfo("UTC")) == date(2026, 3, 29)
