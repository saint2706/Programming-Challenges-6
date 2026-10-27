"""Turn transcripts and notes into utterances: WebVTT/SRT cues, speaker turns, Teams/Zoom
exports, and Markdown or plain notes with bullets, checkboxes, headings and label sections."""

import re

from actionitems.models import Utterance

MAX_LINE_CHARS = (
    4000  # a line longer than this is cut; the parsers stay linear and cheap
)

NAME = r"[A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3}"
CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?"
TIMING_LINE = re.compile(rf"^\s*({CLOCK}(?:[.,]\d+)?)\s*-->\s*{CLOCK}(?:[.,]\d+)?")
SPEAKER_LINE = re.compile(
    rf"^\s*\[?(?:(?P<ts1>{CLOCK})\]?\s+)?(?P<name>{NAME})\]?\s*"
    rf"(?:\((?P<ts2>{CLOCK})\))?\s*:\s+(?P<text>\S.*)$"
)
TEAMS_HEADER = re.compile(rf"^\s*\[?(?P<name>{NAME})\]?\s+(?P<ts>{CLOCK})\s*$")
VOICE_TAG = re.compile(r"<v(?:\.[\w.]+)?\s+([^>]+)>")
MARKUP_TAG = re.compile(r"</?[A-Za-z][^>]*>")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.*?)\s*#*\s*$")
BULLET = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+(?P<body>.*)$")
CHECKBOX = re.compile(r"^\[(?P<mark>[ xX])\]\s*(?P<rest>.*)$")
LABEL = re.compile(
    r"^\s*(?P<label>action items?|actions?|next steps?|to-?\s?dos?|decisions?|questions?"
    r"|notes?|agenda|summary|blockers?|risks?|follow[- ]ups?|attendees?|tl;?dr)"
    r"\s*:\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
BARE_ACTION_HEADING = re.compile(
    r"^\s*(?P<label>action items?|next steps?|to-?\s?dos?|follow[- ]ups?|notes?)\s*:?\s*$",
    re.IGNORECASE,
)
ACTION_SECTIONS = frozenset(
    {
        "action items",
        "action item",
        "action",
        "actions",
        "next steps",
        "next step",
        "todo",
        "todos",
        "to do",
        "to-do",
        "to-dos",
        "follow up",
        "follow-up",
        "follow ups",
        "follow-ups",
        "tasks",
        "task",
        "task list",
    }
)
# Words that look like "Name:" but are section labels or header fields, never speakers.
NOT_SPEAKERS = frozenset(
    {
        "date",
        "time",
        "location",
        "topic",
        "subject",
        "title",
        "project",
        "meeting",
        "when",
        "where",
        "tasks",
        "task",
        "action",
        "action items",
        "actions",
        "agenda",
        "attendees",
        "blocker",
        "blockers",
        "decision",
        "decisions",
        "follow up",
        "follow-up",
        "note",
        "notes",
        "outcome",
        "outcomes",
        "question",
        "questions",
        "reminder",
        "risk",
        "risks",
        "status",
        "summary",
        "tldr",
        "todo",
        "todos",
        "update",
        "next steps",
        "next step",
        "follow ups",
        "follow-ups",
    }
)


def _norm_label(label: str) -> str:
    return " ".join(label.lower().replace("-", " ").split()).replace("to dos", "todos")


def _clock(value: str) -> str:
    """Trim a timestamp to HH:MM:SS (drops milliseconds and comma separators)."""
    value = re.split(r"[.,]", value)[0]
    parts = value.split(":")
    if len(parts) == 2:
        parts = ["00", *parts]
    return ":".join(part.zfill(2) for part in parts)


def _is_speaker(name: str) -> bool:
    return name.strip().lower() not in NOT_SPEAKERS


def _clean_line(raw: str) -> str:
    return raw.strip()[:MAX_LINE_CHARS]


def _cue_utterances(lines: list[str]) -> list[Utterance]:
    """WebVTT and SRT: text lines under each timing line, with voice tags or Name: prefixes."""
    out: list[Utterance] = []
    in_note = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            in_note = False
            i += 1
            continue
        if in_note or stripped.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            in_note = in_note or stripped.startswith("NOTE")
            i += 1
            continue
        timing = TIMING_LINE.match(raw)
        if not timing:
            i += 1
            continue
        stamp = _clock(timing.group(1))
        i += 1
        while i < len(lines) and lines[i].strip():
            text_line = _clean_line(lines[i])
            speaker = None
            voice = VOICE_TAG.search(text_line)
            if voice:
                speaker = voice.group(1).strip()
            text = MARKUP_TAG.sub("", text_line).strip()
            match = SPEAKER_LINE.match(text)
            if speaker is None and match and _is_speaker(match.group("name")):
                speaker, text = match.group("name").strip(), match.group("text").strip()
            if text:
                out.append(
                    Utterance(line=i + 1, text=text, speaker=speaker, timestamp=stamp)
                )
            i += 1
    return out


def _notes_utterances(lines: list[str]) -> list[Utterance]:
    """Speaker transcripts, Teams/Zoom exports, Markdown and plain notes, line by line."""
    out: list[Utterance] = []
    section: str | None = None
    speaker: str | None = None
    stamp: str | None = None
    for number, raw in enumerate(lines, start=1):
        line = _clean_line(raw)
        if not line:
            continue
        heading = HEADING.match(line)
        if heading:
            section = _norm_label(heading.group("text").rstrip(":"))
            speaker = stamp = None
            continue
        bare = BARE_ACTION_HEADING.match(line)
        if bare:
            section = _norm_label(bare.group("label"))
            speaker = stamp = None
            continue
        bullet = BULLET.match(line)
        body = bullet.group("body").strip() if bullet else line
        kind = "bullet" if bullet else "text"
        check = CHECKBOX.match(body) if bullet else None
        if check:
            kind = "checkbox_done" if check.group("mark") in "xX" else "checkbox_open"
            body = check.group("rest").strip()
            if not body:
                continue
        label = LABEL.match(body)
        if label:
            norm = _norm_label(label.group("label"))
            if norm in ACTION_SECTIONS and label.group("rest").strip():
                out.append(
                    Utterance(
                        number,
                        label.group("rest").strip(),
                        None,
                        None,
                        "action_section",
                        "action items",
                    )
                )
                continue
            if not label.group("rest").strip():
                section = norm
                speaker = stamp = None
                continue
            out.append(
                Utterance(number, label.group("rest").strip(), None, None, "text", norm)
            )
            continue
        teams = TEAMS_HEADER.match(body)
        if teams and _is_speaker(teams.group("name")) and not bullet:
            speaker, stamp = teams.group("name").strip(), _clock(teams.group("ts"))
            continue
        turn = SPEAKER_LINE.match(body)
        if turn and _is_speaker(turn.group("name")):
            speaker = turn.group("name").strip()
            when = turn.group("ts1") or turn.group("ts2")
            stamp = _clock(when) if when else stamp
            body = turn.group("text").strip()
            if section in ACTION_SECTIONS and kind in ("text", "bullet"):
                kind = "action_section"
            out.append(Utterance(number, body, speaker, stamp, kind, section))
            continue
        if section in ACTION_SECTIONS and kind in ("text", "bullet"):
            kind = "action_section"
        # A checkbox is a task on its own; a speaker carried down from a turn must not own it.
        line_speaker = None if kind.startswith("checkbox") else speaker
        out.append(Utterance(number, body, line_speaker, stamp, kind, section))
    return out


def parse_document(text: str) -> list[Utterance]:
    """Parse any supported transcript or notes text into utterances with 1-based line numbers."""
    lines = text.split("\n")
    if any(TIMING_LINE.match(line) for line in lines):
        return _cue_utterances(lines)
    return _notes_utterances(lines)
