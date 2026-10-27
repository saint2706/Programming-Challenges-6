"""Due dates: relative phrases resolved against the meeting date, absolute ones via dateparser.

Weekday semantics (documented and tested):
- "Friday", "by Friday", "on Friday", "this Friday": the first Friday on or after the meeting day.
- "next Friday": Friday of the following ISO week (Monday to Sunday), so on a Wednesday it is
  nine days away, not two.
- "end of week" / "EOW": Friday of the meeting's ISO week; on a Saturday or Sunday, the next Friday.
- "next week": Monday of the next ISO week (low confidence, the phrase is vague).
- "EOD", "today", "tonight": the meeting day. "tomorrow": +1 day. "in N days/weeks/months".
- Months are calendar months and clamp to the last day (Jan 31 + 1 month = Feb 28/29).
Dates are plain calendar dates, so DST transitions cannot shift them.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser

from actionitems.errors import InputError

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tues": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thur": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}  # fmt: skip
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip
WD = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
NUM = r"(?:\d{1,2}|" + "|".join(NUMBER_WORDS) + r")"
PREFIX = r"(?:(?:by|on|before|due|until|no later than|for)\s+)?"

# Ordered patterns; the leftmost, then longest, match in a clause is the due phrase.
PATTERNS: list[tuple[str, str]] = [
    (r"end of (?:the |this )?(?:work )?week|\beow\b", "eow"),
    (r"end of (?:the |this )?month|\beom\b", "eom"),
    (
        r"eod(?: today)?|cob|end of (?:the )?day|close of business|today|tonight",
        "today",
    ),
    (r"day after tomorrow", "plus2"),
    (r"tomorrow", "plus1"),
    (r"next week", "next_week"),
    (r"this week", "this_week"),
    (rf"next (?:{WD})", "next_wd"),
    (rf"this (?:coming )?(?:{WD})", "this_wd"),
    (rf"(?:{WD}) next week", "next_wd"),
    (rf"(?:{WD})", "bare_wd"),
    (r"in a (?:couple|few) (?:of )?(?:days|weeks)", "vague_in"),
    (rf"in ({NUM}) (days?|weeks?|months?)", "in_n"),
    (r"the (\d{1,2})(?:st|nd|rd|th)", "dom"),
    (rf"{MONTH}\.? \d{{1,2}}(?:st|nd|rd|th)?(?:,? \d{{4}})?", "abs"),
    (rf"\d{{1,2}}(?:st|nd|rd|th)? {MONTH}\.?(?:,? \d{{4}})?", "abs"),
    (r"\d{4}-\d{2}-\d{2}", "abs"),
    (r"\d{1,2}/\d{1,2}(?:/\d{2,4})?", "abs"),
]
_COMPILED = [(re.compile(rf"\b{PREFIX}(?:{pattern})\b", re.IGNORECASE), kind)
             for pattern, kind in PATTERNS]  # fmt: skip


@dataclass(frozen=True)
class DuePhrase:
    text: str  # as written in the source, prefix included ("by Friday")
    start: int
    end: int
    kind: str
    match: re.Match[str]


@dataclass(frozen=True)
class ResolvedDue:
    due: date
    phrase: str
    confidence: float
    reason: str


def find_due_phrase(text: str) -> DuePhrase | None:
    """Return the leftmost (then longest) due-date phrase in ``text``, if any."""
    best: DuePhrase | None = None
    for regex, kind in _COMPILED:
        for match in regex.finditer(text):
            candidate = DuePhrase(
                match.group(0), match.start(), match.end(), kind, match
            )
            if best is None or (candidate.start, -len(candidate.text)) < (
                best.start,
                -len(best.text),
            ):
                best = candidate
    return best


def week_monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _add_months(day: date, months: int) -> date:
    index = day.month - 1 + months
    year, month = day.year + index // 12, index % 12 + 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


def _upcoming(meeting: date, weekday: int) -> date:
    return meeting + timedelta(days=(weekday - meeting.weekday()) % 7)


def _friday_of_week(meeting: date) -> date:
    friday = week_monday(meeting) + timedelta(days=4)
    return friday if friday >= meeting else friday + timedelta(days=7)


def _absolute(phrase: str, meeting: date) -> ResolvedDue | None:
    phrase = re.sub(
        r"^(?:by|on|before|due|until|no later than|for)\s+",
        "",
        phrase,
        flags=re.IGNORECASE,
    )
    base = datetime.combine(meeting, time(12, 0))
    settings = {"RELATIVE_BASE": base, "PREFER_DATES_FROM": "future"}
    parsed = dateparser.parse(phrase, settings=settings)
    if parsed is None:
        return None
    has_year = re.search(r"\d{4}", phrase) is not None
    if has_year:
        return ResolvedDue(parsed.date(), phrase, 0.9, "explicit date")
    return ResolvedDue(
        parsed.date(), phrase, 0.75, "year inferred as the next future date"
    )


def _resolve_kind(
    kind: str, match: re.Match, phrase: str, meeting: date
) -> ResolvedDue | None:
    if kind == "today":
        return ResolvedDue(meeting, phrase, 0.95, "same day as the meeting")
    if kind == "plus1":
        return ResolvedDue(
            meeting + timedelta(days=1), phrase, 0.95, "the day after the meeting"
        )
    if kind == "plus2":
        return ResolvedDue(
            meeting + timedelta(days=2), phrase, 0.9, "two days after the meeting"
        )
    if kind == "eow":
        return ResolvedDue(
            _friday_of_week(meeting), phrase, 0.85, "Friday of the meeting's week"
        )
    if kind == "eom":
        last = calendar.monthrange(meeting.year, meeting.month)[1]
        return ResolvedDue(date(meeting.year, meeting.month, last), phrase, 0.8,
                           "last day of the meeting's month")  # fmt: skip
    if kind == "next_week":
        due = week_monday(meeting) + timedelta(days=7)
        return ResolvedDue(due, phrase, 0.5, "'next week' read as Monday of next week")
    if kind == "this_week":
        due = _friday_of_week(meeting)
        return ResolvedDue(due, phrase, 0.5, "'this week' read as Friday of this week")
    if kind in ("next_wd", "this_wd", "bare_wd"):
        word = _weekday_in(phrase)
        if word is None:
            return None
        if kind == "next_wd":
            due = week_monday(meeting) + timedelta(days=7 + word)
            return ResolvedDue(
                due, phrase, 0.8, "'next <weekday>' = following ISO week"
            )
        due = _upcoming(meeting, word)
        if kind == "this_wd":
            return ResolvedDue(
                due, phrase, 0.85, "'this <weekday>' = next on or after meeting"
            )
        confidence = (
            0.85 if phrase.lower().startswith(("by", "on", "before", "due")) else 0.8
        )
        return ResolvedDue(
            due, phrase, confidence, "weekday = next on or after the meeting"
        )
    if kind == "vague_in":
        lowered = phrase.lower()
        if "month" in lowered:
            due = _add_months(meeting, 2)
        else:
            due = meeting + timedelta(days=14 if "week" in lowered else 2)
        return ResolvedDue(due, phrase, 0.6, "'a couple of' read as two units")
    if kind == "in_n":
        raw = match.group(1).lower()
        count = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
        unit = match.group(2).lower()
        if unit.startswith("day"):
            due = meeting + timedelta(days=count)
        elif unit.startswith("week"):
            due = meeting + timedelta(days=7 * count)
        else:
            due = _add_months(meeting, count)
        return ResolvedDue(
            due, phrase, 0.9, f"'in {count} {unit}' counted from the meeting"
        )
    if kind == "dom":
        day_number = int(match.group(1))
        return _day_of_month(meeting, day_number, phrase)
    if kind == "abs":
        return _absolute(phrase, meeting)
    return None


def _day_of_month(meeting: date, day_number: int, phrase: str) -> ResolvedDue | None:
    if not 1 <= day_number <= 31:
        return None
    for offset in range(13):
        probe = _add_months(meeting.replace(day=1), offset)
        if day_number <= calendar.monthrange(probe.year, probe.month)[1]:
            due = probe.replace(day=day_number)
            if due >= meeting:
                return ResolvedDue(
                    due, phrase, 0.6, "day of month read as next occurrence"
                )
    return None


def _weekday_in(phrase: str) -> int | None:
    for word in re.findall(r"[a-z]+", phrase.lower()):
        if word in WEEKDAYS:
            return WEEKDAYS[word]
    return None


def resolve_due(phrase_text: str, meeting: date) -> ResolvedDue | None:
    """Resolve one due phrase (as returned by ``find_due_phrase``) against the meeting day."""
    found = find_due_phrase(phrase_text)
    if found is None:
        return None
    try:
        return _resolve_kind(found.kind, found.match, found.text, meeting)
    except (OverflowError, ValueError):  # e.g. "in 99999999 days" runs past date.max
        return None


MIN_YEAR, MAX_YEAR = 1900, 2200


def _checked_meeting(day: date, source: str) -> date:
    """Meeting dates outside a sane range would overflow relative arithmetic; reject them."""
    if not MIN_YEAR <= day.year <= MAX_YEAR:
        raise InputError(f"{source} year must be between {MIN_YEAR} and {MAX_YEAR}")
    return day


def validate_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise InputError(f"unknown timezone: {name!r}") from exc


def parse_meeting_date(value: str, tz: ZoneInfo) -> date:
    """Accept YYYY-MM-DD or an ISO datetime (with or without offset, converted to ``tz``)."""
    text = value.strip()
    try:
        if len(text) == 10:
            return _checked_meeting(date.fromisoformat(text), "--meeting-date")
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is not None:
            moment = moment.astimezone(tz)
    except (ValueError, OverflowError) as exc:
        raise InputError(
            f"--meeting-date must be YYYY-MM-DD or ISO datetime: {value!r}"
        ) from exc
    return _checked_meeting(moment.date(), "--meeting-date")


def date_from_header(text: str, tz: ZoneInfo) -> date | None:
    """A ``Date: 2026-10-27`` (or ``Meeting date:``) line near the top of the document."""
    for line in text.split("\n")[:40]:
        match = re.match(
            r"^\s*(?:meeting\s+)?date\s*:\s*(\S.{0,60})$", line, re.IGNORECASE
        )
        if match:
            parsed = dateparser.parse(match.group(1), settings={"TIMEZONE": tz.key})
            if parsed is not None:
                return _checked_meeting(parsed.date(), "the Date header")
    return None


def date_from_mtime(mtime: datetime, tz: ZoneInfo) -> date:
    """Calendar date of a file's modification time in the meeting's timezone."""
    aware = mtime if mtime.tzinfo is not None else mtime.astimezone()
    return aware.astimezone(tz).date()
