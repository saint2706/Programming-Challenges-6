"""Multi-timezone meeting scheduler core: DST-aware overlap finding.

Given participants each with an IANA timezone and a local working-hours
window, find the UTC intervals where enough of them are simultaneously
within their working hours on a given calendar date -- "calendar date" being
interpreted in a *reference* timezone (the organizer's "let's meet on the
24th, my time"), since a bare date has no meaning without one.

DST-awareness is not a special case here: every local wall-clock time is
built as a timezone-aware ``datetime`` via ``zoneinfo``, which resolves the
correct UTC offset for that exact date automatically, including on the day a
region's clocks change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, available_timezones

UTC = ZoneInfo("UTC")
_WEEKDAYS = frozenset(range(5))  # Mon=0 .. Sun=6


@dataclass(frozen=True)
class Participant:
    name: str
    tz: str  # IANA zone name, e.g. "America/New_York"
    work_start: time
    work_end: time
    days: frozenset[int] = field(default_factory=lambda: _WEEKDAYS)

    def __post_init__(self) -> None:
        if self.work_start >= self.work_end:
            raise ValueError(
                f"{self.name}: work_start ({self.work_start}) must be before work_end "
                f"({self.work_end}); overnight shifts spanning midnight are not supported"
            )
        if not self.days or not self.days.issubset(range(7)):
            raise ValueError(
                f"{self.name}: days must be a non-empty subset of 0..6 (Mon..Sun)"
            )
        ZoneInfo(self.tz)  # raises if the zone name is unknown


@dataclass(frozen=True)
class OverlapWindow:
    start_utc: datetime
    end_utc: datetime
    available: tuple[str, ...]  # participant names available during this window

    def duration_minutes(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds() / 60


def parse_participant_spec(spec: str) -> Participant:
    """Parse ``Name|Area/City|HH:MM-HH:MM[|day,day,...]`` into a Participant.

    Pipe-delimited because IANA zone names and hour ranges both already use
    ``:`` and ``/``. The optional 4th field is a comma-separated list of
    weekday numbers (0=Monday .. 6=Sunday); it defaults to Mon-Fri.
    """
    parts = spec.split("|")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"Invalid participant spec {spec!r}; expected Name|Area/City|HH:MM-HH:MM[|days]"
        )
    name, tz, hours = (p.strip() for p in parts[:3])
    try:
        start_s, end_s = hours.split("-")
        work_start = time.fromisoformat(start_s.strip())
        work_end = time.fromisoformat(end_s.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid hours {hours!r} in spec {spec!r}; expected HH:MM-HH:MM"
        ) from exc

    days = _WEEKDAYS
    if len(parts) == 4 and parts[3].strip():
        try:
            days = frozenset(int(d) for d in parts[3].split(","))
        except ValueError as exc:
            raise ValueError(
                f"Invalid days {parts[3]!r} in spec {spec!r}; expected comma-separated 0..6"
            ) from exc

    return Participant(
        name=name, tz=tz, work_start=work_start, work_end=work_end, days=days
    )


def _participant_intervals(
    participant: Participant, window_start_utc: datetime, window_end_utc: datetime
) -> list[tuple[datetime, datetime]]:
    """UTC (start, end) working intervals for ``participant`` that fall inside
    [window_start_utc, window_end_utc), built by walking their local calendar
    days with a one-day buffer on each side (their local day boundaries
    rarely line up with the reference window's)."""
    tz = ZoneInfo(participant.tz)
    intervals: list[tuple[datetime, datetime]] = []

    probe = window_start_utc.astimezone(tz).date() - timedelta(days=1)
    last = window_end_utc.astimezone(tz).date() + timedelta(days=1)
    while probe <= last:
        if probe.weekday() in participant.days:
            local_start = datetime.combine(probe, participant.work_start, tzinfo=tz)
            local_end = datetime.combine(probe, participant.work_end, tzinfo=tz)
            utc_start = local_start.astimezone(UTC)
            utc_end = local_end.astimezone(UTC)
            clipped_start = max(utc_start, window_start_utc)
            clipped_end = min(utc_end, window_end_utc)
            if clipped_start < clipped_end:
                intervals.append((clipped_start, clipped_end))
        probe += timedelta(days=1)
    return intervals


def find_overlaps(
    participants: list[Participant],
    reference_date: date,
    reference_tz: str,
    min_participants: int | None = None,
) -> list[OverlapWindow]:
    """Find UTC windows where at least ``min_participants`` are at work.

    ``reference_date`` is interpreted as a calendar day in ``reference_tz``
    (midnight to midnight local, whatever that is in UTC -- 23, 24, or 25
    hours on a DST transition day). Every participant's working hours are
    computed from their *own* timezone and clipped to that window.
    """
    if not participants:
        return []
    if len({p.name for p in participants}) != len(participants):
        raise ValueError("Participant names must be unique")

    min_participants = (
        len(participants) if min_participants is None else min_participants
    )
    if not (1 <= min_participants <= len(participants)):
        raise ValueError(
            f"min_participants must be between 1 and {len(participants)}, got {min_participants}"
        )

    ref_zone = ZoneInfo(reference_tz)
    window_start_utc = datetime.combine(
        reference_date, time(0, 0), tzinfo=ref_zone
    ).astimezone(UTC)
    window_end_utc = datetime.combine(
        reference_date + timedelta(days=1), time(0, 0), tzinfo=ref_zone
    ).astimezone(UTC)

    events: list[tuple[datetime, int, str]] = []
    for p in participants:
        for start, end in _participant_intervals(p, window_start_utc, window_end_utc):
            events.append((start, 1, p.name))
            events.append((end, -1, p.name))

    if not events:
        return []

    times = sorted({t for t, _, _ in events})
    deltas_at: dict[datetime, list[tuple[int, str]]] = {}
    for t, d, name in events:
        deltas_at.setdefault(t, []).append((d, name))

    active: set[str] = set()
    segments: list[tuple[datetime, datetime, frozenset[str]]] = []
    for i in range(len(times) - 1):
        for d, name in deltas_at[times[i]]:
            if d == 1:
                active.add(name)
            else:
                active.discard(name)
        segments.append((times[i], times[i + 1], frozenset(active)))

    windows = [
        OverlapWindow(start, end, tuple(sorted(names)))
        for start, end, names in segments
        if len(names) >= min_participants
    ]

    # Merge only truly-adjacent segments with the *identical* participant set.
    merged: list[OverlapWindow] = []
    for w in windows:
        if (
            merged
            and merged[-1].end_utc == w.start_utc
            and merged[-1].available == w.available
        ):
            merged[-1] = OverlapWindow(merged[-1].start_utc, w.end_utc, w.available)
        else:
            merged.append(w)
    return merged


def format_window(window: OverlapWindow, display_tz: str) -> tuple[str, str]:
    """Render a window's start/end localized to ``display_tz``."""
    zone = ZoneInfo(display_tz)
    start_local = window.start_utc.astimezone(zone)
    end_local = window.end_utc.astimezone(zone)
    return start_local.strftime("%Y-%m-%d %H:%M %Z"), end_local.strftime("%H:%M %Z")


def search_timezones(query: str, limit: int = 20) -> list[str]:
    """Case-insensitive substring search over IANA zone names."""
    query = query.lower()
    matches = sorted(z for z in available_timezones() if query in z.lower())
    return matches[:limit]
