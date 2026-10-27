"""Five-field cron expressions (minute hour day-of-month month day-of-week) and the data-interval rule.

Supported: ``*``, lists (``1,3``), ranges (``1-5``), steps (``*/15``, ``1-5/2``). Day-of-week 0 and 7
are Sunday. When both day fields are restricted a day matches if either does (the cron rule).
Times are naive local wall-clock times; there is no timezone handling.

Data intervals are date-grained, so a schedule must fire at most once a day (single minute and hour).
A fire at ``F`` covers the dates from the previous fire's date up to, not including, ``F``'s date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]  # minute, hour, dom, month, dow


@dataclass(frozen=True)
class CronSpec:
    expr: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    days_restricted: bool
    weekdays_restricted: bool


def _field(text: str, low: int, high: int) -> frozenset[int]:
    values: set[int] = set()
    for part in text.split(","):
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            step = int(step_text)
            if step < 1:
                raise ValueError(f"cron step must be at least 1 in {text!r}")
        if part == "*":
            start, end = low, high
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = int(part)
            end = high if step != 1 else start
        if start < low or end > high or start > end:
            raise ValueError(f"cron value out of range {low}-{high} in {text!r}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


def parse(expr: str) -> CronSpec:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"cron needs five fields, got {len(parts)}: {expr!r}")
    minute, hour, dom, month, dow = (
        _field(p, lo, hi) for p, (lo, hi) in zip(parts, _RANGES, strict=True)
    )
    weekdays = frozenset(d % 7 for d in dow)
    return CronSpec(
        expr=expr,
        minutes=minute,
        hours=hour,
        days=dom,
        months=month,
        weekdays=weekdays,
        days_restricted=parts[2] != "*",
        weekdays_restricted=parts[4] != "*",
    )


def _day_matches(spec: CronSpec, day: date) -> bool:
    if day.month not in spec.months:
        return False
    dom_ok = day.day in spec.days
    dow_ok = day.isoweekday() % 7 in spec.weekdays  # Sunday is 0
    if spec.days_restricted and spec.weekdays_restricted:
        return dom_ok or dow_ok
    return dom_ok and dow_ok


def _times_of_day(spec: CronSpec) -> list[tuple[int, int]]:
    return sorted((h, m) for h in spec.hours for m in spec.minutes)


def next_after(spec: CronSpec, after: datetime) -> datetime:
    """The first fire strictly after ``after``."""
    times = _times_of_day(spec)
    day = after.date()
    for _ in range(366 * 9):  # a Feb-29-only schedule repeats within 8 years
        if _day_matches(spec, day):
            for hour, minute in times:
                candidate = datetime(day.year, day.month, day.day, hour, minute)
                if candidate > after:
                    return candidate
        day += timedelta(days=1)
    raise ValueError(f"cron {spec.expr!r} never fires")


def previous_before(spec: CronSpec, before: datetime) -> datetime:
    """The last fire strictly before ``before``."""
    times = _times_of_day(spec)
    day = before.date()
    for _ in range(366 * 9):
        if _day_matches(spec, day):
            for hour, minute in reversed(times):
                candidate = datetime(day.year, day.month, day.day, hour, minute)
                if candidate < before:
                    return candidate
        day -= timedelta(days=1)
    raise ValueError(f"cron {spec.expr!r} has no fire before {before}")


def check_daily(spec: CronSpec) -> None:
    """Reject schedules that fire more than once a day, which date-grained intervals cannot express."""
    if len(spec.minutes) != 1 or len(spec.hours) != 1:
        raise ValueError(
            f"{spec.expr!r} fires more than once a day; a schedule must fire at most once a day"
        )


def interval_for_fire(spec: CronSpec, fire: datetime) -> tuple[date, date]:
    """``[start, end)`` dates covered by the run that fires at ``fire``."""
    check_daily(spec)
    return previous_before(spec, fire).date(), fire.date()


def due_fires(
    spec: CronSpec, after: datetime, now: datetime, limit: int = 1000
) -> list[datetime]:
    """Every fire in ``(after, now]``, oldest first, at most ``limit`` of them."""
    fires: list[datetime] = []
    cursor = after
    while len(fires) < limit:
        nxt = next_after(spec, cursor)
        if nxt > now:
            break
        fires.append(nxt)
        cursor = nxt
    return fires
