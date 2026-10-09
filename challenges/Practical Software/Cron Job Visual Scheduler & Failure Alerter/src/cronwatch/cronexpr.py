"""Five-field cron expressions: parsing, next-run search and plain-English descriptions.

Follows Vixie cron (what ``crontab(5)`` on Linux and macOS documents), including the rule people
trip over: when *both* day-of-month and day-of-week are restricted, a day matches if *either* does.
Times are wall-clock times in a given timezone. A wall-clock time that does not exist (the hour
skipped by spring-forward) never fires; a repeated hour (fall-back) fires once, in its first pass.
"""

import calendar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo

MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
DAY_NAMES = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}
MACROS = {
    "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *", "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@hourly": "0 * * * *",
}  # fmt: skip
SEARCH_YEARS = (
    8  # long enough to find Feb 29 on a Monday, short enough to reject Feb 31 quickly
)
_DAY_LABELS = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
_MONTH_LABELS = [""] + list(calendar.month_name)[1:]


class CronSyntaxError(ValueError):
    """The expression is not valid cron; the message says which field and why."""


@dataclass(frozen=True)
class CronExpr:
    source: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]  # day of month
    months: frozenset[int]
    weekdays: frozenset[int]  # 0 = Sunday
    dom_star: bool
    dow_star: bool
    reboot: bool = False

    def day_matches(self, day: datetime) -> bool:
        in_dom = day.day in self.days
        in_dow = (day.isoweekday() % 7) in self.weekdays
        if self.dom_star and self.dow_star:
            return True
        if self.dom_star:
            return in_dow
        if self.dow_star:
            return in_dom
        return in_dom or in_dow

    def matches(self, when: datetime) -> bool:
        """Whether the wall-clock minute of ``when`` is a scheduled one."""
        if self.reboot:
            return False
        return (
            when.minute in self.minutes
            and when.hour in self.hours
            and when.month in self.months
            and self.day_matches(when)
        )

    def _next_wall_clock(self, after: datetime) -> datetime | None:
        """First scheduled minute strictly after the naive wall-clock time ``after``."""
        t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = t + timedelta(days=366 * SEARCH_YEARS)
        while t <= limit:
            if t.month not in self.months:
                t = (t.replace(day=1, hour=0, minute=0) + timedelta(days=32)).replace(
                    day=1
                )
            elif not self.day_matches(t):
                t = t.replace(hour=0, minute=0) + timedelta(days=1)
            elif t.hour not in self.hours:
                t = t.replace(minute=0) + timedelta(hours=1)
            elif t.minute not in self.minutes:
                t += timedelta(minutes=1)
            else:
                return t
        return None

    def next_after(self, after: datetime, tz: tzinfo | None = None) -> datetime | None:
        """The next run strictly after ``after``, as an aware datetime in ``tz``; None if it never fires.

        ``tz`` defaults to ``after``'s own timezone, else UTC.
        """
        if self.reboot:
            return None
        zone = tz or after.tzinfo or UTC
        cursor = after.astimezone(zone).replace(tzinfo=None) if after.tzinfo else after
        while True:
            candidate = self._next_wall_clock(cursor)
            if candidate is None:
                return None
            aware = candidate.replace(tzinfo=zone)
            if aware.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == candidate:
                return aware
            cursor = (
                candidate  # lands in a spring-forward gap: that minute never happens
            )

    def next_runs(
        self, after: datetime, count: int, tz: tzinfo | None = None
    ) -> list[datetime]:
        runs: list[datetime] = []
        while len(runs) < count:
            following = self.next_after(runs[-1] if runs else after, tz)
            if following is None:
                break
            runs.append(following)
        return runs

    def runs_between(
        self,
        start: datetime,
        end: datetime,
        tz: tzinfo | None = None,
        limit: int = 5000,
    ) -> list[datetime]:
        """Runs in ``(start, end]``."""
        runs: list[datetime] = []
        cursor = start
        while len(runs) < limit:
            following = self.next_after(cursor, tz)
            if following is None or following > end:
                break
            runs.append(following)
            cursor = following
        return runs

    def describe(self) -> str:
        return describe(self)


def _parse_value(
    token: str, names: dict[str, int], lo: int, hi: int, field: str
) -> int:
    lowered = token.lower()
    if lowered in names:
        return names[lowered]
    if not token.isdigit():
        raise CronSyntaxError(
            f"{field}: {token!r} is not a number" + (" or name" if names else "")
        )
    value = int(token)
    if not lo <= value <= hi:
        raise CronSyntaxError(f"{field}: {value} is outside {lo}-{hi}")
    return value


def _parse_field(
    text: str, lo: int, hi: int, field: str, names: dict[str, int] | None = None
) -> frozenset[int]:
    names = names or {}
    values: set[int] = set()
    if not text:
        raise CronSyntaxError(f"{field}: empty")
    for part in text.split(","):
        if not part:
            raise CronSyntaxError(f"{field}: empty item in list {text!r}")
        base, slash, step_text = part.partition("/")
        if slash:
            if not step_text.isdigit() or int(step_text) < 1:
                raise CronSyntaxError(
                    f"{field}: step {step_text!r} must be a positive number"
                )
            step = int(step_text)
        else:
            step = 1
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            first, _, last = base.partition("-")
            start, end = (
                _parse_value(first, names, lo, hi, field),
                _parse_value(last, names, lo, hi, field),
            )
            if start > end:
                raise CronSyntaxError(
                    f"{field}: range {base} runs backwards; write it as two ranges, e.g. {start}-{hi},{lo}-{end}"
                )
        else:
            start = _parse_value(base, names, lo, hi, field)
            end = hi if slash else start
        values.update(range(start, end + 1, step))
    return frozenset(values)


def parse(text: str) -> CronExpr:
    """Parse a five-field expression or an ``@macro``; raises :class:`CronSyntaxError`."""
    stripped = text.strip()
    if stripped.lower() == "@reboot":
        empty: frozenset[int] = frozenset()
        return CronExpr(
            stripped, empty, empty, empty, empty, empty, True, True, reboot=True
        )
    if stripped.startswith("@"):
        expansion = MACROS.get(stripped.lower())
        if expansion is None:
            raise CronSyntaxError(
                f"unknown macro {stripped!r}; use one of {', '.join(sorted(MACROS))} or @reboot"
            )
        stripped_fields = expansion
    else:
        stripped_fields = stripped
    parts = stripped_fields.split()
    if len(parts) != 5:
        raise CronSyntaxError(
            f"expected 5 fields (minute hour day-of-month month day-of-week), got {len(parts)}"
        )
    minute, hour, dom, month, dow = parts
    weekdays = _parse_field(dow, 0, 7, "day-of-week", DAY_NAMES)
    expr = CronExpr(
        source=stripped,
        minutes=_parse_field(minute, 0, 59, "minute"),
        hours=_parse_field(hour, 0, 23, "hour"),
        days=_parse_field(dom, 1, 31, "day-of-month"),
        months=_parse_field(month, 1, 12, "month", MONTH_NAMES),
        weekdays=frozenset(d % 7 for d in weekdays),  # 7 is Sunday too
        dom_star=dom.startswith("*"),
        dow_star=dow.startswith("*"),
    )
    if expr.next_after(datetime(2024, 1, 1, tzinfo=UTC)) is None:
        raise CronSyntaxError(
            "this schedule never fires (for example day 31 of February)"
        )
    return expr


def _clock(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _even_step(values: frozenset[int], lo: int, hi: int) -> int | None:
    """N when ``values`` is exactly ``lo, lo+N, ...`` up to ``hi``; None otherwise."""
    ordered = sorted(values)
    if len(ordered) < 2 or ordered[0] != lo:
        return None
    step = ordered[1] - ordered[0]
    return step if ordered == list(range(lo, hi + 1, step)) else None


def _weekday_text(days: frozenset[int]) -> str:
    if days == frozenset({1, 2, 3, 4, 5}):
        return "Monday to Friday"
    if days == frozenset({0, 6}):
        return "Saturday and Sunday"
    return _join([_DAY_LABELS[d] for d in sorted(days)])


def describe(expr: CronExpr) -> str:
    """Plain-English reading, e.g. ``At 03:00 on Monday to Friday``."""
    if expr.reboot:
        return "At system startup"
    all_minutes, all_hours = len(expr.minutes) == 60, len(expr.hours) == 24
    step = _even_step(expr.minutes, 0, 59)
    if all_minutes and all_hours:
        time_text = "Every minute"
    elif all_hours and step:
        time_text = f"Every {step} minutes"
    elif all_hours and len(expr.minutes) == 1:
        (minute,) = expr.minutes
        time_text = (
            "Every hour on the hour"
            if minute == 0
            else f"At minute {minute} of every hour"
        )
    elif len(expr.hours) * len(expr.minutes) <= 4:
        times = [_clock(h, m) for h in sorted(expr.hours) for m in sorted(expr.minutes)]
        time_text = "At " + _join(times)
    elif (hstep := _even_step(expr.hours, 0, 23)) and len(expr.minutes) == 1:
        (minute,) = expr.minutes
        time_text = f"At minute {minute} of every {hstep}th hour"
    else:
        time_text = f"At minute {_join([str(m) for m in sorted(expr.minutes)])} past hour {_join([str(h) for h in sorted(expr.hours)])}"
    parts = [time_text]
    day_bits = []
    if not expr.dom_star:
        day_bits.append(
            "on day " + _join([str(d) for d in sorted(expr.days)]) + " of the month"
        )
    if not expr.dow_star:
        day_bits.append("on " + _weekday_text(expr.weekdays))
    if day_bits:
        parts.append(" or ".join(day_bits))
    if len(expr.months) != 12:
        parts.append("in " + _join([_MONTH_LABELS[m] for m in sorted(expr.months)]))
    return " ".join(parts)
