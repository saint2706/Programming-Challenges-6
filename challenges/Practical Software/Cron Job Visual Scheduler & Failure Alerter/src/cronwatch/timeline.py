"""A horizontal timeline of what ran and what is about to: one row per job, one column per time slice.

The left of the ``now`` marker is history (green ran, red failed, yellow scheduled but no run was
recorded); the right is the schedule (taller bars mean more runs in that slice). A summary names the
minutes where many jobs fire together, which is the thing worth rearranging in a crontab.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo

from rich.text import Text

from cronwatch.crontab import Job
from cronwatch.store import OK, RUNNING, Run

LABEL_MAX = 22
EMPTY, NOW = "·", "┆"
BARS = [(1, "▂"), (2, "▄"), (4, "▆"), (10, "█")]  # (minimum runs in the slice, glyph)
MAX_INSTANTS = 4000


@dataclass(frozen=True)
class Cell:
    char: str
    style: str


@dataclass(frozen=True)
class Row:
    label: str
    cells: list[Cell]


@dataclass(frozen=True)
class Timeline:
    start: datetime
    end: datetime
    now: datetime
    tz: tzinfo
    slice: timedelta
    axis: str
    rows: list[Row]
    busy: list[tuple[datetime, list[str]]]  # minutes where several jobs start together

    @property
    def width(self) -> int:
        return len(self.axis)

    @property
    def now_column(self) -> int:
        return int((self.now - self.start) / self.slice)


def _bucket(
    moment: datetime, start: datetime, slice_: timedelta, width: int
) -> int | None:
    index = int((moment - start) / slice_)
    return index if 0 <= index < width else None


def _bar(count: int) -> str:
    glyph = BARS[0][1]
    for minimum, candidate in BARS:
        if count >= minimum:
            glyph = candidate
    return glyph


def _axis(
    start: datetime, end: datetime, tz: tzinfo, width: int, slice_: timedelta
) -> str:
    """Hour labels at whole-hour boundaries, spaced so they cannot overlap."""
    axis = [" "] * width
    span_hours = (end - start).total_seconds() / 3600
    step = (
        1
        if span_hours <= 8
        else 2
        if span_hours <= 16
        else 3
        if span_hours <= 30
        else 6
        if span_hours <= 72
        else 24
    )
    tick = start.astimezone(tz).replace(minute=0, second=0, microsecond=0)
    while tick.astimezone(UTC) <= end:
        column = _bucket(tick.astimezone(UTC), start, slice_, width)
        if column is not None and tick.hour % step == 0:
            label = f"{tick:%H:%M}" if step < 24 else f"{tick:%d %b}"
            if column + len(label) <= width and all(
                c == " " for c in axis[max(0, column - 1) : column + len(label) + 1]
            ):
                axis[column : column + len(label)] = label
        tick += timedelta(hours=1)
    return "".join(axis)


def _history_outcomes(
    job: Job, runs: dict[str, list[Run]], start: datetime, slice_: timedelta, width: int
) -> dict[int, str]:
    """slice index -> ``ok`` | ``fail`` | ``run``; a failure in a slice outranks successes in it."""
    outcomes: dict[int, str] = {}
    for run in runs.get(job.job_id or "", []):
        bucket = _bucket(run.started_at.astimezone(UTC), start, slice_, width)
        if bucket is None:
            continue
        outcome = (
            "run" if run.status == RUNNING else "ok" if run.status == OK else "fail"
        )
        outcomes[bucket] = (
            "fail" if "fail" in (outcome, outcomes.get(bucket)) else outcome
        )
    return outcomes


def _inactive_note(job: Job) -> str | None:
    if not job.enabled:
        return "off"
    if job.expr is None:
        return "invalid"
    return "reboot" if job.expr.reboot else None


def build_timeline(
    jobs: list[Job],
    runs: dict[str, list[Run]],
    now: datetime,
    tz: tzinfo,
    width: int = 72,
    past: timedelta = timedelta(hours=6),
    future: timedelta = timedelta(hours=18),
    grace: timedelta = timedelta(minutes=5),
    busy_top: int = 5,
    monitored_since: dict[str, datetime] | None = None,
) -> Timeline:
    """Rows for ``jobs`` over ``[now - past, now + future]``; ``runs`` maps job id to its recorded runs.

    A past slice is marked "no run recorded" only after the job's ``monitored_since`` time: before
    that nobody was watching, and a freshly wrapped job must not look like it missed everything.
    """
    monitored_since = monitored_since or {}
    now = now.astimezone(UTC)
    start, end = now - past, now + future
    slice_ = (end - start) / width
    now_column = _bucket(now, start, slice_, width)
    rows: list[Row] = []
    starts: dict[datetime, list[str]] = defaultdict(list)
    for job in jobs:
        label = (
            job.label
            if len(job.label) <= LABEL_MAX
            else job.label[: LABEL_MAX - 1] + "…"
        ).ljust(LABEL_MAX)
        cells = [Cell(EMPTY, "dim") for _ in range(width)]
        if now_column is not None:
            cells[now_column] = Cell(NOW, "bold cyan")
        if (note := _inactive_note(job)) is not None:
            rows.append(Row(label, _with_note(cells, note)))
            continue
        history = _history_outcomes(job, runs, start, slice_, width)
        upcoming: Counter[int] = Counter()
        for instant in job.expr.runs_between(start, end, tz, limit=MAX_INSTANTS):
            moment = instant.astimezone(UTC)
            bucket = _bucket(moment, start, slice_, width)
            if bucket is None:
                continue
            if moment > now:
                upcoming[bucket] += 1
                starts[moment].append(job.label)
            elif bucket not in history:
                slice_end = start + slice_ * (bucket + 1)
                watched = (
                    job.job_id in monitored_since
                    and moment >= monitored_since[job.job_id]
                )
                if not job.job_id:
                    cells[bucket] = Cell(
                        "·", ""
                    )  # scheduled, outcome unknown: the job is not wrapped
                elif watched and slice_end < now - grace:
                    cells[bucket] = Cell(
                        "○", "yellow"
                    )  # should have run and no record says it did
        for bucket, count in upcoming.items():
            if (
                bucket != now_column
            ):  # keep the "you are here" marker visible for frequent jobs
                cells[bucket] = Cell(_bar(count), "blue")
        glyphs = {
            "ok": Cell("●", "green"),
            "fail": Cell("✗", "bold red"),
            "run": Cell("◐", "cyan"),
        }
        for bucket, outcome in history.items():
            cells[bucket] = glyphs[outcome]
        rows.append(Row(label, cells))
    crowded = [(moment, names) for moment, names in starts.items() if len(names) >= 2]
    crowded.sort(key=lambda item: (-len(item[1]), item[0]))
    return Timeline(
        start,
        end,
        now,
        tz,
        slice_,
        _axis(start, end, tz, width, slice_),
        rows,
        crowded[:busy_top],
    )


def _with_note(cells: list[Cell], note: str) -> list[Cell]:
    """Overwrite the start of an inactive job's row with ``(off)`` / ``(invalid)`` / ``(reboot)``."""
    cells = list(cells)
    for offset, char in enumerate(f" ({note})"):
        if offset < len(cells):
            cells[offset] = Cell(char, "dim italic")
    return cells


def render_timeline(timeline: Timeline) -> Text:
    pad = " " * (LABEL_MAX + 2)
    out = Text()
    out.append(pad + timeline.axis + "\n", style="dim")
    for row in timeline.rows:
        out.append(row.label + "  ")
        for cell in row.cells:
            out.append(cell.char, style=cell.style or None)
        out.append("\n")
    legend = "● ran  ✗ failed  ○ no run recorded  ▂▄▆█ scheduled (more runs → taller)  ┆ now  · not monitored"
    out.append("\n" + pad + legend + "\n", style="dim")
    if timeline.busy:
        out.append("\nBusiest minutes ahead (jobs starting together):\n", style="bold")
        for moment, names in timeline.busy:
            shown = ", ".join(names[:4]) + (
                f" +{len(names) - 4} more" if len(names) > 4 else ""
            )
            out.append(
                f"  {moment.astimezone(timeline.tz):%a %H:%M}  {len(names)} jobs: {shown}\n"
            )
    return out
