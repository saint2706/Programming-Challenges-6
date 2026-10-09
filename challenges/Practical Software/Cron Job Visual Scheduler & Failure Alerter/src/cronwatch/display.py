"""Small formatting helpers shared by the CLI tables and the TUI."""

from datetime import UTC, datetime, timedelta, tzinfo

from cronwatch.crontab import Job
from cronwatch.store import ERROR, FAILED, OK, RUNNING, TIMEOUT, Run

STATUS_GLYPHS = {
    OK: ("●", "green"),
    FAILED: ("✗", "bold red"),
    TIMEOUT: ("⏱", "bold red"),
    ERROR: ("✗", "bold red"),
    RUNNING: ("◐", "cyan"),
}


def humanize(delta: timedelta) -> str:
    """``in 4m``, ``in 2h 5m``, ``3d ago``: two units at most."""
    seconds = int(delta.total_seconds())
    future = seconds >= 0
    seconds = abs(seconds)
    if seconds < 45:
        return "now" if future else "just now"
    minutes = round(seconds / 60)
    if minutes < 60:
        body = f"{minutes}m"
    elif minutes < 600:
        hours, rest = divmod(minutes, 60)
        body = f"{hours}h" + (f" {rest}m" if rest else "")
    elif minutes < 60 * 24:
        body = f"{round(minutes / 60)}h"
    else:
        hours = round(minutes / 60)
        if hours < 240:
            days, rest = divmod(hours, 24)
            body = f"{days}d" + (f" {rest}h" if rest else "")
        else:
            body = f"{round(hours / 24)}d"
    return f"in {body}" if future else f"{body} ago"


def local(moment: datetime, tz: tzinfo, with_date: bool = True) -> str:
    return (
        f"{moment.astimezone(tz):%a %d %b %H:%M}"
        if with_date
        else f"{moment.astimezone(tz):%H:%M}"
    )


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    return (
        f"{minutes}m {rest:02d}s"
        if minutes < 60
        else f"{minutes // 60}h {minutes % 60:02d}m"
    )


def next_run_text(job: Job, now: datetime, tz: tzinfo) -> str:
    if job.error:
        return "invalid schedule"
    if not job.enabled:
        return "disabled"
    if job.expr is None or job.expr.reboot:
        return "at startup" if job.expr else "—"
    following = job.expr.next_after(now.astimezone(UTC), tz)
    if following is None:
        return "never"
    return f"{humanize(following - now)} ({local(following, tz)})"


def last_run_text(run: Run | None, now: datetime, tz: tzinfo) -> tuple[str, str]:
    """(text, rich style) for a job's most recent run."""
    if run is None:
        return "no runs yet", "dim"
    glyph, style = STATUS_GLYPHS.get(run.status, ("?", ""))
    detail = (
        f" exit {run.exit_code}"
        if run.status in {FAILED, ERROR} and run.exit_code is not None
        else ""
    )
    if run.status == TIMEOUT:
        detail = " timed out"
    return f"{glyph} {humanize(run.started_at - now)}{detail}", style
