"""Missed-run detection: a job that never started leaves no run to fail, so nothing would alert.

The wrapper can only report jobs that *ran*. If the cron daemon is stopped, the machine was off, or
the crontab line was broken, silence is the symptom. ``find_missed`` compares what the schedule says
should have happened with what the history says did.
"""

import socket
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo

from cronwatch.alerts import Alert, Alerter
from cronwatch.crontab import Job
from cronwatch.store import JobState, Store

EARLY = timedelta(
    seconds=30
)  # a run may start marginally before the scheduled minute's clock reading
LOOKBACK = timedelta(
    hours=24
)  # after downtime, do not dredge up more than a day of misses
MAX_INSTANTS = 1500


@dataclass(frozen=True)
class Missed:
    job_id: str
    command: str
    instants: tuple[datetime, ...]  # scheduled times with no recorded run


def find_missed(
    job: Job,
    store: Store,
    cursor: datetime,
    now: datetime,
    grace: timedelta,
    tz: tzinfo,
) -> list[datetime]:
    """Scheduled instants in ``(cursor, now - grace]`` that have no run recorded for this job.

    A run counts for the instant it started at or shortly after, but never for a later instant, so one
    late run cannot hide a separate missed one. ``grace`` is how long a run may start after its minute.
    """
    assert job.expr is not None and job.job_id is not None
    horizon = now - grace
    start = max(cursor, horizon - LOOKBACK)
    instants = job.expr.runs_between(start, horizon, tz, limit=MAX_INSTANTS)
    if not instants:
        return []
    runs = [
        r.started_at
        for r in store.runs_between(
            job.job_id, instants[0] - EARLY, horizon + grace + EARLY
        )
    ]
    missed = []
    for index, instant in enumerate(instants):
        following = (
            instants[index + 1]
            if index + 1 < len(instants)
            else job.expr.next_after(instant, tz)
        )
        upper = min(instant + grace, following) if following else instant + grace
        if not any(instant - EARLY <= started < upper for started in runs):
            missed.append(instant)
    return missed


def check_missed(
    jobs: list[Job],
    store: Store,
    alerter: Alerter,
    now: datetime,
    grace: timedelta,
    tz: tzinfo,
    notify: bool = True,
) -> list[Missed]:
    """Evaluate every monitored job once; at most one alert per job per check, however many runs were missed."""
    now = now.astimezone(UTC)
    horizon = now - grace
    found: list[Missed] = []
    for job in jobs:
        if not job.job_id:
            continue
        state = store.get_state(job.job_id)
        evaluable = job.enabled and job.expr is not None and not job.expr.reboot
        if state.monitor_cursor is None:
            # First sighting: we cannot know what happened before, so start counting from now.
            store.save_state(_with_cursor(state, now))
            continue
        if not evaluable:
            # Skipped jobs still advance the cursor to *now* (not now - grace): nothing that happened while
            # the job was paused may be judged after it is re-enabled.
            store.save_state(_with_cursor(state, max(state.monitor_cursor, now)))
            continue
        missed = find_missed(job, store, state.monitor_cursor, now, grace, tz)
        store.save_state(_with_cursor(state, max(state.monitor_cursor, horizon)))
        if not missed:
            continue
        found.append(Missed(job.job_id, job.command, tuple(missed)))
        if notify:
            alerter.dispatch(
                Alert(
                    "missed",
                    job.job_id,
                    socket.gethostname(),
                    now,
                    job.command,
                    scheduled_for=missed[-1],
                    missed_count=len(missed),
                )
            )
    return found


def _with_cursor(state: JobState, cursor: datetime) -> JobState:
    return replace(state, monitor_cursor=cursor)
