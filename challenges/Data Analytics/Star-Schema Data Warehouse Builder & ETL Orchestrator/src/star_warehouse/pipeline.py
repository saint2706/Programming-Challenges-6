"""Runs: one data interval through the DAG, a backfill over months, and the cron scheduler tick.

Every run writes an ``ops.run`` row and one ``ops.task_run`` row per finished task attempt or skip.
A run that was killed leaves its ``ops.run`` row at ``running``; the next run marks it ``interrupted``.
Nothing else needs repair: the killed run's uncommitted task transaction was never committed.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

from star_warehouse import cron
from star_warehouse.dag import Dag, Task, TaskResult
from star_warehouse.dag import run as run_dag
from star_warehouse.paths import project_root
from star_warehouse.tasks import Batch, build_tasks
from star_warehouse.warehouse import connect

REF_LOOKUP = Path("sample_data") / "ref_country_region.csv"


@dataclass
class RunOutcome:
    run_id: str
    status: str  # "succeeded" | "failed"
    start: date
    end: date
    results: dict[str, TaskResult] = field(default_factory=dict)

    @property
    def failed_tasks(self) -> list[str]:
        return [n for n, r in self.results.items() if r.status == "failed"]


def default_feed(root: Path | None = None) -> Path:
    """The full feed when it has been fetched, else the committed sample."""
    root = root or project_root()
    full = root / "data" / "feed.parquet"
    return full if full.exists() else root / "sample_data" / "feed_sample.parquet"


def default_lookup(root: Path | None = None) -> Path:
    return (root or project_root()) / REF_LOOKUP


def _with_faults(
    tasks: list[Task], faults: Mapping[str, Sequence[Exception]] | None
) -> list[Task]:
    """Test hook: the named task raises each queued exception on its next attempts, then runs normally."""
    if not faults:
        return tasks
    out: list[Task] = []
    for task in tasks:
        queue = list(faults.get(task.name, ()))
        if not queue:
            out.append(task)
            continue

        def faulty(task: Task = task, queue: list[Exception] = queue):
            if queue:
                raise queue.pop(0)
            return task.fn()

        out.append(replace(task, fn=faulty))
    return out


def _mark_stale(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "UPDATE ops.run SET status = 'interrupted', finished_at = ? WHERE status = 'running'",
        [datetime.now()],
    )


def run_interval(
    db_path: Path,
    feed: Path,
    lookup: Path,
    start: date,
    end: date,
    *,
    trigger: str = "manual",
    faults: Mapping[str, Sequence[Exception]] | None = None,
    backoff: float = 1.0,
    sleep=time.sleep,
) -> RunOutcome:
    """Load the feed rows that arrived in ``[start, end)`` and bring the star schema up to date."""
    con = connect(db_path)
    try:
        _mark_stale(con)
        run_id = uuid.uuid4().hex
        con.execute(
            "INSERT INTO ops.run VALUES (?, ?, ?, ?, 'running', ?, NULL, NULL)",
            [run_id, start, end, trigger, datetime.now()],
        )
        batch = Batch(con=con, feed=feed, lookup=lookup, start=start, end=end)
        dag = Dag(_with_faults(build_tasks(batch), faults))
        began: dict[str, datetime] = {}

        def observe(event: str, name: str, number: int, status, error) -> None:
            now = datetime.now()
            if event == "start":
                began[name] = now
            elif event == "skip":
                con.execute(
                    "INSERT INTO ops.task_run VALUES (?, ?, 0, 'skipped', ?, ?, ?)",
                    [run_id, name, now, now, error],
                )
            else:
                con.execute(
                    "INSERT INTO ops.task_run VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [run_id, name, number, status, began.get(name, now), now, error],
                )

        results = run_dag(dag, backoff=backoff, sleep=sleep, observer=observe)
        status = (
            "succeeded"
            if all(r.status == "succeeded" for r in results.values())
            else "failed"
        )
        con.execute(
            "UPDATE ops.run SET status = ?, finished_at = ? WHERE run_id = ?",
            [status, datetime.now(), run_id],
        )
        return RunOutcome(run_id, status, start, end, results)
    finally:
        con.close()


def months(first: date, last_exclusive: date) -> list[tuple[date, date]]:
    """Calendar months as ``[start, end)`` pairs, from the month of ``first`` up to ``last_exclusive``."""
    out: list[tuple[date, date]] = []
    cursor = first.replace(day=1)
    while cursor < last_exclusive:
        nxt = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        out.append((cursor, nxt))
        cursor = nxt
    return out


def backfill(
    db_path: Path,
    feed: Path,
    lookup: Path,
    first: date,
    last_exclusive: date,
    *,
    backoff: float = 1.0,
    sleep=time.sleep,
) -> list[RunOutcome]:
    """Run each calendar month in order. Stops at the first failed month: later months depend on it."""
    outcomes: list[RunOutcome] = []
    for start, end in months(first, last_exclusive):
        outcome = run_interval(
            db_path,
            feed,
            lookup,
            start,
            end,
            trigger="backfill",
            backoff=backoff,
            sleep=sleep,
        )
        outcomes.append(outcome)
        if outcome.status != "succeeded":
            break
    return outcomes


def tick(
    db_path: Path,
    feed: Path,
    lookup: Path,
    expr: str,
    now: datetime,
    *,
    name: str = "scheduled",
    faults: Mapping[str, Sequence[Exception]] | None = None,
    backoff: float = 1.0,
    sleep=time.sleep,
) -> list[RunOutcome]:
    """Run every scheduled fire that is due and not yet done, oldest first.

    The first tick ever runs the most recent fire. The stored position advances only after a
    succeeded run, so a failed fire is retried by the next tick instead of being skipped.
    """
    spec = cron.parse(expr)
    cron.check_daily(spec)
    con = connect(db_path)
    try:
        row = con.execute(
            "SELECT last_fire FROM ops.schedule_state WHERE schedule = ?", [name]
        ).fetchone()
    finally:
        con.close()
    if row:
        after = row[0]
    else:  # first tick: the latest fire at or before now is due
        latest = cron.previous_before(spec, now + timedelta(seconds=1))
        after = latest - timedelta(seconds=1)
    outcomes: list[RunOutcome] = []
    for fire in cron.due_fires(spec, after, now):
        start, end = cron.interval_for_fire(spec, fire)
        outcome = run_interval(
            db_path,
            feed,
            lookup,
            start,
            end,
            trigger=f"schedule:{name}",
            faults=faults,
            backoff=backoff,
            sleep=sleep,
        )
        outcomes.append(outcome)
        if outcome.status != "succeeded":
            break
        con = connect(db_path)
        try:
            con.execute("DELETE FROM ops.schedule_state WHERE schedule = ?", [name])
            con.execute("INSERT INTO ops.schedule_state VALUES (?, ?)", [name, fire])
        finally:
            con.close()
    return outcomes
