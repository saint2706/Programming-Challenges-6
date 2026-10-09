"""SQLite run history and alert state. One short-lived connection per operation.

Many wrapper processes can finish at the same moment (every job scheduled for ``0 * * * *``), so
the database runs in WAL mode with a busy timeout, and each call opens, commits and closes its own
connection instead of sharing one across threads or processes.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    command TEXT NOT NULL,
    host TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_s REAL,
    exit_code INTEGER,
    status TEXT NOT NULL,
    output_tail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS runs_by_job ON runs (job_id, started_at DESC);
CREATE INDEX IF NOT EXISTS runs_by_time ON runs (started_at);
CREATE TABLE IF NOT EXISTS job_state (
    job_id TEXT PRIMARY KEY,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    alerted_at TEXT,
    last_status TEXT,
    monitor_cursor TEXT
);
CREATE TABLE IF NOT EXISTS alert_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    run_id INTEGER,
    kind TEXT NOT NULL,
    channel TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
"""
RUNNING, OK, FAILED, TIMEOUT, ERROR = "running", "ok", "failed", "timeout", "error"


def to_iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def from_iso(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


@dataclass(frozen=True)
class Run:
    id: int
    job_id: str
    command: str
    host: str
    started_at: datetime
    ended_at: datetime | None
    duration_s: float | None
    exit_code: int | None
    status: str
    output_tail: str

    @property
    def succeeded(self) -> bool:
        return self.status == OK


@dataclass(frozen=True)
class JobState:
    job_id: str
    consecutive_failures: int = 0
    alerted_at: datetime | None = None
    last_status: str | None = None
    monitor_cursor: datetime | None = None


@dataclass(frozen=True)
class AlertRecord:
    id: int
    job_id: str
    run_id: int | None
    kind: str
    channel: str
    sent_at: datetime
    ok: bool
    detail: str


def _run(row: sqlite3.Row) -> Run:
    return Run(
        row["id"], row["job_id"], row["command"], row["host"], from_iso(row["started_at"]), from_iso(row["ended_at"]),
        row["duration_s"], row["exit_code"], row["status"], row["output_tail"],
    )  # fmt: skip


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=10000")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def start_run(
        self, job_id: str, command: str, host: str, started_at: datetime
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO runs (job_id, command, host, started_at, status) VALUES (?, ?, ?, ?, ?)",
                (job_id, command, host, to_iso(started_at), RUNNING),
            )
            return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        ended_at: datetime,
        exit_code: int | None,
        status: str,
        output_tail: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE runs SET ended_at = ?, duration_s = (julianday(?) - julianday(started_at)) * 86400.0, "
                "exit_code = ?, status = ?, output_tail = ? WHERE id = ?",
                (
                    to_iso(ended_at),
                    to_iso(ended_at),
                    exit_code,
                    status,
                    output_tail,
                    run_id,
                ),
            )

    def get_run(self, run_id: int) -> Run | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row) if row else None

    def runs(
        self, job_id: str | None = None, limit: int = 100, since: datetime | None = None
    ) -> list[Run]:
        clauses, params = [], []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(job_id)
        if since is not None:
            clauses.append("started_at >= ?")
            params.append(to_iso(since))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM runs {where} ORDER BY started_at DESC, id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_run(row) for row in rows]

    def last_runs(self) -> dict[str, Run]:
        """The most recent run of every job."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT r.* FROM runs r JOIN (SELECT job_id, MAX(id) AS id FROM runs GROUP BY job_id) m ON r.id = m.id"
            ).fetchall()
        return {row["job_id"]: _run(row) for row in rows}

    def runs_between(self, job_id: str, start: datetime, end: datetime) -> list[Run]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM runs WHERE job_id = ? AND started_at >= ? AND started_at < ? ORDER BY started_at",
                (job_id, to_iso(start), to_iso(end)),
            ).fetchall()
        return [_run(row) for row in rows]

    def get_state(self, job_id: str) -> JobState:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM job_state WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return JobState(job_id)
        return JobState(
            job_id,
            row["consecutive_failures"],
            from_iso(row["alerted_at"]),
            row["last_status"],
            from_iso(row["monitor_cursor"]),
        )

    def save_state(self, state: JobState) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO job_state (job_id, consecutive_failures, alerted_at, last_status, monitor_cursor) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET consecutive_failures = excluded.consecutive_failures, "
                "alerted_at = excluded.alerted_at, last_status = excluded.last_status, monitor_cursor = excluded.monitor_cursor",
                (
                    state.job_id, state.consecutive_failures, to_iso(state.alerted_at) if state.alerted_at else None,
                    state.last_status, to_iso(state.monitor_cursor) if state.monitor_cursor else None,
                ),
            )  # fmt: skip

    def log_alert(
        self,
        job_id: str,
        run_id: int | None,
        kind: str,
        channel: str,
        sent_at: datetime,
        ok: bool,
        detail: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO alert_log (job_id, run_id, kind, channel, sent_at, ok, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, run_id, kind, channel, to_iso(sent_at), int(ok), detail),
            )

    def alerts(self, limit: int = 50) -> list[AlertRecord]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM alert_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            AlertRecord(
                r["id"],
                r["job_id"],
                r["run_id"],
                r["kind"],
                r["channel"],
                from_iso(r["sent_at"]),
                bool(r["ok"]),
                r["detail"],
            )
            for r in rows
        ]

    def prune(self, before: datetime) -> int:
        """Delete runs and alert log entries older than ``before``; returns the number of runs removed."""
        with self._connect() as db:
            removed = db.execute(
                "DELETE FROM runs WHERE started_at < ?", (to_iso(before),)
            ).rowcount
            db.execute("DELETE FROM alert_log WHERE sent_at < ?", (to_iso(before),))
        return removed
