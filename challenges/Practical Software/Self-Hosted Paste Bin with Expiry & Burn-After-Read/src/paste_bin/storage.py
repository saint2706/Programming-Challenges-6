"""SQLite-backed storage for the paste bin: TTL expiry and burn-after-read.

The one thing that has to be airtight is burn-after-read: two people (or one
person double-clicking / a link-preview bot) hitting the same one-time link
at the same instant must not both see the content. ``get_and_consume`` makes
the read-then-maybe-delete a single manually-controlled SQLite transaction
opened with ``BEGIN IMMEDIATE``, which grabs a write lock *before* the
SELECT. A second connection calling the same method blocks on its own
``BEGIN IMMEDIATE`` until the first transaction commits, so the two calls are
fully serialized even though each opens its own connection -- whoever's
transaction starts first is the only one who ever sees the content.
"""

from __future__ import annotations

import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS pastes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'text',
    created_at REAL NOT NULL,
    expires_at REAL,
    burn_after_read INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class Paste:
    id: str
    content: str
    language: str
    created_at: float
    expires_at: float | None
    burn_after_read: bool

    def is_expired(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.expires_at is not None and self.expires_at <= now


def _row_to_paste(row: sqlite3.Row) -> Paste:
    return Paste(
        id=row["id"],
        content=row["content"],
        language=row["language"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        burn_after_read=bool(row["burn_after_read"]),
    )


class PasteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None (autocommit) so we control transactions
        # explicitly with BEGIN/COMMIT where atomicity actually matters.
        conn = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def create(
        self,
        content: str,
        language: str = "text",
        ttl_seconds: float | None = None,
        burn_after_read: bool = False,
    ) -> Paste:
        paste_id = secrets.token_urlsafe(8)
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO pastes (id, content, language, created_at, expires_at, burn_after_read) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (paste_id, content, language, now, expires_at, int(burn_after_read)),
            )
        finally:
            conn.close()
        return Paste(paste_id, content, language, now, expires_at, burn_after_read)

    def peek(self, paste_id: str) -> Paste | None:
        """Look up without consuming a burn-after-read paste (used for the TTL display)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM pastes WHERE id = ?", (paste_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        paste = _row_to_paste(row)
        return None if paste.is_expired() else paste

    def get_and_consume(self, paste_id: str) -> Paste | None:
        """Fetch a paste, atomically deleting it if it's expired or burn-after-read.

        Returns None if the paste doesn't exist or has expired. Otherwise
        returns the paste content -- even if it was just deleted as a
        consequence of being burn-after-read, so the caller can still render
        it exactly once.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM pastes WHERE id = ?", (paste_id,)
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            paste = _row_to_paste(row)
            if paste.is_expired():
                conn.execute("DELETE FROM pastes WHERE id = ?", (paste_id,))
                conn.execute("COMMIT")
                return None
            if paste.burn_after_read:
                conn.execute("DELETE FROM pastes WHERE id = ?", (paste_id,))
            conn.execute("COMMIT")
            return paste
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def delete_expired(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM pastes WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (time.time(),),
            )
            return cur.rowcount
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM pastes").fetchone()[0]
        finally:
            conn.close()
