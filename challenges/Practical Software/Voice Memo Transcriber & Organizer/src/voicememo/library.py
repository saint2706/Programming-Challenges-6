"""SQLite library: memos, timestamped segments, tags, and an FTS5 index over segment text."""

import hashlib
import json
import re
import shutil
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from voicememo.asr import Segment

SCHEMA = """
CREATE TABLE IF NOT EXISTS memos (
    id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    title TEXT NOT NULL,
    audio_file TEXT NOT NULL DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,
    language TEXT NOT NULL DEFAULT '',
    backend TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    memo_id INTEGER NOT NULL REFERENCES memos(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    words TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS segments_memo ON segments(memo_id, idx);
CREATE TABLE IF NOT EXISTS tags (
    memo_id INTEGER NOT NULL REFERENCES memos(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('auto', 'user')),
    PRIMARY KEY (memo_id, tag)
);
CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts USING fts5(
    text, memo_id UNINDEXED, seg_id UNINDEXED, tokenize = 'unicode61 remove_diacritics 2'
);
"""

MAX_TAG_LENGTH = 40
MAX_QUERY_TERMS = 32
_TERM = re.compile(r"([\w']+)(\*?)")


class LibraryError(Exception):
    """Bad input to the library (unknown memo, invalid tag)."""


@dataclass(frozen=True)
class Memo:
    id: int
    sha256: str
    original_name: str
    title: str
    audio_file: str
    duration: float
    language: str
    backend: str
    status: str
    error: str
    created_at: float


@dataclass(frozen=True)
class SearchHit:
    memo_id: int
    title: str
    start: float
    snippet: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_tag(raw: str) -> str:
    """Lower-case, collapse whitespace, drop control characters, cap the length."""
    # Whitespace becomes a space before control characters are dropped, so "a\tb" stays "a b".
    spaced = re.sub(r"\s+", " ", raw)
    cleaned = "".join(ch for ch in spaced if unicodedata.category(ch)[0] != "C")
    tag = re.sub(r"\s+", " ", cleaned).strip().lower()
    if not tag:
        raise LibraryError("tag must not be empty")
    if len(tag) > MAX_TAG_LENGTH:
        raise LibraryError(f"tag longer than {MAX_TAG_LENGTH} characters")
    return tag


def fts_query(user_text: str) -> str | None:
    """Turn free text into a safe FTS5 query. Every term is quoted, so operators such as
    AND/OR/NOT, NEAR(), column filters, quotes and punctuation are only ever literal text.
    A trailing '*' on a term makes it a prefix match."""
    parts: list[str] = []
    for word, star in _TERM.findall(user_text)[:MAX_QUERY_TERMS]:
        quoted = '"' + word.replace('"', '""') + '"'
        parts.append(quoted + ("*" if star else ""))
    return " ".join(parts) or None


def _safe_extension(name: str) -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    return suffix if re.fullmatch(r"[a-z0-9]{1,8}", suffix) else "bin"


class Library:
    """One SQLite file plus an audio folder. Audio is copied in under its content hash."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.audio_dir = root / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        # The web app runs its handlers on one event-loop thread, which may differ from the
        # thread that opened the library (the test client does this), so skip the thread check.
        self.conn = sqlite3.connect(root / "library.db", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- memos -------------------------------------------------------------------------

    def _memo(self, row: sqlite3.Row) -> Memo:
        return Memo(
            row["id"],
            row["sha256"],
            row["original_name"],
            row["title"],
            row["audio_file"],
            row["duration"],
            row["language"],
            row["backend"],
            row["status"],
            row["error"],
            row["created_at"],
        )

    def get(self, memo_id: int) -> Memo:
        row = self.conn.execute(
            "SELECT * FROM memos WHERE id = ?", (memo_id,)
        ).fetchone()
        if row is None:
            raise LibraryError(f"no memo with id {memo_id}")
        return self._memo(row)

    def find_by_sha(self, sha: str) -> Memo | None:
        row = self.conn.execute(
            "SELECT * FROM memos WHERE sha256 = ?", (sha,)
        ).fetchone()
        return self._memo(row) if row else None

    def all_ids(self) -> list[int]:
        return [
            r["id"]
            for r in self.conn.execute(
                "SELECT id FROM memos WHERE status = 'done' ORDER BY sha256"
            )
        ]

    def list_memos(self, tag: str | None = None, limit: int = 100) -> list[Memo]:
        if tag:
            try:
                key = normalize_tag(tag)
            except (
                LibraryError
            ):  # an unusable filter can match nothing; it is not an error
                return []
            rows = self.conn.execute(
                "SELECT m.* FROM memos m JOIN tags t ON t.memo_id = m.id WHERE t.tag = ? "
                "ORDER BY m.created_at DESC, m.id DESC LIMIT ?",
                (key, limit),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM memos ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [self._memo(r) for r in rows]

    def create_pending(self, sha: str, original_name: str, ext_source: str) -> Memo:
        """Insert a memo row before transcription so a crash leaves a visible record."""
        title = Path(original_name).stem[:200] or "memo"
        ext = _safe_extension(ext_source)
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO memos (sha256, original_name, title, audio_file, status, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (sha, original_name, title, f"{sha}.{ext}", time.time()),
            )
        return self.get(cur.lastrowid)

    def store_audio(self, source: Path, memo: Memo) -> Path:
        """Copy the original into the library under its hash-derived name."""
        target = self.audio_path(memo)
        if not target.exists():
            shutil.copyfile(source, target)
        return target

    def audio_path(self, memo: Memo) -> Path:
        """Only ever built from the hash-derived file name stored in the row, never from user input."""
        if (
            not memo.audio_file
            or "/" in memo.audio_file
            or "\\" in memo.audio_file
            or ":" in memo.audio_file  # Windows drive-relative names such as "C:x.m4a"
            or ".." in memo.audio_file
        ):
            raise LibraryError("memo has no safe audio file")
        return self.audio_dir / memo.audio_file

    def complete(
        self,
        memo_id: int,
        segments: list[Segment],
        *,
        duration: float,
        language: str,
        backend: str,
    ) -> None:
        with self.conn:
            self._replace_segments(memo_id, segments)
            self.conn.execute(
                "UPDATE memos SET status = 'done', error = '', duration = ?, language = ?, backend = ? WHERE id = ?",
                (duration, language, backend, memo_id),
            )

    def _replace_segments(self, memo_id: int, segments: list[Segment]) -> None:
        self.conn.execute("DELETE FROM segments WHERE memo_id = ?", (memo_id,))
        self.conn.execute("DELETE FROM segment_fts WHERE memo_id = ?", (memo_id,))
        for idx, seg in enumerate(segments):
            words = json.dumps(
                [{"text": w.text, "start": w.start, "end": w.end} for w in seg.words]
            )
            cur = self.conn.execute(
                "INSERT INTO segments (memo_id, idx, start, end, text, words) VALUES (?, ?, ?, ?, ?, ?)",
                (memo_id, idx, seg.start, seg.end, seg.text, words),
            )
            self.conn.execute(
                "INSERT INTO segment_fts (text, memo_id, seg_id) VALUES (?, ?, ?)",
                (seg.text, memo_id, cur.lastrowid),
            )

    def fail(self, memo_id: int, message: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE memos SET status = 'failed', error = ? WHERE id = ?",
                (message[:500], memo_id),
            )

    def delete(self, memo_id: int) -> None:
        memo = self.get(memo_id)
        with self.conn:
            self.conn.execute("DELETE FROM segment_fts WHERE memo_id = ?", (memo_id,))
            self.conn.execute("DELETE FROM memos WHERE id = ?", (memo_id,))
        if memo.audio_file:
            target = self.audio_path(memo)
            target.unlink(missing_ok=True)

    # ----- reads --------------------------------------------------------------------------

    def segments(self, memo_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT start, end, text, words FROM segments WHERE memo_id = ? ORDER BY idx",
            (memo_id,),
        )
        return [
            {
                "start": r["start"],
                "end": r["end"],
                "text": r["text"],
                "words": json.loads(r["words"]),
            }
            for r in rows
        ]

    def transcript(self, memo_id: int) -> str:
        return " ".join(s["text"] for s in self.segments(memo_id))

    def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        match = fts_query(query)
        if match is None:
            return []
        rows = self.conn.execute(
            "SELECT f.memo_id AS memo_id, m.title AS title, s.start AS start, "
            "snippet(segment_fts, 0, '[', ']', '...', 12) AS snippet "
            "FROM segment_fts f JOIN segments s ON s.id = f.seg_id JOIN memos m ON m.id = f.memo_id "
            "WHERE segment_fts MATCH ? ORDER BY bm25(segment_fts), f.memo_id, s.idx LIMIT ?",
            (match, limit),
        )
        return [
            SearchHit(r["memo_id"], r["title"], r["start"], r["snippet"]) for r in rows
        ]

    # ----- tags ---------------------------------------------------------------------------

    def tags(self, memo_id: int) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT tag, kind FROM tags WHERE memo_id = ? ORDER BY tag", (memo_id,)
        )
        return [(r["tag"], r["kind"]) for r in rows]

    def add_tag(self, memo_id: int, raw: str) -> str:
        self.get(memo_id)
        tag = normalize_tag(raw)
        with self.conn:
            self.conn.execute(
                "INSERT INTO tags (memo_id, tag, kind) VALUES (?, ?, 'user') "
                "ON CONFLICT(memo_id, tag) DO UPDATE SET kind = 'user'",
                (memo_id, tag),
            )
        return tag

    def remove_tag(self, memo_id: int, raw: str) -> bool:
        tag = normalize_tag(raw)
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM tags WHERE memo_id = ? AND tag = ?", (memo_id, tag)
            )
        return cur.rowcount > 0

    def set_auto_tags(self, assignments: dict[int, list[str]]) -> None:
        """Replace every automatic tag. User-made tags are never touched."""
        with self.conn:
            self.conn.execute("DELETE FROM tags WHERE kind = 'auto'")
            for memo_id, tags in assignments.items():
                for tag in tags:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO tags (memo_id, tag, kind) VALUES (?, ?, 'auto')",
                        (memo_id, normalize_tag(tag)),
                    )

    def tag_counts(self) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT tag, COUNT(*) AS n FROM tags GROUP BY tag ORDER BY n DESC, tag"
        )
        return [(r["tag"], r["n"]) for r in rows]

    def texts_by_memo(self) -> dict[int, str]:
        """Full transcript text for every finished memo, keyed by memo id (sorted by hash)."""
        out: dict[int, str] = {}
        for memo_id in self.all_ids():
            out[memo_id] = self.transcript(memo_id)
        return out
