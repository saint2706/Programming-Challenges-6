"""SQLite project store: items, labels (full history, the last one wins) and retraining rounds."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, text TEXT NOT NULL, gold TEXT);
CREATE TABLE IF NOT EXISTS labels (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id),
    label TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    n_labeled INTEGER NOT NULL,
    n_classes INTEGER NOT NULL,
    change REAL,
    accuracy REAL,
    pred BLOB NOT NULL,
    sig TEXT NOT NULL,
    ts TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        self._db = sqlite3.connect(str(path))
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def get(self, key: str, default=None):
        row = self._db.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return default if row is None else row[0]

    def set(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
        self._db.commit()

    def set_classes(self, classes) -> None:
        self.set("classes", json.dumps(list(classes)))

    def classes(self) -> list[str]:
        return json.loads(self.get("classes", "[]"))

    def add_items(self, texts, gold=None) -> None:
        if self.n_items():
            raise ValueError("this project already has items")
        known = set(self.classes())
        for g in gold or []:
            if g is not None and g not in known:
                raise ValueError(f"gold label {g!r} is not in the class list")
        rows = [(i, t, None if gold is None else gold[i]) for i, t in enumerate(texts)]
        self._db.executemany("INSERT INTO items VALUES (?, ?, ?)", rows)
        self._db.commit()

    def n_items(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def texts(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT text FROM items ORDER BY id")]

    def gold(self) -> list[str | None]:
        return [r[0] for r in self._db.execute("SELECT gold FROM items ORDER BY id")]

    def label(self, item_id: int, label: str) -> None:
        if (
            self._db.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
            is None
        ):
            raise ValueError(f"no item {item_id}")
        classes = self.classes()
        if label not in classes:
            raise ValueError(f"{label!r} is not one of the {len(classes)} classes")
        self._db.execute(
            "INSERT INTO labels (item_id, label, ts) VALUES (?, ?, ?)",
            (int(item_id), label, _now()),
        )
        self._db.commit()

    def current_labels(self) -> dict[int, str]:
        rows = self._db.execute(
            "SELECT item_id, label FROM labels WHERE seq IN (SELECT MAX(seq) FROM labels GROUP BY item_id)"
        )
        return dict(rows.fetchall())

    def history(self, item_id: int) -> list[tuple[str, str]]:
        rows = self._db.execute(
            "SELECT label, ts FROM labels WHERE item_id = ? ORDER BY seq", (item_id,)
        )
        return rows.fetchall()

    def add_round(self, n_labeled, n_classes, change, accuracy, pred, sig) -> None:
        self._db.execute(
            "INSERT INTO rounds (n_labeled, n_classes, change, accuracy, pred, sig, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                int(n_labeled),
                int(n_classes),
                change,
                accuracy,
                np.asarray(pred, dtype=np.int16).tobytes(),
                sig,
                _now(),
            ),
        )
        self._db.commit()

    def rounds(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT n_labeled, n_classes, change, accuracy, pred, sig, ts FROM rounds ORDER BY id"
        )
        return [
            {
                "n_labeled": n,
                "n_classes": k,
                "change": change,
                "accuracy": acc,
                "pred": np.frombuffer(blob, dtype=np.int16),
                "sig": sig,
                "ts": ts,
            }
            for n, k, change, acc, blob, sig, ts in rows
        ]
