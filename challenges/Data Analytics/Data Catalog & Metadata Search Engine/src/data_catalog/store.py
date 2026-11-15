"""The catalog: one SQLite file with datasets, columns, relationships, schema history and two FTS5
indexes (word-level BM25 with stemming, and trigram for substrings inside identifiers)."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from data_catalog.crawl import ColumnInfo, Dataset
from data_catalog.relations import Relation

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(
    key TEXT PRIMARY KEY, kind TEXT, source TEXT, name TEXT, rows INTEGER,
    primary_key TEXT, foreign_keys TEXT, fingerprint TEXT, crawled_at TEXT, columns TEXT);
CREATE TABLE IF NOT EXISTS relationships(
    child TEXT, child_column TEXT, parent TEXT, parent_column TEXT, origin TEXT, inclusion REAL,
    PRIMARY KEY (child, child_column, parent, parent_column));
CREATE TABLE IF NOT EXISTS schema_changes(at TEXT, dataset TEXT, change TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
    dataset UNINDEXED, column_name UNINDEXED, name, context, tags,
    tokenize = 'porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS search_tri USING fts5(
    dataset UNINDEXED, column_name UNINDEXED, text, tokenize = 'trigram');
"""
# bm25 weights for (name, context, tags): a hit in a name outranks one in the surrounding text.
WEIGHTS = "10.0, 2.0, 5.0"


def words(identifier: str) -> str:
    """'l_extendedPrice' -> 'l extended price lextendedprice': splits for word search, keeps the whole."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", identifier)
    parts = re.findall(r"[A-Za-z0-9]+", spaced)
    whole = "".join(parts).lower()
    return " ".join([p.lower() for p in parts] + ([whole] if len(parts) > 1 else []))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def diff_schema(old: list[dict], new: list[ColumnInfo]) -> list[str]:
    before = {c["name"]: c["type"] for c in old}
    after = {c.name: c.type for c in new}
    out = [f"added column {n} {t}" for n, t in after.items() if n not in before]
    out += [f"dropped column {n}" for n in before if n not in after]
    out += [
        f"column {n} type {before[n]} -> {t}"
        for n, t in after.items()
        if n in before and before[n] != t
    ]
    return out


class Catalog:
    def __init__(self, path: Path | str) -> None:
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- writes ---------------------------------------------------------------------------

    def fingerprint(self, key: str) -> str | None:
        row = self.con.execute(
            "SELECT fingerprint FROM datasets WHERE key = ?", [key]
        ).fetchone()
        return row[0] if row else None

    def keys_under(self, source: str) -> list[str]:
        return [
            r[0]
            for r in self.con.execute(
                "SELECT key FROM datasets WHERE source = ?", [source]
            )
        ]

    def upsert(self, ds: Dataset) -> list[str]:
        """Store a dataset and reindex it; returns the schema changes against the stored version."""
        old = self.con.execute(
            "SELECT columns FROM datasets WHERE key = ?", [ds.key]
        ).fetchone()
        changes = (
            ["new dataset"]
            if old is None
            else diff_schema(json.loads(old[0]), ds.columns)
        )
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    ds.key,
                    ds.kind,
                    ds.source,
                    ds.name,
                    ds.rows,
                    json.dumps(ds.primary_key),
                    json.dumps(ds.foreign_keys),
                    ds.fingerprint,
                    _now(),
                    json.dumps([asdict(c) for c in ds.columns]),
                ],
            )
            self._unindex(ds.key)
            context = f"{ds.kind} {Path(ds.source).name} {words(Path(ds.source).stem)}"
            all_tags = sorted({t for c in ds.columns for t in c.tags})
            self.con.execute(
                "INSERT INTO search VALUES (?,?,?,?,?)",
                [
                    ds.key,
                    None,
                    words(ds.name),
                    context + " " + " ".join(words(c.name) for c in ds.columns),
                    " ".join(all_tags),
                ],
            )
            self.con.execute(
                "INSERT INTO search_tri VALUES (?,?,?)", [ds.key, None, ds.name]
            )
            for c in ds.columns:
                self.con.execute(
                    "INSERT INTO search VALUES (?,?,?,?,?)",
                    [
                        ds.key,
                        c.name,
                        words(c.name),
                        f"{words(ds.name)} {context} {c.type} {' '.join(c.samples[:5])}",
                        " ".join(c.tags),
                    ],
                )
                self.con.execute(
                    "INSERT INTO search_tri VALUES (?,?,?)", [ds.key, c.name, c.name]
                )
            for ch in changes:
                self.con.execute(
                    "INSERT INTO schema_changes VALUES (?,?,?)", [_now(), ds.key, ch]
                )
        return changes

    def _unindex(self, key: str) -> None:
        self.con.execute("DELETE FROM search WHERE dataset = ?", [key])
        self.con.execute("DELETE FROM search_tri WHERE dataset = ?", [key])

    def remove(self, key: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM datasets WHERE key = ?", [key])
            self.con.execute(
                "DELETE FROM relationships WHERE child = ? OR parent = ?", [key, key]
            )
            self._unindex(key)
            self.con.execute(
                "INSERT INTO schema_changes VALUES (?,?,?)", [_now(), key, "removed"]
            )

    def set_relations(self, rels: list[Relation]) -> None:
        with self.con:
            self.con.execute("DELETE FROM relationships")
            self.con.executemany(
                "INSERT OR REPLACE INTO relationships VALUES (?,?,?,?,?,?)",
                [
                    (
                        r.child,
                        r.child_column,
                        r.parent,
                        r.parent_column,
                        r.origin,
                        r.inclusion,
                    )
                    for r in rels
                ],
            )

    # ---- reads ----------------------------------------------------------------------------

    def datasets(self) -> list[Dataset]:
        out = []
        for r in self.con.execute("SELECT * FROM datasets ORDER BY key"):
            out.append(
                Dataset(
                    r["kind"],
                    r["source"],
                    r["name"],
                    r["rows"],
                    [ColumnInfo(**c) for c in json.loads(r["columns"])],
                    json.loads(r["primary_key"]),
                    [tuple(f) for f in json.loads(r["foreign_keys"])],
                    r["fingerprint"],
                )
            )
        return out

    def get(self, ref: str) -> dict | None:
        """A dataset by full key, or by bare name when that name is unique."""
        rows = self.con.execute(
            "SELECT * FROM datasets WHERE key = ? OR name = ?", [ref, ref]
        ).fetchall()
        exact = [r for r in rows if r["key"] == ref]
        if not exact and len(rows) != 1:
            return None
        r = (exact or rows)[0]
        d = dict(r)
        d["columns"] = json.loads(d["columns"])
        d["primary_key"] = json.loads(d["primary_key"])
        d["foreign_keys"] = json.loads(d["foreign_keys"])
        d["relationships"] = [
            dict(x)
            for x in self.con.execute(
                "SELECT * FROM relationships WHERE child = ? OR parent = ? ORDER BY origin, child",
                [r["key"], r["key"]],
            )
        ]
        return d

    def relations(self) -> list[dict]:
        return [
            dict(r)
            for r in self.con.execute("SELECT * FROM relationships ORDER BY child")
        ]

    def changes(self, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM schema_changes ORDER BY rowid DESC LIMIT ?"
        return [dict(r) for r in self.con.execute(q, [limit])]

    def search(
        self, query: str, limit: int = 10, tag: str | None = None, mode: str = "hybrid"
    ) -> list[dict]:
        """mode: 'fts' (BM25 on words), 'hybrid' (BM25 fused with trigram substring ranking),
        'like' (substring baseline in name order)."""
        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
        if not terms:
            return []
        if mode == "like":
            return self._like(terms, limit, tag)
        # Each term quoted (no FTS syntax from users), prefix-matched, OR-ed: BM25 ranks rows that
        # match more and rarer terms first.
        match = " OR ".join(f'"{t}"*' for t in terms)
        sql = (
            f"SELECT dataset, column_name, bm25(search, {WEIGHTS}) AS score, tags FROM search "
            "WHERE search MATCH ? ORDER BY score LIMIT ?"
        )
        hits = [dict(r) for r in self.con.execute(sql, [match, limit * 5])]
        if mode == "hybrid":
            hits = self._fuse(hits, self._trigram(terms, limit * 5))
        if tag:
            hits = [h for h in hits if tag in (h["tags"] or "").split()]
        return [self._hit(h) for h in hits[:limit]]

    def _trigram(self, terms: list[str], n: int) -> list[dict]:
        """Substring matches inside identifiers ('price' in 'l_extendedprice'), BM25-ranked."""
        long_terms = [t for t in terms if len(t) >= 3]  # trigram needs three characters
        if not long_terms:
            return []
        tri = " OR ".join(f'"{t}"' for t in long_terms)
        rows = self.con.execute(
            "SELECT dataset, column_name, rank AS score FROM search_tri WHERE search_tri MATCH ? "
            "ORDER BY rank LIMIT ?",
            [tri, n],
        )
        return [
            {**dict(r), "tags": self._tags(r["dataset"], r["column_name"])}
            for r in rows
        ]

    @staticmethod
    def _fuse(*rankings: list[dict], k: int = 60) -> list[dict]:
        """Reciprocal rank fusion: score = sum of 1 / (k + rank) over the rankings a hit is in."""
        fused: dict[tuple, dict] = {}
        for ranking in rankings:
            for rank, h in enumerate(ranking, 1):
                key = (h["dataset"], h["column_name"])
                entry = fused.setdefault(key, {**h, "score": 0.0})
                entry["score"] -= 1 / (k + rank)  # negative, like bm25: lower is better
        return sorted(fused.values(), key=lambda h: h["score"])

    def _tags(self, dataset: str, column: str | None) -> str:
        r = self.con.execute(
            "SELECT tags FROM search WHERE dataset = ? AND column_name IS ?",
            [dataset, column],
        ).fetchone()
        return r[0] if r else ""

    def _like(self, terms: list[str], limit: int, tag: str | None) -> list[dict]:
        hits = []
        for r in self.con.execute(
            "SELECT dataset, column_name, tags FROM search ORDER BY dataset, column_name"
        ):
            name = (r["column_name"] or r["dataset"].rsplit("#", 1)[-1]).lower()
            if any(t in name for t in terms) and (not tag or tag in r["tags"].split()):
                hits.append({**dict(r), "score": 0.0})
        return [self._hit(h) for h in hits[:limit]]

    def _hit(self, h: dict) -> dict:
        name = h["dataset"].rsplit("#", 1)[-1]
        return {
            "dataset": h["dataset"],
            "name": name,
            "column": h["column_name"],
            "kind": "column" if h["column_name"] else "dataset",
            "score": round(h["score"], 3),
            "tags": (h["tags"] or "").split(),
        }
