"""Infer foreign keys that no source declares, across sources.

A pair (child.a -> parent.b) is proposed when all three hold:

1. parent.b is unique and has no NULLs (distinct == rows), so it can be a key;
2. the names agree: equal after dropping a short table prefix (l_orderkey ~ o_orderkey), or the
   child name is the parent table's singular name plus the key (user_id ~ users.id);
3. the values agree: at least `min_inclusion` of child.a's distinct values occur in parent.b.

Names alone over-match (every table has an `id`), and values alone over-match (any small integer
column fits inside a 1..n surrogate key), so both are required.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from data_catalog.crawl import ColumnInfo, Dataset

INT = re.compile(r"int|serial|long", re.IGNORECASE)
TEXT = re.compile(r"char|text|string|varchar", re.IGNORECASE)


@dataclass(frozen=True)
class Relation:
    child: str  # dataset key
    child_column: str
    parent: str
    parent_column: str
    origin: str  # declared | inferred
    inclusion: float


def _family(t: str) -> str:
    if INT.search(t):
        return "int"
    if TEXT.search(t) or t.upper() == "ANY":
        return "text"
    return t.lower()


def core(column: str) -> str:
    """Drop a 1-2 letter table prefix and separators: 'l_orderkey' -> 'orderkey', 'CustomerId' -> 'customerid'."""
    c = re.sub(r"^[a-z]{1,2}_(?=.)", "", column.lower())
    return re.sub(r"[^a-z0-9]", "", c)


def singular(table: str) -> str:
    t = re.sub(r"[^a-z0-9]", "", table.lower().rsplit(".", 1)[-1])
    for suffix, repl in (("ies", "y"), ("s", "")):
        if t.endswith(suffix) and len(t) > len(suffix) + 2 and not t.endswith("ss"):
            return t[: -len(suffix)] + repl
    return t


def names_agree(child_col: str, parent_table: str, parent_col: str) -> bool:
    c, p = core(child_col), core(parent_col)
    if c == p and p not in {"id", "key", "name", "code"}:
        return True
    t = singular(parent_table)
    keyish = p in {"id", "key"} or p.endswith(("id", "key"))
    return keyish and c in {t + p, t + "id", t + "key"}


def is_key(ds: Dataset, col: ColumnInfo) -> bool:
    return ds.rows > 0 and col.nulls == 0 and col.distinct == ds.rows


def infer(
    datasets: list[Dataset],
    values: Callable[[Dataset, str], set[str] | None],
    min_inclusion: float = 0.99,
) -> list[Relation]:
    out = []
    declared = {(ds.key, a) for ds in datasets for a, _, _ in ds.foreign_keys}
    for ds in datasets:
        for a, ref_table, b in ds.foreign_keys:
            parent = next(
                (p for p in datasets if p.source == ds.source and p.name == ref_table),
                None,
            )
            if parent:
                out.append(Relation(ds.key, a, parent.key, b, "declared", 1.0))
    cache: dict[tuple[str, str], set[str] | None] = {}

    def vals(ds: Dataset, col: str) -> set[str] | None:
        if (ds.key, col) not in cache:
            cache[ds.key, col] = values(ds, col)
        return cache[ds.key, col]

    for parent in datasets:
        for pcol in parent.columns:
            if not is_key(parent, pcol):
                continue
            for child in datasets:
                for ccol in child.columns:
                    if child is parent and ccol is pcol:
                        continue
                    if (child.key, ccol.name) in declared or ccol.distinct < 2:
                        continue
                    if _family(ccol.type) != _family(pcol.type):
                        continue
                    if not names_agree(ccol.name, parent.name, pcol.name):
                        continue
                    # two keys with the same name (1:1 or a copy): only keep child -> larger parent
                    if (
                        child is not parent
                        and is_key(child, ccol)
                        and core(ccol.name) == core(pcol.name)
                        and child.rows > parent.rows
                    ):
                        continue
                    cv, pv = vals(child, ccol.name), vals(parent, pcol.name)
                    if not cv or pv is None:
                        continue
                    inc = len(cv & pv) / len(cv)
                    if inc >= min_inclusion:
                        out.append(
                            Relation(
                                child.key,
                                ccol.name,
                                parent.key,
                                pcol.name,
                                "inferred",
                                round(inc, 4),
                            )
                        )
    return out
