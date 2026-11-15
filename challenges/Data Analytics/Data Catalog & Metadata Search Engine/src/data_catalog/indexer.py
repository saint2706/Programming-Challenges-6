"""Crawl sources into the catalog incrementally, then re-infer relationships over everything."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from data_catalog.crawl import crawl_source, discover, distinct_values
from data_catalog.relations import infer
from data_catalog.store import Catalog

VALUE_CAP = 200_000  # columns with more distinct values are not used for FK inference


@dataclass
class Summary:
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    removed: int = 0
    relations: int = 0
    changes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _under(source: str, roots: list[Path]) -> bool:
    p = Path(source)
    return any(p == r or r in p.parents for r in roots)


def crawl_into(cat: Catalog, paths: list[Path], force: bool = False) -> Summary:
    """Profile every source under `paths`. A source whose size and mtime are unchanged is skipped
    unless `force`; datasets that vanished from a crawled path are removed."""
    s = Summary()
    roots = [p.resolve() for p in paths]
    seen: set[str] = set()
    for kind, path in discover(roots):
        src = str(path)
        st = path.stat()
        fp = f"{st.st_size}:{st.st_mtime_ns}"
        existing = cat.keys_under(src)
        if not force and existing and all(cat.fingerprint(k) == fp for k in existing):
            seen.update(existing)
            s.unchanged += len(existing)
            continue
        try:
            found = crawl_source(kind, path)
        except Exception as e:  # noqa: BLE001 - one unreadable file must not stop the crawl
            s.errors.append(f"{src}: {type(e).__name__}: {e}")
            seen.update(existing)
            continue
        for ds in found:
            changes = cat.upsert(ds)
            seen.add(ds.key)
            if changes == ["new dataset"]:
                s.new += 1
            else:
                s.changed += bool(changes)
                s.unchanged += not changes
            s.changes += [f"{ds.name}: {c}" for c in changes if c != "new dataset"]
    for ds in cat.datasets():
        if ds.key not in seen and _under(ds.source, roots):
            cat.remove(ds.key)
            s.removed += 1

    def values(ds, col: str) -> set[str] | None:
        try:
            return distinct_values(ds.kind, ds.source, ds.name, col, VALUE_CAP)
        except Exception:  # noqa: BLE001 - source moved or unreadable since it was cataloged
            return None

    rels = infer(cat.datasets(), values)
    cat.set_relations(rels)
    s.relations = len(rels)
    return s
