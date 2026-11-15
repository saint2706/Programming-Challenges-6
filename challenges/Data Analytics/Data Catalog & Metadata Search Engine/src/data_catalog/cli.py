"""catalog: crawl sources, search and inspect the catalog, serve the search page, run the demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_catalog.indexer import crawl_into
from data_catalog.paths import project_root
from data_catalog.store import Catalog


def _catalog(a: argparse.Namespace) -> Catalog:
    return Catalog(a.catalog)


def cmd_crawl(a: argparse.Namespace) -> int:
    with _catalog(a) as cat:
        s = crawl_into(cat, [Path(p) for p in a.paths], force=a.force)
    print(
        f"{s.new} new, {s.changed} changed, {s.unchanged} unchanged, {s.removed} removed "
        f"datasets; {s.relations} relationships"
    )
    for c in s.changes:
        print(f"  schema change: {c}")
    for e in s.errors:
        print(f"  error: {e}")
    return 1 if s.errors else 0


def cmd_search(a: argparse.Namespace) -> int:
    with _catalog(a) as cat:
        hits = cat.search(a.query, limit=a.limit, tag=a.tag, mode=a.mode)
    for h in hits:
        what = f"{h['name']}.{h['column']}" if h["column"] else f"{h['name']} (dataset)"
        tags = f"  [{', '.join(h['tags'])}]" if h["tags"] else ""
        print(f"{h['score']:8.3f}  {what}{tags}\n          {h['dataset']}")
    if not hits:
        print("no matches")
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    with _catalog(a) as cat:
        d = cat.get(a.dataset)
    if d is None:
        raise SystemExit(
            f"no single dataset matches {a.dataset!r}; use the full key from search"
        )
    if a.json:
        print(json.dumps(d, indent=2, default=str))
        return 0
    print(f"{d['name']}  ({d['kind']}, {d['rows']} rows)\n  {d['key']}")
    if d["primary_key"]:
        print(f"  primary key: {', '.join(d['primary_key'])}")
    for c in d["columns"]:
        tags = f"  [{', '.join(c['tags'])}]" if c["tags"] else ""
        print(
            f"  {c['name']:<20} {c['type']:<12} nulls {c['nulls']:<6} distinct {c['distinct']:<7}"
            f" {c['min']} .. {c['max']}{tags}"
        )
    for r in d["relationships"]:
        print(
            f"  {r['origin']:<8} {r['child'].rsplit('#', 1)[-1]}.{r['child_column']} -> "
            f"{r['parent'].rsplit('#', 1)[-1]}.{r['parent_column']}  ({r['inclusion']:.0%} of values)"
        )
    return 0


def cmd_relations(a: argparse.Namespace) -> int:
    with _catalog(a) as cat:
        for r in cat.relations():
            print(
                f"{r['origin']:<8} {r['child'].rsplit('#', 1)[-1]}.{r['child_column']} -> "
                f"{r['parent'].rsplit('#', 1)[-1]}.{r['parent_column']}  ({r['inclusion']:.0%})"
            )
    return 0


def cmd_changes(a: argparse.Namespace) -> int:
    with _catalog(a) as cat:
        for c in cat.changes(a.limit):
            print(f"{c['at']}  {c['dataset'].rsplit('#', 1)[-1]}: {c['change']}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import os

    import uvicorn

    os.environ["DATA_CATALOG_DB"] = str(Path(a.catalog).resolve())
    os.environ["DATA_CATALOG_HOSTS"] = ",".join(
        filter(None, [os.environ.get("DATA_CATALOG_HOSTS", ""), a.host])
    )
    uvicorn.run("data_catalog.server:app", host=a.host, port=a.port)
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    from data_catalog import demo

    root = project_root()
    demo_dir = root / "data" / "demo"
    if a.rebuild or not (demo_dir / "warehouse.duckdb").exists():
        demo.build(demo_dir)
    res = demo.evaluate(
        demo_dir, root / "data" / "demo_catalog.sqlite", root / "results"
    )
    print(json.dumps({k: res[k] for k in ("foreign_keys", "pii", "search")}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catalog", description=__doc__)
    p.add_argument(
        "--catalog",
        default="catalog.sqlite",
        help="catalog file (default ./catalog.sqlite)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="discover and profile sources (files or folders)")
    c.add_argument("paths", nargs="+")
    c.add_argument(
        "--force", action="store_true", help="re-profile unchanged sources too"
    )
    c.set_defaults(fn=cmd_crawl)

    s = sub.add_parser("search", help="search dataset and column metadata")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--tag", help="only hits carrying this tag, e.g. pii or email")
    s.add_argument("--mode", choices=["hybrid", "fts", "like"], default="hybrid")
    s.set_defaults(fn=cmd_search)

    sh = sub.add_parser("show", help="one dataset: columns, profile, relationships")
    sh.add_argument(
        "dataset", help="dataset key, or a name that is unique in the catalog"
    )
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(fn=cmd_show)

    sub.add_parser("relations", help="declared and inferred foreign keys").set_defaults(
        fn=cmd_relations
    )
    ch = sub.add_parser("changes", help="schema change history")
    ch.add_argument("--limit", type=int, default=50)
    ch.set_defaults(fn=cmd_changes)

    sv = sub.add_parser("serve", help="search page at http://127.0.0.1:8000")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    d = sub.add_parser(
        "demo", help="build the TPC-H demo estate and write results/evaluation.*"
    )
    d.add_argument("--rebuild", action="store_true")
    d.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
