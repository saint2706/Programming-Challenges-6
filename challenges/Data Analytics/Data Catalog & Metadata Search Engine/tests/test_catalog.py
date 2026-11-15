import json
import os
import time

import pytest
from data_catalog.cli import main
from data_catalog.crawl import discover
from data_catalog.indexer import crawl_into
from data_catalog.pii import luhn, tag_column
from data_catalog.relations import core, names_agree, singular
from data_catalog.store import Catalog, words
from fastapi.testclient import TestClient


def names(cat):
    return sorted(d.name for d in cat.datasets())


def rels(cat):
    short = lambda k: k.rsplit("#", 1)[-1]
    return {
        (
            short(r["child"]),
            r["child_column"],
            short(r["parent"]),
            r["parent_column"],
            r["origin"],
        )
        for r in cat.relations()
    }


def test_discover_kinds_and_skips(estate):
    found = sorted((k, p.name) for k, p in discover([estate]))
    assert found == [
        ("csv", "visits.csv"),
        ("duckdb", "wh.duckdb"),
        ("json", "tags.jsonl"),
        ("parquet", "regions.parquet"),
        ("sqlite", "shop.sqlite"),
    ]
    with pytest.raises(FileNotFoundError):
        list(discover([estate / "missing"]))


def test_crawl_profiles_every_source(estate, tmp_path):
    with Catalog(tmp_path / "c.sqlite") as cat:
        s = crawl_into(cat, [estate])
        assert s.errors == [] and s.new == 7
        assert names(cat) == [
            "customers",
            "order_lines",
            "orders",
            "products",
            "regions",
            "tags",
            "visits",
        ]
        d = cat.get("orders")
        assert d["rows"] == 100 and d["primary_key"] == ["order_id"]
        total = next(c for c in d["columns"] if c["name"] == "total")
        assert (total["min"], total["max"], total["distinct"]) == ("1.5", "150.0", 100)
        assert cat.get("products")["primary_key"] == ["sku"]
        assert cat.get("visits")["kind"] == "csv"


def test_declared_and_inferred_relationships(estate, tmp_path):
    with Catalog(tmp_path / "c.sqlite") as cat:
        crawl_into(cat, [estate])
        r = rels(cat)
    assert ("orders", "customer_id", "customers", "id", "declared") in r
    assert ("order_lines", "sku", "products", "sku", "declared") in r
    # across sources, nothing declared: CSV -> SQLite, DuckDB -> SQLite, JSON -> DuckDB
    assert ("visits", "customer_id", "customers", "id", "inferred") in r
    assert ("order_lines", "order_id", "orders", "order_id", "inferred") in r
    assert ("tags", "sku", "products", "sku", "inferred") in r
    assert not any(x[0] == "visits" and x[1] == "visit_id" for x in r)


def test_pii_tags_from_names_and_values(estate, tmp_path):
    with Catalog(tmp_path / "c.sqlite") as cat:
        crawl_into(cat, [estate])
        cols = {c["name"]: c["tags"] for c in cat.get("customers")["columns"]}
        assert cols["email"] == ["email", "pii"]
        email = next(c for c in cat.get("customers")["columns"] if c["name"] == "email")
        assert (email["samples"], email["min"], email["max"]) == ([], None, None)
        assert email["distinct"] == 20  # counts stay; values do not
        assert cols["full_name"] == ["person_name", "pii"]
        assert cols["id"] == []
        visits = {c["name"]: c["tags"] for c in cat.get("visits")["columns"]}
        assert visits["ip_address"] == ["ip_address", "pii"]
        hits = cat.search("address", tag="pii")
        # the dataset document carries the union of its columns' tags, so it matches too
        assert [(h["name"], h["column"]) for h in hits] == [
            ("visits", "ip_address"),
            ("visits", None),
        ]


@pytest.mark.parametrize(
    ("name", "values", "want"),
    [
        ("contact", ["a@b.co", "c.d+x@e.org"], ["email", "pii"]),
        ("note", ["a@b.co", "free text", "more text"], []),
        ("cc", ["4111 1111 1111 1111", "5500 0000 0000 0004"], ["payment_card", "pii"]),
        ("ref", ["4111 1111 1111 1112"], []),
        ("shipped", ["1995-03-15", "1996-01-02"], []),
        ("price", ["17954.55", "1234.56"], []),
        ("contact_no", ["+1 415 555 0100"], ["phone", "pii"]),
        ("tax_id", ["123-45-6789"], ["national_id", "pii"]),
        ("dateOfBirth", [], ["birth_date", "pii"]),
        ("zip", [], ["address", "pii"]),
        ("email_address", [], ["email", "pii"]),
        ("product_name", ["Widget"], []),
    ],
)
def test_tag_column(name, values, want):
    assert tag_column(name, values) == want


def test_luhn():
    assert luhn("4111111111111111") and not luhn("4111111111111112")


def test_name_rules():
    assert core("l_orderkey") == "orderkey" and core("CustomerId") == "customerid"
    assert singular("categories") == "category" and singular("users") == "user"
    assert singular("main.address") == "address"
    assert names_agree("user_id", "users", "id")
    assert not names_agree("id", "users", "id")
    assert not names_agree("order_id", "orders", "o_comment")


def test_search_modes_and_word_splitting(estate, tmp_path):
    assert words("l_extendedPrice") == "l extended price lextendedprice"
    with Catalog(tmp_path / "c.sqlite") as cat:
        crawl_into(cat, [estate])
        top3 = [(h["name"], h["column"]) for h in cat.search("customer email", limit=3)]
        assert ("customers", "email") in top3
        # trigram finds 'region' inside 'regions' and 'region_name' even in FTS-less corners
        hy = [(h["name"], h["column"]) for h in cat.search("egion", mode="hybrid")]
        assert ("regions", "region_name") in hy
        assert cat.search("egion", mode="fts") == []
        assert [h["column"] for h in cat.search("qty", mode="like")] == ["qty"]
        # user text never reaches FTS syntax
        assert isinstance(cat.search('" OR * NEAR( col:x'), list)
        assert cat.search("!!!") == []


def test_incremental_recrawl_detects_schema_changes_and_removals(estate, tmp_path):
    with Catalog(tmp_path / "c.sqlite") as cat:
        crawl_into(cat, [estate])
        s = crawl_into(cat, [estate])
        assert (s.new, s.changed, s.unchanged, s.removed) == (0, 0, 7, 0)
        visits = estate / "files" / "visits.csv"
        visits.write_text(
            "visit_id,customer_id,ip_address,page,referrer\n1,1,10.0.0.1,/a,x\n",
            encoding="utf-8",
        )
        os.utime(visits, (time.time() + 5, time.time() + 5))
        s = crawl_into(cat, [estate])
        assert s.changed == 1 and s.changes == ["visits: added column referrer VARCHAR"]
        (estate / "files" / "nested" / "tags.jsonl").unlink()
        s = crawl_into(cat, [estate])
        assert s.removed == 1 and "tags" not in names(cat)
        log = [c["change"] for c in cat.changes()]
        assert log[:2] == ["removed", "added column referrer VARCHAR"]


def test_unreadable_source_is_reported_not_fatal(estate, tmp_path):
    (estate / "files" / "broken.parquet").write_bytes(b"not parquet")
    with Catalog(tmp_path / "c.sqlite") as cat:
        s = crawl_into(cat, [estate])
    assert s.new == 7
    assert len(s.errors) == 1 and "broken.parquet" in s.errors[0]


def test_cli(estate, tmp_path, capsys):
    db = str(tmp_path / "c.sqlite")
    assert main(["--catalog", db, "crawl", str(estate)]) == 0
    assert "7 new" in capsys.readouterr().out
    main(["--catalog", db, "search", "email", "--limit", "1"])
    assert "customers.email" in capsys.readouterr().out
    main(["--catalog", db, "show", "orders"])
    out = capsys.readouterr().out
    assert "primary key: order_id" in out and "declared" in out
    main(["--catalog", db, "show", "orders", "--json"])
    assert json.loads(capsys.readouterr().out)["rows"] == 100
    main(["--catalog", db, "relations"])
    assert "inferred visits.customer_id -> customers.id" in " ".join(
        capsys.readouterr().out.split()
    )
    main(["--catalog", db, "changes", "--limit", "3"])
    assert "new dataset" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no single dataset"):
        main(["--catalog", db, "show", "nope"])


def test_server(estate, tmp_path, monkeypatch):
    db = tmp_path / "c.sqlite"
    with Catalog(db) as cat:
        crawl_into(cat, [estate])
    monkeypatch.setenv("DATA_CATALOG_DB", str(db))
    from data_catalog.server import app

    c = TestClient(app, base_url="http://127.0.0.1")
    assert c.get("/", headers={"host": "evil.example"}).status_code == 400
    assert "textContent" in c.get("/").text
    hits = c.get("/api/search", params={"q": "email"}).json()
    assert hits[0]["column"] == "email"
    d = c.get("/api/dataset", params={"key": hits[0]["dataset"]}).json()
    assert d["name"] == "customers" and d["columns"][0]["name"] == "id"
    assert c.get("/api/dataset", params={"key": "nope"}).status_code == 404
    assert c.get("/api/search", params={"q": "x" * 201}).status_code == 422
    assert c.get("/api/search", params={"q": "email", "limit": 0}).status_code == 422


def test_duckdb_view_cannot_read_local_files(tmp_path):
    import duckdb

    secret = tmp_path / "secret.csv"
    secret.write_text("k\nTOPSECRET\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    con = duckdb.connect(str(src / "evil.duckdb"))
    con.execute(f"CREATE VIEW leak AS SELECT * FROM read_csv('{secret.as_posix()}')")
    con.close()
    with Catalog(tmp_path / "c.sqlite") as cat:
        s = crawl_into(cat, [src])
        assert s.new == 0 and "Permission" in s.errors[0]
        assert "TOPSECRET" not in json.dumps(
            [d.columns for d in cat.datasets()], default=str
        )


def test_symlinks_inside_crawled_folders_are_skipped(tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("secret\nTOPSECRET\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.csv").write_text("a\n1\n", encoding="utf-8")
    try:
        (src / "link.csv").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks needs privileges on this OS")
    assert [p.name for _, p in discover([src])] == ["ok.csv"]
