import sqlite3
from contextlib import closing

import duckdb
import pytest


@pytest.fixture
def estate(tmp_path):
    """Three sources: SQLite with a declared FK, DuckDB with constraints, a folder of files."""
    src = tmp_path / "src"
    (src / "files" / "nested").mkdir(parents=True)
    with closing(sqlite3.connect(src / "shop.sqlite")) as s:
        s.executescript(
            """
            CREATE TABLE customers(id INTEGER PRIMARY KEY, full_name TEXT, email TEXT);
            CREATE TABLE orders(order_id INTEGER PRIMARY KEY,
                                customer_id INTEGER REFERENCES customers(id), total REAL);
            """
        )
        s.executemany(
            "INSERT INTO customers VALUES (?,?,?)",
            [(i, f"Person {i}", f"p{i}@example.org") for i in range(1, 21)],
        )
        s.executemany(
            "INSERT INTO orders VALUES (?,?,?)",
            [(i, 1 + i % 20, i * 1.5) for i in range(1, 101)],
        )
        s.commit()
    with closing(duckdb.connect(str(src / "wh.duckdb"))) as d:
        d.execute("CREATE TABLE products(sku VARCHAR PRIMARY KEY, price DOUBLE)")
        d.execute("INSERT INTO products SELECT 'SKU' || i, i * 2.0 FROM range(30) t(i)")
        d.execute(
            "CREATE TABLE order_lines(order_id INTEGER, sku VARCHAR REFERENCES products(sku), qty INTEGER)"
        )
        d.execute(
            "INSERT INTO order_lines SELECT 1 + i % 100, 'SKU' || (i % 30), 1 + i % 5 FROM range(300) t(i)"
        )
    (src / "files" / "visits.csv").write_text(
        "visit_id,customer_id,ip_address,page\n"
        + "".join(
            f"{i},{1 + i % 20},10.0.0.{i % 250},/p{i % 7}\n" for i in range(1, 61)
        ),
        encoding="utf-8",
    )
    (src / "files" / "nested" / "tags.jsonl").write_text(
        "".join(f'{{"sku": "SKU{i}", "tag": "t{i % 3}"}}\n' for i in range(30)),
        encoding="utf-8",
    )
    with closing(duckdb.connect()) as d:
        d.execute(
            f"COPY (SELECT range AS region_id, 'R' || range AS region_name FROM range(5)) "
            f"TO '{(src / 'files' / 'regions.parquet').as_posix()}'"
        )
    (src / "files" / ".hidden.csv").write_text("a\n1\n", encoding="utf-8")
    (src / "files" / "notes.txt").write_text("ignored", encoding="utf-8")
    return src
