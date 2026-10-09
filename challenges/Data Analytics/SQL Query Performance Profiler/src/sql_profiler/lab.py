"""Rewrites worth testing, each run as an experiment instead of repeated as folklore.

Every case pairs a query with a rewrite and gets the same treatment: the rows must match (unless the
case says it cannot), then both are timed and their plans diffed. Several of them come out as "no
clear difference" or "same plan" on purpose: DuckDB's optimizer already normalizes most of the
classic anti-patterns, and knowing which advice is obsolete is part of the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from sql_profiler import db
from sql_profiler.compare import Comparison, Timing, compare, time_query
from sql_profiler.plan import profile


@dataclass(frozen=True)
class LabCase:
    id: str
    title: str
    idea: str
    a_sql: str
    b_sql: str
    equivalent: bool = True  # False when the rewrite deliberately returns less
    note: str = ""


LAB: tuple[LabCase, ...] = (
    LabCase(
        "function-on-key",
        "Function on the filter column vs a range",
        "`substr(invoice, 1, 3) = '536'` hides the column from min/max statistics; the equivalent "
        "range lets whole row groups be skipped.",
        "SELECT count(*), sum(quantity) FROM invoice_lines WHERE substr(invoice, 1, 3) = '536'",
        "SELECT count(*), sum(quantity) FROM invoice_lines WHERE invoice >= '536' AND invoice < '537'",
    ),
    LabCase(
        "in-vs-join",
        "IN (subquery) vs JOIN",
        "Folklore says IN is slow and a join is fast.",
        "SELECT count(*) FROM invoice_lines WHERE invoice IN "
        "(SELECT invoice FROM invoices WHERE country = 'France')",
        "SELECT count(*) FROM invoice_lines l JOIN invoices i USING (invoice) WHERE i.country = 'France'",
    ),
    LabCase(
        "correlated-vs-join",
        "Correlated subquery vs join to a pre-aggregate",
        "A subquery per row sounds quadratic; DuckDB decorrelates it into a join.",
        "SELECT i.invoice, (SELECT max(i2.invoice_date) FROM invoices i2 "
        "WHERE i2.customer_id = i.customer_id) AS last_date FROM invoices i",
        "SELECT i.invoice, m.last_date FROM invoices i LEFT JOIN "
        "(SELECT customer_id, max(invoice_date) AS last_date FROM invoices GROUP BY 1) m "
        "ON m.customer_id = i.customer_id",
        note="`IS NOT DISTINCT FROM` in the join would NOT be equivalent: guest invoices have a NULL "
        "customer, and the correlated version gives them NULL while a NULL-safe join gives them the "
        "date of every other guest invoice. The equivalence check catches it.",
    ),
    LabCase(
        "aggregate-before-join",
        "Aggregate before the join vs after",
        "Shrinking 1M lines to 53k invoice totals before joining should help.",
        "SELECT i.country, round(sum(l.quantity * l.price), 2) AS revenue FROM invoice_lines l "
        "JOIN invoices i USING (invoice) GROUP BY 1",
        "SELECT i.country, round(sum(t.revenue), 2) AS revenue FROM "
        "(SELECT invoice, sum(quantity * price) AS revenue FROM invoice_lines GROUP BY 1) t "
        "JOIN invoices i USING (invoice) GROUP BY 1",
    ),
    LabCase(
        "sort-vs-top-n",
        "Sort everything vs ORDER BY ... LIMIT",
        "Sorting a million rows to look at ten. Not the same query by design: the second returns only the top 10.",
        "SELECT * FROM invoice_lines ORDER BY price DESC, invoice, stock_code",
        "SELECT * FROM invoice_lines ORDER BY price DESC, invoice, stock_code LIMIT 10",
        equivalent=False,
    ),
    LabCase(
        "selfjoin-vs-window",
        "Self-join counting earlier rows vs a window function",
        "How many invoices did the same customer have strictly before this one?",
        "SELECT a.invoice, count(b.invoice) AS n_before FROM invoices a LEFT JOIN invoices b "
        "ON b.customer_id = a.customer_id AND b.invoice_date < a.invoice_date "
        "WHERE a.customer_id IS NOT NULL GROUP BY 1",
        "SELECT invoice, count(*) OVER (PARTITION BY customer_id ORDER BY invoice_date "
        "RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 SECOND PRECEDING) AS n_before "
        "FROM invoices WHERE customer_id IS NOT NULL",
        note="A ROWS frame over (date, invoice) would count same-minute invoices as earlier and "
        "return different numbers; the RANGE frame matches the strict `<`.",
    ),
    LabCase(
        "or-join-vs-union",
        "OR in a join condition vs a UNION of equi-joins",
        "Pairs of recent invoices that share a customer or a timestamp.",
        "WITH r AS (SELECT invoice, customer_id, invoice_date FROM invoices WHERE invoice_date >= '2011-11-01') "
        "SELECT count(*) FROM r a JOIN r b ON a.customer_id = b.customer_id OR a.invoice_date = b.invoice_date",
        "WITH r AS (SELECT invoice, customer_id, invoice_date FROM invoices WHERE invoice_date >= '2011-11-01') "
        "SELECT count(*) FROM (SELECT a.invoice AS x, b.invoice AS y FROM r a JOIN r b ON a.customer_id = b.customer_id "
        "UNION SELECT a.invoice, b.invoice FROM r a JOIN r b ON a.invoice_date = b.invoice_date)",
    ),
    LabCase(
        "not-in-vs-not-exists",
        "NOT IN vs NOT EXISTS",
        "Folklore says NOT EXISTS is the fast, null-safe one.",
        "SELECT count(*) FROM invoice_lines WHERE stock_code NOT IN "
        "(SELECT stock_code FROM products WHERE description LIKE '%HEART%')",
        "SELECT count(*) FROM invoice_lines l WHERE NOT EXISTS "
        "(SELECT 1 FROM products p WHERE p.stock_code = l.stock_code AND p.description LIKE '%HEART%')",
    ),
)


def run_case(
    con: duckdb.DuckDBPyConnection, case: LabCase, repeats: int = 9
) -> Comparison:
    return compare(con, case.a_sql, case.b_sql, repeats=repeats)


def run_lab(
    con: duckdb.DuckDBPyConnection, repeats: int = 9
) -> list[tuple[LabCase, Comparison]]:
    return [(case, run_case(con, case, repeats)) for case in LAB]


@dataclass(frozen=True)
class IndexResult:
    label: str
    sql: str
    rows: int
    without: Timing
    with_index: Timing
    scanned_without: int
    scanned_with: int
    scan_type_with: str


# (column, literal): chosen on the full data to span 7 rows, a few hundred, and thousands of matches
INDEX_LOOKUPS = (
    ("invoice", "'536365'"),
    ("stock_code", "'21744'"),
    ("stock_code", "'84828'"),
    ("stock_code", "'85123A'"),
)


def index_lab(root=None, repeats: int = 15) -> list[IndexResult]:
    """Time point lookups before and after ``CREATE INDEX``, on a private connection.

    DuckDB indexes are ART indexes. They change the plan only when the lookup is selective enough,
    and even then a lookup can be slower than a scan that min/max statistics already narrowed.
    """
    con = db.connect(root)
    queries = [
        (
            f"{col} = {lit}",
            f"SELECT count(*), sum(quantity) FROM invoice_lines WHERE {col} = {lit}",
        )
        for col, lit in INDEX_LOOKUPS
    ]
    before = [(time_query(con, q, repeats), profile(con, q)) for _, q in queries]
    for column in ("invoice", "stock_code"):
        con.execute(f"CREATE INDEX ix_{column} ON invoice_lines({column})")
    out = []
    for (label, q), (t0, p0) in zip(queries, before, strict=True):
        t1, p1 = time_query(con, q, repeats), profile(con, q)
        scan0 = next(n for n in p0.nodes if n.name == "TABLE_SCAN")
        scan1 = next(n for n in p1.nodes if n.name == "TABLE_SCAN")
        out.append(
            IndexResult(
                label,
                q,
                scan1.rows,
                t0,
                t1,
                scan0.rows_scanned,
                scan1.rows_scanned,
                scan1.info.get("Type", "?"),
            )
        )
    return out
