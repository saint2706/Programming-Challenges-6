"""The ETL tasks for one data interval: land -> stage -> dimensions -> facts -> aggregate -> reconcile.

Every task is idempotent: it deletes exactly what it is about to write, then writes it, in one
transaction. A re-run therefore leaves the same rows, and a crash inside a task rolls that task back.

Keys are deterministic hashes (``hash(natural key, valid_from)``) rather than sequences, so a rebuilt
dimension gets the same surrogate key it had before, and a fact that still points at it stays valid.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from star_warehouse.dag import Task
from star_warehouse.dq import expect, scalar
from star_warehouse.warehouse import transaction

CRASH_ENV = (
    "STAR_WAREHOUSE_CRASH_AT"  # test hook: exit the process inside the named task
)
FAR_FUTURE = "DATE '9999-12-31'"
ARRIVED = "arrival_date >= ? AND arrival_date < ?"
RAW_COLUMNS = (
    "line_key, invoice, stock_code, description, quantity, invoice_date, price, customer_id, "
    "country, arrival_date"
)
# Cleaning rules are the RFM challenge's, in the same order; a line is counted under the first rule
# that removes it. Returns are cancelled-invoice lines with negative quantity.
EXCLUSION_SQL = r"""
CASE
    WHEN NOT regexp_full_match(stock_code, '\d{5}[A-Za-z]{0,2}') THEN 'non_product'
    WHEN price <= 0 THEN 'bad_price'
    WHEN (is_cancellation AND quantity >= 0) OR (NOT is_cancellation AND quantity <= 0)
        THEN 'bad_quantity'
END
"""


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


@dataclass
class Batch:
    """One interval's run: the warehouse connection, the feed, and the interval ``[start, end)``."""

    con: duckdb.DuckDBPyConnection
    feed: Path
    lookup: Path
    start: date
    end: date

    @property
    def params(self) -> list[date]:
        return [self.start, self.end]

    @property
    def feed_sql(self) -> str:
        return f"read_parquet('{_sql_path(self.feed)}')"


def _crash_point(name: str) -> None:
    if os.environ.get(CRASH_ENV) == name:
        os._exit(75)  # no cleanup, no commit: exactly what a killed process looks like


def static_dims(b: Batch) -> Callable[[], None]:
    def run() -> None:
        with transaction(b.con):
            b.con.execute("DELETE FROM dw.dim_date")
            b.con.execute(
                """
                INSERT INTO dw.dim_date
                SELECT CAST(strftime(d, '%Y%m%d') AS INTEGER), d, year(d), quarter(d), month(d),
                       CAST(date_trunc('month', d) AS DATE), isodow(d), isodow(d) >= 6
                FROM (SELECT CAST(g AS DATE) AS d
                      FROM generate_series(DATE '2009-01-01', DATE '2013-12-31',
                                           INTERVAL 1 DAY) AS t(g))
                """
            )
            b.con.execute("DELETE FROM dw.dim_line_flags")
            b.con.execute(
                """
                INSERT INTO dw.dim_line_flags
                SELECT CAST(i AS INTEGER), (i // 4) % 2 = 1, (i // 2) % 2 = 1, i % 2 = 1
                FROM range(0, 8) AS t(i)
                """
            )
            # The unknown member: sales with no customer id point here instead of being dropped.
            b.con.execute("DELETE FROM dw.dim_customer WHERE customer_key = 0")
            b.con.execute(
                """
                INSERT INTO dw.dim_customer VALUES
                (0, NULL, 'Unknown', DATE '1900-01-01', DATE '9999-12-31', TRUE, 0)
                """
            )

    return run


def land_raw(b: Batch) -> Callable[[], None]:
    def run() -> None:
        with transaction(b.con):
            b.con.execute(f"DELETE FROM raw.feed_lines WHERE {ARRIVED}", b.params)
            b.con.execute(
                f"INSERT INTO raw.feed_lines SELECT {RAW_COLUMNS} FROM {b.feed_sql} "
                f"WHERE {ARRIVED}",
                b.params,
            )

    return run


def gate_landed(b: Batch) -> Callable[[], None]:
    def run() -> None:
        src = b.con.execute(
            f"SELECT count(*), sum(quantity * price) FROM {b.feed_sql} WHERE {ARRIVED}",
            b.params,
        ).fetchone()
        raw = b.con.execute(
            f"SELECT count(*), sum(quantity * price) FROM raw.feed_lines WHERE {ARRIVED}",
            b.params,
        ).fetchone()
        expect("landed rows vs feed rows", raw[0], src[0])
        expect("landed revenue vs feed revenue", raw[1], src[1])

    return run


def stage_lines(b: Batch) -> Callable[[], None]:
    def run() -> None:
        with transaction(b.con):
            b.con.execute(f"DELETE FROM staging.lines WHERE {ARRIVED}", b.params)
            b.con.execute(
                f"""
                INSERT INTO staging.lines
                WITH base AS (
                    SELECT *, invoice LIKE 'C%' AS is_cancellation
                    FROM raw.feed_lines WHERE {ARRIVED}
                ), classified AS (
                    SELECT *, {EXCLUSION_SQL} AS exclusion FROM base
                )
                SELECT line_key, invoice, stock_code, description, quantity, invoice_date, price,
                       customer_id, country, arrival_date, is_cancellation,
                       CASE WHEN is_cancellation THEN 'return' ELSE 'sale' END,
                       exclusion,
                       CAST(quantity * price AS DECIMAL(18, 4))
                FROM classified
                """,
                b.params,
            )

    return run


def gate_staged(b: Batch) -> Callable[[], None]:
    def run() -> None:
        dupes = scalar(
            b.con, "SELECT count(*) - count(DISTINCT line_key) FROM staging.lines"
        )
        expect("duplicate line keys in staging", dupes, 0)
        nulls = scalar(
            b.con,
            f"""SELECT count(*) FROM staging.lines WHERE {ARRIVED} AND (invoice IS NULL
                OR stock_code IS NULL OR invoice_date IS NULL OR quantity IS NULL
                OR price IS NULL OR country IS NULL)""",
            b.params,
        )
        expect("null key or measure columns in batch", nulls, 0)
        # Ledger: every feed line is either kept or excluded by exactly one named rule.
        src = b.con.execute(
            f"SELECT count(*), sum(quantity * price) FROM {b.feed_sql} WHERE {ARRIVED}",
            b.params,
        ).fetchone()
        staged = b.con.execute(
            f"SELECT count(*), sum(revenue) FROM staging.lines WHERE {ARRIVED}",
            b.params,
        ).fetchone()
        expect("staged rows vs feed rows", staged[0], src[0])
        expect("staged revenue vs feed revenue", staged[1], src[1])

    return run


def plan_batch(b: Batch) -> Callable[[], None]:
    """Temp tables that name what this interval changed: the kept lines and the entities they touch."""

    def run() -> None:
        b.con.execute(
            f"""CREATE OR REPLACE TEMP TABLE batch_keys AS
                SELECT line_key, customer_id, stock_code, invoice_date FROM staging.lines
                WHERE {ARRIVED} AND exclusion IS NULL""",
            b.params,
        )
        b.con.execute(
            """CREATE OR REPLACE TEMP TABLE affected_customers AS
               SELECT DISTINCT customer_id FROM batch_keys WHERE customer_id IS NOT NULL"""
        )
        b.con.execute(
            "CREATE OR REPLACE TEMP TABLE affected_products AS SELECT DISTINCT stock_code FROM batch_keys"
        )

    return run


def dim_country(b: Batch) -> Callable[[], None]:
    def run() -> None:
        with transaction(b.con):
            b.con.execute("DELETE FROM dw.dim_country")
            b.con.execute(
                f"""
                INSERT INTO dw.dim_country
                SELECT hash(c.country), c.country, COALESCE(r.region, 'Unknown')
                FROM (SELECT DISTINCT country FROM staging.lines WHERE exclusion IS NULL) c
                LEFT JOIN read_csv('{_sql_path(b.lookup)}', header = true) AS r(country, region)
                  ON r.country = c.country
                """
            )

    return run


def _customer_signature() -> str:
    return (
        "string_agg(country || '|' || CAST(valid_from AS VARCHAR) || '|' "
        "|| CAST(valid_to AS VARCHAR), ';' ORDER BY valid_from)"
    )


def dim_customer(b: Batch) -> Callable[[], None]:
    """SCD type 2 on the customer's country, from the country on each day they bought.

    A day's country is the most frequent one that day (ties: alphabetical). Consecutive days with the
    same country form one version; a version runs until the day before the next one starts.
    """

    def run() -> None:
        with transaction(b.con):
            c = b.con
            c.execute(
                f"""CREATE OR REPLACE TEMP TABLE cust_before AS
                    SELECT customer_id, {_customer_signature()} AS sig FROM dw.dim_customer
                    WHERE customer_id IN (SELECT customer_id FROM affected_customers)
                    GROUP BY customer_id"""
            )
            c.execute(
                """CREATE OR REPLACE TEMP TABLE cust_day AS
                   WITH counts AS (
                       SELECT s.customer_id, s.invoice_date, s.country, count(*) AS n
                       FROM staging.lines s
                       JOIN affected_customers a ON a.customer_id = s.customer_id
                       WHERE s.exclusion IS NULL
                       GROUP BY ALL
                   )
                   SELECT customer_id, invoice_date, country FROM (
                       SELECT *, row_number() OVER (
                           PARTITION BY customer_id, invoice_date ORDER BY n DESC, country) AS rk
                       FROM counts)
                   WHERE rk = 1"""
            )
            c.execute(
                """CREATE OR REPLACE TEMP TABLE cust_runs AS
                   WITH flagged AS (
                       SELECT customer_id, invoice_date, country,
                           CASE WHEN row_number() OVER w = 1
                                  OR lag(country) OVER w <> country THEN 1 ELSE 0 END AS starts
                       FROM cust_day WINDOW w AS (PARTITION BY customer_id ORDER BY invoice_date)
                   ), grouped AS (
                       SELECT *, sum(starts) OVER (
                           PARTITION BY customer_id ORDER BY invoice_date) AS run_no
                       FROM flagged
                   )
                   SELECT customer_id, country, min(invoice_date) AS valid_from, run_no
                   FROM grouped GROUP BY customer_id, country, run_no"""
            )
            c.execute(
                f"""CREATE OR REPLACE TEMP TABLE cust_versions AS
                    SELECT customer_id, country, valid_from,
                        COALESCE(CAST(lead(valid_from) OVER (
                            PARTITION BY customer_id ORDER BY valid_from) - 1 AS DATE),
                            {FAR_FUTURE}) AS valid_to,
                        row_number() OVER (PARTITION BY customer_id ORDER BY valid_from) AS version_no
                    FROM cust_runs"""
            )
            c.execute(
                "DELETE FROM dw.dim_customer WHERE customer_id IN "
                "(SELECT customer_id FROM affected_customers)"
            )
            c.execute(
                f"""INSERT INTO dw.dim_customer
                    SELECT hash(customer_id, valid_from), customer_id, country,
                           valid_from, valid_to, valid_to = {FAR_FUTURE}, version_no
                    FROM cust_versions"""
            )
            c.execute(
                f"""CREATE OR REPLACE TEMP TABLE changed_customers AS
                    SELECT a.customer_id FROM cust_before a
                    FULL OUTER JOIN (SELECT customer_id, {_customer_signature()} AS sig
                                     FROM dw.dim_customer
                                     WHERE customer_id IN (SELECT customer_id FROM affected_customers)
                                     GROUP BY customer_id) b USING (customer_id)
                    WHERE a.sig IS DISTINCT FROM b.sig"""
            )

    return run


def _product_signature() -> str:
    return (
        "string_agg(coalesce(description, '<null>') || '|' || CAST(list_price AS VARCHAR) || '|' "
        "|| CAST(valid_from AS VARCHAR) || '|' || CAST(valid_to AS VARCHAR), ';' "
        "ORDER BY valid_from)"
    )


def dim_product(b: Batch) -> Callable[[], None]:
    """SCD type 2 on (description, list price). The list price is the most frequent unit price per
    product per calendar month (the "price book"); description is the most frequent per month too.
    Consecutive months with the same pair form one version.
    """

    def run() -> None:
        with transaction(b.con):
            c = b.con
            c.execute(
                f"""CREATE OR REPLACE TEMP TABLE prod_before AS
                    SELECT stock_code, {_product_signature()} AS sig FROM dw.dim_product
                    WHERE stock_code IN (SELECT stock_code FROM affected_products)
                    GROUP BY stock_code"""
            )
            c.execute(
                """CREATE OR REPLACE TEMP TABLE prod_base AS
                   SELECT s.stock_code, CAST(date_trunc('month', s.invoice_date) AS DATE) AS month_start,
                          s.price, s.description
                   FROM staging.lines s
                   WHERE s.exclusion IS NULL
                     AND s.stock_code IN (SELECT stock_code FROM affected_products)"""
            )
            c.execute(
                """CREATE OR REPLACE TEMP TABLE prod_month AS
                   WITH price_counts AS (
                       SELECT stock_code, month_start, price, count(*) AS n
                       FROM prod_base GROUP BY ALL),
                   price_mode AS (
                       SELECT stock_code, month_start, price AS list_price FROM (
                           SELECT *, row_number() OVER (
                               PARTITION BY stock_code, month_start ORDER BY n DESC, price) AS rk
                           FROM price_counts) WHERE rk = 1),
                   desc_counts AS (
                       SELECT stock_code, month_start, description, count(*) AS n
                       FROM prod_base GROUP BY ALL),
                   desc_mode AS (
                       SELECT stock_code, month_start, description FROM (
                           SELECT *, row_number() OVER (
                               PARTITION BY stock_code, month_start ORDER BY n DESC, description) AS rk
                           FROM desc_counts) WHERE rk = 1)
                   SELECT p.stock_code, p.month_start, d.description, p.list_price
                   FROM price_mode p JOIN desc_mode d USING (stock_code, month_start)"""
            )
            c.execute(
                """CREATE OR REPLACE TEMP TABLE prod_runs AS
                   WITH flagged AS (
                       SELECT stock_code, month_start, description, list_price,
                           CASE WHEN row_number() OVER w = 1
                                  OR lag(description) OVER w IS DISTINCT FROM description
                                  OR lag(list_price) OVER w IS DISTINCT FROM list_price
                                THEN 1 ELSE 0 END AS starts
                       FROM prod_month WINDOW w AS (PARTITION BY stock_code ORDER BY month_start)
                   ), grouped AS (
                       SELECT *, sum(starts) OVER (
                           PARTITION BY stock_code ORDER BY month_start) AS run_no
                       FROM flagged
                   )
                   SELECT stock_code, description, list_price, min(month_start) AS valid_from, run_no
                   FROM grouped GROUP BY stock_code, description, list_price, run_no"""
            )
            c.execute(
                f"""CREATE OR REPLACE TEMP TABLE prod_versions AS
                    SELECT stock_code, description, list_price, valid_from,
                        COALESCE(CAST(lead(valid_from) OVER (
                            PARTITION BY stock_code ORDER BY valid_from) - 1 AS DATE),
                            {FAR_FUTURE}) AS valid_to,
                        row_number() OVER (PARTITION BY stock_code ORDER BY valid_from) AS version_no
                    FROM prod_runs"""
            )
            c.execute(
                "DELETE FROM dw.dim_product WHERE stock_code IN "
                "(SELECT stock_code FROM affected_products)"
            )
            c.execute(
                f"""INSERT INTO dw.dim_product
                    SELECT hash(stock_code, valid_from), stock_code, description,
                           list_price, valid_from, valid_to, valid_to = {FAR_FUTURE}, version_no
                    FROM prod_versions"""
            )
            c.execute(
                f"""CREATE OR REPLACE TEMP TABLE changed_products AS
                    SELECT a.stock_code FROM prod_before a
                    FULL OUTER JOIN (SELECT stock_code, {_product_signature()} AS sig
                                     FROM dw.dim_product
                                     WHERE stock_code IN (SELECT stock_code FROM affected_products)
                                     GROUP BY stock_code) b USING (stock_code)
                    WHERE a.sig IS DISTINCT FROM b.sig"""
            )

    return run


def gate_dims(b: Batch) -> Callable[[], None]:
    def run() -> None:
        c = b.con
        for table, key in (
            ("dw.dim_customer", "customer_id"),
            ("dw.dim_product", "stock_code"),
        ):
            current = scalar(
                c,
                f"""SELECT count(*) FROM (SELECT {key} FROM {table} WHERE {key} IS NOT NULL
                    GROUP BY {key} HAVING sum(CAST(is_current AS INTEGER)) <> 1)""",
            )
            expect(
                f"{table}: natural keys without exactly one current version", current, 0
            )
            overlaps = scalar(
                c,
                f"""SELECT count(*) FROM (SELECT valid_from, valid_to, lead(valid_from) OVER w AS nxt
                    FROM {table} WHERE {key} IS NOT NULL
                    WINDOW w AS (PARTITION BY {key} ORDER BY valid_from))
                    WHERE valid_from > valid_to OR (nxt IS NOT NULL AND nxt <> valid_to + 1)""",
            )
            expect(f"{table}: versions overlap, gap or are inverted", overlaps, 0)
            dupes = scalar(
                c,
                f"SELECT count(*) - count(DISTINCT {table.split('.')[1][4:]}_key) FROM {table}",
            )
            expect(f"{table}: duplicate surrogate keys", dupes, 0)

    return run


def _rebuild_temp(b: Batch) -> None:
    b.con.execute(
        f"""CREATE OR REPLACE TEMP TABLE rebuild_lines AS
            SELECT s.line_key FROM staging.lines s
            WHERE s.exclusion IS NULL AND (
                {ARRIVED}
                OR s.customer_id IN (SELECT customer_id FROM changed_customers)
                OR s.stock_code IN (SELECT stock_code FROM changed_products))""",
        b.params,
    )


def facts(b: Batch) -> Callable[[], None]:
    """Fact rows for the interval's lines, and for older lines whose customer or product version changed.

    Restating older lines is what makes a late invoice correct: the point-in-time keys of every fact
    that depends on a rebuilt dimension version are re-resolved in the same transaction.
    """

    def run() -> None:
        with transaction(b.con):
            c = b.con
            _rebuild_temp(b)
            c.execute(
                "DELETE FROM dw.fact_sales WHERE line_key IN (SELECT line_key FROM rebuild_lines)"
            )
            c.execute(
                "DELETE FROM dw.fact_returns WHERE line_key IN (SELECT line_key FROM rebuild_lines)"
            )
            _crash_point("facts")
            c.execute(
                """
                INSERT INTO dw.fact_sales
                SELECT s.line_key, s.invoice,
                       CAST(strftime(s.invoice_date, '%Y%m%d') AS INTEGER),
                       CAST(strftime(s.arrival_date, '%Y%m%d') AS INTEGER),
                       p.product_key, COALESCE(cu.customer_key, 0), hash(s.country),
                       CAST(s.is_cancellation AS INTEGER) * 4
                         + CAST(s.customer_id IS NULL AS INTEGER) * 2
                         + CAST(s.quantity < 0 AS INTEGER),
                       s.quantity, s.price, s.revenue,
                       CAST(s.quantity * p.list_price AS DECIMAL(18, 4))
                FROM staging.lines s
                JOIN rebuild_lines r ON r.line_key = s.line_key
                JOIN dw.dim_product p ON p.stock_code = s.stock_code
                     AND s.invoice_date BETWEEN p.valid_from AND p.valid_to
                LEFT JOIN dw.dim_customer cu ON cu.customer_id = s.customer_id
                     AND s.invoice_date BETWEEN cu.valid_from AND cu.valid_to
                WHERE s.kind = 'sale'
                """
            )
            c.execute(
                """
                INSERT INTO dw.fact_returns
                SELECT s.line_key, s.invoice,
                       CAST(strftime(s.invoice_date, '%Y%m%d') AS INTEGER),
                       CAST(strftime(s.arrival_date, '%Y%m%d') AS INTEGER),
                       p.product_key, COALESCE(cu.customer_key, 0), hash(s.country),
                       CAST(s.is_cancellation AS INTEGER) * 4
                         + CAST(s.customer_id IS NULL AS INTEGER) * 2
                         + CAST(s.quantity < 0 AS INTEGER),
                       s.quantity, s.price, s.revenue
                FROM staging.lines s
                JOIN rebuild_lines r ON r.line_key = s.line_key
                JOIN dw.dim_product p ON p.stock_code = s.stock_code
                     AND s.invoice_date BETWEEN p.valid_from AND p.valid_to
                LEFT JOIN dw.dim_customer cu ON cu.customer_id = s.customer_id
                     AND s.invoice_date BETWEEN cu.valid_from AND cu.valid_to
                WHERE s.kind = 'return'
                """
            )
            # Inside the transaction, so a line that fails to resolve rolls the whole task back.
            sales = scalar(
                c,
                "SELECT count(*) FROM staging.lines s JOIN rebuild_lines r USING (line_key) WHERE s.kind = 'sale'",
            )
            got_sales = scalar(
                c,
                "SELECT count(*) FROM dw.fact_sales WHERE line_key IN (SELECT line_key FROM rebuild_lines)",
            )
            expect("sales lines resolved to exactly one fact row", got_sales, sales)
            returns = scalar(
                c,
                "SELECT count(*) FROM staging.lines s JOIN rebuild_lines r USING (line_key) WHERE s.kind = 'return'",
            )
            got_returns = scalar(
                c,
                "SELECT count(*) FROM dw.fact_returns WHERE line_key IN (SELECT line_key FROM rebuild_lines)",
            )
            expect(
                "return lines resolved to exactly one fact row", got_returns, returns
            )

    return run


def gate_facts(b: Batch) -> Callable[[], None]:
    def run() -> None:
        c = b.con
        expect(
            "fact_sales rows vs kept sale lines",
            scalar(c, "SELECT count(*) FROM dw.fact_sales"),
            scalar(
                c,
                "SELECT count(*) FROM staging.lines WHERE kind = 'sale' AND exclusion IS NULL",
            ),
        )
        expect(
            "fact_returns rows vs kept return lines",
            scalar(c, "SELECT count(*) FROM dw.fact_returns"),
            scalar(
                c,
                "SELECT count(*) FROM staging.lines WHERE kind = 'return' AND exclusion IS NULL",
            ),
        )
        orphans = scalar(
            c,
            """SELECT
                 (SELECT count(*) FROM dw.fact_sales f LEFT JOIN dw.dim_product p USING (product_key)
                  WHERE p.product_key IS NULL)
               + (SELECT count(*) FROM dw.fact_sales f LEFT JOIN dw.dim_customer d USING (customer_key)
                  WHERE d.customer_key IS NULL)
               + (SELECT count(*) FROM dw.fact_sales f LEFT JOIN dw.dim_country d USING (country_key)
                  WHERE d.country_key IS NULL)
               + (SELECT count(*) FROM dw.fact_sales f LEFT JOIN dw.dim_date d
                  ON d.date_key = f.invoice_date_key WHERE d.date_key IS NULL)
               + (SELECT count(*) FROM dw.fact_returns f LEFT JOIN dw.dim_product p USING (product_key)
                  WHERE p.product_key IS NULL)
               + (SELECT count(*) FROM dw.fact_returns f LEFT JOIN dw.dim_customer d USING (customer_key)
                  WHERE d.customer_key IS NULL)""",
        )
        expect("orphan foreign keys in facts", orphans, 0)
        total = scalar(
            c,
            """SELECT (SELECT sum(sales_amount) FROM dw.fact_sales)
                    + (SELECT coalesce(sum(return_amount), 0) FROM dw.fact_returns)""",
        )
        kept = scalar(
            c, "SELECT sum(revenue) FROM staging.lines WHERE exclusion IS NULL"
        )
        expect("fact revenue vs kept staged revenue", total, kept)

    return run


def agg_country_month(b: Batch) -> Callable[[], None]:
    def run() -> None:
        with transaction(b.con):
            c = b.con
            c.execute(
                """CREATE OR REPLACE TEMP TABLE touched_months AS
                   SELECT DISTINCT CAST(date_trunc('month', s.invoice_date) AS DATE) AS month_start
                   FROM staging.lines s JOIN rebuild_lines r USING (line_key)"""
            )
            c.execute(
                """DELETE FROM dw.agg_country_month
                   WHERE month_start IN (SELECT month_start FROM touched_months)"""
            )
            c.execute(
                """
                INSERT INTO dw.agg_country_month
                WITH months AS (
                    SELECT date_key FROM dw.dim_date
                    WHERE month_start IN (SELECT month_start FROM touched_months)
                ), lines AS (
                    SELECT invoice_date_key AS dk, country_key, sales_amount AS sales,
                           CAST(0 AS DECIMAL(18, 4)) AS returns, 1 AS sales_n, 0 AS returns_n
                    FROM dw.fact_sales WHERE invoice_date_key IN (SELECT date_key FROM months)
                    UNION ALL
                    SELECT invoice_date_key, country_key, CAST(0 AS DECIMAL(18, 4)), return_amount,
                           0, 1
                    FROM dw.fact_returns WHERE invoice_date_key IN (SELECT date_key FROM months)
                ), summed AS (
                    SELECT d.month_start, l.country_key, sum(l.sales) AS sales_amount,
                           sum(l.returns) AS returns_amount, sum(l.sales_n) AS sales_lines,
                           sum(l.returns_n) AS return_lines
                    FROM lines l JOIN dw.dim_date d ON d.date_key = l.dk
                    GROUP BY d.month_start, l.country_key
                )
                SELECT s.month_start, s.country_key, k.country, s.sales_amount, s.returns_amount,
                       s.sales_amount + s.returns_amount, s.sales_lines, s.return_lines
                FROM summed s JOIN dw.dim_country k ON k.country_key = s.country_key
                """
            )

    return run


def reconcile(b: Batch) -> Callable[[], None]:
    """The aggregate must add up to the facts it summarises, for every month."""

    def run() -> None:
        expect(
            "agg net amount vs fact amounts",
            scalar(b.con, "SELECT sum(net_amount) FROM dw.agg_country_month"),
            scalar(
                b.con,
                """SELECT (SELECT sum(sales_amount) FROM dw.fact_sales)
                        + (SELECT coalesce(sum(return_amount), 0) FROM dw.fact_returns)""",
            ),
        )
        expect(
            "agg line counts vs fact row counts",
            scalar(
                b.con,
                "SELECT sum(sales_lines) + sum(return_lines) FROM dw.agg_country_month",
            ),
            scalar(
                b.con,
                "SELECT (SELECT count(*) FROM dw.fact_sales) + (SELECT count(*) FROM dw.fact_returns)",
            ),
        )

    return run


def build_tasks(b: Batch) -> list[Task]:
    """The DAG for one interval. Upstream failure skips the dependants; unrelated branches still run."""
    return [
        Task(
            "static_dims",
            static_dims(b),
            writes=("dw.dim_date", "dw.dim_line_flags", "dw.dim_customer"),
        ),
        Task("land_raw", land_raw(b), writes=("raw.feed_lines",), retries=1),
        Task("gate_landed", gate_landed(b), deps=("land_raw",)),
        Task(
            "stage_lines",
            stage_lines(b),
            deps=("land_raw",),
            writes=("staging.lines",),
            retries=1,
        ),
        Task("gate_staged", gate_staged(b), deps=("stage_lines", "gate_landed")),
        Task("plan_batch", plan_batch(b), deps=("gate_staged",)),
        Task(
            "dim_country",
            dim_country(b),
            deps=("gate_staged",),
            writes=("dw.dim_country",),
        ),
        Task(
            "dim_customer",
            dim_customer(b),
            deps=("plan_batch",),
            writes=("dw.dim_customer",),
        ),
        Task(
            "dim_product",
            dim_product(b),
            deps=("plan_batch",),
            writes=("dw.dim_product",),
        ),
        Task(
            "gate_dims",
            gate_dims(b),
            deps=("dim_customer", "dim_product", "dim_country", "static_dims"),
        ),
        Task(
            "facts",
            facts(b),
            deps=("gate_dims",),
            writes=("dw.fact_sales", "dw.fact_returns"),
            retries=1,
        ),
        Task("gate_facts", gate_facts(b), deps=("facts",)),
        Task(
            "agg_country_month",
            agg_country_month(b),
            deps=("gate_facts",),
            writes=("dw.agg_country_month",),
        ),
        Task("reconcile", reconcile(b), deps=("agg_country_month",)),
    ]


__all__ = ["Batch", "build_tasks"]
