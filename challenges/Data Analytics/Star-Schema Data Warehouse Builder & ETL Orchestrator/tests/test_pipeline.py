"""Pipeline behaviour: idempotence, incremental equals one-shot, crash recovery, late data, retries,
gates on corrupted data, and the scheduler. Every run uses the committed sample and a temp database."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from datetime import date, datetime

import polars as pl
import pytest
from helpers import LOOKUP, SAMPLE, frame, load, raw_clean
from star_warehouse.dag import PermanentError, TransientError
from star_warehouse.dq import DataQualityError
from star_warehouse.pipeline import backfill, months, run_interval, tick
from star_warehouse.tasks import CRASH_ENV, Batch, gate_dims, gate_facts
from star_warehouse.warehouse import connect, content_hash


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The one-shot build every other path must reproduce exactly."""
    db = tmp_path_factory.mktemp("ref") / "ref.duckdb"
    assert load(db).status == "succeeded"
    con = connect(db)
    digest = content_hash(con)
    con.close()
    return digest


def test_rerun_of_same_interval_leaves_identical_content(tmp_path):
    db = tmp_path / "twice.duckdb"
    assert load(db).status == "succeeded"
    con = connect(db)
    first = content_hash(con)
    con.close()
    assert load(db).status == "succeeded"
    con = connect(db)
    second = content_hash(con)
    runs = con.execute(
        "SELECT count(*) FROM ops.run WHERE status = 'succeeded'"
    ).fetchone()[0]
    con.close()
    assert second == first
    assert runs == 2  # the run log grows; the content does not


def test_quarterly_incremental_equals_one_shot(tmp_path, reference):
    db = tmp_path / "quarters.duckdb"
    quarter_starts = [
        date(2009, 12, 1),
        date(2010, 3, 1),
        date(2010, 6, 1),
        date(2010, 9, 1),
        date(2010, 12, 1),
        date(2011, 3, 1),
        date(2011, 6, 1),
        date(2011, 9, 1),
        date(2011, 12, 1),
        date(2012, 3, 1),
        date(2012, 6, 1),
    ]
    for start, end in itertools.pairwise(quarter_starts):
        outcome = load(db, start=start, end=end)
        assert outcome.status == "succeeded", (start, outcome.failed_tasks)
    con = connect(db)
    digest = content_hash(con)
    con.close()
    assert digest == reference


def test_late_invoice_restates_its_month_and_its_fact(tmp_path):
    # Invoices from before July 2010 that only arrive in July: the July run must restate their month.
    late = raw_clean().filter(
        pl.col("invoice_date") < date(2010, 7, 1),
        pl.col("arrival_date").is_between(date(2010, 7, 1), date(2010, 7, 31)),
    )
    assert late.height > 0, "the sample has no kept invoices that arrive in July 2010"
    month = late.sort("invoice_date")["invoice_date"][0].replace(day=1)
    db = tmp_path / "late.duckdb"
    assert load(db, start=date(2009, 1, 1), end=date(2010, 7, 1)).status == "succeeded"
    con = connect(db)
    lines_before = con.execute(
        "SELECT coalesce(sum(sales_lines + return_lines), 0) FROM dw.agg_country_month "
        "WHERE month_start = ?",
        [month],
    ).fetchone()[0]
    con.close()
    assert load(db, start=date(2010, 7, 1), end=date(2010, 8, 1)).status == "succeeded"

    con = connect(db)
    landed = frame(
        con,
        "SELECT line_key FROM dw.fact_sales UNION ALL SELECT line_key FROM dw.fact_returns",
    )
    lines_after = con.execute(
        "SELECT coalesce(sum(sales_lines + return_lines), 0) FROM dw.agg_country_month "
        "WHERE month_start = ?",
        [month],
    ).fetchone()[0]
    con.close()
    assert set(late["line_key"].to_list()) <= set(landed["line_key"].to_list())
    assert lines_after > lines_before, (
        "the invoice month was not restated by the late arrival"
    )


def test_agg_matches_oracle_after_late_arrivals(tmp_path):
    db = tmp_path / "agg.duckdb"
    assert load(db, start=date(2009, 1, 1), end=date(2010, 7, 1)).status == "succeeded"
    assert load(db, start=date(2010, 7, 1), end=date(2010, 8, 1)).status == "succeeded"
    con = connect(db)
    star = frame(
        con,
        "SELECT month_start, country, net_amount AS net, sales_lines + return_lines AS lines "
        "FROM dw.agg_country_month",
    )
    con.close()
    arrived = raw_clean().filter(pl.col("arrival_date") < date(2010, 8, 1))
    oracle = arrived.group_by(["month_start", "country"]).agg(
        net=pl.col("amount").cast(pl.Float64).sum(), lines=pl.len()
    )
    joined = star.with_columns(pl.col("net").cast(pl.Float64)).join(
        oracle, on=["month_start", "country"], how="full", suffix="_o", coalesce=True
    )
    assert joined.height == star.height == oracle.height
    assert joined["lines"].to_list() == joined["lines_o"].to_list()
    assert (joined["net"] - joined["net_o"]).abs().max() < 1e-6


CHILD = """
import sys
from datetime import date
from pathlib import Path
from star_warehouse.pipeline import run_interval
run_interval(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
             date(2009, 1, 1), date(2013, 1, 1), sleep=lambda s: None)
"""


def test_crash_mid_task_then_rerun_has_no_duplicates(tmp_path, reference):
    db = tmp_path / "crash.duckdb"
    env = {**os.environ, CRASH_ENV: "facts"}
    child = subprocess.run(  # noqa: PLW1510 - the returncode is asserted just below
        [sys.executable, "-c", CHILD, str(db), str(SAMPLE), str(LOOKUP)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 75, child.stderr[-2000:]  # killed, no clean shutdown

    con = connect(db)
    stuck = con.execute("SELECT status FROM ops.run").fetchall()
    rows_before = con.execute("SELECT count(*) FROM dw.fact_sales").fetchone()[0]
    con.close()
    assert stuck == [("running",)]
    assert rows_before == 0, "the facts task committed rows before it was killed"

    outcome = load(db)
    assert outcome.status == "succeeded", outcome.failed_tasks
    con = connect(db)
    statuses = sorted(
        r[0] for r in con.execute("SELECT status FROM ops.run").fetchall()
    )
    dupes = con.execute(
        "SELECT (SELECT count(*) - count(DISTINCT line_key) FROM dw.fact_sales) "
        "+ (SELECT count(*) - count(DISTINCT line_key) FROM dw.fact_returns)"
    ).fetchone()[0]
    digest = content_hash(con)
    con.close()
    assert statuses == ["interrupted", "succeeded"]
    assert dupes == 0
    assert digest == reference


def test_transient_fault_is_retried_and_permanent_fault_is_not(tmp_path):
    db = tmp_path / "faults.duckdb"
    outcome = run_interval(
        db,
        SAMPLE,
        LOOKUP,
        date(2009, 1, 1),
        date(2013, 1, 1),
        faults={"facts": [TransientError("lock timeout (injected)")]},
        backoff=0.0,
        sleep=lambda s: None,
    )
    assert outcome.status == "succeeded"
    attempts = outcome.results["facts"].attempts
    assert [a.status for a in attempts] == ["error", "ok"]

    db2 = tmp_path / "permanent.duckdb"
    outcome = run_interval(
        db2,
        SAMPLE,
        LOOKUP,
        date(2009, 1, 1),
        date(2013, 1, 1),
        faults={"gate_staged": [DataQualityError("injected gate failure")]},
        backoff=0.0,
        sleep=lambda s: None,
    )
    assert outcome.status == "failed"
    assert len(outcome.results["gate_staged"].attempts) == 1  # not retried
    assert outcome.results["dim_customer"].status == "skipped"
    # Skips name the direct blocker; the chain leads back to the failed gate.
    assert (
        outcome.results["plan_batch"].skipped_because == "upstream gate_staged failed"
    )
    assert (
        outcome.results["dim_customer"].skipped_because == "upstream plan_batch skipped"
    )
    assert (
        outcome.results["static_dims"].status == "succeeded"
    )  # unrelated branch still ran
    con = connect(db2)
    logged = con.execute(
        "SELECT task, status FROM ops.task_run WHERE status = 'skipped' ORDER BY task"
    ).fetchall()
    con.close()
    assert ("dim_customer", "skipped") in logged


def test_gate_dims_catches_a_broken_scd2_chain(tmp_path):
    db = tmp_path / "gate.duckdb"
    assert load(db).status == "succeeded"
    con = connect(db)
    con.execute(
        "UPDATE dw.dim_product SET is_current = FALSE WHERE stock_code = "
        "(SELECT min(stock_code) FROM dw.dim_product WHERE is_current)"
    )
    batch = Batch(
        con=con,
        feed=SAMPLE,
        lookup=LOOKUP,
        start=date(2009, 1, 1),
        end=date(2013, 1, 1),
    )
    with pytest.raises(DataQualityError, match="current version"):
        gate_dims(batch)()
    con.close()


def test_gate_facts_catches_an_orphan_foreign_key(tmp_path):
    db = tmp_path / "orphan.duckdb"
    assert load(db).status == "succeeded"
    con = connect(db)
    con.execute(
        "UPDATE dw.fact_sales SET product_key = 1 WHERE line_key = "
        "(SELECT min(line_key) FROM dw.fact_sales)"
    )
    batch = Batch(
        con=con,
        feed=SAMPLE,
        lookup=LOOKUP,
        start=date(2009, 1, 1),
        end=date(2013, 1, 1),
    )
    with pytest.raises(DataQualityError, match="orphan"):
        gate_facts(batch)()
    con.close()


def test_months_cover_the_range_exactly():
    assert months(date(2010, 1, 15), date(2010, 4, 1)) == [
        (date(2010, 1, 1), date(2010, 2, 1)),
        (date(2010, 2, 1), date(2010, 3, 1)),
        (date(2010, 3, 1), date(2010, 4, 1)),
    ]
    assert months(date(2011, 12, 1), date(2012, 2, 1)) == [
        (date(2011, 12, 1), date(2012, 1, 1)),
        (date(2012, 1, 1), date(2012, 2, 1)),
    ]


def test_backfill_runs_each_month_in_order(tmp_path):
    db = tmp_path / "backfill.duckdb"
    outcomes = backfill(
        db,
        SAMPLE,
        LOOKUP,
        date(2010, 1, 1),
        date(2010, 3, 1),
        backoff=0.0,
        sleep=lambda s: None,
    )
    assert [(o.start, o.end) for o in outcomes] == [
        (date(2010, 1, 1), date(2010, 2, 1)),
        (date(2010, 2, 1), date(2010, 3, 1)),
    ]
    assert all(o.status == "succeeded" for o in outcomes)


def test_scheduler_runs_each_due_fire_once_and_retries_failures(tmp_path):
    db = tmp_path / "sched.duckdb"
    expr = "0 2 * * *"
    quiet = {"backoff": 0.0, "sleep": lambda s: None}

    first = tick(db, SAMPLE, LOOKUP, expr, datetime(2010, 3, 5, 3, 0), **quiet)
    assert [(o.start, o.end) for o in first] == [(date(2010, 3, 4), date(2010, 3, 5))]

    second = tick(db, SAMPLE, LOOKUP, expr, datetime(2010, 3, 7, 3, 0), **quiet)
    assert [(o.start, o.end) for o in second] == [
        (date(2010, 3, 5), date(2010, 3, 6)),
        (date(2010, 3, 6), date(2010, 3, 7)),
    ]
    assert tick(db, SAMPLE, LOOKUP, expr, datetime(2010, 3, 7, 3, 0), **quiet) == []

    # A failed fire does not advance the schedule, so the next tick runs it again.
    failing = tick(
        db,
        SAMPLE,
        LOOKUP,
        expr,
        datetime(2010, 3, 8, 3, 0),
        faults={"facts": [PermanentError("injected")]},
        **quiet,
    )
    assert [o.status for o in failing] == ["failed"]
    retry = tick(db, SAMPLE, LOOKUP, expr, datetime(2010, 3, 8, 3, 0), **quiet)
    assert [(o.start, o.end) for o in retry] == [(date(2010, 3, 7), date(2010, 3, 8))]
    assert retry[0].status == "succeeded"


def test_schedule_that_fires_twice_a_day_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at most once a day"):
        tick(
            tmp_path / "x.duckdb", SAMPLE, LOOKUP, "0 2,14 * * *", datetime(2010, 1, 1)
        )
