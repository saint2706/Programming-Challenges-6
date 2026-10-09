import duckdb
import pytest
from helpers import op, scan, wrap
from sql_profiler import db
from sql_profiler.plan import parse_profile, profile, run_with_timeout


def sample_plan():
    return parse_profile(
        wrap(
            op(
                "HASH_JOIN",
                rows=100,
                est="1,000",
                time=0.03,
                children=[
                    scan(
                        "invoice_lines",
                        5000,
                        scanned=9000,
                        est=9000,
                        time=0.06,
                        Filters="price>1",
                    ),
                    scan("products", 40, est=0, time=0.01),
                ],
                Join_Type="INNER",
                Conditions="stock_code = stock_code",
            ),
            latency=0.05,
            cpu=0.1,
        ),
        sql="select 1",
    )


def test_the_explain_wrapper_is_removed_and_the_tree_is_numbered_in_preorder():
    plan = sample_plan()
    assert [n.name for n in plan.nodes] == ["HASH_JOIN", "TABLE_SCAN", "TABLE_SCAN"]
    assert [n.id for n in plan.nodes] == [0, 1, 2]
    assert [n.depth for n in plan.nodes] == [0, 1, 1]
    assert plan.nodes[1].parent is plan.root


def test_a_profile_without_the_wrapper_parses_the_same():
    inner = op("TOP_N", rows=5, children=[scan("products", 10)])
    bare = {"latency": 0.01, "cpu_time": 0.02, "children": [inner]}
    assert [n.name for n in parse_profile(bare).nodes] == ["TOP_N", "TABLE_SCAN"]


def test_numbers_are_read_from_the_profile():
    plan = sample_plan()
    join, big, small = plan.nodes
    assert (join.rows, join.est_rows, join.self_time_s) == (
        100,
        1000,
        0.03,
    )  # "1,000" has a comma
    assert (big.rows_scanned, big.table, big.filters) == (
        9000,
        "invoice_lines",
        "price>1",
    )
    assert join.conditions == "stock_code = stock_code" and join.join_type == "INNER"
    assert small.est_rows is None  # DuckDB writes 0 where it made no estimate
    assert (plan.latency_s, plan.cpu_s, plan.rows_returned) == (0.05, 0.1, 100)
    assert plan.parallelism == pytest.approx(2.0)
    assert plan.rows_scanned == 9040


def test_shares_add_up_to_one():
    plan = sample_plan()
    assert sum(n.share for n in plan.nodes) == pytest.approx(1.0)
    assert plan.nodes[1].share == pytest.approx(0.06 / 0.10)


def test_q_error_is_symmetric_and_one_when_exact():
    a, b = sample_plan().nodes[:2]
    assert a.q_error == pytest.approx(1001 / 101)  # est 1000, got 100
    assert parse_profile(wrap(scan("t", 50, est=50))).root.q_error == 1.0
    over = parse_profile(wrap(scan("t", 1000, est=10))).root
    under = parse_profile(wrap(scan("t", 10, est=1000))).root
    assert over.q_error == pytest.approx(under.q_error)
    assert b.q_error == pytest.approx(9001 / 5001)


def test_signature_ignores_numbers_but_not_structure():
    a = parse_profile(wrap(op("TOP_N", rows=1, children=[scan("t", 5)])))
    b = parse_profile(wrap(op("TOP_N", rows=99, time=1.0, children=[scan("t", 500)])))
    c = parse_profile(wrap(op("TOP_N", rows=1, children=[scan("other", 5)])))
    assert a.signature() == b.signature() != c.signature()


def test_an_empty_profile_is_an_error():
    with pytest.raises(ValueError, match="no operators"):
        parse_profile({"children": []})


def test_dynamic_filter_scans_have_no_comparable_estimate():
    dyn = parse_profile(
        wrap(scan("t", 10, est=1_000_000, Dynamic_Filters="optional: a>=1"))
    ).root
    static = parse_profile(wrap(scan("t", 10, est=1_000_000, Filters="a>=1"))).root
    assert not dyn.estimate_is_comparable
    assert static.estimate_is_comparable


def test_profile_of_a_real_query(con):
    plan = profile(con.cursor(), "SELECT count(*) FROM invoice_lines WHERE price > 5")
    scans = [n for n in plan.nodes if n.name == "TABLE_SCAN"]
    assert [s.table for s in scans] == ["invoice_lines"]
    expected = con.execute(
        "SELECT count(*) FROM invoice_lines WHERE price > 5"
    ).fetchone()[0]
    assert scans[0].rows == expected
    assert plan.rows_returned == 1
    assert plan.latency_s > 0 and plan.sql.startswith("SELECT count")


def test_a_trailing_semicolon_is_fine(con):
    assert profile(con.cursor(), "SELECT 1;").rows_returned == 1


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("DROP TABLE invoices", "only SELECT"),
        ("DELETE FROM invoices", "only SELECT"),
        ("CREATE TABLE x AS SELECT 1", "only SELECT"),
        ("SELECT 1; SELECT 2", "exactly one statement"),
        ("SELEC 1", "cannot parse"),
        ("EXPLAIN SELECT 1", "only SELECT"),
    ],
)
def test_only_a_single_select_is_accepted(con, sql, message):
    with pytest.raises(ValueError, match=message):
        profile(con.cursor(), sql)


def test_a_rejected_statement_changes_nothing(con):
    before = con.execute("SELECT count(*) FROM invoices").fetchone()
    with pytest.raises(ValueError):
        profile(con.cursor(), "DROP TABLE invoices")
    assert con.execute("SELECT count(*) FROM invoices").fetchone() == before


def test_queries_cannot_read_files(con, home):
    with pytest.raises(duckdb.Error, match="disabled"):
        profile(
            con.cursor(),
            f"SELECT * FROM read_parquet('{(home / 'sample_data' / 'products.parquet').as_posix()}')",
        )


def test_check_read_only_returns_the_query(con):
    assert (
        db.check_read_only(con, "WITH a AS (SELECT 1 AS x) SELECT * FROM a")
        == "WITH a AS (SELECT 1 AS x) SELECT * FROM a"
    )


def test_a_runaway_query_is_stopped_and_the_connection_survives(con):
    cur = con.cursor()
    slow = "SELECT sum(a.price * b.price) FROM invoice_lines a, invoice_lines b"
    with pytest.raises(TimeoutError, match="stopped after"):
        profile(cur, slow, timeout_s=0.3)
    assert cur.execute("SELECT 42").fetchone() == (42,)


def test_run_with_timeout_returns_the_result_of_a_quick_call(con):
    assert run_with_timeout(con.cursor(), 5, lambda: 7) == 7


def test_missing_data_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data"):
        db.data_dir(tmp_path)
