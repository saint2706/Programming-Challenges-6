import pytest
from sql_profiler import compare, lab
from sql_profiler.plan import profile


def check(con, a, b):
    return compare.same_results(con.cursor(), a, b)


# --- result equivalence ------------------------------------------------------------------------


def test_identical_rows_in_a_different_order_are_the_same(con):
    r = check(
        con,
        "SELECT stock_code FROM products ORDER BY 1",
        "SELECT stock_code FROM products ORDER BY 1 DESC",
    )
    assert r.same and r.rows_a == r.rows_b


def test_a_missing_row_is_reported_with_which_side_has_it(con):
    r = check(con, "SELECT * FROM range(10)", "SELECT * FROM range(9)")
    assert not r.same
    assert "1 rows only in A" in r.reason and "0 rows only in B" in r.reason


def test_duplicate_counts_matter(con):
    r = check(con, "SELECT 1 AS x UNION ALL SELECT 1", "SELECT 1 AS x")
    assert not r.same and (r.rows_a, r.rows_b) == (2, 1)


def test_nulls_compare_equal_to_nulls(con):
    assert check(con, "SELECT NULL::INT AS x", "SELECT NULL::INT AS x").same


def test_different_columns_are_not_the_same(con):
    r = check(con, "SELECT 1 AS a, 2 AS b", "SELECT 1 AS a")
    assert not r.same and "columns differ" in r.reason


def test_the_null_safe_join_rewrite_is_not_equivalent_to_the_correlated_subquery(con):
    """The pitfall the correlated-vs-join experiment warns about, shown on real data."""
    case = next(c for c in lab.LAB if c.id == "correlated-vs-join")
    assert check(con, case.a_sql, case.b_sql).same
    null_safe = case.b_sql.replace(
        "ON m.customer_id = i.customer_id",
        "ON m.customer_id IS NOT DISTINCT FROM i.customer_id",
    )
    assert null_safe != case.b_sql
    assert not check(con, case.a_sql, null_safe).same


def test_both_queries_must_be_single_selects(con):
    with pytest.raises(ValueError, match="only SELECT"):
        check(con, "SELECT 1", "DROP TABLE products")


# --- verdicts ----------------------------------------------------------------------------------


def fake(a, b, same=True):
    plan = None
    return compare.Comparison(
        compare.Timing(tuple(a)),
        compare.Timing(tuple(b)),
        compare.ResultCheck(same, 1, 1, ""),
        plan,
        plan,
    )


def test_a_clean_win_needs_every_run_to_beat_every_run():
    assert fake([10, 11, 12], [1, 2, 3]).verdict == "b_faster"
    assert fake([1, 2, 3], [10, 11, 12]).verdict == "a_faster"


def test_overlapping_runs_are_no_clear_difference_even_with_a_big_median_ratio():
    result = fake([5, 6, 40], [2, 3, 7])  # medians 6 vs 3, but the ranges overlap
    assert result.speedup == pytest.approx(2.0)
    assert result.verdict == "no_clear_difference"


def test_timing_statistics():
    t = compare.Timing((3.0, 1.0, 2.0))
    assert (t.median_s, t.best_s, t.worst_s) == (2.0, 1.0, 3.0)


# --- real comparisons --------------------------------------------------------------------------


def test_compare_runs_both_queries_and_reads_fewer_rows_for_the_sargable_form(con):
    case = next(c for c in lab.LAB if c.id == "function-on-key")
    result = compare.compare(con.cursor(), case.a_sql, case.b_sql, repeats=3)
    assert result.check.same
    assert len(result.a.runs_s) == len(result.b.runs_s) == 3
    assert (
        result.plan_b.rows_scanned <= result.plan_a.rows_scanned
    )  # strictly fewer on the full table
    assert (
        result.same_plan
    )  # same operators on the same tables; the difference is how much is read


def test_plan_diff_lists_operator_count_changes(con):
    case = next(c for c in lab.LAB if c.id == "sort-vs-top-n")
    diff = compare.plan_diff(
        profile(con.cursor(), case.a_sql), profile(con.cursor(), case.b_sql)
    )
    assert "TOP_N 0 -> 1" in diff


def test_time_query_validates_sql_before_running(con):
    with pytest.raises(ValueError, match="only SELECT"):
        compare.time_query(con.cursor(), "DELETE FROM products")


# --- the experiments ---------------------------------------------------------------------------


def test_experiment_ids_are_unique_and_every_query_is_a_single_select(con):
    ids = [c.id for c in lab.LAB]
    assert len(ids) == len(set(ids))
    for case in lab.LAB:
        for sql in (case.a_sql, case.b_sql):
            profile(con.cursor(), sql)  # raises if it is not one runnable SELECT


@pytest.mark.parametrize("case", lab.LAB, ids=lambda c: c.id)
def test_each_rewrite_returns_the_same_rows_unless_it_says_otherwise(con, case):
    same = check(con, case.a_sql, case.b_sql).same
    assert same == case.equivalent, case.note or case.idea


def test_the_one_non_equivalent_case_is_the_limit_case():
    assert [c.id for c in lab.LAB if not c.equivalent] == ["sort-vs-top-n"]


def test_run_case_returns_a_comparison(con):
    case = lab.LAB[0]
    assert isinstance(lab.run_case(con.cursor(), case, repeats=3), compare.Comparison)


def test_index_lab_reports_each_lookup_with_the_scan_the_planner_chose(home, con):
    results = lab.index_lab(home, repeats=3)
    assert [r.label.split()[0] for r in results] == [
        "invoice",
        "stock_code",
        "stock_code",
        "stock_code",
    ]
    for r in results:
        expected = con.execute(
            f"SELECT count(*) FROM invoice_lines WHERE {r.label}"
        ).fetchone()[0]
        assert r.rows == expected  # the index returns what the table scan does
        assert r.scanned_without >= r.rows
        assert r.scan_type_with in {"Index Scan", "Sequential Scan"}
        assert len(r.with_index.runs_s) == 3
