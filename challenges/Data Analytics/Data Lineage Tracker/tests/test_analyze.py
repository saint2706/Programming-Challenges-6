import pytest
from data_lineage.analyze import AnalysisError, analyze_model

SCHEMA = {
    "t": {"a": "int", "b": "int", "c": "varchar", "d": "date"},
    "u": {"k": "int", "v": "int"},
}


def run(sql):
    return analyze_model("m", sql, SCHEMA)


def col(ml, name):
    return next(c for c in ml.columns if c.name == name)


def deps(ml, name):
    return {(d.table, d.column): d.role for d in col(ml, name).deps}


def test_copy_rename_cast():
    ml = run("SELECT a, b AS bb, CAST(a AS VARCHAR) AS a_text FROM t")
    assert deps(ml, "a") == {("t", "a"): "copy"}
    assert deps(ml, "bb") == {("t", "b"): "rename"}
    assert deps(ml, "a_text") == {("t", "a"): "cast"}


def test_expression_with_two_inputs():
    ml = run("SELECT a + b AS s FROM t")
    assert deps(ml, "s") == {("t", "a"): "expression", ("t", "b"): "expression"}
    assert col(ml, "s").expr == "t.a + t.b"


def test_role_ranking_aggregate_beats_expression():
    ml = run("SELECT sum(a * 2) + 1 AS s FROM t")
    assert col(ml, "s").role == "aggregate"


def test_window_function_depends_on_partition_and_order_columns():
    ml = run("SELECT row_number() OVER (PARTITION BY c ORDER BY d) AS rn FROM t")
    assert set(deps(ml, "rn")) == {("t", "c"), ("t", "d")}
    assert col(ml, "rn").role == "window"


def test_conditional_and_coalesce_depend_on_every_branch():
    ml = run(
        "SELECT CASE WHEN a > 1 THEN b ELSE 0 END AS x, COALESCE(a, b) AS y FROM t"
    )
    assert set(deps(ml, "x")) == {("t", "a"), ("t", "b")}
    assert set(deps(ml, "y")) == {("t", "a"), ("t", "b")}
    assert col(ml, "x").role == "conditional" and col(ml, "y").role == "conditional"


def test_filter_clause_columns_are_part_of_the_value():
    ml = run("SELECT sum(a) FILTER (WHERE b > 1) AS s FROM t")
    assert set(deps(ml, "s")) == {("t", "a"), ("t", "b")}


def test_count_star_has_no_column_inputs_but_is_an_aggregate():
    ml = run("SELECT count(*) AS n, 1 AS one FROM t")
    assert col(ml, "n").deps == [] and col(ml, "n").role == "aggregate"
    assert col(ml, "one").deps == [] and col(ml, "one").role == "constant"


def test_star_and_star_exclude_expand_to_the_real_columns():
    assert [c.name for c in run("SELECT * FROM t").columns] == ["a", "b", "c", "d"]
    assert [c.name for c in run("SELECT * EXCLUDE (c) FROM t").columns] == [
        "a",
        "b",
        "d",
    ]


def test_cte_steps_are_recorded_innermost_last():
    ml = run("WITH x AS (SELECT a * 2 AS a2 FROM t) SELECT a2 + 1 AS r FROM x")
    (dep,) = col(ml, "r").deps
    assert (dep.table, dep.column) == ("t", "a")
    assert dep.steps == ("x.a2 + 1", "t.a * 2")


def test_join_traces_each_side():
    ml = run("SELECT t.a, u.v FROM t JOIN u ON t.a = u.k")
    assert deps(ml, "a") == {("t", "a"): "copy"}
    assert deps(ml, "v") == {("u", "v"): "copy"}
    assert ml.indirect == {("t", "a"): ("join",), ("u", "k"): ("join",)}


def test_using_join_depends_on_both_key_columns():
    ml = run("SELECT k FROM u JOIN (SELECT a AS k FROM t) AS s USING (k)")
    assert {("u", "k"), ("t", "a")} <= set(deps(ml, "k"))


def test_union_all_traces_both_branches():
    ml = run("SELECT a AS x FROM t UNION ALL SELECT k FROM u")
    assert set(deps(ml, "x")) == {("t", "a"), ("u", "k")}


def test_indirect_reasons_are_kept_apart():
    ml = run(
        "SELECT c, sum(b) AS s FROM t WHERE a > 1 GROUP BY c HAVING sum(d IS NOT NULL::int) > 0"
    )
    assert ml.indirect[("t", "a")] == ("filter",)
    assert ml.indirect[("t", "c")] == ("group",)
    assert ml.indirect[("t", "d")] == ("having",)


def test_qualify_columns_are_indirect():
    ml = run(
        "SELECT a FROM t QUALIFY row_number() OVER (PARTITION BY c ORDER BY b) = 1"
    )
    assert ml.indirect == {("t", "b"): ("qualify",), ("t", "c"): ("qualify",)}


def test_group_by_ordinal_is_resolved_to_the_column():
    ml = run("SELECT c, count(*) AS n FROM t GROUP BY 1")
    assert ml.indirect == {("t", "c"): ("group",)}


def test_indirect_columns_inside_a_cte_are_resolved_to_the_real_table():
    ml = run(
        "WITH x AS (SELECT a AS aa, c FROM t WHERE c = 'x') SELECT aa FROM x WHERE aa > 2"
    )
    assert ml.indirect == {("t", "a"): ("filter",), ("t", "c"): ("filter",)}


def test_correlated_subquery_key_is_an_indirect_dependency():
    # sqlglot's own lineage() drops t.a here; the scope walk finds it
    ml = run("SELECT (SELECT max(v) FROM u WHERE u.k = t.a) AS m FROM t")
    assert deps(ml, "m") == {("u", "v"): "aggregate"}
    assert ("t", "a") in ml.indirect and ("u", "k") in ml.indirect


def test_physical_tables_exclude_ctes():
    ml = run("WITH x AS (SELECT a FROM t) SELECT x.a, u.v FROM x JOIN u ON x.a = u.k")
    assert ml.tables == ("t", "u")


def test_columns_regex_is_flagged_opaque_not_silently_dropped():
    ml = run("SELECT COLUMNS('^[ab]$') FROM t")
    assert col(ml, "_col_0").opaque and col(ml, "_col_0").deps == []
    assert any("COLUMNS" in w for w in ml.warnings)


def test_table_function_is_flagged():
    ml = analyze_model("m", "SELECT * FROM read_csv('x.csv')", {})
    assert any("table function" in w for w in ml.warnings)


def test_unknown_column_names_the_model():
    with pytest.raises(AnalysisError, match="m:.*nope"):
        run("SELECT nope FROM t")


def test_unparsable_sql_names_the_model():
    with pytest.raises(AnalysisError, match="m: cannot parse"):
        run("SELEC a FRM t")


def test_non_select_is_rejected():
    with pytest.raises(AnalysisError, match="single SELECT"):
        run("DROP TABLE t")


def test_long_expressions_are_truncated_for_display():
    big = " + ".join(["a"] * 120)
    assert len(col(run(f"SELECT {big} AS s FROM t"), "s").expr) <= 200
