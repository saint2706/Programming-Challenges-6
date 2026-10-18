"""The Rust engine on the real Adult census, checked against polars and DuckDB (independent code)."""

import math

import duckdb
import polars as pl
import pytest
from conftest import ADULT, cells

AGGS = {
    "sum": lambda c: pl.col(c).sum(),
    "mean": lambda c: pl.col(c).mean(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "median": lambda c: pl.col(c).median(),
    "std": lambda c: pl.col(c).std(),
    "var": lambda c: pl.col(c).var(),
    "count": lambda c: pl.col(c).count(),
    "distinct": lambda c: pl.col(c).n_unique(),
}


def close(a, b):
    """``a`` from the engine (None = empty cell), ``b`` from polars (None or NaN = nothing)."""
    a_missing = a is None
    b_missing = b is None or (isinstance(b, float) and math.isnan(b))
    assert a_missing == b_missing, (a, b)
    if not a_missing:
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


@pytest.mark.parametrize("agg", list(AGGS))
@pytest.mark.parametrize("field", ["hours_per_week", "age", "capital_gain"])
def test_every_leaf_and_total_cell_matches_polars(adult, adult_frame, agg, field):
    result = adult.pivot(
        {
            "rows": [{"field": "education"}],
            "cols": [{"field": "sex"}],
            "values": [{"field": field, "agg": agg}],
        }
    )
    got = cells(result)
    expr = AGGS[agg](field).alias("v")
    for (edu, sex), df in adult_frame.group_by("education", "sex"):
        close(got[((1, (edu,)), (1, (sex,)))], df.select(expr).item())
    for (edu,), df in adult_frame.group_by("education"):
        close(got[((1, (edu,)), (0, ()))], df.select(expr).item())
    for (sex,), df in adult_frame.group_by("sex"):
        close(got[((0, ()), (1, (sex,)))], df.select(expr).item())
    close(got[((0, ()), (0, ()))], adult_frame.select(expr).item())


def test_subtotals_match_polars_at_every_level(adult, adult_frame):
    result = adult.pivot(
        {
            "rows": [{"field": "workclass"}, {"field": "marital_status"}],
            "cols": [{"field": "income"}],
            "values": [{"field": "hours_per_week", "agg": "mean"}, {"agg": "count"}],
            "options": {"subtotals": True},
        }
    )
    mean, count = cells(result, 0), cells(result, 1)
    for (wc,), df in adult_frame.group_by("workclass"):
        close(mean[((1, (wc,)), (0, ()))], df["hours_per_week"].mean())
        close(count[((1, (wc,)), (0, ()))], df.height)
    for (wc, ms, inc), df in adult_frame.group_by(
        "workclass", "marital_status", "income"
    ):
        close(mean[((2, (wc, ms)), (1, (inc,)))], df["hours_per_week"].mean())
        close(count[((2, (wc, ms)), (1, (inc,)))], df.height)
    # a subtotal's mean is the mean of all its rows, not the mean of its group means
    wc = "Private"
    group_means = (
        adult_frame.filter(pl.col("workclass") == wc)
        .group_by("marital_status")
        .agg(pl.col("hours_per_week").mean())["hours_per_week"]
    )
    assert not math.isclose(
        mean[((1, (wc,)), (0, ()))], group_means.mean(), rel_tol=1e-6
    )


def test_grouping_by_bins_dates_and_quantiles_matches_polars(adult, adult_frame):
    result = adult.pivot(
        {
            "rows": [{"field": "age", "group": {"kind": "bin", "width": 10}}],
            "values": [{"agg": "count"}],
        }
    )
    got = cells(result)
    binned = adult_frame.with_columns(((pl.col("age") // 10) * 10).alias("b"))
    for (b,), df in binned.group_by("b"):
        assert got[((1, (f"[{b}, {b + 10})",)), (0, ()))] == df.height
    result = adult.pivot(
        {
            "rows": [{"field": "age", "group": {"kind": "quantile", "n": 4}}],
            "values": [{"agg": "count"}],
        }
    )
    sizes = [row[0][0] for row in result["cells"][:-1]]
    assert (
        sum(sizes) == adult_frame.height and max(sizes) / min(sizes) < 1.6
    )  # near-equal bins, ties aside


def test_filters_match_polars(adult, adult_frame):
    result = adult.pivot(
        {
            "rows": [{"field": "occupation"}],
            "values": [{"agg": "count"}],
            "filters": [
                {"field": "age", "op": "between", "values": [30, 40]},
                {"field": "sex", "op": "eq", "value": "Female"},
                {"field": "income", "op": "in", "values": [">50K"]},
                {"field": "native_country", "op": "ne", "value": "United-States"},
            ],
        }
    )
    sub = adult_frame.filter(
        pl.col("age").is_between(30, 40)
        & (pl.col("sex") == "Female")
        & (pl.col("income") == ">50K")
        & (pl.col("native_country") != "United-States")
    )
    assert result["rows_used"] == sub.height
    got = cells(result)
    for (occ,), df in sub.group_by("occupation"):
        assert got[((1, (occ,)), (0, ()))] == df.height


def test_percent_of_row_total_matches_polars(adult, adult_frame):
    result = adult.pivot(
        {
            "rows": [{"field": "education"}],
            "cols": [{"field": "income"}],
            "values": [{"agg": "count", "show_as": {"kind": "pct_row"}}],
        }
    )
    got = cells(result)
    for (edu, inc), df in adult_frame.group_by("education", "income"):
        total = adult_frame.filter(pl.col("education") == edu).height
        close(got[((1, (edu,)), (1, (inc,)))], df.height / total)


def test_top_n_keeps_the_largest_groups_and_sorts_by_value(adult, adult_frame):
    result = adult.pivot(
        {
            "rows": [
                {
                    "field": "occupation",
                    "top": {"n": 3, "measure": "count"},
                    "sort": {"by": "value", "measure": "count"},
                }
            ],
            "values": [{"agg": "count"}],
            "options": {"totals_row": False, "totals_col": False},
        }
    )
    # "?" is an occupation like any other here: it takes part in the ranking
    expected = (
        adult_frame.group_by("occupation").len().sort("len", descending=True).head(3)
    )
    assert [h["labels"][0] for h in result["rows"]] == expected["occupation"].to_list()
    assert [row[0][0] for row in result["cells"]] == expected["len"].to_list()


def test_matches_duckdb_rollup_for_grand_and_group_totals(adult):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM read_csv(?)", [str(_plain_csv())])
    rows = con.execute(
        "SELECT race, sex, sum(hours_per_week), count(*), grouping(race), grouping(sex) FROM t GROUP BY CUBE (race, sex)"
    ).fetchall()
    result = adult.pivot(
        {
            "rows": [{"field": "race"}],
            "cols": [{"field": "sex"}],
            "values": [{"field": "hours_per_week", "agg": "sum"}, {"agg": "count"}],
        }
    )
    total, n = cells(result, 0), cells(result, 1)
    for race, sex, s, c, g_race, g_sex in rows:
        key = ((0, ()) if g_race else (1, (race,)), (0, ()) if g_sex else (1, (sex,)))
        assert total[key] == s and n[key] == c


_csv_cache = {}


def _plain_csv():
    import gzip
    import tempfile
    from pathlib import Path

    if "p" not in _csv_cache:
        tmp = Path(tempfile.mkdtemp()) / "adult.csv"
        tmp.write_bytes(gzip.decompress(ADULT.read_bytes()))
        _csv_cache["p"] = tmp
    return _csv_cache["p"]


def test_calculated_field_and_running_total_against_polars(adult, adult_frame):
    result = adult.pivot(
        {
            "rows": [{"field": "marital_status"}],
            "values": [
                {"agg": "count", "label": "people"},
                {"field": "capital_gain", "agg": "sum", "label": "gain"},
                {"calc": "gain / people", "label": "ratio"},
                {
                    "field": "capital_gain",
                    "agg": "sum",
                    "label": "run",
                    "show_as": {"kind": "running", "along": "rows"},
                },
            ],
            "options": {"totals_col": False},
        }
    )
    ratio = cells(result, 2)
    for (ms,), df in adult_frame.group_by("marital_status"):
        close(ratio[((1, (ms,)), (0, ()))], df["capital_gain"].sum() / df.height)
    leaves = [i for i, h in enumerate(result["rows"]) if h["kind"] == "leaf"]
    running = [result["cells"][i][0][3] for i in leaves]
    singles = [result["cells"][i][0][1] for i in leaves]
    assert running == [sum(singles[: k + 1]) for k in range(len(singles))]
