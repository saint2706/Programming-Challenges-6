import polars as pl
from dist_pipeline.verify import compare
from dist_pipeline.workloads import get


def _table(query_name: str) -> pl.DataFrame:
    """A small hand-built result for the query's own column layout."""
    q = get(query_name)
    cols: dict[str, list] = {k: [1, 2] for k in q.keys}
    for c in q.exact:
        cols.setdefault(c, [10, 20])
    for c in q.approx:
        cols[c] = [100, 200]
    return pl.DataFrame(cols)


def test_identical_results_pass():
    df = _table("q1_filter_count")
    assert compare(df, df, get("q1_filter_count")).ok


def test_changed_exact_value_fails():
    q = get("q1_filter_count")
    df = _table("q1_filter_count")
    bad = df.with_columns(pl.col(q.exact[0]) + 1)
    assert not compare(bad, df, q).ok


def test_float_within_relative_tolerance_passes():
    q = get("q0_scan_sum")
    exp = pl.DataFrame(
        {k: [1, 2] for k in q.keys} | {c: [1.0e6, 2.0e6] for c in q.exact}
    )
    near = exp.with_columns(pl.col(c) * (1 + 1e-12) for c in q.exact)
    assert compare(near, exp, q).ok


def test_approx_column_within_slack_passes_and_beyond_fails():
    q = get("q4_moments_distinct")
    col = next(iter(q.approx))
    exp = _table("q4_moments_distinct")
    within = exp.with_columns(
        pl.col(col) + 2
    )  # 2 distinct values: inside the 5.0 slack
    assert compare(within, exp, q).ok
    beyond = exp.with_columns(pl.col(col) + 500)
    assert not compare(beyond, exp, q).ok
