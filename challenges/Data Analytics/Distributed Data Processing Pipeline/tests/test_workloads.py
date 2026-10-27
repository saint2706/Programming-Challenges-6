import pytest
from dist_pipeline.workloads import QUERIES, get, render_sql


def test_nine_queries_with_unique_names():
    names = [q.name for q in QUERIES]
    assert len(names) == 9
    assert len(set(names)) == 9


def test_every_query_has_keys_and_placeholders_filled():
    for q in QUERIES:
        assert isinstance(q.keys, tuple), q.name  # scalar queries have no keys
        sql = render_sql(q, trips="t", zones="z", distinct="COUNT(DISTINCT x)")
        assert "{" not in sql, q.name


def test_unknown_query_raises():
    with pytest.raises(KeyError):
        get("q99_missing")
