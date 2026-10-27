"""Checks on the full downloaded March 2019 data. Skipped when ``data/`` has not been built."""

import pytest
from dist_pipeline.engines.base import dataset, has_data
from dist_pipeline.engines.duckdb_engine import EXACT_DISTINCT, connect, run_sql
from dist_pipeline.verify import compare, polars_oracle
from dist_pipeline.workloads import get


@pytest.fixture(scope="module")
def ds_1m():
    ds = dataset("1m")
    if not has_data(ds):
        pytest.skip("run `uv run dist-pipeline download` and `prepare` first")
    return ds


@pytest.mark.parametrize("name", ["q0_scan_sum", "q3_high_card_od", "q8_topk_window"])
def test_duckdb_exact_matches_polars_oracle_on_full_march(ds_1m, name):
    q = get(name)
    con = connect(ds_1m)
    try:
        actual = run_sql(con, q, EXACT_DISTINCT)
    finally:
        con.close()
    verdict = compare(actual, polars_oracle(q, ds_1m), q)
    assert verdict.ok, verdict.detail
