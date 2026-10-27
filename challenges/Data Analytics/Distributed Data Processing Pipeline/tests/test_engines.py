"""Every installed engine must agree with the exact DuckDB oracle on the committed fixture.

Engines that are not installed are skipped (the core install has only Polars and DuckDB). The
query loop runs inside one test per engine so each engine starts once.
"""

import pytest
from dist_pipeline.engines import ENGINES, load
from dist_pipeline.engines.base import dataset
from dist_pipeline.verify import compare, duckdb_oracle
from dist_pipeline.workloads import QUERIES
from helpers import FIXTURE_SCALE, installed, spark_blocked


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_engine_matches_oracle(mini_home, engine):
    if not installed(engine):
        pytest.skip(f"{engine} needs the 'engines' extra")
    if engine == "spark" and (why := spark_blocked()):
        pytest.skip(why)
    spec = ENGINES[engine]
    if spec.formats and "parquet" not in spec.formats:
        pytest.skip("no parquet support")
    module = load(spec)
    ds = dataset(FIXTURE_SCALE)
    session = module.start(ds, dict(spec.opts))
    failures = []
    try:
        for q in QUERIES:
            try:
                out = module.run(session, q, ds, dict(spec.opts))
            except NotImplementedError:
                continue  # documented: mrpipe cannot merge the median
            verdict = compare(out, duckdb_oracle(q, ds), q)
            if not verdict.ok:
                failures.append(f"{q.name}: {verdict.detail}")
    finally:
        module.stop(session)
    assert not failures, "\n".join(failures)
