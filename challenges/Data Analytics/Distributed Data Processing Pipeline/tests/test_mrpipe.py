"""The own map-shuffle-reduce pipeline: partitioning must not change any answer."""

import pytest
from dist_pipeline.engines.base import dataset
from dist_pipeline.mrpipe import MrPipeline
from dist_pipeline.verify import compare, duckdb_oracle
from dist_pipeline.workloads import QUERIES
from helpers import FIXTURE_SCALE


@pytest.mark.parametrize(
    "query", [q for q in QUERIES if q.mergeable], ids=lambda q: q.name
)
def test_one_worker_and_three_workers_agree(mini_home, query):
    ds = dataset(FIXTURE_SCALE)
    zones = ds.zones_frame() if query.needs_zones else None
    one = MrPipeline(workers=1)
    try:
        a = one.run(query.name, ds.parts, zones)
    finally:
        one.close()
    three = MrPipeline(workers=3)
    try:
        b = three.run(query.name, ds.parts, zones)
    finally:
        three.close()
    assert compare(a, b, query).ok


@pytest.mark.parametrize(
    "query", [q for q in QUERIES if q.mergeable], ids=lambda q: q.name
)
def test_matches_exact_oracle(mini_home, query):
    ds = dataset(FIXTURE_SCALE)
    zones = ds.zones_frame() if query.needs_zones else None
    pipe = MrPipeline(workers=2)
    try:
        out = pipe.run(query.name, ds.parts, zones)
    finally:
        pipe.close()
    verdict = compare(out, duckdb_oracle(query, ds), query)
    assert verdict.ok, verdict.detail
