"""The empirical check, on pipelines whose true behaviour is known because the test planted it."""

import pytest
from data_lineage import execute
from data_lineage.graph import label
from data_lineage.verify import verify
from helpers import T_CSV, U_CSV, load, make_pipeline


def check(tmp_path, models, sources=None, **kwargs):
    root = make_pipeline(tmp_path, sources or {"t": T_CSV, "u": U_CSV}, models)
    project, graph = load(root)
    return verify(project, graph, **kwargs)


def pairs(items):
    return {(label(p.source), label(p.target)): p for p in items}


def test_a_clean_pipeline_confirms_everything_and_misses_nothing(tmp_path):
    v = check(
        tmp_path,
        {"m": "SELECT a, a + b AS s, c FROM t WHERE d > DATE '2020-01-02'"},
        only_sources=None,
    )
    assert v.ok and not v.missed and not v.noise and not v.failures
    confirmed = pairs(v.confirmed)
    assert confirmed[("t.a", "m.s")].kind == "direct"
    assert confirmed[("t.b", "m.s")].kind == "direct"
    assert confirmed[("t.d", "m.a")].kind == "indirect"  # the filter moves every column
    assert "t.u" not in {s for s, _ in confirmed}


def test_columns_regex_is_a_blind_spot_the_data_check_finds(tmp_path):
    v = check(tmp_path, {"m": "SELECT a, COLUMNS('^b$') FROM t"})
    assert not v.ok
    assert any(
        "m: static analysis lists ['a', '_col_1'], DuckDB produced ['a', 'b']" in s
        for s in v.schema_mismatches
    )
    assert ("t.b", "m.b") in pairs(
        v.missed
    )  # b really feeds m.b; the lineage has no edge for it


def test_a_branch_the_data_never_takes_is_reported_unexercised_not_wrong(tmp_path):
    v = check(tmp_path, {"m": "SELECT COALESCE(a, b) AS x FROM t"})
    assert ("t.a", "m.x") in pairs(v.confirmed)
    assert ("t.b", "m.x") in pairs(v.unexercised)  # a is never NULL, so b never matters
    assert v.ok


def test_the_fallback_is_confirmed_when_the_data_does_reach_it(tmp_path):
    v = check(
        tmp_path,
        {"m": "SELECT COALESCE(x, y) AS z FROM s"},
        sources={"s": "x,y\n1,10\n,20\n3,30"},
        strategies=("shift",),
        only_sources={("s", "y")},
    )
    assert ("s.y", "m.z") in pairs(v.confirmed)  # row 2 has no x, so y decides it


def test_nondeterministic_columns_are_excluded_as_noise(tmp_path):
    v = check(tmp_path, {"m": "SELECT a, random() AS r FROM t"})
    assert v.noise == [("m", "r")]
    assert all(
        p.target != ("m", "r") for p in (*v.confirmed, *v.unexercised, *v.missed)
    )
    assert v.ok


def test_a_run_that_fails_under_a_change_is_reported_not_raised(tmp_path):
    v = check(
        tmp_path,
        {"m": "SELECT CAST(replace(code, 'a', '') AS INTEGER) AS n FROM s"},
        sources={"s": "code\n1a\n2a\n3a"},
        strategies=("shift", "shuffle"),
    )
    assert any(
        "s.code / shift" in f for f in v.failures
    )  # '1a~' -> '1~' cannot be cast
    assert ("s.code", "m.n") not in pairs(v.missed)


def test_ranks_are_caught_by_row_alignment(tmp_path):
    # a permutation of 1..N has the same multiset whatever the data: only the pairing shows it
    v = check(
        tmp_path,
        {"m": "SELECT k, rank() OVER (ORDER BY v) AS r FROM u"},
        strategies=("shuffle",),
    )
    assert v.keys["m"] == "k"
    assert ("u.v", "m.r") in pairs(v.confirmed)


def test_moving_what_a_row_is_called_is_not_a_dependency_of_its_other_columns(tmp_path):
    # shuffling u.k re-pairs keys with values; v does not depend on k
    v = check(tmp_path, {"m": "SELECT k AS kk, v FROM u"}, strategies=("shuffle",))
    assert ("u.k", "m.v") not in pairs(v.missed)
    assert ("u.k", "m.kk") not in pairs(v.missed)


def test_only_sources_and_strategies_limit_the_runs(tmp_path):
    v = check(
        tmp_path,
        {"m": "SELECT a, b FROM t"},
        only_sources={("t", "a")},
        strategies=("shift", "constant"),
    )
    assert v.runs == 2 + 2
    assert {s for s, _ in pairs(v.confirmed)} == {"t.a"}


def test_progress_callback_sees_every_run(tmp_path):
    seen = []
    check(
        tmp_path,
        {"m": "SELECT a FROM t"},
        only_sources={("t", "a")},
        progress=seen.append,
    )
    assert seen == [f"t.a / {s}" for s in execute.STRATEGIES]


def test_to_json_roundtrips_the_verdicts(tmp_path):
    v = check(tmp_path, {"m": "SELECT COALESCE(a, b) AS x FROM t"})
    data = v.to_json()
    assert data["runs"] == v.runs and data["strategies"] == list(execute.STRATEGIES)
    assert {tuple(p["source"]) for p in data["unexercised"]} == {("t", "b")}
    assert data["row_keys"] == {
        "m": "x"
    }  # a, b are unique ints here: the first is the row key


# -- the real pipeline, run once -----------------------------------------------------------------
@pytest.fixture(scope="module")
def real_verification(real_project, real_graph):
    return verify(real_project, real_graph)


def test_real_pipeline_has_no_missed_dependency_and_no_noise(real_verification):
    v = real_verification
    assert v.ok and v.runs == 2 + 10 * len(execute.STRATEGIES)
    assert (
        v.missed == []
        and v.noise == []
        and v.failures == []
        and v.schema_mismatches == []
    )
    assert len(v.confirmed) >= 200


def test_real_pipeline_has_one_unexercised_pair_and_it_is_semantically_inert(
    real_verification,
):
    # last_order_seq = max(row_number() ...) is the number of orders; ORDER BY invoice_date inside
    # the window changes which order is number 1, never how many there are
    assert [
        (label(p.source), label(p.target)) for p in real_verification.unexercised
    ] == [("raw_transactions.invoice_date", "mart_customers.last_order_seq")]


def test_real_pipeline_value_dependencies_are_confirmed_by_every_numeric_strategy(
    real_verification,
):
    p = pairs(real_verification.confirmed)[
        ("raw_transactions.price", "mart_rfm.monetary")
    ]
    assert p.kind == "direct" and set(p.strategies) == set(execute.STRATEGIES)
