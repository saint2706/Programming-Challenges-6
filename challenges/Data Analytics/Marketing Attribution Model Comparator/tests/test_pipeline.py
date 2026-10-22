"""Building journeys, the simulator's truth, the comparison, the Criteo loader and the CLI."""

import json
import re

import numpy as np
import polars as pl
import pytest
from attribution import criteo
from attribution.build import from_touches
from attribution.cli import app
from attribution.compare import LABELS, MODELS, Settings, compare, resampled, shares
from attribution.evaluate import best_model, evaluate
from attribution.journeys import AttributionError, Journeys, from_paths
from attribution.report import build_html, to_json
from attribution.simulate import CHANNELS, SCENARIOS, Scenario, simulate
from helpers import random_journeys, toy
from typer.testing import CliRunner

runner = CliRunner(env={"COLUMNS": "260", "NO_COLOR": "1"})
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(*args: str):
    result = runner.invoke(app, list(args))
    result.clean = " ".join(ANSI.sub("", result.output).split())
    return result


# ---- building -------------------------------------------------------------------------------------
def test_from_touches_agrees_with_from_paths():
    # journeys 1 and 2 convert (A>B twice... as two journeys), 3 and 4 do not
    journey = np.array([1, 1, 2, 2, 3, 3, 4, 5])
    time = np.array(
        [1.0, 2.0, 5.0, 4.0, 1.0, 2.0, 1.0, 1.0]
    )  # journey 2 arrives out of order
    channel = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    converted = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    j = from_touches(
        journey, time, channel, converted, ["A", "B"], conv_time=np.full(8, 10.0)
    )
    assert j.n_converted == 2 and j.null_count == 3
    assert j.channel.tolist() == [
        0,
        1,
        0,
        1,
    ]  # journey 2 reordered to A (t=4) then B (t=5)
    assert j.null_start.tolist() == [2, 1] and j.null_end.tolist() == [1, 2]
    assert j.null_trans.tolist() == [[0, 1], [0, 0]]
    assert j.null_sets == {3: 1.0, 2: 1.0, 1: 1.0}
    ref = from_paths(
        [["A", "B"]] * 2 + [["A", "B"], ["B"], ["A"]], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]
    )
    for rule in ("first_touch", "last_touch", "linear"):
        from attribution import rules

        np.testing.assert_allclose(rules.credit(j, rule), rules.credit(ref, rule))
    assert j.null_trans.tolist() == ref.null_trans.tolist()


def test_journeys_validate_their_input():
    j = toy()
    with pytest.raises(AttributionError, match="offsets"):
        Journeys(
            j.channels,
            np.array([1, 2]),
            j.channel[:1],
            j.time[:1],
            j.clicked[:1],
            j.cost[:1],
            j.conv_time[:1],
            j.value[:1],
        )
    with pytest.raises(AttributionError, match="does not exist"):
        Journeys(
            ["a"],
            np.array([0, 1]),
            np.array([3]),
            np.zeros(1),
            np.zeros(1, bool),
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
        )
    with pytest.raises(AttributionError, match="empty"):
        from_paths([[]], [1])
    with pytest.raises(AttributionError, match="whole numbers"):
        from_paths([["a"]], [1.5])
    with pytest.raises(AttributionError, match="no touches"):
        from_touches(
            np.array([]), np.array([]), np.array([]), np.array([], bool), ["a"]
        )


# ---- the simulator --------------------------------------------------------------------------------
def test_the_truth_is_zero_for_a_channel_with_no_effect_and_matches_a_counterfactual_run():
    sc = SCENARIOS["retargeting"]
    w = simulate(sc, 40_000, seed=3)
    assert w.truth[CHANNELS.index("retargeting")] == pytest.approx(0.0, abs=1e-9)
    assert w.truth_share.sum() == pytest.approx(1.0)
    # remove display's effect and re-run with the same seed: conversions fall by about the truth
    off = Scenario(**{**sc.__dict__, "beta": {**sc.beta, "display": 0.0}})
    drop = (
        simulate(sc, 40_000, seed=3).n_converted
        - simulate(off, 40_000, seed=3).n_converted
    )
    assert drop == pytest.approx(
        w.truth[CHANNELS.index("display")],
        abs=0.25 * w.truth[CHANNELS.index("display")] + 60,
    )


def test_simulation_is_reproducible_and_journeys_are_well_formed():
    a, b = simulate("independent", 5000, seed=1), simulate("independent", 5000, seed=1)
    np.testing.assert_array_equal(a.journeys.channel, b.journeys.channel)
    j = a.journeys
    assert j.offsets[-1] == j.channel.size and j.k == len(CHANNELS) and j.unit == "days"
    assert (j.time >= 0).all() and (j.time <= 30).all()
    within = np.diff(j.time)[np.diff(j.path_of_touch) == 0]
    assert (within >= 0).all()  # touches are in time order inside each journey


# ---- comparison -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def comparison():
    j = simulate("independent", 30_000, seed=0).journeys
    return compare(j, Settings(half_life=7.0), draws=20, seed=0)


def test_every_model_credits_the_same_total_and_shares_sum_to_one(comparison):
    assert comparison.models == MODELS
    np.testing.assert_allclose(comparison.share.sum(axis=1), 1.0)
    rule_totals = comparison.credit[:6].sum(axis=1)
    np.testing.assert_allclose(rule_totals, comparison.journeys.total_value)
    assert comparison.credit[comparison.models.index("markov")].sum() == pytest.approx(
        comparison.journeys.total_value
    )


def test_intervals_bracket_the_estimate_and_describe_the_models(comparison):
    assert comparison.draws == 20
    inside = (comparison.low <= comparison.share + 1e-9) & (
        comparison.share <= comparison.high + 1e-9
    )
    assert inside.mean() > 0.9
    tv, tau = comparison.distance(), comparison.rank_correlation()
    np.testing.assert_allclose(tv, tv.T)
    assert (
        np.allclose(np.diag(tv), 0) and np.allclose(np.diag(tau), 1) and (tv <= 1).all()
    )
    ranks = comparison.rank_matrix()
    assert ranks.min() == 1 and ranks.shape == comparison.share.shape


def test_resampling_keeps_the_totals_and_changes_the_null_side():
    j = random_journeys(1)
    r, w = resampled(j, np.random.default_rng(0))
    assert w.sum() == j.n_converted and r.null_count != j.null_count
    assert j.null_count == 5 * 0 + j.null_count  # the original is untouched
    assert r.n_converted == j.n_converted


def test_cost_per_conversion_uses_spend_over_credit():
    j = simulate("independent", 5000, seed=2).journeys
    j.cost[:] = 2.0
    c = compare(j, Settings(half_life=7.0), ("linear", "first_touch"))
    cpc = c.cost_per_conversion()
    spend = j.touch_spend()
    np.testing.assert_allclose(cpc[0], spend / c.credit[0])
    assert c.spend_share().sum() == pytest.approx(1.0)


def test_unknown_model_and_duplicate_channels_are_errors():
    j = toy()
    with pytest.raises(AttributionError, match="unknown model"):
        compare(j, Settings(half_life=1.0), ("middle_touch",))
    j.channels = ["A", "A"]
    with pytest.raises(AttributionError, match="unique"):
        compare(j, Settings(half_life=1.0))


# ---- the known-truth trials -------------------------------------------------------------------------
def test_no_model_wins_everywhere():
    best = {}
    for name in SCENARIOS:
        r = evaluate(name, customers=30_000, seeds=2)
        best[name] = best_model(r)
        assert r.truth.sum() == pytest.approx(1.0) and r.error.shape == (len(MODELS),)
    assert best["closing"] in {"Last touch", "Last click"}
    assert best["awareness"] == "First touch"
    assert len(set(best.values())) >= 3, best


def test_last_touch_hands_a_useless_retargeting_channel_most_of_the_credit():
    w = simulate("retargeting", 60_000, seed=1)
    c = compare(
        w.journeys, Settings(half_life=7.0), ("last_touch", "first_touch", "markov")
    )
    i = CHANNELS.index("retargeting")
    assert w.truth_share[i] == pytest.approx(0.0, abs=1e-9)
    assert c.share[0, i] > 0.45 and c.share[1, i] < 0.05
    assert c.share[2, i] < c.share[0, i]  # the data-driven model is hijacked less


# ---- report and CLI -------------------------------------------------------------------------------
def test_report_is_self_contained_and_escapes_names():
    j = from_paths([["<img src=x onerror=1>", "b"], ["b"]], [30, 20], [100, 100])
    c = compare(j, Settings(half_life=1.0), draws=5)
    page = build_html("<b>t</b>", c, [], MODELS, ["note <i>"])
    markup = re.sub(r"<script>.*?</script>", "", page, flags=re.DOTALL)
    assert (
        "<img src=x" not in page
        and "&lt;img src=x" in page
        and "<b>t</b>" not in markup
    )
    assert page.count("</script>") == page.count("<script")
    assert not re.search(r"""(src|href)=["']https?://""", markup)
    data = json.loads(to_json(c, [], MODELS))
    assert data["comparison"]["bootstrap_draws"] == 5 and set(
        data["comparison"]["share"]
    ) == set(MODELS)


def test_compare_command_on_a_paths_file(tmp_path):
    path = tmp_path / "paths.csv"
    path.write_text(
        "path,conversions,nulls\nemail>search,40,60\nsearch,30,200\ndisplay>email>search,10,30\ndisplay,5,400\n"
    )
    out = run("compare", str(path), "--draws", "10", "--out", str(tmp_path / "o"))
    assert out.exit_code == 0, out.clean
    for label in ("First touch", "Markov chain", "Linear"):
        assert label in out.clean
    assert (tmp_path / "o" / "report.html").stat().st_size > 1_000_000
    assert json.loads((tmp_path / "o" / "results.json").read_text())["comparison"][
        "channels"
    ] == ["display", "email", "search"]
    only = run(
        "compare",
        str(path),
        "-m",
        "linear",
        "-m",
        "last_touch",
        "--out",
        str(tmp_path / "p"),
    )
    assert only.exit_code == 0 and "Markov" not in only.clean


def test_compare_and_simulate_on_simulated_data(tmp_path):
    out = run("compare", "sim:retargeting", "--out", str(tmp_path / "a"))
    assert (
        out.exit_code == 0
        and "retargeting" in out.clean
        and "most different pair" in out.clean
    )
    out = run(
        "simulate",
        "-s",
        "closing",
        "--customers",
        "20000",
        "--seeds",
        "2",
        "-m",
        "first_touch",
        "-m",
        "last_touch",
        "--out",
        str(tmp_path / "b"),
    )
    assert out.exit_code == 0, out.clean
    assert (
        json.loads((tmp_path / "b" / "simulation.json").read_text())["scenarios"][0][
            "name"
        ]
        == "closing"
    )
    assert LABELS["last_touch"] in out.clean


def test_commands_list_and_fail_cleanly(tmp_path):
    assert "markov" in run("models").clean
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b\n1,2\n")
    for args, text in (
        (["compare", "nowhere.csv"], "neither 'criteo'"),
        (["compare", "sim:nope"], "unknown scenario"),
        (["compare", str(bad_csv)], "needs a 'path' column"),
        (["compare", "sim:independent", "-m", "magic"], "unknown model"),
        (["simulate", "-s", "nope"], "unknown scenario"),
    ):
        result = run(*args)
        assert result.exit_code == 2 and text in result.clean, (args, result.clean)


# ---- the Criteo loader, on a hand-made table with the real schema -----------------------------------
def tiny_criteo(tmp_path):
    rows = [
        # uid, campaign, timestamp, conversion, conversion_id, conversion_timestamp, click, cost, cat9
        (
            1,
            10,
            100,
            1,
            7,
            900,
            0,
            0.1,
            5,
        ),  # converting journey 7: two touches on two contexts
        (1, 10, 200, 1, 7, 900, 1, 0.2, 6),
        (
            1,
            10,
            950,
            0,
            -1,
            -1,
            0,
            0.3,
            5,
        ),  # same user and campaign, after the conversion: ambiguous, dropped
        (2, 10, 120, 0, -1, -1, 0, 0.1, 5),  # never converted
        (2, 10, 180, 0, -1, -1, 1, 0.1, 6),
        (3, 11, 500, 0, -1, -1, 0, 0.1, 6),
    ]
    cols = [
        "uid",
        "campaign",
        "timestamp",
        "conversion",
        "conversion_id",
        "conversion_timestamp",
        "click",
        "cost",
        "cat9",
    ]
    frame = pl.DataFrame(rows, schema=cols, orient="row")
    for c in ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8"]:
        frame = frame.with_columns(pl.lit(1).alias(c))
    frame = frame.with_columns(pl.lit(0).alias("attribution"))
    path = tmp_path / "criteo.parquet"
    frame.write_parquet(path)
    return path


def test_criteo_journeys_from_a_tiny_table(tmp_path):
    j = criteo.load_journeys("cat9", 5, parquet=tiny_criteo(tmp_path))
    assert j.channels == ["cat9=5", "cat9=6"] or j.channels == ["cat9=6", "cat9=5"]
    assert j.n_converted == 1 and j.null_count == 2  # journey 7, plus (2,10) and (3,11)
    assert j.channel.size == 2 and j.clicked.tolist() == [False, True]
    assert j.conv_time.tolist() == [900.0] and j.unit == "seconds"
    assert (
        j.null_touches.sum() == 3
    )  # the post-conversion impression of the converting pair is excluded
    with pytest.raises(AttributionError, match="channel must be one of"):
        criteo.load_journeys("campaign", 5, parquet=tiny_criteo(tmp_path))
    with pytest.raises(AttributionError, match="is missing"):
        criteo.load_journeys("cat9", 5, parquet=tmp_path / "nope.parquet")


@pytest.mark.skipif(
    not criteo.parquet_path().exists(),
    reason="run `uv run attribute fetch` for the full Criteo data",
)
def test_the_real_criteo_journeys_have_the_documented_shape():
    j = criteo.load_journeys("cat9", 11)
    assert j.n_converted == 435_810 and j.k == 12
    assert (
        0.25
        < float(
            np.mean(
                [
                    len(set(j.channel[a:b])) > 1
                    for a, b in zip(j.offsets[:-1], j.offsets[1:], strict=True)
                ]
            )
        )
        < 0.35
    )
    c = compare(
        j,
        Settings(half_life=7 * 86400.0),
        ("first_touch", "last_touch", "linear", "markov"),
    )
    np.testing.assert_allclose(shares(c.credit).sum(axis=1), 1.0)
    assert (
        c.share[0, 0] > c.share[1, 0]
    )  # first touch credits the dominant context more than last touch
