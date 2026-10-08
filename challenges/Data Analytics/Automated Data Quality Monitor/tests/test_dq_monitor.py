"""Unit tests for the checks, on tiny hand-built frames (no files, no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from data_quality import dq_monitor as dq

HERE = Path(__file__).parent.parent


def make_frame(n: int = 200, seed: int = 0, start: str = "2024-01-01") -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = datetime.fromisoformat(start)
    return pl.DataFrame(
        {
            "id": [f"AB-{seed:02d}{i:04d}" for i in range(n)],
            "amount": rng.gamma(3.0, 10.0, n).round(2),
            "qty": rng.integers(1, 6, n),
            "kind": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
            "ts": [
                t0 + timedelta(seconds=int(s))
                for s in np.sort(rng.integers(0, 86_400, n))
            ],
        }
    )


@pytest.fixture(scope="module")
def baseline():
    frames = [(f"b{i}.csv", make_frame(200 + 10 * (i % 3), seed=i)) for i in range(8)]
    return dq.build_baseline(frames)


def fired(result, sev="warn"):
    return result.fired(sev)


# ---------------------------------------------------------------- helpers
def test_shape_of_run_length_encodes_character_classes():
    assert dq.shape_of("TX-004217") == "A{2}-9{6}"
    assert dq.shape_of("ab-12") == "a{2}-9{2}"
    assert dq.shape_of("a") == "a"
    assert dq.shape_of("") == ""


def test_tier_boundaries():
    assert dq.tier(0, 0.01, 0.1) is None
    assert dq.tier(0.005, 0.01, 0.1) == "info"
    assert dq.tier(0.01, 0.01, 0.1) == "warn"
    assert dq.tier(0.1, 0.01, 0.1) == "critical"


def test_bin_counts_uses_right_closed_bins():
    edges = np.array([1.0, 2.0])
    assert dq.bin_counts(np.array([0.5, 1.0, 1.5, 2.0, 2.5]), edges).tolist() == [
        2,
        2,
        1,
    ]


def test_psi_is_zero_for_identical_and_grows_with_shift():
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    assert dq.psi(ref, np.array([25, 25, 25, 25])) == pytest.approx(0.0, abs=1e-12)
    small = dq.psi(ref, np.array([30, 25, 25, 20]))
    big = dq.psi(ref, np.array([70, 10, 10, 10]))
    assert 0 < small < 0.1
    assert 0.25 < big


def test_psi_survives_empty_bins():
    assert np.isfinite(dq.psi(np.array([0.5, 0.5]), np.array([100, 0])))


def test_ks_against_reference_separates_same_from_shifted():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    grid = np.quantile(ref, np.linspace(0, 1, 501))
    _, p_same = dq.ks_against_reference(rng.normal(0, 1, 200), grid, 5000)
    d_shift, p_shift = dq.ks_against_reference(rng.normal(1, 1, 200), grid, 5000)
    assert p_same > 0.001
    assert p_shift < 1e-10 and d_shift > 0.3


def test_logical_type_and_merge():
    assert dq.logical_type(pl.Series([1, 2])) == "int"
    assert dq.logical_type(pl.Series([1.5])) == "float"
    assert dq.logical_type(pl.Series(["x"])) == "string"
    assert dq.logical_type(pl.Series([True])) == "bool"
    assert dq.logical_type(pl.Series([datetime(2024, 1, 1)])) == "datetime"  # noqa: DTZ001 -- polars holds naive datetimes
    assert dq.logical_type(pl.Series([None, None], dtype=pl.String)) == "null"
    assert dq._merge_types(["int", "float"]) == "float"
    assert dq._merge_types(["int", "string"]) == "string"
    assert dq._merge_types(["null", "int"]) == "int"
    assert dq._merge_types(["null"]) == "string"


def test_coerce_numeric_counts_failures_and_treats_nan_as_null():
    s = pl.Series("v", ["1", "2.5", "oops", None, "nan"])
    out, failures = dq.coerce(s, "float")
    assert out.null_count() == 3  # oops, None, nan
    assert failures == 2  # oops and "nan" were present and became null
    out, failures = dq.coerce(pl.Series("v", [1.0, float("nan"), 3.0]), "float")
    assert out.null_count() == 1 and failures == 0


def test_coerce_datetime_parses_mixed_layouts_and_flags_garbage():
    s = pl.Series(
        "t",
        ["2024-01-02 03:04:05", "2024-01-02T03:04:05.5", "2024-01-02", "not a date"],
    )
    out, failures = dq.coerce(s, "datetime")
    assert failures == 1 and out.null_count() == 1
    assert out[2] == datetime(2024, 1, 2)  # noqa: DTZ001 -- polars holds naive datetimes


# ------------------------------------------------------------------ config
def test_default_config_is_not_shared_between_calls():
    a, b = dq.default_config(), dq.default_config()
    a["volume"]["warn_z"] = 99
    assert b["volume"]["warn_z"] == dq.DEFAULT_CONFIG["volume"]["warn_z"]


def test_example_toml_matches_defaults():
    cfg = dq.load_config(HERE / "dq_config.example.toml")
    assert cfg.values == dq.DEFAULT_CONFIG
    assert cfg.severity_overrides == {} and cfg.ignore_columns == frozenset()


@pytest.mark.parametrize(
    "text",
    [
        "[nope]\nx = 1\n",
        "[volume]\nwarn_zz = 3\n",
        '[volume]\nwarn_z = "high"\n',
        "[volume]\nwarn_z = true\n",
        '[severity_overrides]\n"made.up" = "off"\n',
        '[severity_overrides]\n"volume.empty" = "loud"\n',
        "[ignore]\nfoo = 1\n",
        "[volume\n",
    ],
)
def test_bad_configs_are_rejected(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    with pytest.raises(dq.ConfigError):
        dq.load_config(p)


def test_config_merges_over_defaults(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[volume]\nwarn_z = 4\n[ignore]\ncolumns = ["amount"]\n[severity_overrides]\n"schema.new_column" = "info"\n'
    )
    cfg = dq.load_config(p)
    assert cfg["volume"]["warn_z"] == 4.0 and cfg["volume"]["critical_z"] == 6.0
    assert cfg.ignore_columns == {"amount"}
    assert cfg.severity_overrides == {"schema.new_column": "info"}


def test_missing_config_file_is_a_config_error(tmp_path):
    with pytest.raises(dq.ConfigError):
        dq.load_config(tmp_path / "absent.toml")


# ---------------------------------------------------------------- baseline
def test_baseline_learns_kinds_and_keys(baseline):
    cols = {c["name"]: c for c in baseline["columns"]}
    assert (
        cols["id"]["unique"]
        and cols["id"]["kind"] == "text"
        and "A{2}-9{6}" in cols["id"]["shapes"]
    )
    assert (
        cols["amount"]["kind"] == "numeric" and len(cols["amount"]["quantiles"]) == 501
    )
    assert cols["qty"]["kind"] == "numeric" and not cols["qty"]["unique"]
    assert cols["kind"]["kind"] == "categorical" and set(
        cols["kind"]["categories"]
    ) == {"a", "b", "c"}
    assert cols["ts"]["kind"] == "temporal" and cols["ts"]["median_batch_span"] > 0
    assert baseline["row_counts"][:3] == [200, 210, 220]


def test_baseline_is_json_serialisable_and_round_trips(baseline, tmp_path):
    p = tmp_path / "b.json"
    dq.save_baseline(baseline, p)
    assert dq.load_baseline(p) == json.loads(json.dumps(baseline))


def test_single_batch_baseline_never_guesses_keys():
    b = dq.build_baseline([("only.csv", make_frame())])
    assert not any(c["unique"] for c in b["columns"])
    assert any("single baseline batch" in n for n in b["notes"])
    forced = dq.build_baseline([("only.csv", make_frame())], keys=("id",))
    assert {c["name"]: c["unique"] for c in forced["columns"]}["id"]


def test_auto_keys_can_be_disabled():
    frames = [(f"b{i}", make_frame(seed=i)) for i in range(3)]
    assert not any(
        c["unique"] for c in dq.build_baseline(frames, auto_keys=False)["columns"]
    )


def test_high_cardinality_strings_are_text_low_cardinality_are_categories():
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "few": rng.choice(list("abcde"), 500),
            "many": [f"user{i % 400}" for i in range(500)],
        }
    )
    cols = {c["name"]: c for c in dq.build_baseline([("x", df)])["columns"]}
    assert cols["few"]["kind"] == "categorical"
    assert cols["many"]["kind"] == "text"


def test_baseline_notes_short_history_and_optional_columns():
    a = make_frame(seed=1)
    b = make_frame(seed=2).drop("qty")
    base = dq.build_baseline([("a", a), ("b", b)])
    assert any("fewer than 5" in n for n in base["notes"])
    assert any("'qty' is missing" in n for n in base["notes"])
    assert {c["name"]: c["present_batches"] for c in base["columns"]}["qty"] == 1


def test_types_are_merged_across_batches():
    a = pl.DataFrame({"v": [1, 2, 3]})
    b = pl.DataFrame({"v": [1.5, 2.5, 3.5]})
    assert dq.build_baseline([("a", a), ("b", b)])["columns"][0]["type"] == "float"


def test_build_baseline_needs_a_batch():
    with pytest.raises(ValueError):
        dq.build_baseline([])


def test_all_null_column_gets_an_empty_profile():
    df = pl.DataFrame({"v": [None, None, None]}, schema={"v": pl.Float64})
    assert dq.build_baseline([("a", df)])["columns"][0]["kind"] == "empty"


# ------------------------------------------------------------------ schema
def test_identical_frame_is_clean(baseline):
    assert dq.check_batch(baseline, make_frame(seed=50)).alerts == []


def test_missing_new_and_type_change(baseline):
    df = (
        make_frame(seed=50)
        .drop("qty")
        .with_columns(pl.lit(1).alias("brand_new"), pl.col("amount").cast(pl.String))
    )
    res = dq.check_batch(baseline, df)
    assert {(a.check, a.column) for a in res.alerts if a.severity != "info"} >= {
        ("schema.missing_column", "qty"),
        ("schema.new_column", "brand_new"),
        ("schema.type_change", "amount"),
    }
    change = res.by_check("schema.type_change")[0]
    assert change.severity == "critical"  # numeric -> string


def test_int_to_float_is_only_info(baseline):
    df = make_frame(seed=50).with_columns(pl.col("qty").cast(pl.Float64))
    alerts = dq.check_batch(baseline, df).by_check("schema.type_change")
    assert [a.severity for a in alerts] == ["info"]


def test_rename_is_paired_not_reported_as_missing_plus_new(baseline):
    res = dq.check_batch(baseline, make_frame(seed=50).rename({"amount": "amount_usd"}))
    assert res.by_check("schema.possible_rename")[0].metrics["new"] == "amount_usd"
    assert not res.by_check("schema.missing_column") and not res.by_check(
        "schema.new_column"
    )


def test_rename_to_unrelated_name_at_same_position_is_still_found(baseline):
    res = dq.check_batch(baseline, make_frame(seed=50).rename({"kind": "zzz"}))
    assert res.by_check("schema.possible_rename")[0].metrics["old"] == "kind"


def test_unrelated_new_column_of_other_type_is_not_a_rename(baseline):
    df = make_frame(seed=50).drop("kind").with_columns(pl.lit(1.5).alias("qqqq"))
    res = dq.check_batch(baseline, df)
    assert res.by_check("schema.missing_column") and not res.by_check(
        "schema.possible_rename"
    )


def test_column_reorder_is_info_only(baseline):
    df = make_frame(seed=50).select("ts", "kind", "qty", "amount", "id")
    res = dq.check_batch(baseline, df)
    assert [a.check for a in res.alerts] == [
        "schema.column_order"
    ] and res.status == "info"


def test_optional_baseline_column_missing_is_info():
    a, b = make_frame(seed=1), make_frame(seed=2).drop("qty")
    base = dq.build_baseline([("a", a), ("b", b)])
    res = dq.check_batch(base, make_frame(seed=3).drop("qty"))
    assert res.by_check("schema.missing_column")[0].severity == "info"


# ------------------------------------------------------------------ volume
@pytest.mark.parametrize(
    "rows,expected",
    [(0, "volume.empty"), (10, "volume.row_count"), (900, "volume.row_count")],
)
def test_volume_anomalies(baseline, rows, expected):
    df = make_frame(seed=50, n=rows) if rows else make_frame(seed=50).head(0)
    res = dq.check_batch(baseline, df)
    assert expected in fired(res, "critical" if rows in (0, 10, 900) else "warn")


def test_normal_volume_wobble_is_quiet(baseline):
    for n in (180, 205, 225, 240):
        assert not dq.check_batch(baseline, make_frame(seed=n, n=n)).by_check(
            "volume.row_count"
        )


def test_volume_needs_both_significance_and_relative_change():
    # a very steady source: tiny wobble is statistically odd but practically irrelevant
    steady = dq.build_baseline(
        [(f"b{i}", make_frame(seed=i, n=1000)) for i in range(6)]
    )
    assert not dq.check_volume(steady, 1050, dq.default_config())
    assert dq.check_volume(steady, 1400, dq.default_config())


def test_single_batch_baseline_uses_poisson_floor():
    base = dq.build_baseline([("a", make_frame(n=200))])
    assert not dq.check_volume(base, 210, dq.default_config())
    assert dq.check_volume(base, 60, dq.default_config())


def test_empty_batch_skips_value_checks(baseline):
    res = dq.check_batch(baseline, make_frame().head(0))
    assert res.fired("critical") == {"volume.empty"} and res.notes


# ------------------------------------------------------------------- nulls
def test_null_rate_shift(baseline):
    df = make_frame(seed=50, n=300)
    mask = pl.Series([i % 12 == 0 for i in range(300)])
    df = df.with_columns(
        pl.when(mask).then(None).otherwise(pl.col("amount")).alias("amount")
    )
    a = dq.check_batch(baseline, df).by_check("nulls.rate_shift")[0]
    assert a.column == "amount" and a.severity == "warn"
    assert (
        dq.check_batch(
            baseline,
            df.with_columns(
                pl.when(pl.Series([i % 3 == 0 for i in range(300)]))
                .then(None)
                .otherwise(pl.col("amount"))
                .alias("amount")
            ),
        )
        .by_check("nulls.rate_shift")[0]
        .severity
        == "critical"
    )


def test_small_null_blip_below_effect_size_is_ignored(baseline):
    df = make_frame(seed=50, n=400).with_columns(
        pl.when(pl.Series([i < 4 for i in range(400)]))
        .then(None)
        .otherwise(pl.col("amount"))
        .alias("amount")
    )
    assert not dq.check_batch(baseline, df).by_check(
        "nulls.rate_shift"
    )  # 1% < 2 points


def test_all_null_column(baseline):
    df = make_frame(seed=50).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("amount")
    )
    res = dq.check_batch(baseline, df)
    assert res.by_check("nulls.all_null")[0].severity == "critical"
    assert not res.by_check(
        "schema.type_change"
    )  # an all-null column has no type to change


def test_null_rate_dropping_is_info():
    frames = []
    for i in range(6):
        df = make_frame(seed=i)
        frames.append(
            (
                f"b{i}",
                df.with_columns(
                    pl.when(pl.Series([j % 4 == 0 for j in range(200)]))
                    .then(None)
                    .otherwise(pl.col("amount"))
                    .alias("amount")
                ),
            )
        )
    base = dq.build_baseline(frames)
    res = dq.check_batch(base, make_frame(seed=60))
    assert [a.severity for a in res.by_check("nulls.rate_shift")] == ["info"]


# ----------------------------------------------------------------- numeric
def test_numeric_bounds_use_sign_rule():
    spec = dq.build_baseline([("a", pl.DataFrame({"v": np.linspace(1, 100, 300)}))])[
        "columns"
    ][0]
    assert (
        spec["lower_bound"] == 0.0
    )  # non-negative in the baseline, so exactly non-negative
    assert spec["upper_bound"] == pytest.approx(100 + 0.25 * 99)
    neg = dq.build_baseline([("a", pl.DataFrame({"v": np.linspace(-100, -1, 300)}))])[
        "columns"
    ][0]
    assert neg["upper_bound"] == 0.0


def test_out_of_range_values(baseline):
    df = make_frame(seed=50).with_columns(
        pl.when(pl.Series([i < 3 for i in range(200)]))
        .then(-5.0)
        .otherwise(pl.col("amount"))
        .alias("amount")
    )
    a = dq.check_batch(baseline, df).by_check("values.out_of_range")[0]
    assert a.column == "amount" and a.severity == "warn" and a.metrics["count"] == 3


def test_infinity_is_out_of_range(baseline):
    df = make_frame(seed=50).with_columns(
        pl.when(pl.Series([i == 0 for i in range(200)]))
        .then(float("inf"))
        .otherwise(pl.col("amount"))
        .alias("amount")
    )
    assert dq.check_batch(baseline, df).by_check("values.out_of_range")


def test_distribution_shift_needs_effect_size_not_just_significance():
    rng = np.random.default_rng(3)
    big = [(f"b{i}", pl.DataFrame({"v": rng.normal(100, 15, 3000)})) for i in range(4)]
    base = dq.build_baseline(big)
    same = dq.check_batch(base, pl.DataFrame({"v": rng.normal(100, 15, 3000)}))
    tiny = dq.check_batch(base, pl.DataFrame({"v": rng.normal(102, 15, 3000)}))
    real = dq.check_batch(base, pl.DataFrame({"v": rng.normal(125, 15, 3000)}))
    assert not same.by_check("drift.numeric")
    assert not tiny.by_check(
        "drift.numeric"
    )  # KS p-value is tiny at n=3000, PSI and D are not
    assert real.by_check("drift.numeric")[0].severity == "critical"


def test_numeric_drift_skipped_below_min_rows(baseline):
    df = make_frame(seed=50, n=25).with_columns((pl.col("amount") * 5).alias("amount"))
    cfg = dq.default_config()
    cfg["volume"]["min_rel_change"] = 10.0  # keep the volume check out of the way
    assert not dq.check_batch(baseline, df, cfg).by_check("drift.numeric")


def test_constant_baseline_column_detects_any_change():
    base = dq.build_baseline(
        [(f"b{i}", pl.DataFrame({"v": [7.0] * 100})) for i in range(3)]
    )
    res = dq.check_batch(base, pl.DataFrame({"v": [9.0] * 100}))
    assert res.by_check("values.out_of_range") or res.by_check("drift.numeric")
    assert not dq.check_batch(base, pl.DataFrame({"v": [7.0] * 100})).alerts


def test_unparseable_numbers_are_cast_failures(baseline):
    df = make_frame(seed=50).with_columns(
        pl.when(pl.Series([i < 20 for i in range(200)]))
        .then(pl.lit("N/A"))
        .otherwise(pl.col("amount").cast(pl.String))
        .alias("amount")
    )
    res = dq.check_batch(baseline, df)
    assert res.by_check("values.cast_failures")[0].severity == "critical"
    assert res.by_check("schema.type_change")


# ------------------------------------------------------------- categorical
def test_unseen_category(baseline):
    df = make_frame(seed=50).with_columns(
        pl.when(pl.Series([i % 10 == 0 for i in range(200)]))
        .then(pl.lit("zzz"))
        .otherwise(pl.col("kind"))
        .alias("kind")
    )
    a = dq.check_batch(baseline, df).by_check("values.unseen_categories")[0]
    assert a.severity == "critical" and a.metrics["categories"] == 1


def test_long_tail_column_tolerates_new_rare_values():
    rng = np.random.default_rng(0)
    vocab = [f"zone{i}" for i in range(400)]
    weights = 1.0 / np.arange(1, 401)
    weights /= weights.sum()
    frames = [
        (f"b{i}", pl.DataFrame({"z": rng.choice(vocab, 300, p=weights)}))
        for i in range(8)
    ]
    cfg = dq.default_config()
    cfg["profile"]["categorical_max_ratio"] = (
        0.2  # keep this long-tailed column categorical
    )
    base = dq.build_baseline(frames, cfg)
    assert {c["name"]: c for c in base["columns"]}["z"]["unseen_mass"] > 0
    # 8 of 300 rows (2.7%) carry brand-new zones: the baseline's own tail predicts ~3.6%
    rows = list(rng.choice(vocab, 300, p=weights))
    rows[:8] = [f"new{i}" for i in range(8)]
    assert not dq.check_batch(base, pl.DataFrame({"z": rows})).by_check(
        "values.unseen_categories"
    )
    rows[:45] = [f"new{i}" for i in range(45)]  # 15% is a real change
    assert dq.check_batch(base, pl.DataFrame({"z": rows})).by_check(
        "values.unseen_categories"
    )


def test_missing_category(baseline):
    df = make_frame(seed=50).filter(pl.col("kind") != "a")
    df = pl.concat([df, df.head(80)])  # keep the row count plausible
    assert (
        dq.check_batch(baseline, df)
        .by_check("values.missing_categories")[0]
        .metrics["category"]
        == "a"
    )


def test_rare_category_absence_is_not_an_alert():
    rng = np.random.default_rng(0)
    frames = [
        (
            f"b{i}",
            pl.DataFrame(
                {"k": rng.choice(["a", "b", "rare"], 200, p=[0.6, 0.35, 0.05])}
            ),
        )
        for i in range(6)
    ]
    base = dq.build_baseline(frames)
    small = pl.DataFrame(
        {"k": ["a"] * 30 + ["b"] * 20}
    )  # 50 rows: (0.95)^50 = 7.7%, not evidence
    assert not dq.check_batch(base, small).by_check("values.missing_categories")


def test_category_mix_shift(baseline):
    df = make_frame(seed=50, n=300).with_columns(
        pl.Series(
            "kind", np.random.default_rng(1).choice(list("abc"), 300, p=[0.1, 0.3, 0.6])
        )
    )
    a = dq.check_batch(baseline, df).by_check("drift.categorical")[0]
    assert a.column == "kind" and a.metrics["js_divergence"] > 0.1


def test_noisy_but_stable_mix_is_quiet(baseline):
    for seed in range(60, 90):
        assert not dq.check_batch(baseline, make_frame(seed=seed)).by_check(
            "drift.categorical"
        )


# ----------------------------------------------------------- keys / format
def test_duplicate_keys(baseline):
    df = make_frame(seed=50).with_columns(
        pl.when(pl.Series([i < 5 for i in range(200)]))
        .then(pl.lit("AB-500100"))
        .otherwise(pl.col("id"))
        .alias("id")
    )
    a = dq.check_batch(baseline, df).by_check("keys.duplicates")[0]
    assert a.severity == "critical" and a.metrics["duplicates"] == 5


def test_key_format_violation_reports_examples(baseline):
    df = make_frame(seed=50).with_columns(
        pl.when(pl.Series([i < 10 for i in range(200)]))
        .then(pl.lit("bad_key"))
        .otherwise(pl.col("id"))
        .alias("id")
    )
    a = dq.check_batch(baseline, df).by_check("format.unexpected_shape")[0]
    assert a.metrics["examples"] == ["bad_key"] and a.severity == "warn"


def test_key_column_skips_distribution_checks():
    frames = [
        (f"b{i}", pl.DataFrame({"n": np.arange(i * 100, i * 100 + 100)}))
        for i in range(4)
    ]
    base = dq.build_baseline(frames)
    assert base["columns"][0]["unique"]
    # an ever-growing sequence would look like drift and out-of-range forever
    assert not dq.check_batch(base, pl.DataFrame({"n": np.arange(900, 1000)})).alerts


# ---------------------------------------------------------------- freshness
def ts_alerts(baseline, df, as_of=None):
    return dq.check_batch(baseline, df, as_of=as_of)


def test_epoch_zero_timestamps_are_implausible(baseline):
    df = make_frame(seed=50, start="2024-01-02").with_columns(
        pl.when(pl.Series([i < 20 for i in range(200)]))
        .then(pl.lit(0).cast(pl.Datetime("us")))
        .otherwise(pl.col("ts"))
        .alias("ts")
    )
    assert (
        ts_alerts(baseline, df).by_check("freshness.implausible")[0].severity
        == "critical"
    )


def test_span_check_flags_a_batch_covering_another_period(baseline):
    df = make_frame(seed=50, start="2024-01-02").with_columns(
        (pl.col("ts") + pl.duration(seconds=pl.arange(0, 200) * 5000)).alias("ts")
    )
    assert ts_alerts(baseline, df).by_check("freshness.span")


def test_stale_and_future_need_an_as_of_time(baseline):
    df = make_frame(seed=50, start="2024-01-02")
    newest = df["ts"].max()
    assert not ts_alerts(baseline, df).by_check("freshness.stale")
    assert not ts_alerts(baseline, df, newest + timedelta(hours=6)).by_check(
        "freshness.stale"
    )
    stale = ts_alerts(baseline, df, newest + timedelta(days=10)).by_check(
        "freshness.stale"
    )
    assert stale and stale[0].severity == "critical"
    future = ts_alerts(baseline, df, newest - timedelta(hours=12)).by_check(
        "freshness.future"
    )
    assert future


def test_as_of_may_be_timezone_aware(baseline):
    df = make_frame(seed=50, start="2024-01-02")
    aware = (df["ts"].max() + timedelta(days=10)).replace(
        tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    assert ts_alerts(baseline, df, aware).by_check("freshness.stale")


def test_explicit_max_lag_overrides_derived_limit(baseline):
    df = make_frame(seed=50, start="2024-01-02")
    cfg = dq.default_config()
    cfg["freshness"]["max_lag_seconds"] = 3600.0
    late = dq.check_batch(baseline, df, cfg, as_of=df["ts"].max() + timedelta(hours=2))
    assert late.by_check("freshness.stale")[0].severity == "warn"


# -------------------------------------------------------- config in action
def test_severity_override_and_off(baseline):
    df = make_frame(seed=50).with_columns(pl.lit(1).alias("extra"))
    cfg = dq.default_config()
    cfg.severity_overrides["schema.new_column"] = "info"
    assert (
        dq.check_batch(baseline, df, cfg).by_check("schema.new_column")[0].severity
        == "info"
    )
    cfg.severity_overrides["schema.new_column"] = "off"
    assert not dq.check_batch(baseline, df, cfg).alerts


def test_ignored_columns_raise_nothing(baseline):
    df = make_frame(seed=50).with_columns((pl.col("amount") * 10).alias("amount"))
    assert dq.check_batch(baseline, df).by_check("drift.numeric")
    cfg = dq.default_config()
    cfg.ignore_columns = frozenset({"amount"})
    assert not dq.check_batch(baseline, df, cfg).alerts


def test_alerts_are_sorted_worst_first(baseline):
    df = make_frame(seed=50).drop("qty").with_columns(pl.lit(1).alias("extra"))
    sevs = [a.severity for a in dq.check_batch(baseline, df).alerts]
    assert sevs == sorted(sevs, key=lambda s: -dq._RANK[s])


def test_result_status_and_json_shape(baseline):
    res = dq.check_batch(baseline, make_frame(seed=50).drop("qty"))
    d = res.to_dict()
    assert d["status"] == "critical" and d["counts"]["critical"] == 1
    json.dumps(d)  # must be serialisable as-is
    assert dq.check_batch(baseline, make_frame(seed=50)).status == "ok"


# --------------------------------------------------------------------- I/O
def test_read_batch_round_trip_and_types(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("a,b,c\n1,x,2024-01-01 10:00:00\n2,y,2024-01-01 11:00:00\n")
    df = dq.read_batch(p)
    assert [dq.logical_type(df[c]) for c in df.columns] == ["int", "string", "datetime"]


def test_read_batch_header_only_and_empty_and_missing(tmp_path):
    (tmp_path / "h.csv").write_text("a,b\n")
    assert dq.read_batch(tmp_path / "h.csv").height == 0
    (tmp_path / "e.csv").write_text("")
    with pytest.raises(dq.BatchReadError):
        dq.read_batch(tmp_path / "e.csv")
    with pytest.raises(dq.BatchReadError):
        dq.read_batch(tmp_path / "nope.csv")


def test_read_batch_ragged_rows_are_an_error(tmp_path):
    (tmp_path / "r.csv").write_text("a,b\n1,2\n3,4,5,6\n")
    with pytest.raises(dq.BatchReadError):
        dq.read_batch(tmp_path / "r.csv")


def test_load_baseline_rejects_garbage(tmp_path):
    p = tmp_path / "b.json"
    p.write_text("{not json")
    with pytest.raises(dq.BatchReadError):
        dq.load_baseline(p)
    p.write_text('{"version": 99, "columns": []}')
    with pytest.raises(dq.BatchReadError):
        dq.load_baseline(p)


def test_expand_inputs_handles_dirs_globs_and_dedupes(tmp_path):
    for n in ("b.csv", "a.csv"):
        (tmp_path / n).write_text("x\n1\n")
    (tmp_path / "skip.txt").write_text("nope")
    assert [p.name for p in dq.expand_inputs([tmp_path])] == ["a.csv", "b.csv"]
    assert [
        p.name for p in dq.expand_inputs([tmp_path / "*.csv", tmp_path / "a.csv"])
    ] == ["a.csv", "b.csv"]
    with pytest.raises(dq.BatchReadError):
        dq.expand_inputs([tmp_path / "zzz*.csv"])
