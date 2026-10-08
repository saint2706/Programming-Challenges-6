import numpy as np
import polars as pl
import pytest
from drift_monitor import model
from drift_monitor.data import CATEGORICAL, FEATURES, NUMERIC
from drift_monitor.monitor import columns_of, monitor, windows
from helpers import W, make_fitted, synth
from sklearn.metrics import log_loss, roc_auc_score


@pytest.fixture(scope="module")
def fitted():
    return make_fitted()[:4]


def test_windows_partition_the_stream_and_drop_the_partial_tail():
    sl, dropped = windows(1050, 100)
    assert len(sl) == 10 and dropped == 50
    assert [(s.start, s.stop) for s in sl][:2] == [(0, 100), (100, 200)]
    assert windows(99, 100) == ([], 99)


def test_columns_of_has_the_score_and_every_feature(fitted):
    clf, *_ = fitted
    cols = columns_of(clf, synth(300, 2))
    assert list(cols) == ["score", *FEATURES]


def test_a_stream_from_the_baseline_distribution_raises_few_alerts(fitted):
    clf, base, thr, seq = fitted
    res = monitor(clf, base, thr, synth(6000, 3), seq, window=W, delay=4)
    assert res.table.height == 60 and res.dropped_rows == 0
    cells = res.table.height * len(thr)
    assert res.alerts.height / cells < 0.05


def test_a_covariate_shift_trips_that_feature_and_not_the_others(fitted):
    clf, base, thr, seq = fitted
    live = synth(6000, 4, shift_feature="vicdemand", shift=0.5, from_row=3000)
    res = monitor(clf, base, thr, live, seq, window=W, delay=4)
    late = res.alerts.filter(pl.col("window") >= 30)
    assert late.filter(pl.col("signal") == "vicdemand").height >= 25
    other = late.filter(~pl.col("signal").is_in(["vicdemand"]))
    assert other.height <= 0.05 * 30 * len(thr)
    assert (
        res.alerts.filter(
            (pl.col("window") < 30) & (pl.col("signal") == "vicdemand")
        ).height
        <= 3
    )


def test_the_table_has_one_column_per_signal_and_statistic(fitted):
    clf, base, thr, seq = fitted
    res = monitor(clf, base, thr, synth(500, 5), seq, window=W, delay=2)
    for col in ["score", *NUMERIC]:
        for stat in ("psi", "ks", "js", "wasserstein"):
            assert f"{col}_{stat}" in res.table.columns
    for stat in ("psi", "js", "chi2"):
        assert f"{CATEGORICAL[0]}_{stat}" in res.table.columns
    wanted = {
        "window",
        "start",
        "end",
        "score_mean",
        "pred_rate",
        "perf_window",
        "accuracy",
        "logloss",
        "auc",
    }
    assert wanted <= set(res.table.columns)


def test_performance_on_row_t_is_the_metrics_of_window_t_minus_delay(fitted):
    clf, base, thr, seq = fitted
    live = synth(1000, 6)
    delay = 3
    res = monitor(clf, base, thr, live, seq, window=W, delay=delay)
    t = res.table
    assert (
        t["accuracy"][:delay].null_count() == delay
        and t["perf_window"][:delay].null_count() == delay
    )
    for row in range(delay, t.height):
        w = row - delay
        assert t["perf_window"][row] == w
        sl = slice(w * W, (w + 1) * W)
        y, p = live["label"].to_numpy()[sl], model.score(clf, live[sl])
        assert t["accuracy"][row] == pytest.approx(((p > 0.5) == y).mean())
        assert t["logloss"][row] == pytest.approx(log_loss(y, p, labels=[0, 1]))
        assert t["auc"][row] == pytest.approx(roc_auc_score(y, p))


def test_labels_of_the_last_delay_windows_are_never_read(fitted):
    clf, base, thr, seq = fitted
    live = synth(1000, 7)
    broken = live.with_columns(
        pl.when(pl.int_range(pl.len()) >= 700)
        .then(None)
        .otherwise(pl.col("label"))
        .alias("label")
    )
    a = monitor(clf, base, thr, live, seq, window=W, delay=3)
    b = monitor(clf, base, thr, broken, seq, window=W, delay=3)
    assert a.table["accuracy"].to_list() == b.table["accuracy"].to_list()


def test_a_constant_score_window_stays_finite(fitted):
    clf, base, thr, seq = fitted
    live = (
        synth(400, 8)
        .with_columns(pl.lit(0.5).alias(f) for f in NUMERIC)
        .with_columns(pl.lit(3).alias("day"))
    )
    res = monitor(clf, base, thr, live, seq, window=W, delay=2)
    stat_cols = [
        c
        for c in res.table.columns
        if c.endswith(("_psi", "_ks", "_js", "_wasserstein", "_chi2"))
    ]
    assert np.isfinite(res.table.select(stat_cols).to_numpy()).all()


def test_sequential_alarms_are_attributed_to_windows(fitted):
    clf, base, thr, seq = fitted
    live = synth(6000, 9, shift_feature="nswprice", shift=1.0, from_row=3000)
    res = monitor(clf, base, thr, live, seq, window=W, delay=4)
    assert {"window", "detector"} <= set(res.sequential.columns)
    score_alarms = res.sequential.filter(pl.col("detector").str.ends_with("_score"))
    assert score_alarms.height > 0 and score_alarms["window"].min() >= 25
    assert set(res.sequential["detector"]) <= {
        "page_hinkley_score",
        "adwin_score",
        "adwin_error",
    }
