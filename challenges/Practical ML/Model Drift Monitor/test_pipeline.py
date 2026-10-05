import json

import numpy as np
import polars as pl
import pytest

from data import NUMERIC
from pipeline import load_artifacts, run_all

W = 100


def drifting(n=6000, seed=0, step_at=3600):
    """Stationary, then nswprice shifts by +1.0 from row ``step_at``."""
    rng = np.random.default_rng(seed)
    cols = {"day": rng.integers(1, 8, n)}
    for name in NUMERIC:
        cols[name] = rng.random(n)
    cols["nswprice"][step_at:] += 1.0
    p = 1 / (1 + np.exp(-(6 * (np.clip(cols["nswprice"], 0, 1) - 0.5))))
    cols["label"] = (rng.random(n) < p).astype(int)
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("drift")
    report = run_all(
        tmp,
        tmp / "results",
        seed=0,
        n_seeds=2,
        df=drifting(),
        window=W,
        bench_windows=20,
        bench_drift_window=8,
    )
    return report, tmp


def test_report_has_every_section(run):
    report, tmp = run
    assert report["split"] == {"train": 1800, "reference": 600, "live": 3600}
    assert report["window"] == W and report["label_delay"] == 4
    for key in (
        "model",
        "thresholds",
        "sequential_params",
        "natural",
        "benchmark",
        "reference_contiguous",
    ):
        assert key in report, key
    assert (tmp / "results" / "report.json").exists() and (
        tmp / "artifacts.joblib"
    ).exists()
    assert {"accuracy_train_oof", "accuracy_reference", "accuracy_live"} <= set(
        report["model"]
    )


def test_thresholds_are_listed_per_signal_and_statistic_next_to_the_rule_of_thumb(run):
    report, _ = run
    t = report["thresholds"]
    assert "score.ks" in t["calibrated"] and "nswprice.psi" in t["calibrated"]
    assert t["rule_of_thumb_psi"] == 0.2 and t["alpha"] == 0.01


def test_the_shifted_feature_alerts_persistently_from_its_true_start_and_not_before(
    run,
):
    report, _ = run
    step_window = (3600 - 2400) // W  # live starts at row 2400
    wins = report["natural"]["alert_window_list_by_signal"]["nswprice"]
    after = [w for w in wins if w >= step_window]
    before = [w for w in wins if w < step_window]
    assert len(after) >= 0.9 * (36 - step_window)
    assert len(before) <= 3
    # the score is a downstream proxy of the inputs; among the features the cause is localized
    counts = {
        k: v
        for k, v in report["natural"]["alert_windows_by_signal"].items()
        if k != "score"
    }
    assert max(counts, key=counts.get) == "nswprice"


def test_natural_section_lists_accuracy_by_window_and_alarm_windows(run):
    report, _ = run
    n = report["natural"]
    assert len(n["accuracy_by_window"]) == 36 - 4  # released windows only
    assert set(n["alarm_windows"]) >= {
        "score_ks",
        "features_any",
        "adwin_error",
        "page_hinkley_score",
    }
    assert n["dropped_rows"] == 0


def test_benchmark_summary_has_scenarios_detectors_and_chance_rates(run):
    report, _ = run
    rows = report["benchmark"]["summary"]
    assert {r["scenario"].split(":")[0] for r in rows} >= {
        "none",
        "concept",
        "prior",
        "score_noise",
    }
    assert {
        "detection_rate",
        "chance_rate",
        "false_alarm_per_window",
        "mean_delay_windows",
    } <= set(rows[0])
    assert report["benchmark"]["n_seeds"] == 2


def test_report_is_json_safe_and_holds_no_raw_rows(run):
    _report, tmp = run
    text = (tmp / "results" / "report.json").read_text()
    json.loads(text)  # no NaN/Infinity tokens
    assert "NaN" not in text and "Infinity" not in text
    assert len(text) < 400_000


def test_artifacts_reload_and_monitor(run):
    _, tmp = run
    art = load_artifacts(tmp)
    assert {"clf", "baseline", "thresholds", "seq", "window", "delay"} <= set(art)
    assert ("score", "ks") in art["thresholds"]
