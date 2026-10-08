"""fetch -> train -> calibrate -> monitor the live stream -> benchmark -> report.

Writes ``results/report.json`` (aggregates only, no raw rows) and the fitted
artifacts the CLI and dashboard reload from the gitignored ``data/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from drift_monitor import data, model
from drift_monitor.evaluate import (
    alarm_windows,
    benchmark,
    default_scenarios,
    summarize,
)
from drift_monitor.monitor import (
    LABEL_DELAY,
    THRESHOLD,
    WINDOW,
    build_baseline,
    columns_of,
    monitor,
)
from drift_monitor.paths import project_root
from drift_monitor.sequential import calibrate_sequential
from drift_monitor.thresholds import ALPHA, RULE_OF_THUMB, calibrate, null_distribution

HERE = project_root()
RESULTS_DIR = HERE / "results"
ARTIFACTS_FILE = "artifacts.joblib"
DETECTORS = (
    "score_psi",
    "score_ks",
    "score_js",
    "score_wasserstein",
    "features_any",
    "psi_rule_of_thumb",
    "page_hinkley_score",
    "adwin_score",
    "adwin_error",
)


def load_artifacts(data_dir: Path = data.DATA_DIR) -> dict:
    """Reload what ``run_all`` saved.

    joblib is pickle: only ever load the file this pipeline wrote into the local,
    gitignored ``data/`` directory, never one from an untrusted source.
    """
    return joblib.load(Path(data_dir) / ARTIFACTS_FILE)


def load_context(data_dir: Path = data.DATA_DIR) -> tuple[dict, pl.DataFrame]:
    """Saved artifacts plus the live split of the cached dataset."""
    data_dir = Path(data_dir)
    live = data.split(data.load(data_dir / data.PARQUET))[2]
    return load_artifacts(data_dir), live


def run_monitor(art: dict, live: pl.DataFrame):
    """Monitor ``live`` with the saved model, baseline, thresholds and detectors."""
    return monitor(
        art["clf"],
        art["baseline"],
        art["thresholds"],
        live,
        art["seq"],
        window=art["window"],
        delay=art["delay"],
    )


def _clean(obj):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _null_validation(
    clf, base, ref: pl.DataFrame, window: int, seed: int
) -> dict | None:
    """Calibrate on the first half of the reference, test on contiguous windows of the second.

    Compares the i.i.d. and block nulls on data the thresholds never saw: the share
    of (signal, statistic) cells above threshold, which should be about alpha.
    """
    half = ref.height // 2
    if half < window or ref.height - half < window:
        return None
    first, second = columns_of(clf, ref[:half]), columns_of(clf, ref[half:])
    n_windows = (ref.height - half) // window
    out = {}
    for mode in ("iid", "blocks"):
        thr = calibrate(
            null_distribution(base, first, window=window, seed=seed, mode=mode),
            alpha=ALPHA,
        )
        cells, hit_windows = 0, 0
        for w in range(n_windows):
            sl = slice(w * window, (w + 1) * window)
            alerting = 0
            for column, values in second.items():
                for stat, value in base.window_stats(column, values[sl]).items():
                    cells += 1
                    alerting += value > thr[(column, stat)]
            hit_windows += alerting > 0
            out.setdefault(mode, {"alerts": 0})["alerts"] += alerting
        out[mode].update(
            {
                "windows": n_windows,
                "cell_alert_rate": out[mode]["alerts"] / cells,
                "windows_with_any_alert": hit_windows,
            }
        )
        del out[mode]["alerts"]
    return out


def _accuracy(clf, df: pl.DataFrame) -> float:
    return float(((model.score(clf, df) > THRESHOLD) == df["label"].to_numpy()).mean())


def _spearman(a, b) -> float | None:
    pairs = [
        (x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None
    ]
    if len(pairs) < 5:
        return None
    r = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    return float(r) if math.isfinite(r) else None


def _natural(res, thr) -> dict:
    t = res.table
    alarms = alarm_windows(res, thr)
    perf = t.filter(pl.col("perf_window").is_not_null())
    by_window = perf.select(
        pl.col("perf_window").alias("window"), "accuracy", "logloss", "auc"
    ).to_dicts()
    acc_by_window = {r["window"]: r["accuracy"] for r in by_window}
    windows = t["window"].to_list()
    feature_alert_count = {
        w: n
        for w, n in res.alerts.filter(pl.col("signal") != "score")
        .group_by("window")
        .len()
        .iter_rows()
    }
    ks = dict(zip(windows, t["score_ks"].to_list(), strict=True))
    released = sorted(acc_by_window)
    return {
        "n_windows": t.height,
        "dropped_rows": res.dropped_rows,
        "alarm_windows": {d: sorted(alarms[d]) for d in DETECTORS},
        "first_alarm_window": {
            d: (min(alarms[d]) if alarms[d] else None) for d in DETECTORS
        },
        "alert_window_list_by_signal": {
            s: sorted(set(res.alerts.filter(pl.col("signal") == s)["window"].to_list()))
            for s in ["score", *data.FEATURES]
        },
        "alert_windows_by_signal": {
            s: res.alerts.filter(pl.col("signal") == s)["window"].n_unique()
            for s in ["score", *data.FEATURES]
        },
        "accuracy_by_window": by_window,
        "series": t.select(
            "window", "score_ks", "score_psi", "pred_rate", "score_mean"
        ).to_dicts(),
        "spearman_accuracy_vs_score_ks": _spearman(
            [ks[w] for w in released], [acc_by_window[w] for w in released]
        ),
        "spearman_accuracy_vs_feature_alerts": _spearman(
            [feature_alert_count.get(w, 0) for w in released],
            [acc_by_window[w] for w in released],
        ),
    }


def run_all(
    data_dir: Path,
    out_dir: Path,
    seed: int = 0,
    n_seeds: int = 20,
    df: pl.DataFrame | None = None,
    window: int = WINDOW,
    delay: int = LABEL_DELAY,
    bench_windows: int = 40,
    bench_drift_window: int = 15,
    null_mode: str = "blocks",
) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = data.load(data.fetch(data_dir))
    train, ref, live = data.split(df)

    clf = model.fit(train, seed=seed)
    base = build_baseline(clf, train, seed=seed)
    ref_cols = columns_of(clf, ref)
    thr = calibrate(
        null_distribution(base, ref_cols, window=window, seed=seed, mode=null_mode),
        alpha=ALPHA,
    )
    ref_err = ((model.score(clf, ref) > THRESHOLD) != ref["label"].to_numpy()).astype(
        float
    )
    seq = {
        "score": calibrate_sequential(
            ref_cols["score"], window=window, alpha=ALPHA, seed=seed, mode=null_mode
        ),
        "error": calibrate_sequential(
            ref_err, window=window, alpha=ALPHA, seed=seed, mode=null_mode
        ),
    }

    res = monitor(clf, base, thr, live, seq, window=window, delay=delay)
    ref_res = monitor(clf, base, thr, ref, seq, window=window, delay=delay)
    n_cells = max(ref_res.table.height * len(thr), 1)

    oof = model.oof_scores(train, seed=seed)
    live_scores = model.score(clf, live)
    # The benchmark streams are i.i.d. draws from the reference, so they are judged
    # against thresholds calibrated the same way; contiguous-block thresholds are for
    # the natural (contiguous) live stream and would be mismatched here.
    bench_thr = calibrate(
        null_distribution(base, ref_cols, window=window, seed=seed, mode="iid"),
        alpha=ALPHA,
    )
    bench_seq = {
        "score": calibrate_sequential(
            ref_cols["score"], window=window, alpha=ALPHA, seed=seed, mode="iid"
        ),
        "error": calibrate_sequential(
            ref_err, window=window, alpha=ALPHA, seed=seed, mode="iid"
        ),
    }
    runs = benchmark(
        clf,
        base,
        bench_thr,
        bench_seq,
        ref,
        default_scenarios(clf),
        n_seeds=n_seeds,
        n_windows=bench_windows,
        drift_window=bench_drift_window,
        window=window,
        delay=delay,
    )

    report = {
        "seed": seed,
        "window": window,
        "label_delay": delay,
        "split": {"train": train.height, "reference": ref.height, "live": live.height},
        "model": {
            "accuracy_train_oof": float(
                ((oof > THRESHOLD) == train["label"].to_numpy()).mean()
            ),
            "accuracy_reference": _accuracy(clf, ref),
            "accuracy_live": _accuracy(clf, live),
            "auc_live": float(roc_auc_score(live["label"].to_numpy(), live_scores)),
            "up_rate": {
                "train": float(train["label"].mean()),
                "reference": float(ref["label"].mean()),
                "live": float(live["label"].mean()),
            },
        },
        "null_mode": null_mode,
        "null_validation": _null_validation(clf, base, ref, window, seed),
        "thresholds": {
            "alpha": ALPHA,
            "rule_of_thumb_psi": RULE_OF_THUMB["psi"],
            "calibrated": {f"{c}.{s}": v for (c, s), v in sorted(thr.items())},
        },
        "sequential_params": {k: vars(v) for k, v in seq.items()},
        "reference_contiguous": {
            "n_windows": ref_res.table.height,
            "alert_cell_rate": ref_res.alerts.height / n_cells,
            "windows_with_any_alert": ref_res.alerts["window"].n_unique(),
        },
        "natural": _natural(res, thr),
        "benchmark": {
            "n_seeds": n_seeds,
            "n_windows": bench_windows,
            "drift_window": bench_drift_window,
            "summary": summarize(runs).to_dicts(),
        },
    }
    report = _clean(report)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    joblib.dump(
        {
            "clf": clf,
            "baseline": base,
            "thresholds": thr,
            "seq": seq,
            "window": window,
            "delay": delay,
        },
        data_dir / ARTIFACTS_FILE,
    )
    return report
