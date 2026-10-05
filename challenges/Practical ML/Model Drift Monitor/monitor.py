"""Score a stream window by window against the training baseline.

One code path serves the live run and the injected-drift benchmark. For every
full window it computes drift statistics for the model's score and each feature,
raises an alert wherever a statistic is above its calibrated threshold, feeds the
score stream to the sequential detectors, and -- ``delay`` windows later --
releases the window's accuracy, log-loss and AUC, because labels arrive late.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

import model
from data import CATEGORICAL, FEATURES
from sequential import SeqParams, make_detectors, run
from stats import Baseline, class_rate_shift

WINDOW = 336
LABEL_DELAY = 4
THRESHOLD = 0.5

ALERT_SCHEMA = {
    "window": pl.Int64,
    "signal": pl.Utf8,
    "stat": pl.Utf8,
    "value": pl.Float64,
    "threshold": pl.Float64,
}
SEQ_SCHEMA = {"window": pl.Int64, "detector": pl.Utf8}


@dataclass
class MonitorResult:
    table: pl.DataFrame  # one row per window
    alerts: pl.DataFrame  # window, signal, stat, value, threshold
    sequential: pl.DataFrame  # window, detector
    dropped_rows: int


def windows(n_rows: int, window: int = WINDOW) -> tuple[list[slice], int]:
    """Full windows only; the number of trailing rows that did not fill one."""
    n_full = n_rows // window
    return [
        slice(i * window, (i + 1) * window) for i in range(n_full)
    ], n_rows - n_full * window


def columns_of(clf, df: pl.DataFrame) -> dict[str, np.ndarray]:
    """The monitored columns: the model's score and every feature."""
    return {"score": model.score(clf, df), **{f: df[f].to_numpy() for f in FEATURES}}


def build_baseline(clf, train: pl.DataFrame, seed: int = 0) -> Baseline:
    """Baseline from the training split; the score comes from out-of-fold predictions."""
    cols = columns_of(clf, train)
    cols["score"] = model.oof_scores(train, seed=seed)
    return Baseline.fit(cols, categorical=CATEGORICAL)


def _performance(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    both = len(np.unique(y)) == 2
    return {
        "accuracy": float(((p > THRESHOLD) == y).mean()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if both else None,
    }


def monitor(
    clf,
    baseline: Baseline,
    thresholds: dict[tuple[str, str], float],
    df: pl.DataFrame,
    seq: dict[str, SeqParams],
    window: int = WINDOW,
    delay: int = LABEL_DELAY,
) -> MonitorResult:
    """Monitor ``df`` (features + label) in time order.

    ``seq`` holds the calibrated sequential parameters for ``"score"`` and
    ``"error"``. The row for window *t* carries the performance of window
    *t - delay*, the earliest moment its labels exist.
    """
    slices, dropped = windows(df.height, window)
    cols = columns_of(clf, df)
    scores = cols["score"]
    labels = df["label"]
    base_pred_rate = float((baseline.columns["score"].reference > THRESHOLD).mean())

    rows, alerts = [], []
    for t, sl in enumerate(slices):
        row = {"window": t, "start": sl.start, "end": sl.stop}
        for column, values in cols.items():
            for stat, value in baseline.window_stats(column, values[sl]).items():
                row[f"{column}_{stat}"] = value
                limit = thresholds.get((column, stat))
                if limit is not None and value > limit:
                    alerts.append(
                        {
                            "window": t,
                            "signal": column,
                            "stat": stat,
                            "value": value,
                            "threshold": limit,
                        }
                    )
        row["score_mean"] = float(scores[sl].mean())
        row["pred_rate"] = float((scores[sl] > THRESHOLD).mean())
        row["pred_rate_shift"] = class_rate_shift(
            base_pred_rate, scores[sl] > THRESHOLD
        )
        w = t - delay
        row["perf_window"] = w if w >= 0 else None
        perf = {"accuracy": None, "logloss": None, "auc": None}
        if w >= 0:  # the labels of window w exist only from now on
            ws = slices[w]
            perf = _performance(labels[ws].to_numpy().astype(int), scores[ws])
        rows.append({**row, **perf})

    # sequential detectors: the score stream row by row, the error stream as labels are released
    seq_rows = []
    for name, det in make_detectors(seq["score"]).items():
        for i in run(det, scores[: len(slices) * window]):
            seq_rows.append({"window": i // window, "detector": f"{name}_score"})
    err_det = make_detectors(seq["error"])["adwin"]
    for t in range(delay, len(slices)):
        ws = slices[t - delay]
        errors = ((scores[ws] > THRESHOLD) != labels[ws].to_numpy().astype(int)).astype(
            float
        )
        if run(err_det, errors):
            seq_rows.append({"window": t, "detector": "adwin_error"})

    return MonitorResult(
        table=pl.DataFrame(rows),
        alerts=pl.DataFrame(alerts, schema=ALERT_SCHEMA),
        sequential=pl.DataFrame(seq_rows, schema=SEQ_SCHEMA),
        dropped_rows=dropped,
    )
