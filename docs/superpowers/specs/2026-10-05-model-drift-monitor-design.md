# Model Drift Monitor — design

Practical ML #8 (I). Brief: "Compare live prediction distribution against training baseline."

## Understanding (confirmed with the user)

A drift monitor for a deployed classifier: compare what the live model sees and
says against a training baseline and tell harmless drift from harmful drift.
Success = a *benchmarked* monitor, not a demo: false-alarm rate on stationary
data, and detection rate and delay on drift of known type and size, per
detector, plus an honest read of what fires on real drifting data. Decisions
already made: real drifting data **plus** injected drift for ground truth;
predictions + features + delayed-label performance; batch window statistics
**and** sequential detectors; Typer CLI + Streamlit dashboard; thresholds
calibrated from the baseline's own null distribution; own implementations of the
window statistics cross-checked against Evidently; `river` for ADWIN and
Page-Hinkley. Standalone: does not reuse challenge 7's data.

## Data and model

OpenML `electricity` (data_id 151; NSW electricity market, 45,312 half-hourly
rows 1996-1998, 8 features, label UP/DOWN of price vs a moving average), fetched
with scikit-learn's `fetch_openml` into a gitignored `data/` and cached as
parquet. It is the standard concept-drift benchmark, so natural drift exists
(UP rate 39%-48% across eighths of the series).

Rows are used in time order: first 30% **train** (fit a LightGBM classifier; the
score is P(UP)), next 10% **reference** (stationary period for threshold
calibration and the injected-drift benchmark), last 60% **live** (about 27k rows,
about 80 weekly windows). `date` (a normalized day index) is dropped as a
feature: it would only extrapolate. `day` (1-7) is categorical, the rest numeric.

## Windows

A window = 336 consecutive rows (one week). Windows are scored against the
**baseline** = the training rows' score and feature distributions (bin edges and
reference samples come from train only). Labels for window *t* become available
`LABEL_DELAY = 4` windows later; performance metrics for window *t* are computed
at *t + 4* and are never visible earlier.

## Signals

- **Prediction drift** (the brief): PSI (baseline-quantile bins, 10 bins,
  epsilon-smoothed), two-sample KS, Jensen-Shannon distance on the same bins,
  Wasserstein-1 on the score, plus predicted-class rate shift.
- **Feature drift** (to localize the cause): the same four statistics per
  numeric feature, chi-square and JS for `day`.
- **Performance** (delayed labels): accuracy, log-loss, AUC per window.
- **Sequential**: ADWIN and Page-Hinkley on the score stream; ADWIN on the
  0/1 error stream once labels arrive.

## Calibrated thresholds

For each statistic, draw many same-size windows from the **reference** period,
score each against the baseline and take the (1 - alpha) quantile as the
threshold, alpha = 1% per window. This accounts for window-size dependence (KS
and PSI both grow as windows shrink) that fixed rules of thumb ignore. The
rule-of-thumb PSI > 0.2 is reported next to it. Sequential detector parameters
(ADWIN delta, Page-Hinkley lambda) are set on the reference stream to the same
false-alarm target.

## Evaluation

1. **Injected drift.** On reference-period data, build streams of windows with
   drift starting at a known window: covariate shift (one feature shifted by
   0.5, 1, 2 sigma), prior shift (class rate), concept drift (label relationship
   changed so accuracy falls while features are untouched), and score noise.
   Report per detector and magnitude: detection rate within 10 windows, mean
   detection delay in windows, and false-alarm rate on no-drift streams, over many
   seeds with bootstrap CIs.
2. **Natural drift.** Which alarms fire on the real live stream and when, and
   whether they line up with the measured accuracy decline. State plainly which
   drift moves the score distribution and which only moves accuracy (concept
   drift with stable scores is the case score-only monitors miss).

## Components

| File | Role |
|---|---|
| `data.py` | fetch OpenML electricity, time-ordered train/reference/live split |
| `model.py` | LightGBM baseline classifier |
| `stats.py` | PSI, KS, JS, Wasserstein, chi-square, binning from baseline |
| `thresholds.py` | null distributions from reference windows, quantile thresholds |
| `sequential.py` | river ADWIN / Page-Hinkley wrappers with calibrated params |
| `monitor.py` | stream windows -> metric table + alerts (+ delayed performance) |
| `inject.py` | controlled synthetic drift generators |
| `evaluate.py` | injected-drift benchmark, detection delay, false alarms, CIs |
| `pipeline.py` | fetch -> train -> calibrate -> monitor -> benchmark -> `report.json` |
| `cli.py`, `app.py` | Typer (`fetch`, `train`, `monitor`, `report`), Streamlit dashboard |

## Dashboard

Metric time series per signal with alert markers and the threshold line; a
per-feature drift ranking for a chosen window; baseline vs live histograms; the
delayed accuracy line with the alarm timeline beside it.

## Testing

Statistics against hand-computed values and against Evidently (dev dependency)
on the same data; PSI/JS binning edge cases (empty bins, constant feature,
values outside the baseline range); null calibration hits its target false-alarm
rate on fresh stationary windows; injected drift is detected sooner as magnitude
grows; sequential detectors fire on a step change and stay quiet on a stationary
stream; labels are never used before their delay; window slicing leaves no
partial window; CLI via Typer runner and Streamlit via `AppTest`. One end-to-end
test on the real download that skips offline.

## Non-goals

No automatic retraining (that is Practical ML #21), no multi-model comparison,
no claim that an alarm means the model is wrong (the report separates "the
inputs moved" from "accuracy fell"), no streaming infrastructure (Kafka, etc.).
