# Model Drift Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Monitor a deployed classifier for prediction, feature and performance drift against a training baseline, with calibrated alert thresholds, and benchmark the detectors on injected and natural drift.

**Architecture:** Flat modules like Practical ML #5-#7. Pure statistics (`stats`, `thresholds`, `sequential`, `inject`) are separated from IO (`data`, `pipeline`). All window logic lives in `monitor.py` and takes a baseline object, so the benchmark and the live run share one code path.

**Tech Stack:** Python >=3.12, uv, polars, numpy, scipy, scikit-learn, lightgbm, river, typer, streamlit, pytest, evidently (dev only, cross-check).

**Spec:** `docs/superpowers/specs/2026-10-05-model-drift-monitor-design.md`

## Global Constraints

- Folder `challenges/Practical ML/Model Drift Monitor/`; own `pyproject.toml` + `uv.lock`; `requires-python = ">=3.12"`; `>=` bounds; pytest and evidently in the dev group; `data/` gitignored.
- Window = 336 rows; `LABEL_DELAY = 4` windows; per-window false-alarm target alpha = 0.01; PSI 10 baseline-quantile bins; split 30% train / 10% reference / 60% live, strictly in time order; `date` is never a model feature.
- Baseline bin edges, reference samples and score reference come from the train split only; thresholds from the reference split only; the live split is never used to choose anything.
- Labels of window *t* are used no earlier than window *t + LABEL_DELAY*.
- ruff (repo `ruff.toml`) must exit 0 and be run without a pipe that hides the exit code; run `uvx ruff check --fix`, `uvx ruff format` with `--config ../../../ruff.toml`; dprint via `npx --yes dprint@0.57.4 fmt --config dprint.json <files>`.
- Committed artifacts hold aggregates only (`results/report.json`); the dataset is public but is re-downloaded, not committed.

## Review Focus

- Constant feature or constant score in a window -> statistics return 0 drift, not NaN or a crash.
- Live values outside the baseline range (PSI bins must catch them via open-ended outer bins) and categories never seen in train.
- Empty bins on either side -> epsilon smoothing, finite PSI/JS.
- Stream length not a multiple of the window -> the trailing partial window is dropped and counted, never scored.
- Detection delay when drift starts mid-window -> counted from the first window containing drifted rows; a stream with no alarm reports "not detected", not a delay of 0.
- A no-drift benchmark stream must be able to produce zero alarms (rates, not NaN) and the CI code must handle all-zero and all-one outcomes.

---

### Task 1: Scaffold, data, baseline model

**Files:** Create `pyproject.toml`, `pytest.ini`, `data.py`, `model.py`, `test_data.py`, `test_model.py`; modify root `.gitignore` (add `challenges/Practical ML/Model Drift Monitor/data/`).
**Interfaces:** Produces `FEATURES: list[str]` (the 7 numeric features `period, nswprice, nswdemand, vicprice, vicdemand, transfer` plus `day` as int 1-7 -> exactly `["day","period","nswprice","nswdemand","vicprice","vicdemand","transfer"]`), `NUMERIC = FEATURES[1:]`, `CATEGORICAL = ["day"]`, `fetch(data_dir) -> Path` (OpenML id 151 -> `data/electricity.parquet`, columns `FEATURES + ["label"(int 1=UP)]`, row order preserved), `load(path) -> pl.DataFrame`, `split(df, fracs=(0.3,0.1,0.6)) -> (train, reference, live)`; `model.fit(train, seed) -> lgb.LGBMClassifier`, `model.score(clf, df) -> np.ndarray` (P(UP)).
- [ ] Failing tests: `split` is contiguous, ordered, disjoint, sizes within 1 row of the fractions, deterministic; `load` of a tiny synthetic parquet returns `FEATURES + ["label"]` with `label` in {0,1}; `fit` beats the majority rate on a synthetic task with a planted signal, is deterministic per seed, and `score` is in [0,1] with one value per row.
- [ ] Verify `uv add evidently` resolves on Python 3.12; if not, record the ruling (skip the cross-check, test against scipy/hand values only) and say so in the README.
- [ ] Implement; commit `Scaffold Model Drift Monitor, data and baseline model`.

### Task 2: Window statistics

**Files:** Create `stats.py`, `test_stats.py`, `test_stats_crosscheck.py`.
**Interfaces:** Produces `Baseline` dataclass built by `Baseline.fit(train_scores, train_features: dict[str, np.ndarray], categorical: list[str])` holding the quantile bin edges (outer bins open-ended) and the reference sample/proportions per column; `psi(base, live) -> float`, `ks(ref_sample, live) -> float`, `js(base, live) -> float` (Jensen-Shannon *distance*, base 2, in [0,1]), `wasserstein(ref_sample, live) -> float` (W1 divided by the baseline IQR so features are comparable; 0 if IQR is 0), `chi2_pvalue(base_counts, live_counts)`; `Baseline.window_stats(column, live_values) -> dict[str, float]` with keys `psi, ks, js, wasserstein` (numeric) or `psi, js, chi2_p` (categorical); `class_rate_shift(base_rate, live_labels_or_preds)`.
- [ ] Failing tests: PSI of identical data ~0; PSI against a hand-computed two-bin example; PSI symmetric-free (`psi(a,b) != psi(b,a)` is allowed) but always >= 0; JS of identical = 0 and of disjoint support -> 1; KS equals `scipy.stats.ks_2samp`; W1 equals `scipy.stats.wasserstein_distance` over IQR; constant column -> all statistics 0 and finite; live values beyond the baseline min/max land in the outer bins; an unseen category counts in an "other" bin; empty live bins stay finite (epsilon 1e-4).
- [ ] Cross-check (evidently, dev): PSI, KS statistic, JS and Wasserstein from Evidently on the same arrays agree with ours within tolerance, or the test documents the exact definitional difference (e.g. Evidently's JS is a divergence vs ours a distance; compare after squaring).
- [ ] Implement; commit `Add window drift statistics`.

### Task 3: Threshold calibration

**Files:** Create `thresholds.py`, `test_thresholds.py`.
**Interfaces:** Consumes `Baseline`. Produces `null_distribution(base, reference_columns: dict[str, np.ndarray], n_windows=500, window=336, seed=0) -> dict[(column, stat), np.ndarray]` (random contiguous-or-iid windows drawn from the reference split — iid sampling with replacement is the spec'd null; document it), `calibrate(null, alpha=0.01) -> dict[(column, stat), float]` ((1-alpha) quantile), `RULE_OF_THUMB = {"psi": 0.2}`.
- [ ] Failing tests: on a stationary N(0,1) baseline/reference, the empirical exceedance rate of calibrated thresholds on fresh windows is within [0.002, 0.03] for alpha 0.01; smaller windows give larger thresholds for KS and PSI; thresholds are deterministic per seed; a fixed PSI>0.2 rule has a much lower false-alarm rate on that data (shows why calibration matters).
- [ ] Implement; commit `Add calibrated thresholds`.

### Task 4: Sequential detectors

**Files:** Create `sequential.py`, `test_sequential.py`.
**Interfaces:** Produces `class SeqDetector(name, make)` with `.update(x: float) -> bool`, `.reset()`; factories `page_hinkley(delta, threshold)` and `adwin(delta)` (river), `calibrate_sequential(reference_stream: np.ndarray, target_false_alarms=1/…) -> dict[str, dict]` picking Page-Hinkley `threshold` and ADWIN `delta` so the reference stream (stationary) raises at most the target false alarms per window of 336 rows; `run(detector, stream) -> list[int]` indices of alarms.
- [ ] Failing tests: both stay quiet on 5,000 stationary N(0,1) points with calibrated parameters; both alarm within 300 points after a +1.5 sigma step; `reset` clears state; `run` returns sorted indices; calibration returns parameters whose reference false-alarm count <= target.
- [ ] Implement; commit `Add sequential detectors`.

### Task 5: Monitor

**Files:** Create `monitor.py`, `test_monitor.py`.
**Interfaces:** Consumes `Baseline`, thresholds, `SeqDetector`. Produces `windows(n_rows, window=336) -> list[slice]` plus the dropped-row count; `monitor(clf, baseline, thresholds, df, seq, window=336, delay=4) -> MonitorResult` with `.table` (polars, one row per window *t*: `window, start, end, <column>_<stat>` for the score and every feature, `score_mean, pred_rate`, and the *released* performance columns `perf_window, accuracy, logloss, auc`, where `perf_window = t - delay` and those three are the metrics of window `perf_window` (null while `t < delay`)), `.alerts` (long table `window, signal, stat, value, threshold`), `.sequential` (`window, detector` alarms), `.dropped_rows`.
- [ ] Failing tests: a stream drawn from the baseline distribution raises few alerts (< 5% of windows per signal at alpha 0.01 on 60 windows); a covariate shift on one feature trips that feature's alerts and not the others'; windows partition the stream with the partial tail dropped and counted; labels are never read for a window before its delay (assert by passing a label column with NaNs for the last `delay` windows' worth and checking no crash and `accuracy` NaN there); constant score window -> finite, no alert; the performance columns on row t equal the metrics of window t-delay computed directly from that window's labels.
- [ ] Implement; commit `Add window monitor`.

### Task 6: Drift injection and benchmark

**Files:** Create `inject.py`, `evaluate.py`, `test_inject.py`, `test_evaluate.py`.
**Interfaces:** Produces `inject.covariate(df, feature, sigmas, start_row)`, `inject.prior(df, rate, start_row, seed)` (resample so the UP rate becomes `rate` after `start_row`), `inject.concept(df, strength, start_row, seed)` (flip `strength` fraction of labels in rows after `start_row`, features untouched), `inject.score_noise(scores, sd, start_row, seed)`; `evaluate.benchmark(clf, baseline, thresholds, seq, reference_df, scenarios, n_seeds, n_windows=40, drift_window=15) -> pl.DataFrame` rows `(scenario, magnitude, detector, detected, delay_windows)`; `summarize(df) -> pl.DataFrame` with detection rate (Wilson CI), mean delay among detected, false-alarm rate on `scenario="none"`.
- [ ] Failing tests: injectors leave rows before `start_row` untouched and change only what they claim (concept: features identical, labels differ by about `strength`); `benchmark` on a strong covariate shift detects with rate ~1 and on `none` has false-alarm rate <= 0.1; delay is measured from the first window containing drifted rows and "not detected" is `null`, never 0; Wilson CI handles 0/n and n/n; larger magnitudes are detected no later on average.
- [ ] Implement; commit `Add drift injection and benchmark`.

### Task 7: Pipeline

**Files:** Create `pipeline.py`, `test_pipeline.py`.
**Interfaces:** Produces `run_all(data_dir, out_dir, seed=0, n_seeds=20, df=None) -> dict` that splits, fits the model, builds the baseline, calibrates thresholds and sequential parameters on the reference split, monitors the live split, runs the benchmark and writes `results/report.json` plus `data/artifacts.joblib` (clf, baseline, thresholds, seq params); `load_artifacts(data_dir)`.
- [ ] Failing tests on a synthetic drifting dataset (a step in one feature half-way through live): report has thresholds, per-detector benchmark summary, natural-run alarm timeline (first alarm window per signal), accuracy by window, split sizes, and no raw rows; the injected step is alarmed within a few windows of the true start; the report is JSON-safe (no NaN).
- [ ] Implement; commit `Add pipeline`.

### Task 8: CLI and Streamlit

**Files:** Create `cli.py`, `app.py`, `test_cli.py`, `test_app.py`, `conftest.py` (session fixture building a tiny pipeline).
**Interfaces:** CLI `fetch`, `train`, `monitor [--from-window --to-window]`, `report`; `app.py`: signal selector, metric time series with threshold line and alert markers, per-feature drift ranking for a chosen window, baseline vs live histogram, delayed accuracy with alarm timeline.
- [ ] Failing tests: Typer runner on the tiny fixture (`monitor` prints alerts and exits 0; bad window range -> exit 2 with message; `report` without a report file says to run `train`); `AppTest` renders, changing the selected window changes the feature ranking, no exceptions.
- [ ] Implement; commit `Add CLI and Streamlit dashboard`.

### Task 9: Real run, docs, bookkeeping

- [ ] `fetch` + `train` on the real data; read the natural-drift timeline against the accuracy curve; if a monitor/threshold defect appears, fix it with a failing test first.
- [ ] Re-run an alpha sanity check on the real reference split: observed false-alarm rate of calibrated thresholds on the *first live windows before any known drift* is reported, not assumed.
- [ ] README from measured numbers only (baseline model quality, thresholds vs PSI>0.2, injected-drift table per scenario/magnitude/detector with CIs, natural-drift timeline vs accuracy, what score drift misses, Evidently cross-check result, limitations incl. one dataset, iid null, 1 seed for the natural run); root README row 8 -> Implemented, Practical ML 8/30 (27%), Total 63/150 (42%); dprint; ruff exit 0; memory update; commit; push.
