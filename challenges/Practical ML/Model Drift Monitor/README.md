# Model Drift Monitor

**Category:** Practical ML
**Difficulty:** I (brief: "Compare live prediction distribution against training baseline.")

**Status:** Implemented (Python)

A drift monitor for a deployed classifier: it compares each week of live traffic
with the training baseline, raises alerts whose thresholds are *calibrated* rather
than copied from a rule of thumb, and lines the alerts up against the accuracy the
model actually delivered once the labels arrived. It runs on real data that really
does drift (the OpenML electricity series), plus drift injected on a stationary
stream so detection rates can be measured against a known answer. A Typer CLI runs
it; a Streamlit page shows drift over time, what drifted in a chosen window, and
the delayed accuracy.

## Data

[Electricity](https://www.openml.org/d/151) (OpenML data id 151, Harries 1999): 45,312
half-hourly rows from the Australian NSW market, seven features (`day`, `period`,
`nswprice`, `nswdemand`, `vicprice`, `vicdemand`, `transfer`), label = whether the NSW
price goes UP or DOWN against a moving average. `cli.py fetch` downloads it (no token)
into a gitignored `data/`.

Split in time order: **30% train (13,594 rows), 10% reference (4,531), 60% live
(27,187)**. The live stream is cut into windows of **336 rows (one week)**, giving 80
windows (307 trailing rows do not fill a window and are not scored). Labels are assumed
to arrive **4 windows late**, so the accuracy shown at window t is that of window t-4.

## What is monitored

- **Signals:** the model score and each of the seven features.
- **Statistics per window against the training baseline:** PSI (10 baseline-quantile
  bins, open outer bins), Kolmogorov-Smirnov, Jensen-Shannon distance, Wasserstein
  distance divided by the baseline standard deviation, and a chi-square statistic for
  the categorical `day`. Also the predicted-UP rate.
- **Sequential detectors** (`river`) run on the stream of scores and of the model's
  errors: ADWIN and Page-Hinkley. The error stream needs labels, so it fires
  only after the delay.
- **Score baseline is out-of-fold.** A model's scores on its own training rows are
  overconfident, so a baseline built from them would flag healthy traffic. The training
  score distribution comes from blocked 5-fold out-of-fold predictions instead.

## Thresholds: calibrated on the data, not the rule of thumb

A threshold is the 99th percentile of the statistic over windows drawn from the
reference period, so a stationary window exceeds it about 1% of the time. The
conventional "PSI > 0.2" is kept as a comparison detector.

The first calibration resampled reference rows independently (an *i.i.d. null*). It
failed on the real data: on windows of the reference period that the thresholds had
not seen, **39.8% of (signal, statistic) cells exceeded their 99% threshold**, and all
6 of 6 windows alerted. Electricity is not stationary even inside one week-long block,
so rows drawn from the whole reference are less variable than a real, contiguous
window. The default null is now **contiguous blocks** (windows cut in order from the
reference, with circular rotations to get more of them). Validated the same way,
calibrated on the first half of the reference and tested on six unseen windows of the
second half, **11.8% of cells exceed threshold and 2 of 6 windows alert**. That is much
better but still not 1%, because the reference itself drifts between its halves. No
stationary null exists for this series.

The consequence is that block thresholds are wide: for the score, PSI 2.8 and KS 0.49;
for `nswprice` PSI 3.4. The rule-of-thumb PSI > 0.2 on the score alarms in 70 of the 80
live windows, which is not a usable alert.

## Results

All numbers below are from `results/report.json` (seed 0).

### The model

| Accuracy           |       |
| ------------------ | ----- |
| Train, out-of-fold | 0.821 |
| Reference          | 0.775 |
| Live (27,187 rows) | 0.707 |
| Live ROC AUC       | 0.773 |

The UP rate is stable (train 0.445, reference 0.414, live 0.416); accuracy loss is not
a label-balance effect.

### Natural run: what alarmed on the live stream

Windows (of 80) in which each detector alarmed:

| Detector                        | Windows | First |
| ------------------------------- | ------: | ----: |
| Score KS (calibrated)           |      42 |     6 |
| Score PSI (calibrated)          |      27 |     6 |
| Any feature (union of 27 cells) |      80 |     0 |
| ADWIN on scores                 |      26 |     3 |
| Page-Hinkley on scores          |       7 |     7 |
| ADWIN on errors (after delay)   |      14 |     6 |
| Score PSI > 0.2 (rule of thumb) |      70 |     - |

The delayed accuracy has three regimes: windows 4-30 average 0.66, windows 31-50 average
0.61 (low 0.43), and from window 51 it recovers to an average of **0.82**. The score KS
alarm follows it: it fires in **76% of windows 0-50 and 10% of windows 51-79**, and across
windows its Spearman correlation with the delayed accuracy is **-0.75** (-0.60 for the
number of feature alerts). So on the real stream the score distribution is a usable
early warning of accuracy loss, and it is silent when the model recovers.

Per-signal alert counts expose a second finding. `vicprice`, `vicdemand` and `transfer`
alert in 79, 74 and 77 windows, `nswprice` in 57, `nswdemand` in 17, and `day` and
`period` never. The `vic*` and `transfer` columns are **constant through the whole
training split** and start varying around row 17,400, so any variation is huge
drift from the baseline's point of view. But the model never used them (gain 0;
`nswprice` carries 70% of the gain). This is drift that does not matter, and the union
detector reports it as an alert in every window. The per-feature view says *which*
input moved; it cannot say whether the model cares.

### Injected drift (stationary stream, 20 seeds per cell)

The reference rows are resampled into a 40-window stream and drift is injected from
window 15, so the answer is known. These streams are i.i.d. draws, so they are judged
against thresholds calibrated on the matching i.i.d. null, not the block thresholds that
the natural run uses. Detection rate is the share of streams alarmed within the drifted
region; **chance** is the same detector's rate on the no-drift control, so a detector that
alarms constantly cannot look good.

| Scenario                          | Score KS         | ADWIN on errors    | Any feature        |
| --------------------------------- | ---------------- | ------------------ | ------------------ |
| `nswprice` +0.25 / 0.5 sd         | 0 / 0            | 0 / 0.10           | 1.00               |
| `nswprice` +1 sd                  | **1.00**         | 0.40 (7.5 wks)     | 1.00               |
| `nswprice` +2 sd                  | **1.00**         | 1.00 (4.0 wks)     | 1.00               |
| `vicprice` (unused) +0.25 to 2 sd | 0                | 0                  | 1.00               |
| Prior shift to 0.5 / 0.6 / 0.75   | 0 / 0 / 0        | 0.30 / 0.95 / 1.00 | 0.85 / 1.00 / 1.00 |
| Concept drift 0.1 / 0.2 / 0.4     | 0 / 0 / 0        | 0.85 / 1.00 / 1.00 | 0.90               |
| Score noise 0.05 / 0.1 / 0.2      | 0 / 0 / **0.90** | 0.05               | 0.90               |

Chance rates: score KS and ADWIN-on-errors 0.00 in every scenario; **any feature 0.90**.

- **The union-over-features detector is not a detector.** Twenty-seven feature-level cells (6 numeric features x 4 statistics, plus 3 for `day`) at
  1% each, most of them on unused or constant inputs, alarm in 90% of windows with no
  drift at all. Its apparent 1.00 detection rates equal chance. A
  multiple-comparison correction or a model-weighted combination is needed before a
  per-feature alert can page anyone.
- **Score drift sees covariate drift that moves the score, and nothing else.** KS on the
  score catches a 1 sd shift in the most important feature immediately (delay 0.1
  windows) and a large noise increase in the model, but never an unused feature, a
  label-prior shift or a concept change. By design it is label-free.
- **Concept drift is invisible to anything computed on inputs and scores.** Only the
  error-stream ADWIN catches it (0.85 to 1.00 depending on size), at a delay of
  4 to 6.4 windows, which is at least the 4-window label delay. The monitor reports that
  delay rather than hiding it.
- The unused feature `vicprice` is drifted up to 2 sd with zero effect on the model,
  and the model-level detectors correctly stay quiet.

## Cross-check against Evidently

`test_stats_crosscheck.py` compares the statistics with Evidently's (a dev dependency)
on the same data. Categorical PSI and Jensen-Shannon distance, the normalised
Wasserstein distance, and the KS p-value agree. Two choices differ from the original
plan and are deliberate:

- **Wasserstein is divided by the baseline standard deviation**, as Evidently does, not by the IQR.
- **Chi-square is reported as a statistic** (which scales with the window) rather than a p-value
  (which saturates), and its threshold is calibrated like the others.

## Run it

```bash
uv sync
uv run python cli.py fetch      # download the data
uv run python cli.py train      # fit, calibrate, monitor, benchmark (several minutes)
uv run python cli.py report     # print results/report.json
uv run python cli.py monitor --from-window 0 --to-window 20
uv run streamlit run app.py     # dashboard
uv run pytest                   # 85 tests
```

`train` writes `results/report.json` (aggregates only, no rows) and a fitted
`data/artifacts.joblib`. That file is a pickle: only ever load the one this pipeline
wrote locally, never one from elsewhere.

## Limitations

- **One dataset, one seed for the natural run.** The benchmark averages 20 seeds, the real
  stream is a single path through one market.
- **The block null does not reach 1%** (11.8% on six unseen windows), so alert counts on
  this series are an upper bound, and a stationary guarantee cannot be given. The six
  validation windows are few; treat 11.8% as an order of magnitude, not a measurement.
- **Benchmark streams are i.i.d. resamples of the reference**, matched with an i.i.d. null.
  That isolates detection power from non-stationarity, but is more optimistic than the
  natural stream, where thresholds are far wider.
- **Injected drift is synthetic.** It shifts features, labels or scores in ways I chose.
- **The out-of-fold baseline** uses blocked folds on training rows; it is better than
  in-sample scores but not exactly the deployed model's behaviour.
- Score-level alerts are not accuracy alerts: the monitor reports that the inputs or scores
  moved, and the dashboard shows the accuracy beside them rather than inferring harm.
