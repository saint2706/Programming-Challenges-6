# Statistical Anomaly Detector for Time Series

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Ten statistical detectors (no ML, no training) for a univariate time series,
benchmarked against each other on real labelled data and on synthetic series with
planted anomalies. The point of the challenge is less "flag the big values" than
*knowing when each method lies to you*, so the README is mostly about that.

| Family | Methods |
| --- | --- |
| Point (ignores time order) | `zscore` (\|z\| > 3), `iqr` (Tukey fences, k = 1.5), `mad` (Iglewicz-Hoaglin modified z > 3.5), `gesd` (Rosner's Generalized ESD, implemented from the definition) |
| Windowed (looks back only) | `rolling-z`, `rolling-mad` (each point against the 100 before it) |
| Decomposition | `stl-mad`, `stl-z`, `stl-iqr`, `stl-gesd`: robust STL (or MSTL for several periods) removes trend and seasonality, then a point detector runs on the residual |

Plus automatic seasonal-period detection (periodogram candidates, refined and
ranked by the autocorrelation function), precision/recall/F1 at the point and
event level, and a reimplementation of the Numenta Anomaly Benchmark's window
score.

## Run it

```bash
cd "challenges/Data Analytics/Statistical Anomaly Detector for Time Series"

# One report per series: data-quality table, per-method results, interactive chart.
uv run python anomaly_detector.py sample_data/nyc_taxi.csv --labels sample_data/labels.json -o out/nyc_taxi.html
uv run python anomaly_detector.py sample_data/art_daily_jumpsdown.csv --labels sample_data/labels.json -o out/art.html
uv run python anomaly_detector.py sample_data/ambient_temperature_system_failure.csv --labels sample_data/labels.json -o out/ambient.html

# Two seasonalities at once (daily + weekly on 30-minute data) -> MSTL. Takes ~30 s.
uv run python anomaly_detector.py sample_data/nyc_taxi.csv --labels sample_data/labels.json --period 48,336 --methods mad,stl-z,stl-gesd -o out/nyc_mstl.html

# Any two-column CSV (timestamp, value); no labels needed.
uv run python anomaly_detector.py my_metric.csv --time-col ts --value-col cpu --methods mad,stl-gesd --flags-out out/flags.csv -o out/my.html

# More NAB series (downloaded into .cache/nab/, ignored by git), then run on them:
uv run python fetch_nab.py realKnownCause/machine_temperature_system_failure
uv run python anomaly_detector.py .cache/nab/machine_temperature_system_failure.csv --labels .cache/nab/labels.json -o out/machine.html

uv run python benchmark.py --mstl --write out/benchmark_results.md   # everything below, ~3 min
uv run python benchmark.py --only masking fpr                        # or chosen sections
uv run pytest -q # 95 tests
```

Options: `--methods a,b,c` (default all ten), `--period {auto,none,N[,N2]}`
(default `auto`), `--z-threshold`, `--iqr-k`, `--mad-threshold`, `--window`,
`--alpha`, `--max-outliers`, `--stl-seasonal`, `--no-stl-robust`, `--labels` /
`--label-key`, `--flags-out`. Exit code `2` on a bad file, column, method, or
labels file. STL methods are *skipped with a reason* (not run with a guess) when
no seasonal period is given or detected.

The report is one self-contained HTML file (the Plotly bundle is inlined, no
network). Click a method in the legend to show its flagged points; the lower
panel shows the anomaly score of the first shown method against its threshold;
shaded bands are the labelled windows.

## Data

Real series come from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB)
(NAB), **MIT license**, © 2014-2024 Numenta Inc. (`sample_data/NAB_LICENSE.txt`
is a copy). Four series and their labelled windows are vendored (about 750 KB) so
the demo and tests run offline; `fetch_nab.py` fetches others on demand.

| File | Rows | Step | Labelled windows | Why it's here |
| --- | --- | --- | --- | --- |
| `nyc_taxi.csv` | 10,320 | 30 min | 5 (marathon, Thanksgiving, Christmas, New Year, blizzard) | Daily *and* weekly seasonality |
| `ambient_temperature_system_failure.csv` | 7,267 | 1 h | 2 (system failures) | Real gaps in the timestamps |
| `ec2_cpu_utilization_24ae8d.csv` | 4,032 | 5 min | 2 | Not seasonal; heavily discretised values |
| `art_daily_jumpsdown.csv` | 4,032 | 5 min | 1 | Clean daily square-ish wave with a level drop |

`sample_data/labels.json` is the subset of NAB's `combined_windows.json` for
those four. Synthetic data (`synth.py`) is generated, seeded and deterministic.

## Design notes

**Everything is scored so that `flag = score > threshold`,** where the score is
in the units the classic threshold is defined in: |z|, IQR-multiples outside the
box, |modified z|. Generalized ESD is the exception (its cutoff depends on how
many points have been removed), so it reports the plain z for plotting and decides
from Rosner's R_i > λ_i.

**Generalized ESD is implemented, not wrapped.** It repeatedly removes the point
farthest from the mean of what's left, records R_i, and reports the *largest* i
with R_i > λ_i, which is what lets it catch outliers that only look extreme once
their neighbours are gone. It is verified three ways: (1) the NIST/SEMATECH
handbook's worked example (54 points, 3 outliers) reproduces R = 3.118, 2.942,
3.179 and λ = 3.158, 3.151, 3.143 to 2e-3; (2) λ_1 equals the closed-form
Grubbs critical value for n = 20, 54, 200; (3) on clean noise (n = 200, k = 10,
α = 0.05) 4.45% of 2,000 series raise any alarm, inside the ≤ 5% guarantee.

**Rolling detectors score each point against the window *before* it,** never
including itself (a spike shouldn't dilute its own baseline), and stay silent
until a full window exists.

**Median/MAD edge cases.** If more than half the points tie (MAD = 0) the
modified z falls back to the mean absolute deviation (× 1.253314) instead of
dividing by zero, and IQR with zero spread flags anything off the tied value.
A constant series has no anomalies under any detector.

**Regular grid, gaps filled, never flagged.** STL and rolling windows count
samples, so they assume equal spacing. If timestamps sit on a regular grid with
holes (the ambient-temperature file has three, 621 missing hours), the holes are
linearly interpolated so the seasonal phase stays aligned, and those samples are
masked so a detector can never flag data that was invented. If more than 25% of
the grid is missing, or timestamps are off-grid, the file is left alone and the
report says so.

**Data cleaning is counted, not silent.** Missing/unparseable timestamps,
missing/non-numeric/NaN/inf values and duplicate timestamps (first wins) are
dropped and tallied in the report; mixed ISO formats and UTC offsets are
normalised. All data-derived strings in the HTML are escaped.

**Seasonal period detection.** (1) Winsorize outliers and remove a linear trend.
(2) Take the strongest periodogram peaks as candidates. The periodogram is
coarse at long periods, so (3) each candidate is refined to an integer lag by the
autocorrelation function. A long period's ACF peak is nearly flat (1 − cos(2π/P) ≈
2π²/P²), so plain argmax was off by one on 15 of 60 random series (every one with a
period of 67 or more), and STL is unforgiving of that (see below). Scoring each lag by the *comb* of ACF values at its
multiples fixed it: an error of 1 at lag L is an error of j at lag j·L. (4) A
candidate must be a real hump (not stuck at the window edge), must rise ≥ 0.1
above the lowest ACF before it (a smooth AR(1) or drifting series has an ACF that
only decays, so its wiggles have no depth), and must clear 4/√n. (5) The
*smallest* strong candidate wins, so a period-24 signal isn't reported as 48.
Result on 60 random seasonal series (period 6-119, noise, trend, spikes):
**60 exact**; white noise called seasonal in **0 of 200**; white noise, a
constant, and a 0.98-AR(1) series all return "no period" (tested).

**Stack.** NumPy/SciPy for the numerics (Student-t quantiles for Rosner's λ),
`statsmodels` STL/MSTL for decomposition, Polars for CSV parsing and cleaning,
Plotly for the report.

## Findings: where each method fails

All numbers below are reproduced by `benchmark.py`; the ones that matter are
asserted in `test_benchmark.py`. "Event" scores treat a run of consecutive flags
as one detection and credit a hit within ±2 samples of the truth; point scores
count every sample.

### 1. Z-score cannot fire at all for n ≤ 10

Samuelson's inequality: with the sample standard deviation, |z| ≤ (n−1)/√n
*whatever the data*. That is 2.85 at n = 10, below the textbook cutoff of 3.

| n | max possible \|z\| | \|z\| of a 1e9 outlier | Z-score (3) flags it | MAD (3.5) flags it |
| --- | --- | --- | --- | --- |
| 5 | 1.79 | 1.79 | **no** | yes |
| 10 | 2.85 | 2.85 | **no** | yes |
| 11 | 3.02 | 3.02 | yes | yes |
| 30 | 5.29 | 5.29 | yes | yes |

### 2. Outliers mask themselves

100 points of N(0,1) with 10 outliers near +8 (mean of 20 seeds). The cluster
inflates the sample sd 2.66×, so each outlier sits at z ≈ 2.8.

| method | precision | recall | F1 |
| --- | --- | --- | --- |
| zscore | 0.25 | **0.03** | 0.05 |
| iqr | 0.97 | 1.00 | 0.98 |
| mad | 0.99 | 1.00 | 1.00 |
| gesd, `max_outliers=5` | 1.00 | **0.50** | 0.67 |
| gesd, `max_outliers=20` | 0.99 | 1.00 | 1.00 |

The GESD row is the trap in the "robust" method: `max_outliers` is an *upper
bound* on the count and must exceed the truth. With 5 for 10 outliers it can
report at most 5. The default is 5% of n.

### 3. Global methods are blind to seasonality; STL sees through it

Ten spikes of 6σ on a seasonal swing of ±10 (event F1, 10 seeds):

| method | seasonal spikes | + a +40 trend | flatline (stuck sensor) | level shift |
| --- | --- | --- | --- | --- |
| zscore / iqr / mad / gesd | 0.00 | 0.00 | 0.00 | 0.00 |
| rolling-z / rolling-mad | 0.00 | 0.00 | 0.78 / 1.00 | 0.00 |
| stl-mad | 0.84 | 0.91 | 0.64 | 0.57 |
| stl-z | 0.90 | 0.92 | 1.00 | 0.53 |
| stl-iqr | 0.53 | 0.57 | 0.17 | 0.16 |
| **stl-gesd** | **0.98** | **0.99** | **0.95** | **0.85** |

A spike at the seasonal midpoint is well inside the series' overall range, so no
global threshold can see it. A stuck sensor also stays inside the range, which
only the windowed detectors and STL notice. The rolling detectors do nothing for
spikes on a swing this large: the previous window's spread is dominated by the
seasonality. STL's recall is ~1.0 everywhere; the difference between the STL rows is
almost entirely *precision*, which is finding 6.

### 4. STL with the wrong period is worse than useless

Seasonal spikes, true period 24, `stl-z`:

| period given | event F1 | event recall | flagged (10 true spikes) |
| --- | --- | --- | --- |
| **24** (true) | 0.90 | 1.00 | 12.5 |
| 48 (a multiple) | 0.88 | 1.00 | 12.9 |
| 12 | 0.23 | 0.14 | 1.4 |
| 18, 30, 36 | 0.00 | 0.00 | 0.0 |
| 23, 25 (off by one) | 0.04, 0.02 | 0.03, 0.01 | 0.9, 0.3 |

The failure is *silent*: the un-removed seasonality stays in the residual, inflates
its spread, and masks the spikes exactly as in finding 2, so the detector returns
almost nothing and looks healthy. A period of 23 for a true 24 is enough to break
it. A multiple of the true period is fine. That is why period detection is
tested for exactness.

### 5. Rolling windows adapt to what they should flag

Window 50, +8 anomaly on N(0,1) noise (share of anomalous samples flagged):

| scenario | rolling-z | rolling-mad | global mad |
| --- | --- | --- | --- |
| burst of 10 outliers | 0.50 | 1.00 | 1.00 |
| level shift of 60 samples | 0.08 | 0.28 | 1.00 |

Window contamination: the burst's own points enter the window and inflate the
sd that judges the rest of the burst, so rolling-z misses half of it while the
median/MAD version doesn't. Adaptation: on a sustained shift, both stop flagging
once the new level fills the window (rolling-mad after about a third of a window,
rolling-z almost immediately). That's a feature if a level change is the new normal and a
bug if you wanted to be told the whole time.

### 6. STL's residual is *not* Gaussian, and that silently breaks the thresholds

The surprising one. On a **clean** seasonal series (no anomalies at all, 1,200
samples, mean of 6 seeds), false flags per series:

| STL fits | seasonal window | residual excess kurtosis | stl-mad | stl-z | stl-iqr | stl-gesd |
| --- | --- | --- | --- | --- | --- | --- |
| robust | 7 (statsmodels default) | **2.95** | **38.2** | 25.2 | 65.0 | 4.0 |
| robust | 13 | 1.06 | 8.3 | 11.5 | 27.8 | 0.8 |
| robust | 25 (this tool's default) | 0.58 | 4.2 | 8.2 | 18.7 | 0.7 |
| plain | 7 | 0.03 | 0.5 | 3.7 | 7.8 | 0.2 |
| plain | 25 | 0.05 | 0.7 | 4.8 | 8.8 | 0.0 |
| theory, Gaussian | | 0 | 0.6 | 3.2 | 8.4 | < 0.05 |

`robust=True` reweights points by their residual: points with a large residual
are fitted less, which (my reading; I did not isolate it) leaves small residuals
smaller and the large ones larger. What is measured is that the residual ends up
heavy-tailed. The MAD, which measures the middle of the
distribution, then under-states the spread (MAD-σ is only 0.77 of the sd), and a
"3.5σ" cutoff fires ~60× too often (38 flags against a theoretical 0.6). Plain
(non-robust) STL is almost perfectly calibrated. Two consequences:

- The defaults here are `robust=True` with a wider seasonal smoother (`--stl-seasonal 25`,
  down to 4.2 flags), because robustness is what protects the fit when a large
  share of the series is anomalous. Passing `--no-stl-robust` is better calibrated
  and gave the best F1 in a spike-only sweep (event F1, stl-mad: 0.99 plain-25 vs 0.89
  robust-25) but lets heavy contamination bend the fit.
- **`stl-gesd` is the most reliable STL detector**, not `stl-mad` or `stl-iqr`. Its
  test statistic is self-calibrating (the significance level bounds false alarms per
  series), where a fixed cutoff inherits the residual's shape. `stl-iqr` is the worst
  everywhere: the quartile box of a heavy-tailed residual is narrow.

### 7. False-positive rates on clean noise match theory (where theory exists)

400 series of 1,000 i.i.d. N(0,1) points; no anomaly exists, so every flag is false.

| method | measured | theoretical |
| --- | --- | --- |
| zscore (3) | 0.00255 | 0.00270 |
| iqr (1.5) | 0.00724 | 0.00698 |
| mad (3.5) | 0.00049 | 0.00047 |
| rolling-z (window 50) | 0.00453 | 0.00460 |
| gesd (α = 0.05, n = 200), share of series with ≥ 1 flag | 0.0445 | ≤ 0.05 |

Worth noticing: rolling-z's theoretical rate is a *t*-distribution
tail, not the normal one, because the window's own mean and sd are estimated.
With 50 points the normal would predict 0.0027 and be off by 1.7×. The IQR rule
flags 0.7% of perfectly Gaussian data by design, so on 100k points it "finds" ~700
anomalies. Multiple testing is not a bug you can tune away: at these rates, a
1M-sample series produces thousands of false alarms at 3σ, and the more
samples you scan, the more of them there are.

## Real data (NAB labelled windows)

Default parameters, no tuning per series, period auto-detected. "Windows hit"
is how many labelled windows contain at least one detection; NAB score is 100 for
a perfect early detection of every window, 0 for flagging nothing, negative for
false alarms outweighing hits (each false alarm event costs up to 0.11 of a hit).

| Series | Method | Windows hit | Flagged | False-alarm events | Point F1 | NAB |
| --- | --- | --- | --- | --- | --- | --- |
| nyc_taxi | zscore / mad | 1/5 | 1 | 0 | 0.00 | 18.0 |
| nyc_taxi | gesd, rolling-z, rolling-mad | 0/5 | 0 | 0 | 0.00 | 0.0 |
| nyc_taxi | stl-mad (period 48) | 5/5 | 1650 | 224 | 0.17 | −138.5 |
| nyc_taxi | stl-z (period 48) | 4/5 | 366 | 80 | 0.07 | −16.3 |
| nyc_taxi | stl-gesd (period 48) | 2/5 | 9 | 0 | 0.02 | 36.5 |
| nyc_taxi | **stl-z, MSTL 48+336** | **5/5** | 264 | 22 | 0.29 | **75.1** |
| nyc_taxi | stl-gesd, MSTL 48+336 | 5/5 | 399 | 37 | 0.37 | 63.7 |
| ambient_temperature | zscore | 2/2 | 24 | 2 | 0.04 | **84.0** |
| ambient_temperature | iqr | 2/2 | 52 | 4 | 0.08 | 81.6 |
| ambient_temperature | mad, gesd | 0/2 | 0 | 0 | 0.00 | 0.0 |
| ambient_temperature | stl-gesd (period 168) | 1/2 | 32 | 4 | 0.06 | 36.9 |
| ec2_cpu | zscore | 2/2 | 16 | 13 | 0.01 | 58.6 |
| ec2_cpu | gesd | 2/2 | 22 | 15 | 0.03 | 54.1 |
| ec2_cpu | iqr, mad, rolling-mad | 2/2 | 1,000+ | 841+ | 0.15 | −2,162 |
| art_daily_jumpsdown | **stl-z** | 1/1 | 108 | 0 | **0.42** | **92.4** |
| art_daily_jumpsdown | stl-gesd | 1/1 | 114 | 6 | 0.42 | 59.5 |
| art_daily_jumpsdown | mad | 1/1 | 1500 | 13 | 0.12 | 23.8 |
| art_daily_jumpsdown | zscore, iqr, gesd | 0/1 | 0 | 0 | 0.00 | 0.0 |

(Full table, every method on every series: `out/benchmark_results.md` after running
the benchmark.)

What the real data says:

- **No detector wins everywhere; the winner tracks the anomaly's shape.** The
  ambient-temperature failures are large level drops, and plain `zscore` (84.0)
  beats every STL variant, because there's no daily cycle to remove (the ACF has no peak at 24, so auto-detection
  finds only a weekly period). Adding decomposition where the anomaly isn't
  hiding in the seasonality just adds noise. The `art_daily` drop is invisible to
  every global test and found cleanly (92.4, zero false alarms) by `stl-z`.
- **Getting the seasonality right matters more than the scorer.** On NYC taxi, single
  STL at the daily period found 5/5 windows with `stl-mad` but at 224 false-alarm events
  (NAB −138.5); adding the weekly component (MSTL 48+336) drops false-alarm events
  from 80 to 22 for `stl-z` and moves NAB from −16.3 to **+75.1**. The daily-only
  residual still contains the weekly cycle. (MSTL at n ≈ 10k takes about 30 s; STL
  takes 2-10 s.)
- **Discretised data breaks MAD and IQR.** `ec2_cpu` values are mostly a handful of
  distinct levels, so the middle half of the data is a tie: MAD/IQR see a near-zero
  spread and flag ~1,000 points (26% of the series) as anomalous. `zscore` and `gesd`,
  which use the mean and sd, are far better behaved here (58.6 and 54.1). This is the
  one place the non-robust statistic wins on real data.
- **The NAB score punishes noisy detectors severely.** `stl-mad` finding all 5 NYC
  windows scores −138, worse than flagging nothing (0), because it also raises 224 false
  alarm events. Each false alarm costs 0.11 of a hit, so ~9 false alarms cancel one hit.
  Detecting is easy; not crying wolf is the hard part, and it is finding 6 again.
- **Point F1 is low for everything (≤ 0.43)** because NAB windows are wide (~10% of
  the file divided over the windows) and the true anomaly is a small part of each; a
  detector that flags the anomaly's peak covers a fraction of its window. Event-level and
  NAB scores are the fairer lens for this data, which is why the tool reports all three.

## NAB score: how this differs from the official scorer

`evaluate.nab_score` follows Lavin & Ahmad (2015): only the first detection in each
window scores, `A_tp · sigmoid(y)` with y = −(distance to window end)/width so an early
hit is worth up to ~0.99 and a hit on the last sample ~0; a detection outside every window
costs `A_fp · sigmoid(y)` by distance past the previous window (nearly free just after,
full `A_fp` far away, full `A_fp` if none before); an undetected window costs `A_fn`;
weights are the "standard" profile (1, 0.11, 1); normalised so a detector that never
fires is 0 and perfect is 100. The differences: the official scorer consumes a continuous
anomaly-probability stream and chooses the best threshold itself, whereas these detectors
emit binary flags at fixed textbook thresholds; and a run of consecutive flags is collapsed
to one detection (at its first sample). Scores here are comparable to each other, not to
NAB's published leaderboard.

## Limits and what I'd do next

- Univariate only, batch only. Streaming variants (online STL, SPC charts) are the
  separate "Real-Time Anomaly Detection on Streaming Metrics" challenge.
- STL is slow: about 2-10 s for 10k points with `robust=True` (15 outer iterations), MSTL
  ~30 s. Fine for a report, not for an interactive loop.
- Periods are found by position, in samples. Irregular timestamps aren't resampled beyond
  the gap filling described above.
- `MSTL` uses statsmodels' default per-period windows; only `robust` is passed through, not
  `--stl-seasonal`.
- Thresholds are the textbook ones and applied blindly on purpose, to show what the
  defaults do. Choosing a threshold for a given false-alarm budget (say, from the residual's
  empirical tail) is the obvious next step and finding 6 says it would help most for STL.
- The vendored NAB slice is four series. `fetch_nab.py` makes the other ~54 available.

## Where this is actually used

Monitoring and SRE teams run STL-plus-threshold (or its cousins, S-H-ESD and MSTL)
on request rates, latency, and revenue metrics to alert on deviations from the normal
daily/weekly shape, rather than from a static threshold that fires every afternoon. Data
quality teams run robust z-scores on batch statistics; fraud and IoT teams run the same on
sensor and transaction streams.

## Files

| File | What it is |
| --- | --- |
| `detectors.py` | The ten detectors, Generalized ESD, STL/MSTL wrapper, period detection |
| `evaluate.py` | Point / event metrics and the NAB-style window score |
| `anomaly_detector.py` | CSV loading and cleaning, gap filling, analysis, HTML report, CLI |
| `synth.py` | Seeded synthetic series with planted anomalies |
| `benchmark.py` | Failure-mode demonstrations and the real-data table (markdown output) |
| `fetch_nab.py` | Download more NAB series + labels into `.cache/nab/` |
| `test_*.py` | 95 tests (NIST GESD example, Samuelson bound, Gaussian false-positive rates, calibration, gap filling, escaping, CLI) |
| `sample_data/` | 4 vendored NAB series, `labels.json`, `NAB_LICENSE.txt` |
