# Real-Time Anomaly Detection on Streaming Metrics

**Category:** Data Analytics
**Difficulty:** X

**Status:** Implemented (Python)

Statistical process control charts on a live metric stream: a Shewhart individuals chart with the
four Western Electric run rules, an EWMA chart and a two-sided tabular CUSUM, all updated one point
at a time. A FastAPI server replays labelled streams over Server-Sent Events into a live control
chart page, and a CLI watches any CSV or stdin stream and prints alarms as they happen.

The sibling challenge [Statistical Anomaly Detector for Time Series](../Statistical%20Anomaly%20Detector%20for%20Time%20Series/)
runs ten batch detectors over a whole series. This one is strictly online: each chart keeps O(1)
state, sees a value once and has to decide right then.

## Run it

```bash
uv run --locked pytest -q                       # 32 tests
uv run spc-stream serve                         # live page at http://127.0.0.1:8000
uv run spc-stream monitor sample_data/nyc_taxi.csv --chart cusum --baseline 1548 --season 48 --ar1
some_metric_source | uv run spc-stream monitor - --chart ewma   # one value or timestamp,value per line
uv run spc-stream arl                           # rewrites results/arl.{md,json}
uv run spc-stream nab                           # rewrites results/nab.{md,json}
```

## Data

Four labelled streams from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) (MIT,
licence in `sample_data/NAB_LICENSE.txt`), the same files the sibling challenge vendors:
`nyc_taxi` (30-minute taxi demand, 5 windows), `ambient_temperature_system_failure` (hourly, 2),
`ec2_cpu_utilization_24ae8d` (5-minute CPU, 2) and `art_daily_jumpsdown` (synthetic daily cycle
with a level drop, 1). `labels.json` holds NAB's labelled windows.

## Method

**Phase I / Phase II.** The first `baseline_n` points (15% for NAB, matching its probationary
period) estimate the in-control mean and sigma. Sigma is the moving-range estimate MR-bar / 1.128,
the standard individuals-chart choice, because the plain standard deviation is inflated by any
slow drift inside the baseline. From then on every value is standardised, z = (x - mu) / sigma,
and fed to the chart, so all three charts plot on the same sigma axis.

**Charts** (`src/spc_stream/charts.py`):

- Shewhart: WE1 one point beyond 3 sigma, WE2 two of three beyond 2 sigma on one side, WE3 four of
  five beyond 1 sigma on one side, WE4 eight in a row on one side. Any subset can be switched on.
- EWMA, lambda 0.1, L 2.814, with the exact time-varying limits by default (tighter right after a
  start or reset, which gives a fast initial response) and the asymptotic limits as an option.
- CUSUM, k 0.5, h 5: upper and lower sums of z - k and -z - k, clipped at zero.

Every chart restarts after an alarm, and the monitor holds back repeat alarms for a cooldown (12
points by default) so one incident pages once.

**Prewhitening.** Every control chart assumes independent points. Real metrics are autocorrelated
and seasonal, which makes raw charts alarm on the daily cycle. `Prewhitener` optionally takes a
seasonal difference (x_t - x_{t-s}) and then fits AR(1) on Phase I, so the chart watches one-step
forecast residuals. This is the residual-chart approach Montgomery recommends for autocorrelated
data.

## Results

### The charts are correct: simulated ARL against the published tables

`results/arl.md`, 2,000 runs per cell, N(0, 1) data with a step of the given size from the
first point (zero-state ARL):

| Chart                   | Shift | Simulated ARL | Published     |
| ----------------------- | ----- | ------------- | ------------- |
| Shewhart 3 sigma        | 0     | 381.2 ± 8.8   | 370.4 (exact) |
| Shewhart 3 sigma        | 1     | 43.4 ± 1.0    | 43.9 (exact)  |
| Shewhart WE rules 1-4   | 0     | 90.6 ± 1.9    | 91.75         |
| EWMA lambda 0.1 L 2.814 | 0     | 502.5 ± 11.2  | 500           |
| EWMA lambda 0.1 L 2.814 | 1     | 10.45 ± 0.11  | 10.3          |
| CUSUM k 0.5 h 5         | 0     | 458.8 ± 9.7   | 465           |
| CUSUM k 0.5 h 5         | 1     | 10.47 ± 0.13  | 10.4          |
| CUSUM k 0.5 h 5         | 3     | 2.57 ± 0.01   | 2.57          |

Every cell is within 1.4 standard errors of the reference except EWMA at a 3 sigma shift
(2.85 against 2.9, inside the rounding of the published value). Sources: Montgomery's
*Introduction to Statistical Quality Control* for EWMA and CUSUM, Champ and Woodall 1987 for the
run rules, the closed form 1 / P(|z + d| > 3) for the plain Shewhart chart). The table shows the
classic trade-off: switching on all four run rules cuts the false-alarm ARL from 370 to 92, the
price of catching a 1 sigma shift in 9 points instead of 43. EWMA and CUSUM get to about 10 points
while keeping ARL0 near 500.

The simulation also caught a mistake while I built this: I first set L 2.7 from memory and the
in-control ARL came out at 313, not 500. The published table row belongs to L 2.814.

### Real streams: SPC assumptions break, and prewhitening helps only half the time

`results/nab.md` scores each chart with and without prewhitening (season = one day; none for the
EC2 stream). A window counts as hit if any alarm falls inside it; every alarm outside a window is a
false alarm; delay is points from the window start to the first alarm.

| Stream       | Chart    | Prewhiten | Windows hit | False / 1k points | Delay |
| ------------ | -------- | --------- | ----------- | ----------------- | ----- |
| nyc_taxi     | Shewhart | no        | 1/5         | 1.71              | 159   |
| nyc_taxi     | CUSUM    | no        | 4/5         | 2.39              | 128   |
| nyc_taxi     | CUSUM    | yes       | 5/5         | 8.78              | 57    |
| ambient temp | Shewhart | no        | 0/2         | 3.24              | -     |
| ambient temp | EWMA     | yes       | 1/2         | 1.13              | 172   |
| ec2 CPU      | Shewhart | yes       | 2/2         | 14.0              | 23.5  |
| ec2 CPU      | EWMA     | yes       | 2/2         | 3.21              | 94.5  |
| art_daily    | CUSUM    | no        | 1/1         | 0.29              | 224   |
| art_daily    | CUSUM    | yes       | 1/1         | 11.67             | 201   |

What this shows:

- **Raw charts on seasonal data miss or lag.** Phase I sigma from a daily cycle is huge, so the
  limits are wide: the raw Shewhart chart catches 1 of 5 taxi windows.
- **Prewhitening makes charts sensitive, which costs false alarms.** On taxi demand every chart
  hits all five windows with residuals, and the delay halves, but false alarms rise three to eight
  times. On ambient temperature it goes the other way: false alarms drop from 3.4 to 3.9 per 1,000
  to 1.1 for EWMA and CUSUM.
- **Seasonal differencing echoes every anomaly one season later.** On `art_daily_jumpsdown` the
  raw charts are nearly perfect (one false alarm), but x_t - x_{t-288} turns the level drop into a
  second spike exactly one day later, and the CUSUM false alarms go from 1 to 40. A differenced
  chart needs either a model of the season (STL, Holt-Winters) or to skip the echo window after an
  alarm.
- **Spiky streams favour the smoothing charts.** On the EC2 CPU stream the Shewhart chart fires on
  every isolated spike (14 false per 1,000); EWMA and CUSUM average spikes away and fire 4x less.

No setting is best everywhere. That matches why production systems let each metric pick its chart
and its baseline instead of running one global detector.

## Limits and what I'd do next

- Phase I is fixed. A stream whose level drifts permanently (ambient temperature) needs
  re-estimation after an acknowledged change; the hook is there (`Monitor.base`) but not wired to
  any policy.
- The NAB score here is a simple hit/false-alarm/delay count, not NAB's official sigmoid-weighted
  score (the sibling challenge implements a version of it).
- The live page replays files; feeding a real source would mean swapping `load()` for a queue
  consumer, since `Monitor.update` already works one value at a time.

## Where this is actually used

The Western Electric rules come from the company's 1956 *Statistical Quality Control Handbook*,
written for telephone equipment plants, and the same zone tests ship in Minitab as its eight
"tests for special causes". Minitab runs only Test 1 by default and warns that each added test
raises the chance of a false signal, which is the 370-to-92 ARL0 drop in the table above. The NIST/SEMATECH e-Handbook of Statistical Methods,
written for semiconductor manufacturing, documents EWMA control charts and their limit formula in
its process-monitoring chapter.

Healthcare uses these charts for real outcomes. A UK Research Excellence Framework impact case
study by UCL's Clinical Operational Research Unit reports that NHS Blood and Transplant monitors
early transplant outcomes with CUSUM charts alongside variable life-adjusted displays (VLADs).
The same study lists VLAD charts in UK adult cardiac surgery centres, NICOR's paediatric cardiac
analysis, the Care Quality Commission and the US Veterans Health Administration. Risk-adjusted
CUSUM charts for surgical mortality date back to paediatric cardiac surgery in the 1990s.

The newest use is monitoring machine-learning models after deployment. AIM-CU, published in the
DIDSR GitHub organisation and being prepared for the FDA's Catalog of Regulatory Science Tools,
tunes a CUSUM chart to a target false-alarm rate and uses it to flag performance drift in
clinically deployed AI devices. That is the same ARL0/ARL1 calculation this challenge reproduces.

## Files

- `src/spc_stream/charts.py`: Shewhart + run rules, EWMA, CUSUM
- `src/spc_stream/monitor.py`: Phase I baseline, prewhitening, alarm cooldown
- `src/spc_stream/arl.py`: run-length simulation and published reference values
- `src/spc_stream/nab.py`: NAB loading and scoring
- `src/spc_stream/server.py`, `static/index.html`: FastAPI SSE endpoint and canvas chart page
- `src/spc_stream/cli.py`: `spc-stream monitor | arl | nab | serve`
- `results/`: generated ARL and NAB tables (Markdown and JSON)
