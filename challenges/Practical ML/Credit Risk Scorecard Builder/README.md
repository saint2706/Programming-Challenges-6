# Credit Risk Scorecard Builder

**Category:** Practical ML
**Difficulty:** I (brief: "Binning, WOE/IV, logistic regression — the industry-standard method.")

**Status:** Implemented (Python)

Source modules live in `src/credit_scorecard/`; the tests are in `tests/`; the generated report and
charts are in `results/`.

Builds a points-based credit scorecard from the
[Give Me Some Credit](https://www.openml.org/d/45577) data (OpenML 45577: 150,000 borrowers, 6.7%
seriously delinquent within two years): monotonic WOE binning written from scratch, IV and
correlation screening, a logistic regression on the WOE values, and the industry points scaling
(600 points = 50:1 good:bad odds, 20 points to double the odds). It ships a CLI, a Streamlit app that
scores an applicant and lists the reasons behind the score, and an OpenVINO graph that scores
batches on the CPU or the integrated GPU.

```bash
uv run scorecard train                  # fit, evaluate, write results/ (report.md, charts, scorecard.json)
uv run scorecard score --applicant '{"age": 41, "MonthlyIncome": 5200, "RevolvingUtilizationOfUnsecuredLines": 0.92}'
uv run scorecard devices                # OpenVINO devices
uv run scorecard bench --rows 4000000   # NumPy vs OpenVINO CPU/GPU
uv run streamlit run src/credit_scorecard/app.py
uv run pytest -q
```

Data is read as parquet with polars (never pandas) and cached in `data/` on first use. Charts are
Altair (Vega-Lite), exported to PNG with `vl-convert`; there is no matplotlib.

## Results

Stratified random 70/30 split (104,573 train / 44,817 test). The scorecard keeps 7 of the 10 features.

| Model (test set)            |   AUC |  Gini |    KS |
| --------------------------- | ----: | ----: | ----: |
| Scorecard                   | 0.856 | 0.712 | 0.560 |
| Gradient boosting (untuned) | 0.866 | 0.733 | 0.587 |

- The scorecard's AUC has a 95% bootstrap interval of [0.849, 0.864]. Train AUC is 0.852, so there is
  no overfit.
- Interpretability costs about 0.01 AUC against `HistGradientBoostingClassifier` on the raw features.
- Score PSI, train vs test: 0.0003. Every feature's PSI is below 0.001. The split is random, so this
  proves little about drift; the PSI code is there for monitoring a later population.
- Mean predicted PD is 6.88% against 6.70% observed (Brier 0.050).

![WOE by bin](results/woe.png)

![ROC](results/roc.png)

Full tables (the whole scorecard, calibration, selection log) are in
[`results/report.md`](results/report.md).

## The hard parts

**Sentinel codes look like counts.** `96` and `98` in the three past-due columns are not "98 late
payments" but codes, on 269 rows (about 0.2% of the data) that are 54% bad. Binned as numbers they
would be lumped into a ">13 late" tail bin of a handful of rows and make the bin sequence
non-monotone. Each column declares them as special values, and they get their own bin with a WOE of
about -3.0 (the riskiest bin, as it should be). `test_pipeline.py` checks that the special bin is
riskier than every range bin.

**Missingness is a feature.** 20% of `MonthlyIncome` is missing, and those borrowers are *not* worse
risks (5.5% bad against 6.7% overall), so imputing a median would have hidden a real signal. Missing
gets its own bin, and a value that was never missing in training scores neutral (WOE 0), not an error.
`DebtRatio` is also not a ratio for those borrowers: its median is 1,198 when income is missing and
0.30 when it is present (95% of the missing-income rows exceed 1), so it is partly a dollar amount.
Its IV is 0.019, so the screen drops it.

**Monotone bins that still carry the signal.** Pool-adjacent-violators runs in both directions and
keeps the better fit, then undersized or one-class blocks merge into the neighbour that costs the
least (Ward cost). A 5% minimum bin size collapsed "60-89 days late" into one bin because only 4.9% of
borrowers have ever been late in that band; 2% keeps the signal. The binner is tested on a
U-shaped feature, on both directions, on right-closed boundaries and against a hand-computed WOE/IV.

**Screening that explains itself.** Features are dropped for IV below 0.02, for correlation above 0.7
with a stronger feature, or for a negative coefficient. WOE is already a log-odds ratio, so a
healthy coefficient is positive and near 1; a negative one means a feature is fighting a correlated
one. Here that removed "open credit lines" (IV 0.10, coefficient -0.03) and "real-estate loans" (IV
0.04, coefficient -0.08), and the selection log in the report says why for every feature. The kept
coefficients are 0.25-0.64, well under 1, because the three past-due counters overlap.

**Calibration is good in the middle and loose at the safe end.** The points scale holds (19.9 points
per doubling of the odds, against 20 nominal) and the riskiest half of borrowers is within 10% of the
observed rates. The safest 20% are over-predicted by about 2x (1.05% PD against 0.41% observed in the
best decile). This is the model, not the rounding: a logistic regression that is linear in WOE cannot
get steep enough at the extreme, and the unrounded probabilities show the same gap. It is reported
as is, not patched. Integer points add a small upward bias (mean PD 6.88% against 6.70%).

**One test shows the score is honest.** `test_scorecard.py` checks that every 20 points double the
odds, that integer points stay within half a point per feature of the exact log-odds score, and that
the score is a plain sum of per-bin points. KS is checked against `scipy.stats.ks_2samp`, including
tied integer scores.

## OpenVINO scoring

A scorecard is lookups and a sum, so `src/credit_scorecard/ov_scoring.py` expresses it as an OpenVINO
graph with no trained weights: `Greater` + `ReduceSum` for the bin index, `Equal` for sentinel codes,
`x != x` for missing, and `Gather` + `Add` for the points. Measured on the author's laptop (Core
Ultra 7 255H, Arc 140T iGPU, OpenVINO 2026.4.1), scoring 4 million rows:

| Backend             |        Time | Identical scores |
| ------------------- | ----------: | ---------------: |
| NumPy (float64)     | ~690-810 ms |        reference |
| OpenVINO CPU        |     ~370 ms |             100% |
| OpenVINO GPU (iGPU) | ~160-180 ms |             100% |

Two things this surfaced. The Intel GPU plugin defaults to float16, which put 0.09% of rows in the
wrong bin (off by up to 15 points) because values near a cut point are compared at half precision;
compiling with `INFERENCE_PRECISION_HINT=f32` fixes it, and the tests compare every device against
NumPy. The NPU is not supported: it needs static, float16 graphs. The gradient-boosting benchmark
stays on the CPU, since OpenVINO has no tree-ensemble operator.

Timings depend on the machine and are not asserted in tests; parity is.

## Limitations

- The split is random, not out-of-time: the data has no dates. Real scorecards are validated on a
  later period, and a random split flatters stability.
- No reject inference. The data contains only borrowers who were given credit.
- The tail miscalibration described above is not corrected; a Platt or isotonic step on the score
  would be the usual fix.
- Gradient boosting is untuned; it is a reference point, not a tuned ceiling.
- Default bin settings (8 bins, 2% minimum, IV 0.02, correlation 0.7) were picked on this data's
  training split, with no separate tuning set.
