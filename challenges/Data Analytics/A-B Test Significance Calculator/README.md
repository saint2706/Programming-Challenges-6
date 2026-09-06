# A/B Test Significance Calculator

**Category:** Data Analytics
**Difficulty:** B

**Status:** Implemented (Python)

Chi-square for conversion-rate tests, Welch's t-test for continuous metrics,
picked automatically from the data — with a plain-English verdict, not just
a p-value.

## What it does

- **Two input modes.** Point it at a raw per-row CSV (a group column plus an
  outcome column), or skip the CSV entirely and pass already-computed
  summary stats via flags (`--control-conversions`/`--control-total`/... for
  proportions, `--control-mean`/`--control-std`/`--control-n`/... for
  means) — for when you only have aggregated numbers from a dashboard
  export. The two modes are mutually exclusive; mixing them is a usage
  error, not a silent guess.
- **Auto-detects which test to run.** An outcome column with exactly two
  distinct values (0/1, true/false, yes/no, converted/not) is treated as a
  conversion outcome and gets `scipy.stats.chi2_contingency` on the 2×2
  table. Anything else numeric is treated as a continuous metric and gets
  Welch's t-test (`equal_var=False` — the safer default when you haven't
  verified the two groups have equal variance).
- **Effect size, not just significance.** Chi-square and Welch's t alone
  only answer "is there a difference". This also reports the actual
  difference with a 95% confidence interval (Wald CI for proportions,
  Welch–Satterthwaite CI for means) and relative lift / Cohen's d — the
  numbers that answer "how much, and how confident should I be in that
  number".
- **Plain-English verdict**, always printed, e.g.:
  > variant's conversion rate (15.80%, 158/1000) vs control's (13.00%,
  > 130/1000): an absolute difference of +2.80pp (+21.5% relative), 95% CI
  > [-0.27pp, 5.87pp]. This is not statistically significant at
  > alpha=0.05 (chi-square p=0.0745): there isn't enough evidence to
  > conclude a real difference exists -- consider a larger sample size.
- **Optional HTML report** (`-o report.html`): the same verdict plus a bar
  chart with confidence-interval error bars, self-contained like the other
  Data Analytics reports in this repo.

## Design notes

The group column (`group`/`variant`/`arm`/...) is broadly guessable from a
fixed alias list because there's a small, conventional set of names for it.
The *outcome* column generally isn't — `load_time_seconds`,
`order_value_usd`, `nps_score` are all reasonable real column names a fixed
alias list can't anticipate, so `--outcome-col` exists for exactly that
case rather than the tool guessing wrong silently.

## Run it

```bash
cd "challenges/Data Analytics/A-B Test Significance Calculator"

# Raw CSV, auto-detected as a proportions test
uv run --with polars --with plotly --with scipy --with numpy python ab_test_calculator.py sample_data/checkout_conversion.csv -o conversion_report.html

# Raw CSV with an explicit outcome column, auto-detected as a means test
uv run --with polars --with plotly --with scipy --with numpy python ab_test_calculator.py sample_data/page_load_time.csv --outcome-col load_time_seconds

# Summary stats only, no CSV
uv run --with polars --with plotly --with scipy --with numpy python ab_test_calculator.py \
  --control-conversions 120 --control-total 1000 --variant-conversions 145 --variant-total 1000

uv run --with pytest --with polars --with plotly --with scipy --with numpy pytest -q   # 21 tests
```

## Sample data

- `sample_data/checkout_conversion.csv` — 1000 users per group, binary
  `converted` outcome. Random sampling landed this one on the "not
  significant" side (p=0.07) despite the underlying rates being 12% vs
  14.5% — a realistic reminder that a real lift can still fail to clear
  significance at a given sample size.
- `sample_data/page_load_time.csv` — 300 users per group, continuous
  `load_time_seconds` outcome, landed clearly significant (p<0.0001).

Both are synthetic.

## Where this is actually used

Every experimentation platform (Optimizely, GrowthBook, Statsig, a
homegrown "experiments" dashboard) runs exactly this calculation under the
hood for its results page — the two-proportion test for conversion-rate
experiments and Welch's t-test for continuous-metric experiments are the
textbook defaults nearly every one of them ships.
