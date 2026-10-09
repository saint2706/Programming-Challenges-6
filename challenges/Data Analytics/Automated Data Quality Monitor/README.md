# Automated Data Quality Monitor

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Source modules live in `src/data_quality/`; the tests are in `tests/`.

Schema-drift and anomaly alerts on incoming CSV batches. `profile` learns what
"normal" looks like from known-good batches and freezes it in a small JSON
baseline; `check` compares one new batch against it, prints the alerts, writes
an optional HTML + JSON report, and exits non-zero so a pipeline can stop.

## What it checks

| Family       | Alert ids                                                                                              | What fires it                                                                                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Schema       | `schema.missing_column`, `new_column`, `possible_rename`, `type_change`, `column_order`                | Columns added/removed; a missing column paired with a same-typed new one of a similar name or position (reported once, as a rename); dtype changes; reordering.     |
| Volume       | `volume.empty`, `volume.row_count`                                                                     | Row count far outside the baseline batches' own spread, **and** at least 30% away from their mean.                                                                  |
| Nulls        | `nulls.all_null`, `nulls.rate_shift`                                                                   | Exact binomial test against the baseline null rate, **and** at least 2 percentage points. A fall in null rate is only `info`.                                       |
| Values       | `values.cast_failures`, `values.out_of_range`, `values.unseen_categories`, `values.missing_categories` | Values that can't be read as the baseline's type; numbers outside the learned range; categories never seen (net of the column's normal long tail) or suddenly gone. |
| Drift        | `drift.numeric`, `drift.categorical`                                                                   | Numeric: KS test **and** PSI **and** KS effect size. Categorical: chi-square **and** Jensen-Shannon divergence.                                                     |
| Keys, format | `keys.duplicates`, `format.unexpected_shape`                                                           | A column that was a unique key now repeats; text values whose character-class shape (`TX-004217` -> `A{2}-9{6}`) was never seen.                                    |
| Freshness    | `freshness.implausible`, `freshness.span`, `freshness.stale`, `freshness.future`                       | Timestamps years outside the baseline; a batch covering a different length of time; with `--as-of`, a newest value too old or after the as-of time.                 |

Each alert is `info`, `warn` or `critical`. All thresholds live in one TOML
file ([`dq_config.example.toml`](dq_config.example.toml) lists every key with
its default). You can also pin any check to a fixed severity or turn it off, and
skip columns. Unknown sections, keys and check names are errors, so a typo can't
silently disable a check.

## Design notes

**Significance is not enough, and neither is effect size.** Every statistical
check is a two-key lock. With enough rows the KS test flags differences nobody
cares about; with few rows PSI (Population Stability Index) is inflated by pure
sampling noise. So a numeric drift alert needs a small p-value (KS) **and** a
KS distance of at least 0.10 **and** PSI of at least 0.10, and it goes
`critical` at PSI 0.25. Those PSI cut-offs are the credit-scoring rule of
thumb (under 0.1 stable, 0.1-0.25 moderate shift, over 0.25 significant;
Siddiqi, *Credit Risk Scorecards*, 2006). It's a convention, not a theorem, so
they're configurable. The tests prove the point on real data: shifting `fare`
by +10% across ~2,000 rows gives a KS p-value near 1e-15 and **no alert**.

**KS against a stored quantile grid, with the true sample size.** The baseline
keeps 501 quantiles per numeric column (not the raw values), and `check` runs
`ks_2samp` against that grid. The p-value is computed from the baseline's *real*
row count, not the grid's 501, so the baseline's size still counts.

**Volume and nulls have a noise floor.** A steady source still varies by about
`sqrt(mean)` rows, so the volume z-score's scale is
`max(std * sqrt(1 + 1/k), sqrt(mean))`; a one-batch baseline still works. Null
rates are smoothed so a baseline with zero nulls doesn't make the very first
null infinitely surprising.

**Long-tailed categories are not drift.** A zone-name column keeps producing
never-seen values forever. The baseline stores the Good-Turing estimate of that
tail (categories seen exactly once / rows; Good, 1953), and
`values.unseen_categories` only fires on the *excess* over it, with a binomial
test. With the tail term switched off, 6 of the 10 real held-out days raise an
`unseen_categories` warning.
Categories too rare to have 5 expected rows in the batch are pooled into one
"other" bin before the chi-square and Jensen-Shannon (Lin, 1991) checks.

**Keys are inferred, and you can override.** An `int` or `string` column with no
duplicate inside any baseline batch or across all of them, over at least 20
values and at least 2 batches, is treated as a key. Keys skip the distribution
and range checks (an ever-growing sequence would look like drift forever) and
get the uniqueness and format checks instead. `--key COL` forces it,
`--no-auto-keys` disables the guess, and a single-batch baseline never guesses.
Duplicates *across* batches aren't checked: that would mean storing every key.

**Types are the baseline's, not the batch's.** A batch column is cast to the
baseline's logical type. Whatever fails to cast is a `values.cast_failures`
alert with the share of bad values, on top of the `schema.type_change` that a
column of text produces. NaN counts as null everywhere. Only the empty string is
a null token on read; `N/A` is text, so it *is* caught.

**A renamed column is paired, not double-reported.** The old name isn't also
"missing" and the new one isn't also "new". Value-level checks are skipped for a
renamed column, since they'd be comparing it under a name the baseline never saw.

**Freshness needs a reference time.** Without `--as-of`, only the checks that
don't need one run: implausible timestamps (default: more than 365 days outside
the baseline) and the batch's time span vs the baseline's typical span. With
`--as-of`, `stale` (default limit: 3x the typical span, or
`max_lag_seconds`) and `future` join in. Timestamps are naive; a timezone offset
in the CSV isn't parsed.

**Reports treat the CSV as hostile.** Column names, category values, batch names
and example values all come from files you don't control. Everything is passed
through `html.escape`, Plotly embeds its own JSON with `<` escaped, and a test
feeds `</script><script>alert(1)</script>` as a column name and
`<img src=x onerror=alert(1)>` as a category value.

## Run it

```bash
cd "challenges/Data Analytics/Automated Data Quality Monitor"

uv run python -m data_quality.make_sample     # writes sample_data/batches/{train,incoming,drifted}/

# 1. learn a baseline from 21 known-good daily batches
uv run data-quality profile sample_data/batches/train -o out/baseline.json

# 2. check a clean real day -> status OK, exit 0
uv run data-quality check sample_data/batches/incoming/taxi_2019-03-25.csv -b out/baseline.json --as-of 2019-03-26T06:00:00

# 3. check a corrupted one (every fare x1.5) -> critical, exit 1, with an HTML + JSON report
uv run data-quality check sample_data/batches/drifted/taxi_2019-03-22__fare_x1.5.csv -b out/baseline.json --as-of 2019-03-23T06:00:00 --html out/report.html --json out/report.json

uv run python -m data_quality.evaluate        # false-positive and detection rates (about 20 s)
uv run pytest -q                 # 125 tests, about 35 s
```

`profile` takes files, directories (all `*.csv` inside) or globs, plus `-c
config.toml`, `--key COL` and `--no-auto-keys`. `check` takes `-c config.toml`,
`--as-of ISO-TIME`, `--html PATH`, `--json PATH` and `--fail-on
{info,warn,critical}` (default `warn`).

Exit codes: `0` clean (or only alerts below `--fail-on`), `1` alerts at or above
it, `2` unreadable batch/baseline/config. `sample_data/batches/` and `out/` are
generated and git-ignored.

## Does it cry wolf? Does it catch things?

`evaluate.py`, seed 0. Baseline: real taxi days 1-21 (4,465 rows). Clean
batches: the 10 real held-out days plus 300 simulated ones (150-260 random
held-out rows, re-dated onto one day so each is a one-day slice like a real
delivery). Every check ran with all defaults and an as-of time six hours after
the newest trip.

**False positives (warn or worse on a clean batch):** 0 of 10 real days, 5 of
300 simulated, **1.6% overall** (Wilson 95% interval for the simulated ones:
about 0.7% to 3.8%). That's roughly what a per-test alpha of 0.001 across ~40
tests per batch predicts, and the five were two unseen-pickup-zone
batches (about 3.5% of rows in new zones against a 0.7% tail), one `distance`
distribution shift at p = 7e-4, one null-rate blip in two correlated columns
(`dropoff_zone` and `dropoff_borough`, 3.3% vs 0.7%), and one chi-square on the
pickup-zone mix. There's no free lunch: tighten alpha
to trade some of the detection floor below for fewer of these.

**Detection over 30 planted trials each:** 100% for every one of 22 corruptions
(dropped/added/renamed column, reordered columns as `info`, text in a numeric
column, empty batch, 10% of the rows, the batch delivered four times, 15% nulls,
an all-null column, `fare` x1.5, negative fares, a 1000x unit error, cash trips
relabelled, an unseen payment type, a vanished payment type, duplicated keys,
lowercased keys, epoch-zero dates, timestamps 30 days ahead, a 10-day-old batch,
a 5-day spread). The interesting part is the floor:

| Planted change            | 1st step  | 2nd       | 3rd         | 4th        | 5th        |
| ------------------------- | --------- | --------- | ----------- | ---------- | ---------- |
| `fare` scaled by          | x1.05: 0% | x1.1: 13% | x1.25: 100% | x1.5: 100% | x2: 100%   |
| `fare` nulls              | 0.5%: 0%  | 1%: 0%    | 2%: 47%     | 5%: 100%   | 15%: 100%  |
| cash trips -> credit card | 10%: 0%   | 25%: 0%   | 50%: 50%    | 75%: 100%  | 100%: 100% |

A 5-10% drift or a 1% null rate on ~200-row batches is inside normal
batch-to-batch noise, and the monitor stays silent on purpose. Getting those
would need bigger batches or a longer look-back, not a lower threshold.

## Data

`sample_data/taxis.csv` is the 6,433-row NYC taxi sample from
[mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data), a slice of the
[NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
for March 2019 (yellow and green cabs). It is real: real nulls (0.4-0.7% in
`payment`, zone and borough columns), heavy tails (fares up to $150), a real
weekday/weekend rhythm in the daily row counts (149 to 260). It's cut into
daily batches by pickup date (the single Feb-28 trip is folded into March 1).
One column is added, `trip_id` (`TX-000001`...), since the source has no key
and uniqueness/format checks need one.

Days 1-21 train the baseline; days 22-31 are held out as clean incoming
batches. Corruptions live in [`corruptions.py`](src/data_quality/corruptions.py), each a pure
function with the checks it should trigger, so the ground truth is planted and
known rather than eyeballed.

## Where this is actually used

Data teams put a check like this at the door of a pipeline: schema and volume
gates in the style of Great Expectations or Deequ, and PSI / KS drift monitors
on model inputs. The unglamorous failures it exists for are a vendor renaming a
column, a unit change, a half-delivered file, or an upstream join that quietly
starts producing nulls.
