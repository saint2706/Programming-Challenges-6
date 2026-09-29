# Cohort Retention Analysis Tool

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Feed it a raw event log (`user_id`, `timestamp`, optional `event_name`) and it
builds the cohort retention matrix, weighted retention curves, and one
self-contained HTML report: heatmap, curves, and cohort sizes.

## What it does

- **Cohorts from raw events.** A user's cohort is the day/week/month of their
  first event, or of their first `--signup-event` if you name one. Users who
  never fire the signup event are dropped and counted; events *before* the
  signup are dropped and counted.
- **Two retention definitions (`--mode`).**
  - `active` (default, "classic"): cell `(cohort, N)` is the share of the
    cohort with at least one event *in* period `cohort + N`.
  - `return` ("unbounded"): the share with an event in period `cohort + N`
    *or any later period*. It never increases with `N`, and it counts someone
    who skipped a week and came back.
- **Day / week / month granularity.** Weeks start on Monday (ISO); months are
  calendar months.
- **Right-censoring, not fake zeros.** A cell whose period had not (fully)
  happened when the log ended is *unknown*, so it is masked (grey in the
  heatmap, a gap in the curves), never shown as 0%. The weighted average curve
  at offset `N` only uses cohorts that are observable at `N`, so it doesn't
  sag toward zero because the newest cohorts are young.
- **Timestamps normalized to UTC.** ISO strings with `Z` or numeric offsets are
  converted to UTC; naive timestamps are assumed to be UTC; date-only values
  are midnight UTC. Formats can be mixed within one column.
- **Data-quality report.** Rows with a missing `user_id`, a missing timestamp,
  an unparseable timestamp, or an exact duplicate are dropped and counted (each
  row under the first check it fails, so the counts add up). Empty and
  header-only files produce a report with a notice instead of a crash.
- **One HTML file, no internet needed.** The Plotly bundle is inlined once.

## Design notes

**Which period is "still open"?** The period containing the *latest event in
the log* is treated as incomplete and masked (`--include-partial` counts it).
The tool can't know when the export actually ran, only when the last event
happened, so it errs toward masking. The consequence: the newest cohort's
offset 0 is masked too, though its size still shows in the size chart. Passing
`--include-partial` trades that safety for an extra column of data.

**Bucketing is done in UTC**, not in each user's local zone. A
`2024-01-07T23:30:00-05:00` event is `2024-01-08T04:30Z`, which is Monday, so
it opens a new week. Local-time bucketing needs a per-user timezone that a raw
event log doesn't carry. The parsing is offset-based, so DST transitions are
handled by construction: `01:30-05:00` and `03:30-04:00` on the US spring-forward
day come out exactly one real hour apart (tested).

**The parsed UTC values are stored as *naive* datetimes** internally. That
makes UTC calendar bucketing trivial, and it avoids needing an OS time-zone
database just to hold a "UTC" zone (Windows ships none, and Polars panics when
converting a zone-aware column to Python objects there).

**Why the parser coalesces several explicit formats.** Polars infers one format
from the first value it sees, so a file mixing `...Z`, `...+02:00`, and naive
timestamps would silently null out every row that doesn't match the first.
Five explicit formats are tried and coalesced instead, so anything none of them
matches is counted as unparseable rather than lost silently.

**Cohorts with no users are skipped**, not padded with empty rows.

Polars (not pandas) does the grouping; the tiny cohort x offset matrix is then
assembled in plain Python, where the censoring rules are easiest to read and
test.

## Run it

```bash
cd "challenges/Data Analytics/Cohort Retention Analysis Tool"

uv run python cohort_retention.py sample_data/events.csv --granularity week -o report.html
uv run python cohort_retention.py sample_data/events.csv --granularity week --mode return -o return.html
uv run python cohort_retention.py sample_data/messy_events.csv --signup-event signup -o messy.html

uv run pytest -q # 37 tests
```

Options: `--granularity {day,week,month}`, `--mode {active,return}`,
`--signup-event NAME`, `--include-partial`, and `--user-col` / `--time-col` /
`--event-col` if your headers differ. Exit code `2` on a bad file or missing
column.

## Sample data

- `sample_data/events.csv`: 400 synthetic users signing up over 10 consecutive
  weeks, with a **known** retention curve: after week 0, a user is active in week
  `n` with probability `0.60 * 0.80^(n-1)` (60%, 48%, 38%, ...). The log ends
  after 12 weeks, so recent cohorts are censored. Reproduce it byte for byte
  with `uv run python generate_sample.py` (fixed seed); the tests check that the
  tool recovers the known curve within sampling noise.
- `sample_data/messy_events.csv`: a missing user, a missing timestamp, a garbage
  timestamp, an exact duplicate, mixed timestamp formats and UTC offsets around
  the US DST change, and users with no signup event.

## Where this is actually used

Product and growth teams read cohort retention to tell whether a change made
users stick, independent of how many new users showed up. Comparing cohorts
side by side separates "the product got better" from "we signed up a different
crowd".
