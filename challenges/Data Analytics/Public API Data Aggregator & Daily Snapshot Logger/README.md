# Public API Data Aggregator & Daily Snapshot Logger

**Category:** Data Analytics
**Difficulty:** B

**Status:** Implemented (Python)

Polls [Open-Meteo](https://open-meteo.com/) (free, no API key) once per
invocation, upserts the result into SQLite, and builds a longitudinal
dataset one snapshot at a time. A separate `report` command turns
accumulated snapshots into a self-contained HTML trend chart.

## What it does

- **`poll`** — one HTTP call to Open-Meteo's forecast API for a
  location's current conditions + today's daily min/max/precipitation,
  upserted into `snapshots.db` keyed by `(date, location)`. Running it
  twice on the same day updates the row rather than duplicating it —
  verified by a test that polls twice and checks exactly one row remains
  with the second call's values.
- **`seed`** — loads a JSON file of pre-fetched snapshot rows into the
  database through the same `store_snapshot` path `poll` uses. This is
  how the demo dataset works (see below), and it's also how you'd
  backfill from an export if you started tracking a metric that already
  had history elsewhere.
- **`report`** — reads every stored snapshot, ordered by date, and
  renders a self-contained HTML report (temperature line chart with
  daily min/max band, a precipitation bar chart) — same style as the
  other Data Analytics reports in this repo.
- This is meant to run **once a day**, not as a daemon — see
  `daily-snapshot-workflow.yml.example` for how that'd actually be
  scheduled.

## Design notes

**HTTP client**: `httpx`, not `requests` — same simplicity for a single
synchronous call, more actively developed.

**Tests never touch the real network.** `httpx.get` is monkeypatched with
a fixed fixture response in every test, which is what makes the suite
fast and deterministic instead of flaky against a live API and subject to
its rate limits. That said, this was also manually verified against the
real endpoint while building it (`poll` run twice against the live API,
confirmed the second call updated rather than duplicated the row).

**The demo database has no fabricated numbers.** Rather than making up
plausible-looking values to demonstrate the `report` command without
waiting two weeks for `poll` to accumulate real history,
`sample_data/historical_nyc.json` holds 14 days of **real observed
weather** for New York, pulled once from Open-Meteo's free Historical
Weather API (`archive-api.open-meteo.com` — a companion endpoint to the
forecast API used for `poll`, going back decades, no key required). Each
row's `temperature_c`/`humidity_pct`/`wind_speed_kmh` is that day's actual
noon-local hourly reading (the historical endpoint has no "current
conditions" concept, only hourly and daily aggregates, so noon is used as
the stand-in for what `poll`'s "current" reading would have captured);
`temp_max_c`/`temp_min_c`/`precipitation_mm` are the real daily
aggregates. `seed` seeds a *demo* database from this — it's a separate
file from whatever `snapshots.db` your own `poll` runs build up.

**The GitHub Actions workflow is deliberately inert as shipped.** It's a
`.example` file in this folder, not a `.yml` file under
`.github/workflows/` — so merging this PR cannot cause it to start
running against the real API or committing to this repo. Enabling it is
a manual, explicit copy step (see the file's own header comment).

## Run it

```bash
cd "challenges/Data Analytics/Public API Data Aggregator & Daily Snapshot Logger"

# One real poll against the live API
uv run --with httpx python aggregator.py poll --db snapshots.db

# See the trend immediately using 14 days of real historical data
uv run --with httpx --with polars --with plotly python aggregator.py seed --db snapshots_demo.db --input sample_data/historical_nyc.json
uv run --with httpx --with polars --with plotly python aggregator.py report --db snapshots_demo.db -o demo_report.html

uv run --with pytest --with httpx --with polars --with plotly pytest -q   # 9 tests
```

## Sample data

`sample_data/historical_nyc.json` — 14 real days (2026-08-23 to
2026-09-05) of observed weather for New York from Open-Meteo's Historical
Weather API. Not synthetic.

## Where this is actually used

This is "git scraping" (a term Simon Willison coined for it): point a
scheduled job at an API with no bulk-export or history endpoint, and let
the accumulating commits become the historical dataset. It's how a
surprising number of public interest datasets get built — legislative
voting records, transit on-time performance, COVID dashboards early on —
whenever the source only exposes "right now" and someone needs
"over time".
