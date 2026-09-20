# Multi-Timezone Meeting Scheduler

**Category:** Practical Software
**Difficulty:** B (brief: "Find overlapping working hours across N timezones; DST-aware.")

**Status:** Implemented (Python)

Given a list of people, each in their own IANA timezone with their own
working hours, find when enough of them are simultaneously at work on a
given calendar date — correctly, across a DST transition. Ships as both a
CLI (`cli.py`) and a small web UI (`web.py`); both are thin wrappers around
one shared core, `scheduler.py`.

## What "DST-aware" actually requires

A bare date has no meaning on its own — "let's meet on the 24th" is only
answerable once you pick *whose* the 24th is. This tool treats `--date` as a
calendar day in a **reference timezone** (the organizer's), converts that to
a concrete UTC window (which is 23, 24, or 25 hours long depending on
whether the reference zone's clocks change that day), and then, for every
participant, builds their local working-hours interval as a **timezone-aware
`datetime`** via `zoneinfo`. That's the entire trick: `zoneinfo` resolves the
correct UTC offset for the *exact date in question*, so a fixed local 9 AM
is 14:00 UTC on one side of a DST boundary and 13:00 UTC on the other,
automatically — no manual offset tables, no special-casing the transition
day. Two tests construct this directly: `test_dst_spring_forward_shifts_utc_offset`
and `test_dst_fall_back_shifts_utc_offset_back` compare the same participant
on either side of the US 2026 DST boundaries and assert the UTC hour shifts
by exactly one.

## Design

- **`scheduler.py`** — the entire domain model and algorithm, with no CLI or
  web dependency:
  - `Participant(name, tz, work_start, work_end, days=Mon-Fri)` validates
    eagerly: an unknown IANA zone name raises immediately (via `ZoneInfo`),
    and `work_start >= work_end` is rejected with a message naming overnight
    shifts as the explicit unsupported case (see below), rather than
    silently producing an empty or wrapped-around interval.
  - `find_overlaps` builds each participant's UTC working interval(s) for the
    requested day (with a one-day local buffer on each side, since a
    participant's local midnight rarely lines up with the reference
    timezone's), then runs a standard sweep-line over `+1`/`-1` coverage
    events to find where at least `min_participants` overlap.
  - **Segments split whenever the *available set* changes, not just when the
    count crosses the threshold.** If A works 9-17, B works 9-13, and C
    works 13-17, the naive "count >= 1" view sees one continuous 9-17
    stretch — but who's actually available changes at 13:00, so reporting
    that as a single window would be misleading. `find_overlaps` only merges
    two adjacent segments when their participant sets are *identical*,
    covered by `test_segments_split_on_membership_change_not_just_threshold`.
  - `parse_participant_spec("Name|Area/City|HH:MM-HH:MM[|days]")` — pipe-delimited
    because both IANA zone names (`/`) and hour ranges (`:`) already use the
    more "obvious" separators.
  - `search_timezones(query)` — substring search over `zoneinfo.available_timezones()`,
    since nobody remembers the exact IANA spelling on the first try.
- **`cli.py`** (Typer + Rich) — `find` prints a table of overlap windows
  localized to `--display-tz`; `zones` searches IANA names.
- **`web.py`** (FastAPI) — one form, one results table, reusing
  `find_overlaps`/`parse_participant_spec` directly. Rendered as plain
  f-string HTML (no template directory) to stay a self-contained script.

## What it deliberately doesn't do

- **Overnight/cross-midnight shifts are rejected outright**
  (`work_start >= work_end` raises `ValueError` naming the limitation).
  Modeling a shift correctly requires deciding which calendar day it
  "belongs to" and how it interacts with `days`, which is a genuinely
  different feature, not a corner case of this one.
- **No recurring/weekly view** — it answers "does this specific date work,"
  not "what's our standing weekly slot." Run it on a few candidate dates
  near a DST boundary if that matters to you; that's exactly what the DST
  tests do.

## Usage

```bash
cd "challenges/Practical Software/Multi-Timezone Meeting Scheduler"

# CLI
uv run --with typer --with rich --with tzdata python cli.py find \
    --date 2026-09-24 --ref-tz America/New_York --display-tz America/New_York \
    -p "Asha|Asia/Kolkata|09:00-18:00" \
    -p "Ben|Europe/London|09:00-17:00" \
    -p "Cara|America/New_York|09:00-17:00" \
    --min-participants 2

uv run --with typer --with rich --with tzdata python cli.py zones kolkata

# Web UI
uv run --with fastapi --with "uvicorn[standard]" --with python-multipart --with tzdata python web.py
# -> http://127.0.0.1:8001

uv run --with pytest --with tzdata --with typer --with rich \
    --with fastapi --with "uvicorn[standard]" --with python-multipart --with httpx2 pytest -q   # 33 tests
```

**Windows note:** the standard library's `zoneinfo` relies on the OS having
an IANA tz database installed; Windows doesn't ship one, so the `tzdata`
PyPI package (a pure-data fallback `zoneinfo` finds automatically) is a
required dependency here, not optional — every command above includes
`--with tzdata`. On Linux/macOS with a system tz database it's a no-op if
included and unnecessary if omitted.

A worked example: Asha (Kolkata, 9-6), Ben (London, 9-5), and Cara (New
York, 9-5) have **no** 3-way overlap on 2026-09-24 under normal business
hours — a genuine real-world scheduling dead end, not a bug. Adding
`--min-participants 2` shows the actual answer: Asha+Ben overlap for 4.5
hours in the (India) morning, and Ben+Cara overlap for 3 hours in the
(UK/US) afternoon, with no time that works for all three.

## Tests

33 pytest cases across three files. `test_scheduler.py` (19) covers full and
partial overlap across real timezones, no-overlap, `min_participants`
filtering, day-of-week exclusion, both 2026 US DST transitions (spring
forward and fall back) shifting the computed UTC offset by exactly one hour,
overnight-shift rejection, unknown-timezone rejection, duplicate-name
rejection, out-of-range `min_participants`, the available-set segment
splitting described above, display-timezone formatting, and the spec parser
(valid, with custom days, and two invalid-shape cases). `test_cli.py` (7)
and `test_web.py` (7) drive the CLI (`CliRunner`) and web app (`TestClient`)
end-to-end through the same scenarios. A manual smoke test additionally
exercised the live CLI and a running `uvicorn` server over real HTTP.

The suite runs warning-free: Starlette's `TestClient` (used by
`test_web.py`) prefers the `httpx2` package over `httpx` (hence it's in the
test command above), and `pytest.ini` filters one remaining warning that
Starlette 1.6.0's own `testclient.py` emits at import time against anyio
4.15 or newer (a deprecated `anyio.abc.BlockingPortal` alias reference) — an
unfixed upstream version interaction, not something triggered by this suite.
