# Web Traffic Funnel Analyzer

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Feed it a raw clickstream (`user`, `timestamp`, `event name`, optionally a session id and a segment column) and
it shows where visitors drop out, stage by stage: counts, conversion with confidence intervals, how long each
step takes, and whether conversion differs between segments. Output is one self-contained HTML report (Plotly
inlined, no internet needed) plus optional JSON.

## What it does

- **Three funnel semantics (`--mode`).**
  - `ordered` (default): the steps happen in order; other events may happen in between.
  - `strict`: the steps are *consecutive* events, nothing in between.
  - `any-order`: the first N steps all happen, in any order, within the time limit.
    Strict is contained in ordered, which is contained in any-order, and a test checks that on real data.
- **Sessions or users (`--unit`).** With a session id column (`--session-col`) it uses it. Without one it
  sessionizes: a user's events are cut into sessions wherever two neighbours are more than 30 minutes apart
  (`--session-gap`); a gap of exactly 30 minutes stays inside the session.
- **A conversion window (`--window 30m|24h|7d`).** Every step must happen within the window *of the entry*.
- **Per-step statistics.** Entities reaching each step, % of entry and % of the previous step (both with 95%
  Wilson intervals), entities dropped, median and quartile time since the previous step, and the single biggest
  leak with its share of everything lost end to end.
- **Segment breakdown (`--segment-col`, `--segment-level N`, `--top-segments K`).** Conversion to a chosen step
  per segment with Wilson intervals, lift versus all other segments, a Fisher exact test of each segment against
  the rest with Benjamini-Hochberg correction, and a chi-square test across all segments. Segments below
  `--min-segment-size` (30) or outside the top K are merged into `(other)` rather than tested on tiny counts.
  A visit belongs to the segment on its first step-1 event.
- **`a|b` steps.** `--steps view,cart|wishlist,purchase` lets either event count for the second step.
- **Right-censoring (`--drop-censored`).** With a window, an entity that entered less than one window before the
  end of the log can't be judged yet; this drops them (from every step, so the rates aren't biased upward) and says
  how many.
- **Data-quality report.** Rows with a missing user, timestamp or event name, unparseable timestamps, missing
  session ids and exact duplicates are dropped and counted (each row under the first check it fails, so the counts
  add up). Rows that arrive out of time order are kept, re-sorted and counted. A step name that matches no event
  (a typo) is called out in the report and on stderr.
- **Timestamps in the formats real exports use:** ISO with `Z` or offsets, `2019-11-01 00:00:00 UTC`, date-only,
  and bare epoch seconds or milliseconds, mixed freely within one column.

## Design notes

**Every step-1 event starts an attempt, not just the first.** With a time window, a later start can succeed where
the first one timed out (`view@0s, view@100s, cart@110s` with a 20 s window converts, from the second view). Within
one attempt, taking the earliest matching event for each step is optimal because it leaves the most room. With no
window a later start can never get deeper than the first, so it stops after one. The timing reported for an
entity comes from the earliest attempt that reaches its maximum depth.

**Counts are per entity, never per event.** Fifty repeated views followed by five cart clicks is one session that
reached the cart. Nothing a user does over and over can inflate a stage.

**Ties and ordering.** Events are ordered by timestamp, then by row position in the file. Two events in the same
second therefore keep the order they were written in (`cart, view` at the same second does not satisfy
`view -> cart` in ordered mode, but does in any-order mode).

**Correctness is checked against brute force.** The per-entity matchers are compared with oracles that enumerate
every possible combination of events (`itertools.combinations`/`product`) on thousands of random entities, for all
three modes, with and without windows. The end-to-end counts are compared with an oracle that also does its own
grouping, sorting and sessionization in plain Python. Wilson intervals are checked against `scipy`'s implementation.

**Censoring assumes one contiguous log.** "End of the log" is the latest timestamp in the file. The bundled sample
is spliced from eight separate slices (below), so `--drop-censored` and `--unit user` with a window are only
illustrative on it.

**Any-order reports a span, not step gaps.** With no order there is no "time since the previous step", so any-order
reports the shortest time in which the first N steps all happened.

Polars does the reading, cleaning, sessionizing and grouping; the per-entity matching is plain Python, because the
attempt logic is much easier to read and test there.

## Run it

```bash
cd "challenges/Data Analytics/Web Traffic Funnel Analyzer"

uv run python funnel.py sample_data/rees46_sample.csv.gz --session-col user_session --segment-col category_code --segment-level 1 -o report.html
uv run python funnel.py sample_data/rees46_sample.csv.gz -o gap_sessions.html
uv run python funnel.py sample_data/rees46_sample.csv.gz --session-col user_session --mode strict -o strict.html
uv run python funnel.py sample_data/rees46_sample.csv.gz --unit user --window 24h --drop-censored --json result.json -o users.html

uv run pytest -q # 90 tests
```

Options: `--steps`, `--mode {ordered,strict,any-order}`, `--unit {session,user}`, `--window`, `--session-gap`,
`--session-col`, `--segment-col`, `--segment-level`, `--top-segments`, `--min-segment-size`, `--outcome-step`,
`--drop-censored`, `--json`, and `--user-col` / `--time-col` / `--event-col` if your headers differ from the
defaults (`user_id`, `event_time`, `event_type`). Exit code `2` on a bad file, column, step list or duration.
For the Retailrocket dataset the column flags would be
`--user-col visitorid --time-col timestamp --event-col event --steps view,addtocart,transaction`
(epoch milliseconds are parsed; I have not run that file itself).

## Sample data

Real traffic from the **REES46 "eCommerce behavior data from multi-category store"** dataset (an online
electronics/appliances/apparel shop, October-November 2019), not synthetic.

- **Fetching without 14.7 GB.** The full CSV is sorted by time, so `fetch_data.py` uses HTTP Range requests to pull
  eight 8 MB slices spread across the file (different days and hours), trimmed to whole lines, into
  `data_cache/events_raw.csv` (~500k events, git-ignored). `uv run python fetch_data.py` reproduces it.
- **The committed sample** (`sample_data/rees46_sample.csv.gz`, 3.7 MB, 123,141 events, ~30k sessions) keeps 1 in 4
  *sessions* by a stable hash of `user_session`, so every kept session is complete.
  `uv run python fetch_data.py --skip-download --make-sample` rebuilds it byte for byte.
- **Caveats.** Each slice cuts sessions at its edges (a session straddling a boundary loses its tail, which
  slightly undercounts purchases), and `first_event`/`last_event` and the "out-of-order rows" count in the report
  reflect several slices being joined, not real disorder.
- **Source and terms.** REES46 Marketing Platform, published on Kaggle
  ([mkechinov/ecommerce-behavior-data-from-multi-category-store](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store)),
  mirrored on
  [Hugging Face](https://huggingface.co/datasets/kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019)
  (the mirror I download from, since Kaggle needs a login). The dataset card says it is free to use for research,
  books and educational material with attribution; Kaggle's listing may carry a stricter licence (I could not confirm
  which), so check it before reusing the sample outside this portfolio. Thanks to REES46 for publishing it.

## What the real data says

On the committed sample (`view -> cart -> purchase`, native session ids):

| Step     | Sessions | % of entry (95% CI)  | % of previous (95% CI) | Median since previous |
| -------- | -------- | -------------------- | ---------------------- | --------------------- |
| view     | 30,076   | 100%                 | -                      | -                     |
| cart     | 2,454    | 8.16%                | 8.16%                  | 57 s                  |
| purchase | 864      | 2.87% (2.69 to 3.07) | 35.2% (33.3 to 37.1)   | 88 s                  |

- **The leak is at the top.** View to cart loses 27,622 sessions, 95% of everything lost end to end. Cart to
  purchase loses 1,590 more, but 35% of carts convert, which is high.
- **The cart step undercounts purchases by about a third.** Dropping the cart requirement (`--steps view,purchase`)
  gives 1,322 purchasing sessions against 864 through a cart: roughly 35% of sessions that bought never fired a
  `cart` event first. A funnel with a mandatory cart step would report a conversion about a third too low.
- **The three modes genuinely differ.** Purchases reached: strict 780, ordered 864, any-order 873. Strict loses 84
  sessions that browsed something else between the cart and the purchase; any-order gains 9 that added to the cart
  *before* their first view event.
- **Session definition moves the denominator.** The dataset's own session ids give 30,142 sessions; cutting by a
  30-minute gap gives 28,606, and the conversion rates move with them (2.87% versus 3.00% to purchase). "Conversion"
  is only comparable between two reports when both used the same session definition.
- **Category matters a lot.** Conversion to purchase differs across top-level categories (chi-square p = 1e-35):
  electronics 4.36% (2.37x the rest), appliances 2.87% (indistinguishable from the rest, q = 1), computers 1.60%,
  apparel 1.25%, kids 0.23% (1 purchase in 434 sessions). Sessions whose first view has no category convert at 1.80%. Brand shows the same shape: Samsung 5.99% against Apple 4.67%.

## Where this is actually used

Product and growth teams read funnels to decide which step to fix first, and the honest version of that read needs
the three things this tool insists on: an explicit definition of a session, an interval on every rate so a small
segment isn't mistaken for a trend, and a check that the tracking itself isn't dropping the step you're measuring.
