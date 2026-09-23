# Personal Subscription Tracker with Renewal Alerts

**Category:** Practical Software
**Difficulty:** B (brief: "Track recurring costs, notify before renewal, spending summary.")

**Status:** Implemented (Python)

Track every subscription you're actually paying for — cost, billing cycle,
category, next renewal date — see a live spending summary, and get a real
OS desktop notification before each one renews. One process, one SQLite
file, no external services.

## Two things had to be airtight

**Billing-cycle date math.** A monthly subscription renewing on the 31st has
to land on Feb 28 (or 29 in a leap year), not crash or silently roll into
March, and a subscription nobody has touched for several missed cycles has
to catch all the way back up to the future in a single pass, not advance one
step and get stuck in the past again on the next check. `add_months` clamps
the day to the target month's last valid day; `roll_forward` loops
`compute_next_renewal` until the result is strictly after "today," which
terminates because every billing cycle adds at least one day. Covered by
`test_add_months_clamps_short_month`, `test_add_months_clamps_into_leap_february`,
and `test_roll_forward_catches_up_multiple_missed_cycles`.

**Don't nag.** A renewal inside its alert window must fire exactly one
notification, not one per background-thread tick (the checker also runs
on-demand via the dashboard's "run check now" button, so it has to be safe
to call as often as you like). `NotificationLog` rows are keyed on
`(subscription_id, renewal_date)`; `due_for_notification` excludes any
subscription that already has a log row for its *current* `next_renewal`.
Rolling the renewal date forward to the next cycle naturally makes it
notifiable again later. Covered by
`test_due_for_notification_is_idempotent_per_renewal_date` and
`test_run_check_now_is_idempotent`.

## Design

- **`storage.py`** — `add_months` / `compute_next_renewal` / `roll_forward` /
  `monthly_equivalent_cents` are pure functions with zero framework or I/O
  dependency, so the date math and spending-summary math are unit-tested
  directly without spinning up a database. `SubscriptionStore` wraps
  SQLModel (typed models over SQLAlchemy + Pydantic — the FastAPI author's
  own ORM, and a deliberate upgrade over hand-written SQL here) for CRUD,
  rollover, notification tracking, and the spending summary.
- **`notifier.py`** — a two-line wrapper around `plyer.notification.notify`.
  Isolated into its own module so tests can monkeypatch it without a real
  display/notification backend, and so a missing backend (headless box, no
  notification daemon) degrades to a logged warning instead of crashing the
  background scheduler thread.
- **`app.py`** — FastAPI routes, HTML rendering (plain f-string fragments,
  no template directory, per this repo's convention), and a daemon thread
  (started/stopped via `lifespan`, same pattern as the paste bin challenge)
  that calls `run_check_cycle()` hourly. HTMX is used for exactly one thing:
  the "run check now" / "send test notification" buttons swap their result
  into the page without a full reload — everything else is plain forms with
  a 303 redirect.
- **Money is integer cents (`cost_cents: int`), never a float.** Avoids the
  usual `0.1 + 0.2` accumulation problem in the spending summary, which sums
  across every active subscription on every dashboard render.
- **Every user-supplied string is escaped at render time, not input time**
  (`html.escape`), and the delete button's confirmation dialog reads the
  subscription name from a `data-name="..."` attribute at runtime
  (`this.dataset.name`) instead of interpolating it into a JS string literal
  — the latter is escapable-looking but still breakable (a name like
  `x'); alert(1); //` closes the string and injects code once the HTML
  entities are decoded, before the browser's JS parser ever sees it).
  Covered by `test_subscription_name_cannot_break_out_of_delete_confirm_js_string`
  and two more adversarial-payload tests. Currency codes are additionally
  validated server-side (`[A-Za-z]{1,8}`) before they ever reach storage.

## What it deliberately doesn't do

- No accounts/multi-user support, no currency conversion (each subscription
  keeps its own currency code; the spending summary just adds the numbers,
  which is only meaningful if you track everything in one currency) — this
  is a personal single-user tool, not a budgeting SaaS.
- No email delivery — desktop notifications only, per this repo's
  convention of keeping each challenge a single self-contained script with
  no external service credentials to configure.
- No historical spend chart across past months — the summary is a live
  snapshot of currently-active subscriptions, not a time series.

## Usage

```bash
cd "challenges/Practical Software/Personal Subscription Tracker with Renewal Alerts"

uv run --with fastapi --with "uvicorn[standard]" --with sqlmodel --with plyer \
    --with python-multipart python app.py
# -> http://127.0.0.1:8002

# or, with auto-reload during development:
uv run --with fastapi --with "uvicorn[standard]" --with sqlmodel --with plyer \
    --with python-multipart uvicorn app:app --reload

uv run --with fastapi --with "uvicorn[standard]" --with sqlmodel --with plyer \
    --with python-multipart --with httpx2 --with pytest pytest -q   # 52 tests
```

Open `http://127.0.0.1:8002`, add a subscription (name, cost, currency,
category, billing cycle, next renewal date, alert lead time). The dashboard
shows upcoming renewals sorted soonest-first with red/amber/green urgency,
plus a spending summary (monthly and annual totals, broken down by
category, using each subscription's billing-cycle-normalized
monthly-equivalent cost). A background thread checks for due renewals every
hour and fires a real desktop notification; use the "Run renewal check now"
button to trigger and verify that immediately instead of waiting, or "Send
test notification" to fire an unconditional test ping. `GET /health` reports
`{"status": "ok", "subscriptions": <count>}`.

**Live-verified on this Windows machine:** created a subscription due
tomorrow, hit "run check now," and confirmed (via the response, the
notification-history log, and the absence of any "Desktop notification
failed" warning in the server log) that `plyer` actually delivered a real
OS toast notification, not just a logged fallback.

## Tests

52 pytest cases across two files. `test_storage.py` (24) covers the pure
date math described above (month-length clamping, leap years, year
rollover, multi-cycle catch-up), `monthly_equivalent_cents` across all five
billing cycles including a custom interval, and the store's CRUD, rollover,
idempotent-notification, and spending-summary-by-category logic.
`test_app.py` (28) drives the real FastAPI app through `TestClient`:
CRUD + dashboard rendering, empty-name and unknown-billing-cycle rejection,
cancel vs. delete, the spending summary reflecting only active
subscriptions, both notification endpoints (with the real OS notification
call mocked so the automated suite never pops a real toast), and the
adversarial-input suite (a `<script>` payload, a JS-string-breakout payload
targeting the delete confirmation, and an `<img onerror>` payload inside a
notification message — each asserted to render HTML-escaped, never
verbatim) plus server-side currency-code validation.

The suite runs warning-free: Starlette's `TestClient` prefers the `httpx2`
package over `httpx` (hence it's in the test command above), and
`pytest.ini` filters one remaining warning that Starlette 1.6.0's own
`testclient.py` emits at import time against anyio 4.15 or newer (a
reference to a deprecated `anyio.abc.BlockingPortal` alias) — an unfixed
upstream Starlette/anyio version interaction that nothing in this test
suite triggers or can work around.
