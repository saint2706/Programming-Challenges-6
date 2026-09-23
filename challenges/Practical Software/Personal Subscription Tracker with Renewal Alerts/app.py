"""Personal subscription tracker: renewal math, spending summary, and real
desktop notifications before a subscription renews.

Run directly:
    uv run --with fastapi --with uvicorn[standard] --with sqlmodel --with plyer \\
        --with python-multipart python app.py

Or with a reload server:
    uv run --with fastapi --with uvicorn[standard] --with sqlmodel --with plyer \\
        --with python-multipart uvicorn app:app --reload

Pages are rendered as small server-side HTML fragments (no template engine,
no static file directory) to keep this a single self-contained script, per
this repo's convention. HTMX is used for the one place a live action result
matters without a full page reload: the "send test notification" and "run
renewal check now" buttons on the dashboard swap in their result inline.
"""

from __future__ import annotations

import html
import logging
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from notifier import notify
from storage import BILLING_CYCLES, Subscription, SubscriptionStore, local_today

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "subscriptions.db"
CHECK_INTERVAL_SECONDS = 3600  # hourly; use the "run check now" button to
# verify notification behavior immediately instead of waiting for a tick.

store = SubscriptionStore(DB_PATH)

_stop_event = threading.Event()
_scheduler_thread: threading.Thread | None = None


def run_check_cycle(today: date | None = None) -> list[str]:
    """Roll forward any past-due renewals, then notify (once per renewal
    date) for everything within its alert lead time. Returns the list of
    fired notification messages, for both the scheduler log and the
    "run check now" HTTP response."""
    today = today or local_today()
    store.roll_forward_all(today)
    fired: list[str] = []
    for sub in store.due_for_notification(today):
        days_until = (sub.next_renewal - today).days
        when = "today" if days_until == 0 else f"in {days_until} day(s)"
        message = f"{sub.name} renews {when} ({sub.cost_display}) on {sub.next_renewal.isoformat()}"
        notify(title="Subscription renewal coming up", message=message)
        store.log_notification(sub, message)
        fired.append(message)
    return fired


def _scheduler_loop() -> None:
    while not _stop_event.wait(CHECK_INTERVAL_SECONDS):
        try:
            run_check_cycle()
        except Exception:  # pragma: no cover - defensive; keep the thread alive
            logger.exception("Scheduled renewal check failed")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _scheduler_thread
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()
    try:
        yield
    finally:
        _stop_event.set()
        if _scheduler_thread is not None:
            _scheduler_thread.join(timeout=2)


app = FastAPI(title="Subscription Tracker", lifespan=lifespan)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
header { margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: baseline; }
header a.brand { font-weight: 700; font-size: 1.25rem; text-decoration: none; }
nav a { margin-left: 1rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #8884; padding: 0.4rem 0.6rem; text-align: left; }
input, select, button { font-size: 1rem; padding: 0.4rem 0.6rem; }
form .row { margin: 0.75rem 0; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
button { cursor: pointer; }
.urgency-red { border-left: 4px solid #dc2626; }
.urgency-amber { border-left: 4px solid #d97706; }
.urgency-green { border-left: 4px solid #16a34a; }
.card { border: 1px solid #8884; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
.muted { color: #666; font-size: 0.9rem; }
.actions form { display: inline; }
#action-result { margin: 0.5rem 0; }
</style>
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://unpkg.com/htmx.org@2.0.3/dist/htmx.min.js"
        integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq"
        crossorigin="anonymous"></script>
<style>{CSS}
</head>
<body>
<header>
  <a class="brand" href="/">Subscription Tracker</a>
  <nav><a href="/subscriptions/new">+ Add subscription</a><a href="/notifications">Alert log</a></nav>
</header>
<main>{body}</main>
</body>
</html>""")


def _urgency_class(days_until: int) -> str:
    if days_until <= 3:
        return "urgency-red"
    if days_until <= 7:
        return "urgency-amber"
    return "urgency-green"


def _options(values: tuple[str, ...], selected: str) -> str:
    return "".join(
        f'<option value="{v}"{" selected" if v == selected else ""}>{v}</option>'
        for v in values
    )


def render_dashboard() -> HTMLResponse:
    today = local_today()
    store.roll_forward_all(today)
    subs = store.list_active()
    summary = store.spending_summary()

    rows = "".join(
        f"""<tr class="{_urgency_class((s.next_renewal - today).days)}">
<td>{html.escape(s.name)}</td>
<td>{html.escape(s.cost_display)}</td>
<td>{html.escape(s.billing_cycle)}</td>
<td>{html.escape(s.category)}</td>
<td>{s.next_renewal.isoformat()} ({(s.next_renewal - today).days}d)</td>
<td class="actions">
  <a href="/subscriptions/{s.id}/edit">Edit</a>
  <form method="post" action="/subscriptions/{s.id}/cancel"><button>Cancel</button></form>
  <form method="post" action="/subscriptions/{s.id}/delete" data-name="{html.escape(s.name)}"
        onsubmit="return confirm('Delete ' + this.dataset.name + '?')"><button>Delete</button></form>
</td>
</tr>"""
        for s in subs
    )
    table = (
        f"""<table><thead><tr><th>Name</th><th>Cost</th><th>Cycle</th><th>Category</th>
<th>Next renewal</th><th></th></tr></thead><tbody>{rows}</tbody></table>"""
        if subs
        else "<p>No active subscriptions yet.</p>"
    )

    category_rows = "".join(
        f"<tr><td>{html.escape(c.category)}</td><td>{c.monthly_cents / 100:.2f}/mo</td></tr>"
        for c in summary.by_category
    )

    body = f"""
<div class="card">
  <h2>Spending summary</h2>
  <p><strong>{summary.monthly_total_cents / 100:.2f}</strong>/month &middot;
     <strong>{summary.annual_total_cents / 100:.2f}</strong>/year (monthly-equivalent across billing cycles)</p>
  <table><thead><tr><th>Category</th><th>Monthly equivalent</th></tr></thead>
  <tbody>{category_rows or '<tr><td colspan="2">No active subscriptions</td></tr>'}</tbody></table>
</div>

<div class="card">
  <h2>Upcoming renewals</h2>
  {table}
</div>

<div class="card">
  <h2>Notifications</h2>
  <p class="muted">Background check runs every {CHECK_INTERVAL_SECONDS // 60} minutes automatically.
  Use these to verify or trigger it immediately.</p>
  <button hx-post="/notifications/run-check" hx-target="#action-result" hx-swap="innerHTML">Run renewal check now</button>
  <button hx-post="/notifications/test" hx-target="#action-result" hx-swap="innerHTML">Send test notification</button>
  <div id="action-result"></div>
</div>
"""
    return page("Dashboard", body)


def _subscription_form(sub: Subscription | None) -> str:
    action = f"/subscriptions/{sub.id}" if sub else "/subscriptions"
    cost = f"{sub.cost_cents / 100:.2f}" if sub else ""
    name = sub.name if sub else ""
    category = sub.category if sub else "Uncategorized"
    cycle = sub.billing_cycle if sub else "monthly"
    custom_days = sub.custom_interval_days if sub and sub.custom_interval_days else ""
    next_renewal = sub.next_renewal.isoformat() if sub else local_today().isoformat()
    alert_lead = sub.alert_lead_days if sub else 3
    notes = sub.notes or "" if sub else ""
    cancel_url = sub.cancel_url or "" if sub else ""
    currency = sub.currency if sub else "USD"

    return f"""
<h1>{"Edit" if sub else "Add"} subscription</h1>
<form method="post" action="{action}">
  <div class="row">
    <label>Name <input type="text" name="name" value="{html.escape(name)}" required></label>
    <label>Cost <input type="number" step="0.01" min="0" name="cost" value="{cost}" required></label>
    <label>Currency <input type="text" name="currency" value="{html.escape(currency)}" maxlength="8" required></label>
  </div>
  <div class="row">
    <label>Category <input type="text" name="category" value="{html.escape(category)}"></label>
    <label>Billing cycle <select name="billing_cycle" id="billing_cycle">{_options(BILLING_CYCLES, cycle)}</select></label>
    <label>Custom interval (days) <input type="number" min="1" name="custom_interval_days" value="{custom_days}"></label>
  </div>
  <div class="row">
    <label>Next renewal <input type="date" name="next_renewal" value="{next_renewal}" required></label>
    <label>Alert lead time (days) <input type="number" min="0" name="alert_lead_days" value="{alert_lead}" required></label>
  </div>
  <div class="row">
    <label>Cancel URL <input type="text" name="cancel_url" value="{html.escape(cancel_url)}"></label>
  </div>
  <div class="row">
    <label>Notes <input type="text" name="notes" value="{html.escape(notes)}"></label>
  </div>
  <button type="submit">{"Save" if sub else "Add"}</button>
  <a href="/">Cancel</a>
</form>
"""


def render_notifications() -> HTMLResponse:
    history = store.notification_history()
    rows = "".join(
        f"<tr><td>{n.sent_at.strftime('%Y-%m-%d %H:%M:%S')}</td>"
        f"<td>{html.escape(n.message)}</td></tr>"
        for n in history
    )
    body = f"""
<h1>Notification history</h1>
<table><thead><tr><th>Sent at (UTC)</th><th>Message</th></tr></thead>
<tbody>{rows or '<tr><td colspan="2">No notifications sent yet.</td></tr>'}</tbody></table>
"""
    return page("Notifications", body)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return render_dashboard()


@app.get("/subscriptions/new", response_class=HTMLResponse)
def new_subscription_form() -> HTMLResponse:
    return page("Add subscription", _subscription_form(None))


def _parse_cost_cents(cost: str) -> int:
    try:
        return round(float(cost) * 100)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid cost: {cost!r}") from exc


_CURRENCY_RE = re.compile(r"[A-Za-z]{1,8}")


def _validate_currency(currency: str) -> str:
    """Reject anything that isn't a short alphabetic code before it ever
    reaches storage -- not for injection safety (every render site already
    escapes it), but so a stray non-code value can't silently corrupt the
    spending summary/display."""
    value = currency.strip() or "USD"
    if not _CURRENCY_RE.fullmatch(value):
        raise HTTPException(400, f"Invalid currency code: {currency!r}")
    return value


@app.post("/subscriptions")
def create_subscription(
    name: str = Form(...),
    cost: str = Form(...),
    currency: str = Form("USD"),
    category: str = Form("Uncategorized"),
    billing_cycle: str = Form("monthly"),
    custom_interval_days: str = Form(""),
    next_renewal: str = Form(...),
    alert_lead_days: int = Form(3),
    notes: str = Form(""),
    cancel_url: str = Form(""),
) -> RedirectResponse:
    if not name.strip():
        raise HTTPException(400, "Name must not be empty")
    if billing_cycle not in BILLING_CYCLES:
        raise HTTPException(400, f"Unknown billing cycle: {billing_cycle!r}")
    try:
        renewal_date = date.fromisoformat(next_renewal)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid date {next_renewal!r}: {exc}") from exc
    custom_days = int(custom_interval_days) if custom_interval_days.strip() else None
    if billing_cycle == "custom" and not custom_days:
        raise HTTPException(400, "custom billing cycle requires custom_interval_days")

    sub = Subscription(
        name=name.strip(),
        cost_cents=_parse_cost_cents(cost),
        currency=_validate_currency(currency),
        category=category.strip() or "Uncategorized",
        billing_cycle=billing_cycle,
        custom_interval_days=custom_days,
        next_renewal=renewal_date,
        alert_lead_days=alert_lead_days,
        notes=notes.strip() or None,
        cancel_url=cancel_url.strip() or None,
    )
    store.create(sub)
    return RedirectResponse("/", status_code=303)


@app.get("/subscriptions/{sub_id}/edit", response_class=HTMLResponse)
def edit_subscription_form(sub_id: int) -> HTMLResponse:
    sub = store.get(sub_id)
    if sub is None:
        raise HTTPException(404, "Subscription not found")
    return page("Edit subscription", _subscription_form(sub))


@app.post("/subscriptions/{sub_id}")
def update_subscription(
    sub_id: int,
    name: str = Form(...),
    cost: str = Form(...),
    currency: str = Form("USD"),
    category: str = Form("Uncategorized"),
    billing_cycle: str = Form("monthly"),
    custom_interval_days: str = Form(""),
    next_renewal: str = Form(...),
    alert_lead_days: int = Form(3),
    notes: str = Form(""),
    cancel_url: str = Form(""),
) -> RedirectResponse:
    if billing_cycle not in BILLING_CYCLES:
        raise HTTPException(400, f"Unknown billing cycle: {billing_cycle!r}")
    try:
        renewal_date = date.fromisoformat(next_renewal)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid date {next_renewal!r}: {exc}") from exc
    custom_days = int(custom_interval_days) if custom_interval_days.strip() else None

    updated = store.update(
        sub_id,
        name=name.strip(),
        cost_cents=_parse_cost_cents(cost),
        currency=_validate_currency(currency),
        category=category.strip() or "Uncategorized",
        billing_cycle=billing_cycle,
        custom_interval_days=custom_days,
        next_renewal=renewal_date,
        alert_lead_days=alert_lead_days,
        notes=notes.strip() or None,
        cancel_url=cancel_url.strip() or None,
    )
    if updated is None:
        raise HTTPException(404, "Subscription not found")
    return RedirectResponse("/", status_code=303)


@app.post("/subscriptions/{sub_id}/delete")
def delete_subscription(sub_id: int) -> RedirectResponse:
    if not store.delete(sub_id):
        raise HTTPException(404, "Subscription not found")
    return RedirectResponse("/", status_code=303)


@app.post("/subscriptions/{sub_id}/cancel")
def cancel_subscription(sub_id: int) -> RedirectResponse:
    if store.update(sub_id, active=False) is None:
        raise HTTPException(404, "Subscription not found")
    return RedirectResponse("/", status_code=303)


@app.post("/subscriptions/{sub_id}/reactivate")
def reactivate_subscription(sub_id: int) -> RedirectResponse:
    if store.update(sub_id, active=True) is None:
        raise HTTPException(404, "Subscription not found")
    return RedirectResponse("/", status_code=303)


@app.get("/notifications", response_class=HTMLResponse)
def notifications_page() -> HTMLResponse:
    return render_notifications()


@app.post("/notifications/test", response_class=HTMLResponse)
def send_test_notification() -> HTMLResponse:
    sent = notify(
        title="Subscription Tracker test",
        message="This is a test notification from Subscription Tracker.",
    )
    if sent:
        return HTMLResponse("<p>Test notification sent -- check your desktop.</p>")
    return HTMLResponse(
        "<p>Could not deliver a desktop notification on this system "
        "(no backend available); see server logs.</p>"
    )


@app.post("/notifications/run-check", response_class=HTMLResponse)
def run_check_now() -> HTMLResponse:
    fired = run_check_cycle()
    if not fired:
        return HTMLResponse("<p>Checked -- nothing due for notification right now.</p>")
    items = "".join(f"<li>{html.escape(m)}</li>" for m in fired)
    return HTMLResponse(f"<p>Sent {len(fired)} notification(s):</p><ul>{items}</ul>")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "subscriptions": len(store.list_all())}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
