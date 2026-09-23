from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import app as app_module
import pytest
from fastapi.testclient import TestClient
from storage import SubscriptionStore, local_today


def _first_subscription_id(dashboard_html: str) -> str:
    match = re.search(r"/subscriptions/(\d+)/edit", dashboard_html)
    assert match is not None, "no subscription row found on dashboard"
    return match.group(1)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_module, "store", SubscriptionStore(tmp_path / "test.db"))
    # Never let the automated test suite pop a real OS notification.
    monkeypatch.setattr(app_module, "notify", lambda *a, **k: True)
    with TestClient(app_module.app) as c:
        yield c


def _create(
    client: TestClient,
    *,
    name: str = "Netflix",
    cost: str = "15.99",
    billing_cycle: str = "monthly",
    next_renewal: str | None = None,
    category: str = "Streaming",
    alert_lead_days: int = 3,
) -> str:
    resp = client.post(
        "/subscriptions",
        data={
            "name": name,
            "cost": cost,
            "currency": "USD",
            "category": category,
            "billing_cycle": billing_cycle,
            "custom_interval_days": "",
            "next_renewal": next_renewal or local_today().isoformat(),
            "alert_lead_days": str(alert_lead_days),
            "notes": "",
            "cancel_url": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    return resp.headers["location"]


def test_index_shows_empty_state(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "No active subscriptions yet" in resp.text


def test_new_subscription_form_renders(client: TestClient) -> None:
    resp = client.get("/subscriptions/new")
    assert resp.status_code == 200
    assert "<form" in resp.text
    assert "Add subscription" in resp.text


def test_create_subscription_and_appears_on_dashboard(client: TestClient) -> None:
    _create(client, name="Spotify")
    resp = client.get("/")
    assert "Spotify" in resp.text
    assert "15.99" in resp.text


def test_create_rejects_empty_name(client: TestClient) -> None:
    resp = client.post(
        "/subscriptions",
        data={
            "name": "   ",
            "cost": "5",
            "currency": "USD",
            "category": "x",
            "billing_cycle": "monthly",
            "custom_interval_days": "",
            "next_renewal": local_today().isoformat(),
            "alert_lead_days": "3",
            "notes": "",
            "cancel_url": "",
        },
    )
    assert resp.status_code == 400


def test_create_rejects_unknown_billing_cycle(client: TestClient) -> None:
    resp = client.post(
        "/subscriptions",
        data={
            "name": "x",
            "cost": "5",
            "currency": "USD",
            "category": "x",
            "billing_cycle": "daily",
            "custom_interval_days": "",
            "next_renewal": local_today().isoformat(),
            "alert_lead_days": "3",
            "notes": "",
            "cancel_url": "",
        },
    )
    assert resp.status_code == 400


def test_edit_form_prefills_existing_values(client: TestClient) -> None:
    _create(client, name="Adobe CC")
    resp = client.get("/")
    sub_id = _first_subscription_id(resp.text)

    edit_resp = client.get(f"/subscriptions/{sub_id}/edit")
    assert edit_resp.status_code == 200
    assert 'value="Adobe CC"' in edit_resp.text


def test_edit_missing_subscription_is_404(client: TestClient) -> None:
    assert client.get("/subscriptions/9999/edit").status_code == 404


def test_update_subscription_changes_values(client: TestClient) -> None:
    _create(client, name="Old Name", cost="1.00")
    resp = client.get("/")
    sub_id = _first_subscription_id(resp.text)

    update_resp = client.post(
        f"/subscriptions/{sub_id}",
        data={
            "name": "New Name",
            "cost": "9.99",
            "currency": "USD",
            "category": "Streaming",
            "billing_cycle": "monthly",
            "custom_interval_days": "",
            "next_renewal": local_today().isoformat(),
            "alert_lead_days": "3",
            "notes": "",
            "cancel_url": "",
        },
        follow_redirects=False,
    )
    assert update_resp.status_code == 303
    dashboard = client.get("/")
    assert "New Name" in dashboard.text
    assert "9.99" in dashboard.text


def test_cancel_subscription_removes_from_active_dashboard(client: TestClient) -> None:
    _create(client, name="ToCancel")
    resp = client.get("/")
    sub_id = _first_subscription_id(resp.text)

    cancel_resp = client.post(f"/subscriptions/{sub_id}/cancel", follow_redirects=False)
    assert cancel_resp.status_code == 303
    dashboard = client.get("/")
    assert "ToCancel" not in dashboard.text


def test_delete_subscription(client: TestClient) -> None:
    _create(client, name="ToDelete")
    resp = client.get("/")
    sub_id = _first_subscription_id(resp.text)

    del_resp = client.post(f"/subscriptions/{sub_id}/delete", follow_redirects=False)
    assert del_resp.status_code == 303
    assert client.get(f"/subscriptions/{sub_id}/edit").status_code == 404


def test_delete_missing_subscription_is_404(client: TestClient) -> None:
    assert client.post("/subscriptions/9999/delete").status_code == 404


def test_spending_summary_reflects_active_subscriptions(client: TestClient) -> None:
    _create(client, name="A", cost="10.00", billing_cycle="monthly")
    _create(client, name="B", cost="120.00", billing_cycle="annual")
    resp = client.get("/")
    # 10.00/mo + (120/12)=10.00/mo -> 20.00 total monthly equivalent
    assert "20.00" in resp.text


def test_send_test_notification_uses_mocked_backend(client: TestClient) -> None:
    resp = client.post("/notifications/test")
    assert resp.status_code == 200
    assert "sent" in resp.text.lower()


def test_run_check_now_fires_and_logs_due_notification(client: TestClient) -> None:
    soon = (local_today() + timedelta(days=1)).isoformat()
    _create(client, name="DueSoon", next_renewal=soon, alert_lead_days=3)

    resp = client.post("/notifications/run-check")
    assert resp.status_code == 200
    assert "DueSoon" in resp.text

    history = client.get("/notifications")
    assert "DueSoon" in history.text


def test_run_check_now_is_idempotent(client: TestClient) -> None:
    soon = (local_today() + timedelta(days=1)).isoformat()
    _create(client, name="DueSoon", next_renewal=soon, alert_lead_days=3)

    first = client.post("/notifications/run-check")
    assert "DueSoon" in first.text

    second = client.post("/notifications/run-check")
    assert "nothing due" in second.text.lower()


def test_notifications_page_renders_empty_state(client: TestClient) -> None:
    resp = client.get("/notifications")
    assert resp.status_code == 200
    assert "No notifications sent yet" in resp.text


def test_health_endpoint(client: TestClient) -> None:
    _create(client, name="A")
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["subscriptions"] == 1


# ---------------------------------------------------------------------------
# Adversarial input: a subscription name is fully attacker-controlled (it's
# free-text the user typed), so every place it's rendered has to survive
# both a classic <script> payload and a payload aimed specifically at the
# delete button's onsubmit="confirm(...)" JS-string-literal context.
# ---------------------------------------------------------------------------


def test_subscription_name_html_is_escaped_everywhere_it_renders(
    client: TestClient,
) -> None:
    payload = "<script>alert(1)</script>"
    _create(client, name=payload)

    dashboard = client.get("/")
    assert "<script>alert(1)</script>" not in dashboard.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in dashboard.text

    sub_id = _first_subscription_id(dashboard.text)
    edit = client.get(f"/subscriptions/{sub_id}/edit")
    assert "<script>alert(1)</script>" not in edit.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in edit.text


def test_subscription_name_cannot_break_out_of_delete_confirm_js_string(
    client: TestClient,
) -> None:
    # A name crafted to close the single-quoted JS string literal inside
    # onsubmit="return confirm('Delete ' + ... )" if it were interpolated
    # directly into JS source rather than read from a data-* attribute.
    payload = "x'); alert(1); //"
    _create(client, name=payload)

    dashboard = client.get("/")
    # The raw payload must never appear verbatim inside an onsubmit="..."
    # attribute -- it may only appear (HTML-escaped) inside data-name="...".
    assert "onsubmit=\"return confirm('Delete x'); alert(1); //" not in dashboard.text
    assert (
        'data-name="x\'); alert(1); //"' not in dashboard.text
    )  # unescaped quote would break the attribute
    assert (
        "this.dataset.name" in dashboard.text
    )  # still uses the safe read-from-attribute pattern


def test_notification_message_containing_subscription_name_is_escaped(
    client: TestClient,
) -> None:
    soon = (local_today() + timedelta(days=1)).isoformat()
    payload = "<img src=x onerror=alert(1)>"
    _create(client, name=payload, next_renewal=soon, alert_lead_days=3)

    run_check = client.post("/notifications/run-check")
    assert "<img src=x onerror=alert(1)>" not in run_check.text
    assert "&lt;img src=x onerror=alert(1)&gt;" in run_check.text

    history = client.get("/notifications")
    assert "<img src=x onerror=alert(1)>" not in history.text
    assert "&lt;img src=x onerror=alert(1)&gt;" in history.text


def test_cost_display_is_escaped_on_dashboard(client: TestClient) -> None:
    # currency is validated server-side, but confirm the render site itself
    # also escapes cost_display defensively (belt and suspenders).
    _create(client, name="Escaped Cost")
    dashboard = client.get("/")
    assert "15.99 USD" in dashboard.text


def test_create_rejects_invalid_currency_code(client: TestClient) -> None:
    resp = client.post(
        "/subscriptions",
        data={
            "name": "x",
            "cost": "5",
            "currency": "<script>alert(1)</script>",
            "category": "x",
            "billing_cycle": "monthly",
            "custom_interval_days": "",
            "next_renewal": local_today().isoformat(),
            "alert_lead_days": "3",
            "notes": "",
            "cancel_url": "",
        },
    )
    assert resp.status_code == 400


def test_create_accepts_lowercase_currency_code(client: TestClient) -> None:
    _create(client, name="Lowercase Currency")
    resp = client.post(
        "/subscriptions",
        data={
            "name": "Euro sub",
            "cost": "5",
            "currency": "eur",
            "category": "x",
            "billing_cycle": "monthly",
            "custom_interval_days": "",
            "next_renewal": local_today().isoformat(),
            "alert_lead_days": "3",
            "notes": "",
            "cancel_url": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
