from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from storage import (
    Subscription,
    SubscriptionStore,
    add_months,
    compute_next_renewal,
    local_today,
    monthly_equivalent_cents,
    roll_forward,
)

# ---------------------------------------------------------------------------
# Pure date math
# ---------------------------------------------------------------------------


def test_add_months_clamps_short_month() -> None:
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)  # 2026 not a leap year


def test_add_months_clamps_into_leap_february() -> None:
    assert add_months(date(2028, 1, 31), 1) == date(2028, 2, 29)  # 2028 is a leap year


def test_add_months_quarterly_30_day_month() -> None:
    assert add_months(date(2026, 1, 31), 3) == date(2026, 4, 30)


def test_add_months_annual_leap_day_rolls_to_non_leap_year() -> None:
    assert add_months(date(2028, 2, 29), 12) == date(2029, 2, 28)


def test_add_months_handles_year_rollover() -> None:
    assert add_months(date(2026, 12, 15), 2) == date(2027, 2, 15)


@pytest.mark.parametrize(
    ("cycle", "current", "expected"),
    [
        ("weekly", date(2026, 1, 1), date(2026, 1, 8)),
        ("monthly", date(2026, 1, 15), date(2026, 2, 15)),
        ("quarterly", date(2026, 1, 15), date(2026, 4, 15)),
        ("annual", date(2026, 1, 15), date(2027, 1, 15)),
    ],
)
def test_compute_next_renewal_standard_cycles(
    cycle: str, current: date, expected: date
) -> None:
    assert compute_next_renewal(current, cycle) == expected


def test_compute_next_renewal_custom_interval() -> None:
    assert compute_next_renewal(date(2026, 1, 1), "custom", 45) == date(2026, 2, 15)


def test_compute_next_renewal_custom_requires_positive_interval() -> None:
    with pytest.raises(ValueError, match="custom_interval_days"):
        compute_next_renewal(date(2026, 1, 1), "custom", None)
    with pytest.raises(ValueError, match="custom_interval_days"):
        compute_next_renewal(date(2026, 1, 1), "custom", 0)


def test_compute_next_renewal_unknown_cycle_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown billing cycle"):
        compute_next_renewal(date(2026, 1, 1), "daily")


def test_roll_forward_advances_past_today() -> None:
    result = roll_forward(date(2026, 1, 1), "monthly", None, today=date(2026, 1, 15))
    assert result == date(2026, 2, 1)


def test_roll_forward_catches_up_multiple_missed_cycles() -> None:
    # Last renewal was 4 months ago; a single call must catch all the way up
    # to the next renewal strictly after "today", not stop after one step.
    result = roll_forward(date(2026, 1, 1), "monthly", None, today=date(2026, 5, 10))
    assert result == date(2026, 6, 1)


def test_roll_forward_noop_when_already_future() -> None:
    result = roll_forward(date(2026, 6, 1), "monthly", None, today=date(2026, 5, 10))
    assert result == date(2026, 6, 1)


def test_monthly_equivalent_cents_across_cycles() -> None:
    weekly = Subscription(
        name="w", cost_cents=1000, billing_cycle="weekly", next_renewal=local_today()
    )
    monthly = Subscription(
        name="m", cost_cents=1000, billing_cycle="monthly", next_renewal=local_today()
    )
    quarterly = Subscription(
        name="q", cost_cents=3000, billing_cycle="quarterly", next_renewal=local_today()
    )
    annual = Subscription(
        name="a", cost_cents=1200, billing_cycle="annual", next_renewal=local_today()
    )
    assert monthly_equivalent_cents(weekly) == pytest.approx(1000 * 52 / 12)
    assert monthly_equivalent_cents(monthly) == 1000
    assert monthly_equivalent_cents(quarterly) == pytest.approx(1000)
    assert monthly_equivalent_cents(annual) == 100


def test_monthly_equivalent_cents_custom_interval() -> None:
    sub = Subscription(
        name="c",
        cost_cents=6000,
        billing_cycle="custom",
        custom_interval_days=60,
        next_renewal=local_today(),
    )
    # ~30.44 days/month average, over a 60-day cycle -> roughly half a cycle's cost per month
    assert monthly_equivalent_cents(sub) == pytest.approx(6000 * 30.436875 / 60)


# ---------------------------------------------------------------------------
# Store: CRUD, rollover, notifications, spending summary
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> SubscriptionStore:
    return SubscriptionStore(tmp_path / "test.db")


def _make(store: SubscriptionStore, **overrides: object) -> Subscription:
    defaults: dict[str, object] = {
        "name": "Netflix",
        "cost_cents": 1599,
        "category": "Streaming",
        "billing_cycle": "monthly",
        "next_renewal": local_today(),
        "alert_lead_days": 3,
    }
    defaults.update(overrides)
    return store.create(Subscription(**defaults))  # type: ignore[arg-type]


def test_create_and_get(store: SubscriptionStore) -> None:
    sub = _make(store)
    fetched = store.get(sub.id)  # type: ignore[arg-type]
    assert fetched is not None
    assert fetched.name == "Netflix"


def test_update_fields(store: SubscriptionStore) -> None:
    sub = _make(store)
    updated = store.update(sub.id, name="Netflix Premium", cost_cents=2299)  # type: ignore[arg-type]
    assert updated is not None
    assert updated.name == "Netflix Premium"
    assert updated.cost_cents == 2299


def test_update_missing_returns_none(store: SubscriptionStore) -> None:
    assert store.update(9999, name="x") is None


def test_delete(store: SubscriptionStore) -> None:
    sub = _make(store)
    assert store.delete(sub.id) is True  # type: ignore[arg-type]
    assert store.get(sub.id) is None  # type: ignore[arg-type]
    assert store.delete(sub.id) is False  # type: ignore[arg-type]


def test_list_active_excludes_cancelled(store: SubscriptionStore) -> None:
    active = _make(store, name="Active")
    cancelled = _make(store, name="Cancelled")
    store.update(cancelled.id, active=False)  # type: ignore[arg-type]

    listed = {s.name for s in store.list_active()}
    assert listed == {"Active"}
    assert active.name in listed


def test_roll_forward_all_advances_overdue_subscriptions(
    store: SubscriptionStore,
) -> None:
    past_due = _make(store, name="Overdue", next_renewal=local_today().replace(day=1))
    store.update(past_due.id, next_renewal=date(2020, 1, 1))  # type: ignore[arg-type]

    rolled = store.roll_forward_all(today=local_today())
    assert rolled == 1
    refreshed = store.get(past_due.id)  # type: ignore[arg-type]
    assert refreshed is not None
    assert refreshed.next_renewal > local_today()


def test_due_for_notification_respects_lead_time(store: SubscriptionStore) -> None:
    today = date(2026, 3, 1)
    soon = _make(store, name="Soon", next_renewal=date(2026, 3, 3), alert_lead_days=3)
    far = _make(store, name="Far", next_renewal=date(2026, 4, 1), alert_lead_days=3)

    due = store.due_for_notification(today=today)
    names = {s.name for s in due}
    assert soon.name in names
    assert far.name not in names


def test_due_for_notification_is_idempotent_per_renewal_date(
    store: SubscriptionStore,
) -> None:
    today = date(2026, 3, 1)
    sub = _make(store, name="Soon", next_renewal=date(2026, 3, 2), alert_lead_days=3)

    first_pass = store.due_for_notification(today=today)
    assert len(first_pass) == 1
    store.log_notification(first_pass[0], "test message")

    second_pass = store.due_for_notification(today=today)
    assert second_pass == []

    # A NEW renewal date (next cycle) should become notifiable again.
    store.update(sub.id, next_renewal=date(2026, 4, 2))  # type: ignore[arg-type]
    third_pass = store.due_for_notification(today=date(2026, 4, 1))
    assert len(third_pass) == 1


def test_due_for_notification_ignores_cancelled(store: SubscriptionStore) -> None:
    sub = _make(store, name="Soon", next_renewal=local_today(), alert_lead_days=3)
    store.update(sub.id, active=False)  # type: ignore[arg-type]
    assert store.due_for_notification(today=local_today()) == []


def test_notification_history_orders_most_recent_first(
    store: SubscriptionStore,
) -> None:
    sub = _make(store)
    store.log_notification(sub, "first")
    store.log_notification(sub, "second")

    history = store.notification_history()
    assert [h.message for h in history] == ["second", "first"]


def test_spending_summary_aggregates_monthly_equivalent_by_category(
    store: SubscriptionStore,
) -> None:
    _make(
        store, name="A", cost_cents=1000, billing_cycle="monthly", category="Streaming"
    )
    _make(
        store, name="B", cost_cents=1200, billing_cycle="annual", category="Streaming"
    )
    _make(
        store, name="C", cost_cents=3000, billing_cycle="quarterly", category="Software"
    )

    summary = store.spending_summary()
    by_cat = {c.category: c.monthly_cents for c in summary.by_category}
    assert by_cat["Streaming"] == pytest.approx(1000 + 100)
    assert by_cat["Software"] == pytest.approx(1000)
    assert summary.monthly_total_cents == pytest.approx(1000 + 100 + 1000)
    assert summary.annual_total_cents == pytest.approx(summary.monthly_total_cents * 12)


def test_spending_summary_excludes_cancelled_subscriptions(
    store: SubscriptionStore,
) -> None:
    _make(store, name="Active", cost_cents=1000)
    cancelled = _make(store, name="Cancelled", cost_cents=5000)
    store.update(cancelled.id, active=False)  # type: ignore[arg-type]

    summary = store.spending_summary()
    assert summary.monthly_total_cents == pytest.approx(1000)
