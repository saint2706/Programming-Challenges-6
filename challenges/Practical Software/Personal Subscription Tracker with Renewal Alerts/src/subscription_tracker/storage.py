"""SQLModel-backed storage for the subscription tracker: renewal math,
spending summaries, and idempotent notification tracking.

The one thing that has to be airtight is billing-cycle date math: a monthly
subscription renewing on the 31st has to land on Feb 28 (or 29), not crash
or silently roll into March, and a subscription that's been left untouched
for several missed cycles has to catch back up to the future in one pass
rather than staying stuck in the past. `add_months` / `compute_next_renewal`
/ `roll_forward` are pure functions with no framework or I/O dependency so
they're trivially unit-testable.

The second thing that has to be airtight is "don't nag": a renewal within
the alert window must fire exactly one notification per renewal date, not
once per background-thread tick. `NotificationLog` rows are keyed on
(subscription_id, renewal_date), and `due_for_notification` excludes any
subscription that already has a log row for its *current* next_renewal, so
re-running the check as often as you like is safe.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, select

BILLING_CYCLES = ("weekly", "monthly", "quarterly", "annual", "custom")


def local_today() -> date:
    """The current local calendar date. Uses an explicit timezone-aware
    `now()` rather than the naive `date.today()` (flagged by this repo's
    lint config, DTZ011) -- and local time is genuinely what "today" should
    mean for a tool that runs on the user's own machine tracking their own
    renewals, not UTC."""
    return datetime.now().astimezone().date()


# Average month length (365.2425 / 12), used only to express a custom
# interval-days subscription as a monthly-equivalent cost for the spending
# summary -- never used for date arithmetic itself.
_AVG_DAYS_PER_MONTH = 30.436875


class Subscription(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    cost_cents: int
    currency: str = "USD"
    category: str = "Uncategorized"
    billing_cycle: str = "monthly"
    custom_interval_days: int | None = None
    next_renewal: date
    alert_lead_days: int = 3
    notes: str | None = None
    cancel_url: str | None = None
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def cost_display(self) -> str:
        return f"{self.cost_cents / 100:.2f} {self.currency}"


class NotificationLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    subscription_id: int = Field(foreign_key="subscription.id")
    renewal_date: date
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


@dataclass
class CategorySpend:
    category: str
    monthly_cents: float


@dataclass
class SpendingSummary:
    monthly_total_cents: float
    annual_total_cents: float
    by_category: list[CategorySpend] = field(default_factory=list)


def add_months(d: date, months: int) -> date:
    """Add `months` calendar months to `d`, clamping the day to the target
    month's last valid day (e.g. Jan 31 + 1 month -> Feb 28/29, not a
    ValueError or a silent roll into March)."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)


def compute_next_renewal(
    current: date, cycle: str, custom_interval_days: int | None = None
) -> date:
    if cycle == "weekly":
        from datetime import timedelta

        return current + timedelta(days=7)
    if cycle == "monthly":
        return add_months(current, 1)
    if cycle == "quarterly":
        return add_months(current, 3)
    if cycle == "annual":
        return add_months(current, 12)
    if cycle == "custom":
        from datetime import timedelta

        if not custom_interval_days or custom_interval_days <= 0:
            raise ValueError(
                "custom billing cycle requires a positive custom_interval_days"
            )
        return current + timedelta(days=custom_interval_days)
    raise ValueError(f"Unknown billing cycle: {cycle!r}")


def roll_forward(
    next_renewal: date, cycle: str, custom_interval_days: int | None, today: date
) -> date:
    """Advance `next_renewal` past `today`, handling any number of missed
    cycles in one call (e.g. a monthly subscription untouched for 4 months
    catches all the way up, not just one step)."""
    renewal = next_renewal
    # Bounded by construction: each iteration strictly increases `renewal`
    # (all billing cycles add >= 1 day), so this always terminates.
    while renewal <= today:
        renewal = compute_next_renewal(renewal, cycle, custom_interval_days)
    return renewal


def monthly_equivalent_cents(sub: Subscription) -> float:
    if sub.billing_cycle == "weekly":
        return sub.cost_cents * 52 / 12
    if sub.billing_cycle == "monthly":
        return float(sub.cost_cents)
    if sub.billing_cycle == "quarterly":
        return sub.cost_cents / 3
    if sub.billing_cycle == "annual":
        return sub.cost_cents / 12
    if sub.billing_cycle == "custom":
        if not sub.custom_interval_days or sub.custom_interval_days <= 0:
            return 0.0
        return sub.cost_cents * (_AVG_DAYS_PER_MONTH / sub.custom_interval_days)
    raise ValueError(f"Unknown billing cycle: {sub.billing_cycle!r}")


class SubscriptionStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(self.engine)

    # -- CRUD -----------------------------------------------------------

    def create(self, sub: Subscription) -> Subscription:
        with Session(self.engine) as session:
            session.add(sub)
            session.commit()
            session.refresh(sub)
            return sub

    def get(self, sub_id: int) -> Subscription | None:
        with Session(self.engine) as session:
            return session.get(Subscription, sub_id)

    def update(self, sub_id: int, **fields: object) -> Subscription | None:
        with Session(self.engine) as session:
            sub = session.get(Subscription, sub_id)
            if sub is None:
                return None
            for key, value in fields.items():
                setattr(sub, key, value)
            session.add(sub)
            session.commit()
            session.refresh(sub)
            return sub

    def delete(self, sub_id: int) -> bool:
        with Session(self.engine) as session:
            sub = session.get(Subscription, sub_id)
            if sub is None:
                return False
            session.delete(sub)
            session.commit()
            return True

    def list_active(self) -> list[Subscription]:
        with Session(self.engine) as session:
            rows = session.exec(
                select(Subscription)
                .where(Subscription.active == True)
                .order_by(Subscription.next_renewal)
            ).all()
            return list(rows)

    def list_all(self) -> list[Subscription]:
        with Session(self.engine) as session:
            rows = session.exec(
                select(Subscription).order_by(Subscription.next_renewal)
            ).all()
            return list(rows)

    # -- Renewal rollover -------------------------------------------------

    def roll_forward_all(self, today: date | None = None) -> int:
        """Advance every active subscription's `next_renewal` past `today`.
        Idempotent and cheap to call on every dashboard render as well as
        every background scheduler tick -- a subscription already in the
        future is untouched."""
        today = today or local_today()
        rolled = 0
        with Session(self.engine) as session:
            subs = session.exec(
                select(Subscription).where(Subscription.active == True)
            ).all()
            for sub in subs:
                if sub.next_renewal <= today:
                    sub.next_renewal = roll_forward(
                        sub.next_renewal,
                        sub.billing_cycle,
                        sub.custom_interval_days,
                        today,
                    )
                    session.add(sub)
                    rolled += 1
            session.commit()
        return rolled

    # -- Notifications ----------------------------------------------------

    def due_for_notification(self, today: date | None = None) -> list[Subscription]:
        """Active subscriptions whose next_renewal is within their alert
        lead time and that have not already been notified for *this*
        renewal date."""
        today = today or local_today()
        with Session(self.engine) as session:
            subs = session.exec(
                select(Subscription).where(Subscription.active == True)
            ).all()
            due: list[Subscription] = []
            for sub in subs:
                days_until = (sub.next_renewal - today).days
                if days_until < 0 or days_until > sub.alert_lead_days:
                    continue
                already_logged = session.exec(
                    select(NotificationLog).where(
                        NotificationLog.subscription_id == sub.id,
                        NotificationLog.renewal_date == sub.next_renewal,
                    )
                ).first()
                if already_logged is None:
                    due.append(sub)
            return due

    def log_notification(self, sub: Subscription, message: str) -> NotificationLog:
        with Session(self.engine) as session:
            log = NotificationLog(
                subscription_id=sub.id,  # type: ignore[arg-type]
                renewal_date=sub.next_renewal,
                message=message,
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log

    def notification_history(self, limit: int = 50) -> list[NotificationLog]:
        with Session(self.engine) as session:
            rows = session.exec(
                select(NotificationLog)
                .order_by(NotificationLog.sent_at.desc())
                .limit(limit)
            ).all()
            return list(rows)

    # -- Spending summary ---------------------------------------------------

    def spending_summary(self) -> SpendingSummary:
        by_category: dict[str, float] = {}
        monthly_total = 0.0
        for sub in self.list_active():
            m = monthly_equivalent_cents(sub)
            monthly_total += m
            by_category[sub.category] = by_category.get(sub.category, 0.0) + m
        categories = [
            CategorySpend(category=cat, monthly_cents=cents)
            for cat, cents in sorted(by_category.items(), key=lambda kv: -kv[1])
        ]
        return SpendingSummary(
            monthly_total_cents=monthly_total,
            annual_total_cents=monthly_total * 12,
            by_category=categories,
        )
