"""Database tables. Amounts are integer minor units, quantities integer ten-thousandths, rates
integer basis points (see ``money``)."""

from datetime import UTC, date, datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

DRAFT, ISSUED, VOID = "draft", "issued", "void"


def _now() -> datetime:
    return datetime.now(UTC)


class Client(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    email: str = ""
    address: str = ""
    country: str = "US"
    tax_id: str = ""
    currency: str = "USD"
    terms_days: int = 14
    notes: str = ""
    archived: bool = False


class Invoice(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("number"),)

    id: int | None = Field(default=None, primary_key=True)
    number: str | None = Field(default=None, index=True)  # assigned when issued
    client_id: int = Field(foreign_key="client.id", index=True)
    status: str = DRAFT  # draft | issued | void; paid/overdue are derived, never stored
    currency: str
    issue_date: date | None = None
    due_date: date | None = None
    discount_kind: str | None = None  # "percent" (basis points) | "fixed" (minor units)
    discount_value: int = 0
    reference: str = ""  # the client's PO number
    notes: str = ""
    recurring_id: int | None = Field(default=None, foreign_key="recurring.id")
    # Seller and buyer exactly as they were when the invoice was issued (JSON): editing a client
    # later must not rewrite history.
    snapshot: str | None = None
    created_at: datetime = Field(default_factory=_now)


class LineItem(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    invoice_id: int = Field(foreign_key="invoice.id", index=True)
    position: int
    description: str
    qty_e4: int
    unit_minor: int
    tax_bp: int = 0


class Payment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    invoice_id: int = Field(foreign_key="invoice.id", index=True)
    amount_minor: int
    paid_on: date
    method: str = ""
    note: str = ""
    created_at: datetime = Field(default_factory=_now)


class Recurring(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    client_id: int = Field(foreign_key="client.id")
    frequency: str  # weekly | monthly | quarterly | yearly
    interval: int = 1
    anchor_day: int  # day of month the schedule is anchored to (month-end clamping)
    next_date: date
    end_date: date | None = None
    active: bool = True
    auto_send: bool = False
    currency: str
    terms_days: int = 14
    discount_kind: str | None = None
    discount_value: int = 0
    notes: str = ""
    lines_json: str  # [{"description", "qty_e4", "unit_minor", "tax_bp"}, ...]


class EmailLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    invoice_id: int = Field(foreign_key="invoice.id", index=True)
    kind: str  # invoice | reminder
    stage: int | None = None  # reminder stage: days after due date
    recipient: str
    subject: str
    status: str  # sent | failed | dry-run
    error: str = ""
    message_id: str = ""
    smtp_response: str = ""
    attempts: int = 1
    attempted_at: datetime = Field(default_factory=_now)


class NumberSequence(SQLModel, table=True):
    scope: str = Field(primary_key=True)  # e.g. "INV-2026-#"
    last: int = 0
