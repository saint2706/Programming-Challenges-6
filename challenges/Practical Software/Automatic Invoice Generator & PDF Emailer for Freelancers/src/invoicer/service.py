"""Business rules over the database: clients, invoices, numbering, payments, recurring
invoices and reminders. The CLI and the web UI are thin layers over these functions."""

import calendar
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from email.utils import parseaddr

from sqlalchemy import func, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select

from invoicer import calc
from invoicer.calc import Discount, LineInput, Totals
from invoicer.config import Settings, number_scope
from invoicer.errors import InvoicerError
from invoicer.models import (
    DRAFT,
    ISSUED,
    VOID,
    Client,
    EmailLog,
    Invoice,
    LineItem,
    Payment,
    Recurring,
)
from invoicer.money import (
    MoneyError,
    format_percent,
    normalize_currency,
    parse_percent,
    parse_quantity,
    to_minor,
)

# Deliberately narrower than RFC 5322: no quoting, comments, groups or routes, which are
# only ever a way to make a header mean something else.
_BAD = r"@\s<>,;:()\[\]\\\"'\x00-\x1f"
EMAIL_RE = re.compile(rf"[^{_BAD}]+@[^{_BAD}]+\.[^{_BAD}]+")
FREQUENCIES = ("weekly", "monthly", "quarterly", "yearly")
MAX_DESCRIPTION = 500


# --------------------------------------------------------------------------- clients


def _same(column, value: str):
    """Case-insensitive equality. Not ILIKE: that would treat ``%`` and ``_`` in a name as
    wildcards."""
    return func.lower(column) == value.lower()


def validate_email(address: str) -> str:
    address = address.strip()
    if not EMAIL_RE.fullmatch(address) or parseaddr(address)[1] != address:
        raise InvoicerError(f"{address!r} is not a valid email address")
    return address


def add_client(
    session: Session,
    name: str,
    *,
    email: str = "",
    address: str = "",
    country: str = "US",
    tax_id: str = "",
    currency: str = "USD",
    terms_days: int = 14,
    notes: str = "",
) -> Client:
    name = name.strip()
    if not name:
        raise InvoicerError("client name is required")
    if session.exec(select(Client).where(_same(Client.name, name))).first():
        raise InvoicerError(f"a client named {name!r} already exists")
    client = Client(name=name, address=address, tax_id=tax_id, notes=notes)
    return _fill_client(client, email, country, currency, terms_days, session)


def _fill_client(
    client: Client,
    email: str,
    country: str,
    currency: str,
    terms_days: int,
    session: Session,
) -> Client:
    client.email = validate_email(email) if email.strip() else ""
    country = country.strip().upper()
    if len(country) != 2 or not country.isalpha():
        raise InvoicerError("country must be a two-letter ISO 3166 code")
    client.country = country
    try:
        client.currency = normalize_currency(currency)
    except MoneyError as exc:
        raise InvoicerError(str(exc)) from exc
    if terms_days < 0:
        raise InvoicerError("payment terms cannot be negative")
    client.terms_days = terms_days
    session.add(client)
    session.flush()
    return client


def edit_client(session: Session, client: Client, **changes: object) -> Client:
    allowed = {
        "name",
        "email",
        "address",
        "country",
        "tax_id",
        "currency",
        "terms_days",
        "notes",
    }
    if unknown := set(changes) - allowed:
        raise InvoicerError(f"unknown client field(s): {', '.join(sorted(unknown))}")
    values = {k: v for k, v in changes.items() if v is not None}
    name = str(values.pop("name", client.name)).strip()
    if not name:
        raise InvoicerError("client name is required")
    other = session.exec(select(Client).where(_same(Client.name, name))).first()
    if other and other.id != client.id:
        raise InvoicerError(f"a client named {name!r} already exists")
    client.name = name
    for key in ("address", "tax_id", "notes"):
        if key in values:
            setattr(client, key, str(values[key]))
    return _fill_client(
        client,
        str(values.get("email", client.email)),
        str(values.get("country", client.country)),
        str(values.get("currency", client.currency)),
        int(values.get("terms_days", client.terms_days)),
        session,
    )


def find_client(session: Session, key: str) -> Client:
    """By id, exact name (any case) or unique email."""
    key = key.strip()
    if key.isdigit() and (client := session.get(Client, int(key))):
        return client
    matches = session.exec(select(Client).where(_same(Client.name, key))).all()
    if not matches:
        matches = session.exec(select(Client).where(_same(Client.email, key))).all()
    if not matches:  # a unique name prefix is enough ("acme" for "Acme GmbH")
        matches = session.exec(
            select(Client).where(col(Client.name).istartswith(key, autoescape=True))
        ).all()
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise InvoicerError(f"{key!r} matches several clients; use the id")
    raise InvoicerError(f"no client {key!r}")


def delete_client(session: Session, client: Client) -> None:
    used = session.exec(select(Invoice).where(Invoice.client_id == client.id)).first()
    scheduled = session.exec(
        select(Recurring).where(Recurring.client_id == client.id)
    ).first()
    if used or scheduled:
        client.archived = True  # keep the history; just hide the client
        session.add(client)
        return
    session.delete(client)


# --------------------------------------------------------------------------- invoices


@dataclass(frozen=True)
class LineSpec:
    description: str
    qty_e4: int
    unit_minor: int
    tax_bp: int = 0


def parse_line(spec: str, currency: str, default_tax_bp: int = 0) -> LineSpec:
    """``"Design work | 10 | 85.00 | 20"``: description | quantity | unit price | tax % (optional)."""
    parts = [p.strip() for p in spec.rsplit("|", 3)]
    if len(parts) == 3:
        parts.append("")
    if len(parts) != 4 or not parts[0]:
        raise InvoicerError(
            f"line {spec!r}: expected 'description | quantity | price | tax %'"
        )
    description, qty, price, tax = parts
    try:
        return make_line(description, qty, price, tax or None, currency, default_tax_bp)
    except MoneyError as exc:
        raise InvoicerError(f"line {spec!r}: {exc}") from exc


def make_line(
    description: str,
    qty: str | int,
    price: str | int,
    tax_percent: str | int | None,
    currency: str,
    default_tax_bp: int = 0,
) -> LineSpec:
    description = description.strip()
    if not description:
        raise InvoicerError("a line needs a description")
    if len(description) > MAX_DESCRIPTION:
        raise InvoicerError(f"line description is over {MAX_DESCRIPTION} characters")
    try:
        unit = to_minor(price, currency)
        if unit < 0:
            raise MoneyError("unit price must not be negative")
        tax = (
            default_tax_bp if tax_percent in (None, "") else parse_percent(tax_percent)
        )
        return LineSpec(description, parse_quantity(qty), unit, tax)
    except MoneyError as exc:
        raise InvoicerError(str(exc)) from exc


def parse_discount(text_value: str | None, currency: str) -> Discount | None:
    """``"10%"`` is a percentage, ``"25.00"`` a fixed amount."""
    if not text_value or not text_value.strip():
        return None
    value = text_value.strip()
    try:
        if value.endswith("%"):
            return Discount("percent", parse_percent(value[:-1]))
        return Discount("fixed", to_minor(value, currency))
    except MoneyError as exc:
        raise InvoicerError(f"discount {text_value!r}: {exc}") from exc


def _discount_of(inv: Invoice | Recurring) -> Discount | None:
    if inv.discount_kind is None:
        return None
    return Discount(inv.discount_kind, inv.discount_value)  # type: ignore[arg-type]


def create_invoice(
    session: Session,
    client: Client,
    lines: Sequence[LineSpec],
    *,
    currency: str | None = None,
    discount: Discount | None = None,
    notes: str = "",
    reference: str = "",
    recurring_id: int | None = None,
) -> Invoice:
    if client.archived:
        raise InvoicerError(f"client {client.name!r} is archived")
    if not lines:
        raise InvoicerError("an invoice needs at least one line")
    try:
        currency = normalize_currency(currency or client.currency)
        calc.compute(
            [LineInput(x.qty_e4, x.unit_minor, x.tax_bp) for x in lines], discount
        )
    except MoneyError as exc:
        raise InvoicerError(str(exc)) from exc
    inv = Invoice(
        client_id=client.id,
        currency=currency,
        discount_kind=discount.kind if discount else None,
        discount_value=discount.value if discount else 0,
        notes=notes,
        reference=reference,
        recurring_id=recurring_id,
    )
    session.add(inv)
    session.flush()
    for pos, spec in enumerate(lines, 1):
        session.add(LineItem(invoice_id=inv.id, position=pos, **asdict(spec)))
    session.flush()
    return inv


def get_invoice(session: Session, key: str) -> Invoice:
    """By number (``INV-2026-0001``) or numeric id (drafts have no number yet)."""
    key = key.strip()
    inv = session.exec(select(Invoice).where(Invoice.number == key)).first()
    if inv is None and key.isdigit():
        inv = session.get(Invoice, int(key))
    if inv is None:
        raise InvoicerError(f"no invoice {key!r}")
    return inv


def invoice_lines(session: Session, inv: Invoice) -> list[LineItem]:
    return list(
        session.exec(
            select(LineItem)
            .where(LineItem.invoice_id == inv.id)
            .order_by(LineItem.position)
        )
    )


def compute_totals(inv: Invoice, lines: Sequence[LineItem]) -> Totals:
    return calc.compute(
        [LineInput(x.qty_e4, x.unit_minor, x.tax_bp) for x in lines], _discount_of(inv)
    )


def payments_of(session: Session, inv: Invoice) -> list[Payment]:
    return list(
        session.exec(
            select(Payment)
            .where(Payment.invoice_id == inv.id)
            .order_by(Payment.paid_on, Payment.id)
        )
    )


def lock_invoice(session: Session, inv: Invoice) -> Invoice:
    """Take SQLite's write lock and reload the row, so a check made next cannot go stale.

    Two processes (or a double-clicked button) can both load a draft and both decide to issue
    it; whoever holds the lock first wins and the other re-reads the new status.
    """
    session.execute(text("UPDATE invoice SET id = id WHERE id = :i"), {"i": inv.id})
    session.refresh(inv)
    return inv


def issue_invoice(
    session: Session, inv: Invoice, settings: Settings, issue_date: date | None = None
) -> Invoice:
    """Draft -> issued: assigns the next number and freezes seller/buyer details.

    The counter update, the number and the status change commit together, so a failure anywhere
    leaves no gap in the sequence; the unique constraint on ``number`` backs this up if two
    processes ever race.
    """
    lock_invoice(session, inv)
    if inv.status != DRAFT:
        raise InvoicerError(f"invoice {inv.number or inv.id} is already {inv.status}")
    client = session.get(Client, inv.client_id)
    issue_date = issue_date or date.today()
    inv.number = _next_number(session, settings, issue_date)
    inv.issue_date = issue_date
    inv.due_date = issue_date + timedelta(days=client.terms_days)
    inv.status = ISSUED
    inv.snapshot = json.dumps(
        {
            "seller": _seller_snapshot(settings),
            "buyer": {
                "name": client.name,
                "address": client.address,
                "country": client.country,
                "email": client.email,
                "tax_id": client.tax_id,
            },
        }
    )
    session.add(inv)
    try:
        session.flush()
    except IntegrityError as exc:
        raise InvoicerError(
            f"invoice number {inv.number} is already taken; retry"
        ) from exc
    return inv


def _seller_snapshot(settings: Settings) -> dict:
    b = settings.business
    return {
        k: getattr(b, k)
        for k in (
            "name",
            "address",
            "country",
            "email",
            "tax_id",
            "iban",
            "bic",
            "payment_note",
        )
    }


def _next_number(session: Session, settings: Settings, issue_date: date) -> str:
    scope = number_scope(settings.number_format, issue_date.year, issue_date.month)
    # INSERT OR IGNORE then UPDATE takes SQLite's write lock before the counter is read, so two
    # issuers are serialised and the read below sees this transaction's own increment.
    session.execute(
        text("INSERT OR IGNORE INTO numbersequence (scope, last) VALUES (:s, 0)"),
        {"s": scope},
    )
    session.execute(
        text("UPDATE numbersequence SET last = last + 1 WHERE scope = :s"), {"s": scope}
    )
    seq = session.execute(
        text("SELECT last FROM numbersequence WHERE scope = :s"), {"s": scope}
    ).scalar_one()
    return settings.number_format.format(
        year=issue_date.year, month=issue_date.month, seq=seq
    )


def record_payment(
    session: Session,
    inv: Invoice,
    amount: str,
    *,
    paid_on: date | None = None,
    method: str = "",
    note: str = "",
) -> Payment:
    lock_invoice(session, inv)  # the balance check below must see every earlier payment
    if inv.status != ISSUED:
        raise InvoicerError(
            f"only issued invoices take payments (this one is {inv.status})"
        )
    try:
        minor = to_minor(amount, inv.currency)
    except MoneyError as exc:
        raise InvoicerError(str(exc)) from exc
    if minor <= 0:
        raise InvoicerError("payment amount must be greater than zero")
    totals = compute_totals(inv, invoice_lines(session, inv))
    balance = totals.total - sum(p.amount_minor for p in payments_of(session, inv))
    if minor > balance:
        raise InvoicerError(
            "payment exceeds the balance due; overpayments are not recorded"
        )
    payment = Payment(
        invoice_id=inv.id,
        amount_minor=minor,
        paid_on=paid_on or date.today(),
        method=method,
        note=note,
    )
    session.add(payment)
    session.flush()
    return payment


def void_invoice(session: Session, inv: Invoice) -> Invoice:
    """Cancel an issued invoice. Its number stays used, so the sequence has no gap."""
    lock_invoice(session, inv)  # a payment recorded a moment ago must block the void
    if inv.status != ISSUED:
        raise InvoicerError(
            f"only issued invoices can be voided (this one is {inv.status})"
        )
    if payments_of(session, inv):
        raise InvoicerError("an invoice with payments cannot be voided")
    inv.status = VOID
    session.add(inv)
    return inv


def delete_draft(session: Session, inv: Invoice) -> None:
    if inv.status != DRAFT:
        raise InvoicerError(
            "only drafts can be deleted; void an issued invoice instead"
        )
    for line in invoice_lines(session, inv):
        session.delete(line)
    # Without relationship() the unit of work does not order deletes by foreign key, so the
    # lines must be gone before the invoice row is.
    session.flush()
    session.delete(inv)


# --------------------------------------------------------------------------- views


@dataclass(frozen=True)
class InvoiceView:
    invoice: Invoice
    client: Client
    lines: list[LineItem]
    totals: Totals
    payments: list[Payment]
    emails: list[EmailLog]
    paid: int
    balance: int
    status: str  # draft | issued | sent | partial | overdue | paid | void
    days_overdue: int


def effective_status(
    inv: Invoice, total: int, paid: int, sent: bool, today: date
) -> str:
    if inv.status in (DRAFT, VOID):
        return inv.status
    if paid >= total:
        return "paid"
    if inv.due_date and inv.due_date < today:
        return "overdue"
    if paid > 0:
        return "partial"
    return "sent" if sent else "issued"


def load_view(session: Session, inv: Invoice, today: date | None = None) -> InvoiceView:
    today = today or date.today()
    lines = invoice_lines(session, inv)
    totals = compute_totals(inv, lines)
    payments = payments_of(session, inv)
    emails = list(
        session.exec(
            select(EmailLog)
            .where(EmailLog.invoice_id == inv.id)
            .order_by(EmailLog.attempted_at)
        )
    )
    paid = sum(p.amount_minor for p in payments)
    sent = any(e.status == "sent" and e.kind == "invoice" for e in emails)
    status = effective_status(inv, totals.total, paid, sent, today)
    overdue = (today - inv.due_date).days if inv.due_date and status == "overdue" else 0
    return InvoiceView(
        inv,
        session.get(Client, inv.client_id),
        lines,
        totals,
        payments,
        emails,
        paid,
        0 if inv.status == VOID else totals.total - paid,  # a void invoice owes nothing
        status,
        overdue,
    )


def list_views(
    session: Session,
    *,
    status: str | None = None,
    client: Client | None = None,
    today: date | None = None,
) -> list[InvoiceView]:
    query = select(Invoice).order_by(col(Invoice.id).desc())
    if client is not None:
        query = query.where(Invoice.client_id == client.id)
    views = [load_view(session, inv, today) for inv in session.exec(query)]
    return [v for v in views if status is None or v.status == status]


def outstanding(views: Sequence[InvoiceView]) -> dict[str, int]:
    """Money still owed on issued invoices, per currency."""
    totals: dict[str, int] = {}
    for v in views:
        if v.invoice.status == ISSUED and v.balance > 0:
            totals[v.invoice.currency] = totals.get(v.invoice.currency, 0) + v.balance
    return totals


# --------------------------------------------------------------------------- recurring


def add_months(day: date, months: int, anchor_day: int) -> date:
    """Move ``months`` forward, clamping to the month's length but remembering the anchor day,
    so a schedule on the 31st goes Jan 31, Feb 28, Mar 31 rather than drifting to the 28th."""
    year, month0 = divmod(day.year * 12 + day.month - 1 + months, 12)
    month = month0 + 1
    return date(year, month, min(anchor_day, calendar.monthrange(year, month)[1]))


def advance(day: date, frequency: str, interval: int, anchor_day: int) -> date:
    if frequency == "weekly":
        return day + timedelta(weeks=interval)
    months = {"monthly": 1, "quarterly": 3, "yearly": 12}[frequency] * interval
    return add_months(day, months, anchor_day)


def add_recurring(
    session: Session,
    name: str,
    client: Client,
    lines: Sequence[LineSpec],
    *,
    frequency: str,
    start: date,
    interval: int = 1,
    end: date | None = None,
    discount: Discount | None = None,
    auto_send: bool = False,
    notes: str = "",
    currency: str | None = None,
) -> Recurring:
    if frequency not in FREQUENCIES:
        raise InvoicerError(f"frequency must be one of {', '.join(FREQUENCIES)}")
    if interval < 1:
        raise InvoicerError("interval must be at least 1")
    if end is not None and end < start:
        raise InvoicerError("end date is before the start date")
    if not name.strip():
        raise InvoicerError("a recurring invoice needs a name")
    if session.exec(select(Recurring).where(Recurring.name == name.strip())).first():
        raise InvoicerError(f"a recurring invoice named {name!r} already exists")
    currency = normalize_currency(currency or client.currency)
    try:
        calc.compute(
            [LineInput(x.qty_e4, x.unit_minor, x.tax_bp) for x in lines], discount
        )
    except MoneyError as exc:
        raise InvoicerError(str(exc)) from exc
    if not lines:
        raise InvoicerError("a recurring invoice needs at least one line")
    rec = Recurring(
        name=name.strip(),
        client_id=client.id,
        frequency=frequency,
        interval=interval,
        anchor_day=start.day,
        next_date=start,
        end_date=end,
        auto_send=auto_send,
        currency=currency,
        terms_days=client.terms_days,
        discount_kind=discount.kind if discount else None,
        discount_value=discount.value if discount else 0,
        notes=notes,
        lines_json=json.dumps([asdict(x) for x in lines]),
    )
    session.add(rec)
    session.flush()
    return rec


def run_recurring(
    session: Session, settings: Settings, as_of: date, max_catchup: int = 12
) -> list[tuple[Recurring, Invoice]]:
    """Create and issue every invoice that is due on or before ``as_of``.

    Each template's ``next_date`` moves forward in the same transaction as the invoice, so
    running this twice creates nothing the second time. After a long gap at most ``max_catchup``
    invoices per template are created per run; the next run continues.
    """
    created: list[tuple[Recurring, Invoice]] = []
    templates = session.exec(select(Recurring).where(Recurring.active)).all()
    for rec in templates:
        count = 0
        client = session.get(Client, rec.client_id)
        while rec.next_date <= as_of and count < max_catchup:
            if rec.end_date and rec.next_date > rec.end_date:
                break
            lines = [LineSpec(**d) for d in json.loads(rec.lines_json)]
            inv = create_invoice(
                session,
                client,
                lines,
                currency=rec.currency,
                discount=_discount_of(rec),
                notes=rec.notes,
                recurring_id=rec.id,
            )
            issue_invoice(session, inv, settings, rec.next_date)
            created.append((rec, inv))
            rec.next_date = advance(
                rec.next_date, rec.frequency, rec.interval, rec.anchor_day
            )
            count += 1
        if rec.end_date and rec.next_date > rec.end_date:
            rec.active = False
        session.add(rec)
    session.flush()
    return created


# --------------------------------------------------------------------------- reminders


@dataclass(frozen=True)
class Reminder:
    view: InvoiceView
    stage: int  # the days-after-due threshold that was reached


def due_reminders(session: Session, settings: Settings, as_of: date) -> list[Reminder]:
    """Overdue invoices that have reached a reminder stage not yet sent.

    One reminder per invoice per run, for the highest stage reached, and only ``sent`` emails
    count: a failed attempt is retried on the next run instead of being skipped for good.
    """
    result = []
    for view in list_views(session, today=as_of):
        if view.status != "overdue":
            continue
        reached = [d for d in settings.reminder_days if d <= view.days_overdue]
        already = [
            e.stage
            for e in view.emails
            if e.kind == "reminder" and e.status == "sent" and e.stage is not None
        ]
        last = max(already, default=0)
        pending = [d for d in reached if d > last]
        if pending:
            result.append(Reminder(view, max(pending)))
    return result


def describe_discount(inv: Invoice, currency: str | None = None) -> str:
    from invoicer.money import plain_amount

    if inv.discount_kind == "percent":
        return f"{format_percent(inv.discount_value)}%"
    if inv.discount_kind == "fixed":
        return plain_amount(inv.discount_value, currency or inv.currency)
    return ""
