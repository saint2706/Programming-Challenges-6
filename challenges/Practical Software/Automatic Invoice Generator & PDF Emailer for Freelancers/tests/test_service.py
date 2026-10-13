import json
import threading
from dataclasses import replace
from datetime import date, timedelta

import pytest
from invoicer import service
from invoicer.calc import Discount
from invoicer.db import init_db, make_engine, open_session
from invoicer.errors import InvoicerError
from invoicer.models import EmailLog, Invoice, NumberSequence
from sqlmodel import Session, select

# ------------------------------------------------------------------------------ clients


def test_client_validation(session):
    with pytest.raises(InvoicerError, match="name is required"):
        service.add_client(session, "  ")
    with pytest.raises(InvoicerError, match="valid email"):
        service.add_client(session, "A", email="not-an-email")
    with pytest.raises(InvoicerError, match="two-letter"):
        service.add_client(session, "A", country="DEU")
    with pytest.raises(InvoicerError, match="currency"):
        service.add_client(session, "A", currency="XX")
    with pytest.raises(InvoicerError, match="negative"):
        service.add_client(session, "A", terms_days=-1)


def test_duplicate_client_names_are_refused_case_insensitively(session, acme):
    with pytest.raises(InvoicerError, match="already exists"):
        service.add_client(session, "ACME gmbh")


def test_find_client_by_id_name_email_and_prefix(session, acme):
    assert service.find_client(session, str(acme.id)) is acme
    assert service.find_client(session, "acme gmbh") is acme
    assert service.find_client(session, "AP@acme.example") is acme
    assert service.find_client(session, "acm") is acme
    with pytest.raises(InvoicerError, match="no client"):
        service.find_client(session, "nobody")


def test_find_client_ambiguous_prefix_and_wildcards(session, acme):
    service.add_client(session, "Acme Ltd")
    with pytest.raises(InvoicerError, match="several"):
        service.find_client(session, "acme")
    with pytest.raises(InvoicerError, match="no client"):
        service.find_client(session, "%")  # a LIKE wildcard must not match everything


def test_edit_client_and_delete_or_archive(session, acme, make_invoice):
    service.edit_client(session, acme, email="new@acme.example", terms_days=30)
    assert (acme.email, acme.terms_days) == ("new@acme.example", 30)
    with pytest.raises(InvoicerError, match="unknown client field"):
        service.edit_client(session, acme, bogus=1)
    make_invoice()
    service.delete_client(session, acme)  # has invoices: archived, not deleted
    assert acme.archived
    other = service.add_client(session, "Temp")
    service.delete_client(session, other)
    session.flush()
    assert service.find_client(session, str(acme.id)) is acme
    with pytest.raises(InvoicerError):
        service.find_client(session, "Temp")


def test_archived_clients_cannot_be_invoiced(session, acme):
    acme.archived = True
    with pytest.raises(InvoicerError, match="archived"):
        service.create_invoice(session, acme, [service.parse_line("x | 1 | 1", "EUR")])


# --------------------------------------------------------------------------- line parsing


def test_parse_line():
    line = service.parse_line("Design | 10 | 85.00 | 19", "EUR")
    assert (line.description, line.qty_e4, line.unit_minor, line.tax_bp) == (
        "Design",
        100_000,
        8500,
        1900,
    )
    assert service.parse_line("Item | 1 | 5", "EUR").tax_bp == 0
    assert service.parse_line("Item | 1 | 5", "EUR", default_tax_bp=700).tax_bp == 700
    # The description may itself contain a pipe: only the last three fields are structural.
    assert service.parse_line("A | B | 2 | 3.5 | 20", "EUR").description == "A | B"


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "just text",
        "x | 0 | 5",
        "x | 1 | -5",
        "x | 1 | 5.005",
        "x | 1 | 5 | 101",
        "| 1 | 5",
    ],
)
def test_bad_lines(spec):
    with pytest.raises(InvoicerError):
        service.parse_line(spec, "EUR")


def test_overlong_description():
    with pytest.raises(InvoicerError, match="characters"):
        service.parse_line("x" * 501 + " | 1 | 5", "EUR")


def test_parse_discount():
    assert service.parse_discount("10%", "EUR") == Discount("percent", 1000)
    assert service.parse_discount("25.50", "EUR") == Discount("fixed", 2550)
    assert service.parse_discount(None, "EUR") is None
    assert service.parse_discount("  ", "EUR") is None
    with pytest.raises(InvoicerError):
        service.parse_discount("150%", "EUR")


# --------------------------------------------------------------------------- numbering


def test_numbers_are_sequential_and_start_per_year(session, settings, make_invoice):
    from datetime import date as d

    assert [make_invoice().number for _ in range(3)] == [
        "INV-2026-0001",
        "INV-2026-0002",
        "INV-2026-0003",
    ]
    assert make_invoice(issue_on=d(2027, 1, 2)).number == "INV-2027-0001"
    assert make_invoice().number == "INV-2026-0004"  # the 2026 run continues


def test_custom_format_and_monthly_scope(session, settings, acme):
    settings = replace(settings, number_format="{year}{month:02d}/{seq:03d}")
    line = [service.parse_line("x | 1 | 1", "EUR")]

    def issue(on):
        return service.issue_invoice(
            session, service.create_invoice(session, acme, line), settings, on
        ).number

    assert issue(date(2026, 1, 5)) == "202601/001"
    assert issue(date(2026, 1, 9)) == "202601/002"
    assert issue(date(2026, 2, 1)) == "202602/001"


def test_drafts_have_no_number_and_issuing_twice_fails(session, settings, acme):
    inv = service.create_invoice(
        session, acme, [service.parse_line("x | 1 | 1", "EUR")]
    )
    assert inv.number is None and inv.status == "draft"
    service.issue_invoice(session, inv, settings, date(2026, 3, 1))
    with pytest.raises(InvoicerError, match="already issued"):
        service.issue_invoice(session, inv, settings)


def test_a_failed_issue_leaves_no_gap(engine, settings):
    line = [service.parse_line("x | 1 | 1", "EUR")]
    with open_session(engine) as s:
        c = service.add_client(s, "Acme GmbH", currency="EUR", country="DE")
        ids = [service.create_invoice(s, c, line).id for _ in range(2)]
    with pytest.raises(RuntimeError), open_session(engine) as s:
        service.issue_invoice(s, s.get(Invoice, ids[0]), settings, date(2026, 5, 1))
        raise RuntimeError("something later in the same transaction fails")
    with open_session(engine) as s:
        inv = service.issue_invoice(
            s, s.get(Invoice, ids[1]), settings, date(2026, 5, 1)
        )
        assert (
            inv.number == "INV-2026-0001"
        )  # the rolled-back number was never consumed


def test_concurrent_issuers_never_duplicate_a_number(engine, settings):
    line = [service.parse_line("x | 1 | 1", "EUR")]
    with open_session(engine) as s:
        c = service.add_client(s, "Acme GmbH", currency="EUR", country="DE")
        ids = [service.create_invoice(s, c, line).id for _ in range(12)]
    errors = []

    def worker(invoice_id):
        try:
            with open_session(engine) as s:
                service.issue_invoice(
                    s, s.get(Invoice, invoice_id), settings, date(2026, 6, 1)
                )
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in ids]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    with open_session(engine) as s:
        numbers = sorted(i.number for i in s.exec(select(Invoice)))
    assert numbers == [f"INV-2026-{n:04d}" for n in range(1, 13)]
    with open_session(engine) as s:
        assert s.get(NumberSequence, "INV-2026-#").last == 12


def test_issue_sets_due_date_from_client_terms_and_freezes_parties(
    session, settings, acme, make_invoice
):
    acme.terms_days = 30
    inv = make_invoice()
    assert inv.due_date == date(2026, 10, 31)
    snap = json.loads(inv.snapshot)
    assert snap["seller"]["name"] == "Ada Lovelace Consulting"
    assert snap["buyer"]["name"] == "Acme GmbH"
    service.edit_client(session, acme, name="Acme Renamed", address="Elsewhere 9")
    assert (
        json.loads(inv.snapshot)["buyer"]["name"] == "Acme GmbH"
    )  # history is not rewritten


# ------------------------------------------------------------------------- payments / status


def test_payment_flow_and_statuses(session, make_invoice):
    inv = make_invoice()  # 1000.00 + 19 % = 1190.00, due 2026-10-15
    view = service.load_view(session, inv, date(2026, 10, 10))
    assert (view.status, view.balance) == ("issued", 119_000)
    service.record_payment(
        session, inv, "190", paid_on=date(2026, 10, 11), method="bank"
    )
    assert service.load_view(session, inv, date(2026, 10, 12)).status == "partial"
    assert (
        service.load_view(session, inv, date(2026, 10, 16)).status == "overdue"
    )  # partial but late
    service.record_payment(session, inv, "1000.00")
    view = service.load_view(session, inv, date(2026, 12, 1))
    assert (view.status, view.balance, view.days_overdue) == ("paid", 0, 0)


def test_payment_validation(session, make_invoice):
    inv = make_invoice()
    for bad in ("0", "-5", "abc", "1190.01", "10.005"):
        with pytest.raises(InvoicerError):
            service.record_payment(session, inv, bad)
    service.record_payment(session, inv, "1190")
    with pytest.raises(InvoicerError, match="exceeds"):
        service.record_payment(session, inv, "0.01")


def test_only_issued_invoices_take_payments(session, acme):
    draft = service.create_invoice(
        session, acme, [service.parse_line("x | 1 | 10", "EUR")]
    )
    with pytest.raises(InvoicerError, match="only issued"):
        service.record_payment(session, draft, "1")


def test_sent_status_follows_a_successful_email(session, make_invoice):
    inv = make_invoice()
    session.add(
        EmailLog(
            invoice_id=inv.id,
            kind="invoice",
            recipient="a@b.c",
            subject="s",
            status="failed",
        )
    )
    assert service.load_view(session, inv, date(2026, 10, 2)).status == "issued"
    session.add(
        EmailLog(
            invoice_id=inv.id,
            kind="invoice",
            recipient="a@b.c",
            subject="s",
            status="sent",
        )
    )
    assert service.load_view(session, inv, date(2026, 10, 2)).status == "sent"


def test_zero_value_invoice_is_paid_immediately(session, acme, settings):
    inv = service.create_invoice(
        session, acme, [service.parse_line("Freebie | 1 | 0", "EUR")]
    )
    service.issue_invoice(session, inv, settings, date(2026, 1, 1))
    assert service.load_view(session, inv, date(2026, 6, 1)).status == "paid"


def test_void_and_delete_rules(session, acme, settings, make_invoice):
    draft = service.create_invoice(
        session, acme, [service.parse_line("x | 1 | 10", "EUR")]
    )
    with pytest.raises(InvoicerError, match="only issued"):
        service.void_invoice(session, draft)
    service.delete_draft(session, draft)
    session.flush()
    assert session.get(Invoice, draft.id) is None

    inv = make_invoice()
    with pytest.raises(InvoicerError, match="only drafts"):
        service.delete_draft(session, inv)
    service.record_payment(session, inv, "10")
    with pytest.raises(InvoicerError, match="payments"):
        service.void_invoice(session, inv)

    other = make_invoice()
    service.void_invoice(session, other)
    assert service.load_view(session, other).status == "void"
    assert service.load_view(session, other).balance == 0
    assert other.number == "INV-2026-0002"  # voiding keeps the number used
    assert make_invoice().number == "INV-2026-0003"
    with pytest.raises(InvoicerError):
        service.record_payment(session, other, "1")


def test_get_invoice_by_number_and_id(session, make_invoice):
    inv = make_invoice()
    assert service.get_invoice(session, "INV-2026-0001") is inv
    assert service.get_invoice(session, str(inv.id)) is inv
    with pytest.raises(InvoicerError, match="no invoice"):
        service.get_invoice(session, "INV-1999-0001")


def test_invoice_needs_lines_and_valid_discount(session, acme):
    with pytest.raises(InvoicerError, match="at least one line"):
        service.create_invoice(session, acme, [])
    with pytest.raises(InvoicerError, match="exceed"):
        service.create_invoice(
            session,
            acme,
            [service.parse_line("x | 1 | 10", "EUR")],
            discount=Discount("fixed", 5000),
        )


def test_multi_currency_invoice(session, settings):
    client = service.add_client(session, "Tokyo KK", country="JP", currency="JPY")
    lines = [service.parse_line("Consulting | 3 | 10000 | 10", "JPY")]
    inv = service.create_invoice(session, client, lines)
    service.issue_invoice(session, inv, settings, date(2026, 4, 1))
    view = service.load_view(session, inv, date(2026, 4, 2))
    assert (inv.currency, view.totals.total) == ("JPY", 33_000)  # no decimals in yen
    with pytest.raises(InvoicerError):
        service.record_payment(session, inv, "100.5")


def test_list_filter_and_outstanding_per_currency(
    session, settings, acme, make_invoice
):
    a = make_invoice()
    b = make_invoice()
    service.record_payment(session, b, "1190")
    jp = service.add_client(session, "Tokyo KK", country="JP", currency="JPY")
    c = service.issue_invoice(
        session,
        service.create_invoice(
            session, jp, [service.parse_line("x | 1 | 5000", "JPY")]
        ),
        settings,
    )
    views = service.list_views(session, today=date(2026, 10, 5))
    assert {v.invoice.id for v in views} == {a.id, b.id, c.id}
    assert [
        v.invoice.id
        for v in service.list_views(session, status="paid", today=date(2026, 10, 5))
    ] == [b.id]
    assert [
        v.invoice.id
        for v in service.list_views(session, client=jp, today=date(2026, 10, 5))
    ] == [c.id]
    assert service.outstanding(views) == {"EUR": 119_000, "JPY": 5000}


# ---------------------------------------------------------------------------- recurring


@pytest.mark.parametrize(
    ("start", "freq", "interval", "anchor", "expected"),
    [
        (date(2026, 1, 31), "monthly", 1, 31, date(2026, 2, 28)),
        (
            date(2026, 2, 28),
            "monthly",
            1,
            31,
            date(2026, 3, 31),
        ),  # back to the anchor day
        (date(2024, 1, 31), "monthly", 1, 31, date(2024, 2, 29)),  # leap year
        (
            date(2026, 11, 15),
            "monthly",
            3,
            15,
            date(2027, 2, 15),
        ),  # across the year end
        (date(2026, 1, 31), "quarterly", 1, 31, date(2026, 4, 30)),
        (date(2024, 2, 29), "yearly", 1, 29, date(2025, 2, 28)),
        (date(2026, 10, 1), "weekly", 2, 1, date(2026, 10, 15)),
    ],
)
def test_advance_is_calendar_safe(start, freq, interval, anchor, expected):
    assert service.advance(start, freq, interval, anchor) == expected


def test_recurring_creates_due_invoices_and_is_idempotent(session, settings, acme):
    rec = service.add_recurring(
        session,
        "Retainer",
        acme,
        [service.parse_line("Retainer | 1 | 500 | 19", "EUR")],
        frequency="monthly",
        start=date(2026, 1, 31),
    )
    created = service.run_recurring(session, settings, date(2026, 3, 31))
    assert [inv.issue_date for _, inv in created] == [
        date(2026, 1, 31),
        date(2026, 2, 28),
        date(2026, 3, 31),
    ]
    assert [inv.number for _, inv in created] == [
        "INV-2026-0001",
        "INV-2026-0002",
        "INV-2026-0003",
    ]
    assert all(
        inv.recurring_id == rec.id and inv.status == "issued" for _, inv in created
    )
    assert rec.next_date == date(2026, 4, 30)
    assert (
        service.run_recurring(session, settings, date(2026, 3, 31)) == []
    )  # nothing twice


def test_recurring_catch_up_is_capped_per_run(session, settings, acme):
    rec = service.add_recurring(
        session,
        "Weekly",
        acme,
        [service.parse_line("x | 1 | 10", "EUR")],
        frequency="weekly",
        start=date(2026, 1, 1),
    )
    first = service.run_recurring(session, settings, date(2026, 12, 31), max_catchup=5)
    assert len(first) == 5
    assert rec.next_date == date(2026, 1, 1) + timedelta(weeks=5)
    assert (
        len(service.run_recurring(session, settings, date(2026, 12, 31), max_catchup=5))
        == 5
    )


def test_recurring_end_date_deactivates(session, settings, acme):
    rec = service.add_recurring(
        session,
        "Short",
        acme,
        [service.parse_line("x | 1 | 10", "EUR")],
        frequency="monthly",
        start=date(2026, 1, 1),
        end=date(2026, 2, 15),
    )
    created = service.run_recurring(session, settings, date(2026, 6, 1))
    assert len(created) == 2
    assert not rec.active
    assert service.run_recurring(session, settings, date(2026, 7, 1)) == []


def test_recurring_copies_discount_and_uses_frozen_prices(session, settings, acme):
    service.add_recurring(
        session,
        "Disc",
        acme,
        [service.parse_line("x | 2 | 100 | 20", "EUR")],
        frequency="monthly",
        start=date(2026, 1, 1),
        discount=Discount("percent", 1000),
    )
    ((_, inv),) = service.run_recurring(session, settings, date(2026, 1, 1))
    assert service.load_view(session, inv).totals.total == 21_600  # (200 - 20) * 1.2


def test_recurring_validation(session, acme):
    line = [service.parse_line("x | 1 | 1", "EUR")]
    with pytest.raises(InvoicerError, match="frequency"):
        service.add_recurring(
            session, "A", acme, line, frequency="daily", start=date(2026, 1, 1)
        )
    with pytest.raises(InvoicerError, match="interval"):
        service.add_recurring(
            session,
            "A",
            acme,
            line,
            frequency="monthly",
            start=date(2026, 1, 1),
            interval=0,
        )
    with pytest.raises(InvoicerError, match="before"):
        service.add_recurring(
            session,
            "A",
            acme,
            line,
            frequency="monthly",
            start=date(2026, 2, 1),
            end=date(2026, 1, 1),
        )
    service.add_recurring(
        session, "A", acme, line, frequency="monthly", start=date(2026, 1, 1)
    )
    with pytest.raises(InvoicerError, match="already exists"):
        service.add_recurring(
            session, "A", acme, line, frequency="monthly", start=date(2026, 1, 1)
        )


# ---------------------------------------------------------------------------- reminders


def _log(session, inv, kind, status, stage=None):
    session.add(
        EmailLog(
            invoice_id=inv.id,
            kind=kind,
            stage=stage,
            recipient="a@b.c",
            subject="s",
            status=status,
        )
    )
    session.flush()


def test_reminder_stages(session, settings, make_invoice):
    inv = make_invoice()  # due 2026-10-15; stages 7, 14, 30 days after due
    due = lambda day: [(r.stage) for r in service.due_reminders(session, settings, day)]
    assert due(date(2026, 10, 15)) == []  # due today is not overdue
    assert due(date(2026, 10, 21)) == []  # 6 days late: before the first stage
    assert due(date(2026, 10, 22)) == [7]
    _log(session, inv, "reminder", "sent", 7)
    assert due(date(2026, 10, 25)) == []  # stage 7 done, 14 not reached
    assert due(date(2026, 10, 29)) == [14]
    # A long silence: only the highest stage reached is sent, once, and earlier ones are skipped.
    assert due(date(2026, 12, 31)) == [30]
    _log(session, inv, "reminder", "sent", 30)
    assert due(date(2027, 3, 1)) == []


def test_failed_reminders_are_retried(session, settings, make_invoice):
    inv = make_invoice()
    _log(session, inv, "reminder", "failed", 7)
    assert [
        r.stage for r in service.due_reminders(session, settings, date(2026, 10, 23))
    ] == [7]


def test_paid_partial_and_void_invoices_get_no_reminders(
    session, settings, make_invoice
):
    paid, partial, void = make_invoice(), make_invoice(), make_invoice()
    service.record_payment(session, paid, "1190")
    service.record_payment(session, partial, "10")
    service.void_invoice(session, void)
    assert [
        r.view.invoice.id
        for r in service.due_reminders(session, settings, date(2026, 11, 30))
    ] == [partial.id]


def test_session_helper_rolls_back_on_error(engine):
    with pytest.raises(RuntimeError), open_session(engine) as s:
        service.add_client(s, "Ghost")
        raise RuntimeError("boom")
    with open_session(engine) as s:
        assert service.list_views(s) == []
        with pytest.raises(InvoicerError):
            service.find_client(s, "Ghost")


def test_foreign_keys_are_enforced(tmp_path):
    from invoicer.models import Payment
    from sqlalchemy.exc import IntegrityError

    engine = make_engine(tmp_path / "fk.db")
    init_db(engine)
    with pytest.raises(IntegrityError), Session(engine) as s:
        s.add(Payment(invoice_id=999, amount_minor=1, paid_on=date(2026, 1, 1)))
        s.commit()
