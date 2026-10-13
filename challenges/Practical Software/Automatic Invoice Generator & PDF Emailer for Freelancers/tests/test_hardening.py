"""Regression tests for problems found in the security review of the first version."""

import threading
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest
from invoicer import mailer, service
from invoicer.cli import main
from invoicer.db import open_session
from invoicer.errors import InvoicerError
from invoicer.models import EmailLog, Invoice
from invoicer.money import MoneyError, parse_quantity, to_minor
from invoicer.web import create_app
from litestar.testing import TestClient
from smtpserver import smtp_server
from sqlmodel import select

PDF = b"%PDF-1.7 fake"


def run(*argv):
    return main([str(a) for a in argv])


# ------------------------------------------------------------------ atomic state changes


def test_two_sessions_cannot_both_issue_the_same_draft(engine, settings):
    line = [service.parse_line("x | 1 | 10", "EUR")]
    with open_session(engine) as s:
        client = service.add_client(s, "Acme", currency="EUR")
        inv_id = service.create_invoice(s, client, line).id
    results, barrier = [], threading.Barrier(2)

    def issue():
        with open_session(engine) as s:
            inv = s.get(Invoice, inv_id)  # both load it as a draft
            barrier.wait()
            try:
                service.issue_invoice(s, inv, settings, date(2026, 5, 1))
                results.append("issued")
            except InvoicerError as exc:
                results.append(str(exc))

    threads = [threading.Thread(target=issue) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(r[:6] for r in results) == ["invoic", "issued"]
    with open_session(engine) as s:
        inv = s.get(Invoice, inv_id)
        assert inv.number == "INV-2026-0001"
        # The loser did not burn a number: the next invoice is 0002, not 0003.
        other = service.create_invoice(s, service.find_client(s, "acme"), line)
        assert (
            service.issue_invoice(s, other, settings, date(2026, 5, 2)).number
            == "INV-2026-0002"
        )


def test_payments_cannot_overshoot_when_recorded_concurrently(engine, settings):
    line = [service.parse_line("x | 1 | 100", "EUR")]
    with open_session(engine) as s:
        client = service.add_client(s, "Acme", currency="EUR")
        inv = service.issue_invoice(
            s, service.create_invoice(s, client, line), settings
        )
        inv_id = inv.id
    outcomes, barrier = [], threading.Barrier(4)

    def pay():
        with open_session(engine) as s:
            inv = s.get(Invoice, inv_id)
            barrier.wait()
            try:
                service.record_payment(s, inv, "60")
                outcomes.append("ok")
            except InvoicerError:
                outcomes.append("refused")

    threads = [threading.Thread(target=pay) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert outcomes.count("ok") == 1  # 60 + 60 would exceed the 100.00 balance
    with open_session(engine) as s:
        assert service.load_view(s, s.get(Invoice, inv_id)).paid == 6000


# ------------------------------------------------------------------- duplicate email sends


def view_of(session, inv):
    return service.load_view(session, inv, date(2026, 10, 5))


def test_a_pending_claim_blocks_a_second_send(session, settings, make_invoice):
    inv = make_invoice()
    session.add(
        EmailLog(
            invoice_id=inv.id,
            kind="invoice",
            recipient="a@b.cd",
            subject="s",
            status="pending",
        )
    )
    session.flush()
    with pytest.raises(InvoicerError, match="already being sent"):
        mailer.deliver(session, view_of(session, inv), settings, PDF)


def test_a_stale_claim_from_a_crashed_process_expires(session, settings, make_invoice):
    inv = make_invoice()
    old = datetime.now(UTC) - mailer.PENDING_TTL - timedelta(minutes=1)
    session.add(
        EmailLog(
            invoice_id=inv.id,
            kind="invoice",
            recipient="a@b.cd",
            subject="s",
            status="pending",
            attempted_at=old,
        )
    )
    session.flush()
    with smtp_server() as server:
        cfg = replace(
            settings, email=replace(settings.email, host=server.host, port=server.port)
        )
        assert mailer.deliver(session, view_of(session, inv), cfg, PDF).status == "sent"


def test_already_sent_needs_an_explicit_resend(session, settings, make_invoice):
    inv = make_invoice()
    with smtp_server() as server:
        cfg = replace(
            settings, email=replace(settings.email, host=server.host, port=server.port)
        )
        assert mailer.deliver(session, view_of(session, inv), cfg, PDF).status == "sent"
        with pytest.raises(InvoicerError, match="already sent"):
            mailer.deliver(session, view_of(session, inv), cfg, PDF)
        assert (
            mailer.deliver(session, view_of(session, inv), cfg, PDF, resend=True).status
            == "sent"
        )
        assert len(server.messages) == 2


def test_concurrent_deliveries_send_exactly_one_message(engine, settings):
    line = [service.parse_line("x | 1 | 10", "EUR")]
    with open_session(engine) as s:
        client = service.add_client(s, "Acme", email="ap@acme.example", currency="EUR")
        inv_id = service.issue_invoice(
            s, service.create_invoice(s, client, line), settings
        ).id
    results, barrier = [], threading.Barrier(2)
    with smtp_server(
        delay=1.0
    ) as server:  # the first send is still in flight when the second starts
        cfg = replace(
            settings, email=replace(settings.email, host=server.host, port=server.port)
        )

        def send():
            with open_session(engine) as s:
                view = service.load_view(s, s.get(Invoice, inv_id))
                barrier.wait()
                try:
                    results.append(mailer.deliver(s, view, cfg, PDF).status)
                except InvoicerError as exc:
                    results.append(str(exc))

        threads = [threading.Thread(target=send) for _ in range(2)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(server.messages) == 1
    assert "sent" in results and len(results) == 2
    with open_session(engine) as s:
        assert [r.status for r in s.exec(select(EmailLog))] == ["sent"]


def test_an_interrupted_send_does_not_leave_a_blocking_claim(
    session, settings, make_invoice, monkeypatch
):
    inv = make_invoice()

    def boom(cfg, msg):
        raise KeyboardInterrupt

    monkeypatch.setattr(mailer, "send_message", boom)
    with pytest.raises(KeyboardInterrupt):
        mailer.deliver(session, view_of(session, inv), settings, PDF)
    (row,) = session.exec(select(EmailLog)).all()
    assert row.status == "failed" and "interrupted" in row.error
    monkeypatch.undo()
    with smtp_server() as server:
        cfg = replace(
            settings, email=replace(settings.email, host=server.host, port=server.port)
        )
        assert mailer.deliver(session, view_of(session, inv), cfg, PDF).status == "sent"


def test_mail_uses_the_issue_time_seller_like_the_pdf(session, settings, make_invoice):
    inv = make_invoice()
    renamed = replace(
        settings,
        business=replace(settings.business, name="Someone Else", payment_note="NEW"),
    )
    msg = mailer.build_message(view_of(session, inv), renamed, PDF)
    body = msg.get_body().get_content()
    assert "Ada Lovelace Consulting" in msg["Subject"] and "Someone Else" not in body
    assert "Please pay by bank transfer." in body and "NEW" not in body


# ------------------------------------------------------------------------ input validation


@pytest.mark.parametrize(
    "address",
    [
        "a(b@c.co",
        "grp:evil@x.co",
        "a@b.co;c@d.co",
        "a\x00b@c.co",
        "a\\b@c.co",
        "a[b]@c.co",
        '"a b"@c.co',
        "a@b",
        "<a@b.co>",
    ],
)
def test_odd_addresses_are_refused(address):
    with pytest.raises(InvoicerError):
        service.validate_email(address)


@pytest.mark.parametrize(
    "address", ["ada@example.com", "first.last+tag@sub.example.org", "x_y-z@a-b.co.uk"]
)
def test_ordinary_addresses_pass(address):
    assert service.validate_email(address) == address


def test_absurd_amounts_are_refused_not_silently_altered():
    for bad in (
        "100000000000000000000000000000.01",
        "1e30",
        "1e17",
        "99999999999999.00",
    ):
        with pytest.raises(MoneyError, match="too large"):
            to_minor(bad, "USD")
    assert to_minor("9999999999999.99", "USD") == 999_999_999_999_999
    with pytest.raises(MoneyError, match="too large"):
        parse_quantity("100000001")


def test_largest_allowed_invoice_computes_exactly(session, settings, acme):
    """10**8 units at 10**13 each is 10**21: far beyond Decimal's default 28 digits once tax is
    applied, and beyond SQLite's 64-bit integers, yet it must neither lose a cent nor crash."""
    from invoicer.calc import LineInput, compute

    big = compute([LineInput(10**12, 10**15, 1900)])
    assert (
        big.net == 10**23
        and big.tax == 19 * 10**21 + 10**21 * 0
        and big.total == big.net + big.tax
    )
    with pytest.raises(InvoicerError):
        service.record_payment(
            session,
            service.issue_invoice(
                session,
                service.create_invoice(
                    session, acme, [service.parse_line("x | 1 | 10", "EUR")]
                ),
                settings,
            ),
            "1e30",
        )


def test_money_errors_are_clean_cli_errors(_isolated_home, capsys):
    run("client", "add", "Acme")
    for argv in (
        ["invoice", "new", "--client", "acme", "--item", "x | 1 | 1e30"],
        ["invoice", "new", "--client", "acme", "--item", "x | 1e9 | 5"],
    ):
        assert run(*argv) == 1
        err = capsys.readouterr().err
        assert err.startswith("error:") and "Traceback" not in err


def test_a_missing_job_file_is_a_clean_error(_isolated_home, capsys, tmp_path):
    run("client", "add", "Acme")
    assert (
        run("invoice", "new", "--client", "acme", "--file", tmp_path / "nope.toml") == 1
    )
    assert capsys.readouterr().err.startswith("error: cannot read")


def test_a_client_with_a_recurring_invoice_is_archived_not_deleted(
    session, settings, acme
):
    service.add_recurring(
        session,
        "R",
        acme,
        [service.parse_line("x | 1 | 10", "EUR")],
        frequency="monthly",
        start=date(2026, 1, 1),
    )
    service.delete_client(session, acme)
    session.flush()
    assert acme.archived


def test_a_void_invoice_owes_nothing(session, make_invoice):
    inv = make_invoice()
    service.void_invoice(session, inv)
    assert service.load_view(session, inv).balance == 0


# ----------------------------------------------------------------------- web / serve


@pytest.fixture
def web(settings, engine):
    with TestClient(create_app(settings), base_url="http://localhost:8000") as client:
        yield client


def test_the_csrf_cookie_is_not_readable_by_scripts(web):
    response = web.get("/invoices")
    cookie = response.headers["set-cookie"].lower()
    assert "csrftoken=" in cookie and "httponly" in cookie and "samesite=lax" in cookie


def test_cross_origin_posts_are_refused_even_with_a_valid_token(
    web, make_invoice, session
):
    inv = make_invoice()
    session.commit()
    web.get("/invoices")
    token = web.cookies["csrftoken"]
    headers = {"x-csrftoken": token}
    evil = web.post(
        f"/invoices/{inv.id}/void",
        headers={**headers, "origin": "http://localhost:3000"},
    )
    assert evil.status_code == 403
    same = web.post(
        f"/invoices/{inv.id}/void",
        headers={**headers, "origin": "http://localhost:8000"},
    )
    assert same.status_code == 200
    noheader = web.post(
        f"/invoices/{inv.id}/pay", headers=headers, data={"amount": "1"}
    )
    assert noheader.status_code in (
        200,
        409,
    )  # no Origin (curl, tests): token alone suffices


def test_buttons_disable_while_a_request_is_in_flight(web, make_invoice, session):
    inv = make_invoice()
    session.commit()
    html = web.get(f"/invoices/{inv.id}").text
    assert html.count('hx-disabled-elt="find button"') >= 3
    assert '"allowEval":false' in web.get("/invoices").text


def test_serve_refuses_non_loopback_hosts(_isolated_home, capsys):
    for host in ("0.0.0.0", "192.168.1.5", "example.com", "::"):
        assert run("serve", "--host", host) == 1
        assert "refusing to listen" in capsys.readouterr().err


def test_config_check_warns_about_plaintext_credentials(_isolated_home, capsys):
    (_isolated_home / "invoicer.toml").write_text(
        '[email]\nhost = "smtp.example.com"\nsecurity = "none"\nusername = "u"\n',
        encoding="utf-8",
    )
    assert run("config-check") == 1
    assert "unencrypted" in capsys.readouterr().out
