import dataclasses
import email
import re
import socket
from datetime import date
from email import policy

import pytest
from aiosmtpd.controller import Controller
from invoicer import service
from invoicer.db import open_session
from invoicer.models import EmailLog, Invoice
from invoicer.web import create_app
from litestar.testing import TestClient
from sqlmodel import select

HX = {"HX-Request": "true"}


class Inbox:
    """In-process SMTP handler: keeps every message and can refuse recipients on demand."""

    def __init__(self) -> None:
        self.messages: list[email.message.EmailMessage] = []
        self.refuse = False

    async def handle_RCPT(self, server, session, envelope, address, options):
        if self.refuse:
            return "550 mailbox unavailable"
        envelope.rcpt_tos.append(address)
        return "250 OK"

    async def handle_DATA(self, server, session, envelope):
        self.messages.append(
            email.message_from_bytes(envelope.content, policy=policy.default)
        )
        return "250 OK"


@pytest.fixture
def inbox():
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    handler = Inbox()
    controller = Controller(handler, hostname="127.0.0.1", port=port)
    controller.start()
    try:
        yield handler, port
    finally:
        controller.stop()


@pytest.fixture
def web(settings, engine):
    """A test client for the app; data is created through the shared ``session`` fixture."""
    return TestClient(create_app(settings), base_url="http://localhost")


@pytest.fixture
def issued_id(session, make_invoice):
    inv = make_invoice()
    session.commit()
    return inv.id


def _token(client: TestClient) -> str:
    """Make sure the CSRF cookie exists; the token is the cookie value."""
    if "csrftoken" not in client.cookies:
        client.get("/healthz")
    return client.cookies["csrftoken"]


def act(client: TestClient, path: str, data: dict | None = None, *, htmx: bool = True):
    """POST the way the UI does: htmx sends the token in a header, plain forms in a field."""
    token = _token(client)
    if htmx:
        return client.post(path, data=data or {}, headers={**HX, "x-csrftoken": token})
    return client.post(path, data={**(data or {}), "_csrf_token": token})


def _assert_no_inline_code(html: str) -> None:
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert "<style" not in html
    assert not re.search(r"\sstyle=", html)
    assert not re.search(r"\son[a-z]+=", html)


def test_healthz_and_root_redirect(web):
    assert web.get("/healthz").text == "ok"
    response = web.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/invoices"


def test_security_headers_and_no_inline_code(web, issued_id):
    for path in ("/invoices", f"/invoices/{issued_id}", "/clients", "/nope"):
        response = web.get(path)
        assert response.headers["content-security-policy"].startswith(
            "default-src 'self'; style-src 'self'; script-src 'self'"
        )
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
    assert web.get("/invoices").headers["cache-control"] == "no-store"
    for path in ("/invoices", f"/invoices/{issued_id}", "/clients", "/nope"):
        _assert_no_inline_code(web.get(path).text)


def test_list_full_page_and_htmx_partial(web, session, make_invoice, acme):
    make_invoice()
    service.create_invoice(
        session, acme, [service.parse_line("Draft work | 1 | 5 | 0", "EUR")]
    )
    session.commit()

    full = web.get("/invoices").text
    assert "<html" in full and "draft 2" in full and "INV-2026-0001" in full

    partial = web.get("/invoices?status=draft", headers=HX).text
    assert "<html" not in partial
    assert "draft 2" in partial and "INV-2026-0001" not in partial

    filtered = web.get("/invoices?status=draft").text
    assert "<html" in filtered and "draft 2" in filtered
    assert "INV-2026-0001" not in filtered


def test_list_rejects_unknown_status(web):
    assert web.get("/invoices?status=bogus").status_code == 400


def test_outstanding_totals(web, session, issued_id):
    page = web.get("/invoices").text
    assert "Outstanding EUR" in page and "1,190.00" in page

    act(
        web, f"/invoices/{issued_id}/pay", {"amount": "190.00", "paid_on": "2026-10-05"}
    )
    page = web.get("/invoices").text
    assert "1,000.00" in page


def test_detail_page_shows_parties_lines_totals_and_log(web, issued_id):
    page = web.get(f"/invoices/{issued_id}").text
    assert "Ada Lovelace Consulting" in page
    assert "Acme GmbH" in page
    assert "Consulting" in page
    assert "1,190.00" in page
    assert "Not emailed yet." in page
    assert "No payments yet." in page


def test_unknown_invoice_is_404_page(web):
    for response in (
        web.get("/invoices/9999"),
        web.get("/invoices/9999/pdf"),
    ):
        assert response.status_code == 404
        assert "Error 404" in response.text


def test_pdf_download(web, issued_id):
    response = web.get(f"/invoices/{issued_id}/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert (
        response.headers["content-disposition"]
        == 'inline; filename="INV-2026-0001.pdf"'
    )
    assert response.content.startswith(b"%PDF")


def test_issue_draft(web, session, acme):
    draft = service.create_invoice(
        session, acme, [service.parse_line("Work | 1 | 10 | 0", "EUR")]
    )
    session.commit()

    response = act(web, f"/invoices/{draft.id}/issue")
    assert response.status_code == 200
    assert "<html" not in response.text
    assert "Issued INV-2026-0001." in response.text

    again = act(web, f"/invoices/{draft.id}/issue", htmx=False)
    assert again.status_code == 409
    assert "already issued" in again.text
    assert "<html" in again.text
    with open_session(web.app.state.engine) as s:
        assert s.get(Invoice, draft.id).number == "INV-2026-0001"


def test_pay_flow_and_errors(web, session, issued_id):
    response = act(
        web,
        f"/invoices/{issued_id}/pay",
        {"amount": "5000.00", "paid_on": "2026-10-05", "method": "bank"},
    )
    assert response.status_code == 409
    assert "exceeds the balance due" in response.text

    response = act(web, f"/invoices/{issued_id}/pay", {"amount": "abc"})
    assert response.status_code == 409 and "abc" in response.text

    response = act(
        web, f"/invoices/{issued_id}/pay", {"amount": "1.00", "paid_on": "yesterday"}
    )
    assert response.status_code == 409 and "paid on must be a date" in response.text

    response = act(
        web,
        f"/invoices/{issued_id}/pay",
        {"amount": "1190.00", "paid_on": "2026-10-05", "method": "bank transfer"},
    )
    assert response.status_code == 200
    assert "Payment recorded." in response.text
    assert "paid" in response.text and "bank transfer" in response.text


def test_paying_a_draft_is_refused(web, session, acme):
    draft = service.create_invoice(
        session, acme, [service.parse_line("Work | 1 | 10 | 0", "EUR")]
    )
    session.commit()
    response = act(web, f"/invoices/{draft.id}/pay", {"amount": "10.00"})
    assert response.status_code == 409
    assert "only issued invoices take payments" in response.text


def test_void_flow(web, session, issued_id):
    act(web, f"/invoices/{issued_id}/pay", {"amount": "1.00"})
    response = act(web, f"/invoices/{issued_id}/void")
    assert response.status_code == 409
    assert "cannot be voided" in response.text

    with open_session(web.app.state.engine) as s:
        inv = s.get(Invoice, issued_id)
        for payment in service.payments_of(s, inv):
            s.delete(payment)
    response = act(web, f"/invoices/{issued_id}/void")
    assert response.status_code == 200
    assert "Voided INV-2026-0001." in response.text
    assert "void" in response.text


def test_send_delivers_pdf_and_logs(web, settings, session, issued_id, inbox):
    handler, port = inbox
    mail = dataclasses.replace(
        settings.email, host="127.0.0.1", port=port, security="none"
    )
    app = create_app(dataclasses.replace(settings, email=mail))
    client = TestClient(app, base_url="http://localhost")

    response = act(client, f"/invoices/{issued_id}/send")
    assert response.status_code == 200
    assert "Sent INV-2026-0001 to ap@acme.example." in response.text

    assert len(handler.messages) == 1
    message = handler.messages[0]
    assert message["To"] == "ap@acme.example"
    attachment = next(message.iter_attachments())
    assert attachment.get_content_type() == "application/pdf"
    assert attachment.get_payload(decode=True).startswith(b"%PDF")

    with open_session(app.state.engine) as s:
        logs = s.exec(select(EmailLog)).all()
        assert [(row.status, row.recipient) for row in logs] == [
            ("sent", "ap@acme.example")
        ]

    again = act(client, f"/invoices/{issued_id}/send")
    assert again.status_code == 409
    assert "already sent" in again.text
    assert len(handler.messages) == 1

    resent = act(client, f"/invoices/{issued_id}/send", {"resend": "1"})
    assert resent.status_code == 200
    assert len(handler.messages) == 2


def test_failed_send_is_logged_and_shown(web, settings, issued_id, inbox):
    handler, port = inbox
    handler.refuse = True
    mail = dataclasses.replace(
        settings.email, host="127.0.0.1", port=port, security="none"
    )
    app = create_app(dataclasses.replace(settings, email=mail))
    client = TestClient(app, base_url="http://localhost")

    response = act(client, f"/invoices/{issued_id}/send")
    assert response.status_code == 502
    assert "Sending failed" in response.text
    assert "failed" in response.text
    assert handler.messages == []

    with open_session(app.state.engine) as s:
        (row,) = s.exec(select(EmailLog)).all()
        assert row.status == "failed" and row.error


def test_send_refused_for_drafts(web, session, acme):
    draft = service.create_invoice(
        session, acme, [service.parse_line("Work | 1 | 10 | 0", "EUR")]
    )
    session.commit()
    response = act(web, f"/invoices/{draft.id}/send")
    assert response.status_code == 409
    assert "only issued invoices can be sent" in response.text


def test_xss_is_escaped_everywhere(web, session, make_invoice, acme):
    evil_client = service.add_client(
        session, "<script>alert(1)</script>", currency="EUR"
    )
    spec = service.parse_line('"><img src=x onerror=alert(1)> | 1 | 10 | 0', "EUR")
    inv = service.create_invoice(session, evil_client, [spec])
    service.issue_invoice(session, inv, web.app.state.settings, date(2026, 10, 1))
    session.commit()

    pages = [
        web.get("/invoices").text,
        web.get("/invoices", headers=HX).text,
        web.get(f"/invoices/{inv.id}").text,
        web.get("/clients").text,
    ]
    for page in pages:
        assert "<script>alert(1)" not in page
        assert "<img src=x onerror" not in page
    detail = pages[2]
    assert "&lt;img src=x onerror=alert(1)&gt;" in detail
    assert "&#34;&gt;&lt;img" in detail


def test_csrf_rejects_missing_or_wrong_token(web, issued_id):
    _token(web)
    response = web.post(f"/invoices/{issued_id}/void")
    assert response.status_code == 403

    response = web.post(
        f"/invoices/{issued_id}/void", headers={"x-csrftoken": "forged"}
    )
    assert response.status_code == 403

    with open_session(web.app.state.engine) as s:
        assert s.get(Invoice, issued_id).status == "issued"


def test_csrf_accepts_token_from_form_field(web, issued_id):
    response = act(web, f"/invoices/{issued_id}/void", htmx=False)
    assert (
        response.status_code == 200
    )  # no payments, so voiding is allowed; not a CSRF refusal
    assert "Voided INV-2026-0001." in response.text


def test_csrf_token_is_in_the_page_for_htmx(web):
    page = web.get("/invoices").text
    token = web.cookies["csrftoken"]
    assert f'hx-headers=\'{{"x-csrftoken": "{token}"}}\'' in page


def test_foreign_host_header_is_rejected(web):
    response = web.get("/healthz", headers={"host": "evil.example"})
    assert response.status_code == 400


def test_local_hosts_with_port_are_accepted(settings):
    # The test transport cannot parse "[::1]:8000" as a base_url, so set the Host header directly.
    client = TestClient(
        create_app(settings, port=8000), base_url="http://localhost:8000"
    )
    for host in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000", "[::1]"):
        assert client.get("/healthz", headers={"host": host}).text == "ok"


def test_clients_page_lists_clients(web, session, acme):
    session.commit()
    page = web.get("/clients").text
    assert "Acme GmbH" in page and "ap@acme.example" in page


def test_static_assets_are_served(web):
    css = web.get("/static/app.css")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert web.get("/static/htmx.min.js").status_code == 200
