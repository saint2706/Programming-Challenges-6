from dataclasses import replace
from datetime import date

import pytest
from invoicer import mailer, service
from invoicer.errors import InvoicerError
from invoicer.models import EmailLog
from smtpserver import make_test_ca, smtp_server
from sqlmodel import select

PDF = b"%PDF-1.7 fake pdf bytes"


def view_of(session, inv, today=date(2026, 10, 5)):
    return service.load_view(session, inv, today)


def configured(settings, server, **changes):
    changes = {"security": "none", **changes}
    email = replace(settings.email, host=server.host, port=server.port, **changes)
    return replace(settings, email=email)


def logs(session):
    return list(session.exec(select(EmailLog).order_by(EmailLog.id)))


def test_message_content_and_attachment(session, settings, make_invoice):
    msg = mailer.build_message(view_of(session, make_invoice()), settings, PDF)
    assert msg["To"] == "ap@acme.example"
    assert msg["From"] == "Ada Lovelace Consulting <ada@example.com>"
    assert msg["Subject"] == "Invoice INV-2026-0001 from Ada Lovelace Consulting"
    body = msg.get_body().get_content()
    assert "INV-2026-0001" in body and "€1,190.00" in body and "2026-10-15" in body
    assert "Please pay by bank transfer." in body
    (attachment,) = list(msg.iter_attachments())
    assert attachment.get_filename() == "INV-2026-0001.pdf"
    assert attachment.get_content_type() == "application/pdf"
    assert attachment.get_payload(decode=True) == PDF
    assert msg["Message-ID"].endswith("@example.com>")


def test_reminder_message(session, settings, make_invoice):
    view = view_of(session, make_invoice(), date(2026, 10, 25))
    msg = mailer.build_message(view, settings, PDF, kind="reminder")
    assert msg["Subject"] == "Reminder: invoice INV-2026-0001 is 10 days overdue"
    assert "was due on 2026-10-15" in msg.get_body().get_content()


def test_recipient_override_and_validation(session, settings, make_invoice, acme):
    view = view_of(session, make_invoice())
    msg = mailer.build_message(view, settings, PDF, to=["a@x.example", "b@y.example"])
    assert msg["To"] == "a@x.example, b@y.example"
    for bad in (
        "nobody",
        "a@b",
        "x@y.z\nBcc: evil@evil.example",
        "a b@c.de",
        "<a@b.cd>",
    ):
        with pytest.raises(InvoicerError):
            mailer.build_message(view, settings, PDF, to=[bad])
    acme.email = ""
    with pytest.raises(InvoicerError, match="no email address"):
        mailer.build_message(view_of(session, view.invoice), settings, PDF)


def test_header_injection_through_a_client_name_is_neutralised(
    session, settings, make_invoice, acme
):
    service.edit_client(session, acme, name="Evil\nBcc: victim@example.com")
    view = view_of(session, make_invoice())
    # The snapshot-free name flows into the body only, never into a header.
    msg = mailer.build_message(view, settings, PDF)
    assert msg["Bcc"] is None
    assert "Evil" in msg.get_body().get_content()


def test_unconfigured_sender(session, settings, make_invoice):
    settings = replace(settings, email=replace(settings.email, from_address=""))
    with pytest.raises(InvoicerError, match="from_address"):
        mailer.build_message(view_of(session, make_invoice()), settings, PDF)


def test_bcc_header(session, settings, make_invoice):
    settings = replace(settings, email=replace(settings.email, bcc="books@example.com"))
    assert (
        mailer.build_message(view_of(session, make_invoice()), settings, PDF)["Bcc"]
        == "books@example.com"
    )


def test_successful_delivery_is_logged(session, settings, make_invoice):
    with smtp_server() as server:
        cfg = configured(settings, server)
        view = view_of(session, make_invoice())
        outcome = mailer.deliver(session, view, cfg, PDF)
        assert outcome.status == "sent" and outcome.attempts == 1
        (received,) = server.messages
    assert received["Subject"] == "Invoice INV-2026-0001 from Ada Lovelace Consulting"
    assert next(received.iter_attachments()).get_payload(decode=True) == PDF
    (row,) = logs(session)
    assert (row.status, row.kind, row.recipient) == (
        "sent",
        "invoice",
        "ap@acme.example",
    )
    assert (
        row.message_id == received["Message-ID"] and row.smtp_response == "250 accepted"
    )
    assert service.load_view(session, view.invoice, date(2026, 10, 5)).status == "sent"


def test_permanent_failure_is_not_retried(session, settings, make_invoice):
    with smtp_server(replies=["550 mailbox unavailable"]) as server:
        outcome = mailer.deliver(
            session, view_of(session, make_invoice()), configured(settings, server), PDF
        )
        assert server.handler.data_calls == 1
    assert outcome.status == "failed" and outcome.attempts == 1
    assert "550" in outcome.error
    (row,) = logs(session)
    assert row.status == "failed" and "550" in row.error


def test_transient_failure_is_retried_with_backoff(session, settings, make_invoice):
    sleeps = []
    with smtp_server(replies=["451 try again later", "451 still busy"]) as server:
        outcome = mailer.deliver(
            session,
            view_of(session, make_invoice()),
            configured(settings, server),
            PDF,
            retries=2,
            backoff=0.5,
            sleep=sleeps.append,
        )
        assert server.handler.data_calls == 3 and len(server.messages) == 1
    assert (outcome.status, outcome.attempts) == ("sent", 3)
    assert sleeps == [0.5, 1.0]  # exponential
    assert [r.status for r in logs(session)] == [
        "sent"
    ]  # one row per delivery, not per attempt


def test_transient_failure_gives_up_after_the_retries(session, settings, make_invoice):
    with smtp_server(replies=["451 a", "451 b", "451 c", "451 d"]) as server:
        outcome = mailer.deliver(
            session,
            view_of(session, make_invoice()),
            configured(settings, server),
            PDF,
            retries=1,
            sleep=lambda _: None,
        )
        assert server.handler.data_calls == 2
    assert (outcome.status, outcome.attempts) == ("failed", 2)
    assert logs(session)[0].attempts == 2


def test_refused_recipient_is_permanent(session, settings, make_invoice):
    with smtp_server(refuse_rcpt=["ap@acme.example"]) as server:
        outcome = mailer.deliver(
            session, view_of(session, make_invoice()), configured(settings, server), PDF
        )
        assert server.handler.data_calls == 0
    assert outcome.status == "failed" and outcome.attempts == 1


def test_connection_refused_is_transient_and_logged(session, settings, make_invoice):
    from smtpserver import free_port

    cfg = replace(
        settings, email=replace(settings.email, host="127.0.0.1", port=free_port())
    )
    outcome = mailer.deliver(
        session,
        view_of(session, make_invoice()),
        cfg,
        PDF,
        retries=1,
        sleep=lambda _: None,
    )
    assert (outcome.status, outcome.attempts) == ("failed", 2)
    assert logs(session)[0].error


def test_unconfigured_host_fails_cleanly(session, settings, make_invoice):
    cfg = replace(settings, email=replace(settings.email, host=""))
    outcome = mailer.deliver(session, view_of(session, make_invoice()), cfg, PDF)
    assert outcome.status == "failed" and "email.host" in outcome.error


def test_login_uses_the_password_from_the_environment(
    session, settings, make_invoice, monkeypatch
):
    with smtp_server(credentials=("ada", "s3cret")) as server:
        cfg = configured(settings, server, username="ada", password_env="TEST_SMTP_PW")
        view = view_of(session, make_invoice())
        monkeypatch.delenv("TEST_SMTP_PW", raising=False)
        missing = mailer.deliver(session, view, cfg, PDF)
        assert missing.status == "failed" and "TEST_SMTP_PW" in missing.error

        monkeypatch.setenv("TEST_SMTP_PW", "wrong")
        wrong = mailer.deliver(session, view, cfg, PDF, retries=3, sleep=lambda _: None)
        assert (
            wrong.status == "failed" and wrong.attempts == 1
        )  # a bad login is not retried
        assert not server.messages

        monkeypatch.setenv("TEST_SMTP_PW", "s3cret")
        assert mailer.deliver(session, view, cfg, PDF).status == "sent"
        assert len(server.messages) == 1


def test_starttls_verifies_certificates(session, settings, make_invoice, tmp_path):
    ca, cert, key = make_test_ca(tmp_path)
    view = view_of(session, make_invoice())
    with smtp_server(tls=(cert, key)) as server:
        base = configured(settings, server, security="starttls")
        # Without trusting the test CA the connection must fail: certificates are never skipped.
        untrusted = mailer.deliver(session, view, base, PDF, retries=0)
        assert untrusted.status == "failed" and "certificate" in untrusted.error.lower()
        assert not server.messages
        trusted = replace(base, email=replace(base.email, ca_file=ca))
        assert mailer.deliver(session, view, trusted, PDF).status == "sent"
        assert len(server.messages) == 1


def test_a_failure_never_raises_out_of_deliver(session, settings, make_invoice):
    view = view_of(session, make_invoice())
    cfg = replace(settings, email=replace(settings.email, host="256.256.256.256"))
    assert mailer.deliver(session, view, cfg, PDF, retries=0).status == "failed"


def test_message_body_uses_the_sandbox(session, settings, make_invoice, acme):
    service.edit_client(session, acme, name="{{ 7*7 }} {{ ''.__class__ }}")
    body = (
        mailer.build_message(view_of(session, make_invoice()), settings, PDF)
        .get_body()
        .get_content()
    )
    assert "{{ 7*7 }}" in body  # client data is substituted as text, never evaluated
