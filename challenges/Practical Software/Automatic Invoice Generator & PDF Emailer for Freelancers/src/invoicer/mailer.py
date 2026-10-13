"""Sending invoices and reminders over SMTP, and recording every attempt.

* Previews are the caller's business (``build_message``); ``deliver`` always sends, and
  previews are not logged because nothing happened.
* Every real attempt has one ``EmailLog`` row: ``pending`` while in flight (which also stops a
  duplicate send), then ``sent`` with the Message-ID or ``failed`` with the error. Only ``sent`` rows count as delivered, so a failure is retried
  by the next run instead of being skipped for good.
* Transient failures (connection trouble, 4xx replies) are retried with backoff; permanent ones
  (5xx, refused recipients, bad login) are not.
* The password comes from the environment, TLS certificates are always verified (``ca_file``
  adds a private CA), and recipients are validated so a stray newline cannot become a header.
"""

import json
import smtplib
import ssl
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

from jinja2.sandbox import SandboxedEnvironment
from sqlmodel import Session, select

from invoicer.config import EmailSettings, Settings
from invoicer.errors import InvoicerError
from invoicer.models import EmailLog
from invoicer.money import format_money
from invoicer.render import pdf_filename
from invoicer.service import InvoiceView, lock_invoice, validate_email

SUBJECTS = {
    "invoice": "Invoice {{ number }} from {{ seller }}",
    "reminder": "Reminder: invoice {{ number }} is {{ days_overdue }} days overdue",
}
BODIES = {
    "invoice": """\
Hello {{ client }},

Please find attached invoice {{ number }} for {{ total }}, due on {{ due }}.
{% if payment_note %}
{{ payment_note }}
{% endif %}
Thank you,
{{ seller }}
""",
    "reminder": """\
Hello {{ client }},

This is a reminder that invoice {{ number }} ({{ balance }} outstanding) was due on {{ due }}
and is now {{ days_overdue }} days overdue. A copy is attached.

If you have already paid, please ignore this message and accept our thanks.

{{ seller }}
""",
}
_TRANSIENT_CODES = range(400, 500)
PENDING_TTL = timedelta(minutes=15)


class MailError(InvoicerError):
    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


@dataclass(frozen=True)
class Outcome:
    status: str  # sent | failed
    recipient: str
    subject: str
    error: str = ""
    message_id: str = ""
    attempts: int = 1


def recipients_for(view: InvoiceView, override: Sequence[str] = ()) -> list[str]:
    addresses = [validate_email(a) for a in override] or (
        [validate_email(view.client.email)] if view.client.email else []
    )
    if not addresses:
        raise InvoicerError(f"{view.client.name} has no email address; pass --to")
    return addresses


def build_message(
    view: InvoiceView,
    settings: Settings,
    pdf: bytes,
    *,
    kind: str = "invoice",
    to: Sequence[str] = (),
    as_of: date | None = None,
) -> EmailMessage:
    cfg = settings.email
    if not cfg.from_address:
        raise InvoicerError("email.from_address (or business.email) is not configured")
    inv = view.invoice
    env = SandboxedEnvironment(autoescape=False)  # plain text mail: nothing to escape
    cur, loc = inv.currency, settings.locale
    # The same frozen seller details the PDF shows, so mail and attachment cannot disagree.
    seller = (
        json.loads(inv.snapshot)["seller"] if inv.snapshot else vars(settings.business)
    )
    context = {
        "number": inv.number,
        "client": view.client.name,
        "seller": seller["name"],
        "total": format_money(view.totals.total, cur, loc),
        "balance": format_money(view.balance, cur, loc),
        "due": inv.due_date.isoformat() if inv.due_date else "",
        "days_overdue": view.days_overdue,
        "payment_note": seller["payment_note"],
    }
    msg = EmailMessage()
    msg["From"] = formataddr((cfg.from_name, validate_email(cfg.from_address)))
    msg["To"] = ", ".join(recipients_for(view, to))
    if cfg.bcc:
        msg["Bcc"] = validate_email(cfg.bcc)
    msg["Subject"] = env.from_string(SUBJECTS[kind]).render(context)
    msg["Message-ID"] = make_msgid(domain=cfg.from_address.rpartition("@")[2] or None)
    msg.set_content(env.from_string(BODIES[kind]).render(context))
    msg.add_attachment(
        pdf,
        maintype="application",
        subtype="pdf",
        filename=pdf_filename(inv.number, inv.id),
    )
    return msg


def _connect(cfg: EmailSettings) -> smtplib.SMTP:
    # Certificates and host names are always verified; a private CA is added, never skipped.
    context = ssl.create_default_context(
        cafile=str(cfg.ca_file) if cfg.ca_file else None
    )
    if cfg.security == "ssl":
        return smtplib.SMTP_SSL(
            cfg.host, cfg.port, timeout=cfg.timeout, context=context
        )
    server = smtplib.SMTP(cfg.host, cfg.port, timeout=cfg.timeout)
    if cfg.security == "starttls":
        server.starttls(context=context)
    return server


def send_message(cfg: EmailSettings, msg: EmailMessage) -> str:
    """One delivery attempt. Returns the server's reply; raises ``MailError``."""
    if not cfg.host:
        raise MailError("email.host is not configured", transient=False)
    try:
        with _connect(cfg) as server:
            if cfg.username:
                password = cfg.password
                if not password:
                    raise MailError(
                        f"set the {cfg.password_env} environment variable",
                        transient=False,
                    )
                server.login(cfg.username, password)
            refused = server.send_message(msg)
            if refused:
                raise MailError(
                    f"recipients refused: {sorted(refused)}", transient=False
                )
            return "250 accepted"
    except MailError:
        raise
    except smtplib.SMTPResponseException as exc:
        text = exc.smtp_error.decode(errors="replace") if exc.smtp_error else ""
        raise MailError(
            f"{exc.smtp_code} {text}", transient=exc.smtp_code in _TRANSIENT_CODES
        ) from exc
    except (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused) as exc:
        raise MailError(f"refused: {exc}", transient=False) from exc
    except (smtplib.SMTPException, ssl.SSLError, TimeoutError, OSError) as exc:
        # Connection resets, DNS trouble and timeouts usually clear up on their own.
        raise MailError(f"{type(exc).__name__}: {exc}", transient=True) from exc


def _reserve(
    session: Session,
    view: InvoiceView,
    msg: EmailMessage,
    kind: str,
    stage: int | None,
    resend: bool,
) -> EmailLog:
    """Claim the right to send, durably, before the slow SMTP conversation starts.

    The check and the ``pending`` row happen under the invoice's write lock and are committed
    at once, so a second request (a double-clicked button, a concurrent ``remind``) sees the
    claim and refuses instead of sending the same message twice. A claim older than
    ``PENDING_TTL`` is treated as abandoned by a crashed process.
    """
    inv = view.invoice
    lock_invoice(session, inv)
    now = datetime.now(UTC)
    rows = session.exec(
        select(EmailLog).where(EmailLog.invoice_id == inv.id, EmailLog.kind == kind)
    ).all()
    same = [r for r in rows if kind == "invoice" or r.stage == stage]
    for row in same:
        stamp = (
            row.attempted_at.replace(tzinfo=UTC)
            if row.attempted_at.tzinfo is None
            else row.attempted_at
        )
        if row.status == "pending" and now - stamp < PENDING_TTL:
            raise InvoicerError(f"a {kind} for {inv.number} is already being sent")
    if not resend and any(r.status == "sent" for r in same):
        raise InvoicerError(
            f"{inv.number} was already sent; pass resend to send it again"
        )
    claim = EmailLog(
        invoice_id=inv.id,
        kind=kind,
        stage=stage,
        recipient=str(msg["To"]),
        subject=str(msg["Subject"]),
        status="pending",
        message_id=str(msg["Message-ID"]),
    )
    session.add(claim)
    session.commit()
    return claim


def deliver(
    session: Session,
    view: InvoiceView,
    settings: Settings,
    pdf: bytes,
    *,
    kind: str = "invoice",
    stage: int | None = None,
    to: Sequence[str] = (),
    resend: bool = False,
    retries: int = 2,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Outcome:
    """Send one message and log the outcome.

    Raises ``InvoicerError`` for things the caller can fix before any sending (bad recipient,
    already sent, a send in progress); a delivery failure is never raised, only recorded and
    returned. This commits the session once, to publish the claim made by ``_reserve``.
    """
    msg = build_message(view, settings, pdf, kind=kind, to=to)
    claim = _reserve(session, view, msg, kind, stage, resend)
    attempts, last_error, reply, status = 0, "", "", "failed"
    try:
        for attempt in range(retries + 1):
            attempts = attempt + 1
            try:
                reply = send_message(settings.email, msg)
            except MailError as exc:
                last_error = str(exc)
                if not exc.transient or attempt == retries:
                    break
                sleep(backoff * 2**attempt)
            else:
                status, last_error = "sent", ""
                break
    except (
        BaseException
    ) as exc:  # e.g. Ctrl-C: do not leave a claim that blocks for 15 minutes
        claim.status, claim.error = "failed", f"interrupted: {type(exc).__name__}"
        session.add(claim)
        session.commit()
        raise
    claim.status, claim.error, claim.smtp_response, claim.attempts = (
        status,
        last_error,
        reply,
        attempts,
    )
    session.add(claim)
    session.flush()
    return Outcome(
        status,
        str(msg["To"]),
        str(msg["Subject"]),
        last_error,
        str(msg["Message-ID"]),
        attempts,
    )
