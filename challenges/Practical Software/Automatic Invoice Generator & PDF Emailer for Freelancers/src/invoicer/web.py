"""Local web UI: Litestar routes over ``service``, ``render``, ``mailer`` and ``present``.

Frontend: htmx 2.0.4, vendored as ``static/htmx.min.js`` from cdn.jsdelivr.net/npm/htmx.org@2.0.4.
Pages load nothing from third parties; ``static/app.js`` only tells htmx to swap error responses.
"""

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from jinja2 import Environment, PackageLoader, StrictUndefined
from litestar import Litestar, Request, Response, get, post
from litestar.config.allowed_hosts import AllowedHostsConfig
from litestar.config.csrf import CSRFConfig
from litestar.datastructures import State
from litestar.enums import MediaType, RequestEncodingType
from litestar.exceptions import HTTPException, NotFoundException
from litestar.params import Body, PathParameter, QueryParameter
from litestar.response import Redirect
from litestar.static_files import create_static_files_router
from litestar.utils.empty import value_or_default
from litestar.utils.scope.state import ScopeState
from sqlmodel import Session, col, select

from invoicer import mailer, present, render, service
from invoicer.config import Settings
from invoicer.db import init_db, make_engine, open_session
from invoicer.errors import InvoicerError
from invoicer.models import ISSUED, Client, Invoice
from invoicer.money import format_money

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
STATUSES = ("draft", "issued", "sent", "partial", "overdue", "paid", "void")
FILTERS = (("", "All"), *((s, s.capitalize()) for s in STATUSES))
StatusFilter = Literal["draft", "issued", "sent", "partial", "overdue", "paid", "void"]
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
CSP = (
    "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


@dataclass(frozen=True)
class Flash:
    """Outcome of an action, shown above the refreshed invoice. ``status`` is the HTTP code."""

    text: str
    ok: bool = True
    status: int = 200


@dataclass
class PayForm:
    amount: str
    paid_on: str = ""
    method: str = ""


@dataclass
class SendForm:
    resend: str = ""


# --------------------------------------------------------------------------- rendering


def _respond(
    request: Request,
    page: str,
    context: dict,
    *,
    fragment: str | None = None,
    status: int = 200,
) -> Response[str]:
    """Full page normally; the fragment when htmx asks for it (``HX-Request`` header)."""
    template = fragment if fragment and "HX-Request" in request.headers else page
    token = ScopeState.from_scope(request.scope).csrf_token
    context = {
        "flash": None,
        "csrf_token": value_or_default(token, ""),
        **context,
    }
    html = request.app.state.env.get_template(template).render(context)
    return Response(content=html, media_type=MediaType.HTML, status_code=status)


def _detail(
    request: Request, view: service.InvoiceView, flash: Flash | None = None
) -> Response[str]:
    settings = request.app.state.settings
    sent = any(e.kind == "invoice" and e.status == "sent" for e in view.emails)
    return _respond(
        request,
        "web/invoice.html",
        {
            "view": view,
            "doc": render.build_context(view, settings),
            "emails": present.view_dict(view)["emails"],
            "sent": sent,
            "flash": flash,
        },
        fragment="web/_invoice_detail.html",
        status=flash.status if flash else 200,
    )


# --------------------------------------------------------------------------- helpers


def _find(session: Session, invoice_id: Annotated[int, PathParameter()]) -> Invoice:
    inv = session.get(Invoice, invoice_id)
    if inv is None:
        raise NotFoundException(detail=f"There is no invoice {invoice_id}.")
    return inv


def _mutate(
    request: Request,
    invoice_id: Annotated[int, PathParameter()],
    change: Callable[[Session, Invoice, Settings], Flash],
) -> Response[str]:
    """Run one action and show the invoice again.

    A refused action (``InvoicerError``) rolls back its partial writes and comes back as a 409
    flash rather than an exception. Some failures are recorded rather than raised (a failed send
    is logged as an EmailLog row), so the caller returns them as a ``Flash`` and they commit.
    """
    settings = request.app.state.settings
    with open_session(request.app.state.engine) as session:
        inv = _find(session, invoice_id)
        try:
            flash = change(session, inv, settings)
        except InvoicerError as exc:
            session.rollback()
            flash = Flash(str(exc), ok=False, status=409)
        return _detail(request, service.load_view(session, inv), flash)


def _issue(session: Session, inv: Invoice, settings: Settings) -> Flash:
    service.issue_invoice(session, inv, settings)
    return Flash(f"Issued {inv.number}.")


def _pay(session: Session, inv: Invoice, settings: Settings, form: PayForm) -> Flash:
    try:
        paid_on = date.fromisoformat(form.paid_on) if form.paid_on else None
    except ValueError as exc:
        raise InvoicerError(
            f"paid on must be a date such as 2026-10-13: {exc}"
        ) from exc
    service.record_payment(
        session, inv, form.amount, paid_on=paid_on, method=form.method
    )
    return Flash("Payment recorded.")


def _void(session: Session, inv: Invoice, settings: Settings) -> Flash:
    service.void_invoice(session, inv)
    return Flash(f"Voided {inv.number}.")


def _send(session: Session, inv: Invoice, settings: Settings, form: SendForm) -> Flash:
    if inv.status != ISSUED:
        raise InvoicerError(
            f"only issued invoices can be sent (this one is {inv.status})"
        )
    view = service.load_view(session, inv)
    if form.resend != "1" and any(
        e.kind == "invoice" and e.status == "sent" for e in view.emails
    ):
        raise InvoicerError(f"{inv.number} was already sent; send it again to confirm")
    pdf = render.render_pdf(view, settings)
    outcome = mailer.deliver(session, view, settings, pdf, resend=form.resend == "1")
    if outcome.status == "sent":
        return Flash(f"Sent {inv.number} to {outcome.recipient}.")
    return Flash(
        f"Sending failed after {outcome.attempts} attempt(s): {outcome.error}",
        ok=False,
        status=502,
    )


# --------------------------------------------------------------------------- routes


@get("/", sync_to_thread=False)
def root() -> Redirect:
    return Redirect(path="/invoices")


@get("/invoices", sync_to_thread=True)
def invoices(
    request: Request, status: Annotated[StatusFilter | None, QueryParameter()] = None
) -> Response[str]:
    with open_session(request.app.state.engine) as session:
        everything = service.list_views(session)
    views = [v for v in everything if status is None or v.status == status]
    return _respond(
        request,
        "web/invoices.html",
        {
            "views": views,
            "status": status,
            "filters": FILTERS,
            "outstanding": service.outstanding(everything),
        },
        fragment="web/_invoice_table.html",
    )


@get("/invoices/{invoice_id:int}", sync_to_thread=True)
def invoice_detail(
    request: Request, invoice_id: Annotated[int, PathParameter()]
) -> Response[str]:
    with open_session(request.app.state.engine) as session:
        view = service.load_view(session, _find(session, invoice_id))
        return _detail(request, view)


@get("/invoices/{invoice_id:int}/pdf", sync_to_thread=True)
def invoice_pdf(
    request: Request, invoice_id: Annotated[int, PathParameter()]
) -> Response[bytes]:
    settings = request.app.state.settings
    with open_session(request.app.state.engine) as session:
        inv = _find(session, invoice_id)
        pdf = render.render_pdf(service.load_view(session, inv), settings)
        name = render.pdf_filename(inv.number, inv.id)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{name}"'},
    )


@post("/invoices/{invoice_id:int}/issue", sync_to_thread=True)
def issue(
    request: Request, invoice_id: Annotated[int, PathParameter()]
) -> Response[str]:
    return _mutate(request, invoice_id, _issue)


@post("/invoices/{invoice_id:int}/pay", sync_to_thread=True)
def pay(
    request: Request,
    invoice_id: Annotated[int, PathParameter()],
    data: Annotated[PayForm, Body(media_type=RequestEncodingType.URL_ENCODED)],
) -> Response[str]:
    return _mutate(request, invoice_id, lambda s, i, st: _pay(s, i, st, data))


@post("/invoices/{invoice_id:int}/void", sync_to_thread=True)
def void(
    request: Request, invoice_id: Annotated[int, PathParameter()]
) -> Response[str]:
    return _mutate(request, invoice_id, _void)


@post("/invoices/{invoice_id:int}/send", sync_to_thread=True)
def send(
    request: Request,
    invoice_id: Annotated[int, PathParameter()],
    data: Annotated[SendForm, Body(media_type=RequestEncodingType.URL_ENCODED)],
) -> Response[str]:
    return _mutate(request, invoice_id, lambda s, i, st: _send(s, i, st, data))


@get("/clients", sync_to_thread=True)
def clients(request: Request) -> Response[str]:
    with open_session(request.app.state.engine) as session:
        rows = session.exec(
            select(Client)
            .where(col(Client.archived).is_(False))
            .order_by(col(Client.name))
        ).all()
    return _respond(request, "web/clients.html", {"clients": rows})


@get("/healthz", media_type=MediaType.TEXT, sync_to_thread=False)
def healthz() -> str:
    return "ok"


# --------------------------------------------------------------------------- hooks


def _security_headers(response: Response) -> Response:
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    # getattr: file responses (static assets) have no media_type attribute.
    if getattr(response, "media_type", None) == MediaType.HTML:
        response.headers["Cache-Control"] = "no-store"
    return response


def _same_origin_only(request: Request) -> Response | None:
    """Refuse a state-changing request whose Origin is not this server.

    Browsers always send Origin on form posts; a page on another localhost port is the
    realistic attacker for a local tool, and port numbers do not separate cookies.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    origin = request.headers.get("origin")
    if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
        return _security_headers(
            Response(
                "cross-origin request refused",
                status_code=403,
                media_type=MediaType.TEXT,
            )
        )
    return None


def _error_page(request: Request, exc: Exception) -> Response[str]:
    if isinstance(exc, HTTPException):
        code, message = exc.status_code, str(exc.detail)
    elif isinstance(exc, InvoicerError):
        code, message = 500, str(exc)
    else:
        logger.error("unhandled error in %s", request.url.path, exc_info=exc)
        code, message = 500, "Something went wrong. The details are in the server log."
    # Exception responses skip the after_request hook, so the headers are applied here too.
    return _security_headers(
        _respond(
            request,
            "web/error.html",
            {"code": code, "flash": Flash(message, ok=False, status=code)},
            fragment="web/_flash.html",
            status=code,
        )
    )


def create_app(settings: Settings, *, port: int = 8000) -> Litestar:
    """``port`` must match the port the server listens on: browsers send it in the Host header,
    and Litestar's allowed-hosts check compares the whole header."""
    engine = make_engine(settings.db_path)
    init_db(engine)
    env = Environment(
        loader=PackageLoader("invoicer", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )
    env.globals["money"] = lambda minor, currency: format_money(
        minor, currency, settings.locale
    )
    hosts = [*LOCAL_HOSTS, *(f"{host}:{port}" for host in LOCAL_HOSTS)]
    return Litestar(
        route_handlers=[
            root,
            invoices,
            invoice_detail,
            invoice_pdf,
            issue,
            pay,
            void,
            send,
            clients,
            healthz,
            create_static_files_router(path="/static", directories=[STATIC_DIR]),
        ],
        state=State({"engine": engine, "settings": settings, "env": env}),
        # HttpOnly: the token is embedded in each page, so scripts never need to read the cookie,
        # and cookies are shared across ports, so a script on another localhost port could.
        csrf_config=CSRFConfig(secret=secrets.token_hex(32), cookie_httponly=True),
        before_request=_same_origin_only,
        allowed_hosts=AllowedHostsConfig(allowed_hosts=hosts),
        after_request=_security_headers,
        exception_handlers={Exception: _error_page},
    )
