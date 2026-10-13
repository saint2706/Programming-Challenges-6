"""``invoicer`` command line."""

import json
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich import box
from rich.console import Console
from rich.table import Column, Table

from invoicer import mailer, render, service
from invoicer.config import CONFIG_NAME, SAMPLE_CONFIG, Settings, load_settings
from invoicer.db import init_db, make_engine, open_session
from invoicer.errors import InvoicerError
from invoicer.models import DRAFT, ISSUED
from invoicer.money import plain_amount
from invoicer.present import view_dict

app = App(
    name="invoicer", help="Create invoices, render PDFs, email them and track payment."
)
client_app = App(name="client", help="The client book.")
invoice_app = App(name="invoice", help="Create, issue, send and settle invoices.")
recurring_app = App(name="recurring", help="Invoices that repeat on a schedule.")
app.command(client_app)
app.command(invoice_app)
app.command(recurring_app)

console = Console(highlight=False)
Json = Annotated[bool, Parameter(name="--json", help="Machine-readable output.")]
MAX_JOB_FILE = 1_000_000


def _engine(settings: Settings):
    settings.root.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.db_path)
    init_db(engine)
    return engine


def _print_json(data) -> None:
    print(json.dumps(data, indent=2))


def _money(minor: int, currency: str) -> str:
    return f"{plain_amount(minor, currency)} {currency}"


# ------------------------------------------------------------------------------ setup


@app.command
def init() -> None:
    """Write a sample invoicer.toml (if none exists) and create the database."""
    settings = load_settings()
    path = settings.root / CONFIG_NAME
    settings.root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"{path} already exists; left unchanged")
    else:
        path.write_text(SAMPLE_CONFIG, encoding="utf-8")
        print(f"wrote {path}; edit it, then run `invoicer config-check`")
    _engine(settings)
    print(f"database: {settings.db_path}")


@app.command
def config_check() -> None:
    """Load the configuration and report anything that would stop PDFs or email working."""
    settings = load_settings()
    b, e = settings.business, settings.email
    print(
        f"config:   {settings.root / CONFIG_NAME} ({'found' if (settings.root / CONFIG_NAME).exists() else 'missing; defaults'})"
    )
    print(f"business: {b.name} ({b.country}), default currency {b.currency}")
    print(f"numbering: {settings.number_format}")
    problems = []
    if not b.address:
        problems.append("business.address is empty")
    if not e.host:
        problems.append("email.host is not set; sending will fail")
    elif e.username and not e.password:
        problems.append(f"environment variable {e.password_env} is not set")
    if e.username and e.security == "none" and e.host not in ("127.0.0.1", "localhost"):
        problems.append(
            "email.security = 'none' with a username sends the password unencrypted"
        )
    try:
        render.load_weasyprint()
        print("pdf:      WeasyPrint is available")
    except InvoicerError as exc:
        problems.append(str(exc))
    for p in problems:
        print(f"problem:  {p}")
    if problems:
        raise InvoicerError(f"{len(problems)} problem(s) found")
    print("ok")


# ----------------------------------------------------------------------------- clients


@client_app.command(name="add")
def client_add(
    name: str,
    *,
    email: str = "",
    address: str = "",
    country: str | None = None,
    tax_id: str = "",
    currency: str | None = None,
    terms_days: int = 14,
    notes: str = "",
) -> None:
    """Add a client. Country and currency default to your own."""
    settings = load_settings()
    with open_session(_engine(settings)) as session:
        client = service.add_client(
            session,
            name,
            email=email,
            address=address.replace("\\n", "\n"),
            country=country or settings.business.country,
            tax_id=tax_id,
            currency=currency or settings.business.currency,
            terms_days=terms_days,
            notes=notes,
        )
        print(f"client {client.id}: {client.name}")


@client_app.command(name="list")
def client_list(*, all: bool = False, json_output: Json = False) -> None:
    """List clients (archived ones only with --all)."""
    from sqlmodel import select

    from invoicer.models import Client

    with open_session(_engine(load_settings())) as session:
        clients = [
            c
            for c in session.exec(select(Client).order_by(Client.name))
            if all or not c.archived
        ]
        rows = [
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "country": c.country,
                "currency": c.currency,
                "terms_days": c.terms_days,
                "archived": c.archived,
            }
            for c in clients
        ]
    if json_output:
        return _print_json(rows)
    table = Table("ID", "Name", "Email", "Country", "Cur", "Terms")
    for r in rows:
        table.add_row(
            str(r["id"]),
            r["name"],
            r["email"],
            r["country"],
            r["currency"],
            f"{r['terms_days']}d",
        )
    console.print(table)


@client_app.command(name="edit")
def client_edit(
    client: str,
    *,
    name: str | None = None,
    email: str | None = None,
    address: str | None = None,
    country: str | None = None,
    tax_id: str | None = None,
    currency: str | None = None,
    terms_days: int | None = None,
    notes: str | None = None,
) -> None:
    """Change a client. Issued invoices keep the details they were issued with."""
    with open_session(_engine(load_settings())) as session:
        found = service.find_client(session, client)
        service.edit_client(
            session,
            found,
            name=name,
            email=email,
            address=address.replace("\\n", "\n") if address is not None else None,
            country=country,
            tax_id=tax_id,
            currency=currency,
            terms_days=terms_days,
            notes=notes,
        )
        print(f"updated client {found.id}: {found.name}")


@client_app.command(name="rm")
def client_rm(client: str) -> None:
    """Delete a client, or archive them if they have invoices."""
    with open_session(_engine(load_settings())) as session:
        found = service.find_client(session, client)
        had_invoices = bool(service.list_views(session, client=found))
        service.delete_client(session, found)
        print(f"{'archived' if had_invoices else 'deleted'} {found.name}")


# ---------------------------------------------------------------------------- invoices


def _read_job_file(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_JOB_FILE:
            raise InvoicerError(f"{path.name} is over 1 MB")
        text = path.read_text(encoding="utf-8")
        data = (
            json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
        )
    except (
        OSError,
        ValueError,
    ) as exc:  # JSONDecodeError and TOMLDecodeError are ValueErrors
        raise InvoicerError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InvoicerError(f"{path.name} must contain an object")
    return data


def _lines_from_file(
    items: list, currency: str, default_tax: int
) -> list[service.LineSpec]:
    lines = []
    for n, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise InvoicerError(
                f"item {n} must be a table of description/qty/price/tax"
            )
        lines.append(
            service.make_line(
                str(item.get("description", "")),
                item.get("qty", 1),
                item.get("price", ""),
                item.get("tax"),
                currency,
                default_tax,
            )
        )
    return lines


@invoice_app.command(name="new")
def invoice_new(
    *,
    client: str | None = None,
    item: Annotated[
        list[str] | None,
        Parameter(
            name="--item", help="'description | qty | price | tax%'; repeatable."
        ),
    ] = None,
    file: Annotated[
        Path | None, Parameter(help="TOML/JSON with client, items, discount, notes.")
    ] = None,
    currency: str | None = None,
    discount: Annotated[
        str | None, Parameter(help="'10%' or a fixed amount like '25.00'.")
    ] = None,
    tax: Annotated[
        str | None, Parameter(help="Default tax % for lines that give none.")
    ] = None,
    notes: str = "",
    reference: str = "",
    issue: Annotated[bool, Parameter(help="Issue it now: assigns the number.")] = False,
    on: Annotated[
        date | None, Parameter(name="--date", help="Issue date (default today).")
    ] = None,
) -> None:
    """Create a draft invoice (add --issue to number and freeze it straight away)."""
    from invoicer.money import MoneyError, parse_percent

    settings = load_settings()
    data = _read_job_file(file) if file else {}
    client_key = client or data.get("client")
    if not client_key:
        raise InvoicerError("choose a client with --client")
    with open_session(_engine(settings)) as session:
        found = service.find_client(session, str(client_key))
        cur = currency or data.get("currency") or found.currency
        try:
            default_tax = parse_percent(tax if tax is not None else data.get("tax", 0))
        except MoneyError as exc:
            raise InvoicerError(str(exc)) from exc
        lines = [service.parse_line(spec, cur, default_tax) for spec in (item or [])]
        lines += _lines_from_file(data.get("items", []), cur, default_tax)
        inv = service.create_invoice(
            session,
            found,
            lines,
            currency=cur,
            discount=service.parse_discount(discount or data.get("discount"), cur),
            notes=notes or str(data.get("notes", "")),
            reference=reference or str(data.get("reference", "")),
        )
        if issue:
            service.issue_invoice(session, inv, settings, on)
        label = inv.number if inv.number else f"draft {inv.id}"
        view = service.load_view(session, inv)
        print(
            f"created {label} for {found.name}: {_money(view.totals.total, inv.currency)}"
        )


@invoice_app.command(name="list")
def invoice_list(
    *,
    status: str | None = None,
    client: str | None = None,
    as_of: date | None = None,
    json_output: Json = False,
) -> None:
    """List invoices; status is draft, issued, sent, partial, overdue, paid or void."""
    with open_session(_engine(load_settings())) as session:
        found = service.find_client(session, client) if client else None
        views = service.list_views(session, status=status, client=found, today=as_of)
        owed = service.outstanding(views)
        rows = [view_dict(v) for v in views]
    if json_output:
        return _print_json(
            {
                "invoices": rows,
                "outstanding": {c: plain_amount(m, c) for c, m in owed.items()},
            }
        )
    table = Table(
        Column("Number", no_wrap=True),
        Column("Client", overflow="ellipsis", max_width=14),
        Column("Due", no_wrap=True),
        Column("Cur", no_wrap=True),
        Column("Total", justify="right", no_wrap=True),
        Column("Balance", justify="right", no_wrap=True),
        Column("Status", no_wrap=True),
        box=box.SIMPLE_HEAD,
    )
    for r in rows:
        table.add_row(
            r["number"] or f"(draft {r['id']})",
            r["client"],
            r["due_date"] or "",
            r["currency"],
            r["total"],
            r["balance"],
            r["status"],
        )
    console.print(table)
    for cur, minor in owed.items():
        console.print(f"outstanding: {_money(minor, cur)}")


@invoice_app.command(name="show")
def invoice_show(
    number: str, *, as_of: date | None = None, json_output: Json = False
) -> None:
    """One invoice in full: lines, totals, payments and every email attempt."""
    with open_session(_engine(load_settings())) as session:
        view = service.load_view(session, service.get_invoice(session, number), as_of)
        data = view_dict(view)
    if json_output:
        return _print_json(data)
    cur = data["currency"]
    print(
        f"{data['number'] or '(draft ' + str(data['id']) + ')'}  {data['status']}  {data['client']}"
    )
    print(
        f"issued {data['issue_date'] or '-'}  due {data['due_date'] or '-'}"
        + (f"  ({data['days_overdue']} days overdue)" if data["days_overdue"] else "")
    )
    for line in data["lines"]:
        print(
            f"  {line['description']}: {line['quantity']} x {line['unit_price']} @ {line['tax_percent']}% = {line['net']}"
        )
    print(
        f"subtotal {data['subtotal']}  discount {data['discount']}  net {data['net']}"
    )
    for g in data["tax"]:
        print(f"tax {g['rate_percent']}%: {g['tax']}")
    print(
        f"total {data['total']} {cur}  paid {data['paid']}  balance {data['balance']}"
    )
    for p in data["payments"]:
        print(f"payment {p['paid_on']}: {p['amount']} {p['method']}".rstrip())
    for e in data["emails"]:
        print(
            f"email {e['at']} {e['kind']} to {e['to']}: {e['status']}"
            + (f" ({e['error']})" if e["error"] else "")
        )


@invoice_app.command(name="issue")
def invoice_issue(
    number: str, *, on: Annotated[date | None, Parameter(name="--date")] = None
) -> None:
    """Assign the next number and freeze the invoice (draft -> issued)."""
    settings = load_settings()
    with open_session(_engine(settings)) as session:
        inv = service.issue_invoice(
            session, service.get_invoice(session, number), settings, on
        )
        print(f"issued {inv.number}, due {inv.due_date}")


@invoice_app.command(name="pdf")
def invoice_pdf(
    number: str,
    *,
    output: Annotated[Path | None, Parameter(name=["--output", "-o"])] = None,
    facturx: Annotated[
        bool | None, Parameter(help="Embed Factur-X XML (default: from the config).")
    ] = None,
) -> None:
    """Render the invoice to PDF (PDF/A-3b with Factur-X XML when issued)."""
    settings = load_settings()
    with open_session(_engine(settings)) as session:
        inv = service.get_invoice(session, number)
        pdf = render.render_pdf(
            service.load_view(session, inv), settings, facturx=facturx
        )
        target = output or settings.pdf_dir / render.pdf_filename(inv.number, inv.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf)
    print(f"wrote {target} ({len(pdf) // 1024} KB)")


@invoice_app.command(name="send")
def invoice_send(
    number: str,
    *,
    to: Annotated[
        list[str] | None, Parameter(name="--to", help="Override the client's address.")
    ] = None,
    send: Annotated[
        bool, Parameter(help="Actually send. Without it nothing leaves this machine.")
    ] = False,
    resend: Annotated[bool, Parameter(help="Send again even if already sent.")] = False,
    retries: int = 2,
) -> None:
    """Email the invoice as a PDF attachment and record the result."""
    settings = load_settings()
    with open_session(_engine(settings)) as session:
        inv = service.get_invoice(session, number)
        if inv.status != ISSUED:
            hint = " (issue it first)" if inv.status == DRAFT else ""
            raise InvoicerError(
                f"{inv.number or 'draft ' + str(inv.id)} is {inv.status}{hint}"
            )
        view = service.load_view(session, inv)
        if (
            any(e.kind == "invoice" and e.status == "sent" for e in view.emails)
            and not resend
        ):
            raise InvoicerError(
                f"{inv.number} was already sent; use --resend to send it again"
            )
        pdf = render.render_pdf(view, settings)
        msg = mailer.build_message(view, settings, pdf, to=to or ())
        if not send:
            print(f"dry run: would send {inv.number} to {msg['To']}")
            print(f"subject: {msg['Subject']}  attachment: {len(pdf) // 1024} KB PDF")
            print("nothing was sent; add --send to deliver it")
            return
        outcome = mailer.deliver(
            session, view, settings, pdf, to=to or (), retries=retries, resend=resend
        )
    if outcome.status != "sent":
        print(
            f"FAILED after {outcome.attempts} attempt(s): {outcome.error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"sent {inv.number} to {outcome.recipient} ({outcome.message_id})")


@invoice_app.command(name="pay")
def invoice_pay(
    number: str,
    amount: str,
    *,
    on: Annotated[date | None, Parameter(name="--date")] = None,
    method: str = "",
    note: str = "",
) -> None:
    """Record a payment (partial payments are fine; overpayments are refused)."""
    with open_session(_engine(load_settings())) as session:
        inv = service.get_invoice(session, number)
        service.record_payment(
            session, inv, amount, paid_on=on, method=method, note=note
        )
        view = service.load_view(session, inv)
        print(
            f"{inv.number}: {view.status}, balance {_money(view.balance, inv.currency)}"
        )


@invoice_app.command(name="void")
def invoice_void(number: str) -> None:
    """Cancel an issued invoice with no payments; its number stays used."""
    with open_session(_engine(load_settings())) as session:
        inv = service.void_invoice(session, service.get_invoice(session, number))
        print(f"voided {inv.number}")


@invoice_app.command(name="delete")
def invoice_delete(number: str) -> None:
    """Delete a draft."""
    with open_session(_engine(load_settings())) as session:
        service.delete_draft(session, service.get_invoice(session, number))
        print(f"deleted draft {number}")


# --------------------------------------------------------------------------- recurring


@recurring_app.command(name="add")
def recurring_add(
    name: str,
    *,
    client: str,
    item: Annotated[
        list[str], Parameter(name="--item", help="'description | qty | price | tax%'")
    ],
    frequency: str = "monthly",
    start: date,
    interval: int = 1,
    end: date | None = None,
    discount: str | None = None,
    tax: str = "0",
    auto_send: Annotated[
        bool, Parameter(help="`recurring run --send` also emails them.")
    ] = False,
    notes: str = "",
) -> None:
    """Define an invoice that repeats; `recurring run` creates what is due."""
    from invoicer.money import parse_percent

    with open_session(_engine(load_settings())) as session:
        found = service.find_client(session, client)
        default_tax = parse_percent(tax)
        lines = [service.parse_line(spec, found.currency, default_tax) for spec in item]
        rec = service.add_recurring(
            session,
            name,
            found,
            lines,
            frequency=frequency,
            start=start,
            interval=interval,
            end=end,
            discount=service.parse_discount(discount, found.currency),
            auto_send=auto_send,
            notes=notes,
        )
        print(f"recurring {rec.id}: {rec.name}, next {rec.next_date}")


@recurring_app.command(name="list")
def recurring_list() -> None:
    """Show the recurring definitions."""
    from sqlmodel import select

    from invoicer.models import Recurring

    with open_session(_engine(load_settings())) as session:
        for r in session.exec(select(Recurring).order_by(Recurring.id)):
            state = "active" if r.active else "finished"
            every = (
                f"every {r.interval} {r.frequency}" if r.interval > 1 else r.frequency
            )
            print(f"{r.id}  {r.name}: {every}, next {r.next_date}, {state}")


@recurring_app.command(name="run")
def recurring_run(
    *,
    as_of: date | None = None,
    send: Annotated[
        bool, Parameter(help="Email auto-send invoices as they are created.")
    ] = False,
) -> None:
    """Create and issue every recurring invoice that is due (safe to run daily)."""
    settings = load_settings()
    as_of = as_of or date.today()
    failures = 0
    with open_session(_engine(settings)) as session:
        created = service.run_recurring(session, settings, as_of)
        for rec, inv in created:
            print(f"created {inv.number} from {rec.name!r} ({inv.issue_date})")
            if send and rec.auto_send:
                view = service.load_view(session, inv, as_of)
                try:
                    outcome = mailer.deliver(
                        session, view, settings, render.render_pdf(view, settings)
                    )
                except (
                    InvoicerError
                ) as exc:  # no address, already sent, send in progress
                    print(f"  not sent: {exc}")
                    failures += 1
                    continue
                print(
                    f"  {outcome.status} to {outcome.recipient}"
                    + (f": {outcome.error}" if outcome.error else "")
                )
                failures += outcome.status != "sent"
        if not created:
            print("nothing due")
    if failures:
        raise SystemExit(1)


@recurring_app.command(name="stop")
def recurring_stop(name: str) -> None:
    """Stop a recurring definition (invoices already created are untouched)."""
    from sqlmodel import select

    from invoicer.models import Recurring

    with open_session(_engine(load_settings())) as session:
        rec = session.exec(select(Recurring).where(Recurring.name == name)).first()
        if rec is None:
            raise InvoicerError(f"no recurring invoice named {name!r}")
        rec.active = False
        session.add(rec)
        print(f"stopped {rec.name}")


# ------------------------------------------------------------------------------ reminders


@app.command
def remind(
    *,
    as_of: date | None = None,
    send: Annotated[
        bool, Parameter(help="Actually send. Without it, only list what is due.")
    ] = False,
) -> None:
    """Find overdue invoices that reached a reminder stage and email each one once per stage."""
    settings = load_settings()
    as_of = as_of or date.today()
    failures = 0
    with open_session(_engine(settings)) as session:
        due = service.due_reminders(session, settings, as_of)
        if not due:
            print("no reminders due")
        for r in due:
            v = r.view
            label = f"{v.invoice.number} ({v.client.name}, {v.days_overdue} days overdue, stage {r.stage})"
            if not send:
                print(f"would remind: {label}")
                continue
            pdf = render.render_pdf(v, settings)
            try:
                outcome = mailer.deliver(
                    session, v, settings, pdf, kind="reminder", stage=r.stage
                )
            except InvoicerError as exc:  # no address, or another run is sending it
                print(f"skipped: {label} - {exc}")
                failures += 1
                continue
            print(
                f"{outcome.status}: {label}"
                + (f" - {outcome.error}" if outcome.error else "")
            )
            failures += outcome.status != "sent"
    if failures:
        raise SystemExit(1)


@app.command
def summary(*, as_of: date | None = None) -> None:
    """Money outstanding per currency, and how much of it is overdue."""
    with open_session(_engine(load_settings())) as session:
        views = service.list_views(session, today=as_of)
        owed = service.outstanding(views)
        late = service.outstanding([v for v in views if v.status == "overdue"])
    if not owed:
        print("nothing outstanding")
    for cur, minor in owed.items():
        print(
            f"{cur}: {plain_amount(minor, cur)} outstanding, {plain_amount(late.get(cur, 0), cur)} overdue"
        )


@app.command
def serve(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local web UI. Loopback only: it has no login, and anyone who can reach it can
    issue, void and email invoices."""
    import uvicorn

    from invoicer.web import LOCAL_HOSTS, create_app

    if host.strip("[]") not in {h.strip("[]") for h in LOCAL_HOSTS}:
        raise InvoicerError(
            f"refusing to listen on {host}: the web UI has no authentication, so it only "
            "binds to localhost. Put it behind an authenticating reverse proxy or an SSH "
            "tunnel if you need remote access."
        )

    app = create_app(load_settings(), port=port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    try:
        result = app(argv, result_action="return_value")
    except InvoicerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
