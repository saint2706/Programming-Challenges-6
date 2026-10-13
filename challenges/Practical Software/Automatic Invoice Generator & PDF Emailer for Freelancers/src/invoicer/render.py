"""Invoice -> HTML (Jinja2) -> PDF/A-3b (WeasyPrint) -> optionally Factur-X (embedded CII XML).

Two safety properties worth stating:

* Templates are rendered in Jinja's sandbox with autoescaping on, so a client called
  ``<script>`` or ``{{ 7*7 }}`` appears as text, and a user-supplied template cannot reach
  Python internals.
* WeasyPrint is only allowed to load ``data:`` URLs. An invoice's text comes from the database;
  without this a ``<img src="file:///...">`` or ``http://internal/...`` smuggled into a template
  or field would read local files or call internal services while the PDF is built.
"""

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

from babel.dates import format_date
from jinja2 import ChoiceLoader, FileSystemLoader, PackageLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from invoicer.config import Settings
from invoicer.errors import InvoicerError
from invoicer.models import DRAFT, VOID
from invoicer.money import format_money, format_percent, format_quantity
from invoicer.service import InvoiceView, describe_discount

LABELS = {
    "number": "Invoice no.",
    "issued": "Issued",
    "due": "Due",
    "reference": "Reference",
    "from": "From",
    "bill_to": "Bill to",
    "tax_id": "VAT ID",
    "description": "Description",
    "qty": "Qty",
    "unit_price": "Unit price",
    "tax": "Tax",
    "amount": "Amount",
    "subtotal": "Subtotal",
    "discount": "Discount",
    "net": "Net",
    "total": "Total",
    "paid": "Paid",
    "balance": "Balance due",
    "payment": "Payment details",
}
_STAMPS = {"paid": "PAID", VOID: "VOID", DRAFT: "DRAFT"}
_MAX_LOGO_BYTES = 2_000_000


def _pango_dirs() -> list[Path]:
    home = Path.home()
    return [
        home / "scoop" / "apps" / "msys2" / "current" / "ucrt64" / "bin",
        Path("C:/msys64/ucrt64/bin"),
    ]


def load_weasyprint():
    """Import WeasyPrint, pointing it at MSYS2's Pango on Windows when it is not configured.

    WeasyPrint needs the Pango/GLib DLLs, which pip cannot install on Windows. Set
    ``WEASYPRINT_DLL_DIRECTORIES`` yourself to use another install.
    """
    if sys.platform == "win32" and not os.environ.get("WEASYPRINT_DLL_DIRECTORIES"):
        for candidate in _pango_dirs():
            if (candidate / "libgobject-2.0-0.dll").exists():
                os.environ["WEASYPRINT_DLL_DIRECTORIES"] = str(candidate)
                break
    try:
        import weasyprint
    except OSError as exc:  # missing shared libraries
        raise InvoicerError(
            "WeasyPrint cannot load Pango. Linux: install libpango-1.0-0; macOS: brew install "
            "pango; Windows: install MSYS2 (scoop install msys2, then pacman -S "
            "mingw-w64-ucrt-x86_64-pango) or set WEASYPRINT_DLL_DIRECTORIES. "
            f"({exc})"
        ) from exc
    return weasyprint


def _safe_fetcher(weasyprint):
    """Only ``data:`` URLs, and no redirects: nothing outside the document can be loaded."""
    from weasyprint.urls import URLFetcher

    return URLFetcher(allowed_protocols=("data",), allow_redirects=False)


def _environment(settings: Settings) -> SandboxedEnvironment:
    loaders = []
    if settings.template_dir is not None:
        loaders.append(FileSystemLoader(str(settings.template_dir)))
    loaders.append(PackageLoader("invoicer", "templates"))
    return SandboxedEnvironment(
        loader=ChoiceLoader(loaders), autoescape=True, undefined=StrictUndefined
    )


def _logo_data_uri(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) > _MAX_LOGO_BYTES:
        raise InvoicerError(f"logo {path.name} is over 2 MB")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime.startswith("image/"):
        raise InvoicerError(f"logo {path.name} is not an image")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _parties(view: InvoiceView, settings: Settings) -> tuple[dict, dict]:
    if view.invoice.snapshot:
        snap = json.loads(view.invoice.snapshot)
        return snap["seller"], snap["buyer"]
    b, c = settings.business, view.client
    seller = {
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
    buyer = {
        k: getattr(c, k) for k in ("name", "address", "country", "email", "tax_id")
    }
    return seller, buyer


def build_context(view: InvoiceView, settings: Settings) -> dict:
    inv, totals = view.invoice, view.totals
    cur, loc = inv.currency, settings.locale
    money = lambda minor: format_money(minor, cur, loc)
    seller, buyer = _parties(view, settings)
    day = lambda d: format_date(d, format="medium", locale=loc) if d else ""
    stamp = _STAMPS.get(view.status)
    number = inv.number or "DRAFT"
    return {
        "lang": loc.split("_")[0],
        "title": f"Invoice {number}",
        "heading": "Invoice",
        "labels": LABELS,
        "number": number,
        "issue_date": day(inv.issue_date),
        "due_date": day(inv.due_date),
        "reference": inv.reference,
        "seller": seller,
        "buyer": buyer,
        "logo": _logo_data_uri(settings.business.logo),
        "lines": [
            {
                "description": line.description,
                "qty": format_quantity(line.qty_e4),
                "unit_price": money(line.unit_minor),
                "tax": f"{format_percent(line.tax_bp)}%",
                "net": money(r.net),
            }
            for line, r in zip(view.lines, totals.lines, strict=True)
        ],
        "totals": {
            "subtotal": money(totals.subtotal),
            "discount": money(totals.discount) if totals.discount else "",
            "net": money(totals.net),
            "tax_groups": [
                {"rate": f"{format_percent(g.rate_bp)}%", "tax": money(g.tax)}
                for g in totals.tax_groups
            ],
            "total": money(totals.total),
            "paid": money(view.paid) if view.paid else "",
            "balance": money(view.balance),
        },
        "discount_label": describe_discount(inv)
        if inv.discount_kind == "percent"
        else "",
        "notes": inv.notes,
        "stamp": stamp,
        "stamp_class": view.status if stamp else "",
    }


def render_html(view: InvoiceView, settings: Settings) -> str:
    template = _environment(settings).get_template("invoice.html")
    return template.render(build_context(view, settings))


def render_pdf(
    view: InvoiceView, settings: Settings, *, facturx: bool | None = None
) -> bytes:
    """PDF/A-3b bytes; with ``facturx`` (default: the config's choice, for issued invoices) the
    BASIC WL XML is embedded so accounting software can read the invoice without OCR."""
    weasyprint = load_weasyprint()
    html = render_html(view, settings)
    want_xml = settings.facturx if facturx is None else facturx
    want_xml = want_xml and bool(
        view.invoice.snapshot
    )  # drafts have no number or frozen parties
    variant = "pdf/a-3b" if want_xml else None
    try:
        pdf = weasyprint.HTML(
            string=html, url_fetcher=_safe_fetcher(weasyprint)
        ).write_pdf(pdf_variant=variant)
    except Exception as exc:  # WeasyPrint raises many unrelated types
        raise InvoicerError(f"PDF rendering failed: {exc}") from exc
    if not want_xml:
        return pdf
    import facturx as facturx_lib

    from invoicer.facturx_xml import LEVEL, build_xml

    seller = json.loads(view.invoice.snapshot)["seller"]
    return facturx_lib.generate_from_binary(
        pdf,
        build_xml(view),
        flavor="factur-x",
        level=LEVEL,
        pdf_metadata={
            "author": seller["name"],
            "title": f"Invoice {view.invoice.number}",
            "subject": f"Invoice {view.invoice.number} from {seller['name']}",
        },
    )


def pdf_filename(number: str | None, invoice_id: int) -> str:
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in (number or f"draft-{invoice_id}")
    )
    return f"{safe}.pdf"
