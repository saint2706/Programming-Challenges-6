"""The machine-readable half of a Factur-X invoice: a CII XML at the BASIC WL profile
("without lines": header, parties, VAT breakdown and totals), built with the ``factur-x``
library, which validates it against the official XSD before returning it.

The XSD checks structure only. The profile's business rules (Schematron) need a Saxon server
and are not run here.
"""

import json

import facturx

from invoicer.errors import InvoicerError
from invoicer.money import plain_amount
from invoicer.service import InvoiceView

LEVEL = "basicwl"
SPEC_ID = "urn:factur-x.eu:1p0:basicwl"


def _party(info: dict) -> dict:
    lines = [ln.strip() for ln in info.get("address", "").splitlines() if ln.strip()]
    party = {"name": info["name"], "country_code": info["country"]}
    if lines:
        party["addr_l1"] = lines[0]
    if len(lines) > 1:
        party["addr_l2"] = ", ".join(lines[1:])
    if info.get("tax_id"):
        party["vat_identifier"] = info["tax_id"].replace(" ", "")
    return party


def build_data(view: InvoiceView) -> dict:
    inv, totals = view.invoice, view.totals
    cur = inv.currency
    snap = json.loads(inv.snapshot or "{}")
    if not snap:
        raise InvoicerError("only issued invoices can be exported as e-invoices")
    money = lambda minor: plain_amount(minor, cur)

    # One VAT breakdown per rate; a 0 % rate is category Z (zero rated), anything else S (standard).
    groups = [
        {
            "BT-116": money(g.taxable),
            "BT-117": money(g.tax),
            "BT-118": "S" if g.rate_bp else "Z",
            "BT-119": f"{g.rate_bp / 100:.2f}",
        }
        for g in totals.tax_groups
    ]
    data = {
        "BT-1": inv.number,
        "BT-2": inv.issue_date,
        "BT-3": "380",  # commercial invoice
        "BT-5": cur,
        "BT-9": inv.due_date,
        "BT-24": SPEC_ID,
        "BG-4": _party(snap["seller"]),
        "BG-7": _party(snap["buyer"]),
        # BASIC WL requires a non-empty delivery section. Services are delivered to the buyer, so
        # that is stated rather than inventing a delivery date.
        "BG-13": {
            "name": snap["buyer"]["name"],
            "country_code": snap["buyer"]["country"],
        },
        "BG-23": groups,
        "BT-106": money(totals.subtotal),
        "BT-109": money(totals.net),
        "BT-110": money(totals.tax),
        "BT-112": money(totals.total),
        "BT-115": money(view.balance),
    }
    if view.paid:
        data["BT-113"] = money(view.paid)
    if inv.reference:
        data["BT-13"] = inv.reference
    if totals.discount:
        # A document-level allowance must name its VAT category, so the discount is reported
        # per tax rate: the share each rate group's lines received.
        per_rate: dict[int, int] = {}
        for line in totals.lines:
            per_rate[line.tax_bp] = per_rate.get(line.tax_bp, 0) + line.discount
        data["BT-107"] = money(totals.discount)
        data["BG-20"] = [
            {
                "amount": money(amount),
                "reason": "Discount",
                "vat_category_code": "S" if rate else "Z",
                "vat_rate": f"{rate / 100:.2f}",
            }
            for rate, amount in sorted(per_rate.items())
            if amount
        ]
    iban = snap["seller"].get("iban")
    if iban:
        data["BT-81"] = "58"  # SEPA credit transfer
        data["BT-84"] = iban
    return data


def build_xml(view: InvoiceView) -> bytes:
    try:
        return facturx.generate_cii_xml(
            build_data(view), level=LEVEL, check_xsd=True, check_schematron=False
        )
    except ValueError as exc:
        raise InvoicerError(f"cannot build the e-invoice XML: {exc}") from exc
