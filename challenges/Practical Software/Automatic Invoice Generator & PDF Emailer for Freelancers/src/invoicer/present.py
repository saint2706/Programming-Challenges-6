"""Plain-data views of invoices, shared by ``--json`` output, the CLI tables and the web UI."""

from invoicer.money import format_percent, format_quantity, plain_amount
from invoicer.service import InvoiceView


def view_dict(view: InvoiceView) -> dict:
    inv, t = view.invoice, view.totals
    cur = inv.currency
    amt = lambda minor: plain_amount(minor, cur)
    return {
        "id": inv.id,
        "number": inv.number,
        "status": view.status,
        "client": view.client.name,
        "currency": cur,
        "issue_date": inv.issue_date.isoformat() if inv.issue_date else None,
        "due_date": inv.due_date.isoformat() if inv.due_date else None,
        "days_overdue": view.days_overdue,
        "reference": inv.reference,
        "notes": inv.notes,
        "lines": [
            {
                "description": line.description,
                "quantity": format_quantity(line.qty_e4),
                "unit_price": amt(line.unit_minor),
                "tax_percent": format_percent(line.tax_bp),
                "net": amt(r.net),
            }
            for line, r in zip(view.lines, t.lines, strict=True)
        ],
        "subtotal": amt(t.subtotal),
        "discount": amt(t.discount),
        "net": amt(t.net),
        "tax": [
            {"rate_percent": format_percent(g.rate_bp), "tax": amt(g.tax)}
            for g in t.tax_groups
        ],
        "total": amt(t.total),
        "paid": amt(view.paid),
        "balance": amt(view.balance),
        "payments": [
            {
                "amount": amt(p.amount_minor),
                "paid_on": p.paid_on.isoformat(),
                "method": p.method,
                "note": p.note,
            }
            for p in view.payments
        ],
        "emails": [
            {
                "kind": e.kind,
                "stage": e.stage,
                "to": e.recipient,
                "status": e.status,
                "at": e.attempted_at.isoformat(timespec="seconds"),
                "attempts": e.attempts,
                "error": e.error,
            }
            for e in view.emails
        ],
    }
