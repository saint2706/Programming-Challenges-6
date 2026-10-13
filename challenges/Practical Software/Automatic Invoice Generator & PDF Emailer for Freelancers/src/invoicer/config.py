"""``invoicer.toml``: who you are, how invoices are numbered, and how mail is sent.

Secrets never live in the file: the SMTP password is read from the environment variable named by
``email.password_env``.
"""

import os
import re
import string
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from invoicer.errors import InvoicerError
from invoicer.money import MoneyError, normalize_currency
from invoicer.paths import project_root

CONFIG_NAME = "invoicer.toml"
SECURITY_MODES = ("starttls", "ssl", "none")
_NUMBER_FIELDS = {"year", "month", "seq"}

SAMPLE_CONFIG = """\
# invoicer configuration. Run `uv run invoicer config-check` after editing.

[business]
name = "Ada Lovelace Consulting"
address = "12 Analytical Way\\nLondon N1 9GU"
country = "GB"                # ISO 3166 two-letter code (required for e-invoices)
email = "ada@example.com"
tax_id = "GB123456789"        # VAT number; shown on invoices and in the e-invoice XML
currency = "GBP"              # default for new clients and invoices
iban = "GB29NWBK60161331926819"
bic = "NWBKGB2L"
payment_note = "Please pay by bank transfer, quoting the invoice number."
# logo = "logo.png"           # embedded in the PDF; relative to this file

[numbering]
# Fields: {year} {month} {seq}; the sequence restarts whenever the rest of the number changes.
format = "INV-{year}-{seq:04d}"

[email]
host = "smtp.example.com"
port = 587
security = "starttls"         # starttls | ssl | none
username = "ada@example.com"
password_env = "INVOICER_SMTP_PASSWORD"
from_address = "ada@example.com"
from_name = "Ada Lovelace Consulting"
# bcc = "books@example.com"
# ca_file = "private-ca.pem"   # trust an extra CA; certificates are always verified

[reminders]
days_after_due = [7, 14, 30]  # at most one reminder per invoice per stage

[pdf]
locale = "en_GB"              # number/date formatting on the invoice
facturx = true                # embed a Factur-X (BASIC WL) XML in a PDF/A-3b file
# template_dir = "templates"  # your own invoice.html overrides the built-in one
"""


@dataclass(frozen=True)
class Business:
    name: str = "Your Business"
    address: str = ""
    country: str = "US"
    email: str = ""
    tax_id: str = ""
    currency: str = "USD"
    iban: str = ""
    bic: str = ""
    payment_note: str = ""
    logo: Path | None = None


@dataclass(frozen=True)
class EmailSettings:
    host: str = ""
    port: int = 587
    security: str = "starttls"
    username: str = ""
    password_env: str = "INVOICER_SMTP_PASSWORD"
    from_address: str = ""
    from_name: str = ""
    bcc: str = ""
    timeout: float = 30.0
    ca_file: Path | None = None  # extra CA to trust, e.g. for a private mail server

    @property
    def password(self) -> str | None:
        return os.environ.get(self.password_env) if self.password_env else None


@dataclass(frozen=True)
class Settings:
    business: Business = field(default_factory=Business)
    email: EmailSettings = field(default_factory=EmailSettings)
    number_format: str = "INV-{year}-{seq:04d}"
    reminder_days: tuple[int, ...] = (7, 14, 30)
    locale: str = "en_US"
    facturx: bool = True
    template_dir: Path | None = None
    root: Path = field(default_factory=project_root)

    @property
    def db_path(self) -> Path:
        return self.root / "invoicer.db"

    @property
    def pdf_dir(self) -> Path:
        return self.root / "pdf"


def validate_number_format(fmt: str) -> str:
    """Only ``{year}``, ``{month}`` and exactly one ``{seq}``: ``str.format`` on arbitrary
    templates can reach object attributes, so everything else is refused."""
    seq = 0
    try:
        parsed = list(string.Formatter().parse(fmt))
    except ValueError as exc:
        raise InvoicerError(
            f"numbering.format {fmt!r} is not a valid format: {exc}"
        ) from exc
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if name not in _NUMBER_FIELDS or conversion or (spec and "{" in spec):
            raise InvoicerError(
                f"numbering.format may only use {{year}}, {{month}}, {{seq}}: {fmt!r}"
            )
        seq += name == "seq"
    if seq != 1:
        raise InvoicerError("numbering.format needs exactly one {seq}")
    try:
        fmt.format(year=2026, month=1, seq=1)
    except (ValueError, TypeError) as exc:
        raise InvoicerError(
            f"numbering.format {fmt!r} cannot format numbers: {exc}"
        ) from exc
    return fmt


def number_scope(fmt: str, year: int, month: int) -> str:
    """Identifies one numbering run: the format with {seq} blanked, e.g. ``INV-2026-#``."""
    blanked = re.sub(r"\{seq[^}]*\}", "#", fmt)
    return blanked.format(year=year, month=month)


def _currency(code: str) -> str:
    try:
        return normalize_currency(code)
    except MoneyError as exc:
        raise InvoicerError(f"business.currency: {exc}") from exc


def _table(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise InvoicerError(f"[{key}] must be a table")
    return value


def _str(table: dict, key: str, default: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise InvoicerError(f"{key} must be a string")
    return value


def load_settings(path: Path | None = None) -> Settings:
    root = project_root()
    path = path or root / CONFIG_NAME
    if not path.exists():
        return Settings(root=root)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InvoicerError(f"cannot read {path}: {exc}") from exc
    base = path.parent
    b, e = _table(data, "business"), _table(data, "email")
    n, r, p = _table(data, "numbering"), _table(data, "reminders"), _table(data, "pdf")
    logo = b.get("logo")
    business = Business(
        name=_str(b, "name", Business.name),
        address=_str(b, "address", ""),
        country=_str(b, "country", Business.country).upper(),
        email=_str(b, "email", ""),
        tax_id=_str(b, "tax_id", ""),
        currency=_currency(_str(b, "currency", Business.currency)),
        iban=_str(b, "iban", "").replace(" ", ""),
        bic=_str(b, "bic", ""),
        payment_note=_str(b, "payment_note", ""),
        logo=(base / logo) if isinstance(logo, str) else None,
    )
    if len(business.country) != 2 or not business.country.isalpha():
        raise InvoicerError("business.country must be a two-letter ISO 3166 code")
    security = _str(e, "security", "starttls")
    if security not in SECURITY_MODES:
        raise InvoicerError(
            f"email.security must be one of {', '.join(SECURITY_MODES)}"
        )
    port = e.get("port", 587)
    if not isinstance(port, int) or not 0 < port < 65536:
        raise InvoicerError("email.port must be a port number")
    email = EmailSettings(
        host=_str(e, "host", ""),
        port=port,
        security=security,
        username=_str(e, "username", ""),
        password_env=_str(e, "password_env", EmailSettings.password_env),
        from_address=_str(e, "from_address", business.email),
        from_name=_str(e, "from_name", business.name),
        bcc=_str(e, "bcc", ""),
        timeout=float(e.get("timeout", 30)),
        ca_file=(base / e["ca_file"]) if isinstance(e.get("ca_file"), str) else None,
    )
    days = r.get("days_after_due", [7, 14, 30])
    if not isinstance(days, list) or not all(
        isinstance(d, int) and d > 0 for d in days
    ):
        raise InvoicerError(
            "reminders.days_after_due must be a list of positive integers"
        )
    template_dir = p.get("template_dir")
    return Settings(
        business=business,
        email=email,
        number_format=validate_number_format(_str(n, "format", Settings.number_format)),
        reminder_days=tuple(sorted(set(days))),
        locale=_str(p, "locale", "en_US"),
        facturx=bool(p.get("facturx", True)),
        template_dir=(base / template_dir) if isinstance(template_dir, str) else None,
        root=root,
    )
