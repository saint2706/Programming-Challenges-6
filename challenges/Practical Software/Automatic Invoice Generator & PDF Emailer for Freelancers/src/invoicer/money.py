"""Exact money: integer minor units (cents), with the minor-unit count taken from ISO 4217.

Floats never touch an amount. Quantities are integers in ten-thousandths and tax rates are
integers in basis points (1 bp = 0.01 %), so every calculation is integer or ``Decimal``.
"""

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from babel.numbers import format_currency, get_currency_precision

QTY_SCALE = 10_000  # quantities are stored as integer ten-thousandths: 1.5 -> 15000
BP_SCALE = 10_000  # tax rates and percentage discounts are basis points: 20 % -> 2000
MAX_AMOUNT = 10**13  # largest price or payment, in whole currency units
MAX_QUANTITY = 10**8


class MoneyError(ValueError):
    pass


def exponent(currency: str) -> int:
    """Number of minor-unit digits: 2 for EUR/USD, 0 for JPY, 3 for KWD."""
    try:
        return get_currency_precision(_code(currency))
    except Exception as exc:  # babel raises varied errors for unknown codes
        raise MoneyError(f"unknown currency {currency!r}") from exc


def _code(currency: str) -> str:
    code = currency.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise MoneyError(f"currency must be a 3-letter ISO 4217 code, got {currency!r}")
    return code


def normalize_currency(currency: str) -> str:
    code = _code(currency)
    exponent(code)
    return code


def round_half_up(value: Decimal) -> int:
    """Commercial rounding: .5 rounds away from zero, as invoices conventionally do."""
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _decimal(text: str | int | Decimal, what: str, limit: int = MAX_AMOUNT) -> Decimal:
    if isinstance(text, float):
        raise MoneyError(f"{what} must not be a float")
    try:
        value = Decimal(str(text).strip().replace(",", ""))
    except InvalidOperation as exc:
        raise MoneyError(f"{what} {text!r} is not a number") from exc
    if not value.is_finite():
        raise MoneyError(f"{what} {text!r} is not a finite number")
    # Checked before any arithmetic: Decimal silently rounds past 28 digits, and SQLite
    # integers overflow at 2**63, so absurd values must be refused, not quietly altered.
    if abs(value) > limit:
        raise MoneyError(f"{what} {text!r} is too large (the limit is {limit:,})")
    return value


def to_minor(amount: str | int | Decimal, currency: str) -> int:
    """``'12.50'`` -> 1250 for EUR. More decimals than the currency has is an error, not a
    silent rounding: ``'10.005'`` could be a typo for ``'10.05'``."""
    value = _decimal(amount, "amount")
    digits = exponent(currency)
    scaled = value * (10**digits)
    if scaled != scaled.to_integral_value():
        raise MoneyError(
            f"{amount!r} has more than {digits} decimal places for {currency}"
        )
    return int(scaled)


def from_minor(minor: int, currency: str) -> Decimal:
    return Decimal(minor).scaleb(-exponent(currency))


def format_money(minor: int, currency: str, locale: str = "en_US") -> str:
    return format_currency(from_minor(minor, currency), _code(currency), locale=locale)


def plain_amount(minor: int, currency: str) -> str:
    """Machine-readable ``1234.50``: used in the e-invoice XML and CLI tables."""
    return f"{from_minor(minor, currency):.{exponent(currency)}f}"


def parse_quantity(text: str | int | Decimal) -> int:
    """``'1.5'`` -> 15000; at most four decimals, and strictly positive."""
    value = _decimal(text, "quantity", MAX_QUANTITY)
    scaled = value * QTY_SCALE
    if scaled != scaled.to_integral_value():
        raise MoneyError(f"quantity {text!r} has more than 4 decimal places")
    if scaled <= 0:
        raise MoneyError("quantity must be greater than zero")
    return int(scaled)


def format_quantity(qty_e4: int) -> str:
    text = f"{Decimal(qty_e4) / QTY_SCALE:f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_percent(text: str | int | Decimal) -> int:
    """``'20'`` -> 2000 bp, ``'7.5'`` -> 750; at most two decimals, 0-100 %."""
    value = _decimal(text, "percentage", BP_SCALE)
    scaled = value * 100
    if scaled != scaled.to_integral_value():
        raise MoneyError(f"percentage {text!r} has more than 2 decimal places")
    if not 0 <= scaled <= BP_SCALE:
        raise MoneyError("percentage must be between 0 and 100")
    return int(scaled)


def format_percent(bp: int) -> str:
    text = f"{Decimal(bp) / 100:f}"
    return text.rstrip("0").rstrip(".") if "." in text else text
