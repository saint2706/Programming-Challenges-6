from decimal import Decimal

import pytest
from invoicer import money
from invoicer.money import MoneyError


@pytest.mark.parametrize(
    ("currency", "digits"), [("EUR", 2), ("usd", 2), ("JPY", 0), ("KWD", 3), ("BHD", 3)]
)
def test_exponent_follows_iso_4217(currency, digits):
    assert money.exponent(currency) == digits


@pytest.mark.parametrize("bad", ["", "EU", "EURO", "12E", "ZZZZ"])
def test_bad_currency_codes(bad):
    with pytest.raises(MoneyError):
        money.normalize_currency(bad)


@pytest.mark.parametrize(
    ("text", "currency", "minor"),
    [
        ("12.50", "EUR", 1250),
        ("12.5", "EUR", 1250),
        ("12", "EUR", 1200),
        ("1,234.56", "EUR", 123456),
        ("0.01", "EUR", 1),
        ("500", "JPY", 500),
        ("1.234", "KWD", 1234),
        (7, "EUR", 700),
        (Decimal("3.10"), "EUR", 310),
    ],
)
def test_to_minor(text, currency, minor):
    assert money.to_minor(text, currency) == minor


@pytest.mark.parametrize(
    ("text", "currency"),
    [
        ("10.005", "EUR"),
        ("1.5", "JPY"),
        ("1.2345", "KWD"),
        ("abc", "EUR"),
        ("nan", "EUR"),
        ("inf", "EUR"),
    ],
)
def test_to_minor_refuses_instead_of_rounding(text, currency):
    with pytest.raises(MoneyError):
        money.to_minor(text, currency)


def test_floats_are_refused():
    with pytest.raises(MoneyError, match="float"):
        money.to_minor(0.1 + 0.2, "EUR")


def test_minor_round_trip():
    assert money.plain_amount(123456, "EUR") == "1234.56"
    assert money.plain_amount(5, "EUR") == "0.05"
    assert money.plain_amount(500, "JPY") == "500"
    assert money.plain_amount(1234, "KWD") == "1.234"
    assert money.from_minor(-250, "EUR") == Decimal("-2.50")


def test_format_money_is_locale_aware():
    assert money.format_money(123456, "EUR", "en_GB") == "€1,234.56"
    assert (
        money.format_money(123456, "EUR", "de_DE").replace("\xa0", " ") == "1.234,56 €"
    )
    assert money.format_money(500, "JPY", "en_US") == "¥500"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0.5", 1), ("1.5", 2), ("2.5", 3), ("2.4999", 2), ("-0.5", -1), ("-2.5", -3)],
)
def test_round_half_up_rounds_away_from_zero(value, expected):
    # Python's round() would give banker's rounding (2.5 -> 2); invoices round .5 up.
    assert money.round_half_up(Decimal(value)) == expected


@pytest.mark.parametrize(
    ("text", "qty_e4"),
    [("1", 10_000), ("1.5", 15_000), ("0.0001", 1), ("12.3456", 123_456)],
)
def test_parse_quantity(text, qty_e4):
    assert money.parse_quantity(text) == qty_e4


@pytest.mark.parametrize("bad", ["0", "-1", "0.00001", "x", ""])
def test_bad_quantities(bad):
    with pytest.raises(MoneyError):
        money.parse_quantity(bad)


def test_quantity_formatting():
    assert money.format_quantity(15_000) == "1.5"
    assert money.format_quantity(10_000) == "1"
    assert money.format_quantity(1) == "0.0001"
    assert money.format_quantity(1_000_000) == "100"


@pytest.mark.parametrize(
    ("text", "bp"),
    [("20", 2000), ("7.5", 750), ("0", 0), ("100", 10_000), ("19.25", 1925)],
)
def test_parse_percent(text, bp):
    assert money.parse_percent(text) == bp
    assert money.parse_percent(money.format_percent(bp)) == bp


@pytest.mark.parametrize("bad", ["-1", "100.01", "1.234", "x"])
def test_bad_percentages(bad):
    with pytest.raises(MoneyError):
        money.parse_percent(bad)
