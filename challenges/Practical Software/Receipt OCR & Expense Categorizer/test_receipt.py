"""Tests for receipt parsing from OCR text."""

from datetime import datetime

import pytest

from receipt import extract_date, extract_merchant, extract_total, parse_receipt


class TestExtractMerchant:
    def test_first_line_is_merchant(self):
        lines = ["WHOLE FOODS MARKET", "123 Main St", "Date: 01/15/2026"]
        assert extract_merchant(lines) == "WHOLE FOODS MARKET"

    def test_empty_lines_skipped(self):
        lines = ["", "  ", "STARBUCKS"]
        assert extract_merchant(lines) == "STARBUCKS"

    def test_empty_input(self):
        assert extract_merchant([]) == "Unknown"

    def test_whitespace_trimmed(self):
        lines = ["  McDonald's  ", "123 Street"]
        assert extract_merchant(lines) == "McDonald's"

    def test_very_short_merchant_name(self):
        lines = ["A"]
        assert extract_merchant(lines) == "Unknown"

    def test_merchant_truncated_to_100_chars(self):
        long_name = "A" * 150
        lines = [long_name]
        assert len(extract_merchant(lines)) == 100


class TestExtractDate:
    def test_mm_dd_yyyy_format(self):
        lines = ["RECEIPT", "Date: 01/15/2026"]
        result = extract_date(lines)
        assert result is not None
        assert result.month == 1
        assert result.day == 15
        assert result.year == 2026

    def test_dd_mm_yyyy_format_swapped_month_day(self):
        lines = ["RECEIPT", "Date: 15/01/2026"]
        result = extract_date(lines)
        assert result is not None
        assert result.month == 1
        assert result.day == 15
        assert result.year == 2026

    def test_yyyy_mm_dd_format(self):
        lines = ["2026-05-10 Transaction"]
        result = extract_date(lines)
        assert result is not None
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 10

    def test_month_name_and_day(self):
        lines = ["RECEIPT", "Jan 5, 2026"]
        result = extract_date(lines)
        assert result is not None
        assert result.month == 1
        assert result.day == 5
        assert result.year == 2026

    def test_month_name_alternate_format(self):
        lines = ["5 January 2026"]
        result = extract_date(lines)
        assert result is not None
        assert result.month == 1
        assert result.day == 5
        assert result.year == 2026

    def test_various_separators(self):
        assert extract_date(["Date: 12/25/2025"]) is not None
        assert extract_date(["Date: 12-25-2025"]) is not None

    def test_no_date_found(self):
        lines = ["NO DATE HERE", "Just some text"]
        assert extract_date(lines) is None

    def test_invalid_date(self):
        lines = ["Date: 13/32/2026"]
        assert extract_date(lines) is None


class TestExtractTotal:
    def test_total_keyword_with_amount(self):
        lines = ["Item 1", "TOTAL: $42.50"]
        result = extract_total(lines)
        assert result is not None
        assert abs(result - 42.50) < 0.01

    def test_amount_due_keyword(self):
        lines = ["Subtotal: $40", "Tax: $3.50", "Amount Due: $43.50"]
        result = extract_total(lines)
        assert result is not None

    def test_grand_total_keyword(self):
        lines = ["Items...", "GRAND TOTAL $99.99"]
        result = extract_total(lines)
        assert result is not None
        assert abs(result - 99.99) < 0.01

    def test_amount_with_comma_thousands(self):
        lines = ["TOTAL: $1,234.56"]
        result = extract_total(lines)
        assert result is not None
        assert abs(result - 1234.56) < 0.01

    def test_last_amount_as_fallback(self):
        lines = ["Item: $5.99", "Tax: $0.50", "TOTAL $6.49"]
        result = extract_total(lines)
        assert result is not None

    def test_no_amount_found(self):
        lines = ["NO NUMBERS HERE"]
        assert extract_total(lines) is None

    def test_malformed_total(self):
        lines = ["TOTAL: ABC"]
        assert extract_total(lines) is None

    def test_dollar_sign_prefix(self):
        lines = ["Grand Total $ 19.99"]
        result = extract_total(lines)
        assert result is not None


class TestParseReceipt:
    def test_full_receipt_parsing(self):
        lines = [
            "WHOLE FOODS MARKET",
            "123 Main Street",
            "Date: 01/15/2026",
            "Apples",
            "$3.99",
            "Bread",
            "$4.49",
            "Milk",
            "$3.99",
            "TAX",
            "$1.10",
            "TOTAL: $13.57",
        ]
        receipt = parse_receipt(lines)

        assert receipt.merchant == "WHOLE FOODS MARKET"
        assert receipt.date is not None
        assert receipt.date.month == 1
        assert receipt.date.day == 15
        assert receipt.date.year == 2026
        assert receipt.total is not None
        assert receipt.category == "Other"

    def test_partial_receipt_parsing(self):
        lines = ["STARBUCKS"]
        receipt = parse_receipt(lines)

        assert receipt.merchant == "STARBUCKS"
        assert receipt.date is None
        assert receipt.total is None

    def test_receipt_preserves_raw_text(self):
        lines = ["Merchant", "Date: 01/01/2026", "Total: $10.00"]
        receipt = parse_receipt(lines)

        assert receipt.raw_text == lines

    def test_receipt_has_unique_id(self):
        lines = ["Test"]
        receipt1 = parse_receipt(lines)
        receipt2 = parse_receipt(lines)

        assert receipt1.id != receipt2.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
