"""Receipt parsing from raw OCR text.

Pure parsing functions that operate on line strings extracted from OCR,
independent of any particular OCR library. This module handles all the messy
regex and heuristic work to extract structured fields from inconsistent
receipt formats, and is extensively tested with fixture data rather than
live OCR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Receipt:
    """A parsed receipt with extracted structured fields."""

    merchant: str
    category: str = "Other"
    date: datetime | None = None
    total: float | None = None
    raw_text: list[str] = field(default_factory=list)
    image_path: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))


def extract_merchant(lines: list[str]) -> str:
    """Extract merchant name from OCR lines.

    First non-empty line is typically the merchant name, but handle various
    cases: may be followed by location, may be all-caps, may be garbled by OCR.
    """
    if not lines:
        return "Unknown"

    candidates = [line.strip() for line in lines if line.strip()]
    if not candidates:
        return "Unknown"

    first = candidates[0]
    if first and len(first) > 2:
        return first[:100]

    return "Unknown"


def extract_date(lines: list[str]) -> datetime | None:
    """Extract date from OCR lines.

    Receipts use many date formats:
    - MM/DD/YYYY, MM-DD-YYYY, DD/MM/YYYY
    - Jan 5, 2026 or 5 January 2026
    - 2026-01-05
    - Single word month names with day/year scattered
    """
    date_patterns = [
        r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})",
        r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})",
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})[,\s]+(\d{4})",
        r"(\d{1,2})\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})",
    ]

    full_text = " ".join(lines).lower()

    for pattern in date_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            try:
                groups = match.groups()
                if len(groups) >= 2:
                    if pattern == date_patterns[0]:
                        month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                        if month > 12:
                            month, day = day, month
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            return datetime(year, month, day)
                    elif pattern == date_patterns[1]:
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            return datetime(year, month, day)
                    elif pattern in [date_patterns[2], date_patterns[3]]:
                        for month_num, month_name in enumerate(
                            [
                                "jan",
                                "feb",
                                "mar",
                                "apr",
                                "may",
                                "jun",
                                "jul",
                                "aug",
                                "sep",
                                "oct",
                                "nov",
                                "dec",
                            ],
                            1,
                        ):
                            if month_name in full_text[max(0, match.start() - 10) : match.end() + 10]:
                                if pattern == date_patterns[2]:
                                    day, year = int(groups[0]), int(groups[1])
                                else:
                                    day, year = int(groups[0]), int(groups[1])
                                if 1 <= day <= 31:
                                    return datetime(year, month_num, day)
                        break
            except (ValueError, IndexError):
                continue

    return None


def extract_total(lines: list[str]) -> float | None:
    """Extract total amount from OCR lines.

    Looks for lines containing 'TOTAL', 'Amount', 'Due', followed by
    a currency amount. Handles $X.XX, X.XX, commas in thousands.
    """
    total_keywords = [
        r"total",
        r"amount\s+due",
        r"grand\s+total",
        r"sum",
        r"balance",
    ]

    full_text = " ".join(lines).lower()

    for keyword_pattern in total_keywords:
        matches = list(re.finditer(keyword_pattern, full_text))
        for match in matches:
            start_pos = match.end()
            search_text = " ".join(lines)[start_pos : start_pos + 100]

            amount_pattern = r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\.\d{2})"
            amount_match = re.search(amount_pattern, search_text)
            if amount_match:
                amount_str = amount_match.group(1).replace(",", "")
                try:
                    return float(amount_str)
                except ValueError:
                    continue

    amount_pattern = r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\.\d{2})"
    amounts = re.findall(amount_pattern, " ".join(lines))
    if amounts:
        try:
            last_amount = amounts[-1].replace(",", "")
            return float(last_amount)
        except ValueError:
            pass

    return None


def parse_receipt(raw_text: list[str]) -> Receipt:
    """Parse a receipt from raw OCR text lines."""
    merchant = extract_merchant(raw_text)
    date = extract_date(raw_text)
    total = extract_total(raw_text)

    return Receipt(
        merchant=merchant,
        date=date,
        total=total,
        category="Other",
        raw_text=raw_text,
    )
