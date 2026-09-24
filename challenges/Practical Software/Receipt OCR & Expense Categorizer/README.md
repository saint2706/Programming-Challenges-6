# Receipt OCR & Expense Categorizer

**Category:** Practical Software
**Difficulty:** Intermediate (brief: "Pretrained OCR + rule/ML categorization by merchant/keywords.")

**Status:** Implemented (Python)

A receipt scanner that extracts text from receipt images using PaddleOCR,
parses structured fields (merchant, date, total), automatically categorizes
expenses by merchant name, and stores everything in SQLite for reporting
and analysis. No cloud APIs, no per-image fees — all processing happens
locally.

## The actually-hard part: OCR text is messy and receipts are anarchic

The brief sounds like paddleocr.ocr(image) → structured data, but receipts
are the opposite of structured:

- **Wildly different layouts.** A grocery store receipt is 20 lines long and
  narrow. A restaurant receipt is a crumpled thermal paper with fading text.
  A gas station slip is barely readable. No two receipts follow the same format.
- **OCR hallucinations and character confusion.** PaddleOCR reads a capital I
  as a 1, or an S as a 5. A smudged $ becomes a different character. A
  folded receipt has entire lines missing. The extracted text is a suggestion,
  not ground truth.
- **Inconsistent date and number formats.** Dates are 01/15/2026, 01-15-2026,
  Jan 15, 2026, 2026-01-15, or "15 January 2026" — sometimes with the
  month spelled out, sometimes abbreviated, sometimes in a locale not English.
  Totals are $42.50, 42.50, 42,50, or split across lines.
- **Merchant names are unreliable OCR targets.** The business name might be
  a logo that OCR entirely skips. The first line might be a location or
  receipt number, not the merchant. A chain store may print as "STORE #123"
  or show only the franchise name, not the chain.

The parsing strategy is deliberately defensive and heuristic:

- **Merchant extraction:** Use the first non-empty line as a heuristic, with
  graceful fallback to "Unknown" if it's too short or missing.
- **Date extraction:** Regex patterns for each common format (MM/DD/YYYY,
  YYYY-MM-DD, month names, etc.), with month/day swap logic to handle
  ambiguity (is 01/15/2026 January 15 or an invalid format?) and fallback
  to None if parsing fails.
- **Total extraction:** Search for keywords ("TOTAL", "Amount Due", "Balance"),
  then scan forward for the first currency amount, with a fallback to the
  last amount in the receipt (often the total).
- **Categorization:** Two-stage: keyword rules first (fast, exact match), then
  fuzzy matching against a merchant database (handles typos and OCR errors).

This is why eceipt.py is tested extensively with fixture text (not live OCR)
and why the OCR layer (ocr.py) is deliberately thin and swappable — the hard
part is making parsing robust to garbage input, not the OCR itself.

## Design

- **ocr.py** — Thin wrapper around PaddleOCR. Lazily initializes the model
  on first use and reuses it. Returns a list of extracted text lines. This is
  the only file allowed to import paddleocr, so the rest of the system is
  testable without OCR model downloads during normal test runs.
- **eceipt.py** — Receipt dataclass (merchant, date, total, category,
  raw_text, image_path, id) and pure parsing functions (xtract_merchant,
  xtract_date, xtract_total, and the convenience parse_receipt) that
  operate on OCR text via regex and heuristics. Extensively tested with fixture
  OCR output (garbage, typos, format variations) rather than live images.
- **categorizer.py** — Keyword-rule table mapping merchant keywords to
  categories (Groceries, Dining, Transport, Utilities, Shopping, Entertainment,
  Health, Other), plus apidfuzz-based fuzzy merchant matching (using the
  uzz.ratio scorer for whole-string matching, not partial). Two-stage
  categorization: keyword rules first (fast), then fuzzy match against known
  merchants, then "Other".
- **storage.py** — SQLite CRUD (insert, retrieve, update, delete, list,
  filter by category/month, category/monthly totals). Schema is simple
  (one receipts table), and operations are row-oriented, not normalized.
- **cli.py** — Typer app with commands: scan <image> (OCR one receipt),
  import-folder <dir> (batch OCR all images in a folder), list (show
  receipts with filters), eport (category breakdown and monthly totals),
  ecategorize <id> <category> (manual override), delete <id> (remove a
  receipt). Default SQLite path is ~/.expense_tracker/receipts.db.

## Usage

`ash
cd "challenges/Practical Software/Receipt OCR & Expense Categorizer"

# Scan a single receipt
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py scan /path/to/receipt.jpg

# Batch-import all JPGs from a folder
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py import-folder /path/to/receipt/folder

# List all receipts
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py list

# Filter receipts by category
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py list --category Groceries

# Filter by month (YYYY-MM)
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py list --month 2026-01

# Show expense report
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py report

# Report for a specific month
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py report --month 2026-01

# Manually recategorize a receipt
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py recategorize <receipt-id> Transport

# Delete a receipt
uv run --with paddleocr --with paddlepaddle --with typer --with pillow \
    python cli.py delete <receipt-id>

# Run all tests (no PaddleOCR downloads — mocked)
uv run --with paddleocr --with paddlepaddle --with typer --with rapidfuzz \
    --with pillow --with pytest pytest -q      # 84 tests
`

**PaddleOCR Installation Note:** On Windows with Python 3.12, both paddleocr
and paddlepaddle must be installed. The first scan will download the OCR model
weights (~500MB) — this is one-time and takes a few minutes. Subsequent scans
are fast. If model download fails in your environment, install failures will
be clear at import time, and you can then swap ocr.py's implementation
(the interface is minimal and intentionally swappable).

## Supported categories

- **Groceries** — supermarkets, grocery stores, meal kits
- **Dining** — restaurants, cafes, food delivery apps
- **Transport** — rideshare (Uber, Lyft), gas stations, parking, airlines
- **Utilities** — internet, phone, electricity, water
- **Shopping** — clothing, general retail, online marketplaces
- **Entertainment** — streaming services, movies, concerts, gaming
- **Health** — pharmacies, gyms, medical, dental
- **Other** — everything else

## Tests

84 pytest cases across four files.

	est_receipt.py (13 cases) covers extraction functions directly with fixture
OCR text: multiple date formats (MM/DD/YYYY, month names, different separators,
swapped month/day detection), total extraction with keywords ("TOTAL", "Amount
Due"), currency formats (dollar sign, commas, decimal), and graceful failures
(missing fields → None, not crashes).

	est_categorizer.py (22 cases) tests keyword rules for each category, case
insensitivity, partial merchant matching, fuzzy matching with typo tolerance,
and the "Other" fallback.

	est_storage.py (21 cases) covers CRUD round-trips against a temporary SQLite
database, filtering by category and month, updating categories, category/monthly
totals, and JSON serialization of receipt raw_text.

	est_cli.py (28 cases) drives the Typer CLI via CliRunner with mocked OCR
calls (no real PaddleOCR model downloads): scan success/failure, import-folder
batch processing, list with filters, report generation, recategorization,
deletion, and error cases.

All tests run without downloading the PaddleOCR model — OCR calls are mocked
in CLI and storage tests. The only time the model downloads is during a real
scan or import-folder command.

The suite runs warning-free. pytest.ini is empty (no custom config needed).
