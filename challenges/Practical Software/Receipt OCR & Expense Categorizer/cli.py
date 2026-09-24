"""Command-line interface for receipt scanning and expense management.

Typer-based CLI with commands to scan receipts, import batches, view logs,
generate reports, and recategorize entries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from categorizer import categorize
from ocr import extract_text
from receipt import parse_receipt
from storage import (
    add_receipt,
    delete_receipt,
    get_category_totals,
    get_monthly_totals,
    get_receipt,
    init_db,
    list_receipts,
    update_category,
)

app = typer.Typer(help="Receipt OCR and expense categorizer")

DEFAULT_DB = Path.home() / ".expense_tracker" / "receipts.db"


@app.command()
def scan(
    image_path: str = typer.Argument(..., help="Path to receipt image file"),
    db: str = typer.Option(str(DEFAULT_DB), help="Path to SQLite database"),
    category_override: str | None = typer.Option(None, "--category", "-c", help="Override auto-detected category"),
) -> None:
    """Scan a receipt image, extract text, categorize, and store."""
    db_path = Path(db)
    init_db(db_path)

    image_path = Path(image_path)
    if not image_path.exists():
        typer.echo(f"Error: Image file not found: {image_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {image_path}...")
    raw_text = extract_text(image_path)

    if not raw_text:
        typer.echo("Error: No text extracted from image", err=True)
        raise typer.Exit(1)

    receipt = parse_receipt(raw_text)
    receipt = receipt.__class__(
        merchant=receipt.merchant,
        date=receipt.date,
        total=receipt.total,
        category=category_override or categorize(receipt.merchant),
        raw_text=receipt.raw_text,
        image_path=str(image_path),
        id=receipt.id,
    )

    add_receipt(db_path, receipt)

    typer.echo(f"Receipt stored: ID={receipt.id}")
    typer.echo(f"  Merchant: {receipt.merchant}")
    typer.echo(f"  Date: {receipt.date}")
    typer.echo(f"  Total: ${receipt.total:.2f}" if receipt.total else "  Total: Unknown")
    typer.echo(f"  Category: {receipt.category}")


@app.command()
def import_folder(
    folder: str = typer.Argument(..., help="Path to folder with receipt images"),
    db: str = typer.Option(str(DEFAULT_DB), help="Path to SQLite database"),
    pattern: str = typer.Option("*.jpg", help="File pattern to match (e.g., *.png)"),
) -> None:
    """Batch-import all images from a folder."""
    db_path = Path(db)
    init_db(db_path)

    folder_path = Path(folder)
    if not folder_path.is_dir():
        typer.echo(f"Error: Folder not found: {folder_path}", err=True)
        raise typer.Exit(1)

    images = sorted(folder_path.glob(pattern))
    if not images:
        typer.echo(f"No images matching '{pattern}' found in {folder_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(images)} images to process")

    success_count = 0
    fail_count = 0

    with typer.progressbar(images, label="Processing") as progress:
        for image_path in progress:
            try:
                raw_text = extract_text(str(image_path))
                if raw_text:
                    receipt = parse_receipt(raw_text)
                    receipt = receipt.__class__(
                        merchant=receipt.merchant,
                        date=receipt.date,
                        total=receipt.total,
                        category=categorize(receipt.merchant),
                        raw_text=receipt.raw_text,
                        image_path=str(image_path),
                        id=receipt.id,
                    )
                    add_receipt(db_path, receipt)
                    success_count += 1
                else:
                    fail_count += 1
                    typer.echo(f"  Failed: {image_path.name} (no text extracted)", err=True)
            except Exception as e:
                fail_count += 1
                typer.echo(f"  Failed: {image_path.name} ({e})", err=True)

    typer.echo(f"Imported: {success_count} receipts, {fail_count} failures")


@app.command()
def list(
    db: str = typer.Option(str(DEFAULT_DB), help="Path to SQLite database"),
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category"),
    month: str | None = typer.Option(None, "--month", "-m", help="Filter by month (YYYY-MM)"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of receipts to show"),
) -> None:
    """List stored receipts."""
    db_path = Path(db)
    if not db_path.exists():
        typer.echo("No receipts found. Use 'scan' or 'import-folder' to add receipts.", err=True)
        raise typer.Exit(1)

    receipts = list_receipts(db_path, category=category, month=month)[:limit]

    if not receipts:
        typer.echo("No receipts match the criteria")
        return

    typer.echo(f"Found {len(receipts)} receipts:\n")

    for r in receipts:
        date_str = r.date.strftime("%Y-%m-%d") if r.date else "Unknown"
        total_str = f"${r.total:.2f}" if r.total else "Unknown"
        typer.echo(f"[{r.id[:8]}...] {date_str} | {r.merchant:30} | {total_str:>10} | {r.category}")


@app.command()
def report(
    db: str = typer.Option(str(DEFAULT_DB), help="Path to SQLite database"),
    month: str | None = typer.Option(None, "--month", "-m", help="Show report for specific month (YYYY-MM)"),
) -> None:
    """Show expense report by category and/or month."""
    db_path = Path(db)
    if not db_path.exists():
        typer.echo("No receipts found.", err=True)
        raise typer.Exit(1)

    if month:
        typer.echo(f"Category breakdown for {month}:\n")
        totals = get_category_totals(db_path, month=month)
    else:
        typer.echo("Category breakdown (all time):\n")
        totals = get_category_totals(db_path)

    if not totals:
        typer.echo("No data available")
        return

    grand_total = sum(totals.values())

    for category, total in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (total / grand_total * 100) if grand_total > 0 else 0
        typer.echo(f"  {category:20} ${total:>10.2f}  ({percentage:>5.1f}%)")

    typer.echo(f"  {'-' * 45}")
    typer.echo(f"  {'Total':20} ${grand_total:>10.2f}")

    if not month:
        typer.echo("\n\nMonthly totals:\n")
        monthly = get_monthly_totals(db_path)
        for month_key, total in sorted(monthly.items(), reverse=True):
            typer.echo(f"  {month_key}  ${total:>10.2f}")


@app.command()
def recategorize(
    receipt_id: str = typer.Argument(..., help="Receipt ID to recategorize"),
    new_category: str = typer.Argument(..., help="New category"),
    db: str = typer.Option(str(DEFAULT_DB), help="Path to SQLite database"),
) -> None:
    """Manually override the category of a receipt."""
    db_path = Path(db)

    receipt = get_receipt(db_path, receipt_id)
    if not receipt:
        typer.echo(f"Error: Receipt not found: {receipt_id}", err=True)
        raise typer.Exit(1)

    if new_category not in [
        "Groceries",
        "Dining",
        "Transport",
        "Utilities",
        "Shopping",
        "Entertainment",
        "Health",
        "Other",
    ]:
        typer.echo(f"Error: Invalid category '{new_category}'", err=True)
        typer.echo(
            "Valid categories: Groceries, Dining, Transport, Utilities, Shopping, Entertainment, Health, Other",
            err=True,
        )
        raise typer.Exit(1)

    update_category(db_path, receipt_id, new_category)
    typer.echo(f"Updated receipt {receipt_id[:8]}... to category '{new_category}'")


@app.command()
def delete(
    receipt_id: str = typer.Argument(..., help="Receipt ID to delete"),
    db: str = typer.Option(str(DEFAULT_DB), help="Path to SQLite database"),
) -> None:
    """Delete a receipt from the database."""
    db_path = Path(db)

    receipt = get_receipt(db_path, receipt_id)
    if not receipt:
        typer.echo(f"Error: Receipt not found: {receipt_id}", err=True)
        raise typer.Exit(1)

    if typer.confirm(f"Delete receipt {receipt_id[:8]}...?"):
        delete_receipt(db_path, receipt_id)
        typer.echo("Receipt deleted")
    else:
        typer.echo("Cancelled")


if __name__ == "__main__":
    app()
