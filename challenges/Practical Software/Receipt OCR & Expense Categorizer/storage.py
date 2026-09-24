"""SQLite storage for receipt records with CRUD operations."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from receipt import Receipt


def init_db(db_path: str | Path) -> None:
    """Initialize the SQLite database schema."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS receipts (
            id TEXT PRIMARY KEY,
            merchant TEXT NOT NULL,
            date TEXT,
            total REAL,
            category TEXT NOT NULL,
            raw_text TEXT,
            image_path TEXT,
            created_at TEXT NOT NULL
        )
    """
    )

    conn.commit()
    conn.close()


def add_receipt(db_path: str | Path, receipt: Receipt) -> None:
    """Store a parsed receipt in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    raw_text_str = "\n".join(receipt.raw_text) if receipt.raw_text else ""
    date_str = receipt.date.isoformat() if receipt.date else None

    cursor.execute(
        """
        INSERT INTO receipts (id, merchant, date, total, category, raw_text, image_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            receipt.id,
            receipt.merchant,
            date_str,
            receipt.total,
            receipt.category,
            raw_text_str,
            receipt.image_path,
            datetime.now().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def get_receipt(db_path: str | Path, receipt_id: str) -> Receipt | None:
    """Retrieve a receipt by ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM receipts WHERE id = ?", (receipt_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return _row_to_receipt(row)


def list_receipts(
    db_path: str | Path,
    category: str | None = None,
    month: str | None = None,
) -> list[Receipt]:
    """List receipts, optionally filtered by category and/or month.

    month should be in format "YYYY-MM" (e.g., "2026-01").
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM receipts WHERE 1=1"
    params = []

    if category:
        query += " AND category = ?"
        params.append(category)

    if month:
        query += " AND date LIKE ?"
        params.append(f"{month}%")

    query += " ORDER BY date DESC, created_at DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [_row_to_receipt(row) for row in rows]


def update_category(db_path: str | Path, receipt_id: str, new_category: str) -> None:
    """Update the category of an existing receipt."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("UPDATE receipts SET category = ? WHERE id = ?", (new_category, receipt_id))

    conn.commit()
    conn.close()


def delete_receipt(db_path: str | Path, receipt_id: str) -> None:
    """Delete a receipt by ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))

    conn.commit()
    conn.close()


def get_category_totals(db_path: str | Path, month: str | None = None) -> dict[str, float]:
    """Get total spending by category, optionally filtered by month.

    Returns a dict mapping category names to total amounts.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT category, SUM(total) FROM receipts WHERE total IS NOT NULL"
    params = []

    if month:
        query += " AND date LIKE ?"
        params.append(f"{month}%")

    query += " GROUP BY category ORDER BY SUM(total) DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return {category: total for category, total in rows}


def get_monthly_totals(db_path: str | Path) -> dict[str, float]:
    """Get total spending by month (YYYY-MM format).

    Returns a dict mapping month strings to total amounts, ordered by month descending.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT strftime('%Y-%m', date) as month, SUM(total)
        FROM receipts
        WHERE date IS NOT NULL AND total IS NOT NULL
        GROUP BY month
        ORDER BY month DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    return {month: total for month, total in rows if month}


def _row_to_receipt(row: tuple) -> Receipt:
    """Convert a database row to a Receipt object."""
    id_, merchant, date_str, total, category, raw_text_str, image_path, created_at = row

    date = None
    if date_str:
        try:
            date = datetime.fromisoformat(date_str)
        except ValueError:
            pass

    raw_text = raw_text_str.split("\n") if raw_text_str else []

    return Receipt(
        merchant=merchant,
        date=date,
        total=total,
        category=category,
        raw_text=raw_text,
        image_path=image_path,
        id=id_,
    )
