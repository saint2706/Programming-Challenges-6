"""Tests for SQLite storage operations."""

from datetime import datetime
from pathlib import Path

import pytest

from receipt import Receipt
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


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_receipts.db"
    init_db(db_path)
    return db_path


class TestInitDB:
    def test_creates_database_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        assert db_path.exists()

    def test_creates_receipts_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        assert db_path.exists()


class TestAddReceipt:
    def test_add_single_receipt(self, tmp_db):
        receipt = Receipt(
            merchant="Test Store",
            category="Groceries",
            date=datetime(2026, 1, 15),
            total=42.50,
            raw_text=["Line 1", "Line 2"],
            image_path="/path/to/image.jpg",
            id="test-id-1",
        )

        add_receipt(tmp_db, receipt)

        stored = get_receipt(tmp_db, "test-id-1")
        assert stored is not None
        assert stored.merchant == "Test Store"
        assert stored.total == 42.50

    def test_add_receipt_with_none_fields(self, tmp_db):
        receipt = Receipt(
            merchant="Unknown",
            category="Other",
            date=None,
            total=None,
            id="test-id-2",
        )

        add_receipt(tmp_db, receipt)

        stored = get_receipt(tmp_db, "test-id-2")
        assert stored is not None
        assert stored.date is None
        assert stored.total is None


class TestGetReceipt:
    def test_retrieve_existing_receipt(self, tmp_db):
        receipt = Receipt(
            merchant="Store A",
            category="Dining",
            date=datetime(2026, 1, 10),
            total=25.00,
            id="test-1",
        )

        add_receipt(tmp_db, receipt)
        retrieved = get_receipt(tmp_db, "test-1")

        assert retrieved is not None
        assert retrieved.merchant == "Store A"
        assert retrieved.total == 25.00

    def test_retrieve_nonexistent_receipt(self, tmp_db):
        result = get_receipt(tmp_db, "nonexistent")
        assert result is None


class TestListReceipts:
    def test_list_all_receipts(self, tmp_db):
        for i in range(3):
            receipt = Receipt(
                merchant=f"Store {i}",
                category="Groceries",
                date=datetime(2026, 1, i + 1),
                total=float(10 * (i + 1)),
                id=f"test-{i}",
            )
            add_receipt(tmp_db, receipt)

        all_receipts = list_receipts(tmp_db)
        assert len(all_receipts) == 3

    def test_filter_by_category(self, tmp_db):
        for category in ["Groceries", "Dining", "Transport"]:
            receipt = Receipt(
                merchant=f"Store {category}",
                category=category,
                date=datetime(2026, 1, 1),
                total=10.00,
                id=f"test-{category}",
            )
            add_receipt(tmp_db, receipt)

        groceries = list_receipts(tmp_db, category="Groceries")
        assert len(groceries) == 1
        assert groceries[0].category == "Groceries"

    def test_filter_by_month(self, tmp_db):
        for month in [1, 2, 3]:
            receipt = Receipt(
                merchant="Store",
                category="Groceries",
                date=datetime(2026, month, 1),
                total=10.00,
                id=f"test-m{month}",
            )
            add_receipt(tmp_db, receipt)

        jan_receipts = list_receipts(tmp_db, month="2026-01")
        assert len(jan_receipts) == 1

    def test_filter_by_both_category_and_month(self, tmp_db):
        for month in [1, 2]:
            for category in ["Groceries", "Dining"]:
                receipt = Receipt(
                    merchant="Store",
                    category=category,
                    date=datetime(2026, month, 1),
                    total=10.00,
                    id=f"test-{month}-{category}",
                )
                add_receipt(tmp_db, receipt)

        results = list_receipts(tmp_db, category="Groceries", month="2026-01")
        assert len(results) == 1


class TestUpdateCategory:
    def test_update_receipt_category(self, tmp_db):
        receipt = Receipt(
            merchant="Store",
            category="Other",
            date=None,
            total=10.00,
            id="test-update",
        )

        add_receipt(tmp_db, receipt)
        update_category(tmp_db, "test-update", "Groceries")

        updated = get_receipt(tmp_db, "test-update")
        assert updated.category == "Groceries"


class TestDeleteReceipt:
    def test_delete_existing_receipt(self, tmp_db):
        receipt = Receipt(
            merchant="Store",
            category="Groceries",
            total=10.00,
            id="test-delete",
        )

        add_receipt(tmp_db, receipt)
        assert get_receipt(tmp_db, "test-delete") is not None

        delete_receipt(tmp_db, "test-delete")
        assert get_receipt(tmp_db, "test-delete") is None


class TestGetCategoryTotals:
    def test_sum_by_category(self, tmp_db):
        receipts_data = [
            ("Groceries", datetime(2026, 1, 1), 20.00),
            ("Groceries", datetime(2026, 1, 5), 30.00),
            ("Dining", datetime(2026, 1, 10), 25.00),
        ]

        for i, (category, date, total) in enumerate(receipts_data):
            receipt = Receipt(
                merchant="Store",
                category=category,
                date=date,
                total=total,
                id=f"test-sum-{i}",
            )
            add_receipt(tmp_db, receipt)

        totals = get_category_totals(tmp_db)
        assert totals["Groceries"] == 50.00
        assert totals["Dining"] == 25.00

    def test_filter_by_month(self, tmp_db):
        receipts_data = [
            ("Groceries", datetime(2026, 1, 1), 20.00),
            ("Groceries", datetime(2026, 2, 1), 30.00),
        ]

        for i, (category, date, total) in enumerate(receipts_data):
            receipt = Receipt(
                merchant="Store",
                category=category,
                date=date,
                total=total,
                id=f"test-month-{i}",
            )
            add_receipt(tmp_db, receipt)

        jan_totals = get_category_totals(tmp_db, month="2026-01")
        assert jan_totals["Groceries"] == 20.00


class TestGetMonthlyTotals:
    def test_sum_by_month(self, tmp_db):
        for month in [1, 2, 3]:
            for i in range(2):
                receipt = Receipt(
                    merchant="Store",
                    category="Groceries",
                    date=datetime(2026, month, 1),
                    total=10.00,
                    id=f"test-monthly-{month}-{i}",
                )
                add_receipt(tmp_db, receipt)

        monthly = get_monthly_totals(tmp_db)
        assert monthly["2026-01"] == 20.00
        assert monthly["2026-02"] == 20.00
        assert monthly["2026-03"] == 20.00

    def test_monthly_totals_ordered(self, tmp_db):
        for month in [1, 3, 2]:
            receipt = Receipt(
                merchant="Store",
                category="Groceries",
                date=datetime(2026, month, 1),
                total=10.00,
                id=f"test-order-{month}",
            )
            add_receipt(tmp_db, receipt)

        monthly = get_monthly_totals(tmp_db)
        months_list = list(monthly.keys())
        assert months_list == ["2026-03", "2026-02", "2026-01"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
