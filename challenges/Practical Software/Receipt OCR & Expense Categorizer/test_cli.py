"""Tests for the CLI interface with mocked OCR."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli import app
from receipt import Receipt
from storage import add_receipt, get_receipt, init_db, list_receipts

runner = CliRunner()


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_receipts.db"
    init_db(db_path)
    return db_path


class TestScanCommand:
    def test_scan_receipt_success(self, tmp_db, tmp_path):
        image_path = tmp_path / "test.jpg"
        image_path.touch()

        with patch("cli.extract_text") as mock_ocr:
            mock_ocr.return_value = [
                "WHOLE FOODS MARKET",
                "123 Main St",
                "Date: 01/15/2026",
                "Apples $3.99",
                "TOTAL: $42.50",
            ]

            result = runner.invoke(app, ["scan", str(image_path), "--db", str(tmp_db)])

        assert result.exit_code == 0
        assert "Receipt stored" in result.stdout

    def test_scan_image_not_found(self, tmp_db):
        result = runner.invoke(app, ["scan", "/nonexistent/image.jpg", "--db", str(tmp_db)])
        assert result.exit_code == 1

    def test_scan_with_category_override(self, tmp_db, tmp_path):
        image_path = tmp_path / "test.jpg"
        image_path.touch()

        with patch("cli.extract_text") as mock_ocr:
            mock_ocr.return_value = ["STORE NAME", "Total: $50.00"]

            result = runner.invoke(
                app,
                ["scan", str(image_path), "--db", str(tmp_db), "--category", "Transport"],
            )

        assert result.exit_code == 0
        assert "Transport" in result.stdout

    def test_scan_no_text_extracted(self, tmp_db, tmp_path):
        image_path = tmp_path / "empty.jpg"
        image_path.touch()

        with patch("cli.extract_text") as mock_ocr:
            mock_ocr.return_value = []

            result = runner.invoke(app, ["scan", str(image_path), "--db", str(tmp_db)])

        assert result.exit_code == 1


class TestImportFolderCommand:
    def test_import_folder_success(self, tmp_db, tmp_path):
        img1 = tmp_path / "receipt1.jpg"
        img2 = tmp_path / "receipt2.jpg"
        img1.touch()
        img2.touch()

        with patch("cli.extract_text") as mock_ocr:
            mock_ocr.return_value = ["STORE", "Total: $20.00"]

            result = runner.invoke(
                app,
                ["import-folder", str(tmp_path), "--db", str(tmp_db), "--pattern", "*.jpg"],
            )

        assert result.exit_code == 0

    def test_import_folder_not_found(self, tmp_db):
        result = runner.invoke(
            app,
            ["import-folder", "/nonexistent/folder", "--db", str(tmp_db)],
        )
        assert result.exit_code == 1

    def test_import_folder_no_matches(self, tmp_db, tmp_path):
        result = runner.invoke(
            app,
            ["import-folder", str(tmp_path), "--db", str(tmp_db), "--pattern", "*.png"],
        )
        assert result.exit_code == 1


class TestListCommand:
    def test_list_receipts(self, tmp_db):
        for i in range(3):
            receipt = Receipt(
                merchant=f"Store {i}",
                category="Groceries",
                date=datetime(2026, 1, i + 1),
                total=float(10 * (i + 1)),
                id=f"test-{i}",
            )
            add_receipt(tmp_db, receipt)

        result = runner.invoke(app, ["list", "--db", str(tmp_db)])
        assert result.exit_code == 0
        assert "Store 0" in result.stdout

    def test_list_filter_by_category(self, tmp_db):
        for category in ["Groceries", "Dining"]:
            receipt = Receipt(
                merchant="Store",
                category=category,
                date=datetime(2026, 1, 1),
                total=10.00,
                id=f"test-{category}",
            )
            add_receipt(tmp_db, receipt)

        result = runner.invoke(app, ["list", "--db", str(tmp_db), "--category", "Groceries"])
        assert result.exit_code == 0
        assert "1 receipts" in result.stdout

    def test_list_filter_by_month(self, tmp_db):
        for month in [1, 2]:
            receipt = Receipt(
                merchant="Store",
                category="Groceries",
                date=datetime(2026, month, 1),
                total=10.00,
                id=f"test-m{month}",
            )
            add_receipt(tmp_db, receipt)

        result = runner.invoke(app, ["list", "--db", str(tmp_db), "--month", "2026-01"])
        assert result.exit_code == 0

    def test_list_no_receipts(self, tmp_db):
        result = runner.invoke(app, ["list", "--db", str(tmp_db)])
        assert result.exit_code == 0 or result.exit_code == 1


class TestReportCommand:
    def test_report_all_time(self, tmp_db):
        for category in ["Groceries", "Dining"]:
            receipt = Receipt(
                merchant="Store",
                category=category,
                date=datetime(2026, 1, 1),
                total=25.00,
                id=f"test-{category}",
            )
            add_receipt(tmp_db, receipt)

        result = runner.invoke(app, ["report", "--db", str(tmp_db)])
        assert result.exit_code == 0
        assert "Groceries" in result.stdout
        assert "Dining" in result.stdout

    def test_report_specific_month(self, tmp_db):
        receipt = Receipt(
            merchant="Store",
            category="Groceries",
            date=datetime(2026, 1, 1),
            total=50.00,
            id="test-1",
        )
        add_receipt(tmp_db, receipt)

        result = runner.invoke(app, ["report", "--db", str(tmp_db), "--month", "2026-01"])
        assert result.exit_code == 0
        assert "2026-01" in result.stdout

    def test_report_no_data(self, tmp_db):
        result = runner.invoke(app, ["report", "--db", str(tmp_db)])
        assert result.exit_code == 0 or result.exit_code == 1


class TestRecategorizeCommand:
    def test_recategorize_receipt(self, tmp_db):
        receipt = Receipt(
            merchant="Store",
            category="Groceries",
            total=10.00,
            id="test-recat",
        )
        add_receipt(tmp_db, receipt)

        result = runner.invoke(
            app,
            ["recategorize", "test-recat", "Dining", "--db", str(tmp_db)],
            input="y\n",
        )
        assert result.exit_code == 0

    def test_recategorize_invalid_category(self, tmp_db):
        receipt = Receipt(
            merchant="Store",
            category="Groceries",
            total=10.00,
            id="test-invalid",
        )
        add_receipt(tmp_db, receipt)

        result = runner.invoke(
            app,
            ["recategorize", "test-invalid", "InvalidCategory", "--db", str(tmp_db)],
        )
        assert result.exit_code == 1

    def test_recategorize_nonexistent_receipt(self, tmp_db):
        result = runner.invoke(
            app,
            ["recategorize", "nonexistent", "Dining", "--db", str(tmp_db)],
        )
        assert result.exit_code == 1


class TestDeleteCommand:
    def test_delete_receipt(self, tmp_db):
        receipt = Receipt(
            merchant="Store",
            category="Groceries",
            total=10.00,
            id="test-delete",
        )
        add_receipt(tmp_db, receipt)

        result = runner.invoke(
            app,
            ["delete", "test-delete", "--db", str(tmp_db)],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "Receipt deleted" in result.stdout

    def test_delete_nonexistent_receipt(self, tmp_db):
        result = runner.invoke(
            app,
            ["delete", "nonexistent", "--db", str(tmp_db)],
        )
        assert result.exit_code == 1

    def test_delete_cancelled(self, tmp_db):
        receipt = Receipt(
            merchant="Store",
            category="Groceries",
            total=10.00,
            id="test-cancel",
        )
        add_receipt(tmp_db, receipt)

        result = runner.invoke(
            app,
            ["delete", "test-cancel", "--db", str(tmp_db)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
