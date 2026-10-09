import polars as pl
import pytest
from helpers import raw
from rfm_explorer import data


def run(*rows):
    kept, ledger = data.clean(raw(*rows))
    return kept, {
        r["rule"]: (r["rows"], r["revenue"]) for r in ledger.iter_rows(named=True)
    }


def test_product_codes_with_letter_suffix_are_kept():
    kept, _ = run(
        ("1", "85123A", 2, "2011-01-01", 3.0, 10, "UK"),
        ("1", "22423", 1, "2011-01-01", 5.0, 10, "UK"),
    )
    assert kept.height == 2
    assert kept["revenue"].to_list() == [6.0, 5.0]


@pytest.mark.parametrize(
    "code",
    [
        "POST",
        "M",
        "D",
        "BANK CHARGES",
        "AMAZONFEE",
        "ADJUST",
        "DOT",
        "gift_0001_20",
        "TEST001",
        None,
    ],
)
def test_non_product_codes_are_dropped_and_counted(code):
    kept, ledger = run(("1", code, 1, "2011-01-01", 10.0, 10, "UK"))
    assert kept.is_empty()
    assert ledger["non_product"] == (1, 10.0)


def test_zero_or_negative_price_is_dropped():
    kept, ledger = run(
        ("1", "22423", 1, "2011-01-01", 0.0, 10, "UK"),
        ("2", "22423", 1, "2011-01-01", -5.0, 10, "UK"),
    )
    assert kept.is_empty()
    assert ledger["bad_price"][0] == 2


def test_returns_are_kept_but_inconsistent_signs_are_not():
    kept, ledger = run(
        ("C10", "22423", -2, "2011-01-02", 4.0, 10, "UK"),  # a real return
        (
            "C11",
            "22423",
            2,
            "2011-01-02",
            4.0,
            10,
            "UK",
        ),  # cancellation with positive quantity
        (
            "12",
            "22423",
            -2,
            "2011-01-02",
            4.0,
            10,
            "UK",
        ),  # negative quantity on a normal invoice
    )
    assert kept["invoice"].to_list() == ["C10"]
    assert kept["is_return"].to_list() == [True]
    assert kept["revenue"].to_list() == [-8.0]
    assert ledger["bad_quantity"][0] == 2


def test_guest_rows_are_dropped_last_and_their_money_is_reported():
    kept, ledger = run(
        ("1", "22423", 1, "2011-01-01", 7.0, None, "UK"),  # guest
        (
            "2",
            "POST",
            1,
            "2011-01-01",
            9.0,
            None,
            "UK",
        ),  # guest AND non-product: counted once, as non-product
        ("3", "22423", 1, "2011-01-01", 2.0, 10, "UK"),
    )
    assert kept["customer_id"].to_list() == [10]
    assert ledger["non_product"] == (1, 9.0)
    assert ledger["no_customer"] == (1, 7.0)


def test_every_input_row_is_either_kept_or_in_the_ledger():
    rows = [
        ("1", "22423", 1, "2011-01-01", 7.0, None, "UK"),
        ("2", "POST", 1, "2011-01-01", 9.0, 5, "UK"),
        ("3", "22423", 1, "2011-01-01", 0.0, 5, "UK"),
        ("4", "22423", 3, "2011-01-01", 2.0, 5, "UK"),
        ("C5", "22423", -1, "2011-01-01", 2.0, 5, "UK"),
    ]
    kept, ledger = data.clean(raw(*rows))
    assert kept.height + ledger["rows"].sum() == len(rows)


def test_find_data_file_prefers_the_full_dataset_then_the_sample(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data"):
        data.find_data_file(tmp_path)
    (tmp_path / "sample_data").mkdir()
    sample = tmp_path / data.SAMPLE_FILE
    sample.write_bytes(b"")
    assert data.find_data_file(tmp_path) == sample
    (tmp_path / "data").mkdir()
    full = tmp_path / data.FULL_FILE
    full.write_bytes(b"")
    assert data.find_data_file(tmp_path) == full


def test_bundled_sample_cleans_to_valid_lines():
    kept, ledger = data.clean(
        data.load_raw(data.find_data_file().parent.parent / data.SAMPLE_FILE)
    )
    assert kept.height > 10_000
    assert kept["customer_id"].null_count() == 0
    assert (kept.filter(~pl.col("is_return"))["revenue"] > 0).all()
    assert (kept.filter(pl.col("is_return"))["revenue"] < 0).all()
    assert ledger["rows"].sum() > 0
