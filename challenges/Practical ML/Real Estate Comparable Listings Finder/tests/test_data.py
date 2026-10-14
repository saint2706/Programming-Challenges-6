from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from comp_finder import data
from helpers import synthetic_raw


def test_clean_sorts_by_date_and_counts_prior_sales(sales):
    sold = sales["sold"].to_numpy()
    assert (sold[:-1] <= sold[1:]).all()
    assert sales["n_prior"].max() == 1
    assert (sales["n_prior"] > 0).sum() == 20
    # a property's first sale has n_prior 0 and the second has 1
    twice = sales.filter(pl.col("id").is_duplicated()).sort("id", "sold")
    assert (
        twice.group_by("id")
        .agg(pl.col("n_prior"))
        .select(pl.col("n_prior").list.sort().list.to_array(2))
        .to_series()
        .to_list()
        == [[0, 1]] * 20
    )


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        data.clean(synthetic_raw(50).drop("lat"))


def test_clean_drops_non_positive_prices():
    raw = synthetic_raw(100, resales=0)
    raw = raw.with_columns(
        pl.when(pl.col("id") == raw["id"][0])
        .then(0.0)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    assert data.clean(raw).height == 99


def test_windows_partition_the_table(sales):
    n_fit, n_val, n_test = data.window_counts(
        sales, data.Windows(date(2014, 12, 1), date(2015, 2, 1))
    )
    assert n_fit + n_val + n_test == sales.height
    sold = sales["sold"]
    assert (sold[:n_fit] < date(2014, 12, 1)).all()
    assert (sold[n_fit : n_fit + n_val] < date(2015, 2, 1)).all()
    assert (sold[n_fit + n_val :] >= date(2015, 2, 1)).all()


def test_real_data_shape(real_sales):
    assert real_sales.height == 21_613
    assert str(real_sales["sold"].min()) == "2014-05-02"
    assert str(real_sales["sold"].max()) == "2015-05-27"
    assert real_sales["zipcode"].n_unique() == 70
    assert int((real_sales["n_prior"] > 0).sum()) == 177
    assert real_sales["price"].min() > 0
    n_fit, n_val, n_test = data.window_counts(real_sales, data.Windows())
    assert (n_fit, n_val, n_test) == (14_633, 2_228, 4_752)


def test_subject_frame_matches_the_sales_schema(sales):
    rec = {
        "lat": 47.6,
        "long": -122.3,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 1,
        "grade": 8,
        "condition": 3,
    }
    row = data.subject_frame(rec, date(2015, 6, 1), sales)
    assert row.schema == sales.schema and row.height == 1
    assert np.isnan(row["price"][0]) and row["id"][0] == -1
    assert row["sqft_living15"][0] == 1800  # defaults to the property's own size
    assert row["zipcode"][0] in set(sales["zipcode"])  # nearest sale's zip


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"lat": 40.0}, "outside King County"),
        ({"sqft_living": 0}, "must be positive"),
        ({"colour": "red"}, "unknown fields"),
        ({"grade": None}, "missing fields"),
    ],
)
def test_subject_frame_validates(sales, patch, message):
    rec = {
        "lat": 47.6,
        "long": -122.3,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 1,
        "grade": 8,
        "condition": 3,
    } | patch
    with pytest.raises(ValueError, match=message):
        data.subject_frame(rec, date(2015, 6, 1), sales)
