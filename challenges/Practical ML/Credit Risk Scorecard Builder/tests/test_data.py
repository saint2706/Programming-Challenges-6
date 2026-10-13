from __future__ import annotations

import polars as pl
from credit_scorecard import data


def _raw_row(age: int = 40, bad: str = "0") -> dict:
    row = {f: 1.0 for f in data.FEATURES}
    row["age"] = age
    row[data.TARGET] = bad
    return row


def test_clean_drops_duplicates_and_impossible_ages_and_returns_float_features():
    raw = pl.DataFrame(
        [_raw_row(), _raw_row(), _raw_row(age=0), _raw_row(age=55, bad="1")]
    )
    cleaned = data.clean(raw)
    assert cleaned.height == 2
    assert cleaned["age"].to_list() == [40.0, 55.0]
    assert cleaned[data.TARGET].dtype == pl.Int8 and cleaned[data.TARGET].to_list() == [
        0,
        1,
    ]
    assert all(cleaned[f].dtype == pl.Float64 for f in data.FEATURES)


def test_real_data_cleans_to_the_expected_shape():
    cleaned = data.clean(data.load_raw())
    assert cleaned.columns == [*data.FEATURES, data.TARGET]
    assert cleaned.height == 149_390
    assert cleaned.is_duplicated().sum() == 0
    assert cleaned["age"].min() >= 18
    assert set(cleaned[data.TARGET].unique().to_list()) == {0, 1}
    assert 0.066 < cleaned[data.TARGET].mean() < 0.068


def test_sentinel_codes_survive_cleaning_for_the_binner_to_handle():
    cleaned = data.clean(data.load_raw())
    for feature, codes in data.SPECIAL_VALUES.items():
        assert cleaned.filter(pl.col(feature).is_in(codes)).height > 100


def test_split_is_stratified_deterministic_and_disjoint():
    cleaned = data.clean(data.load_raw())
    train, test = data.split(cleaned)
    again_train, _ = data.split(cleaned)
    assert train.height + test.height == cleaned.height
    assert abs(train[data.TARGET].mean() - test[data.TARGET].mean()) < 0.001
    assert train.equals(again_train)
    assert train.vstack(test).unique().height == cleaned.height  # no row on both sides
