from __future__ import annotations

import data
import numpy as np
import pandas as pd


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "SalePrice": [100_000.0, 200_000.0, 300_000.0],
            "LotArea": [1000, np.nan, 3000],
            "Alley": [np.nan, "Grvl", np.nan],  # NONE_MEANS_ABSENT
            "Electrical": ["SBrkr", np.nan, "FuseA"],  # ordinary missing value
        }
    )


def test_load_ames_housing_uses_cache_when_present(tmp_path, monkeypatch):
    cache_path = tmp_path / "ames_housing.csv"
    monkeypatch.setattr(data, "CACHE_PATH", cache_path)
    _tiny_frame().to_csv(cache_path, index=False)

    called = False

    def fake_fetch(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("fetch_openml should not be called when the cache exists")

    monkeypatch.setattr(data, "fetch_openml", fake_fetch)

    df = data.load_ames_housing()
    assert not called
    assert list(df["SalePrice"]) == [100_000.0, 200_000.0, 300_000.0]


def test_load_ames_housing_fetches_and_caches_when_missing(tmp_path, monkeypatch):
    cache_path = tmp_path / "ames_housing.csv"
    monkeypatch.setattr(data, "CACHE_PATH", cache_path)
    assert not cache_path.exists()

    class FakeDataset:
        frame = _tiny_frame()

    calls = []

    def fake_fetch(*, data_id, as_frame):
        calls.append((data_id, as_frame))
        return FakeDataset()

    monkeypatch.setattr(data, "fetch_openml", fake_fetch)

    df = data.load_ames_housing()
    assert calls == [(data.DATA_ID, True)]
    assert cache_path.exists()
    assert list(df["SalePrice"]) == [100_000.0, 200_000.0, 300_000.0]


def test_load_ames_housing_force_refresh_refetches_even_with_cache(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "ames_housing.csv"
    monkeypatch.setattr(data, "CACHE_PATH", cache_path)
    _tiny_frame().to_csv(cache_path, index=False)

    class FakeDataset:
        frame = _tiny_frame()

    calls = []
    monkeypatch.setattr(
        data, "fetch_openml", lambda **kwargs: (calls.append(kwargs), FakeDataset())[1]
    )

    data.load_ames_housing(force_refresh=True)
    assert len(calls) == 1


def test_numeric_and_categorical_columns_splits_by_dtype_and_excludes_target_and_id():
    df = _tiny_frame()
    numeric, categorical = data.numeric_and_categorical_columns(df)
    assert "Id" not in numeric and "Id" not in categorical
    assert "SalePrice" not in numeric and "SalePrice" not in categorical
    assert numeric == ["LotArea"]
    assert set(categorical) == {"Alley", "Electrical"}


def test_numeric_and_categorical_columns_works_without_target_present():
    df = _tiny_frame().drop(columns=["SalePrice"])
    numeric, categorical = data.numeric_and_categorical_columns(df)
    assert numeric == ["LotArea"]
    assert set(categorical) == {"Alley", "Electrical"}


def test_clean_features_drops_id_column():
    cleaned = data.clean_features(_tiny_frame())
    assert "Id" not in cleaned.columns


def test_clean_features_fills_none_means_absent_with_literal_none_string():
    cleaned = data.clean_features(_tiny_frame())
    assert list(cleaned["Alley"]) == ["None", "Grvl", "None"]


def test_clean_features_fills_ordinary_categorical_missing_with_sentinel():
    cleaned = data.clean_features(_tiny_frame())
    assert list(cleaned["Electrical"]) == ["SBrkr", "Missing", "FuseA"]


def test_clean_features_leaves_numeric_nan_untouched():
    cleaned = data.clean_features(_tiny_frame())
    assert cleaned["LotArea"].isna().sum() == 1


def test_clean_features_leaves_no_nan_in_any_categorical_column():
    cleaned = data.clean_features(_tiny_frame())
    _, categorical_cols = data.numeric_and_categorical_columns(cleaned)
    assert cleaned[categorical_cols].isna().sum().sum() == 0


def test_train_test_split_frame_log1p_transforms_target():
    df = data.clean_features(_tiny_frame())
    X_train, X_test, y_train_log, y_test_log = data.train_test_split_frame(
        df, test_size=1 / 3, random_state=0
    )
    all_y_log = pd.concat([y_train_log, y_test_log]).sort_index()
    expected = np.log1p(df["SalePrice"])
    pd.testing.assert_series_equal(all_y_log, expected, check_names=False)
    assert "SalePrice" not in X_train.columns and "SalePrice" not in X_test.columns


def test_train_test_split_frame_is_reproducible_given_same_seed():
    df = data.clean_features(_tiny_frame())
    split_a = data.train_test_split_frame(df, test_size=1 / 3, random_state=5)
    split_b = data.train_test_split_frame(df, test_size=1 / 3, random_state=5)
    pd.testing.assert_frame_equal(split_a[0], split_b[0])
    pd.testing.assert_frame_equal(split_a[1], split_b[1])
