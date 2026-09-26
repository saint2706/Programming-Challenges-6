from __future__ import annotations

import data
import numpy as np
import pandas as pd


def _tiny_raw_frame() -> pd.DataFrame:
    """Mimics the real OpenML export's quirks: quote-wrapped categorical
    values, a blank (quoted-space) TotalCharges, and duplicate rows."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "Contract": [
                "Month-to-month",
                "'One year'",
                "Month-to-month",
                "'Two year'",
                "Month-to-month",
                "'One year'",
            ],
            "MonthlyCharges": [29.85, 56.95, 29.85, 70.0, 29.85, 56.95],
            "TotalCharges": ["29.85", "1889.5", "29.85", "' '", "29.85", "1889.5"],
            "Churn": ["No", "No", "No", "Yes", "No", "No"],
        }
    )


def test_load_telco_churn_uses_cache_when_present(tmp_path, monkeypatch):
    cache_path = tmp_path / "telco_churn.csv"
    monkeypatch.setattr(data, "CACHE_PATH", cache_path)
    _tiny_raw_frame().to_csv(cache_path, index=False)

    def fake_fetch(*args, **kwargs):
        raise AssertionError("fetch_openml should not be called when the cache exists")

    monkeypatch.setattr(data, "fetch_openml", fake_fetch)

    df = data.load_telco_churn()
    assert list(df["Churn"]) == ["No", "No", "No", "Yes", "No", "No"]


def test_load_telco_churn_fetches_and_caches_when_missing(tmp_path, monkeypatch):
    cache_path = tmp_path / "telco_churn.csv"
    monkeypatch.setattr(data, "CACHE_PATH", cache_path)
    assert not cache_path.exists()

    class FakeDataset:
        frame = _tiny_raw_frame()

    calls = []

    def fake_fetch(*, data_id, as_frame):
        calls.append((data_id, as_frame))
        return FakeDataset()

    monkeypatch.setattr(data, "fetch_openml", fake_fetch)

    df = data.load_telco_churn()
    assert calls == [(data.DATA_ID, True)]
    assert cache_path.exists()
    assert len(df) == 6


def test_load_telco_churn_force_refresh_refetches_even_with_cache(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "telco_churn.csv"
    monkeypatch.setattr(data, "CACHE_PATH", cache_path)
    _tiny_raw_frame().to_csv(cache_path, index=False)

    class FakeDataset:
        frame = _tiny_raw_frame()

    calls = []
    monkeypatch.setattr(
        data, "fetch_openml", lambda **kwargs: (calls.append(kwargs), FakeDataset())[1]
    )

    data.load_telco_churn(force_refresh=True)
    assert len(calls) == 1


def test_clean_features_strips_quote_wrapping_from_categorical_values():
    cleaned = data.clean_features(_tiny_raw_frame())
    assert set(cleaned["Contract"]) == {"Month-to-month", "One year", "Two year"}
    assert not any(v.startswith("'") for v in cleaned["Contract"])


def test_clean_features_fills_blank_total_charges_with_zero_not_nan():
    cleaned = data.clean_features(_tiny_raw_frame())
    # The one originally-blank ("' '") row is the "'Two year'" row.
    row = cleaned[cleaned["Contract"] == "Two year"].iloc[0]
    assert row["TotalCharges"] == 0.0


def test_clean_features_parses_ordinary_total_charges_as_float():
    cleaned = data.clean_features(_tiny_raw_frame())
    assert cleaned["TotalCharges"].dtype == np.float64
    assert set(cleaned["TotalCharges"]) == {29.85, 1889.5, 0.0}


def test_clean_features_maps_churn_to_binary_int():
    cleaned = data.clean_features(_tiny_raw_frame())
    assert cleaned["Churn"].dtype.kind == "i"
    assert set(cleaned["Churn"].unique()) == {0, 1}


def test_clean_features_drops_exact_duplicate_rows():
    raw = _tiny_raw_frame()
    assert len(raw) == 6  # 3 duplicates of one row, 2 of another, 1 unique
    cleaned = data.clean_features(raw)
    assert len(cleaned) == 3
    assert cleaned.duplicated().sum() == 0


def test_numeric_and_categorical_columns_splits_by_dtype_and_excludes_target():
    cleaned = data.clean_features(_tiny_raw_frame())
    numeric, categorical = data.numeric_and_categorical_columns(cleaned)
    assert "Churn" not in numeric and "Churn" not in categorical
    assert set(numeric) == {"MonthlyCharges", "TotalCharges"}
    assert set(categorical) == {"gender", "Contract"}


def test_numeric_and_categorical_columns_works_without_target_present():
    cleaned = data.clean_features(_tiny_raw_frame()).drop(columns=["Churn"])
    numeric, categorical = data.numeric_and_categorical_columns(cleaned)
    assert set(numeric) == {"MonthlyCharges", "TotalCharges"}
    assert set(categorical) == {"gender", "Contract"}


def _balanced_cleaned_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "MonthlyCharges": rng.uniform(20, 100, size=n),
            "TotalCharges": rng.uniform(0, 5000, size=n),
            "Churn": [1 if i % 3 == 0 else 0 for i in range(n)],  # ~33% positive
        }
    )


def test_train_test_split_frame_excludes_target_from_features():
    df = _balanced_cleaned_frame()
    X_train, X_test, y_train, y_test = data.train_test_split_frame(df, test_size=0.25)
    assert "Churn" not in X_train.columns and "Churn" not in X_test.columns
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_train_test_split_frame_stratifies_on_churn_rate():
    df = _balanced_cleaned_frame()
    overall_rate = df["Churn"].mean()
    _, _, y_train, y_test = data.train_test_split_frame(
        df, test_size=0.25, random_state=0
    )
    # Stratified split keeps the churn rate close to the overall rate in
    # both splits, well within what an unstratified split on this small a
    # dataset could swing by chance.
    assert abs(y_train.mean() - overall_rate) < 0.05
    assert abs(y_test.mean() - overall_rate) < 0.05


def test_train_test_split_frame_is_reproducible_given_same_seed():
    df = _balanced_cleaned_frame()
    split_a = data.train_test_split_frame(df, test_size=0.25, random_state=5)
    split_b = data.train_test_split_frame(df, test_size=0.25, random_state=5)
    pd.testing.assert_frame_equal(split_a[0], split_b[0])
    pd.testing.assert_frame_equal(split_a[1], split_b[1])
