import numpy as np
import polars as pl
import pytest

from data import CATEGORICAL, FEATURES, NUMERIC, load, split


def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    cols = {"day": rng.integers(1, 8, n)}
    for name in NUMERIC:
        cols[name] = rng.random(n)
    cols["label"] = (rng.random(n) < 0.4).astype(int)
    return pl.DataFrame(cols).with_row_index("row")


def test_feature_lists_match_the_spec():
    assert FEATURES == [
        "day",
        "period",
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer",
    ]
    assert NUMERIC == FEATURES[1:] and CATEGORICAL == ["day"]


def test_split_is_contiguous_ordered_disjoint_and_close_to_the_fractions():
    df = _frame(1001)
    train, ref, live = split(df)
    assert train.height + ref.height + live.height == df.height
    assert abs(train.height - 0.3 * 1001) <= 1 and abs(ref.height - 0.1 * 1001) <= 1
    rows = [x["row"].to_list() for x in (train, ref, live)]
    assert rows[0] + rows[1] + rows[2] == list(range(1001))  # contiguous, in time order


def test_split_is_deterministic_and_validates_fractions():
    df = _frame(500)
    a, b = split(df), split(df)
    assert all(x.equals(y) for x, y in zip(a, b, strict=True))
    with pytest.raises(ValueError, match="sum to 1"):
        split(df, fracs=(0.5, 0.5, 0.5))


def test_load_returns_features_and_a_binary_label(tmp_path):
    p = tmp_path / "e.parquet"
    _frame(50).drop("row").write_parquet(p)
    df = load(p)
    assert df.columns == [*FEATURES, "label"]
    assert set(df["label"].unique().to_list()) <= {0, 1}
