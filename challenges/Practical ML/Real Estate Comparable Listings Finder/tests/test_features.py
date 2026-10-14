from __future__ import annotations

import numpy as np
import polars as pl
from comp_finder import features
from comp_finder.engine import Engine


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def test_projection_distance_matches_haversine(real_sales):
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, real_sales.height, (2, 500))
    lat, lon = real_sales["lat"].to_numpy(), real_sales["long"].to_numpy()
    e, n = features.project(lat, lon)
    flat = np.hypot(e[i] - e[j], n[i] - n[j])
    true = haversine_km(lat[i], lon[i], lat[j], lon[j])
    far = true > 1
    assert np.max(np.abs(flat[far] / true[far] - 1)) < 0.01


def test_build_has_the_declared_columns_and_no_nans(sales):
    x = features.build(sales)
    assert x.shape == (sales.height, len(features.COLUMNS))
    assert np.isfinite(x).all()
    assert set(features.ATTRIBUTES) == set(features.COLUMNS) - set(features.LOCATION)


def test_eff_year_is_the_later_of_built_and_renovated(sales):
    x = features.build(sales)
    col = features.COLUMNS.index("eff_year")
    expected = np.maximum(
        sales["yr_built"].to_numpy(), sales["yr_renovated"].to_numpy()
    )
    assert (x[:, col] == expected).all()


def test_scaler_winsorises_outliers(real_sales):
    x = features.build(real_sales)
    sc = features.Scaler.fit(x)
    bed = features.COLUMNS.index("bedrooms")
    assert x[:, bed].max() == 33
    assert sc.clip(x)[:, bed].max() < 12  # the 33-bedroom typo is pulled back
    z = sc.transform(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9)


def test_scaler_statistics_come_from_the_fit_rows_only(sales):
    n_fit = 1800
    base = Engine(sales, n_fit)
    shifted = sales.with_columns(
        pl.when(pl.int_range(pl.len()) >= n_fit)
        .then(pl.col("sqft_living") * 5)
        .otherwise(pl.col("sqft_living"))
        .alias("sqft_living")
    )
    other = Engine(shifted, n_fit)
    assert np.array_equal(base.scaler.mean, other.scaler.mean)
    assert np.array_equal(base.scaler.std, other.scaler.std)
    assert np.array_equal(base.scaler.hi, other.scaler.hi)
