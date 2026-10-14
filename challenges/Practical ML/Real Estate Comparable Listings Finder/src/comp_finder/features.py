"""Engineered property features and their scaling.

Fifteen columns describe a property: where it is (km east/north of downtown Seattle, so Euclidean
distance is kilometres), how big, how good, how old, and what its neighbours look like. Two
choices worth knowing:

* Winsorising and standardising statistics come from the fit window only (``Scaler.fit``), so a
  later sale never moves the scale an earlier one is measured on.
* ``eff_year`` is the later of year built and year renovated, so age does not depend on the sale
  date and two comps of different vintage compare on the same footing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

DOWNTOWN = (47.6062, -122.3321)  # lat, long
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * float(np.cos(np.radians(DOWNTOWN[0])))

COLUMNS = [
    "east_km",
    "north_km",
    "log_living",
    "log_lot",
    "bedrooms",
    "bathrooms",
    "floors",
    "has_basement",
    "grade",
    "condition",
    "view",
    "waterfront",
    "eff_year",
    "log_living15",
    "log_lot15",
]
LABELS = {
    "east_km": "East of downtown (km)",
    "north_km": "North of downtown (km)",
    "log_living": "Living area",
    "log_lot": "Lot size",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "floors": "Floors",
    "has_basement": "Basement",
    "grade": "Build grade",
    "condition": "Condition",
    "view": "View",
    "waterfront": "Waterfront",
    "eff_year": "Year built/renovated",
    "log_living15": "Neighbour living area",
    "log_lot15": "Neighbour lot size",
}
LOCATION = ["east_km", "north_km"]
ATTRIBUTES = [c for c in COLUMNS if c not in LOCATION]  # what an appraiser adjusts for
ATTR_IDX = [COLUMNS.index(c) for c in ATTRIBUTES]
RAW_INPUTS = [
    "lat",
    "long",
    "sqft_living",
    "sqft_lot",
    "bedrooms",
    "bathrooms",
    "floors",
    "sqft_basement",
    "grade",
    "condition",
    "view",
    "waterfront",
    "yr_built",
    "yr_renovated",
    "sqft_living15",
    "sqft_lot15",
]


def project(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kilometres east and north of downtown (local equirectangular, fine within ~80 km)."""
    return (lon - DOWNTOWN[1]) * KM_PER_DEG_LON, (lat - DOWNTOWN[0]) * KM_PER_DEG_LAT


def build(df: pl.DataFrame) -> np.ndarray:
    """The raw (unclipped, unscaled) engineered matrix, one row per sale."""
    g = {c: df[c].to_numpy().astype(np.float64) for c in RAW_INPUTS}
    east, north = project(g["lat"], g["long"])
    cols = {
        "east_km": east,
        "north_km": north,
        "log_living": np.log(g["sqft_living"]),
        "log_lot": np.log(g["sqft_lot"]),
        "bedrooms": g["bedrooms"],
        "bathrooms": g["bathrooms"],
        "floors": g["floors"],
        "has_basement": (g["sqft_basement"] > 0).astype(np.float64),
        "grade": g["grade"],
        "condition": g["condition"],
        "view": g["view"],
        "waterfront": g["waterfront"],
        "eff_year": np.maximum(g["yr_built"], g["yr_renovated"]),
        "log_living15": np.log(g["sqft_living15"]),
        "log_lot15": np.log(g["sqft_lot15"]),
    }
    return np.column_stack([cols[c] for c in COLUMNS])


@dataclass
class Scaler:
    """Winsorise at the 0.1/99.9 percentiles of the fit rows, then z-score."""

    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        lo, hi = np.percentile(x, [0.1, 99.9], axis=0)
        clipped = np.clip(x, lo, hi)
        std = clipped.std(axis=0)
        return cls(lo, hi, clipped.mean(axis=0), np.where(std > 0, std, 1.0))

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lo, self.hi)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (self.clip(x) - self.mean) / self.std
