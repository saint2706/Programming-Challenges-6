"""A linear hedonic price model: appraisal-style adjustments and the market-time index.

``log(price) ~ attributes + location controls + zip dummies + month dummies``, ridge-solved on the
fit window. One model, two jobs:

* the attribute coefficients are the per-feature adjustment an appraiser would put on the grid
  (a comp with 200 fewer square feet is worth ``beta * delta`` less than the subject);
* the month coefficients are the market index that rolls a comp's price forward to the subject's
  sale month. Months after the fit window are extrapolated (``flat`` or the ``trend`` of the last
  six fitted months), because at appraisal time the target month has not been observed yet.

Location is a control, never an adjustment: a comp that is far away is a worse comp, not a comp that
needs a dollar figure for its address.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comp_finder.features import ATTR_IDX

EXTRAPOLATIONS = ("flat", "trend")
TREND_MONTHS = 6


@dataclass
class LinearHedonic:
    beta: np.ndarray  # per z-unit effect on log price, aligned with features.COLUMNS
    zips: list[str]
    zip_effect: np.ndarray
    months: np.ndarray  # fitted months (year*12 + month), ascending
    month_effect: np.ndarray  # log index per fitted month, first month = 0
    intercept: float

    @classmethod
    def fit(
        cls,
        z: np.ndarray,
        zipcode: np.ndarray,
        month: np.ndarray,
        log_price: np.ndarray,
    ) -> LinearHedonic:
        zips = sorted(set(zipcode.tolist()))
        months = np.unique(month)
        zip_cols = (zipcode[:, None] == np.array(zips)[None, :]).astype(np.float64)
        month_cols = (month[:, None] == months[None, 1:]).astype(
            np.float64
        )  # first month = base
        a = np.column_stack([np.ones(len(z)), z, zip_cols, month_cols])
        penalty = np.full(a.shape[1], 1e-3)
        penalty[0] = 0.0
        coef = np.linalg.solve(a.T @ a + np.diag(penalty), a.T @ log_price)
        d = z.shape[1]
        return cls(
            beta=coef[1 : 1 + d],
            zips=zips,
            zip_effect=coef[1 + d : 1 + d + len(zips)],
            months=months,
            month_effect=np.concatenate([[0.0], coef[1 + d + len(zips) :]]),
            intercept=float(coef[0]),
        )

    def index(self, month: np.ndarray, extrapolate: str = "flat") -> np.ndarray:
        """Log market index at each month; beyond the fitted months it is extrapolated."""
        if extrapolate not in EXTRAPOLATIONS:
            raise ValueError(f"extrapolate must be one of {EXTRAPOLATIONS}")
        month = np.asarray(month)
        out = np.interp(month, self.months, self.month_effect)  # holds the ends flat
        if extrapolate == "trend" and len(self.months) >= 3:
            tail = slice(-TREND_MONTHS, None)
            slope = np.polyfit(self.months[tail], self.month_effect[tail], 1)[0]
            ahead = np.maximum(month - self.months[-1], 0)
            out = out + slope * ahead
        return out

    def location_effect(self, zipcode: np.ndarray) -> np.ndarray:
        table = dict(zip(self.zips, self.zip_effect, strict=True))
        return np.array([table.get(z, 0.0) for z in zipcode.tolist()])

    def predict_log(
        self,
        z: np.ndarray,
        zipcode: np.ndarray,
        month: np.ndarray,
        extrapolate: str = "flat",
    ) -> np.ndarray:
        return (
            self.intercept
            + z @ self.beta
            + self.location_effect(zipcode)
            + self.index(month, extrapolate)
        )

    def adjustments(self, z_subject: np.ndarray, z_comp: np.ndarray) -> np.ndarray:
        """Per-attribute log adjustment moving a comp to the subject, ``(..., n_attributes)``."""
        return (z_subject - z_comp)[..., ATTR_IDX] * self.beta[ATTR_IDX]

    def ridge_weights(self) -> np.ndarray:
        """|beta| per feature: the (poor for location) linear take on feature importance."""
        return np.abs(self.beta)
