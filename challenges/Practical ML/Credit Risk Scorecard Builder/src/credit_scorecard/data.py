"""Give Me Some Credit: fetch, cache, clean and split.

150,000 borrowers, 6.7% of whom were seriously delinquent (90+ days past due) within two years. The
public copy lives on OpenML as dataset 45577 and is read as parquet with polars, so pandas is never
imported.

Three quirks matter for scorecard building and are handled here or in the binning step:

* The three past-due counters use ``96`` and ``98`` as *codes*, not counts (269 rows, 54% bad). A
  naive bin would merge them with "13+ late payments" and break monotonicity, so they are declared
  special values and get their own bin.
* ``MonthlyIncome`` (20% missing) and ``NumberOfDependents`` (2.6%) are missing, and for borrowers
  with no income ``DebtRatio`` holds an absolute dollar amount, not a ratio. Missingness is itself
  informative, so it is a bin of its own.
* 609 exact duplicate rows and one borrower aged 0. Duplicates are dropped before the split so a row
  cannot sit in both train and test.
"""

from __future__ import annotations

import polars as pl
from sklearn.model_selection import train_test_split

from credit_scorecard.paths import project_root

PARQUET_URL = "https://data.openml.org/datasets/0004/45577/dataset_45577.pq"
CACHE_PATH = project_root() / "data" / "give_me_some_credit.parquet"
TARGET = "SeriousDlqin2yrs"  # 1 = bad (seriously delinquent), 0 = good

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late",
    "NumberRealEstateLoansOrLines": "Real-estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late",
    "NumberOfDependents": "Dependents",
}
# Sentinel codes that stand for "unknown / not a count", per feature.
SPECIAL_VALUES = {
    "NumberOfTime30-59DaysPastDueNotWorse": [96.0, 98.0],
    "NumberOfTimes90DaysLate": [96.0, 98.0],
    "NumberOfTime60-89DaysPastDueNotWorse": [96.0, 98.0],
}


def load_raw(*, force_refresh: bool = False) -> pl.DataFrame:
    """The raw dataset, fetched from OpenML on first use and read from the parquet cache after."""
    if not force_refresh and CACHE_PATH.exists():
        return pl.read_parquet(CACHE_PATH)
    df = pl.read_parquet(PARQUET_URL)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CACHE_PATH)
    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Features as Float64 plus an Int8 ``TARGET`` (1 = bad); duplicates and impossible ages dropped."""
    return (
        df.with_columns(
            pl.col(TARGET).cast(pl.Utf8).cast(pl.Int8),
            *[pl.col(c).cast(pl.Float64) for c in FEATURES],
        )
        .unique(maintain_order=True)
        .filter(pl.col("age") >= 18)
        .select([*FEATURES, TARGET])
    )


def split(
    df: pl.DataFrame, *, test_size: float = 0.3, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stratified train/test split. Rows are random, not out-of-time: the data has no dates."""
    idx_train, idx_test = train_test_split(
        range(df.height),
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET].to_numpy(),
    )
    return df[sorted(idx_train)], df[sorted(idx_test)]
