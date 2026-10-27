"""The bundled datasets and the one loader every entry point uses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hypothesis_wizard.paths import project_root


@dataclass(frozen=True)
class Dataset:
    key: str
    file: str
    title: str
    source: str
    license: str
    note: str


DATASETS: dict[str, Dataset] = {
    d.key: d
    for d in [
        Dataset(
            "penguins",
            "penguins.csv",
            "Palmer penguins (344 birds, 3 species)",
            "Gorman, Williams & Fraser (2014), via allisonhorst/palmerpenguins",
            "CC0 1.0",
            "Rows with NA kept in the file; NA is dropped per analysis.",
        ),
        Dataset(
            "iris",
            "iris.csv",
            "Fisher's iris (150 flowers, 3 species)",
            "Fisher (1936); UCI Machine Learning Repository, dataset 53",
            "CC BY 4.0",
            "Petal length has very unequal spread across species.",
        ),
        Dataset(
            "wine_red",
            "wine_red.csv",
            "Red wine quality (1,599 wines, quality rated 3 to 8)",
            "Cortez et al. (2009); UCI Machine Learning Repository, dataset 186",
            "CC BY 4.0",
            "Quality is an ordinal rating; residual sugar has a long right tail.",
        ),
        Dataset(
            "sleep",
            "sleep.csv",
            "Hours of sleep gained on two drugs (10 patients, paired)",
            "Student (1908), R datasets::sleep",
            "Published historical table (R's datasets package)",
            "Each patient ID appears in both groups: a paired design.",
        ),
        Dataset(
            "toothgrowth",
            "toothgrowth.csv",
            "Tooth growth in guinea pigs (60 animals, supplement x dose)",
            "Crampton (1947), R datasets::ToothGrowth",
            "Published historical table (R's datasets package)",
            "Small groups of 10; dose has three levels.",
        ),
        Dataset(
            "ucb_admissions",
            "ucb_admissions.csv",
            "UC Berkeley graduate admissions 1973 (4,526 applicants, 6 departments)",
            "Bickel, Hammel & O'Connell (1975), R datasets::UCBAdmissions",
            "Published historical table (R's datasets package)",
            "Stored as counts (column Freq); expand with expand_counts before use.",
        ),
    ]
}


def data_dir() -> Path:
    return project_root() / "sample_data"


def load(key: str) -> pl.DataFrame:
    """A bundled dataset by key (see ``DATASETS``)."""
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose one of {', '.join(DATASETS)}")
    return read_table(data_dir() / DATASETS[key].file)


def read_table(path: Path) -> pl.DataFrame:
    """A CSV with ``NA`` and blanks as nulls and numeric columns typed as numbers."""
    return pl.read_csv(path, null_values=["NA", ""], infer_schema_length=10_000)


def expand_counts(df: pl.DataFrame, count_col: str) -> pl.DataFrame:
    """One row per counted case: a table with a ``Freq`` column becomes individual rows."""
    rows: list[dict] = []
    for record in df.to_dicts():
        count = int(record.pop(count_col))
        rows.extend(dict(record) for _ in range(count))
    return pl.DataFrame(rows)
