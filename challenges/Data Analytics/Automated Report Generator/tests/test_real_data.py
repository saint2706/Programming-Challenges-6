"""What the engine says about the three real tables: checked against facts about the data itself."""

import polars as pl
import pytest
from helpers import bundled
from report_generator.data import DATASETS, load
from report_generator.engine import build_report


@pytest.fixture(scope="module")
def bike():
    spec, frame = load("bike")
    return build_report(
        frame, title=spec.title, time=spec.time, row_label=spec.row_label
    )


@pytest.fixture(scope="module")
def ames():
    spec, frame = load("ames")
    return build_report(
        frame, title=spec.title, time=spec.time, row_label=spec.row_label
    )


@pytest.fixture(scope="module")
def retail():
    spec, frame = load("retail")
    return build_report(
        frame,
        title=spec.title,
        time=spec.time,
        ignore=spec.ignore,
        row_label=spec.row_label,
    )


def ids(report):
    return {f.id for f in report.candidate_findings}


def test_bundled_tables_load_with_the_documented_shapes():
    assert load("bike")[1].shape == (731, 13)
    assert load("ames")[1].shape == (2930, 17)
    assert load("retail")[1].shape == (87862, 9)  # eight columns plus revenue
    assert set(DATASETS) == {"bike", "retail", "ames"}
    with pytest.raises(FileNotFoundError, match="neither a bundled dataset"):
        load("nope.csv")


def test_bike_overview_and_identity(bike):
    assert bike.overview.text.startswith(
        "The table has 731 days and 13 columns, covering 2011-01-01 to 2012-12-31"
    )
    note = next(f for f in bike.notes if f.id.startswith("quality:identity"))
    df = bundled("bike")
    assert (df["rentals"] == df["casual"] + df["registered"]).all()
    assert note.facts["total"].text == "rentals" and note.facts["n"].value == 731
    # parts of an identity are not reported as "correlated with" their total
    assert not any(
        {"rentals", "casual"} == set(f.columns) for f in bike.candidate_findings
    )


def test_bike_total_rentals_grew_year_on_year(bike):
    f = next(f for f in bike.candidate_findings if f.id == "trend:rentals")
    df = bundled("bike").with_columns(pl.col("date").dt.year().alias("y"))
    totals = (
        df.group_by("y").agg(pl.col("rentals").sum()).sort("y")["rentals"].to_list()
    )
    assert totals[1] / totals[0] > 1.5  # 2012 vs 2011 in the raw data
    assert f.stats["direction"] == "up" and f.facts["change"].value > 0.5


def test_bike_summer_is_busier_than_winter_and_warmth_tracks_rentals(bike):
    assert "season:temp_c" in ids(bike) or "season:feels_like_c" in ids(bike)
    g = next(f for f in bike.candidate_findings if f.id == "group:season:rentals")
    assert (g.facts["hi"].text, g.facts["lo"].text) == ("summer", "winter")
    c = [
        f
        for f in bike.candidate_findings
        if f.type == "correlation" and "rentals" in f.columns
    ]
    assert c and all(f.template == "correlation.pos" for f in c)


def test_ames_sales_are_seasonal_and_neighbourhood_drives_price(ames):
    s = next(f for f in ames.candidate_findings if f.id == "season:__rows__")
    assert s.facts["peak"].text in {"June", "July"} and s.stats["cycles"] >= 4
    g = next(
        f for f in ames.candidate_findings if f.id == "group:Neighborhood:Sale_Price"
    )
    df = bundled("ames")
    med = (
        df.group_by("Neighborhood")
        .agg(pl.col("Sale_Price").median().alias("m"), pl.len().alias("n"))
        .filter(pl.col("n") >= 20)
    )
    assert g.facts["hi"].text == med.sort("m")["Neighborhood"][-1]
    assert g.facts["lo"].text == med.sort("m")["Neighborhood"][0]
    assert any(i.startswith("correlation:Gr_Liv_Area:Sale_Price") for i in ids(ames))


def test_ames_final_month_is_not_mistaken_for_a_collapse(ames):
    assert not any(
        f.type in ("anomaly", "latest") and "__rows__" in f.columns
        for f in ames.candidate_findings
    )
    assert not any(f.type == "shift" for f in ames.candidate_findings)


def test_retail_is_dominated_by_one_country_and_a_few_customers(retail):
    c = next(
        f for f in retail.candidate_findings if f.id == "concentration:country:rows"
    )
    assert c.facts["top"].text == "United Kingdom"
    df = bundled("retail")
    share = (df["country"] == "United Kingdom").mean()
    assert c.facts["share"].value == pytest.approx(share)
    e = next(
        f
        for f in retail.candidate_findings
        if f.id.startswith("concentration:customer_id")
    )
    assert e.facts["share"].value >= 0.6


def test_retail_notes_cover_returns_and_the_seasonal_peak(retail):
    note = next(f for f in retail.notes if f.id == "quality:negative:quantity")
    assert note.facts["n"].value == int((bundled("retail")["quantity"] < 0).sum())
    s = [f for f in retail.candidate_findings if f.type == "seasonality"]
    assert s and all(f.facts["peak"].text in {"October", "November"} for f in s)


def test_the_unfinished_final_week_never_reaches_a_series(retail):
    from report_generator.profile import profile_table
    from report_generator.series import ROWS, build_series

    p = profile_table(
        bundled("retail"), ignore=("invoice", "description"), row_label="invoice lines"
    )
    s = build_series(p, ROWS)
    assert (
        any("incomplete" in d for d in s.dropped) and str(s.dates[-1]) == "2011-11-29"
    )
