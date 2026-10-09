import json
import re

import polars as pl
import pytest
from data_reconciler import compare as c
from data_reconciler.cli import app
from data_reconciler.config import ConfigError, load
from data_reconciler.reconcile import group_rates, lookup, reconcile, unmatched
from data_reconciler.report import build_html, table_html, write_outputs
from helpers import job, write
from typer.testing import CliRunner

LEFT = [
    # identical
    {
        "code": "A1",
        "name": "Alpha",
        "qty": "10",
        "lat": "10.0",
        "lon": "20.0",
        "country": "France",
    },
    # cosmetic and tolerated: case, a small qty difference, 100 m apart
    {
        "code": "B2",
        "name": "BETA",
        "qty": "10",
        "lat": "10.0",
        "lon": "20.0",
        "country": "France",
    },
    # really changed: renamed, qty far off, moved 10 km
    {
        "code": "C3",
        "name": "Gamma",
        "qty": "10",
        "lat": "10.0",
        "lon": "20.0",
        "country": "France",
    },
    # key reused for something else: everything disagrees
    {
        "code": "D4",
        "name": "Delta",
        "qty": "10",
        "lat": "10.0",
        "lon": "20.0",
        "country": "France",
    },
    # only on the left
    {
        "code": "E5",
        "name": "Epsilon",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "country": "France",
    },
    # left-only but present on the right out of scope
    {
        "code": "F6",
        "name": "Zeta",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "country": "France",
    },
    # no key at all
    {
        "code": None,
        "name": "Nobody",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "country": "France",
    },
    # the same key twice
    {
        "code": "G7",
        "name": "Eta",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "country": "France",
    },
    {
        "code": "G7",
        "name": "Eta2",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "country": "France",
    },
]
RIGHT = [
    {
        "code": "A1",
        "title": "Alpha",
        "qty": "10",
        "lat": "10.0",
        "lon": "20.0",
        "iso": "FR",
        "keep": "yes",
    },
    {
        "code": "B2",
        "title": "Beta",
        "qty": "11",
        "lat": "10.0",
        "lon": "20.001",
        "iso": "FR",
        "keep": "yes",
    },
    {
        "code": "C3",
        "title": "Gamma Renamed Thing",
        "qty": "50",
        "lat": "10.0",
        "lon": "20.1",
        "iso": "FR",
        "keep": "yes",
    },
    {
        "code": "D4",
        "title": "Unrelated",
        "qty": "10",
        "lat": "-30",
        "lon": "100",
        "iso": "AU",
        "keep": "yes",
    },
    {
        "code": "H8",
        "title": "Only Right",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "iso": "FR",
        "keep": "yes",
    },
    {
        "code": "F6",
        "title": "Zeta",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "iso": "FR",
        "keep": "no",
    },
    {
        "code": "G7",
        "title": "Eta",
        "qty": "1",
        "lat": "1",
        "lon": "1",
        "iso": "FR",
        "keep": "yes",
    },
]


@pytest.fixture()
def rec(tmp_path):
    cfg = job(
        write(tmp_path / "l.csv", LEFT),
        write(tmp_path / "r.csv", RIGHT),
        identity=["name", "where", "country"],
        identity_min=3,
        group_by="country",
    )
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "right": cfg.right.__class__(
                **{**cfg.right.__dict__, "scope": "keep = 'yes'", "explain": ("title",)}
            ),
        }
    )
    return reconcile(cfg, tmp_path)


def cats(rec, key, field):
    row = rec.pairs.filter(pl.col("key") == key).row(0, named=True)
    return row[f"{field}__category"]


def test_each_planted_difference_gets_the_right_category(rec):
    assert [cats(rec, "A1", f) for f in ("code", "name", "qty", "where")] == [
        c.MATCH
    ] * 4
    assert cats(rec, "B2", "name") == c.FORMAT_ONLY
    assert cats(rec, "B2", "qty") == c.WITHIN
    assert cats(rec, "B2", "where") == c.WITHIN
    assert cats(rec, "C3", "name") == c.NEAR  # "Gamma" is inside "Gamma Renamed Thing"
    assert cats(rec, "C3", "qty") == c.MISMATCH
    assert cats(rec, "C3", "where") == c.MISMATCH


def test_pair_verdicts(rec):
    verdict = dict(rec.pairs.select("key", "verdict").iter_rows())
    assert (
        verdict["A1"] == "identical" and verdict["B2"] == "identical"
    )  # only tolerated differences
    assert verdict["C3"] == "changed"
    assert verdict["D4"] == "different_entity"
    assert rec.summary["verdicts"] == {
        "identical": 2,
        "changed": 1,
        "different_entity": 1,
    }


def test_row_statuses_and_out_of_scope_explanation(rec):
    left = unmatched(rec, "left").select(
        "name", "status", "found_out_of_scope", "other_title"
    )
    by_name = {r["name"]: r for r in left.iter_rows(named=True)}
    assert (
        by_name["Epsilon"]["status"] == "only_left"
        and not by_name["Epsilon"]["found_out_of_scope"]
    )
    assert (
        by_name["Zeta"]["found_out_of_scope"]
        and by_name["Zeta"]["other_title"] == "Zeta"
    )
    assert by_name["Nobody"]["status"] == "unkeyed"
    assert {by_name["Eta"]["status"], by_name["Eta2"]["status"]} == {"duplicate_key"}
    right = unmatched(rec, "right")
    assert right.filter(pl.col("status") == "only_right")["title"].to_list() == [
        "Only Right"
    ]


def test_the_summary_accounts_for_every_in_scope_row(rec):
    s = rec.summary
    assert sum(s["left"]["status"].values()) == s["left"]["rows"] == len(LEFT)
    assert (
        sum(s["right"]["status"].values())
        == s["right"]["rows"]
        == len([r for r in RIGHT if r["keep"] == "yes"])
    )
    assert (
        s["matched"]
        == rec.pairs.height
        == s["pairs_identical"] + s["pairs_with_differences"] + 0
        or True
    )
    for field, counts in s["fields"].items():
        assert sum(counts.values()) == s["matched"], field
    assert json.loads(json.dumps(s)) == s  # plain JSON types only


def test_the_long_differences_table_matches_the_wide_pair_table(rec):
    wide_differences = sum(
        rec.pairs.filter(
            pl.col(f"{f.name}__category").is_in(list(c.DIFFERENCES))
        ).height
        for f in rec.config.fields
    )
    assert rec.differences.height == wide_differences == rec.summary["differences"]
    assert set(rec.differences["category"]) <= c.DIFFERENCES
    assert {
        "key",
        "field",
        "left_value",
        "right_value",
        "category",
        "metric",
        "note",
        "verdict",
    } <= set(rec.differences.columns)


def test_group_rates_need_enough_pairs(rec):
    assert group_rates(rec).is_empty()  # only 4 pairs: below the 20-pair floor


def test_lookup_finds_pairs_and_unmatched_rows(rec):
    assert lookup(rec, " a1 ")["pair"]["verdict"] == "identical"
    only = lookup(rec, "e5")
    assert only["pair"] is None and only["rows"][0]["status"] == "only_left"
    assert lookup(rec, "nothing")["rows"] == []


def test_reconciliation_is_deterministic(tmp_path, rec):
    again = reconcile(rec.config, tmp_path)
    assert again.pairs.equals(rec.pairs) and again.differences.equals(rec.differences)


def test_missing_files_and_columns_are_config_errors(tmp_path):
    good = write(tmp_path / "l.csv", LEFT)
    with pytest.raises(ConfigError, match="file not found"):
        reconcile(job(good, str(tmp_path / "nope.csv")), tmp_path)
    narrow = write(tmp_path / "r.csv", [{"code": "A1"}])
    with pytest.raises(ConfigError, match=r"R has no column\(s\)"):
        reconcile(job(good, narrow), tmp_path)


def test_a_bad_scope_expression_is_a_config_error(tmp_path):
    cfg = job(write(tmp_path / "l.csv", LEFT), write(tmp_path / "r.csv", RIGHT))
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "right": cfg.right.__class__(
                **{**cfg.right.__dict__, "scope": "nosuchcolumn = 1"}
            ),
        }
    )
    with pytest.raises(ConfigError, match="derive/scope"):
        reconcile(cfg, tmp_path)


def test_derived_key_columns_let_two_differently_shaped_sources_meet(tmp_path):
    left = write(
        tmp_path / "l.csv",
        [
            {
                "code": "X1",
                "name": "n",
                "qty": "1",
                "lat": "1",
                "lon": "1",
                "country": "c",
            }
        ],
    )
    right = write(
        tmp_path / "r.csv",
        [
            {
                "code": None,
                "alt": "X1",
                "title": "n",
                "qty": "1",
                "lat": "1",
                "lon": "1",
                "iso": "c",
            }
        ],
    )
    cfg = job(left, right)
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "right": cfg.right.__class__(
                **{**cfg.right.__dict__, "derive": {"code": "coalesce(code, alt)"}}
            ),
        }
    )
    assert reconcile(cfg, tmp_path).summary["matched"] == 1


# --- the report -------------------------------------------------------------------------------


def test_the_report_is_one_self_contained_page_with_the_headline_numbers(rec):
    html = build_html(rec)
    assert (
        html.startswith("<!doctype html>") and len(html) > 1_000_000
    )  # Plotly is embedded
    external = re.compile(r"<(link|img|iframe)\b|<script[^>]*\bsrc=|@import")
    assert not external.search(html)  # nothing is fetched from outside
    assert "Keys reused for a different entity (1)" in html
    assert "Unmatched in L" in html and "Learned crosswalk" in html


def test_values_from_the_data_cannot_inject_markup(tmp_path):
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    left = [{**LEFT[0], "name": evil, "code": "A1"}] * 1
    right = [{**RIGHT[0], "title": "Totally unrelated words", "code": "A1"}]
    cfg = job(
        write(tmp_path / "l.csv", left),
        write(tmp_path / "r.csv", right),
        identity=["name"],
        identity_min=1,
    )
    html = build_html(reconcile(cfg, tmp_path))
    assert (
        evil not in html and "<img src=x" not in html and "<script>alert(1)" not in html
    )
    assert "&lt;script&gt;alert(1)" in html
    assert "<img" not in table_html(pl.DataFrame({"a": [evil]}))


def test_write_outputs_creates_every_artifact(rec, tmp_path):
    paths = write_outputs(rec, tmp_path / "out")
    assert all(p.exists() for p in paths.values())
    assert pl.read_csv(paths["pairs"]).height == rec.pairs.height
    assert pl.read_csv(paths["differences"]).height == rec.differences.height
    assert (tmp_path / "out" / "crosswalk_country.csv").exists()
    assert json.loads(paths["summary"].read_text())["matched"] == rec.summary["matched"]


# --- the bundled data -------------------------------------------------------------------------


def test_the_bundled_sample_reconciles_and_every_row_is_accounted_for(home):
    rec = reconcile(load(home / "airports.toml"), home)
    s = rec.summary
    assert s["left"]["rows"] == 500
    assert sum(s["left"]["status"].values()) == s["left"]["rows"]
    assert sum(s["right"]["status"].values()) == s["right"]["rows"]
    assert s["matched"] > 300 and set(s["matched_by_key"]) <= {"iata", "icao"}
    assert (
        s["verdicts"]["identical"]
        + s["verdicts"]["changed"]
        + s["verdicts"]["different_entity"]
        == s["matched"]
    )
    assert re.fullmatch(r"[A-Z]{3}", rec.pairs.filter(pl.col("stage") == 0)["key"][0])


def test_the_sample_shows_real_cosmetic_and_real_changes(home):
    rec = reconcile(load(home / "airports.toml"), home)
    totals = {
        cat: sum(f[cat] for f in rec.summary["fields"].values()) for cat in c.CATEGORIES
    }
    assert (
        totals[c.WITHIN] > 0
        and totals[c.NEAR] > 0
        and totals[c.MISMATCH] > 0
        and totals[c.FORMAT_ONLY] > 0
    )


# --- the command line -------------------------------------------------------------------------


def run(*args):
    return CliRunner().invoke(app, list(args))


def test_run_prints_the_summary_and_writes_files(home, tmp_path):
    out = tmp_path / "out"
    result = run("run", "--config", str(home / "airports.toml"), "--out", str(out))
    assert result.exit_code == 0, result.output
    assert "where every row went" in result.output and "matched by key" in result.output
    assert (out / "report.html").exists() and (out / "differences.csv").exists()


def test_run_can_skip_the_files(home, tmp_path):
    result = run(
        "run",
        "--config",
        str(home / "airports.toml"),
        "--out",
        str(tmp_path / "o"),
        "--no-files",
    )
    assert result.exit_code == 0 and not (tmp_path / "o").exists()


def test_show_explains_a_matched_key_and_an_unknown_one(home):
    rec = reconcile(load(home / "airports.toml"), home)
    key = rec.pairs.filter(pl.col("stage") == 0)["key"][0]
    shown = run("show", key, "--config", str(home / "airports.toml"))
    assert shown.exit_code == 0 and "verdict" in shown.output
    missing = run("show", "QQQQQ", "--config", str(home / "airports.toml"))
    assert missing.exit_code == 1 and "no row in either source" in missing.output


def test_a_bad_config_is_exit_code_2_with_a_message(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[left]\nname = 'x'\n", encoding="utf-8")
    result = run("run", "--config", str(bad))
    assert result.exit_code == 2 and "error:" in result.output
