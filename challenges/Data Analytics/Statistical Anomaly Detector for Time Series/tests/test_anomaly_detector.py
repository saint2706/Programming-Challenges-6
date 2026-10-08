from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ts_anomaly import anomaly_detector as ad
from ts_anomaly.detectors import Params
from ts_anomaly.synth import base_series

HERE = Path(__file__).parent.parent
SAMPLE = HERE / "sample_data"
LABELS = SAMPLE / "labels.json"


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def hourly_csv(path: Path, values, start="2024-03-01", skip=()) -> Path:
    ts = pl.datetime_range(
        pl.datetime(*map(int, start.split("-"))),
        pl.datetime(*map(int, start.split("-"))) + pl.duration(hours=len(values) - 1),
        "1h",
        eager=True,
    )
    rows = [
        f"{t:%Y-%m-%d %H:%M:%S},{v}"
        for i, (t, v) in enumerate(zip(ts, values, strict=True))
        if i not in set(skip)
    ]
    return write_csv(path, "timestamp,value\n" + "\n".join(rows) + "\n")


# --------------------------------------------------------------------- loading


def test_vendored_nab_files_load_cleanly():
    nyc = ad.load_series(SAMPLE / "nyc_taxi.csv")
    assert len(nyc.values) == 10320
    assert nyc.quality.step_seconds == 1800
    assert nyc.quality.gaps_filled == 0 and not nyc.quality.irregular
    assert str(nyc.timestamps[0]).startswith("2014-07-01T00:00")
    ambient = ad.load_series(SAMPLE / "ambient_temperature_system_failure.csv")
    assert ambient.quality.rows_read == 7267
    assert ambient.quality.gaps_filled == 621  # three holes in an hourly series
    assert ambient.filled.sum() == 621
    assert len(ambient.values) == 7267 + 621
    assert ambient.quality.rows_used == 7267


def test_bad_rows_are_dropped_and_counted_under_the_first_check_they_fail(tmp_path):
    p = write_csv(
        tmp_path / "messy.csv",
        "timestamp,value\n"
        "2024-01-01 00:00:00,1.0\n"
        ",2.0\n"  # missing timestamp
        "not-a-date,3.0\n"  # unparseable timestamp
        "2024-01-01 03:00:00,\n"  # missing value
        "2024-01-01 04:00:00,abc\n"  # non-numeric value
        "2024-01-01 05:00:00,nan\n"  # non-finite
        "2024-01-01 06:00:00,inf\n"  # non-finite
        "2024-01-01 02:00:00,5.0\n"  # out of order (sorted later)
        "2024-01-01 02:00:00,9.0\n"  # duplicate timestamp: first wins
        "2024-01-01 01:00:00,4.0\n",
    )
    ts = ad.load_series(p)
    q = ts.quality
    assert q.rows_read == 10
    assert (q.null_timestamp, q.unparseable_timestamp, q.null_value) == (1, 1, 1)
    assert (q.non_numeric_value, q.non_finite_value, q.duplicate_timestamps) == (
        1,
        2,
        1,
    )
    assert q.out_of_order
    assert ts.values.tolist() == [1.0, 4.0, 5.0]
    assert np.all(np.diff(ts.timestamps) > np.timedelta64(0, "us"))


def test_mixed_timestamp_formats_and_offsets_are_normalised_to_utc(tmp_path):
    p = write_csv(
        tmp_path / "mixed.csv",
        "timestamp,value\n2024-01-01T00:00:00Z,1\n2024-01-01T02:00:00+01:00,2\n2024-01-01 02:00:00,3\n2024-01-01,4\n",
    )
    ts = ad.load_series(p)
    # 00:00Z, 01:00Z (=02:00+01:00), 02:00 (naive = UTC), and midnight (a duplicate of the first)
    assert ts.quality.duplicate_timestamps == 1
    assert [str(t)[:16] for t in ts.timestamps] == [
        "2024-01-01T00:00",
        "2024-01-01T01:00",
        "2024-01-01T02:00",
    ]


def test_holes_in_a_regular_grid_are_filled_and_marked(tmp_path):
    values = np.arange(20, dtype=float)
    ts = ad.load_series(hourly_csv(tmp_path / "gaps.csv", values, skip={5, 6, 12}))
    assert len(ts.values) == 20
    assert ts.quality.gaps_filled == 3
    assert ts.filled.nonzero()[0].tolist() == [5, 6, 12]
    assert np.allclose(ts.values, values)  # a straight line interpolates exactly
    assert ts.quality.rows_used == 17


def test_mostly_empty_or_off_grid_series_is_left_alone(tmp_path):
    sparse = ad.load_series(
        hourly_csv(
            tmp_path / "sparse.csv", np.arange(40.0), skip=set(range(1, 40, 2)) - {39}
        )
    )
    assert (
        sparse.quality.gaps_filled == 0
    )  # ~half the grid missing: refuse to invent it
    assert sparse.quality.irregular

    off = write_csv(
        tmp_path / "off.csv",
        "timestamp,value\n2024-01-01 00:00:00,1\n2024-01-01 01:00:00,2\n2024-01-01 02:20:00,3\n2024-01-01 03:00:00,4\n",
    )
    ts = ad.load_series(off)
    assert ts.quality.irregular and ts.quality.gaps_filled == 0
    assert any("regular grid" in n for n in ts.quality.notes)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "empty"),
        ("timestamp,value\n", "no usable rows"),
        ("timestamp,other\n2024-01-01,1\n", "no column named 'value'"),
        ("timestamp,value\nbad,1\n,2\n", "no usable rows"),
    ],
)
def test_unusable_files_raise_series_error(tmp_path, text, match):
    with pytest.raises(ad.SeriesError, match=match):
        ad.load_series(write_csv(tmp_path / "bad.csv", text))


def test_custom_column_names(tmp_path):
    p = write_csv(
        tmp_path / "c.csv", "when,reading\n2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n"
    )
    ts = ad.load_series(p, time_col="when", value_col="reading")
    assert ts.values.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------- labels


def test_load_windows_matches_by_file_name_or_explicit_key(tmp_path):
    assert len(ad.load_windows(LABELS, "nyc_taxi.csv")) == 5
    assert ad.load_windows(
        LABELS, "whatever.csv", key="artificialWithAnomaly/art_daily_jumpsdown.csv"
    ) == [("2014-04-10 16:15:00.000000", "2014-04-12 01:45:00.000000")]
    with pytest.raises(ad.SeriesError, match="0 entries match"):
        ad.load_windows(LABELS, "missing.csv")
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps({"a/x.csv": [], "b/x.csv": []}))
    with pytest.raises(ad.SeriesError, match="2 entries match"):
        ad.load_windows(dup, "x.csv")
    with pytest.raises(ad.SeriesError, match="no entry"):
        ad.load_windows(dup, "x.csv", key="nope")
    dup.write_text("{not json")
    with pytest.raises(ad.SeriesError, match="cannot read labels"):
        ad.load_windows(dup, "x.csv")
    dup.write_text(json.dumps({"x.csv": [["only-one"]]}))
    with pytest.raises(ad.SeriesError, match="start, end"):
        ad.load_windows(dup, "x.csv")


# -------------------------------------------------------------------- analysis


def seasonal_series(tmp_path, n=480, spikes=(100, 250, 400)):
    rng = np.random.default_rng(0)
    x = base_series(n, 24, rng=rng)
    for s in spikes:
        x[s] += 9
    return ad.load_series(hourly_csv(tmp_path / "s.csv", np.round(x, 4)))


def test_analyze_auto_detects_the_period_and_runs_every_method(tmp_path):
    ts = seasonal_series(tmp_path)
    an = ad.analyze(ts)
    assert an.periods == [24] and an.period_source == "auto"
    assert not an.skipped
    assert set(an.detections) == set(ad.METHODS)
    assert an.detections["stl-gesd"].flags[[100, 250, 400]].all()


def test_stl_methods_are_skipped_with_a_reason_when_there_is_no_seasonality(tmp_path):
    rng = np.random.default_rng(0)
    ts = ad.load_series(
        hourly_csv(tmp_path / "n.csv", np.round(rng.normal(0, 1, 300), 4))
    )
    an = ad.analyze(ts)
    assert an.periods == []
    assert set(an.skipped) == set(ad.STL_METHODS)
    assert "no seasonality detected" in an.skipped["stl-mad"]
    assert "mad" in an.detections
    none = ad.analyze(seasonal_series(tmp_path), ["stl-mad"], periods="none")
    assert "no seasonal period given" in none.skipped["stl-mad"]


def test_explicit_periods_bypass_detection(tmp_path):
    an = ad.analyze(seasonal_series(tmp_path), ["stl-mad"], periods=[24])
    assert an.period_source == "given" and an.period_estimate is None
    assert an.detections["stl-mad"].params["periods"] == [24]


def test_unknown_methods_and_impossible_windows(tmp_path):
    ts = seasonal_series(tmp_path)
    with pytest.raises(ad.SeriesError, match="unknown method"):
        ad.analyze(ts, ["magic"])
    an = ad.analyze(ts, ["rolling-z"], params=Params(window=10_000))
    assert "rolling-z" in an.skipped


def test_interpolated_samples_are_never_flagged(tmp_path):
    ts = seasonal_series(tmp_path)
    ts.filled[50] = True
    ts.values[50] += 500  # absurd, but it was "not observed"
    an = ad.analyze(ts, ["zscore", "mad"], periods="none")
    assert not an.detections["zscore"].flags[50]
    assert not an.detections["mad"].flags[50]


# ------------------------------------------------------------ evaluation/report


def test_evaluate_detection_against_labelled_windows():
    ts = ad.load_series(SAMPLE / "art_daily_jumpsdown.csv")
    windows = ad.load_windows(LABELS, "art_daily_jumpsdown.csv")
    an = ad.analyze(ts, ["mad", "zscore"], periods="none")
    hit = ad.evaluate_detection(ts, an.detections["mad"], windows)
    miss = ad.evaluate_detection(ts, an.detections["zscore"], windows)
    assert (hit.nab.windows_hit, hit.nab.windows_total) == (1, 1)
    assert miss.nab.windows_hit == 0 and miss.nab.normalized == 0.0
    assert 0 < hit.point_precision <= 1 and 0 < hit.point_recall <= 1


def test_report_escapes_data_derived_strings(tmp_path):
    ts = seasonal_series(tmp_path)
    evil = '<img src=x onerror="alert(1)">.csv'
    ts.name = evil
    an = ad.analyze(ts, ["zscore", "stl-mad"], periods="none")
    an.skipped["<script>alert(2)</script>"] = "<b>why</b>"
    an.series.quality.notes.append("<svg onload=alert(3)>")
    html = ad.build_report(an)
    for payload in (
        '<img src=x onerror="alert(1)">',
        "<script>alert(2)</script>",
        "<b>why</b>",
        "<svg onload=alert(3)>",
    ):
        assert payload not in html
    assert "&lt;img src=x" in html and "&lt;script&gt;alert(2)" in html


def test_report_structure_with_and_without_labels(tmp_path):
    ts = ad.load_series(SAMPLE / "art_daily_jumpsdown.csv")
    an = ad.analyze(ts, ["mad", "stl-gesd"], periods=[288])
    plain = ad.build_report(an)
    scored = ad.build_report(an, ad.load_windows(LABELS, "art_daily_jumpsdown.csv"))
    assert "Data quality" in plain and "Detector results" in plain
    assert "NAB score" not in plain
    assert "NAB score" in scored and "1/1" in scored
    assert plain.count("Plotly.newPlot") == 1  # one figure, one inlined bundle
    assert plain.startswith("<!doctype html>")
    # nothing was fetched from the network
    assert (
        'src="http' not in plain
        and "cdn.plot.ly" not in plain.split("Plotly.newPlot")[0][-3000:]
    )


def test_report_with_no_detector_output_still_renders(tmp_path):
    an = ad.analyze(seasonal_series(tmp_path), ["stl-mad"], periods="none")
    html = ad.build_report(an)
    assert "No detector produced a result" in html
    assert "skipped: no seasonal period given" in html


# ------------------------------------------------------------------------- CLI


def test_cli_end_to_end_with_labels_and_flags_csv(tmp_path, capsys):
    out, flags = tmp_path / "r.html", tmp_path / "f.csv"
    code = ad.main(
        [
            str(SAMPLE / "art_daily_jumpsdown.csv"),
            "--labels",
            str(LABELS),
            "--methods",
            "mad,rolling-mad,stl-z",
            "--period",
            "288",
            "-o",
            str(out),
            "--flags-out",
            str(flags),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "4032 samples" in text and "period: [288] (given)" in text
    assert "stl-z" in text and "windows 1/1" in text
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    df = pl.read_csv(flags)
    assert df.columns == [
        "timestamp",
        "value",
        "interpolated",
        "mad",
        "rolling_mad",
        "stl_z",
    ]
    assert df.height == 4032 and df["mad"].sum() == 1500


def test_cli_reports_errors_with_exit_code_2(tmp_path, capsys):
    assert ad.main([str(tmp_path / "missing.csv")]) == 2
    bad = write_csv(tmp_path / "bad.csv", "a,b\n1,2\n")
    assert ad.main([str(bad), "-o", str(tmp_path / "o.html")]) == 2
    assert "no column named 'timestamp'" in capsys.readouterr().err
    ok = SAMPLE / "art_daily_jumpsdown.csv"
    assert (
        ad.main([str(ok), "--methods", "sorcery", "-o", str(tmp_path / "o.html")]) == 2
    )
    assert (
        ad.main(
            [
                str(ok),
                "--labels",
                str(tmp_path / "nolabels.json"),
                "-o",
                str(tmp_path / "o.html"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "o.html").exists()


def test_cli_rejects_a_bad_period_argument():
    with pytest.raises(SystemExit) as exc:
        ad.main([str(SAMPLE / "nyc_taxi.csv"), "--period", "weekly"])
    assert exc.value.code == 2


def test_cli_stl_options_are_passed_through(tmp_path, capsys):
    code = ad.main(
        [
            str(SAMPLE / "art_daily_jumpsdown.csv"),
            "--methods",
            "stl-z",
            "--period",
            "288",
            "--no-stl-robust",
            "--stl-seasonal",
            "13",
            "-o",
            str(tmp_path / "o.html"),
        ]
    )
    assert code == 0
    assert "stl-z" in capsys.readouterr().out
