# ruff: noqa: DTZ001  (the module under test deliberately returns naive-UTC datetimes)
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cohort_retention as cr
import generate_sample
import polars as pl
import pytest

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_data" / "events.csv"
MESSY = HERE / "sample_data" / "messy_events.csv"


def events(rows: list[tuple[str, ...]]) -> pl.DataFrame:
    cols = ["user_id", "timestamp", "event_name"][: len(rows[0])]
    return pl.DataFrame(rows, schema=dict.fromkeys(cols, pl.String), orient="row")


# Weeks start on Mondays: w0 = 2024-01-01, w1 = 01-08, w2 = 01-15, w3 = 01-22.
# u3 skips w1 but returns in w2, which is what separates the two modes.
HAND = events(
    [
        ("u1", "2024-01-01"),
        ("u1", "2024-01-09"),
        ("u1", "2024-01-16"),
        ("u2", "2024-01-02"),
        ("u2", "2024-01-10"),
        ("u3", "2024-01-03"),
        ("u3", "2024-01-17"),
        ("u4", "2024-01-08"),
        ("u4", "2024-01-16"),
        ("u5", "2024-01-09"),
        ("u6", "2024-01-22 12:00:00"),  # last event: w3 is still open
    ]
)


class TestMatrixMath:
    def test_active_mode_hand_computed(self):
        r = cr.analyze(HAND, granularity="week", mode="active")
        assert r.cohort_labels == ["2024-01-01", "2024-01-08", "2024-01-22"]
        assert r.cohort_sizes == [3, 2, 1]
        assert r.n_offsets == 4
        assert r.counts == [[3, 2, 2, None], [2, 1, None, None], [None] * 4]
        assert r.pct[0][:3] == pytest.approx([100.0, 200 / 3, 200 / 3])
        assert r.pct[0][3] is None
        assert r.pct[1][:2] == pytest.approx([100.0, 50.0])
        assert r.pct[2] == [None] * 4

    def test_weighted_average_ignores_censored_cohorts(self):
        r = cr.analyze(HAND, granularity="week", mode="active")
        # offset 1: cohorts w0+w1 observed -> (2+1)/(3+2); offset 2: only w0 -> 2/3
        assert r.average_curve[:3] == pytest.approx([100.0, 60.0, 200 / 3])
        assert r.average_curve[3] is None

    def test_return_mode_counts_later_activity(self):
        r = cr.analyze(HAND, granularity="week", mode="return")
        # u3 is absent in w1 but back in w2 -> counts as retained at offset 1
        assert r.counts[0] == [3, 3, 2, None]
        assert r.counts[1][:2] == [2, 1]
        assert r.average_curve[1] == pytest.approx(80.0)  # (3+1)/5

    def test_return_curve_is_monotone_non_increasing(self):
        r = cr.analyze(HAND, granularity="week", mode="return")
        row = [v for v in r.counts[0] if v is not None]
        assert row == sorted(row, reverse=True)

    def test_include_partial_reveals_last_period_as_real_zero(self):
        r = cr.analyze(HAND, granularity="week", include_partial=True)
        assert r.counts[0][3] == 0  # observed, nobody came back: 0%, not masked
        assert r.pct[0][3] == 0.0
        assert r.counts[2][0] == 1
        assert r.pct[2][0] == 100.0

    def test_partial_period_is_masked_by_default_not_zero(self):
        r = cr.analyze(HAND)
        assert r.pct[0][3] is None
        assert r.pct[2][0] is None

    def test_single_period_of_data_has_nothing_observable(self):
        r = cr.analyze(events([("a", "2024-01-01"), ("b", "2024-01-02")]))
        assert r.cohort_sizes == [2]
        assert r.pct == [[None]]
        assert r.average_curve == [None]

    def test_offset_zero_is_always_full_when_observable(self):
        r = cr.analyze(HAND)
        assert all(row[0] == 100.0 for row in r.pct if row[0] is not None)


class TestGranularity:
    def test_week_boundary_sunday_vs_monday(self):
        df = events(
            [
                ("a", "2024-01-07T23:59:59Z"),  # Sunday -> w0
                ("b", "2024-01-08T00:00:00Z"),  # Monday -> w1
                ("a", "2024-01-20"),
            ]
        )
        assert cr.analyze(df, granularity="week").cohort_labels == [
            "2024-01-01",
            "2024-01-08",
        ]

    def test_day_boundary(self):
        df = events(
            [
                ("a", "2024-01-01 23:59:59"),
                ("b", "2024-01-02 00:00:00"),
                ("a", "2024-01-05"),
            ]
        )
        r = cr.analyze(df, granularity="day")
        assert r.cohort_labels == ["2024-01-01", "2024-01-02"]
        assert r.cohort_sizes == [1, 1]

    def test_month_boundary_and_year_rollover(self):
        df = events(
            [
                ("a", "2023-12-31T23:59:59Z"),
                ("b", "2024-01-01T00:00:00Z"),
                ("a", "2024-01-15"),  # a is active one month after the Dec cohort
                ("b", "2024-03-01"),
            ]
        )
        r = cr.analyze(df, granularity="month")
        assert r.cohort_labels == ["2023-12", "2024-01"]
        assert r.n_offsets == 4  # Dec, Jan, Feb, Mar
        assert r.counts[0][:3] == [1, 1, 0]
        assert r.counts[1][:3] == [1, 0, None]  # March is the still-open month

    def test_period_helpers_roundtrip(self):
        for gran, sample in [
            ("day", "2024-02-29"),
            ("week", "2024-02-26"),
            ("month", "2024-02-01"),
        ]:
            frame = pl.DataFrame({"t": [datetime.fromisoformat(sample)]})
            idx = frame.select(cr.period_index_expr("t", gran)).item()
            assert cr.period_start(idx, gran).isoformat() == sample
            assert cr.period_start(idx + 1, gran) > cr.period_start(idx, gran)

    def test_invalid_granularity_or_mode(self):
        with pytest.raises(ValueError, match="granularity"):
            cr.analyze(HAND, granularity="year")
        with pytest.raises(ValueError, match="mode"):
            cr.analyze(HAND, mode="bogus")


class TestTimestamps:
    def test_mixed_formats_all_normalise_to_utc(self):
        df = pl.DataFrame(
            {
                "t": [
                    "2024-03-10T12:00:00Z",
                    "2024-03-10T14:00:00+02:00",
                    "2024-03-10 12:00:00",
                    "2024-03-10 07:00:00-05:00",
                    "2024-03-10T12:00:00.250",
                    "2024-03-10",
                    "garbage",
                    None,
                ]
            }
        )
        out = df.select(cr.parse_timestamps("t")).to_series().to_list()
        noon = datetime(2024, 3, 10, 12)
        assert out[:4] == [noon] * 4
        assert out[4] == datetime(2024, 3, 10, 12, 0, 0, 250000)
        assert out[5] == datetime(2024, 3, 10)
        assert out[6:] == [None, None]

    def test_dst_spring_forward_offsets_are_one_utc_hour_apart(self):
        # New York, 2024-03-10: 02:00 EST jumps to 03:00 EDT. 01:30 EST and 03:30 EDT
        # look 2h apart on the wall clock but are exactly 1h apart in real time.
        df = pl.DataFrame(
            {"t": ["2024-03-10T01:30:00-05:00", "2024-03-10T03:30:00-04:00"]}
        )
        a, b = df.select(cr.parse_timestamps("t")).to_series().to_list()
        assert a == datetime(2024, 3, 10, 6, 30)
        assert (b - a).total_seconds() == 3600

    def test_local_date_differs_from_utc_date_cohort_uses_utc(self):
        df = events(
            [
                ("a", "2024-01-07T23:30:00-05:00"),  # = Mon 01-08 04:30Z -> w1
                ("b", "2024-01-08T01:00:00+02:00"),  # = Sun 01-07 23:00Z -> w0
                ("a", "2024-01-20"),
            ]
        )
        r = cr.analyze(df, granularity="week")
        assert r.cohort_labels == ["2024-01-01", "2024-01-08"]
        assert r.cohort_sizes == [1, 1]

    def test_whitespace_around_values_is_tolerated(self):
        df = events([(" a ", " 2024-01-01 "), ("a", "2024-01-09")])
        r = cr.analyze(df)
        assert r.quality.users == 1
        assert r.quality.unparseable_timestamp == 0


class TestDataQuality:
    def test_messy_file_counts_and_no_signup_event(self):
        q = cr.analyze(cr.read_events(MESSY)).quality
        assert (q.rows_read, q.null_user_id, q.null_timestamp) == (12, 1, 1)
        assert (q.unparseable_timestamp, q.duplicate_rows) == (1, 1)
        assert q.rows_used == 8 and q.users == 5
        dropped = (
            q.null_user_id
            + q.null_timestamp
            + q.unparseable_timestamp
            + q.duplicate_rows
        )
        assert q.rows_read - dropped == q.rows_used

    def test_signup_event_cohorts_and_drops(self):
        r = cr.analyze(cr.read_events(MESSY), signup_event="signup")
        q = r.quality
        assert q.users_without_signup == 2  # d and e never signed up
        assert q.events_before_signup == 0
        assert q.rows_used == 5
        # 2024-03-10 is a Sunday -> a & b in week of 03-04; c signs up Monday 03-11
        assert r.cohort_labels == ["2024-03-04", "2024-03-11"]
        assert r.cohort_sizes == [2, 1]
        assert r.pct == [[100.0, 0.0, None], [100.0, None, None]]

    def test_events_before_signup_are_dropped(self):
        df = events(
            [
                ("x", "2024-01-01", "session"),  # pre-signup browsing
                ("x", "2024-01-09", "signup"),
                ("x", "2024-01-16", "session"),
                ("y", "2024-01-02", "session"),  # never signs up
                ("z", "2024-01-30", "signup"),
            ]
        )
        r = cr.analyze(df, signup_event="signup")
        assert r.quality.events_before_signup == 1
        assert r.quality.users_without_signup == 1
        assert r.cohort_labels == ["2024-01-08", "2024-01-29"]
        assert r.counts[0][:2] == [1, 1]

    def test_exact_duplicates_do_not_inflate_counts(self):
        df = events([("a", "2024-01-01"), ("a", "2024-01-01"), ("a", "2024-01-09")])
        r = cr.analyze(df)
        assert r.quality.duplicate_rows == 1
        assert r.cohort_sizes == [1]

    def test_all_rows_invalid_gives_empty_result(self):
        df = events([("", "2024-01-01"), ("a", "nope"), ("b", "")])
        r = cr.analyze(df)
        assert r.cohort_labels == []
        q = r.quality
        assert (q.null_user_id, q.unparseable_timestamp, q.null_timestamp) == (1, 1, 1)

    def test_signup_event_requires_event_column(self):
        with pytest.raises(ValueError, match="event_name"):
            cr.analyze(events([("a", "2024-01-01")]), signup_event="signup")

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="user_id"):
            cr.analyze(pl.DataFrame({"who": ["a"], "timestamp": ["2024-01-01"]}))

    def test_custom_column_names(self):
        df = pl.DataFrame(
            {"uid": ["a", "a"], "ts": ["2024-01-01", "2024-01-09"], "e": ["x", "y"]}
        )
        r = cr.analyze(df, user_col="uid", time_col="ts", event_col="e")
        assert r.cohort_sizes == [1]


class TestEmptyInput:
    def test_zero_byte_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_bytes(b"")
        r = cr.analyze(cr.read_events(p))
        assert r.cohort_labels == [] and r.quality.rows_read == 0

    def test_header_only_file(self, tmp_path):
        p = tmp_path / "h.csv"
        p.write_text("user_id,timestamp\n")
        r = cr.analyze(cr.read_events(p))
        assert r.cohort_labels == [] and r.quality.rows_read == 0

    def test_empty_report_renders_notice_not_charts(self):
        page = cr.build_report(cr.analyze(pl.DataFrame()), "empty.csv")
        assert "No usable events" in page
        assert "Plotly" not in page


class TestSampleData:
    def test_committed_sample_matches_generator(self):
        rows = generate_sample.generate()
        lines = SAMPLE.read_text().splitlines()
        assert lines[0] == "user_id,timestamp,event_name"
        assert [tuple(line.split(",")) for line in lines[1:]] == rows

    def test_recovers_known_retention_curve(self):
        r = cr.analyze(cr.read_events(SAMPLE))
        assert r.quality.users == generate_sample.N_USERS
        assert r.average_curve[0] == 100.0
        for n in (1, 2, 3):
            expected = 100 * generate_sample.true_retention(n)
            assert r.average_curve[n] == pytest.approx(expected, abs=5.0)
        observed = [v for v in r.average_curve if v is not None]
        assert observed[1] > observed[4] > observed[7]  # decays

    def test_censoring_staircase(self):
        r = cr.analyze(cr.read_events(SAMPLE))
        assert len(r.cohort_labels) == generate_sample.N_SIGNUP_WEEKS
        last_complete = generate_sample.LOG_WEEKS - 2  # week 11 is the still-open week
        for cohort, row in enumerate(r.pct):
            for n, v in enumerate(row):
                assert (v is None) == (cohort + n > last_complete), (cohort, n)


class TestReportAndCli:
    def test_report_is_self_contained_and_escaped(self):
        r = cr.analyze(cr.read_events(SAMPLE))
        page = cr.build_report(r, "<script>alert(1)</script>.csv")
        assert "<script>alert(1)</script>.csv" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;.csv" in page
        assert "<script src=" not in page  # Plotly is inlined, never fetched
        assert page.count("Plotly.newPlot") == 3  # heatmap, curves, sizes
        assert "not yet observable" in page

    def test_cli_writes_report(self, tmp_path, capsys):
        out = tmp_path / "r.html"
        argv = [
            str(SAMPLE),
            "--granularity",
            "week",
            "--mode",
            "return",
            "-o",
            str(out),
        ]
        assert cr.main(argv) == 0
        assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
        assert "400 users" in capsys.readouterr().out

    def test_cli_month_granularity(self, tmp_path):
        out = tmp_path / "m.html"
        assert cr.main([str(SAMPLE), "--granularity", "month", "-o", str(out)]) == 0

    def test_cli_bad_column_exits_2(self, tmp_path, capsys):
        p = tmp_path / "bad.csv"
        p.write_text("who,when\na,2024-01-01\n")
        assert cr.main([str(p), "-o", str(tmp_path / "x.html")]) == 2
        assert "missing required column" in capsys.readouterr().err

    def test_cli_missing_file_exits_2(self, tmp_path):
        assert (
            cr.main([str(tmp_path / "nope.csv"), "-o", str(tmp_path / "x.html")]) == 2
        )

    def test_cli_empty_file_still_writes_report(self, tmp_path):
        p = tmp_path / "e.csv"
        p.write_bytes(b"")
        out = tmp_path / "e.html"
        assert cr.main([str(p), "-o", str(out)]) == 0
        assert "No usable events" in out.read_text(encoding="utf-8")
