from datetime import UTC, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

from cronwatch.crontab import Crontab
from cronwatch.store import FAILED, OK, RUNNING, Store
from cronwatch.timeline import LABEL_MAX, NOW, build_timeline, render_timeline

NOW_T = datetime(2026, 10, 9, 12, 20, tzinfo=UTC)
WIDTH = 48  # 24h window at 30 minutes per slice: easy arithmetic
KW = {"width": WIDTH, "past": timedelta(hours=6), "future": timedelta(hours=18)}


def wrapped(job_id, schedule):
    return f"{schedule} /x/cronwatch run --job {job_id} -- sh -c 'true'\n"


def build(text, runs=None, since=None, **kw):
    jobs = Crontab.from_text(text).jobs()
    return build_timeline(
        jobs, runs or {}, NOW_T, UTC, monitored_since=since, **{**KW, **kw}
    )


def chars(timeline, row=0):
    return "".join(cell.char for cell in timeline.rows[row].cells)


def col(timeline, when):
    return int((when - timeline.start) / timeline.slice)


def record(store, job, when, status=OK):
    run_id = store.start_run(job, "c", "h", when)
    store.finish_run(
        run_id, when + timedelta(seconds=2), 0 if status == OK else 1, status, ""
    )
    return store.get_run(run_id)


def test_geometry_the_now_marker_and_slice_size():
    timeline = build(wrapped("j", "0 * * * *"))
    assert timeline.slice == timedelta(minutes=30) and timeline.width == WIDTH
    assert timeline.now_column == 12  # six hours of history = 12 slices
    assert chars(timeline)[12] == NOW
    assert timeline.start == NOW_T - timedelta(
        hours=6
    ) and timeline.end == NOW_T + timedelta(hours=18)


def test_future_runs_become_bars_in_the_right_slices():
    row = chars(build(wrapped("j", "0 * * * *")))
    # 13:00 is slice 13 (12:30-13:00 ends...) hourly => every second slice from 13
    bars = [i for i, c in enumerate(row) if c == "▂"]
    assert (
        bars[0] == 13 and all(b - a == 2 for a, b in pairwise(bars)) and len(bars) == 18
    )


def test_more_runs_per_slice_make_taller_bars():
    narrow = {
        "width": 24,
        "past": timedelta(hours=1),
        "future": timedelta(hours=11),
    }  # 30-minute slices
    every_5 = chars(build(wrapped("j", "*/5 * * * *"), **narrow))
    every_minute = chars(build(wrapped("j", "* * * * *"), **narrow))
    assert set(every_5[every_5.index("┆") + 1 :]) == {"▆"}  # six runs per slice
    assert set(every_minute[every_minute.index("┆") + 1 :]) == {
        "█"
    }  # thirty per slice: the tallest bar
    assert "▄" in chars(
        build(wrapped("j", "0,20,40 * * * *"))
    )  # three per hour = two in some slices


def test_the_now_marker_survives_a_slice_that_also_has_upcoming_runs():
    timeline = build(wrapped("j", "*/5 * * * *"))
    assert chars(timeline)[timeline.now_column] == NOW


def test_history_glyphs_ok_failed_and_a_failure_outranks_a_success_in_the_same_slice(
    tmp_path,
):
    store = Store(tmp_path / "t.db")
    ok_at = NOW_T - timedelta(hours=1, minutes=20)  # 11:00
    failed_at = NOW_T - timedelta(hours=2, minutes=20)  # 10:00
    mixed = NOW_T - timedelta(
        hours=3, minutes=20
    )  # 09:00: a success then a failure in one slice
    runs = [
        record(store, "j", ok_at),
        record(store, "j", failed_at, FAILED),
        record(store, "j", mixed),
        record(store, "j", mixed + timedelta(minutes=2), FAILED),
    ]
    timeline = build(wrapped("j", "0 * * * *"), {"j": runs})
    row = chars(timeline)
    assert row[col(timeline, ok_at)] == "●"
    assert row[col(timeline, failed_at)] == "✗"
    assert row[col(timeline, mixed)] == "✗"


def test_a_running_job_shows_as_in_progress(tmp_path):
    store = Store(tmp_path / "t.db")
    run_id = store.start_run("j", "c", "h", NOW_T - timedelta(minutes=10))
    row = chars(build(wrapped("j", "10 * * * *"), {"j": [store.get_run(run_id)]}))
    assert RUNNING and "◐" in row


def test_scheduled_but_unrecorded_runs_are_flagged_only_once_watched():
    text = wrapped("j", "0 * * * *")
    assert "○" not in chars(build(text))  # nobody was monitoring yet: not a failure
    watched_from = NOW_T - timedelta(hours=3)  # 09:20
    timeline = build(text, since={"j": watched_from})
    flagged = [i for i, c in enumerate(chars(timeline)) if c == "○"]
    # 10:00 and 11:00 should have run and did not; 12:00's slice is still inside the grace period
    assert flagged == [
        col(timeline, NOW_T.replace(hour=10, minute=0)),
        col(timeline, NOW_T.replace(hour=11, minute=0)),
    ]
    assert all(i > col(timeline, watched_from) for i in flagged)


def test_the_most_recent_slice_gets_a_grace_period():
    text = wrapped("j", "*/10 * * * *")
    row = chars(
        build(
            text,
            since={"j": NOW_T - timedelta(hours=6)},
            past=timedelta(hours=1),
            future=timedelta(hours=1),
            width=12,
        )
    )
    assert row[row.index("┆") - 1] != "○"  # within grace of "now": not judged yet


def test_unwrapped_jobs_show_scheduled_history_as_unknown():
    row = chars(build("0 * * * * /usr/bin/thing\n"))
    assert "·" in row[:12] and "○" not in row and "●" not in row


def test_inactive_jobs_are_labelled_not_drawn():
    text = "#cronwatch:off# 0 9 * * * /paused\n@reboot /boot\n61 * * * * bad\n"
    timeline = build(text)
    assert [
        chars(timeline, i).lstrip("·┆").startswith(" (")
        or " (" in chars(timeline, i)[:12]
        for i in range(3)
    ] == [True] * 3
    assert (
        "(off)" in chars(timeline, 0)
        and "(reboot)" in chars(timeline, 1)
        and "(invalid)" in chars(timeline, 2)
    )
    assert "▂" not in "".join(chars(timeline, i) for i in range(3))


def test_busiest_minutes_group_jobs_that_start_together_biggest_first():
    text = (
        "".join(wrapped(f"j{i}", "0 15 * * *") for i in range(3))
        + wrapped("solo", "30 15 * * *")
        + wrapped("pair1", "0 20 * * *")
        + wrapped("pair2", "0 20 * * *")
    )
    busy = build(text).busy
    assert [(m.hour, len(names)) for m, names in busy] == [(15, 3), (20, 2)]
    assert busy[0][1] == ["j0", "j1", "j2"]


def test_busy_list_is_capped_and_ignores_single_starts():
    text = wrapped("a", "0 * * * *") + wrapped("b", "0 * * * *")
    assert len(build(text, busy_top=3).busy) == 3
    assert build(wrapped("a", "0 * * * *")).busy == []


def test_axis_has_whole_hour_labels_in_the_display_timezone_without_overlap():
    berlin = ZoneInfo("Europe/Berlin")
    timeline = build_timeline(
        Crontab.from_text(wrapped("j", "0 * * * *")).jobs(), {}, NOW_T, berlin, **KW
    )
    labels = timeline.axis.split()
    assert labels[0] == "08:00" or labels[0].endswith(
        ":00"
    )  # UTC 06:20 = 08:20 in Berlin (CEST)
    assert all(label.endswith(":00") for label in labels) and len(labels) == len(
        set(labels)
    )
    assert len(timeline.axis) == WIDTH


def test_long_labels_are_truncated_and_short_ones_padded_to_one_column_width():
    text = wrapped("a-very-long-job-name-that-keeps-going", "0 * * * *") + wrapped(
        "x", "0 * * * *"
    )
    labels = [row.label for row in build(text).rows]
    assert all(len(label) == LABEL_MAX for label in labels) and labels[0].endswith("…")


def test_render_includes_axis_rows_legend_and_busy_summary():
    plain = render_timeline(
        build(wrapped("alpha", "0 15 * * *") + wrapped("beta", "0 15 * * *"))
    ).plain
    assert "alpha" in plain and "beta" in plain and "now" in plain
    assert (
        "Busiest minutes ahead" in plain
        and "2 jobs: alpha, beta" in plain
        and "Fri 15:00" in plain
    )


def test_dst_does_not_distort_the_window_which_is_always_utc_exact():
    ny = ZoneInfo("America/New_York")
    around_fall_back = datetime(2026, 11, 1, 5, 0, tzinfo=UTC)
    timeline = build_timeline(
        Crontab.from_text(wrapped("j", "30 1 * * *")).jobs(),
        {},
        around_fall_back,
        ny,
        **KW,
    )
    assert timeline.end - timeline.start == timedelta(hours=24)
