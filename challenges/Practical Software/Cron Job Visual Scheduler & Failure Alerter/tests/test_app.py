import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from cronwatch.alerts import Alerter
from cronwatch.app import Confirm, CronWatchApp, JobEditor, preview_text
from cronwatch.config import Config
from cronwatch.crontab import FileCrontab, Session
from cronwatch.store import FAILED, OK, Store
from textual.widgets import DataTable, Input, Static, TabbedContent

NOW = datetime(2026, 10, 9, 12, 20, tzinfo=UTC)
WRAPPER = "/x/cronwatch"
CRONTAB = """\
# nightly database dump
0 2 * * * /usr/bin/pg_backup --all
*/15 * * * * /x/cronwatch run --job pinger -- sh -c 'echo pong'
0 9 * * 1-5 /usr/bin/report.sh
"""


class Recording:
    name = "recording"

    def __init__(self, error=None):
        self.alerts, self.error = [], error

    def send(self, alert):
        if self.error:
            raise self.error
        self.alerts.append(alert)


@pytest.fixture
def parts(tmp_path):
    path = tmp_path / "crontab"
    path.write_text(CRONTAB)
    store = Store(tmp_path / "c.db")
    session = Session(FileCrontab(path), tmp_path / "backups")
    channel = Recording()
    return path, store, session, channel


@pytest.fixture
def app(parts):
    _path, store, session, channel = parts
    return CronWatchApp(
        session,
        store,
        Config(),
        UTC,
        Alerter([channel], store),
        WRAPPER,
        clock=lambda: NOW,
    )


def record(store, job, minutes_ago, status=OK, output=""):
    started = NOW - timedelta(minutes=minutes_ago)
    run_id = store.start_run(job, "cmd", "host", started)
    store.finish_run(
        run_id, started + timedelta(seconds=3), 0 if status == OK else 2, status, output
    )


def table_rows(app):
    table = app.query_one("#jobs", DataTable)
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


async def type_into(pilot, selector, text):
    field = pilot.app.screen.query_one(selector, Input)
    field.value = text
    await pilot.pause()


async def test_lists_every_job_with_meaning_next_run_and_last_result(app, parts):
    record(parts[1], "pinger", 4, FAILED)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        rows = table_rows(app)
    assert [r[1] for r in rows] == ["nightly database dump", "pinger", "report.sh"]
    assert (
        rows[0][3] == "At 02:00"
        and "in 14h" in rows[0][4]
        and rows[0][5] == "not monitored"
    )
    assert (
        rows[1][0] == "✗"
        and "exit 2" in rows[1][5]
        and rows[1][3] == "Every 15 minutes"
    )
    assert rows[1][6] == "echo pong"  # the wrapper boilerplate is hidden
    assert rows[2][3] == "At 09:00 on Monday to Friday"


async def test_highlighting_a_job_updates_the_details_pane(app):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("3")
        await pilot.press("down")
        await pilot.pause()
        details = str(app.query_one("#details-view", Static).render())
    assert (
        "pinger" in details
        and "Every 15 minutes" in details
        and "echo pong" in details
        and "Next runs" in details
    )


async def test_history_tab_lists_runs_and_shows_the_highlighted_runs_output(app, parts):
    record(parts[1], "pinger", 30, OK, "all good\n")
    record(parts[1], "pinger", 10, FAILED, "Traceback: boom\n")
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("2")
        await pilot.press("down")
        await pilot.pause()
        runs = app.query_one("#runs", DataTable)
        assert runs.row_count == 2
        newest = str(app.query_one("#output", Static).render())
        await pilot.press(
            "tab"
        )  # move focus to the runs table, then down to the older run
        runs.focus()
        await pilot.press("down")
        await pilot.pause()
        older = str(app.query_one("#output", Static).render())
    assert "Traceback: boom" in newest and "all good" in older


async def test_history_for_an_unmonitored_job_explains_how_to_start(app):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("2")
        await pilot.pause()
        assert "press w" in str(app.query_one("#output", Static).render())


async def test_timeline_tab_draws_a_row_per_job(app):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("1")
        await pilot.pause()
        text = str(app.query_one("#timeline-view", Static).render())
    assert "pinger" in text and "report.sh" in text and "┆" in text


async def test_space_toggles_a_job_in_the_crontab_file_and_back(app, parts):
    path = parts[0]
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("space")
        await pilot.pause()
        assert "#cronwatch:off# 0 2 * * * /usr/bin/pg_backup --all" in path.read_text()
        assert table_rows(app)[0][0] == "○" and table_rows(app)[0][4] == "disabled"
        await pilot.press("space")
        await pilot.pause()
    assert path.read_text() == CRONTAB


async def test_delete_asks_for_confirmation_and_n_cancels(app, parts):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, Confirm)
        await pilot.press("n")
        await pilot.pause()
        assert parts[0].read_text() == CRONTAB
        await pilot.press("d")
        await pilot.press("y")
        await pilot.pause()
    assert "pg_backup" not in parts[0].read_text()
    assert "0 9 * * 1-5 /usr/bin/report.sh" in parts[0].read_text()


async def test_wrap_and_unwrap_with_w(app, parts):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("w")
        await pilot.pause()
        wrapped = parts[0].read_text()
        assert (
            "/x/cronwatch run --job pg_backup -- sh -c '/usr/bin/pg_backup --all'"
            in wrapped
        )
        assert (
            table_rows(app)[0][1] == "pg_backup" or "pg_backup" in table_rows(app)[0][1]
        )
        await pilot.press("w")
        await pilot.pause()
    assert parts[0].read_text() == CRONTAB


async def test_add_job_shows_a_live_preview_then_saves_wrapped_by_default(app, parts):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, JobEditor)
        await type_into(pilot, "#schedule", "30 4 * * 1-5")
        preview = str(app.screen.query_one("#preview", Static).render())
        assert (
            "At 04:30 on Monday to Friday" in preview
            and "Next: Mon 12 Oct 04:30" in preview
        )
        await type_into(pilot, "#command", "/usr/local/bin/cleanup --fast")
        await type_into(pilot, "#description", "weekday cleanup")
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, JobEditor)
    text = parts[0].read_text()
    assert "# cronwatch: weekday cleanup" in text
    assert (
        "30 4 * * 1-5 /x/cronwatch run --job cleanup -- sh -c '/usr/local/bin/cleanup --fast'"
        in text
    )


async def test_an_invalid_schedule_keeps_the_editor_open_and_writes_nothing(app, parts):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("a")
        await pilot.pause()
        await type_into(pilot, "#schedule", "61 * * * *")
        await type_into(pilot, "#command", "x")
        assert "minute: 61 is outside 0-59" in str(
            app.screen.query_one("#preview", Static).render()
        )
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, JobEditor)
        await pilot.press("escape")
        await pilot.pause()
    assert parts[0].read_text() == CRONTAB


async def test_edit_keeps_the_wrapper_job_id_and_timeout(parts, tmp_path):
    path, store, session, channel = parts
    path.write_text(
        "*/15 * * * * /x/cronwatch run --job pinger --timeout 90 -- sh -c 'echo pong'\n"
    )
    session.reload()
    app = CronWatchApp(
        session,
        store,
        Config(),
        UTC,
        Alerter([channel], store),
        WRAPPER,
        clock=lambda: NOW,
    )
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("e")
        await pilot.pause()
        assert (
            app.screen.query_one("#command", Input).value == "echo pong"
        )  # shown without the wrapper
        await type_into(pilot, "#schedule", "*/20 * * * *")
        await type_into(pilot, "#command", "echo ping")
        await pilot.press("enter")
        await pilot.pause()
    assert (
        path.read_text()
        == "*/20 * * * * /x/cronwatch run --job pinger --timeout 90 -- sh -c 'echo ping'\n"
    )


async def test_undo_restores_the_previous_crontab(app, parts):
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("space")
        await pilot.press("u")
        await pilot.pause()
        assert parts[0].read_text() == CRONTAB
        await pilot.press("u")
        await pilot.pause()
        assert "nothing to undo" in str(app.query_one("#status", Static).render())


async def test_an_external_edit_is_a_conflict_that_reload_resolves(app, parts):
    path = parts[0]
    async with app.run_test(size=(160, 50)) as pilot:
        path.write_text(CRONTAB + "5 5 * * * /someone/elses/job\n")
        await pilot.press("space")
        await pilot.pause()
        assert "changed since it was loaded" in str(
            app.query_one("#status", Static).render()
        )
        assert "#cronwatch:off#" not in path.read_text()  # nothing was overwritten
        await pilot.press("r")
        await pilot.pause()
        assert len(table_rows(app)) == 4
        await pilot.press("space")
        await pilot.pause()
    assert (
        "someone/elses/job" in path.read_text()
        and "#cronwatch:off# 0 2" in path.read_text()
    )


async def test_test_alert_goes_through_the_channels_and_reports(app, parts):
    channel = parts[3]
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("t")
        for _ in range(50):
            await pilot.pause()
            if "Test alert sent" in str(app.query_one("#status", Static).render()):
                break
            await asyncio.sleep(0.05)
        assert "Test alert sent via recording" in str(
            app.query_one("#status", Static).render()
        )
    assert [a.kind for a in channel.alerts] == ["test"]


async def test_test_alert_without_channels_says_so(parts):
    _, store, session, _ = parts
    app = CronWatchApp(
        session, store, Config(), UTC, Alerter([], store), WRAPPER, clock=lambda: NOW
    )
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("t")
        await pilot.pause()
        assert "No alert channels configured" in str(
            app.query_one("#status", Static).render()
        )


async def test_alerts_tab_shows_delivery_history(app, parts):
    parts[1].log_alert("pinger", None, "failed", "webhook", NOW, False, "HTTP 500")
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("4")
        await pilot.pause()
        table = app.query_one("#alert-log", DataTable)
        assert table.row_count == 1
        assert "HTTP 500" in str(table.get_row_at(0)[4])
        assert app.query_one("#tabs", TabbedContent).active == "alerts"


async def test_an_empty_crontab_still_renders(tmp_path):
    store = Store(tmp_path / "c.db")
    session = Session(FileCrontab(tmp_path / "empty"), tmp_path / "b")
    app = CronWatchApp(
        session, store, Config(), UTC, Alerter([], store), WRAPPER, clock=lambda: NOW
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("3")
        await pilot.pause()
        assert app.query_one("#jobs", DataTable).row_count == 0
        assert "Press" in str(app.query_one("#details-view", Static).render())
        await pilot.press(
            "e", "space", "w", "d"
        )  # every job action is a harmless no-op
        await pilot.pause()


def test_preview_text_covers_valid_invalid_and_percent_cases():
    text, valid = preview_text("0 3 * * *", "backup", True, NOW, UTC)
    assert valid and "At 03:00" in text and "Next: Sat 10 Oct 03:00" in text
    text, valid = preview_text("nonsense", "backup", True, NOW, UTC)
    assert not valid and "expected 5 fields" in text
    assert preview_text("0 3 * * *", "", True, NOW, UTC)[1] is False
    text, valid = preview_text("0 3 * * *", "date +%F", True, NOW, UTC)
    assert not valid and "unescaped %" in text
    assert (
        preview_text("0 3 * * *", "date +%F", False, NOW, UTC)[1] is True
    )  # fine when not wrapping
