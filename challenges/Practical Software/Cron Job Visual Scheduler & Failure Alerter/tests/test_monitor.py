from datetime import UTC, timedelta

import pytest
from cronwatch.alerts import Alerter
from cronwatch.crontab import Crontab
from cronwatch.monitor import check_missed, find_missed
from cronwatch.store import OK, Store
from helpers import at

GRACE = timedelta(minutes=5)
WRAP = "/x/cronwatch run --job {id} -- sh -c 'true'"


class Recording:
    name = "recording"

    def __init__(self):
        self.alerts = []

    def send(self, alert):
        self.alerts.append(alert)


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "m.db")
    channel = Recording()
    return store, channel, Alerter([channel], store)


def jobs(text):
    return Crontab.from_text(text).jobs()


def hourly(job_id="hourly", schedule="0 * * * *", prefix=""):
    return jobs(f"{prefix}{schedule} {WRAP.format(id=job_id)}\n")


def ran(store, job_id, when):
    run_id = store.start_run(job_id, "c", "h", when)
    store.finish_run(run_id, when + timedelta(seconds=5), 0, OK, "")


def hour(n):  # 2026-10-09 12:00 UTC + n hours
    return at(60 * n)


def test_first_sighting_starts_the_clock_and_never_alerts(env):
    store, channel, alerter = env
    assert check_missed(hourly(), store, alerter, hour(5), GRACE, UTC) == []
    assert store.get_state("hourly").monitor_cursor == hour(5)
    assert channel.alerts == []


def test_a_job_that_ran_every_time_is_not_missed(env):
    store, channel, alerter = env
    check_missed(hourly(), store, alerter, hour(0), GRACE, UTC)
    for n in range(1, 5):
        ran(store, "hourly", hour(n) + timedelta(seconds=2))
    assert (
        check_missed(
            hourly(), store, alerter, hour(4) + timedelta(minutes=10), GRACE, UTC
        )
        == []
    )
    assert channel.alerts == []


def test_a_skipped_run_is_reported_with_its_exact_scheduled_time_and_alerted_once(env):
    store, channel, alerter = env
    check_missed(hourly(), store, alerter, hour(0), GRACE, UTC)
    for n in (1, 3, 4):  # hour 2 never ran
        ran(store, "hourly", hour(n) + timedelta(seconds=1))
    now = hour(4) + timedelta(minutes=10)
    (missed,) = check_missed(hourly(), store, alerter, now, GRACE, UTC)
    assert missed.job_id == "hourly" and missed.instants == (hour(2),)
    (alert,) = channel.alerts
    assert (
        alert.kind == "missed"
        and alert.scheduled_for == hour(2)
        and alert.job_id == "hourly"
    )
    assert (
        check_missed(hourly(), store, alerter, now + timedelta(minutes=1), GRACE, UTC)
        == []
    )  # not re-reported
    assert len(channel.alerts) == 1


def test_instants_still_inside_the_grace_period_are_not_judged_yet(env):
    store, _, alerter = env
    check_missed(hourly(), store, alerter, hour(0), GRACE, UTC)
    assert (
        check_missed(
            hourly(), store, alerter, hour(1) + timedelta(minutes=3), GRACE, UTC
        )
        == []
    )
    assert (
        len(
            check_missed(
                hourly(), store, alerter, hour(1) + timedelta(minutes=6), GRACE, UTC
            )
        )
        == 1
    )


def test_a_run_that_started_a_little_late_still_counts(env):
    store, _, alerter = env
    check_missed(hourly(), store, alerter, hour(0), GRACE, UTC)
    ran(store, "hourly", hour(1) + timedelta(minutes=2, seconds=30))
    assert (
        check_missed(
            hourly(), store, alerter, hour(1) + timedelta(minutes=10), GRACE, UTC
        )
        == []
    )


def test_one_late_run_cannot_hide_a_separate_missed_one(env):
    store, _, alerter = env
    five = hourly("five", "*/5 * * * *")
    check_missed(five, store, alerter, at(0), GRACE, UTC)
    ran(store, "five", at(5))
    ran(
        store, "five", at(15)
    )  # :10 is missing; the :15 run must not be credited to :10
    (missed,) = check_missed(
        five, store, alerter, at(24), GRACE, UTC
    )  # judges :05, :10, :15
    assert missed.instants == (at(10),)


def test_a_long_outage_is_capped_at_a_day_and_produces_a_single_alert(env):
    store, channel, alerter = env
    check_missed(hourly(), store, alerter, hour(0), GRACE, UTC)
    (missed,) = check_missed(
        hourly(), store, alerter, hour(24 * 5), GRACE, UTC
    )  # five days later, nothing ran
    assert 22 <= len(missed.instants) <= 24
    assert len(channel.alerts) == 1 and channel.alerts[0].missed_count == len(
        missed.instants
    )


def test_disabled_jobs_are_ignored_and_do_not_build_up_a_backlog(env):
    store, channel, alerter = env
    enabled = hourly()
    disabled = hourly(prefix="#cronwatch:off# ")
    check_missed(enabled, store, alerter, hour(0), GRACE, UTC)
    check_missed(disabled, store, alerter, hour(48), GRACE, UTC)  # paused for two days
    assert channel.alerts == []
    assert (
        check_missed(
            enabled, store, alerter, hour(48) + timedelta(minutes=10), GRACE, UTC
        )
        == []
    )  # re-enabled: no storm
    assert store.get_state("hourly").monitor_cursor > hour(48)


def test_reboot_unmonitored_and_invalid_jobs_are_skipped(env):
    store, channel, alerter = env
    text = f"@reboot {WRAP.format(id='boot')}\n0 * * * * /bin/true\n61 * * * * {WRAP.format(id='bad')}\n"
    everything = jobs(text)
    check_missed(everything, store, alerter, hour(0), GRACE, UTC)
    assert check_missed(everything, store, alerter, hour(30), GRACE, UTC) == []
    assert channel.alerts == []


def test_notify_false_reports_without_alerting(env):
    store, channel, alerter = env
    check_missed(hourly(), store, alerter, hour(0), GRACE, UTC)
    assert (
        len(check_missed(hourly(), store, alerter, hour(3), GRACE, UTC, notify=False))
        == 1
    )
    assert channel.alerts == []


def test_find_missed_is_a_pure_query_over_the_history(env):
    store, _, _ = env
    job = hourly()[0]
    ran(store, "hourly", hour(1))
    assert find_missed(job, store, hour(0), hour(3), GRACE, UTC) == [hour(2)]
    assert find_missed(job, store, hour(0), hour(3), GRACE, UTC) == [
        hour(2)
    ]  # nothing was consumed


def test_runs_of_another_job_do_not_count(env):
    store, _, _ = env
    ran(store, "other", hour(1))
    assert find_missed(
        hourly()[0], store, hour(0), hour(2) + timedelta(minutes=10), GRACE, UTC
    ) == [hour(1), hour(2)]
