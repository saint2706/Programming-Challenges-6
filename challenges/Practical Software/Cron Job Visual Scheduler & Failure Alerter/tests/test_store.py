import threading
from datetime import timedelta

import pytest
from cronwatch.store import FAILED, OK, RUNNING, JobState, Store
from helpers import T0, at


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data" / "cronwatch.db")


def record(
    store, job, minutes, exit_code=0, status=OK, command="cmd", duration=30, output=""
):
    run_id = store.start_run(job, command, "host1", at(minutes))
    store.finish_run(
        run_id, at(minutes) + timedelta(seconds=duration), exit_code, status, output
    )
    return run_id


def test_a_started_run_is_visible_as_running_until_finished(store):
    run_id = store.start_run("backup", "cmd", "h", T0)
    run = store.get_run(run_id)
    assert (
        run.status == RUNNING
        and run.ended_at is None
        and run.exit_code is None
        and not run.succeeded
    )
    store.finish_run(run_id, T0 + timedelta(seconds=90), 0, OK, "done\n")
    run = store.get_run(run_id)
    assert run.succeeded and run.exit_code == 0 and run.output_tail == "done\n"
    assert run.duration_s == pytest.approx(90, abs=0.01) and run.started_at == T0


def test_creates_missing_directories_and_reopens_existing_database(tmp_path):
    path = tmp_path / "a" / "b" / "c.db"
    first = Store(path)
    record(first, "j", 0)
    assert len(Store(path).runs()) == 1  # schema creation is idempotent and keeps data


def test_runs_are_newest_first_and_filterable(store):
    record(store, "a", 0)
    record(store, "b", 5)
    record(store, "a", 10, exit_code=1, status=FAILED)
    assert [(r.job_id, r.started_at) for r in store.runs()] == [
        ("a", at(10)),
        ("b", at(5)),
        ("a", at(0)),
    ]
    assert [r.exit_code for r in store.runs("a")] == [1, 0]
    assert len(store.runs(limit=2)) == 2
    assert [r.job_id for r in store.runs(since=at(5))] == ["a", "b"]


def test_last_runs_gives_one_latest_run_per_job(store):
    record(store, "a", 0)
    record(store, "a", 10, exit_code=2, status=FAILED)
    record(store, "b", 5)
    last = store.last_runs()
    assert set(last) == {"a", "b"} and last["a"].exit_code == 2 and last["b"].succeeded


def test_runs_between_is_half_open(store):
    for minute in (0, 5, 10):
        record(store, "a", minute)
    assert [r.started_at for r in store.runs_between("a", at(5), at(10))] == [at(5)]


def test_job_state_defaults_round_trips_and_updates(store):
    assert store.get_state("new") == JobState("new")
    state = JobState("j", 3, at(1), "failed", at(2))
    store.save_state(state)
    assert store.get_state("j") == state
    store.save_state(JobState("j", 0, None, "ok", at(9)))
    assert store.get_state("j") == JobState("j", 0, None, "ok", at(9))


def test_alert_log(store):
    store.log_alert("j", 7, "failed", "webhook", at(1), True, "")
    store.log_alert("j", 7, "failed", "email", at(1), False, "refused")
    first, second = store.alerts()
    assert (second.channel, second.ok) == ("webhook", True) and (
        first.channel,
        first.ok,
        first.detail,
    ) == ("email", False, "refused")


def test_prune_removes_old_runs_and_alerts_only(store):
    record(store, "a", -60 * 24 * 100)
    record(store, "a", 0)
    store.log_alert("a", None, "failed", "w", at(-60 * 24 * 100), True, "")
    assert store.prune(at(-60 * 24 * 90)) == 1
    assert len(store.runs()) == 1 and store.alerts() == []


def test_concurrent_writers_do_not_lose_runs_or_hit_database_locked(store):
    errors = []

    def worker(n):
        try:
            for i in range(15):
                record(store, f"job{n}", i)
        except Exception as exc:  # noqa: BLE001 - collected and asserted on below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.runs(limit=1000)) == 90
