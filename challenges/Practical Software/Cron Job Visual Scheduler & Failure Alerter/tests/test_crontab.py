import subprocess
from datetime import datetime

import pytest
from cronwatch.crontab import (
    ConflictError,
    Crontab,
    CrontabError,
    FileCrontab,
    Session,
    SystemCrontab,
    has_unescaped_percent,
    suggest_job_id,
    unwrap_command,
    wrap_command,
    wrapper_job_id,
)

SAMPLE = """\
# My crontab -- hand curated since 2019
MAILTO=ops@example.com
PATH=/usr/local/bin:/usr/bin

# nightly database dump
0 2 * * * /usr/local/bin/pg_backup --all >> /var/log/pg.log 2>&1
*/5 * * * * curl -fsS https://example.com/ping
@daily /usr/bin/find /tmp -mtime +7 -delete
#cronwatch:off# 30 4 * * 1 /opt/weekly.sh
# 0 0 * * * commented_out_job
this is not a crontab line at all really
61 * * * * bad_schedule_job
"""


def test_round_trip_is_byte_identical_for_a_messy_real_world_crontab():
    assert Crontab.from_text(SAMPLE).to_text() == SAMPLE


def test_empty_text_and_missing_trailing_newline():
    assert Crontab.from_text("").to_text() == "" and Crontab.from_text("").lines == []
    assert (
        Crontab.from_text("0 0 * * * x").to_text() == "0 0 * * * x\n"
    )  # always newline-terminated


def test_line_classification():
    kinds = [line.kind for line in Crontab.from_text(SAMPLE).lines]
    # the prose line has six words, so it is job-shaped: it shows up as a job with an error, which is
    # right because cron itself would reject the whole crontab over it
    assert kinds == [
        "comment",
        "env",
        "env",
        "blank",
        "comment",
        "job",
        "job",
        "job",
        "job",
        "comment",
        "job",
        "job",
    ]
    assert Crontab.from_text("just words\n").lines[0].kind == "other"


def test_jobs_expose_schedule_command_state_and_description():
    jobs = Crontab.from_text(SAMPLE).jobs()
    backup, ping, daily, weekly, prose, bad = jobs
    assert prose.error and prose.expr is None
    assert backup.schedule == "0 2 * * *" and backup.command.startswith(
        "/usr/local/bin/pg_backup"
    )
    assert backup.description == "nightly database dump"
    assert ping.description is None and daily.schedule == "@daily"
    assert not weekly.enabled and weekly.command == "/opt/weekly.sh"
    assert bad.error and "minute" in bad.error and bad.expr is None


def test_commented_out_job_is_just_a_comment_not_a_disabled_job():
    assert all(
        "commented_out_job" not in j.command for j in Crontab.from_text(SAMPLE).jobs()
    )


def test_environment_lines():
    assert Crontab.from_text(SAMPLE).environment() == {
        "MAILTO": "ops@example.com",
        "PATH": "/usr/local/bin:/usr/bin",
    }


def test_label_prefers_job_id_then_description_then_command_name():
    jobs = Crontab.from_text(SAMPLE).jobs()
    assert jobs[0].label == "nightly database dump" and jobs[1].label == "curl"
    wrapped = Crontab.from_text(
        "* * * * * /x/cronwatch run --job nightly -- sh -c 'true'\n"
    ).jobs()[0]
    assert wrapped.label == "nightly" and wrapped.monitored


def test_edits_touch_only_the_edited_line():
    crontab = Crontab.from_text(SAMPLE)
    before = [line.raw for line in crontab.lines]
    index = crontab.jobs()[1].index
    crontab.update_job(index, "*/10 * * * *", "curl -fsS https://example.com/ping2")
    after = [line.raw for line in crontab.lines]
    assert after[index] == "*/10 * * * * curl -fsS https://example.com/ping2"
    assert [a for i, a in enumerate(after) if i != index] == [
        b for i, b in enumerate(before) if i != index
    ]


def test_toggle_disables_and_re_enables_reversibly():
    crontab = Crontab.from_text(SAMPLE)
    index = crontab.jobs()[0].index
    original = crontab.lines[index].raw
    crontab.set_enabled(index, False)
    assert (
        crontab.lines[index].raw == f"#cronwatch:off# {original}"
        and not crontab.job(index).enabled
    )
    crontab.set_enabled(index, True)
    assert crontab.lines[index].raw == original


def test_cannot_enable_a_job_with_an_invalid_schedule():
    crontab = Crontab.from_text("#cronwatch:off# 99 * * * * x\n")
    assert crontab.jobs()[0].error
    with pytest.raises(CrontabError, match="invalid schedule"):
        crontab.set_enabled(0, True)


def test_add_with_description_and_delete_removes_only_our_own_note():
    crontab = Crontab.from_text("# user comment\n0 1 * * * a\n")
    index = crontab.add_job("*/5 * * * *", "b --flag", "ping every five")
    assert crontab.to_text().endswith(
        "# cronwatch: ping every five\n*/5 * * * * b --flag\n"
    )
    assert crontab.job(index).description == "ping every five"
    crontab.delete_job(index)
    assert crontab.to_text() == "# user comment\n0 1 * * * a\n"
    crontab.delete_job(1)
    assert (
        crontab.to_text() == "# user comment\n"
    )  # the user's own comment is never deleted


@pytest.mark.parametrize(
    ("schedule", "command", "message"),
    [
        ("61 * * * *", "x", "minute"),
        ("* * * * *", "  ", "empty"),
        ("* * * * *", "a\nb", "single line"),
        ("* * *", "x", "5 fields"),
    ],
)
def test_invalid_edits_are_rejected_before_anything_changes(schedule, command, message):
    crontab = Crontab.from_text("0 0 * * * keep\n")
    with pytest.raises(CrontabError, match=message):
        crontab.add_job(schedule, command)
    assert crontab.to_text() == "0 0 * * * keep\n"


def test_edit_or_delete_of_a_non_job_line_is_an_error():
    crontab = Crontab.from_text("# comment\n0 0 * * * x\n")
    for operation in (
        lambda: crontab.delete_job(0),
        lambda: crontab.update_job(5, "* * * * *", "x"),
    ):
        with pytest.raises(CrontabError, match="no job"):
            operation()


def test_set_env_replaces_or_inserts_above_the_first_job():
    crontab = Crontab.from_text("MAILTO=a\n# c\n0 0 * * * x\n")
    crontab.set_env("MAILTO", "b")
    crontab.set_env("CRONWATCH_HOME", "/home/me/my data")
    assert (
        crontab.to_text()
        == "MAILTO=b\n# c\nCRONWATCH_HOME='/home/me/my data'\n0 0 * * * x\n"
    )


WRAPPER = "/home/me/.venv/bin/cronwatch"


def test_wrap_keeps_pipes_redirects_and_quotes_inside_the_job():
    original = "cd /srv && ./dump.sh | gzip > \"/b/$(date +\\%F).gz\" 2>&1; echo 'done'"
    wrapped = wrap_command(original, "dump", WRAPPER, timeout=300)
    assert wrapped.startswith(f"{WRAPPER} run --job dump --timeout 300 -- sh -c ")
    assert unwrap_command(wrapped) == original
    assert wrapper_job_id(wrapped) == "dump"


def test_wrap_then_unwrap_round_trips_through_the_crontab():
    crontab = Crontab.from_text("0 2 * * * /usr/bin/backup --full\n")
    crontab.wrap_job(0, "backup", WRAPPER)
    assert crontab.job(0).job_id == "backup" and "sh -c" in crontab.job(0).command
    crontab.unwrap_job(0)
    assert crontab.to_text() == "0 2 * * * /usr/bin/backup --full\n"


def test_wrap_preserves_the_disabled_state():
    crontab = Crontab.from_text("#cronwatch:off# 0 2 * * * job\n")
    crontab.wrap_job(0, "j", WRAPPER)
    assert not crontab.job(0).enabled and crontab.lines[0].raw.startswith(
        "#cronwatch:off# 0 2 * * * "
    )


def test_wrapper_with_spaces_in_its_path_still_unwraps():
    wrapper = "'/home/my user/bin/cronwatch'"
    assert unwrap_command(wrap_command("echo hi", "x", wrapper)) == "echo hi"
    assert (
        wrapper_job_id(wrap_command("echo hi", "x", "/usr/bin/python3 -m cronwatch"))
        == "x"
    )


def test_unescaped_percent_is_refused_but_escaped_is_fine():
    assert has_unescaped_percent("date +%F") and not has_unescaped_percent("date +\\%F")
    with pytest.raises(CrontabError, match="unescaped %"):
        wrap_command("date +%F", "d", WRAPPER)
    wrap_command("date +\\%F", "d", WRAPPER)


def test_wrap_rejects_duplicate_ids_bad_ids_and_double_wrapping():
    crontab = Crontab.from_text("0 0 * * * a\n1 0 * * * b\n")
    crontab.wrap_job(0, "same", WRAPPER)
    with pytest.raises(CrontabError, match="already used"):
        crontab.wrap_job(1, "same", WRAPPER)
    with pytest.raises(CrontabError, match="only letters"):
        crontab.wrap_job(1, "bad id!", WRAPPER)
    with pytest.raises(CrontabError, match="already wrapped"):
        crontab.wrap_job(0, "other", WRAPPER)


def test_unwrap_of_a_hand_edited_or_unwrapped_job_is_refused():
    with pytest.raises(CrontabError, match="not wrapped"):
        unwrap_command("/usr/bin/backup")
    with pytest.raises(CrontabError, match="trailing text"):
        unwrap_command(f"{WRAPPER} run --job x -- sh -c 'a' 'b'")


def test_suggest_job_id_slugs_and_deduplicates():
    assert suggest_job_id("/usr/local/bin/backup.sh --all", set()) == "backup"
    assert suggest_job_id("/usr/local/bin/backup.sh", {"backup"}) == "backup-2"
    assert suggest_job_id("", set()) == "job"
    assert suggest_job_id("(weird) thing", set()) == "weird"


# --- backends and the safe-write session ---


def test_file_backend_round_trip_and_missing_file(tmp_path):
    backend = FileCrontab(tmp_path / "sub" / "crontab")
    assert backend.read() == ""
    backend.write("0 0 * * * x\n")
    assert backend.read() == "0 0 * * * x\n"
    assert [p.name for p in (tmp_path / "sub").iterdir()] == [
        "crontab"
    ]  # no temp files left behind


def fake_run(stdout="", stderr="", returncode=0):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    run.calls = calls
    return run


def test_system_backend_commands_and_user_flag():
    run = fake_run(stdout="0 0 * * * x\n")
    assert SystemCrontab(run=run).read() == "0 0 * * * x\n"
    assert run.calls[0][0] == ["crontab", "-l"]
    SystemCrontab("deploy", run=run).write("new\n")
    command, kwargs = run.calls[1]
    assert command == ["crontab", "-u", "deploy", "-"] and kwargs["input"] == "new\n"


def test_system_backend_treats_no_crontab_as_empty_and_other_errors_as_errors():
    assert (
        SystemCrontab(run=fake_run(returncode=1, stderr="no crontab for alice")).read()
        == ""
    )
    with pytest.raises(CrontabError, match="crontab -l failed"):
        SystemCrontab(run=fake_run(returncode=1, stderr="permission denied")).read()
    with pytest.raises(CrontabError, match="old crontab is unchanged"):
        SystemCrontab(run=fake_run(returncode=1, stderr='"-":3: bad minute')).write("x")


def test_system_backend_missing_binary_and_timeout():
    def missing(*_a, **_k):
        raise FileNotFoundError

    def slow(*_a, **_k):
        raise subprocess.TimeoutExpired("crontab", 30)

    with pytest.raises(CrontabError, match="not found"):
        SystemCrontab(run=missing).read()
    with pytest.raises(CrontabError, match="timed out"):
        SystemCrontab(run=slow).read()


@pytest.fixture
def session(tmp_path):
    backend = FileCrontab(tmp_path / "crontab")
    backend.write("0 0 * * * keep\n")
    ticks = iter(range(1000))
    clock = lambda: datetime(2026, 1, 1, 0, 0, next(ticks) % 60, next(ticks))  # noqa: DTZ001
    return Session(backend, tmp_path / "backups", clock)


def test_apply_writes_backs_up_the_previous_text_and_updates_state(session, tmp_path):
    session.apply(lambda c: c.add_job("* * * * *", "new"))
    assert (tmp_path / "crontab").read_text() == "0 0 * * * keep\n* * * * * new\n"
    backups = list((tmp_path / "backups").iterdir())
    assert len(backups) == 1 and backups[0].read_text() == "0 0 * * * keep\n"
    assert [j.command for j in session.crontab.jobs()] == ["keep", "new"]


def test_a_failing_mutation_leaves_everything_untouched(session, tmp_path):
    with pytest.raises(CrontabError):
        session.apply(lambda c: c.add_job("nonsense", "x"))
    assert (tmp_path / "crontab").read_text() == "0 0 * * * keep\n"
    assert (
        not (tmp_path / "backups").exists()
        and session.crontab.to_text() == "0 0 * * * keep\n"
    )


def test_external_edit_between_load_and_save_is_a_conflict_not_an_overwrite(
    session, tmp_path
):
    (tmp_path / "crontab").write_text("0 0 * * * keep\n5 5 * * * someone-elses-job\n")
    with pytest.raises(ConflictError, match="reload"):
        session.apply(lambda c: c.add_job("* * * * *", "mine"))
    assert (
        "someone-elses-job" in (tmp_path / "crontab").read_text()
        and "mine" not in (tmp_path / "crontab").read_text()
    )
    session.reload()
    session.apply(lambda c: c.add_job("* * * * *", "mine"))
    assert "someone-elses-job" in session.loaded_text and "mine" in session.loaded_text


def test_undo_restores_each_previous_state_in_order(session, tmp_path):
    session.apply(lambda c: c.add_job("1 * * * *", "one"))
    session.apply(lambda c: c.add_job("2 * * * *", "two"))
    assert session.can_undo
    session.undo()
    assert (tmp_path / "crontab").read_text() == "0 0 * * * keep\n1 * * * * one\n"
    session.undo()
    assert (tmp_path / "crontab").read_text() == "0 0 * * * keep\n"
    assert not session.can_undo
    with pytest.raises(CrontabError, match="nothing to undo"):
        session.undo()


def test_undo_also_refuses_to_clobber_an_external_edit(session, tmp_path):
    session.apply(lambda c: c.add_job("1 * * * *", "one"))
    (tmp_path / "crontab").write_text("external\n")
    with pytest.raises(ConflictError):
        session.undo()


def test_backups_are_pruned_to_a_bounded_number(session, tmp_path):
    for i in range(60):
        session.apply(lambda c, i=i: c.add_job("* * * * *", f"job{i}"))
    assert len(list((tmp_path / "backups").iterdir())) == 50
