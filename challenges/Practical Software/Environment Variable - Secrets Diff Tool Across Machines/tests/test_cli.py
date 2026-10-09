import json
import os
import stat

import pytest
from envdiff.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def run(*args, env=None, input=None):
    return runner.invoke(app, [str(a) for a in args], env=env, input=input)


@pytest.fixture
def envs(tmp_path):
    (tmp_path / "prod.env").write_text(
        "APP_ENV=production\nPORT=80\nSECRET_KEY=sk-prod-1\nONLY_PROD=1\n"
    )
    (tmp_path / "stg.env").write_text(
        "APP_ENV=staging\nPORT=80\nSECRET_KEY=sk-prod-1\n"
    )
    return tmp_path


def test_identical_files_exit_zero_and_say_so(tmp_path):
    (tmp_path / "a.env").write_text("A=1\n")
    (tmp_path / "b.env").write_text("A=1\n")
    result = run("diff", tmp_path / "a.env", tmp_path / "b.env")
    assert result.exit_code == 0 and "No differences" in result.output


def test_differences_exit_one_and_list_only_differing_variables(envs):
    result = run("diff", envs / "prod.env", envs / "stg.env")
    assert result.exit_code == 1
    assert "APP_ENV" in result.output and "ONLY_PROD" in result.output
    assert (
        "PORT" not in result.output.split("Findings")[0].split("VARIABLE")[1]
    )  # same everywhere: hidden


def test_all_flag_lists_identical_variables_too(envs):
    result = run("diff", envs / "prod.env", envs / "stg.env", "--all")
    assert "PORT" in result.output


def test_labels_ignore_and_missing_reporting(envs):
    result = run(
        "diff", f"prod={envs / 'prod.env'}", f"stg={envs / 'stg.env'}", "-i", "APP_*"
    )
    assert (
        "APP_ENV" not in result.output
        and "missing in stg" in result.output
        and "1 ignored" in result.output
    )


def test_json_output_is_machine_readable_and_complete(envs):
    result = run("diff", envs / "prod.env", envs / "stg.env", "--as", "json")
    data = json.loads(result.output)
    assert data["summary"] == {
        "variables": 4,
        "same": 2,
        "differs": 1,
        "missing": 1,
        "ignored": 0,
    }
    keys = {k["key"]: k for k in data["keys"]}
    assert (
        keys["ONLY_PROD"]["absent"] == ["stg.env"] and keys["PORT"]["status"] == "same"
    )
    assert "fingerprints" not in keys["PORT"]
    assert any(f["code"] == "shared-secret" for f in data["findings"])


def test_fingerprints_flag_adds_short_prints_that_match_exactly_when_values_match(envs):
    data = json.loads(
        run(
            "diff",
            envs / "prod.env",
            envs / "stg.env",
            "--as",
            "json",
            "--fingerprints",
        ).output
    )
    secret = next(k for k in data["keys"] if k["key"] == "SECRET_KEY")["fingerprints"]
    app_env = next(k for k in data["keys"] if k["key"] == "APP_ENV")["fingerprints"]
    assert secret["prod.env"] == secret["stg.env"] and len(secret["prod.env"]) == 8
    assert app_env["prod.env"] != app_env["stg.env"]


def test_markdown_output_has_a_table(envs):
    result = run("diff", envs / "prod.env", envs / "stg.env", "--as", "markdown")
    assert "| Variable | prod.env | stg.env | Verdict |" in result.output


def test_strict_turns_warnings_into_a_failure(tmp_path):
    (tmp_path / "a.env").write_text("API_TOKEN=\n")
    assert run("diff", tmp_path / "a.env").exit_code == 0
    assert run("diff", tmp_path / "a.env", "--strict").exit_code == 1
    assert run("diff", tmp_path / "a.env", "--strict", "--no-lint").exit_code == 0


def test_missing_file_is_exit_two_with_a_message(tmp_path):
    result = run("diff", tmp_path / "nope.env")
    assert (
        result.exit_code == 2
        and "error:" in result.output
        and "No such file" in result.output
    )


def test_stdin_can_be_one_source_but_not_two(tmp_path):
    (tmp_path / "a.env").write_text("A=1\n")
    assert run("diff", tmp_path / "a.env", "-", input="A=1\n").exit_code == 0
    assert run("diff", "-", "-", input="A=1\n").exit_code == 2


def test_env_spec_compares_against_the_process_environment(tmp_path, monkeypatch):
    (tmp_path / "a.env").write_text("ENVDIFF_TEST_VAR=1\n")
    monkeypatch.setenv("ENVDIFF_TEST_VAR", "1")
    result = run("diff", tmp_path / "a.env", "env:", "-i", "*", "--all")
    assert result.exit_code == 0
    result = run("diff", tmp_path / "a.env", "env:", "--as", "json")
    row = next(
        k for k in json.loads(result.output)["keys"] if k["key"] == "ENVDIFF_TEST_VAR"
    )
    assert row["status"] == "same"


def test_reveal_prints_only_the_requested_variable_and_warns_for_secrets(envs):
    result = run(
        "diff",
        envs / "prod.env",
        envs / "stg.env",
        "--reveal",
        "APP_ENV",
        "--reveal",
        "SECRET_KEY",
    )
    assert "'production'" in result.output and "'staging'" in result.output
    assert "warning: revealing SECRET_KEY" in result.output
    assert "PORT = " not in result.output


def test_reveal_of_an_unknown_variable_warns(envs):
    result = run("diff", envs / "prod.env", "--reveal", "NOPE")
    assert "no source defines it" in result.output


def test_format_option_changes_how_quotes_are_read(tmp_path):
    (tmp_path / "a.env").write_text('K="v"\n')
    (tmp_path / "b.env").write_text("K=v\n")
    assert run("diff", tmp_path / "a.env", tmp_path / "b.env").exit_code == 0
    result = run("diff", tmp_path / "a.env", tmp_path / "b.env", "--format", "raw")
    assert result.exit_code == 1 and "only quoting differs" in result.output


def test_snapshot_workflow_end_to_end(envs, tmp_path):
    env = {"ENVDIFF_PASSPHRASE": "a long shared passphrase"}
    out = tmp_path / "prod.json"
    made = run("snapshot", envs / "prod.env", "-o", out, "--label", "prod", env=env)
    assert made.exit_code == 0 and out.exists()
    data = json.loads(out.read_text())
    assert data["label"] == "prod" and "sk-prod-1" not in out.read_text()
    result = run("diff", out, envs / "stg.env", "--as", "json", env=env)
    assert result.exit_code == 1
    assert {k["key"]: k["status"] for k in json.loads(result.output)["keys"]}[
        "APP_ENV"
    ] == "differs"


def test_snapshot_to_stdout_and_passphrase_file(envs, tmp_path):
    (tmp_path / "pw").write_text("from a file\nignored second line\n")
    result = run("snapshot", envs / "prod.env", "--passphrase-file", tmp_path / "pw")
    assert result.exit_code == 0 and json.loads(result.output)["envdiff_snapshot"] == 1


def test_snapshot_requires_a_passphrase(envs):
    result = run("snapshot", envs / "prod.env")
    assert result.exit_code == 2 and "ENVDIFF_PASSPHRASE" in result.output


def test_snapshot_refuses_a_snapshot(envs, tmp_path):
    env = {"ENVDIFF_PASSPHRASE": "pw"}
    snap = tmp_path / "s.json"
    run("snapshot", envs / "prod.env", "-o", snap, env=env)
    result = run("snapshot", snap, env=env)
    assert result.exit_code == 2 and "already a snapshot" in result.output


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """An `ssh` that serves files from a directory standing in for the remote host."""
    remote = tmp_path / "remote"
    remote.mkdir()
    log = tmp_path / "ssh.log"
    script = tmp_path / "bin" / "ssh"
    script.parent.mkdir()
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        'while [ "$1" != "--" ]; do shift; done; shift\n'
        'host="$1"; shift\n'
        f'cd "{remote}" || exit 1\n'
        'case "$host" in *down*) echo "ssh: connect to host $host port 22: Connection refused" >&2; exit 255;; esac\n'
        'exec sh -c "$*"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    return remote, log


def test_ssh_sources_end_to_end_through_the_real_subprocess_path(fake_ssh, tmp_path):
    remote, log = fake_ssh
    (remote / "app.env").write_text("APP_ENV=production\nPORT=80\n")
    (tmp_path / "local.env").write_text("APP_ENV=development\nPORT=80\n")
    result = run(
        "diff", "prod=deploy@web1:app.env", tmp_path / "local.env", "--as", "json"
    )
    data = json.loads(result.output)
    assert result.exit_code == 1
    assert [s["label"] for s in data["sources"]] == ["prod", "local.env"]
    assert {k["key"]: k["status"] for k in data["keys"]} == {
        "APP_ENV": "differs",
        "PORT": "same",
    }
    logged = log.read_text()
    assert "BatchMode=yes" in logged and "deploy@web1" in logged


def test_ssh_failure_is_exit_two_and_names_the_source(fake_ssh):
    result = run("diff", "web-down:/etc/app.env")
    assert (
        result.exit_code == 2
        and "Connection refused" in result.output
        and "web-down:/etc/app.env" in result.output
    )


def test_ssh_remote_file_missing_reports_cat_error(fake_ssh):
    result = run("diff", "web1:nonexistent.env")
    assert result.exit_code == 2 and "nonexistent.env" in result.output
