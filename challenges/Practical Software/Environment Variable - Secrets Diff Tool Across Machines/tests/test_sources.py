import io
import subprocess
from pathlib import Path

import pytest
from envdiff.snapshot import snapshot_to_dict
from envdiff.sources import SourceError, SourceSpec, fetch_ssh, load_sources, parse_spec


def never(_):
    return False


def test_plain_paths_and_labels():
    assert parse_spec("a/.env", never) == SourceSpec("file", "a/.env", None, "a/.env")
    spec = parse_spec("prod=a/.env", never)
    assert (spec.kind, spec.label, spec.path) == ("file", "prod", "a/.env")


def test_equals_sign_in_a_real_filename_is_not_a_label():
    assert parse_spec("a=b.env", lambda p: p == "a=b.env").label is None


def test_ssh_specs_with_and_without_user_and_label():
    spec = parse_spec("deploy@prod.example.com:/srv/app/.env", never)
    assert (spec.kind, spec.destination, spec.path) == (
        "ssh",
        "deploy@prod.example.com",
        "/srv/app/.env",
    )
    assert parse_spec("web1:.env", never).destination == "web1"
    labelled = parse_spec("prod=deploy@web1:/srv/.env", never)
    assert (labelled.label, labelled.kind, labelled.destination) == (
        "prod",
        "ssh",
        "deploy@web1",
    )
    assert labelled.default_label == "web1"


def test_ssh_ipv6_host():
    assert parse_spec("[2001:db8::1]:/srv/.env", never).destination == "[2001:db8::1]"


def test_windows_drive_letters_and_existing_local_files_are_not_ssh():
    assert parse_spec("C:\\proj\\.env", never).kind == "file"
    assert parse_spec("web1:.env", lambda p: p == "web1:.env").kind == "file"


def test_dash_leading_host_cannot_smuggle_an_ssh_option():
    spec = parse_spec("-oProxyCommand=evil:/x", never)
    assert spec.kind == "file"


def test_stdin_and_env_specs():
    assert parse_spec("-").kind == "stdin"
    assert parse_spec("laptop=env:").kind == "env"


class Recorder:
    def __init__(self, stdout=b"A=1\n", returncode=0, stderr=b""):
        self.calls = []
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return self.result


def test_ssh_command_is_batch_mode_option_terminated_and_quotes_the_path():
    spec = parse_spec("deploy@web1:/srv/my app/it's;rm -rf x/.env", never)
    run = Recorder()
    assert fetch_ssh(spec, run) == "A=1\n"
    command, kwargs = run.calls[0]
    assert command[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    assert command[5:7] == ["--", "deploy@web1"]
    assert command[7] == "cat -- '/srv/my app/it'\"'\"'s;rm -rf x/.env'"
    assert kwargs["timeout"] == 30 and kwargs["capture_output"]


def test_ssh_failure_reports_the_last_stderr_line_and_no_content():
    run = Recorder(
        stdout=b"leaked=content",
        returncode=255,
        stderr=b"warn\nPermission denied (publickey).\n",
    )
    with pytest.raises(SourceError) as error:
        fetch_ssh(parse_spec("web1:/x", never), run)
    assert "Permission denied" in str(error.value) and "leaked" not in str(error.value)


def test_ssh_missing_binary_and_timeout_are_source_errors():
    def missing(*_a, **_k):
        raise FileNotFoundError

    def slow(*_a, **_k):
        raise subprocess.TimeoutExpired("ssh", 30)

    spec = parse_spec("web1:/x", never)
    with pytest.raises(SourceError, match="not installed"):
        fetch_ssh(spec, missing)
    with pytest.raises(SourceError, match="timed out"):
        fetch_ssh(spec, slow)


def test_binary_and_oversized_data_are_rejected():
    with pytest.raises(SourceError, match="binary"):
        fetch_ssh(parse_spec("web1:/x", never), Recorder(stdout=b"A=1\0\x01\x02"))
    with pytest.raises(SourceError, match="larger than"):
        fetch_ssh(
            parse_spec("web1:/x", never), Recorder(stdout=b"A" * (5 * 1024 * 1024 + 1))
        )


def write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_files_share_one_hasher_so_equal_values_get_equal_fingerprints(tmp_path):
    a, b = write(tmp_path / "a.env", "K=v\n"), write(tmp_path / "b.env", "K=v\n")
    one, two = load_sources([parse_spec(a), parse_spec(b)])
    assert one.entries["K"].fingerprints == two.entries["K"].fingerprints


def test_separate_runs_use_different_random_keys(tmp_path):
    path = write(tmp_path / "a.env", "K=v\n")
    first = load_sources([parse_spec(path)])[0]
    second = load_sources([parse_spec(path)])[0]
    assert first.entries["K"].fingerprints != second.entries["K"].fingerprints


def test_missing_file_is_a_clean_error(tmp_path):
    with pytest.raises(SourceError, match="No such file"):
        load_sources([parse_spec(str(tmp_path / "nope.env"))])


def test_duplicate_labels_are_made_unique(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    a, b = (
        write(tmp_path / "x" / ".env", "A=1\n"),
        write(tmp_path / "y" / ".env", "A=1\n"),
    )
    assert [s.label for s in load_sources([parse_spec(a), parse_spec(b)])] == [
        ".env",
        ".env#2",
    ]


def test_labels_override_defaults(tmp_path):
    a = write(tmp_path / "a.env", "A=1\n")
    assert load_sources([parse_spec(f"prod={a}")])[0].label == "prod"


def test_stdin_source():
    (source,) = load_sources([parse_spec("-")], stdin=io.BytesIO(b"A=1\n"))
    assert source.label == "stdin" and "A" in source.entries


def test_env_source_reads_the_given_environment_without_parsing():
    (source,) = load_sources([parse_spec("env:")], environ={"HOME": "/h", "TOKEN": "t"})
    assert set(source.entries) == {"HOME", "TOKEN"} and source.fmt == "environment"


def test_format_override_applies(tmp_path):
    path = write(tmp_path / "a.env", 'K="v"\n')
    dotenv, raw = (
        load_sources([parse_spec(path)], fmt)[0] for fmt in ["dotenv", "raw"]
    )
    assert dotenv.entries["K"].fingerprints != raw.entries["K"].fingerprints


def test_reveal_keeps_only_requested_values(tmp_path):
    path = write(tmp_path / "a.env", "SHOW=visible\nHIDE=hidden\n")
    (source,) = load_sources([parse_spec(path)], reveal=frozenset({"SHOW"}))
    assert source.revealed == {"SHOW": "visible"}


def make_snapshot(tmp_path, name, text, passphrase="pw"):
    import json

    (source,) = load_sources(
        [parse_spec(write(tmp_path / f"{name}.env", text))], passphrase=passphrase
    )
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(snapshot_to_dict(source)))
    return str(path)


def test_snapshot_compares_with_live_file_under_the_same_passphrase(tmp_path):
    snap = make_snapshot(tmp_path, "host", "K=v\nOTHER=1\n")
    live = write(tmp_path / "live.env", "K=v\nOTHER=2\n")
    a, b = load_sources([parse_spec(snap), parse_spec(live)], passphrase="pw")
    assert a.entries["K"].fingerprints == b.entries["K"].fingerprints
    assert a.entries["OTHER"].fingerprints != b.entries["OTHER"].fingerprints


def test_snapshot_with_live_file_requires_a_passphrase(tmp_path):
    snap = make_snapshot(tmp_path, "host", "K=v\n")
    live = write(tmp_path / "live.env", "K=v\n")
    with pytest.raises(SourceError, match="ENVDIFF_PASSPHRASE"):
        load_sources([parse_spec(snap), parse_spec(live)])


def test_snapshot_made_with_another_passphrase_is_rejected(tmp_path):
    snap = make_snapshot(tmp_path, "host", "K=v\n", passphrase="one")
    live = write(tmp_path / "live.env", "K=v\n")
    with pytest.raises(SourceError, match="different passphrase"):
        load_sources([parse_spec(snap), parse_spec(live)], passphrase="two")


def test_two_snapshots_need_matching_keys_but_no_passphrase(tmp_path):
    one = make_snapshot(tmp_path, "a", "K=v\n", "pw")
    two = make_snapshot(tmp_path, "b", "K=v\n", "pw")
    sources = load_sources([parse_spec(one), parse_spec(two)])
    assert [s.fmt for s in sources] == ["snapshot", "snapshot"]
    other = make_snapshot(tmp_path, "c", "K=v\n", "different")
    with pytest.raises(SourceError, match="different passphrase"):
        load_sources([parse_spec(one), parse_spec(other)])


def test_corrupt_snapshot_is_a_clean_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"envdiff_snapshot": 1, "entries": {"K": {"fp": [1]}}}')
    with pytest.raises(SourceError, match="not a valid envdiff snapshot"):
        load_sources([parse_spec(str(bad))])
