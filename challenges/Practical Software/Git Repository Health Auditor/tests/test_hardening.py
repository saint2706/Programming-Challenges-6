"""Regression tests for the independent security review (items A1 to A8 and the hardening nits).

Each test names the review item it guards. Timing bounds are generous: they catch quadratic
behaviour on adversarial input, not small slowdowns.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import gitaudit.safety as safety_module
import pytest
from conftest import requires_git
from gitaudit import audit as audit_module
from gitaudit import cli as cli_module
from gitaudit import git as git_module
from gitaudit import hygiene
from gitaudit import secrets as secrets_module
from gitaudit.audit import run_audit
from gitaudit.cli import main
from gitaudit.config import Settings, discover_config, parse_config
from gitaudit.findings import Severity
from gitaudit.git import GitError, GitRepo, TreeEntry
from gitaudit.reporters import render_sarif
from gitaudit.safety import atomic_write, cli_arg, redact, sanitize
from gitaudit.secrets import Allowlist, content_finding, scan_text
from helpers import _CLEAN_ENV, GIT, fake_aws_key_id
from helpers import git as run_git

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


# ----- A1: the signing, editor and credential programs are pinned on the command line ------


@requires_git
def test_argv_pins_programs_before_the_subcommand(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    args = GitRepo(repo.path)._argv(["log"])
    pins = (
        "gpg.program=gpg",
        "gpg.openpgp.program=gpg",
        "gpg.ssh.program=ssh-keygen",
        "gpg.x509.program=gpgsm",
        "gpg.format=openpgp",
        "core.sshCommand=ssh",
        "core.editor=true",
        "sequence.editor=true",
        "credential.helper=",
        "i18n.logOutputEncoding=UTF-8",
    )
    for pin in pins:
        assert pin in args, pin
        assert args.index(pin) < args.index("log"), pin


@requires_git
@pytest.mark.skipif(
    os.name == "nt", reason="needs a POSIX shell script as the gpg program"
)
def test_repository_config_cannot_choose_a_gpg_program(repo_factory, tmp_path):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    marker = tmp_path / "ran"
    script = tmp_path / "evil-gpg"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\n", encoding="utf-8")
    script.chmod(0o755)
    with (repo.path / ".git" / "config").open("a", encoding="utf-8") as fh:
        fh.write(f'[gpg]\n\tprogram = {script}\n[gpg "ssh"]\n\tprogram = {script}\n')
    run_audit(Settings(repo=repo.path, check_signatures=True, check_identities=True))
    assert not marker.exists()


# ----- A2: hostile author and subject text cannot shift the NUL-separated fields -----------


def _raw_commit(repo: Path, author: bytes, subject: bytes) -> str:
    body = (
        b"tree " + EMPTY_TREE.encode() + b"\nauthor " + author + b" 1700000000 +0000\n"
        b"committer Test <t@example.org> 1700000000 +0000\n\n" + subject + b"\n"
    )
    proc = subprocess.run(
        [GIT, "-C", str(repo), "hash-object", "-t", "commit", "-w", "--stdin"],
        input=body,
        capture_output=True,
        env=_CLEAN_ENV,
        check=True,
    )
    sha = proc.stdout.decode("ascii").strip()
    run_git(repo, "update-ref", "refs/heads/main", sha)
    return sha


@requires_git
@pytest.mark.parametrize("junk", ["\x0c", "\x1f", "\x85", "\u2028"])
def test_hostile_author_and_subject_keep_their_fields(repo_factory, junk):
    repo = repo_factory()
    author = f"Eve{junk}good <evil@corp.example>".encode()
    subject = f"fix{junk}the thing".encode()
    _raw_commit(repo.path, author, subject)
    entries = list(GitRepo(repo.path).commits())
    assert len(entries) == 1
    entry = entries[0]
    assert entry["author"] == f"Eve{junk}good"
    assert entry["email"] == "evil@corp.example"
    assert entry["subject"] == f"fix{junk}the thing"


def test_sanitize_escapes_unicode_line_and_paragraph_separators():
    assert sanitize("a\u2028b\u2029c") == "a\\u2028b\\u2029c"


def test_cli_arg_rejects_a_leading_at_sign():
    assert cli_arg("@evil", "<x>") == "<x>"


def test_commit_log_disables_external_diff_and_textconv(repo_factory, monkeypatch):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    seen: dict[str, list[str]] = {}

    def fake_records(self, args, sep=b"\0"):
        seen["args"] = list(args)
        return iter(())

    monkeypatch.setattr(GitRepo, "records", fake_records)
    list(GitRepo(repo.path).commits())
    assert "--no-ext-diff" in seen["args"]
    assert "--no-textconv" in seen["args"]


# ----- A3: records() is linear, capped, and a timeout is an error ---------------------------


class _FakeProc:
    def __init__(self, data: bytes) -> None:
        self.stdout = io.BytesIO(data)


def _fake_repo(monkeypatch, data: bytes) -> GitRepo:
    repo = GitRepo.__new__(GitRepo)

    @contextmanager
    def fake_stream(args, feed=None):
        yield _FakeProc(data)

    monkeypatch.setattr(repo, "_stream", fake_stream)
    return repo


def test_records_splits_across_read_chunks(monkeypatch):
    data = b"a\0bb\0" + b"c" * 70000 + b"\0" + b"d"
    repo = _fake_repo(monkeypatch, data)
    assert list(repo.records(["x"])) == [b"a", b"bb", b"c" * 70000, b"d"]


def test_records_refuses_one_record_over_the_cap(monkeypatch):
    monkeypatch.setattr(git_module, "MAX_RECORD", 1000)
    repo = _fake_repo(monkeypatch, b"x" * 1500 + b"\0")
    with pytest.raises(GitError, match="record over"):
        list(repo.records(["log"]))


def test_records_cap_applies_before_a_separator_arrives(monkeypatch):
    monkeypatch.setattr(git_module, "MAX_RECORD", 1000)
    repo = _fake_repo(monkeypatch, b"x" * 1500)
    with pytest.raises(GitError, match="record over"):
        list(repo.records(["log"]))


@requires_git
def test_stream_timeout_raises_instead_of_ending_quietly(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    handle = GitRepo(repo.path)
    handle.timeout = 0.001  # the timer fires before git can finish
    with pytest.raises(GitError, match="timed out"):
        list(handle.records(["rev-list", "--all"]))


# ----- A7: a real nonzero exit is an error; stopping early is not --------------------------


@requires_git
def test_nonzero_exit_after_full_read_raises(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    with pytest.raises(GitError, match="exited"):
        list(GitRepo(repo.path).records(["log", "--format=%H", "no-such-ref-zz"]))


@requires_git
def test_stopping_early_is_not_an_error(repo_factory):
    repo = repo_factory()
    for n in range(3):
        repo.write("a.txt", f"v{n}")
        repo.commit(f"c{n}")
    gen = GitRepo(repo.path).records(["rev-list", "HEAD"], sep=b"\n")
    first = next(gen)
    gen.close()
    assert len(first) == 40


# ----- A8: gitfile, commondir, alternates and [include] are refused before git runs --------


@requires_git
def test_gitfile_pointing_at_a_unc_path_is_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text("gitdir: //server/share/repo\n", encoding="utf-8")
    with pytest.raises(GitError, match="network path"):
        GitRepo(work)


@requires_git
def test_commondir_on_a_unc_path_is_refused(tmp_path):
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    (work / ".git" / "commondir").write_text(
        "//server/share/common\n", encoding="utf-8"
    )
    with pytest.raises(GitError, match="network path"):
        GitRepo(work)


@requires_git
def test_alternates_outside_the_repository_are_refused(repo_factory, tmp_path):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    other = tmp_path / "elsewhere" / "objects"
    other.mkdir(parents=True)
    (repo.path / ".git" / "objects" / "info" / "alternates").write_text(
        f"{other}\n", encoding="utf-8"
    )
    with pytest.raises(GitError, match="outside the repository"):
        GitRepo(repo.path)


@requires_git
def test_unc_alternate_is_refused(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    (repo.path / ".git" / "objects" / "info" / "alternates").write_text(
        "//server/share/objects\n", encoding="utf-8"
    )
    with pytest.raises(GitError, match="network path"):
        GitRepo(repo.path)


@requires_git
def test_config_with_include_is_refused(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    with (repo.path / ".git" / "config").open("a", encoding="utf-8") as fh:
        fh.write("[include]\n\tpath = /tmp/elsewhere.cfg\n")
    with pytest.raises(GitError, match="include"):
        GitRepo(repo.path)


@requires_git
def test_cli_exits_two_on_a_refused_repository(tmp_path, capsys):
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text("gitdir: //server/share/repo\n", encoding="utf-8")
    assert main([str(work), "--format", "json"]) == 2
    capsys.readouterr()


# ----- A4 and A5: .gitmodules URLs are parsed linearly and never echo credentials -----------


def _gitmodules_findings(monkeypatch, text: str):
    monkeypatch.setattr(
        hygiene,
        "_read_text_blobs",
        lambda git, shas, limit: [("sha", text.encode("utf-8"))],
    )
    entries = [TreeEntry("100644", "blob", "sha", len(text), ".gitmodules")]
    return hygiene._submodule_findings(None, entries)


def test_long_submodule_url_line_is_fast(monkeypatch):
    text = '[submodule "x"]\n\turl = ' + "a" * 65536 + "\n"
    start = time.perf_counter()
    out = _gitmodules_findings(monkeypatch, text)
    assert time.perf_counter() - start < 2.0
    assert out == []


def test_colon_run_in_submodule_url_is_fast(monkeypatch):
    text = '[submodule "x"]\n\turl = x://' + ":" * 32000 + "@h\n"
    start = time.perf_counter()
    out = _gitmodules_findings(monkeypatch, text)
    assert time.perf_counter() - start < 2.0
    assert out and out[0].severity == Severity.HIGH


@pytest.mark.parametrize(
    "url, fragments",
    [
        ("http://ghp_TOKEN123456@github.example/org/repo.git", ("TOKEN123456",)),
        ("https://github.example/org/repo.git?token=QuerySecret99", ("QuerySecret99",)),
        ("https://deploy:Pa@ss/Word9X@git.example/repo.git", ("Pa@ss", "Word9X")),
    ],
)
def test_credentialed_submodule_urls_never_echo_the_secret(monkeypatch, url, fragments):
    out = _gitmodules_findings(monkeypatch, f'[submodule "x"]\n\turl = {url}\n')
    assert out, "a credentialed URL must be reported"
    assert out[0].severity == Severity.HIGH
    blob = json.dumps([f.to_dict() for f in out])
    for fragment in fragments:
        assert fragment not in blob


def test_plain_https_submodule_url_is_not_reported(monkeypatch):
    assert (
        _gitmodules_findings(
            monkeypatch, '[submodule "x"]\n\turl = https://git.example/r.git\n'
        )
        == []
    )


# ----- A6: truncation never hides findings; the exit code uses every finding ---------------


@requires_git
def test_tree_findings_are_not_truncated_by_top(repo_factory):
    repo = repo_factory()
    for n in range(30):
        repo.write(f"junk{n}.log", "x")
    repo.commit("logs")
    handle = GitRepo(repo.path)
    entries = handle.ls_tree(repo.sha())
    findings = hygiene.tree_findings(handle, entries, Settings(top=1))
    assert sum(f.rule_id == "HY001" for f in findings) == 30


@requires_git
def test_high_finding_after_many_low_findings_still_fails(repo_factory, capsys):
    repo = repo_factory()
    for n in range(300):
        repo.write(f"junk{n}.log", "x")
    repo.write(
        ".gitmodules", '[submodule "x"]\n\turl = https://u:pw@host.example/x.git\n'
    )
    repo.commit("logs and a credentialed submodule")
    code = main(
        [
            str(repo.path),
            "--format",
            "json",
            "--no-history",
            "--top",
            "1",
            "--fail-on",
            "high",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert any(
        f["rule"] == "HY008" and f["severity"] == "high" for f in report["findings"]
    )
    assert code == 1


# ----- A3 budgets and caps, HY013 and HY014 ------------------------------------------------


@requires_git
def test_scan_budget_marks_partial_and_sarif_unsuccessful(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "x" * 100)
    repo.commit("init")
    result = run_audit(Settings(repo=repo.path, history=False, scan_budget_bytes=1))
    assert result.partial
    assert any(
        f.rule_id == "HY013" and "scan budget" in f.title for f in result.findings
    )
    sarif = json.loads(render_sarif(result))
    assert sarif["runs"][0]["invocations"][0]["executionSuccessful"] is False


@requires_git
def test_complete_audit_is_not_partial(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "x")
    repo.commit("init")
    result = run_audit(Settings(repo=repo.path, history=False))
    assert not result.partial
    assert not any(f.rule_id == "HY013" for f in result.findings)


@requires_git
def test_total_secret_hit_cap_is_reported(repo_factory, monkeypatch):
    monkeypatch.setattr(audit_module, "MAX_TOTAL_SECRET_HITS", 1)
    repo = repo_factory()
    key = fake_aws_key_id()
    repo.write("one.txt", f"a = '{key}'\n")
    repo.write("two.txt", f"b = '{key}'\n")
    repo.commit("two secrets")
    result = run_audit(Settings(repo=repo.path, history=False))
    assert result.partial
    assert sum(f.rule_id == "SEC001" for f in result.findings) == 1
    assert any(
        f.rule_id == "HY013" and "total secret hit cap" in f.title
        for f in result.findings
    )


def test_per_blob_hit_cap_limits_hits(monkeypatch):
    monkeypatch.setattr(secrets_module, "MAX_HITS_PER_BLOB", 2)
    key = fake_aws_key_id()
    data = "\n".join(f"k = '{key}'" for _ in range(5)).encode()
    assert len(scan_text(data, Allowlist())) == 2


@requires_git
def test_skipped_blobs_are_reported_as_info(repo_factory):
    repo = repo_factory()
    repo.write("image.bin", b"\0\1\2binary")
    repo.commit("binary")
    result = run_audit(Settings(repo=repo.path, history=False))
    skipped = [f for f in result.findings if f.rule_id == "HY014"]
    assert skipped and skipped[0].severity == Severity.INFO


@requires_git
def test_allow_path_suppresses_content_secrets(repo_factory):
    repo = repo_factory()
    repo.write("notes/creds.txt", f"key = '{fake_aws_key_id()}'\n")
    repo.commit("notes")
    open_result = run_audit(Settings(repo=repo.path, history=False))
    allowed = run_audit(
        Settings(repo=repo.path, history=False, allow_paths=("notes/*",))
    )
    assert any(f.rule_id == "SEC001" for f in open_result.findings)
    assert not any(f.rule_id == "SEC001" for f in allowed.findings)


# ----- Nits: stopwords, redaction, fingerprints, atomic writes, config, CLI ---------------


def test_password_is_no_longer_a_blanket_stopword():
    assert not Allowlist().value_allowed("MyPassword9xQ")


def test_short_values_are_masked_entirely():
    assert redact("abcdefghij") == "********** (len 10)"
    assert "abcd" not in redact("abcdefghijklmno")


def test_fingerprint_separates_same_shape_hits_by_position():
    def fp(line: int, occurrence: int) -> str:
        return content_finding(
            "SEC001", "a.txt", "c1", "b1", line, "AKIA***", "high", occurrence
        ).fingerprint

    assert fp(1, 0) == fp(1, 0)
    assert fp(1, 0) != fp(1, 1)
    assert fp(1, 0) != fp(2, 0)


def test_atomic_write_leaves_only_the_target(tmp_path):
    target = tmp_path / "out.json"
    atomic_write(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_atomic_write_removes_its_temp_file_on_failure(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(safety_module.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(tmp_path / "out.json", "x")
    assert list(tmp_path.iterdir()) == []


def test_honor_inline_allow_is_a_config_key():
    assert parse_config({"gitaudit": {"honor_inline_allow": True}}) == {
        "honor_inline_allow": True
    }


def test_tool_that_is_not_a_table_is_refused():
    with pytest.raises(ValueError, match="tool"):
        parse_config({"tool": "x"})


def test_non_table_tool_does_not_crash_discovery(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "pyproject.toml").write_text('tool = "x"\n', encoding="utf-8")
    assert discover_config(cwd, tmp_path / "elsewhere") is None


def test_unexpected_error_exits_two(repo_factory, monkeypatch, capsys):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")

    def boom(settings):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "run_audit", boom)
    assert main([str(repo.path), "--format", "json"]) == 2
    assert "internal error" in capsys.readouterr().err


# ----- Second review: alternates quoting, include keys, storage discovery, partial exit ------


@requires_git
def test_quoted_alternate_line_is_refused(repo_factory):
    # Git C-unquotes a quoted alternates line, so the plain-text checks cannot judge it.
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    (repo.path / ".git" / "objects" / "info" / "alternates").write_text(
        '"/elsewhere/objects"\n', encoding="utf-8"
    )
    with pytest.raises(GitError, match="quoted"):
        GitRepo(repo.path)


@requires_git
@pytest.mark.parametrize(
    "section",
    [
        "[include]\n\tpath = /tmp/a.cfg\n",
        "[INCLUDE]\n\tPATH = /tmp/b.cfg\n",
        '[includeIf "gitdir:/x"]\n\tpath = /tmp/c.cfg\n',
        '[include "sub"]\n\tpath = /tmp/d.cfg\n',
    ],
)
def test_include_spellings_are_refused(repo_factory, section):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    with (repo.path / ".git" / "config").open("a", encoding="utf-8") as fh:
        fh.write(section)
    with pytest.raises(GitError, match="include"):
        GitRepo(repo.path)


@requires_git
def test_include_in_config_worktree_is_refused(repo_factory):
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    (repo.path / ".git" / "config.worktree").write_text(
        "[include]\n\tpath = /tmp/e.cfg\n", encoding="utf-8"
    )
    with pytest.raises(GitError, match="include"):
        GitRepo(repo.path)


@requires_git
def test_config_git_cannot_parse_is_refused(repo_factory):
    # Git rejects "[ include ]" as a bad line, so no reader can disagree about it: fail closed.
    repo = repo_factory()
    repo.write("a.txt", "a")
    repo.commit("init")
    with (repo.path / ".git" / "config").open("a", encoding="utf-8") as fh:
        fh.write("[ include ]\n\tpath = /tmp/f.cfg\n")
    with pytest.raises(GitError, match="cannot read"):
        GitRepo(repo.path)


@requires_git
def test_bare_repository_is_discovered_as_itself(tmp_path):
    bare = tmp_path / "bare.git"
    subprocess.run([GIT, "init", "--bare", "-q", str(bare)], check=True, env=_CLEAN_ENV)
    assert GitRepo(bare).git_dir == bare.resolve()


@requires_git
def test_gitfile_is_discovered_at_the_directory_git_reports(repo_factory, tmp_path):
    store = repo_factory()
    store.write("a.txt", "a")
    store.commit("init")
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {store.path / '.git'}\n", encoding="utf-8")
    assert GitRepo(work).git_dir == (store.path / ".git").resolve()


@requires_git
def test_include_in_a_gitfile_target_is_refused(repo_factory, tmp_path):
    store = repo_factory()
    store.write("a.txt", "a")
    store.commit("init")
    with (store.path / ".git" / "config").open("a", encoding="utf-8") as fh:
        fh.write("[include]\n\tpath = /tmp/g.cfg\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {store.path / '.git'}\n", encoding="utf-8")
    with pytest.raises(GitError, match="include"):
        GitRepo(work)


@requires_git
def test_nested_directory_is_discovered_at_the_enclosing_repository(repo_factory):
    repo = repo_factory()
    repo.write("sub/a.txt", "a")
    repo.commit("init")
    assert GitRepo(repo.path / "sub").git_dir == (repo.path / ".git").resolve()


@requires_git
def test_partial_audit_fails_the_run_by_default(repo_factory, capsys):
    repo = repo_factory()
    repo.write("a.txt", "x" * 100)
    repo.commit("init")
    args = [str(repo.path), "--fail-on", "none", "--format", "json", "--no-history"]
    args += ["--scan-budget", "1"]
    assert main(args) == 1
    assert '"HY013"' in capsys.readouterr().out


@requires_git
def test_allow_partial_lets_a_partial_audit_pass(repo_factory, capsys):
    repo = repo_factory()
    repo.write("a.txt", "x" * 100)
    repo.commit("init")
    args = [str(repo.path), "--fail-on", "none", "--format", "json", "--no-history"]
    args += ["--scan-budget", "1", "--allow-partial"]
    assert main(args) == 0
    out = json.loads(capsys.readouterr().out)
    hy013 = [f for f in out["findings"] if f["rule"] == "HY013"]
    assert hy013 and hy013[0]["severity"] == "info"


@requires_git
def test_allow_path_skips_content_only_when_every_path_is_allowed(repo_factory):
    # The same secret under an allowed path and a non-allowed one: the non-allowed path must scan.
    repo = repo_factory()
    key = fake_aws_key_id()
    repo.write("docs/a.txt", f"a = '{key}'\n")
    repo.write("src/a.txt", f"a = '{key}'\n")
    repo.commit("same blob twice")
    result = run_audit(Settings(repo=repo.path, history=False, allow_paths=("src/*",)))
    assert any(f.rule_id == "SEC001" for f in result.findings)


@requires_git
def test_allow_path_skips_content_when_all_paths_are_allowed(repo_factory):
    repo = repo_factory()
    key = fake_aws_key_id()
    repo.write("docs/a.txt", f"a = '{key}'\n")
    repo.write("src/a.txt", f"a = '{key}'\n")
    repo.commit("same blob twice")
    result = run_audit(
        Settings(repo=repo.path, history=False, allow_paths=("docs/*", "src/*"))
    )
    assert not any(f.rule_id == "SEC001" for f in result.findings)
