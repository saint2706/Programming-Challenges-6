"""Pure-function unit tests: no git process is started here."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from gitaudit import hygiene
from gitaudit.baseline import apply_baseline, load_baseline, write_baseline
from gitaudit.blobs import (
    BlobInfo,
    gc_findings,
    is_archive_or_binary_path,
    is_binary,
    lfs_findings,
    oversized_findings,
)
from gitaudit.config import Settings, build_settings, parse_config
from gitaudit.findings import CATALOG, Finding, Severity, info
from gitaudit.safety import (
    atomic_write,
    check_output_path,
    fingerprint,
    is_valid_ref_name,
    parse_size,
    redact,
    sanitize,
)
from gitaudit.secrets import (
    Allowlist,
    content_finding,
    path_findings,
    path_rule_ids,
    scan_text,
    shannon,
)
from helpers import fake_aws_key_id, secret_fixtures

# ---------------------------------------------------------------- safety ------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("plain", "plain"),
        ("tab\there", "tab here"),
        ("\x1b[31mred", "\\x1b[31mred"),
        ("bidi\u202eevil", "bidi\\u202eevil"),
        ("nul\x00", "nul\\x00"),
        ("line\nbreak", "line\\x0abreak"),
    ],
)
def test_sanitize_escapes_controls(raw, expected):
    assert sanitize(raw) == expected


@pytest.mark.parametrize("limit", [1, 5, 20])
def test_sanitize_respects_limit(limit):
    out = sanitize("x" * 500, limit)
    assert len(out) <= limit


def test_sanitize_handles_non_strings():
    assert sanitize(12) == "12"
    assert sanitize(None) == "None"


@pytest.mark.parametrize(
    "value, expected",
    [
        (5, 5),
        ("5", 5),
        ("512KB", 512_000),
        ("5MiB", 5 * 1024**2),
        ("1GB", 10**9),
        ("2 KiB", 2048),
        ("0", 0),
    ],
)
def test_parse_size_valid(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize(
    "value", [-1, "-5", "5 XB", "abc", True, "9" * 40, "1.5MB", ""]
)
def test_parse_size_rejects(value):
    with pytest.raises(ValueError):
        parse_size(value)


@pytest.mark.parametrize(
    "name, ok",
    [
        ("main", True),
        ("feature/login-2", True),
        ("release_1.0", True),
        ("-dash", False),
        ("a..b", False),
        ("with space", False),
        ("semi;colon", False),
        ("", False),
        ("x" * 251, False),
    ],
)
def test_ref_name_validation(name, ok):
    assert is_valid_ref_name(name) is ok


def test_redact_short_secret_fully_masked():
    assert redact("abc") == "*** (len 3)"


def test_redact_keeps_only_prefix_and_suffix():
    value = "ghp_" + "9fK3mQ2vR7xT1bN8cW4eH6yU0sA5dG3jL2pZ"
    out = redact(value)
    assert value not in out
    assert out.startswith("ghp_***") and out.endswith("(len 40)")
    assert "9fK3mQ2v" not in out


def test_fingerprint_is_stable_and_separator_safe():
    assert fingerprint("a", "b") == fingerprint("a", "b")
    assert fingerprint("ab", "c") != fingerprint("a", "bc")
    assert len(fingerprint("x")) == 64


def test_atomic_write_round_trip(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_write_refuses_huge_payload(tmp_path, monkeypatch):
    from gitaudit import safety

    monkeypatch.setattr(safety, "MAX_OUTPUT_FILE_BYTES", 10)
    with pytest.raises(ValueError):
        atomic_write(tmp_path / "big.txt", "x" * 11)


def test_check_output_path_refuses_directory(tmp_path):
    with pytest.raises(ValueError):
        check_output_path(tmp_path)


def test_check_output_path_refuses_missing_parent(tmp_path):
    with pytest.raises(ValueError):
        check_output_path(tmp_path / "nope" / "x.json")


def test_check_output_path_refuses_forbidden_root(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    with pytest.raises(ValueError):
        check_output_path(git_dir / "x.json", forbidden_root=git_dir)


def test_check_output_path_allows_sibling(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    assert (
        check_output_path(tmp_path / "report.json", forbidden_root=git_dir).name
        == "report.json"
    )


def test_check_output_path_refuses_symlink(tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        check_output_path(link)


# -------------------------------------------------------------- findings ------------------


def test_severity_parse_is_case_insensitive():
    assert Severity.parse(" High ") is Severity.HIGH


def test_severity_parse_rejects_unknown():
    with pytest.raises(ValueError):
        Severity.parse("urgent")


@pytest.mark.parametrize(
    "severity, level",
    [
        (Severity.INFO, "note"),
        (Severity.LOW, "note"),
        (Severity.MEDIUM, "warning"),
        (Severity.HIGH, "error"),
        (Severity.CRITICAL, "error"),
    ],
)
def test_sarif_levels(severity, level):
    assert severity.sarif_level == level


def test_catalog_has_every_family():
    prefixes = {rid[:2] for rid in CATALOG}
    assert {"BR", "BL", "SE", "PT", "HY"} <= prefixes
    assert len(CATALOG) == 6 + 4 + 15 + 6 + 14


@pytest.mark.parametrize("rule", sorted(CATALOG))
def test_every_rule_has_description_and_category(rule):
    rule_info = info(rule)
    assert rule_info.description and rule_info.category
    assert rule_info.rule_id == rule


def _finding(**kw) -> Finding:
    base = {
        "rule_id": "BR001",
        "severity": Severity.LOW,
        "confidence": "high",
        "title": "t",
        "category": "branches",
        "evidence": "e",
        "remediation": "r",
        "fingerprint": "f" * 64,
    }
    base.update(kw)
    return Finding(**base)


def test_finding_with_suppressed_is_a_copy():
    original = _finding()
    marked = original.with_suppressed(True)
    assert marked.suppressed and not original.suppressed


def test_finding_to_dict_uses_labels():
    data = _finding(severity=Severity.CRITICAL).to_dict()
    assert data["severity"] == "critical"
    assert data["rule"] == "BR001"


# ----------------------------------------------------------------- config -----------------


def test_defaults_are_sane():
    s = Settings()
    assert s.stale_days == 90 and s.fail_on == "high" and s.history is True
    assert s.fail_threshold() is Severity.HIGH


def test_fail_on_none_disables_threshold():
    assert Settings(fail_on="none").fail_threshold() is None


@pytest.mark.parametrize(
    "table, error",
    [
        ({"stale_days": -1}, "stale_days"),
        ({"stale_days": "90"}, "stale_days"),
        ({"stale_days": True}, "stale_days"),
        ({"top": 0}, "top"),
        ({"history": "yes"}, "history"),
        ({"fail_on": "urgent"}, "severity"),
        ({"since": "2 years; rm -rf"}, "since"),
        ({"allow_regexes": ["("]}, "invalid allow regex"),
        ({"allow_regexes": ["x" * 201]}, "200"),
        ({"stopwords": "abc"}, "list"),
        ({"bogus": 1}, "unknown setting"),
        ({"default_branch": 5}, "string"),
    ],
)
def test_config_validation_errors(table, error):
    with pytest.raises(ValueError, match=error):
        parse_config({"gitaudit": table})


def test_config_accepts_secrets_subtable():
    values = parse_config(
        {"gitaudit": {"stale_days": 30, "secrets": {"allow_paths": ["docs/*"]}}}
    )
    assert values == {"stale_days": 30, "allow_paths": ("docs/*",)}


def test_config_rejects_unknown_secrets_key():
    with pytest.raises(ValueError):
        parse_config({"gitaudit": {"secrets": {"nope": []}}})


def test_config_pyproject_table_is_read():
    values = parse_config({"tool": {"gitaudit": {"top": 5}}})
    assert values == {"top": 5}


def test_config_missing_table_is_empty():
    assert parse_config({"project": {"name": "x"}}) == {}


def test_config_size_strings_are_parsed():
    assert parse_config({"gitaudit": {"max_blob_size": "2MiB"}}) == {
        "max_blob_size": 2 * 1024**2
    }


def test_build_settings_cli_beats_file(tmp_path):
    # Config lives in a directory outside the audited repo, so discovery applies.
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "gitaudit.toml").write_text(
        "[gitaudit]\nstale_days = 10\n", encoding="utf-8"
    )
    s = build_settings(repo, None, cfg, {"stale_days": None})
    assert s.stale_days == 10
    s2 = build_settings(repo, None, cfg, {"stale_days": 3})
    assert s2.stale_days == 3
    s3 = build_settings(repo, None, tmp_path, {"stale_days": 7})
    assert s3.stale_days == 7 and s3.config_source is None


def test_build_settings_pyproject_only_with_table(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert build_settings(repo, None, cfg, {}).config_source is None
    (cfg / "pyproject.toml").write_text("[tool.gitaudit]\ntop = 4\n", encoding="utf-8")
    s = build_settings(repo, None, cfg, {})
    assert s.top == 4 and s.config_source is not None


def test_build_settings_rejects_invalid_fail_on(tmp_path):
    with pytest.raises(ValueError):
        build_settings(tmp_path, None, tmp_path, {"fail_on": "nope"})


def test_build_settings_missing_explicit_config(tmp_path):
    with pytest.raises(ValueError):
        build_settings(tmp_path, tmp_path / "missing.toml", tmp_path, {})


def test_build_settings_rejects_oversized_config(tmp_path):
    from gitaudit import config as config_module

    path = tmp_path / "gitaudit.toml"
    path.write_text(
        "# " + "x" * (config_module.MAX_CONFIG_BYTES + 10), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="too large"):
        config_module.load_config_file(path)


def test_build_settings_rejects_bad_toml(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "gitaudit.toml").write_text("[gitaudit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot parse"):
        build_settings(repo, None, cfg, {})


# ------------------------------------------------------------------ blobs -----------------


def test_is_binary_sniffs_nul():
    assert is_binary(b"abc\x00def")
    assert not is_binary(b"plain text\n")


@pytest.mark.parametrize(
    "path, expected",
    [
        ("a/b.zip", True),
        ("x.PNG", False),
        ("lib.so", True),
        ("doc.pdf", True),
        ("a.py", False),
    ],
)
def test_archive_or_binary_suffixes(path, expected):
    assert is_archive_or_binary_path(path) is expected


def _blob(size, path="big.bin", in_head=True, sha="a" * 40):
    return BlobInfo(sha=sha, size=size, path=path, commit="c" * 40, in_head=in_head)


def test_oversized_blob_rules_and_severity():
    s = Settings(max_blob_size=1000)
    out = oversized_findings(
        [_blob(2000), _blob(20_000, sha="b" * 40, in_head=False)], s
    )
    by_rule = {f.rule_id: f for f in out}
    assert set(by_rule) == {"BL001", "BL002"}
    assert by_rule["BL001"].severity is Severity.MEDIUM  # 2x the limit, in head
    assert by_rule["BL002"].severity is Severity.MEDIUM  # history only, never HIGH here


def test_oversized_head_blob_over_ten_times_limit_is_high():
    out = oversized_findings([_blob(20_000)], Settings(max_blob_size=1000))
    assert out[0].rule_id == "BL001" and out[0].severity is Severity.HIGH


def test_oversized_blob_below_limit_is_ignored():
    assert oversized_findings([_blob(10)], Settings(max_blob_size=1000)) == []


def test_oversized_blob_ranked_and_capped():
    s = Settings(max_blob_size=10, top=2)
    blobs = [_blob(100 + i, path=f"f{i}", sha=f"{i:040d}") for i in range(5)]
    out = oversized_findings(blobs, s)
    assert len(out) == 2
    assert out[0].extra["size"] > out[1].extra["size"]


def test_oversized_remediation_is_inert_for_hostile_path():
    s = Settings(max_blob_size=10)
    out = oversized_findings([_blob(100, path="a;rm -rf /")], s)
    assert ";" not in out[0].remediation.split("#")[0]


def test_lfs_candidate_by_suffix_and_by_sniff():
    s = Settings(lfs_threshold=100, max_blob_size=10**9, top=10)
    blobs = [
        BlobInfo("1" * 40, 500, "model.bin", None, True),
        BlobInfo("2" * 40, 500, "notes.txt", None, True),
        BlobInfo("3" * 40, 500, "data.dat", None, True),
    ]
    out = lfs_findings(blobs, s, binary_shas={"3" * 40})
    paths = {f.path for f in out}
    assert paths == {"model.bin", "data.dat"}
    assert all(f.rule_id == "BL003" for f in out)


def test_lfs_skips_below_threshold():
    out = lfs_findings(
        [BlobInfo("1" * 40, 5, "a.bin", None, True)], Settings(lfs_threshold=100), set()
    )
    assert out == []


def test_gc_advice_thresholds():
    assert gc_findings({"count": 10}) == []
    out = gc_findings({"count": 7000, "size-pack": 12, "garbage": 0})
    assert len(out) == 1 and out[0].rule_id == "BL004"
    assert "7000 loose objects" in out[0].evidence
    assert gc_findings({"garbage": 2})[0].rule_id == "BL004"


# ----------------------------------------------------------------- secrets ----------------


def test_shannon_entropy_bounds():
    assert shannon("") == 0.0
    assert shannon("aaaa") == 0.0
    assert shannon("abcd") == pytest.approx(2.0)


FIXTURE_RULES = {
    "aws_key": "SEC001",
    "github_classic": "SEC003",
    "github_fine": "SEC004",
    "gitlab": "SEC005",
    "slack": "SEC006",
    "slack_hook": "SEC007",
    "stripe": "SEC008",
    "google": "SEC009",
    "jwt": "SEC011",
    "url_password": "SEC012",
    "npm_token": "SEC014",
    "private_key_header": "SEC010",
}


@pytest.mark.parametrize("name, rule", sorted(FIXTURE_RULES.items()))
def test_each_provider_fixture_fires_its_rule(name, rule):
    text = secret_fixtures()[name] + "\n"
    hits = scan_text(text.encode("utf-8"), Allowlist())
    assert rule in {h.rule_id for h in hits}, (name, [h.rule_id for h in hits])


def test_hits_never_contain_the_secret_value():
    value = secret_fixtures()["github_classic"]
    hits = scan_text(f"token = '{value}'\n".encode(), Allowlist())
    assert hits and all(value not in h.shape for h in hits)


def test_placeholder_values_are_allowed():
    text = b'password = "changeme-placeholder-value"\n'
    assert all(h.rule_id != "SEC013" for h in scan_text(text, Allowlist()))


def test_allowlist_regex_suppresses_value():
    value = fake_aws_key_id()
    allow = Allowlist(regexes=[r"Q3XJ7K"])
    assert scan_text(f"x = '{value}'\n".encode(), allow) == []


def test_allowlist_stopwords_are_case_insensitive():
    allow = Allowlist(stopwords=["MyCorpWord"])
    assert allow.value_allowed("prefix-mycorpword-suffix")


def test_inline_marker_is_ignored_by_default():
    value = fake_aws_key_id()
    line = f"x = '{value}'  # gitaudit:allow\n".encode()
    assert scan_text(line, Allowlist())


def test_inline_marker_skips_line_when_opted_in():
    value = fake_aws_key_id()
    line = f"x = '{value}'  # gitaudit:allow\n".encode()
    assert scan_text(line, Allowlist(honor_inline=True)) == []


def test_secret_at_window_boundary_is_found():
    value = fake_aws_key_id()
    line = b"a" * 4000 + b" " + value.encode() + b" " + b"z" * 300
    rules = {h.rule_id for h in scan_text(line, Allowlist())}
    assert "SEC001" in rules


def test_long_line_scan_completes_quickly():
    data = (b"x" * 2_000_000) + b"\n"
    assert scan_text(data, Allowlist()) == []


def test_data_uri_payload_is_ignored():
    blob = b"data:image/png;base64," + b"QUFB" * 2000
    assert scan_text(b"img = '" + blob + b"'\n", Allowlist(), entropy=True) == []


def test_entropy_rule_flags_random_literal_and_skips_hashes():
    random_token = "Qz8nW3pL9xV2kR7mT4yB6cH1"
    hits = scan_text(f'k = "{random_token}"\n'.encode(), Allowlist())
    assert "SEC015" in {h.rule_id for h in hits}
    sha = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    assert "SEC015" not in {
        h.rule_id for h in scan_text(f'h = "{sha}"\n'.encode(), Allowlist())
    }


def test_entropy_can_be_disabled():
    random_token = "Qz8nW3pL9xV2kR7mT4yB6cH1"
    hits = scan_text(f'k = "{random_token}"\n'.encode(), Allowlist(), entropy=False)
    assert "SEC015" not in {h.rule_id for h in hits}


def test_private_key_block_is_critical_rule_with_header_shape():
    header = secret_fixtures()["private_key_header"]
    hits = scan_text((header + "\n").encode(), Allowlist())
    assert any(h.rule_id == "SEC010" and "RSA PRIVATE KEY" in h.shape for h in hits)


@pytest.mark.parametrize(
    "path, rules",
    [
        (".env", ["PTH001"]),
        ("config/.env.local", ["PTH001"]),
        (".env.example", []),
        ("keys/id_ed25519", ["PTH002"]),
        ("certs/server.pem", ["PTH003"]),
        ("infra/terraform.tfstate", ["PTH004"]),
        ("ops/kubeconfig", ["PTH005"]),
        ("home/.aws/credentials", ["PTH006"]),
        ("README.md", []),
    ],
)
def test_path_rules(path, rules):
    assert path_rule_ids(path) == rules


def test_path_findings_downgrade_pem_and_respect_allowlist():
    allow = Allowlist(paths=["fixtures/*"])
    out = path_findings("certs/a.pem", "b" * 40, "c" * 40, Allowlist())
    assert out and out[0].severity is Severity.MEDIUM
    assert path_findings("fixtures/.env", "b" * 40, None, allow) == []


def test_content_finding_is_redacted_and_carries_line():
    f = content_finding(
        "SEC003", "src/a.py", "c" * 40, "d" * 40, 7, "ghp_***xy (len 40)", "high"
    )
    assert f.line == 7 and f.rule_id == "SEC003" and f.path == "src/a.py"
    assert "ghp_***xy" in f.evidence


# --------------------------------------------------------------- hygiene (pure) -----------


@pytest.mark.parametrize(
    "path, expected",
    [
        (".DS_Store", True),
        ("pkg/__pycache__/x.pyc", True),
        ("a/b/.pytest_cache/x", True),
        ("src/main.py", False),
        ("notes.log", True),
    ],
)
def test_ignorable_paths(path, expected):
    assert hygiene.ignorable_path(path) is expected


def test_identity_findings_flag_placeholders_only():
    counts = Counter({"dev@example.com": 3, "real@corp.io": 9, "": 1, "bad-no-at": 2})
    out = hygiene.identity_findings(counts, Settings(top=10))
    assert len(out) == 3
    assert all(f.rule_id == "HY012" for f in out)
    assert not any("real@corp.io" in f.title for f in out)


def test_commit_findings_huge_by_file_count():
    commit = {
        "sha": "e" * 40,
        "subject": "import",
        "changes": [("A", f"f{i}", "0" * 40, "1" * 40) for i in range(5)],
    }
    out = hygiene.commit_findings(commit, Settings(huge_commit_files=3), {})
    assert [f.rule_id for f in out] == ["HY005"]


def test_commit_findings_small_commit_is_clean():
    commit = {
        "sha": "e" * 40,
        "subject": "x",
        "changes": [("A", "a", "0" * 40, "1" * 40)],
    }
    assert hygiene.commit_findings(commit, Settings(), {"1" * 40: 10}) == []


def test_commit_findings_huge_by_bytes():
    commit = {
        "sha": "e" * 40,
        "subject": "x",
        "changes": [("A", "a", "0" * 40, "1" * 40)],
    }
    out = hygiene.commit_findings(
        commit, Settings(huge_commit_bytes=100), {"1" * 40: 500}
    )
    assert out and out[0].rule_id == "HY005"


def test_signature_findings_accept_good_codes():
    assert hygiene.signature_findings([("a" * 40, "G"), ("b" * 40, "U")]) == []
    out = hygiene.signature_findings([("a" * 40, "G"), ("b" * 40, "N")])
    assert out and out[0].rule_id == "HY011"


# ------------------------------------------------------------------ baseline ------------


def _f(rule: str, fp: str) -> Finding:
    return _finding(rule_id=rule, fingerprint=fp)


def test_baseline_round_trip(tmp_path):
    findings = [_f("SEC001", "a" * 64), _f("BR001", "b" * 64), _f("BR001", "b" * 64)]
    target = tmp_path / "base.json"
    count = write_baseline(target, findings, None)
    assert count == 2
    loaded = load_baseline(target)
    assert loaded == {"a" * 64, "b" * 64}
    marked = apply_baseline(findings, loaded)
    assert all(f.suppressed for f in marked)


def test_baseline_only_suppresses_listed(tmp_path):
    accepted = frozenset({"a" * 64})
    out = apply_baseline([_f("X", "a" * 64), _f("Y", "c" * 64)], accepted)
    assert [f.suppressed for f in out] == [True, False]


def test_baseline_file_contains_no_evidence(tmp_path):
    write_baseline(tmp_path / "b.json", [_f("SEC001", "a" * 64)], None)
    text = (tmp_path / "b.json").read_text(encoding="utf-8")
    assert "evidence" not in text and "title" not in text


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not json", "not valid JSON"),
        ('{"version": 2, "findings": []}', "version 1"),
        ('{"version": 1, "findings": {}}', "list"),
        ('{"version": 1, "findings": [1]}', "object"),
        ('{"version": 1, "findings": [{"fingerprint": "XYZ"}]}', "64"),
        ('{"version": 1, "findings": [{"fingerprint": "' + "A" * 64 + '"}]}', "64"),
    ],
)
def test_baseline_validation(tmp_path, payload, message):
    path = tmp_path / "b.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_baseline(path)


def test_baseline_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_baseline(tmp_path / "none.json")


def test_baseline_too_large(tmp_path, monkeypatch):
    from gitaudit import baseline

    monkeypatch.setattr(baseline, "MAX_BASELINE_BYTES", 10)
    path = tmp_path / "b.json"
    path.write_text("x" * 50, encoding="utf-8")
    with pytest.raises(ValueError, match="too large"):
        load_baseline(path)


def test_baseline_refuses_symlinked_output(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        write_baseline(link, [], None)


# ------------------------------------------------------------------ reporters (pure) ----


def test_json_and_sarif_are_valid_documents():
    from gitaudit.audit import AuditResult
    from gitaudit.reporters import render_json, render_sarif

    result = AuditResult(
        repo="r",
        ref="HEAD",
        commit="a" * 40,
        default_branch="main",
        findings=[
            _finding(rule_id="SEC001", severity=Severity.HIGH, path="a/b.txt", line=3)
        ],
    )
    doc = json.loads(render_json(result))
    assert doc["findings"][0]["rule"] == "SEC001"
    sarif = json.loads(render_sarif(result))
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert run["results"][0]["level"] == "error"
    assert (
        run["results"][0]["locations"][0]["physicalLocation"]["region"]["startLine"]
        == 3
    )


def test_sarif_escapes_path_uri():
    from gitaudit.audit import AuditResult
    from gitaudit.reporters import render_sarif

    result = AuditResult(
        "r", "HEAD", "a" * 40, "main", [_finding(path="we ird#name?.txt")]
    )
    uri = json.loads(render_sarif(result))["runs"][0]["results"][0]["locations"][0][
        "physicalLocation"
    ]["artifactLocation"]["uri"]
    assert " " not in uri and "#" not in uri and "?" not in uri


def test_html_escapes_hostile_strings():
    from gitaudit.audit import AuditResult
    from gitaudit.reporters import render_html

    nasty = "<script>alert(1)</script>"
    result = AuditResult(
        nasty, nasty, "a" * 40, nasty, [_finding(title=nasty, path=nasty)]
    )
    page = render_html(result)
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
    assert "Content-Security-Policy" in page and "script-src" not in page


def test_table_renders_hostile_markup_as_text(capsys):
    from gitaudit.audit import AuditResult
    from gitaudit.reporters import render_table
    from rich.console import Console

    hostile = "[bold red]pwned[/] \x1b[2J"
    result = AuditResult("r", "HEAD", "a" * 40, "main", [_finding(title=hostile)])
    console = Console(file=__import__("io").StringIO(), width=120, force_terminal=False)
    render_table(result, console)
    out = console.file.getvalue()
    assert "[bold red]pwned[/]" in out
    assert "\x1b" not in out


def test_table_limit_reports_remainder():
    import io

    from gitaudit.audit import AuditResult
    from gitaudit.reporters import render_table
    from rich.console import Console

    findings = [
        _finding(fingerprint=str(i) * 64, rule_id=f"BR00{i % 6 + 1}") for i in range(3)
    ]
    result = AuditResult("r", "HEAD", "a" * 40, "main", findings)
    console = Console(file=io.StringIO(), width=120)
    render_table(result, console, limit=1)
    assert "2 more" in console.file.getvalue()


def test_audit_result_failing_threshold():
    from gitaudit.audit import AuditResult

    result = AuditResult(
        "r",
        "HEAD",
        "a" * 40,
        "main",
        [
            _finding(severity=Severity.MEDIUM),
            _finding(severity=Severity.HIGH, suppressed=True),
        ],
    )
    assert result.failing(Severity.HIGH) == []
    assert len(result.failing(Severity.MEDIUM)) == 1
    assert result.failing(None) == []
