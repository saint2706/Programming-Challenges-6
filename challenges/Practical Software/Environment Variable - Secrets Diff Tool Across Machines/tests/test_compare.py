from envdiff.compare import compare, compare_key
from helpers import src


def result(key, *texts, fmt="dotenv"):
    sources = [src(f"s{i}", t, fmt) for i, t in enumerate(texts)]
    return compare_key(key, sources)


def test_identical_values_share_a_group_and_are_same():
    r = result("A", "A=1\n", "A=1\n")
    assert (r.status, r.groups, r.absent, r.relations) == (
        "same",
        {"s0": 0, "s1": 0},
        [],
        [],
    )


def test_groups_are_numbered_by_first_appearance():
    r = result("A", "A=x\n", "A=y\n", "A=x\n", "A=z\n")
    assert r.groups == {"s0": 0, "s1": 1, "s2": 0, "s3": 2}
    assert r.status == "differs"


def test_key_present_in_some_sources_only_is_missing():
    r = result("A", "A=1\n", "B=2\n", "A=1\n")
    assert (r.status, r.absent) == ("missing", ["s1"])


def test_differs_and_missing_are_both_reported():
    r = result("A", "A=1\n", "A=2\n", "B=0\n")
    assert r.status == "differs" and r.absent == ["s2"]


def test_empty_is_present_not_missing_and_differs_from_a_value():
    r = result("A", "A=\n", "A=1\n")
    assert r.status == "differs" and r.absent == [] and r.relations == [(0, 1, "empty")]


def test_relations_name_whitespace_quote_and_case_differences():
    assert result("A", "A=abc\n", 'A="abc "\n').relations == [(0, 1, "whitespace")]
    assert result("A", "A=abc\n", "A=ABC\n").relations == [(0, 1, "case")]
    assert result("A", "A=abc\n", "A=abc\n", fmt="raw").relations == []
    quoted = [src("a", "A=abc\n", "raw"), src("b", 'A="abc"\n', "raw")]
    assert compare_key("A", quoted).relations == [(0, 1, "quotes")]


def test_relations_are_pairwise_so_three_way_diffs_still_explain_each_pair():
    r = result("A", "A=false\n", 'A="false "\n', "A=FALSE\n")
    assert r.relations == [(0, 1, "whitespace"), (0, 2, "case")]


def test_unrelated_values_have_no_relation():
    assert result("A", "A=one\n", "A=two\n").relations == []


def test_passthrough_is_its_own_group_and_never_equals_a_value():
    r = result("A", "A\n", "A=1\n")
    assert r.groups == {"s0": -1, "s1": 0}
    assert r.status == "differs" and r.relations == [(-1, 0, "pass-through")]
    assert result("A", "A\n", "A\n").status == "same"


def test_compare_lists_the_union_of_keys_sorted_and_counts():
    report = compare([src("a", "B=1\nA=1\n"), src("b", "A=2\nC=3\n")])
    assert [k.key for k in report.keys] == ["A", "B", "C"]
    assert (report.count("differs"), report.count("missing"), report.count("same")) == (
        1,
        2,
        0,
    )
    assert report.has_differences


def test_no_differences_when_everything_matches():
    report = compare([src("a", "A=1\n"), src("b", "A=1\n")])
    assert not report.has_differences


def test_ignore_globs_remove_keys_from_results_and_findings():
    a = src("a", "PATH=/x\nAWS_KEY_ID=\nA=1\n")
    b = src("b", "PATH=/y\nAWS_KEY_ID=z\nA=1\n")
    report = compare([a, b], ignore=["PATH", "AWS_*"])
    assert [k.key for k in report.keys] == ["A"] and report.ignored == 2
    assert not any(f.key in {"PATH", "AWS_KEY_ID"} for f in report.findings)


def codes(report, severity=None):
    return {(f.code, f.key) for f in report.findings if severity in (None, f.severity)}


def test_shared_secret_across_sources_is_flagged_but_shared_plain_config_is_not():
    report = compare(
        [
            src("prod", "SECRET_KEY=abc123\nPORT=80\n"),
            src("stg", "SECRET_KEY=abc123\nPORT=80\n"),
        ]
    )
    assert ("shared-secret", "SECRET_KEY") in codes(report)
    assert ("shared-secret", "PORT") not in codes(report)
    finding = next(f for f in report.findings if f.code == "shared-secret")
    assert finding.sources == ["prod", "stg"]


def test_different_secrets_are_not_shared():
    report = compare([src("a", "SECRET_KEY=one\n"), src("b", "SECRET_KEY=two\n")])
    assert ("shared-secret", "SECRET_KEY") not in codes(report)


def test_shared_empty_or_placeholder_secrets_are_not_reported_as_shared():
    report = compare(
        [
            src("a", "API_TOKEN=\nDB_PASSWORD=changeme\n"),
            src("b", "API_TOKEN=\nDB_PASSWORD=changeme\n"),
        ]
    )
    assert not any(f.code == "shared-secret" for f in report.findings)
    assert ("empty-secret", "API_TOKEN") in codes(report) and (
        "placeholder",
        "DB_PASSWORD",
    ) in codes(report)


def test_value_hygiene_findings():
    report = compare([src("a", 'A=" pad"\nSOME_TOKEN=\nB=changeme\n', "dotenv")])
    assert {
        ("edge-whitespace", "A"),
        ("empty-secret", "SOME_TOKEN"),
        ("placeholder", "B"),
    } <= codes(report)


def test_parse_issues_become_findings_with_line_numbers():
    report = compare([src("a", "A=1\nA=2\nbad line\nB=x y\n")])
    messages = {f.code: f.message for f in report.findings}
    assert "line 2" in messages["parse:duplicate-key"]
    assert "line 3" in messages["parse:invalid-line"]
    assert (
        next(f for f in report.findings if f.code == "parse:unquoted-space").severity
        == "info"
    )


def test_shape_mismatch_is_informational_and_ignores_redacted_secrets():
    report = compare(
        [
            src("a", "DEBUG=true\nSECRET_KEY=1234567\n"),
            src("b", "DEBUG=maybe\nSECRET_KEY=abcdefg\n"),
        ]
    )
    assert ("shape-mismatch", "DEBUG") in codes(report, "info")
    assert ("shape-mismatch", "SECRET_KEY") not in codes(report)


def test_lint_can_be_turned_off_and_a_single_source_is_lint_only():
    sources = [src("a", "SECRET_TOKEN=\n")]
    assert compare(sources, lint=False).findings == []
    report = compare(sources)
    assert not report.has_differences and report.has_warnings
