"""The two guards: every printed number is a declared fact (audit), and each fact recomputes (verify)."""

from dataclasses import replace

import pytest
from helpers import bundled, planted
from report_generator import findings as fnd
from report_generator import realize as rz
from report_generator.engine import build_report
from report_generator.verify import failed, verify_findings


@pytest.fixture(scope="module")
def report():
    return build_report(planted(0), row_label="days")


def test_every_template_is_digit_free_and_has_its_facts():
    for name, template in rz.TEMPLATES.items():
        assert not rz.DIGIT.search(template), name


def test_every_finding_type_the_engine_makes_uses_a_known_template(report):
    seen = {f.template for f in report.candidate_findings} | {
        f.template for f in report.notes
    }
    assert seen <= set(rz.TEMPLATES)
    for f in [*report.candidate_findings, *report.notes, report.overview]:
        assert rz.audit(f) == [], (f.id, rz.audit(f))


def test_audit_catches_a_number_smuggled_into_the_sentence(report):
    f = next(f for f in report.findings if f.type == "trend")
    f.text = f.text.replace("climbed", "climbed 99%")
    try:
        assert any("numbers shown" in p or "not the one" in p for p in rz.audit(f))
    finally:
        f.text = rz.sentence(f)


def test_audit_catches_a_fact_whose_text_no_longer_matches_its_value(report):
    f = next(f for f in report.findings if f.type == "trend")
    good = f.facts["change"]
    f.facts["change"] = fnd.Fact(good.value * 2, good.text, good.kind)
    try:
        assert any("does not match value" in p for p in rz.audit(f))
    finally:
        f.facts["change"] = good


def test_audit_ignores_digits_inside_names_from_the_data():
    f = fnd.Finding("x", "quality", "Data notes", ("Q1 2012",), "quality.missing",
                    {"col": fnd.label("Q1 2012 sales"), "share": fnd.pct(0.12)}, 0.5)  # fmt: skip
    f.text = rz.sentence(f)
    assert f.text == "Q1 2012 sales is missing in 12% of rows." and rz.audit(f) == []


def test_missing_fact_is_a_clear_error():
    f = fnd.Finding(
        "x",
        "quality",
        "Data notes",
        (),
        "quality.missing",
        {"col": fnd.label("a")},
        0.1,
    )
    with pytest.raises(KeyError, match="needs facts"):
        rz.sentence(f)


@pytest.mark.parametrize(
    ("kind", "value", "text"),
    [("count", 1234567, "1,234,567"), ("pct", 0.034, "3.4%"), ("pct", 0.253, "25%"), ("ratio", 2.64, "2.6x"),
     ("ratio", 12.4, "12x"), ("p", 0.0004, "p < 0.001"), ("p", 0.0123, "p = 0.012"), ("num", 5.0, "5"),
     ("num", 0.04213, "0.0421"), ("num", 12345.6, "12,346"), ("num", 12.34, "12.3"), ("num", 1.234, "1.23")],
)  # fmt: skip
def test_number_formats(kind, value, text):
    fact = {
        "count": fnd.count,
        "pct": fnd.pct,
        "ratio": fnd.ratio,
        "p": fnd.pvalue,
        "num": fnd.num,
    }[kind](value)
    assert fact.text == text and fnd.check_text(fact)


def test_verification_reproduces_the_planted_report(report):
    for fid, checks in report.checks.items():
        assert not failed(checks), fid
    assert {
        "overview",
        "trend:visits",
        "season:visits",
        "group:plan:spend",
        "correlation:x:y",
    } <= set(report.checks)


def test_verification_catches_a_tampered_fact_and_the_engine_drops_it():
    r = build_report(planted(0), row_label="days", verify=False)
    tampered = next(f for f in r.findings if f.id == "group:plan:spend")
    tampered.facts["hi_med"] = fnd.num(tampered.facts["hi_med"].value * 1.5)
    tampered.text = rz.sentence(tampered)
    checks = verify_findings(r.profile, [tampered])[tampered.id]
    assert [c.name for c in failed(checks)] == ["high median"]


def test_a_finding_that_cannot_be_recomputed_counts_as_failed(report):
    broken = replace(
        next(f for f in report.findings if f.type == "trend"), chart={"kind": "series"}
    )
    checks = verify_findings(report.profile, [broken])[broken.id]
    assert failed(checks) and "recompute failed" in checks[0].name


def test_real_tables_verify_cleanly():
    for name, kw in [
        ("bike", {"time": "date", "row_label": "days"}),
        ("ames", {"time": "sale_month", "row_label": "sales"}),
        (
            "retail",
            {"ignore": ("invoice", "description"), "row_label": "invoice lines"},
        ),
    ]:
        r = build_report(bundled(name), **kw)
        assert r.dropped == [], (name, r.dropped)
        assert sum(len(v) for v in r.checks.values()) >= 20
