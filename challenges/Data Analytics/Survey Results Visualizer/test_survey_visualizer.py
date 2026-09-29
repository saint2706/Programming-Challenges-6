"""Tests for the Survey Results Visualizer.

Run with:  uv run pytest -q
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest
from survey_visualizer import (
    analyze_item,
    analyze_survey,
    build_likert_figure,
    build_word_cloud,
    detect_roles,
    main,
    read_survey,
    render_report,
    score_column,
    summarize_text,
    tokenize,
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_CSV = SAMPLE_DIR / "employee_engagement.csv"


def frame(**columns: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame(columns, schema={k: pl.String for k in columns})


# ---- scoring ------------------------------------------------------------


def test_five_point_labels_are_case_and_whitespace_insensitive() -> None:
    mode, scale, codes, unknown, missing = score_column(
        [
            "Strongly agree",
            " agree ",
            "NEUTRAL",
            "Disagree.",
            "strongly  disagree",
            None,
            "",
        ]
    )
    assert (mode, scale) == ("labels", 5)
    assert codes == [5, 4, 3, 2, 1, None, None]
    assert not unknown
    assert missing == 2


def test_somewhat_label_promotes_column_to_seven_point() -> None:
    _, scale, codes, _, _ = score_column(
        ["Somewhat agree", "Agree", "Neutral", "Strongly disagree"]
    )
    assert scale == 7
    assert codes == [5, 6, 4, 1]


def test_numeric_scale_is_inferred_from_maximum() -> None:
    assert score_column(["1", "3", "5", "2"])[1] == 5
    assert score_column(["1", "3", "7", "2"])[1] == 7


def test_float_looking_integers_parse_but_fractions_are_unknown() -> None:
    _, _, codes, unknown, _ = score_column(["4.0", "2.5", "5"])
    assert codes == [4, None, 5]
    assert unknown == {"2.5": 1}


def test_unknown_labels_are_reported_not_dropped_silently() -> None:
    _, _, codes, unknown, missing = score_column(
        ["Agree", "Maybe", "Maybe", "Agree", "N/A"]
    )
    assert codes[1] is None
    assert unknown == {"Maybe": 2}
    assert missing == 1  # N/A counts as skipped, not as an unrecognized answer


def test_out_of_range_numbers_are_unknown() -> None:
    _, _, _, unknown, _ = score_column(["1", "2", "9", "3"], scale_override=5)
    assert unknown == {"9": 1}


def test_seven_point_label_on_forced_five_point_scale_is_unknown() -> None:
    _, _, codes, unknown, _ = score_column(
        ["Somewhat agree", "Agree"], scale_override=5
    )
    assert codes == [None, 4]
    assert unknown == {"Somewhat agree": 1}


def test_reverse_coding_flips_scale() -> None:
    raw = ["Strongly agree", "Strongly agree", "Disagree"]
    normal, _ = analyze_item("q", raw)
    flipped, _ = analyze_item("q", raw, reverse=True)
    assert normal.counts == [0, 1, 0, 0, 2]
    assert flipped.counts == [2, 0, 0, 1, 0]
    assert flipped.mean == pytest.approx(5 + 1 - normal.mean)
    assert flipped.reverse


# ---- item statistics ----------------------------------------------------


def test_item_statistics() -> None:
    raw = (
        ["Strongly agree"] * 2
        + ["Agree"] * 3
        + ["Neutral"]
        + ["Disagree"] * 2
        + ["Strongly disagree"] * 2
    )
    item, _ = analyze_item("q", [*raw, None])
    assert item.n == 10
    assert item.missing == 1
    assert item.mean == pytest.approx((2 * 5 + 3 * 4 + 3 + 2 * 2 + 2) / 10)
    assert item.top2_pct == pytest.approx(50.0)
    assert item.bottom2_pct == pytest.approx(40.0)
    assert item.neutral_pct == pytest.approx(10.0)
    assert item.net == pytest.approx(10.0)


def test_all_missing_item_has_no_mean() -> None:
    item, _ = analyze_item("q", [None, "", "n/a"], scale=5)
    assert item.n == 0
    assert item.mean is None
    assert item.net == 0.0


# ---- role detection -----------------------------------------------------


def test_detect_roles_on_sample_data() -> None:
    roles = detect_roles(read_survey(SAMPLE_CSV))
    assert "I enjoy the work I do" in roles.likert
    assert "work-life balance (1-5)" in roles.likert
    assert len([n for n in roles.likert if n.startswith("I would recommend")]) == 1
    assert roles.text == ["what_works_well", "what_to_improve"]
    assert roles.categorical == ["department", "tenure"]
    assert roles.ignored == ["respondent_id"]


def test_detect_roles_blank_and_numeric_non_likert() -> None:
    df = frame(
        empty=[None, "", " "] * 5,
        age=[str(20 + 3 * i) for i in range(15)],
        answers=["1", "2", "5", "3", "4"] * 3,
    )
    roles = detect_roles(df)
    assert roles.blank == ["empty"]
    assert "age" in roles.ignored  # 15 distinct values, not categorical, not 1..7
    assert "answers" in roles.likert


def test_explicit_lists_override_autodetection() -> None:
    df = frame(a=["x", "y", "z"], b=["Agree", "Agree", "Disagree"])
    roles = detect_roles(df, likert=["a"], text=["b"])
    assert list(roles.likert) == ["a"]
    assert roles.text == ["b"]


def test_ignore_and_reverse_lists() -> None:
    df = frame(a=["Agree"] * 3, b=["Agree"] * 3)
    roles = detect_roles(df, reverse=["a"], ignore=["b"])
    assert roles.likert == {"a": {"reverse": True}}
    assert roles.ignored == ["b"]


def test_unknown_column_and_conflicts_raise() -> None:
    df = frame(a=["Agree"])
    with pytest.raises(ValueError, match="not found"):
        detect_roles(df, likert=["nope"])
    with pytest.raises(ValueError, match="both likert and text"):
        detect_roles(df, likert=["a"], text=["a"])


# ---- text analysis ------------------------------------------------------


def test_tokenize_drops_stopwords_short_words_and_punctuation() -> None:
    assert tokenize("The meetings are TOO long, and I don't like them!") == [
        "meetings",
        "long",
        "like",
    ]


def test_tokenize_handles_curly_apostrophes_and_digits() -> None:
    assert tokenize("Can’t stand 9am standups") == ["stand", "standups"]


def test_summarize_text_orders_ties_alphabetically() -> None:
    s = summarize_text("q", ["pizza salad", "salad pizza", "burger", None, "  "])
    assert s.terms == [("pizza", 2), ("salad", 2), ("burger", 1)]
    assert (s.responses, s.blank) == (3, 2)


def test_all_blank_text_column() -> None:
    df = frame(q=[None, "", "  "], score=["Agree"] * 3)
    analysis = analyze_survey(df, text=["q"])
    assert analysis.texts[0].responses == 0
    assert any("no open-ended responses" in n for n in analysis.notes)
    assert "no meaningful words" in render_report(analysis)


# ---- word cloud ---------------------------------------------------------


def test_word_cloud_is_deterministic_and_places_everything() -> None:
    terms = [(f"word{chr(97 + i)}", 30 - i) for i in range(25)]
    svg1, skipped1 = build_word_cloud(terms)
    svg2, skipped2 = build_word_cloud(terms)
    assert svg1 == svg2
    assert skipped1 == skipped2 == 0
    assert svg1.count("<text ") == 25


def test_word_cloud_font_scales_with_count() -> None:
    svg, _ = build_word_cloud([("big", 50), ("small", 1)])
    sizes = {
        word: float(size)
        for size, word in re.findall(
            r'font-size="([\d.]+)".*?</title>(\w+)</text>', svg
        )
    }
    assert sizes["big"] > sizes["small"]


def test_word_cloud_skips_words_that_cannot_fit() -> None:
    svg, skipped = build_word_cloud(
        [("extraordinarily" * 3, 5), ("cat", 1)], width=100, height=60
    )
    assert skipped == 1
    assert "cat" in svg


def test_word_cloud_empty() -> None:
    assert build_word_cloud([]) == ("", 0)


def test_word_cloud_boxes_do_not_overlap() -> None:
    terms = [(f"term{i}", 40 - i) for i in range(30)]
    svg, skipped = build_word_cloud(terms)
    assert skipped == 0
    boxes = []
    for x, y, size, word in re.findall(
        r'<text x="([\d.]+)" y="([\d.]+)" font-size="([\d.]+)".*?</title>(\w+)</text>',
        svg,
    ):
        w, h = len(word) * float(size) * 0.58, float(size) * 0.9
        boxes.append(
            (float(x) - w / 2, float(y) - h / 2, float(x) + w / 2, float(y) + h / 2)
        )
    assert len(boxes) == 30
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not (a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1])


# ---- breakdown ----------------------------------------------------------


def test_breakdown_groups_and_missing_bucket() -> None:
    df = frame(
        group=["a", "a", "b", None],
        q=["Strongly agree", "Agree", "Disagree", "Neutral"],
    )
    analysis = analyze_survey(df, likert=["q"], breakdown="group")
    assert analysis.breakdown["a"]["q"] == {"n": 2, "mean": 4.5, "top2": 100.0}
    assert analysis.breakdown["b"]["q"]["top2"] == 0.0
    assert "(missing)" in analysis.breakdown


def test_breakdown_caps_number_of_groups() -> None:
    df = frame(group=[f"g{i}" for i in range(20)], q=["Agree"] * 20)
    analysis = analyze_survey(df, likert=["q"], breakdown="group")
    assert len(analysis.breakdown) == 12
    assert analysis.breakdown_groups_omitted == 8
    assert any("omitted" in n for n in analysis.notes)


def test_breakdown_validation() -> None:
    df = frame(g=["a"], q=["Agree"])
    with pytest.raises(ValueError, match="not found"):
        analyze_survey(df, breakdown="zzz")
    with pytest.raises(ValueError, match="Likert item"):
        analyze_survey(df, likert=["q"], breakdown="q")
    with pytest.raises(ValueError, match="scale"):
        analyze_survey(df, scale=4)


# ---- figures / report ---------------------------------------------------


def test_likert_figure_sorts_by_net_agreement_and_centers_neutral() -> None:
    good, _ = analyze_item("good", ["Strongly agree"] * 5)
    bad, _ = analyze_item("bad", ["Strongly disagree"] * 5)
    mixed, _ = analyze_item("mixed", ["Agree", "Neutral", "Disagree", "Neutral"])
    fig = build_likert_figure([bad, mixed, good])
    assert list(fig.data[0].y) == ["good", "mixed", "bad"]
    neutral = [t for t in fig.data if t.name == "Neutral"]
    assert len(neutral) == 2
    assert list(neutral[0].x) == [-x for x in neutral[1].x]
    assert neutral[0].x[1] == pytest.approx(-25.0)  # 50% neutral split 25 / 25
    assert sum(t.showlegend for t in neutral) == 1


def test_full_report_on_sample_data() -> None:
    analysis = analyze_survey(
        read_survey(SAMPLE_CSV),
        reverse=["I often think about leaving the company"],
        breakdown="department",
        title="Engagement",
    )
    out = render_report(analysis)
    assert out.count("plotly.js v") == 1  # bundle embedded exactly once
    assert "5-point items" in out and "7-point items" in out
    assert "Breakdown by department" in out
    assert '<svg class="cloud"' in out
    assert "reverse-coded" in out
    assert "not recognized" in out  # 'Maybe' answers surfaced
    assert "<script src" not in out  # fully offline: no external script tags


def test_report_is_deterministic() -> None:
    a = render_report(analyze_survey(read_survey(SAMPLE_CSV), breakdown="department"))
    b = render_report(analyze_survey(read_survey(SAMPLE_CSV), breakdown="department"))
    assert a == b


def test_no_likert_columns_still_renders() -> None:
    df = frame(
        q=["Slow builds and flaky tests hurt us", "Slow reviews", "Docs are missing"]
    )
    out = render_report(analyze_survey(df, text=["q"]))
    assert "0 Likert item" in out
    assert "builds" in out


# ---- XSS ----------------------------------------------------------------

PAYLOAD = "<script>XSSMARK</script><img src=x onerror=XSSMARK>"


def test_respondent_text_column_names_and_groups_are_escaped() -> None:
    likert_col, text_col = f"Q1 {PAYLOAD}", f"feedback {PAYLOAD}"
    df = pl.DataFrame(
        {
            likert_col: ["Agree", "Disagree", "Agree", "Neutral"],
            "team": [PAYLOAD, "ops", "ops", PAYLOAD],
            text_col: [PAYLOAD, "great tooling", PAYLOAD + " tooling", None],
        }
    )
    analysis = analyze_survey(
        df, likert=[likert_col], text=[text_col], breakdown="team", title=PAYLOAD
    )
    out = render_report(analysis)
    assert "<script>XSSMARK" not in out
    assert "<img src=x onerror=XSSMARK>" not in out
    assert (
        "&lt;img src=x onerror=XSSMARK&gt;" in out
    )  # sample response rendered as inert text
    # Plotly JSON-escapes '<' and '&', so the chart labels can't break out either.
    assert "onerror=XSSMARK" not in out.replace("&lt;img src=x onerror=XSSMARK&gt;", "")


def test_word_cloud_escapes_terms() -> None:
    svg, _ = build_word_cloud([("<b>x</b>", 3)])
    assert "<b>" not in svg
    assert "&lt;b&gt;" in svg


# ---- CLI ----------------------------------------------------------------


def test_cli_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    code = main(
        [str(SAMPLE_CSV), "-o", str(out), "--config", str(SAMPLE_DIR / "config.json")]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "Employee Engagement Survey" in text
    assert "Breakdown by department" in text


def test_cli_flags_override_config(tmp_path: Path) -> None:
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"title": "From config", "breakdown": "department"}))
    out = tmp_path / "r.html"
    args = [str(SAMPLE_CSV), "-o", str(out), "--config", str(config)]
    main([*args, "--title", "From flag", "--breakdown", "tenure"])
    text = out.read_text(encoding="utf-8")
    assert "From flag" in text and "From config" not in text
    assert "Breakdown by tenure" in text


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main([str(SAMPLE_CSV), "-o", str(tmp_path / "x.html"), "--likert", "nope"]) == 2
    )
    assert "not found" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.csv")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"bogus": 1}')
    assert main([str(SAMPLE_CSV), "--config", str(bad)]) == 2
    assert "unknown config key" in capsys.readouterr().err
