# the SVG parsed here is generated and escaped by the code under test, never external input
import xml.etree.ElementTree as ET
from itertools import pairwise

import pytest
from helpers import op, scan, wrap
from sql_profiler import lab, render
from sql_profiler.plan import parse_profile, profile
from sql_profiler.rules import Finding, diagnose


def plan_of(root, sql="select 1", **kw):
    return parse_profile(wrap(root, **kw), sql=sql)


def tree():
    return plan_of(
        op(
            "TOP_N",
            rows=10,
            time=0.001,
            Top="10",
            children=[
                op(
                    "HASH_JOIN",
                    rows=1000,
                    est=50,
                    time=0.08,
                    Join_Type="INNER",
                    Conditions="a = b",
                    children=[
                        scan("invoice_lines", 40_000, scanned=1_000_000, time=0.02),
                        scan("products", 300, time=0.001),
                    ],
                )
            ],
        )
    )


def svg_root(plan, findings=()):
    return ET.fromstring(render.plan_svg(plan, findings))


def test_the_svg_is_well_formed_and_draws_every_operator_and_edge():
    plan = tree()
    root = svg_root(plan)
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}g")) == 4  # one group per operator
    assert len(root.findall(f"{ns}path")) == 3  # one edge per parent-child link
    text = ET.tostring(root, encoding="unicode")
    for label in ("TOP_N", "HASH_JOIN", "invoice_lines", "products"):
        assert label in text


def test_each_box_says_rows_estimate_time_and_how_far_off_it_is():
    text = ET.tostring(svg_root(tree()), encoding="unicode")
    assert "rows 1,000 (est 50)" in text
    assert "20x off" in text  # 1001/51
    assert "80.0 ms" in text


def test_tooltips_carry_the_full_detail():
    text = ET.tostring(svg_root(tree()), encoding="unicode")
    assert "rows scanned 1,000,000" in text
    assert "Conditions: a = b" in text


def test_boxes_do_not_overlap_and_parents_sit_above_their_children():
    plan = plan_of(
        op(
            "HASH_JOIN",
            rows=1,
            children=[
                op("HASH_JOIN", rows=1, children=[scan("a", 1), scan("b", 1)]),
                scan("c", 1),
                scan("d", 1),
            ],
        )
    )
    centres, total = render._layout(plan.root)
    by_depth: dict[int, list[float]] = {}
    for n in plan.nodes:
        by_depth.setdefault(n.depth, []).append(centres[n.id])
        assert (
            0 <= centres[n.id] - render.BOX_W / 2
            and centres[n.id] + render.BOX_W / 2 <= total
        )
    for xs in by_depth.values():
        xs.sort()
        assert all(b - a >= render.BOX_W for a, b in pairwise(xs))
    join = plan.root
    kids = [centres[c.id] for c in join.children]
    assert min(kids) <= centres[join.id] <= max(kids)


def test_a_critical_finding_outlines_its_box_in_red_and_info_does_not():
    plan = tree()
    crit = Finding("x", "critical", 1, "t", "d", "s")
    info = Finding("y", "info", 2, "t", "d", "s")
    svg = render.plan_svg(plan, [crit, info])
    assert svg.count('stroke="var(--crit)"') == 1
    assert 'stroke="var(--warn)"' not in svg


def test_the_estimate_badge_is_not_shown_for_a_dynamic_filter_scan():
    plan = plan_of(
        scan(
            "t", 100, scanned=1_000_000, est=1_000_000, Dynamic_Filters="optional: a>=1"
        )
    )
    assert "off" not in ET.tostring(svg_root(plan), encoding="unicode").replace(
        "Dynamic", ""
    )


def test_query_text_cannot_inject_markup_into_the_report():
    evil = "</pre><script>alert(1)</script><img src=x onerror=alert(2)>"
    plan = plan_of(
        scan("<b>t</b>", 20_000, scanned=60_000, Filters=f"(lower({evil}) = 'x')"),
        sql=f"select '{evil}'",
    )
    html = render.report_html(plan, diagnose(plan), title=evil)
    assert "<script" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
    assert html.count("<pre>") == 1  # the injected </pre> did not close ours early


def test_text_tree_has_one_indented_line_per_operator():
    lines = render.text_tree(tree()).splitlines()
    assert [len(l) - len(l.lstrip()) for l in lines] == [0, 2, 4, 4]
    assert "rows=1,000 (est 50)" in lines[1]


def test_operator_rows_are_sorted_by_time():
    rows = render.operator_rows(tree())
    assert next(r["operator"] for r in rows) == "HASH_JOIN"
    assert [r["self_ms"] for r in rows] == sorted(
        (r["self_ms"] for r in rows), reverse=True
    )
    assert rows[0]["share_pct"] == pytest.approx(80 / 102 * 100, abs=0.1)


def test_the_report_is_one_self_contained_page():
    plan = tree()
    html = render.report_html(plan, diagnose(plan), title="My query")
    assert html.startswith("<!doctype html>") and "<title>My query</title>" in html
    for external in ("http://", "https://", "<link", "src="):
        assert external not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "Findings" in html and "Operators by self time" in html


def test_a_report_for_a_clean_plan_says_so():
    html = render.report_html(plan_of(scan("t", 10)), [], title="t")
    assert "No findings" in html


def test_a_real_plan_renders_end_to_end(con):
    case = lab.LAB[-3]
    plan = profile(con.cursor(), case.a_sql)
    html = render.report_html(plan, diagnose(plan))
    svg_start = html.index("<svg")
    ET.fromstring(html[svg_start : html.index("</svg>") + 6])
