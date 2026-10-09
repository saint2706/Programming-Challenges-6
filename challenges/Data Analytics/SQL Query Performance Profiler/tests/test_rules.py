import pytest
from helpers import op, scan, wrap
from sql_profiler import lab
from sql_profiler.plan import parse_profile, profile
from sql_profiler.rules import Thresholds, diagnose


def found(root, **thresholds):
    plan = parse_profile(wrap(root))
    return diagnose(plan, Thresholds(**thresholds)) if thresholds else diagnose(plan)


def rules_of(findings):
    return [f.rule for f in findings]


def only(findings, rule):
    hits = [f for f in findings if f.rule == rule]
    assert len(hits) == 1, rules_of(findings)
    return hits[0]


# --- estimates ---------------------------------------------------------------------------------


def test_misestimate_severity_follows_the_size_of_the_miss():
    assert (
        only(found(scan("t", 20_000, est=1_000)), "estimate.off").severity == "warn"
    )  # 20x
    f = only(found(scan("t", 200_000, est=1_000)), "estimate.off")  # 200x
    assert f.severity == "critical"
    assert (
        "200,000" in f.detail and "1,000" in f.detail
    )  # the evidence is in the message


def test_a_small_miss_or_a_tiny_table_is_not_a_finding():
    assert "estimate.off" not in rules_of(found(scan("t", 5_000, est=1_000)))  # 5x
    assert "estimate.off" not in rules_of(
        found(scan("t", 90, est=3))
    )  # 30x but 90 rows


def test_no_estimate_no_finding():
    assert "estimate.off" not in rules_of(found(op("ORDER_BY", rows=1_000_000)))


def test_a_scan_with_only_a_dynamic_filter_is_not_blamed_on_the_estimator():
    root = scan(
        "invoice_lines",
        100,
        scanned=1_000_000,
        est=1_000_000,
        Dynamic_Filters="optional: a>=1",
    )
    assert "estimate.off" not in rules_of(found(root))


def test_delim_joins_do_not_report_comparable_rows():
    assert "estimate.off" not in rules_of(
        found(op("LEFT_DELIM_JOIN", rows=0, est=50_000))
    )


# --- filters and scans -------------------------------------------------------------------------


def test_function_on_the_filter_column_is_named():
    f = only(
        found(
            scan(
                "invoices",
                24_000,
                scanned=150_000,
                Filters="(strftime(invoice_date, '%Y') = '2011')",
            )
        ),
        "filter.function_on_column",
    )
    assert f.severity == "warn" and "strftime" in f.title
    assert "150,000" in f.detail


def test_function_on_a_mid_size_scan_is_only_info_and_on_a_small_one_silent():
    filt = "(substr(invoice, 1, 3) = '536')"
    assert (
        only(
            found(scan("t", 100, scanned=50_000, Filters=filt)),
            "filter.function_on_column",
        ).severity
        == "info"
    )
    assert "filter.function_on_column" not in rules_of(
        found(scan("t", 100, scanned=5_000, Filters=filt))
    )


@pytest.mark.parametrize(
    "plain",
    [
        "invoice>='536' AND invoice<'537'",
        "country IN ('France', 'Spain')",
        "price>1.0 OR quantity<0",
        "(customer_id IS NOT NULL)",
        "description BETWEEN 'A' AND 'C'",
    ],
)
def test_plain_column_filters_are_not_function_calls(plain):
    assert "filter.function_on_column" not in rules_of(
        found(scan("t", 10, scanned=500_000, Filters=plain))
    )


def test_a_filter_that_stays_above_its_input_and_drops_nearly_everything():
    f = only(found(op("FILTER", rows=500, children=[scan("t", 80_000)])), "filter.late")
    assert "99%" in f.title


def test_a_filter_that_keeps_most_rows_is_fine():
    assert "filter.late" not in rules_of(
        found(op("FILTER", rows=60_000, children=[scan("t", 80_000)]))
    )


def test_a_scan_that_reads_a_million_to_keep_a_few():
    f = only(
        found(scan("t", 800, scanned=1_000_000, Filters="price>50.0")),
        "scan.reads_much_keeps_little",
    )
    assert "0.08%" in f.detail


def test_a_selective_join_driven_scan_is_not_a_filter_finding():
    assert "scan.reads_much_keeps_little" not in rules_of(
        found(scan("t", 800, scanned=1_000_000, Dynamic_Filters="optional: a>=1"))
    )


# --- joins -------------------------------------------------------------------------------------


def test_cartesian_product_thresholds():
    small = op(
        "CROSS_PRODUCT", rows=900_000, children=[scan("a", 900), scan("b", 1000)]
    )
    big = op(
        "CROSS_PRODUCT",
        rows=100_000_000,
        children=[scan("a", 10_000), scan("b", 10_000)],
    )
    tiny = op("CROSS_PRODUCT", rows=100, children=[scan("a", 10), scan("b", 10)])
    assert "join.no_equality_key" not in rules_of(found(small))  # 900k pairs
    assert (
        only(
            found(
                op(
                    "CROSS_PRODUCT",
                    rows=1_000_000,
                    children=[scan("a", 1000), scan("b", 1000)],
                )
            ),
            "join.no_equality_key",
        ).severity
        == "warn"
    )
    f = only(found(big), "join.no_equality_key")
    assert f.severity == "critical" and "cartesian product" in f.title
    assert "join.no_equality_key" not in rules_of(found(tiny))


def test_a_range_join_is_a_join_without_an_equality_key():
    root = op(
        "PIECEWISE_MERGE_JOIN",
        rows=4_000_000,
        children=[scan("a", 3000), scan("b", 3000)],
        Conditions="d < d",
    )
    assert "join.no_equality_key" in rules_of(found(root))


def test_hash_join_with_a_bigger_build_side():
    # children[0] probes, children[1] is built into the hash table
    bad = op(
        "HASH_JOIN", rows=10, children=[scan("small", 50_000), scan("big", 200_000)]
    )
    good = op(
        "HASH_JOIN", rows=10, children=[scan("big", 200_000), scan("small", 50_000)]
    )
    close = op("HASH_JOIN", rows=10, children=[scan("a", 150_000), scan("b", 200_000)])
    assert "join.big_build_side" in rules_of(found(bad))
    assert "join.big_build_side" not in rules_of(found(good))
    assert "join.big_build_side" not in rules_of(found(close))


def test_join_that_multiplies_rows():
    root = op(
        "HASH_JOIN", rows=5_000_000, children=[scan("a", 100_000), scan("b", 60_000)]
    )
    f = only(found(root), "join.row_explosion")
    assert "50x" in f.title
    assert "join.row_explosion" not in rules_of(
        found(
            op(
                "HASH_JOIN",
                rows=150_000,
                children=[scan("a", 100_000), scan("b", 60_000)],
            )
        )
    )


# --- the rest ----------------------------------------------------------------------------------


def test_full_sort_only_above_the_row_threshold_and_louder_when_it_dominates():
    heavy = op(
        "ORDER_BY", rows=1_000_000, time=0.9, children=[scan("t", 1_000_000, time=0.1)]
    )
    light = op(
        "ORDER_BY", rows=1_000_000, time=0.01, children=[scan("t", 1_000_000, time=0.9)]
    )
    assert only(found(heavy), "sort.full").severity == "warn"
    assert only(found(light), "sort.full").severity == "info"
    assert "sort.full" not in rules_of(found(op("ORDER_BY", rows=50_000, time=0.9)))
    assert "sort.full" not in rules_of(
        found(op("TOP_N", rows=10, time=0.9, children=[scan("t", 1_000_000)]))
    )


def test_the_same_table_scanned_twice():
    root = op(
        "HASH_JOIN",
        rows=10,
        children=[
            scan("invoices", 20_000, Filters="a>1"),
            scan("invoices", 20_000, Filters="b>1"),
        ],
    )
    f = only(found(root), "scan.repeated")
    assert "invoices is scanned 2 times" == f.title


def test_a_row_fetch_by_dynamic_filter_is_not_a_repeated_scan():
    # TOP_N late materialization: the second pass reads by row id and returns a sliver
    fetch = scan(
        "invoice_lines", 11, scanned=1_000_000, est=1_000_000, Dynamic_Filters=""
    )
    root = op(
        "HASH_JOIN",
        rows=10,
        children=[
            fetch,
            op("TOP_N", rows=10, children=[scan("invoice_lines", 1_000_000)]),
        ],
    )
    assert "scan.repeated" not in rules_of(found(root))


def test_group_by_that_barely_reduces():
    root = op("HASH_GROUP_BY", rows=900_000, children=[scan("t", 1_000_000)])
    assert "aggregate.barely_reduces" in rules_of(found(root))
    assert "aggregate.barely_reduces" not in rules_of(
        found(op("HASH_GROUP_BY", rows=34, children=[scan("t", 1_000_000)]))
    )


def test_a_decorrelated_subquery_is_reported_as_information():
    f = only(
        found(op("LEFT_DELIM_JOIN", rows=0, children=[scan("a", 10), scan("b", 10)])),
        "subquery.decorrelated",
    )
    assert f.severity == "info"


def test_dominant_operator_needs_real_time_to_matter():
    big = op("HASH_GROUP_BY", rows=10, time=0.09, children=[scan("t", 100, time=0.01)])
    assert only(found(big), "time.dominant_operator").title.startswith(
        "HASH_GROUP_BY is 90%"
    )
    microscopic = op(
        "HASH_GROUP_BY", rows=10, time=0.0009, children=[scan("t", 100, time=0.0001)]
    )
    assert "time.dominant_operator" not in rules_of(found(microscopic))


def test_findings_are_most_severe_first_then_in_plan_order():
    root = op(
        "HASH_JOIN",
        rows=5_000_000,
        children=[
            scan("a", 100_000, scanned=100_000, Filters="(lower(x) = 'a')"),
            scan("b", 200_000, est=100),
        ],
    )
    severities = [f.severity for f in found(root)]
    assert severities == sorted(severities, key=("critical", "warn", "info").index)


def test_thresholds_can_be_tightened():
    root = scan("t", 5_000, est=1_000)  # 5x
    assert "estimate.off" not in rules_of(found(root))
    assert "estimate.off" in rules_of(
        found(root, misestimate_warn=2.0, misestimate_min_rows=100)
    )


def test_a_clean_plan_has_no_findings():
    assert found(op("TOP_N", rows=10, children=[scan("t", 100)])) == []


# --- real plans, on the committed sample (thresholds scaled down to its size) --------------------

SMALL = {
    "scan_min_rows": 1_000,
    "wide_scan_rows": 10_000,
    "sort_rows": 10_000,
    "dominant_min_s": 0.0,
    "pairwise_warn": 1e5,
    "explosion_ratio": 3.0,
    "explosion_min_rows": 1_000,
}


@pytest.mark.parametrize(
    ("case_id", "rule"),
    [
        ("function-on-key", "filter.function_on_column"),
        ("sort-vs-top-n", "sort.full"),
        ("or-join-vs-union", "join.no_equality_key"),
        ("selfjoin-vs-window", "join.row_explosion"),
    ],
)
def test_the_slow_side_of_an_experiment_trips_its_rule_and_the_fast_side_does_not(
    con, case_id, rule
):
    case = next(c for c in lab.LAB if c.id == case_id)
    slow = diagnose(profile(con.cursor(), case.a_sql), Thresholds(**SMALL))
    fast = diagnose(profile(con.cursor(), case.b_sql), Thresholds(**SMALL))
    assert rule in rules_of(slow)
    assert rule not in rules_of(fast)
