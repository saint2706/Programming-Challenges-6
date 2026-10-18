//! The pivot engine on small tables whose answers can be checked by eye.

use pivot_core::pivot::pivot;
use pivot_core::spec::{HeaderKind, PivotResult, PivotSpec};
use pivot_core::table::{LoadOptions, Table};
use serde_json::{Value, json};

const SALES: &str = "region,product,month,units,price,when\n\
north,apple,jan,10,2.0,2024-01-05\n\
north,apple,feb,20,2.0,2024-02-10\n\
north,pear,jan,5,3.0,2024-01-20\n\
south,apple,jan,7,2.5,2024-01-25\n\
south,pear,feb,12,3.5,2024-03-02\n\
south,pear,mar,3,3.5,2024-03-15\n\
east,apple,mar,9,2.0,2024-03-30\n\
east,,feb,4,,2024-02-02\n";

fn table() -> Table {
    Table::from_csv(SALES.as_bytes(), LoadOptions::default()).unwrap()
}

fn run(v: Value) -> PivotResult {
    let spec: PivotSpec = serde_json::from_value(v).unwrap();
    pivot(&table(), &spec).unwrap()
}

fn err(v: Value) -> String {
    let spec: PivotSpec = serde_json::from_value(v).unwrap();
    pivot(&table(), &spec).unwrap_err().to_string()
}

fn row_labels(r: &PivotResult) -> Vec<String> {
    r.rows
        .iter()
        .map(|h| match h.kind {
            HeaderKind::Total => "TOTAL".to_string(),
            HeaderKind::Subtotal => format!(
                "{} subtotal",
                h.labels
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("/")
            ),
            HeaderKind::Leaf => h
                .labels
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
                .join("/"),
        })
        .collect()
}

fn col_labels(r: &PivotResult) -> Vec<String> {
    r.cols
        .iter()
        .map(|h| match h.kind {
            HeaderKind::Total => "TOTAL".to_string(),
            _ => h
                .labels
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
                .join("/"),
        })
        .collect()
}

/// `cells[row][col][0]` as a plain grid of numbers (None -> NaN) for comparing.
fn grid(r: &PivotResult, measure: usize) -> Vec<Vec<f64>> {
    r.cells
        .iter()
        .map(|row| row.iter().map(|c| c[measure].unwrap_or(f64::NAN)).collect())
        .collect()
}

fn same(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
    for (ra, rb) in a.iter().zip(b) {
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(rb) {
            assert!(
                (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-9,
                "{a:?} vs {b:?}"
            );
        }
    }
}

const N: f64 = f64::NAN;

#[test]
fn sum_by_region_and_product_with_totals() {
    let r = run(json!({
        "rows": [{"field": "region"}],
        "cols": [{"field": "product"}],
        "values": [{"field": "units", "agg": "sum"}]
    }));
    assert_eq!(row_labels(&r), ["east", "north", "south", "TOTAL"]);
    assert_eq!(col_labels(&r), ["apple", "pear", "(blank)", "TOTAL"]);
    same(
        &grid(&r, 0),
        &[
            vec![9.0, N, 4.0, 13.0],
            vec![30.0, 5.0, N, 35.0],
            vec![7.0, 15.0, N, 22.0],
            vec![46.0, 20.0, 4.0, 70.0],
        ],
    );
    assert_eq!((r.rows_used, r.rows_total), (8, 8));
    assert_eq!(r.measures[0].label, "sum(units)");
}

#[test]
fn totals_can_be_switched_off() {
    let r = run(json!({
        "rows": [{"field": "region"}], "cols": [{"field": "product"}],
        "values": [{"field": "units"}],
        "options": {"totals_row": false, "totals_col": false}
    }));
    assert_eq!(row_labels(&r), ["east", "north", "south"]);
    assert_eq!(col_labels(&r), ["apple", "pear", "(blank)"]);
}

#[test]
fn no_column_field_gives_one_column() {
    let r = run(json!({"rows": [{"field": "region"}], "values": [{"field": "units"}]}));
    assert_eq!(r.cols.len(), 1);
    same(
        &grid(&r, 0),
        &[vec![13.0], vec![35.0], vec![22.0], vec![70.0]],
    );
}

#[test]
fn no_fields_at_all_is_one_grand_total_and_count_is_the_default() {
    let r = run(json!({}));
    assert_eq!((r.rows.len(), r.cols.len()), (1, 1));
    assert_eq!(r.cells[0][0][0], Some(8.0));
    assert_eq!(r.measures[0].label, "count");
}

#[test]
fn every_aggregation_agrees_with_hand_counting_on_one_group() {
    // north/apple units: 10, 20
    let spec = |agg: &str| {
        json!({"rows": [{"field": "region"}], "values": [{"field": "units", "agg": agg}],
        "filters": [{"field": "region", "op": "eq", "value": "north"}], "options": {"totals_row": false}})
    };
    let cell = |agg: &str| run(spec(agg)).cells[0][0][0];
    assert_eq!(cell("sum"), Some(35.0));
    assert_eq!(cell("count"), Some(3.0));
    assert_eq!(cell("distinct"), Some(3.0));
    assert_eq!(cell("mean"), Some(35.0 / 3.0));
    assert_eq!(cell("median"), Some(10.0));
    assert_eq!(cell("min"), Some(5.0));
    assert_eq!(cell("max"), Some(20.0));
    assert_eq!(cell("first"), Some(10.0));
    assert_eq!(cell("last"), Some(5.0));
    assert_eq!(cell("mode"), Some(5.0)); // all different: the smallest
    let var = ((10.0f64 - 35.0 / 3.0).powi(2)
        + (20.0f64 - 35.0 / 3.0).powi(2)
        + (5.0f64 - 35.0 / 3.0).powi(2))
        / 2.0;
    assert!((cell("var").unwrap() - var).abs() < 1e-9);
    assert!((cell("std").unwrap() - var.sqrt()).abs() < 1e-9);
}

#[test]
fn weighted_mean() {
    let r = run(json!({"rows": [{"field": "region"}],
        "values": [{"field": "price", "agg": "weighted_mean", "weight": "units"}], "options": {"totals_row": false}}));
    // north: (2*10 + 2*20 + 3*5) / 35
    assert!((r.cells[1][0][0].unwrap() - 75.0 / 35.0).abs() < 1e-12);
    // east: the row with no price carries no weight in the mean
    assert!((r.cells[0][0][0].unwrap() - 2.0).abs() < 1e-12);
}

#[test]
fn counting_a_field_skips_missing_values_but_counting_rows_does_not() {
    let r = run(json!({"values": [
        {"field": "product", "agg": "count", "label": "with product"},
        {"agg": "count", "label": "rows"},
        {"field": "product", "agg": "distinct", "label": "kinds"},
        {"field": "price", "agg": "count", "label": "priced"}
    ]}));
    let c = &r.cells[0][0];
    assert_eq!(
        (c[0], c[1], c[2], c[3]),
        (Some(7.0), Some(8.0), Some(2.0), Some(7.0))
    );
}

#[test]
fn subtotals_sit_where_asked_and_are_exact() {
    let spec = |pos: &str| {
        json!({
            "rows": [{"field": "region"}, {"field": "product"}],
            "values": [{"field": "units"}, {"field": "units", "agg": "mean", "label": "avg"}],
            "options": {"subtotals": true, "subtotal_position": pos, "totals_col": false}
        })
    };
    let bottom = run(spec("bottom"));
    assert_eq!(
        row_labels(&bottom),
        [
            "east/apple",
            "east/(blank)",
            "east subtotal",
            "north/apple",
            "north/pear",
            "north subtotal",
            "south/apple",
            "south/pear",
            "south subtotal",
            "TOTAL"
        ]
    );
    // the north subtotal mean is the mean of the three rows (35/3), not the mean of the two group means
    assert!((bottom.cells[5][0][1].unwrap() - 35.0 / 3.0).abs() < 1e-12);
    assert_eq!(bottom.cells[5][0][0], Some(35.0));
    assert_eq!(bottom.rows[2].kind, HeaderKind::Subtotal);
    let top = run(spec("top"));
    assert_eq!(row_labels(&top)[0], "east subtotal");
    assert_eq!(row_labels(&top)[1], "east/apple");
    assert_eq!(top.rows[0].depth, 1);
}

#[test]
fn subtotals_need_two_fields() {
    let r = run(
        json!({"rows": [{"field": "region"}], "values": [{"field": "units"}], "options": {"subtotals": true}}),
    );
    assert_eq!(row_labels(&r), ["east", "north", "south", "TOTAL"]);
}

#[test]
fn sort_by_value_and_by_label_descending() {
    let by_value = run(
        json!({"rows": [{"field": "region", "sort": {"by": "value", "measure": "sum(units)"}}],
        "values": [{"field": "units"}], "options": {"totals_row": false, "totals_col": false}}),
    );
    assert_eq!(row_labels(&by_value), ["north", "south", "east"]);
    let ascending = run(
        json!({"rows": [{"field": "region", "sort": {"by": "value", "measure": "sum(units)", "desc": false}}],
        "values": [{"field": "units"}], "options": {"totals_row": false, "totals_col": false}}),
    );
    assert_eq!(row_labels(&ascending), ["east", "south", "north"]);
    let desc = run(
        json!({"rows": [{"field": "region", "sort": {"by": "label", "desc": true}}],
        "values": [{"field": "units"}], "options": {"totals_row": false, "totals_col": false}}),
    );
    assert_eq!(row_labels(&desc), ["south", "north", "east"]);
}

#[test]
fn blank_stays_last_even_when_sorting_descending() {
    let r = run(
        json!({"cols": [{"field": "product", "sort": {"by": "label", "desc": true}}], "values": [{"field": "units"}]}),
    );
    assert_eq!(col_labels(&r), ["pear", "apple", "(blank)", "TOTAL"]);
}

#[test]
fn top_n_keeps_the_best_groups_within_each_parent() {
    let r = run(json!({
        "rows": [{"field": "region", "top": {"n": 2, "measure": "sum(units)"}}, {"field": "product", "top": {"n": 1, "measure": "sum(units)"}}],
        "values": [{"field": "units"}], "options": {"totals_row": false, "totals_col": false}
    }));
    assert_eq!(row_labels(&r), ["north/apple", "south/pear"]);
}

#[test]
fn filters_and_hide_blank() {
    let r = run(
        json!({"rows": [{"field": "product"}], "values": [{"field": "units"}],
        "filters": [{"field": "units", "op": "ge", "value": 5}]}),
    );
    assert_eq!(row_labels(&r), ["apple", "pear", "TOTAL"]);
    assert_eq!(r.rows_used, 6);
    let r = run(
        json!({"rows": [{"field": "product", "hide_blank": true}], "values": [{"field": "units"}]}),
    );
    assert_eq!(row_labels(&r), ["apple", "pear", "TOTAL"]);
    assert_eq!(r.rows_used, 7);
    assert_eq!(r.cells[2][0][0], Some(66.0));
}

#[test]
fn no_matching_rows_is_an_empty_result_with_a_warning() {
    let r = run(
        json!({"rows": [{"field": "region"}], "values": [{"field": "units"}],
        "filters": [{"field": "region", "op": "eq", "value": "mars"}]}),
    );
    assert!(r.rows.is_empty() && r.cells.is_empty());
    assert_eq!(r.warnings, ["no rows match the filters"]);
}

#[test]
fn grouping_dates_numbers_and_bands() {
    let r = run(
        json!({"rows": [{"field": "when", "group": {"kind": "month"}}], "values": [{"agg": "count"}], "options": {"totals_row": false, "totals_col": false}}),
    );
    assert_eq!(row_labels(&r), ["Jan", "Feb", "Mar"]);
    same(&grid(&r, 0), &[vec![3.0], vec![2.0], vec![3.0]]);
    assert_eq!(r.row_fields, ["when (month)"]);
    let r = run(
        json!({"rows": [{"field": "units", "group": {"kind": "bin", "width": 10.0}}], "values": [{"agg": "count"}], "options": {"totals_row": false, "totals_col": false}}),
    );
    assert_eq!(row_labels(&r), ["[0, 10)", "[10, 20)", "[20, 30)"]);
    same(&grid(&r, 0), &[vec![5.0], vec![2.0], vec![1.0]]);
    let r = run(
        json!({"rows": [{"field": "when", "group": {"kind": "year_month"}}, {"field": "when", "group": {"kind": "weekday"}}],
        "values": [{"agg": "count"}], "options": {"totals_row": false, "totals_col": false}}),
    );
    assert!(row_labels(&r)[0].starts_with("2024-01/"));
}

#[test]
fn show_values_as_percentages() {
    let r = run(
        json!({"rows": [{"field": "region"}], "cols": [{"field": "product", "hide_blank": true}],
        "values": [
            {"field": "units", "label": "grand", "show_as": {"kind": "pct_grand"}},
            {"field": "units", "label": "row", "show_as": {"kind": "pct_row"}},
            {"field": "units", "label": "col", "show_as": {"kind": "pct_col"}}
        ]}),
    );
    // rows: east, north, south, TOTAL; cols: apple, pear, TOTAL; total units 66
    let c = |ri: usize, ci: usize, m: usize| r.cells[ri][ci][m].unwrap();
    assert!((c(1, 0, 0) - 30.0 / 66.0).abs() < 1e-12);
    assert!((c(1, 0, 1) - 30.0 / 35.0).abs() < 1e-12);
    assert!((c(1, 0, 2) - 30.0 / 46.0).abs() < 1e-12);
    assert!(
        (c(3, 2, 0) - 1.0).abs() < 1e-12
            && (c(3, 2, 1) - 1.0).abs() < 1e-12
            && (c(3, 2, 2) - 1.0).abs() < 1e-12
    );
    assert!((c(0, 2, 1) - 1.0).abs() < 1e-12);
    assert!(r.measures.iter().all(|m| m.percent));
}

#[test]
fn percent_of_parent_and_index() {
    let r = run(
        json!({"rows": [{"field": "region"}, {"field": "product", "hide_blank": true}],
        "values": [{"field": "units", "show_as": {"kind": "pct_parent_row"}}],
        "options": {"subtotals": true, "totals_col": false, "subtotal_position": "bottom"}}),
    );
    // north/apple is 30 of the north subtotal 35
    let i = row_labels(&r)
        .iter()
        .position(|l| l == "north/apple")
        .unwrap();
    assert!((r.cells[i][0][0].unwrap() - 30.0 / 35.0).abs() < 1e-12);
    let j = row_labels(&r)
        .iter()
        .position(|l| l == "north subtotal")
        .unwrap();
    assert!((r.cells[j][0][0].unwrap() - 35.0 / 66.0).abs() < 1e-12); // the parent of a subtotal is the grand total

    let idx = run(
        json!({"rows": [{"field": "region"}], "cols": [{"field": "product", "hide_blank": true}],
        "values": [{"field": "units", "show_as": {"kind": "index"}}]}),
    );
    // index = cell * grand / (row total * col total) for north/apple
    assert!((idx.cells[1][0][0].unwrap() - 30.0 * 66.0 / (35.0 * 46.0)).abs() < 1e-12);
}

#[test]
fn difference_running_total_and_rank_along_an_axis() {
    let r = run(
        json!({"rows": [{"field": "month", "sort": {"by": "label"}}],
        "values": [
            {"field": "units", "label": "sum"},
            {"field": "units", "label": "diff", "show_as": {"kind": "diff", "along": "rows"}},
            {"field": "units", "label": "pct", "show_as": {"kind": "pct_diff", "along": "rows"}},
            {"field": "units", "label": "run", "show_as": {"kind": "running", "along": "rows"}},
            {"field": "units", "label": "rank", "show_as": {"kind": "rank", "along": "rows"}}
        ], "options": {"totals_row": false, "totals_col": false}}),
    );
    // months sort by label: feb, jan, mar with sums 36, 22, 12
    assert_eq!(row_labels(&r), ["feb", "jan", "mar"]);
    let col = |m: usize| -> Vec<Option<f64>> { r.cells.iter().map(|row| row[0][m]).collect() };
    assert_eq!(col(0), [Some(36.0), Some(22.0), Some(12.0)]);
    assert_eq!(col(1), [None, Some(-14.0), Some(-10.0)]);
    assert_eq!(
        col(2)
            .iter()
            .map(|v| v.map(|x| (x * 1000.0).round()))
            .collect::<Vec<_>>(),
        [None, Some(-389.0), Some(-455.0)]
    );
    assert_eq!(col(3), [Some(36.0), Some(58.0), Some(70.0)]);
    assert_eq!(col(4), [Some(1.0), Some(2.0), Some(3.0)]);
}

#[test]
fn calculated_fields_work_in_every_cell_including_totals() {
    let r = run(json!({"rows": [{"field": "region"}],
        "values": [
            {"field": "units", "label": "units"},
            {"agg": "count", "label": "orders"},
            {"calc": "units / orders", "label": "units per order"},
            {"calc": "round([units per order] * 100, 0)", "label": "x100"}
        ], "options": {"totals_col": false}}));
    // east: 13 units / 2 orders; total: 70 / 8
    assert_eq!(r.cells[0][0][2], Some(6.5));
    assert_eq!(r.cells[3][0][2], Some(8.75));
    assert_eq!(r.cells[3][0][3], Some(875.0));
    assert_eq!(r.measures[2].agg, "calc");
}

#[test]
fn a_calculated_field_that_divides_by_zero_leaves_the_cell_empty() {
    let r = run(json!({"rows": [{"field": "region"}], "values": [
        {"field": "units", "label": "u"}, {"calc": "u / 0", "label": "bad"}]}));
    assert!(r.cells.iter().all(|row| row[0][1].is_none()));
}

#[test]
fn measure_names_are_made_unique() {
    let r = run(json!({"values": [{"field": "units"}, {"field": "units"}]}));
    assert_eq!(r.measures[0].label, "sum(units)");
    assert_eq!(r.measures[1].label, "sum(units) (2)");
}

#[test]
fn mistakes_are_explained() {
    assert!(err(json!({"rows": [{"field": "nope"}]})).contains("no field called 'nope'"));
    assert!(err(json!({"values": [{"field": "product", "agg": "sum"}]})).contains("it is text"));
    assert!(err(json!({"values": [{"agg": "sum"}]})).contains("needs a field"));
    assert!(
        err(json!({"values": [{"field": "price", "agg": "weighted_mean"}]}))
            .contains("needs a \"weight\"")
    );
    assert!(
        err(json!({"values": [{"calc": "ghost + 1"}]}))
            .contains("'ghost', which is not an earlier measure")
    );
    assert!(err(json!({"values": [{"calc": "1 +"}]})).contains("formula"));
    assert!(err(json!({"rows": [{"field": "region", "sort": {"by": "value", "measure": "ghost"}}], "values": [{"field": "units"}]})).contains("no measure called 'ghost'"));
    assert!(err(json!({"rows": [{"field": "region"}], "cols": [{"field": "product"}], "values": [{"field": "units"}], "options": {"max_cells": 3}})).contains("over the limit"));
    assert!(
        err(json!({"rows": [{"field": "region", "group": {"kind": "month"}}]}))
            .contains("not a date")
    );
}

#[test]
fn unknown_keys_in_a_request_are_rejected_not_ignored() {
    let e = serde_json::from_value::<PivotSpec>(json!({"rows": [{"field": "region", "sortt": 1}]}))
        .unwrap_err();
    assert!(e.to_string().contains("unknown field `sortt`"), "{e}");
}

#[test]
fn rows_and_columns_may_use_the_same_field_twice_with_different_grouping() {
    let r = run(
        json!({"rows": [{"field": "when", "group": {"kind": "month"}}], "cols": [{"field": "when", "group": {"kind": "weekday"}}],
        "values": [{"agg": "count"}]}),
    );
    let total: f64 = r.cells.last().unwrap().last().unwrap()[0].unwrap();
    assert_eq!(total, 8.0);
}
