//! The engine against a deliberately naive reference: for every header of every random pivot, filter
//! the rows with a plain loop and aggregate them with the textbook formula. Whatever the hash
//! aggregation, prefix merging and subtotal logic do, they must agree.

use pivot_core::pivot::pivot;
use pivot_core::spec::{HeaderKind, PivotSpec};
use pivot_core::table::{LoadOptions, Table};
use proptest::prelude::*;
use serde_json::json;

#[derive(Debug, Clone)]
struct Row {
    a: Option<usize>,
    b: Option<usize>,
    n: Option<i32>,
    f: Option<i32>, // tenths
}

const A: [&str; 3] = ["x", "y", "z"];
const B: [&str; 2] = ["p", "q"];

fn csv(rows: &[Row]) -> String {
    let mut s = String::from("a,b,n,f\n");
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{}\n",
            r.a.map_or("", |i| A[i]),
            r.b.map_or("", |i| B[i]),
            r.n.map_or(String::new(), |v| v.to_string()),
            r.f.map_or(String::new(), |v| format!("{:.1}", f64::from(v) / 10.0))
        ));
    }
    s
}

fn label(row: &Row, field: &str) -> String {
    match field {
        "a" => row.a.map_or("(blank)", |i| A[i]).to_string(),
        _ => row.b.map_or("(blank)", |i| B[i]).to_string(),
    }
}

fn value(row: &Row, field: &str) -> Option<f64> {
    match field {
        "n" => row.n.map(f64::from),
        _ => row.f.map(|v| f64::from(v) / 10.0),
    }
}

/// The textbook answer for one cell. `None` means "no rows at all": an empty cell.
fn naive(rows: &[&Row], agg: &str, field: &str) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    let vals: Vec<f64> = rows.iter().filter_map(|r| value(r, field)).collect();
    match agg {
        "count" => return Some(vals.len() as f64),
        "distinct" => {
            let mut v = vals.clone();
            v.sort_by(f64::total_cmp);
            v.dedup();
            return Some(v.len() as f64);
        }
        _ => {}
    }
    if vals.is_empty() {
        return None;
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = || {
        (vals.len() > 1).then(|| vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0))
    };
    match agg {
        "sum" => Some(vals.iter().sum()),
        "mean" => Some(mean),
        "min" => Some(vals.iter().copied().fold(f64::INFINITY, f64::min)),
        "max" => Some(vals.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        "var" => var(),
        "std" => var().map(f64::sqrt),
        "median" => {
            let mut v = vals.clone();
            v.sort_by(f64::total_cmp);
            Some(if v.len() % 2 == 1 {
                v[v.len() / 2]
            } else {
                (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
            })
        }
        "mode" => {
            let mut v = vals.clone();
            v.sort_by(f64::total_cmp);
            let mut best = (v[0], 0);
            for &x in &v {
                let c = v.iter().filter(|&&y| y == x).count();
                if c > best.1 {
                    best = (x, c);
                }
            }
            Some(best.0)
        }
        "first" => Some(vals[0]),
        "last" => Some(*vals.last().unwrap()),
        _ => unreachable!(),
    }
}

fn arb_rows() -> impl Strategy<Value = Vec<Row>> {
    prop::collection::vec(
        (
            prop::option::weighted(0.85, 0..3usize),
            prop::option::weighted(0.85, 0..2usize),
            prop::option::weighted(0.8, -9..10i32),
            prop::option::weighted(0.8, -50..50i32),
        )
            .prop_map(|(a, b, n, f)| Row { a, b, n, f }),
        0..60,
    )
}

fn arb_dims() -> impl Strategy<Value = Vec<&'static str>> {
    prop::sample::select(vec![
        vec![],
        vec!["a"],
        vec!["b"],
        vec!["a", "b"],
        vec!["b", "a"],
    ])
}

fn arb_agg() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec![
        "sum", "count", "distinct", "mean", "min", "max", "median", "std", "var", "mode", "first",
        "last",
    ])
}

fn cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

proptest! {
    // 300 cases by default; PROPTEST_CASES=20000 for a long soak
    #![proptest_config(ProptestConfig::with_cases(cases()))]

    #[test]
    fn every_cell_matches_the_naive_reference(
        rows in arb_rows(),
        row_dims in arb_dims(),
        col_dims in arb_dims(),
        agg in arb_agg(),
        numeric in prop::sample::select(vec!["n", "f"]),
        subtotals in any::<bool>(),
        top in any::<bool>(),
    ) {
        let table = Table::from_csv(csv(&rows).as_bytes(), LoadOptions::default()).unwrap();
        // a table whose columns are all blank would be typed as text and refuse numeric aggregations
        prop_assume!(table.columns[2].is_numeric() && table.columns[3].is_numeric());
        let dim = |f: &&str| json!({"field": f});
        let spec: PivotSpec = serde_json::from_value(json!({
            "rows": row_dims.iter().map(dim).collect::<Vec<_>>(),
            "cols": col_dims.iter().map(dim).collect::<Vec<_>>(),
            "values": [{"field": numeric, "agg": agg}],
            "options": {"subtotals": subtotals, "totals_row": top, "totals_col": !top},
        })).unwrap();
        let result = pivot(&table, &spec).unwrap();

        // every row of the table is somewhere: the leaf cells of a count-of-rows pivot add up
        let matches = |h: &pivot_core::spec::Header, dims: &[&str], row: &Row| {
            (0..h.depth).all(|i| h.labels[i].as_deref() == Some(label(row, dims[i]).as_str()))
        };
        for (ri, rh) in result.rows.iter().enumerate() {
            for (ci, ch) in result.cols.iter().enumerate() {
                let in_cell: Vec<&Row> = rows.iter().filter(|r| matches(rh, &row_dims, r) && matches(ch, &col_dims, r)).collect();
                let want = naive(&in_cell, agg, numeric);
                let got = result.cells[ri][ci][0];
                match (want, got) {
                    (Some(w), Some(g)) => prop_assert!((w - g).abs() <= 1e-9 * w.abs().max(1.0), "{agg} of {numeric} at ({ri},{ci}): want {w}, got {g}"),
                    (None, None) => {}
                    (w, g) => prop_assert!(false, "{agg} of {numeric} at ({ri},{ci}) [{:?}/{:?}]: want {w:?}, got {g:?}", rh.kind, ch.kind),
                }
            }
        }
        prop_assert_eq!(result.rows_used, rows.len());
        // header shapes: depth never exceeds the field count, totals have depth 0
        for h in result.rows.iter().chain(&result.cols) {
            prop_assert!(h.depth <= 2);
            if h.kind == HeaderKind::Total { prop_assert_eq!(h.depth, 0); }
        }
    }

    #[test]
    fn sums_and_counts_are_additive_across_the_layout(rows in arb_rows(), subtotals in any::<bool>()) {
        let table = Table::from_csv(csv(&rows).as_bytes(), LoadOptions::default()).unwrap();
        prop_assume!(table.columns[2].is_numeric());
        let spec: PivotSpec = serde_json::from_value(json!({
            "rows": [{"field": "a"}, {"field": "b"}], "cols": [{"field": "b"}],
            "values": [{"field": "n", "agg": "sum"}, {"agg": "count"}],
            "options": {"subtotals": subtotals},
        })).unwrap();
        let r = pivot(&table, &spec).unwrap();
        prop_assume!(!r.rows.is_empty());
        let grand = r.cells.last().unwrap().last().unwrap();
        prop_assert_eq!(grand[1], Some(rows.len() as f64));
        let sum_all: f64 = rows.iter().filter_map(|r| r.n).map(f64::from).sum();
        if rows.iter().any(|r| r.n.is_some()) {
            prop_assert_eq!(grand[0], Some(sum_all));
        }
        // leaf rows add up to the total row, for every column
        for (ci, _) in r.cols.iter().enumerate() {
            let leaf_sum: f64 = r.rows.iter().enumerate().filter(|(_, h)| h.kind == HeaderKind::Leaf).filter_map(|(ri, _)| r.cells[ri][ci][1]).sum();
            prop_assert_eq!(Some(leaf_sum), r.cells.last().unwrap()[ci][1]);
        }
    }

    #[test]
    fn shuffling_the_input_rows_never_changes_a_sum_or_a_count(rows in arb_rows(), seed in 0u64..1000) {
        let table = Table::from_csv(csv(&rows).as_bytes(), LoadOptions::default()).unwrap();
        prop_assume!(table.columns[2].is_numeric());
        let mut shuffled = rows.clone();
        // a deterministic permutation: rotate by a seed-dependent amount, then reverse
        if !shuffled.is_empty() { let k = (seed as usize) % shuffled.len(); shuffled.rotate_left(k); }
        shuffled.reverse();
        let t2 = Table::from_csv(csv(&shuffled).as_bytes(), LoadOptions::default()).unwrap();
        let spec: PivotSpec = serde_json::from_value(json!({
            "rows": [{"field": "a"}], "cols": [{"field": "b"}],
            "values": [{"field": "n", "agg": "sum"}, {"agg": "count"}, {"field": "n", "agg": "median"}],
        })).unwrap();
        let (r1, r2) = (pivot(&table, &spec).unwrap(), pivot(&t2, &spec).unwrap());
        prop_assert_eq!(r1.cells, r2.cells);
    }
}
