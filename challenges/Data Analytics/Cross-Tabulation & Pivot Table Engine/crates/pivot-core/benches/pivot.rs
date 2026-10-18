//! Criterion benchmarks: parsing a CSV, and three pivots over a million synthetic rows.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use pivot_core::pivot::pivot;
use pivot_core::spec::PivotSpec;
use pivot_core::table::{LoadOptions, Table};
use serde_json::json;
use std::hint::black_box;

/// A million rows from a small linear congruential generator, so the benchmark needs no data file.
fn csv(rows: usize) -> String {
    let regions = ["north", "south", "east", "west", "centre"];
    let products: Vec<String> = (0..40).map(|i| format!("product-{i:02}")).collect();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    let mut out = String::from("region,product,year,units,price\n");
    for _ in 0..rows {
        let (r, p, y, u) = (next() % 5, next() % 40, 2015 + next() % 10, next() % 200);
        out.push_str(&format!(
            "{},{},{},{},{}.{}\n",
            regions[r],
            products[p],
            y,
            u,
            1 + next() % 90,
            next() % 100
        ));
    }
    out
}

fn spec(v: serde_json::Value) -> PivotSpec {
    serde_json::from_value(v).unwrap()
}

fn benches(c: &mut Criterion) {
    let text = csv(1_000_000);
    let table = Table::from_csv(text.as_bytes(), LoadOptions::default()).unwrap();

    let mut g = c.benchmark_group("million rows");
    g.sample_size(10);
    g.bench_function("parse and type the CSV", |b| {
        b.iter_batched(
            || (),
            |_| Table::from_csv(black_box(text.as_bytes()), LoadOptions::default()).unwrap(),
            BatchSize::PerIteration,
        )
    });

    let small = spec(
        json!({"rows": [{"field": "region"}], "cols": [{"field": "year"}], "values": [{"field": "units", "agg": "sum"}]}),
    );
    g.bench_function("sum by region x year", |b| {
        b.iter(|| pivot(black_box(&table), &small).unwrap())
    });

    let wide = spec(json!({
        "rows": [{"field": "region"}, {"field": "product"}], "cols": [{"field": "year"}],
        "values": [{"field": "units", "agg": "mean"}, {"field": "price", "agg": "sum"}],
        "options": {"subtotals": true}
    }));
    g.bench_function("two row fields with subtotals, two measures", |b| {
        b.iter(|| pivot(black_box(&table), &wide).unwrap())
    });

    let median = spec(
        json!({"rows": [{"field": "product"}], "cols": [{"field": "region"}], "values": [{"field": "units", "agg": "median"}]}),
    );
    g.bench_function("median by product x region", |b| {
        b.iter(|| pivot(black_box(&table), &median).unwrap())
    });
    g.finish();
}

criterion_group!(group, benches);
criterion_main!(group);
