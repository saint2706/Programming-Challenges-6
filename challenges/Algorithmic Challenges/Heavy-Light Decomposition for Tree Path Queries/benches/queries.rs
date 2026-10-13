//! Per-operation cost of the three real structures at n = 1,000,000 on four tree shapes.
//!
//! Each iteration runs a batch of 1,024 operations; criterion's throughput line divides by that.
//! `path_add` pairs every +1 with a -1 so values never drift.

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use hld::shapes::{Rng, Shape, generate};
use hld::{BalancedHld, ClassicHld, Layout, MaybeSubtree, PathOps, TopTree, Tree};

const N: usize = 1_000_000;
const BATCH: usize = 1024;

fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(BATCH as u64));
}

fn pairs(n: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = Rng::new(seed);
    (0..BATCH).map(|_| (rng.below(n), rng.below(n))).collect()
}

fn bench_structure<S: MaybeSubtree>(
    c: &mut Criterion,
    name: &str,
    shape: Shape,
    layout: &Layout,
    values: &[i64],
    workloads: &[(&str, Vec<(usize, usize)>)],
) {
    let mut s = S::from_weights(layout.clone(), values);
    for (label, pairs) in workloads {
        let case = format!("{}/{label}", shape.name());
        let mut group = c.benchmark_group(format!("path_query/{case}"));
        configure(&mut group);
        group.bench_function(name, |b| {
            b.iter(|| {
                for &(u, v) in pairs {
                    black_box(s.path_query(u, v));
                }
            })
        });
        group.finish();

        let mut group = c.benchmark_group(format!("path_add/{case}"));
        configure(&mut group);
        group.bench_function(name, |b| {
            b.iter(|| {
                for &(u, v) in pairs {
                    s.path_add(u, v, 1);
                    s.path_add(u, v, -1);
                }
            })
        });
        group.finish();
    }
    let (_, random) = &workloads[0];
    let mut group = c.benchmark_group(format!("point_set/{}", shape.name()));
    configure(&mut group);
    group.bench_function(name, |b| {
        b.iter(|| {
            for &(u, _) in random {
                s.set(u, 7);
            }
        })
    });
    group.finish();

    if s.maybe_subtree(0).is_some() {
        let mut group = c.benchmark_group(format!("subtree_query/{}", shape.name()));
        configure(&mut group);
        group.bench_function(name, |b| {
            b.iter(|| {
                for &(u, _) in random {
                    black_box(s.maybe_subtree(u));
                }
            })
        });
        group.finish();
    }
}

fn shapes(c: &mut Criterion) {
    for shape in [
        Shape::Random,
        Shape::Binary,
        Shape::Adversarial,
        Shape::Path,
    ] {
        let g = generate(shape, N, 42);
        let layout = Layout::new(&Tree::new(g.n, &g.edges).unwrap(), g.root).unwrap();
        let mut rng = Rng::new(9);
        let values: Vec<i64> = (0..N).map(|_| rng.below(2001) as i64 - 1000).collect();
        let mut workloads = vec![("random", pairs(N, 1))];
        if shape == Shape::Adversarial {
            // The deepest light-nested vertex against the root, over and over.
            workloads.push(("far", vec![g.far; BATCH]));
        }
        bench_structure::<ClassicHld>(c, "classic", shape, &layout, &values, &workloads);
        bench_structure::<BalancedHld>(c, "balanced", shape, &layout, &values, &workloads);
        bench_structure::<TopTree>(c, "toptree", shape, &layout, &values, &workloads);
    }
}

fn build(c: &mut Criterion) {
    let g = generate(Shape::Random, N, 42);
    let layout = Layout::new(&Tree::new(g.n, &g.edges).unwrap(), g.root).unwrap();
    let values = vec![1i64; N];
    let mut group = c.benchmark_group("build/random-1M");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("classic", |b| {
        b.iter(|| black_box(ClassicHld::from_weights(layout.clone(), &values)))
    });
    group.bench_function("balanced", |b| {
        b.iter(|| black_box(BalancedHld::from_weights(layout.clone(), &values)))
    });
    group.bench_function("toptree", |b| {
        b.iter(|| black_box(TopTree::from_weights(layout.clone(), &values)))
    });
    group.bench_function("layout only", |b| {
        let tree = Tree::new(g.n, &g.edges).unwrap();
        b.iter(|| black_box(Layout::new(&tree, g.root).unwrap()))
    });
    group.finish();
}

fn all(c: &mut Criterion) {
    shapes(c);
    build(c);
}

criterion_group!(benches, all);
criterion_main!(benches);
