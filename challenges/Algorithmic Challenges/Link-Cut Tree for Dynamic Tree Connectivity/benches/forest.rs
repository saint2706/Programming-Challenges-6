//! Per-operation cost of the dynamic forests on trees of 100,000 vertices.
//!
//! * `churn`: a cyclic stream of cut-and-relink pairs mixed with connectivity queries, replayed
//!   on a pre-built forest (the stream undoes itself, so it can loop). Three mixes: update-heavy
//!   (10% queries), query-heavy (90%), and query-heavy on 64 hot vertices, which a self-adjusting
//!   tree can exploit and an oblivious one cannot.
//! * `build`: link a whole tree from isolated vertices, per edge.
//! * `path` / `subtree`: aggregate queries on a built forest, for the structures that offer them.
//! * `connectivity`: the offline link-cut-tree solver against rebuilding a union-find per query.

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
};
use lct::connectivity::{random_events, solve_offline, solve_rebuild};
use lct::shapes::{Op, Rng, Shape, build, cyclic_stream, generate, replay};
use lct::{EulerTourTree, Forest, PathLct, PathOps, SubtreeLct, SubtreeOps, UfoTree};

const N: usize = 100_000;
const SHAPES: [Shape; 4] = [Shape::Random, Shape::Path, Shape::Star, Shape::Binary];

fn configure(group: &mut BenchmarkGroup<'_, WallTime>, elements: usize) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(elements as u64));
}

fn built<F: Forest>(edges: &[(usize, usize)]) -> F {
    let mut f = F::new(N);
    for &(u, v) in edges {
        f.link(u, v);
    }
    f
}

fn bench_churn_one<F: Forest>(
    c: &mut Criterion,
    name: &str,
    case: &str,
    edges: &[(usize, usize)],
    stream: &[Op],
) {
    let mut group = c.benchmark_group(format!("churn/{case}"));
    configure(&mut group, stream.len());
    let mut f = built::<F>(edges);
    group.bench_function(name, |b| b.iter(|| black_box(replay(&mut f, stream))));
    group.finish();
}

fn churn(c: &mut Criterion) {
    for shape in SHAPES {
        let edges = generate(shape, N, 1);
        for (mix, query_percent, hot) in [("update", 10, 0), ("query", 90, 0), ("hot", 90, 64)] {
            let stream = cyclic_stream::<PathLct>(N, &edges, 5_000, query_percent, hot, 7);
            let case = format!("{}/{mix}", shape.name());
            bench_churn_one::<PathLct>(c, "lct", &case, &edges, &stream);
            bench_churn_one::<SubtreeLct>(c, "subtree-lct", &case, &edges, &stream);
            bench_churn_one::<EulerTourTree>(c, "ett", &case, &edges, &stream);
            bench_churn_one::<UfoTree>(c, "ufo", &case, &edges, &stream);
        }
    }
}

fn build_one<F: Forest>(c: &mut Criterion, name: &str, shape: Shape, edges: &[(usize, usize)]) {
    let mut group = c.benchmark_group(format!("build/{}", shape.name()));
    configure(&mut group, edges.len());
    group.bench_function(name, |b| {
        b.iter_batched(
            || F::new(N),
            |mut f| {
                for &(u, v) in edges {
                    f.link(u, v);
                }
                f
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

fn builds(c: &mut Criterion) {
    for shape in SHAPES {
        let edges = generate(shape, N, 1);
        build_one::<PathLct>(c, "lct", shape, &edges);
        build_one::<SubtreeLct>(c, "subtree-lct", shape, &edges);
        build_one::<EulerTourTree>(c, "ett", shape, &edges);
        build_one::<UfoTree>(c, "ufo", shape, &edges);
    }
}

fn pairs(seed: u64) -> Vec<(usize, usize)> {
    let mut rng = Rng::new(seed);
    (0..1024).map(|_| (rng.below(N), rng.below(N))).collect()
}

fn aggregates(c: &mut Criterion) {
    for shape in SHAPES {
        let edges = generate(shape, N, 1);
        let queries = pairs(3);

        let mut group = c.benchmark_group(format!("path/{}", shape.name()));
        configure(&mut group, queries.len());
        let mut a = build::<PathLct>(N, &edges, 5);
        group.bench_function("lct", |b| {
            b.iter(|| {
                queries.iter().for_each(|&(u, v)| {
                    black_box(a.path_query(u, v));
                })
            })
        });
        let mut s = build::<SubtreeLct>(N, &edges, 5);
        group.bench_function("subtree-lct", |b| {
            b.iter(|| {
                queries.iter().for_each(|&(u, v)| {
                    black_box(s.path_query(u, v));
                })
            })
        });
        group.finish();

        // Subtree queries go across a tree edge, one at a time, in a random orientation.
        let across: Vec<(usize, usize)> = (0..1024)
            .map(|i| {
                let (a, b) = edges[(i * 97) % edges.len()];
                if i % 2 == 0 { (a, b) } else { (b, a) }
            })
            .collect();
        let mut group = c.benchmark_group(format!("subtree/{}", shape.name()));
        configure(&mut group, across.len());
        group.bench_function("subtree-lct sum", |b| {
            b.iter(|| {
                across.iter().for_each(|&(p, v)| {
                    black_box(s.subtree_query(p, v));
                })
            })
        });
        let mut e = build::<EulerTourTree>(N, &edges, 5);
        group.bench_function("ett sum", |b| {
            b.iter(|| {
                across.iter().for_each(|&(p, v)| {
                    black_box(e.subtree_query(p, v));
                })
            })
        });
        group.bench_function("ett min/max", |b| {
            b.iter(|| {
                across.iter().for_each(|&(p, v)| {
                    black_box(e.subtree_stats(p, v));
                })
            })
        });
        group.finish();
    }
}

fn connectivity(c: &mut Criterion) {
    let (n, events) = (2_000, random_events(2_000, 4_000, 20_000, 30, 11));
    let mut group = c.benchmark_group("connectivity");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(events.len() as u64));
    group.bench_function("offline link-cut tree", |b| {
        b.iter(|| black_box(solve_offline(n, &events).unwrap()))
    });
    group.bench_function("union-find rebuilt per query", |b| {
        b.iter(|| black_box(solve_rebuild(n, &events).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, churn, builds, aggregates, connectivity);
criterion_main!(benches);
