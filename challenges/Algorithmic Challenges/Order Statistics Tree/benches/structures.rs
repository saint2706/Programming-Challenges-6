//! Head-to-head: every order-statistic structure on insert / rank / select /
//! mixed workloads at n = 1e3 .. 1e6. Keys are uniform in `0..2^22` so the
//! Fenwick tree (fixed universe) can play too.
//!
//! Each rank/select/mixed iteration performs `BATCH` operations; criterion's
//! `Throughput::Elements` turns that into ns-per-op numbers.

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
};
use ost::rng::SplitMix64;
use ost::{
    AvlTree, Fenwick, NaiveBTreeMap, OrderStatBTree, OrderStatistics, RedBlackTree, SortedVec,
    Treap,
};

const UNIVERSE: usize = 1 << 22;
const BATCH: usize = 1024;
const SIZES: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];

fn keys(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.below(UNIVERSE)).collect()
}

fn build<S: OrderStatistics<usize>>(mut s: S, ks: &[usize]) -> S {
    for &k in ks {
        s.insert(k);
    }
    s
}

/// `max_n`: skip the structure above this size (its cost is quadratic/linear).
fn bench_structure<S, F>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    n: usize,
    max_n: usize,
    make: F,
) where
    S: OrderStatistics<usize>,
    F: Fn() -> S,
{
    if n > max_n {
        return;
    }
    let ks = keys(n, 1);
    let probes = keys(BATCH, 2);

    g.throughput(Throughput::Elements(n as u64));
    g.bench_function(format!("insert/{name}/{n}"), |b| {
        b.iter_batched(&make, |s| black_box(build(s, &ks)), BatchSize::PerIteration)
    });

    let mut s = build(make(), &ks);
    g.throughput(Throughput::Elements(BATCH as u64));
    g.bench_function(format!("rank/{name}/{n}"), |b| {
        b.iter(|| probes.iter().map(|p| s.rank(p)).sum::<usize>())
    });
    let idxs: Vec<usize> = probes.iter().map(|p| p % n).collect();
    g.bench_function(format!("select/{name}/{n}"), |b| {
        b.iter(|| idxs.iter().filter_map(|&i| s.select(i)).sum::<usize>())
    });

    // Steady state: insert a key, delete a random existing element, rank, select.
    let mut rng = SplitMix64::new(3);
    g.bench_function(format!("mixed/{name}/{n}"), |b| {
        b.iter(|| {
            let mut acc = 0;
            for i in 0..BATCH {
                match i % 4 {
                    0 => s.insert(rng.below(UNIVERSE)),
                    1 => {
                        let victim = s.select(rng.below(s.len())).unwrap();
                        s.remove(&victim);
                    }
                    2 => acc += s.rank(&rng.below(UNIVERSE)),
                    _ => acc += s.select(rng.below(s.len())).unwrap(),
                }
            }
            black_box(acc)
        })
    });
}

fn all(c: &mut Criterion) {
    let mut g = c.benchmark_group("ost");
    g.sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for n in SIZES {
        bench_structure(
            &mut g,
            "btree16",
            n,
            usize::MAX,
            OrderStatBTree::<usize, 16>::new,
        );
        bench_structure(&mut g, "avl", n, usize::MAX, AvlTree::new);
        bench_structure(&mut g, "treap", n, usize::MAX, Treap::new);
        bench_structure(&mut g, "redblack", n, usize::MAX, RedBlackTree::new);
        bench_structure(&mut g, "fenwick", n, usize::MAX, || Fenwick::new(UNIVERSE));
        bench_structure(&mut g, "sortedvec", n, 100_000, SortedVec::new);
        bench_structure(&mut g, "naive_btreemap", n, 100_000, NaiveBTreeMap::new);
    }
    g.finish();
}

criterion_group!(benches, all);
criterion_main!(benches);
