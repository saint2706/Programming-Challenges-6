//! Node-size (minimum degree `T`) tuning and bulk-load vs incremental build,
//! at n = 1e6 uniformly random keys.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use ost::OrderStatBTree;
use ost::rng::SplitMix64;

const N: usize = 1_000_000;
const BATCH: usize = 1024;
const UNIVERSE: usize = 1 << 22;

fn tune<const T: usize>(c: &mut Criterion, ks: &[usize], probes: &[usize]) {
    let mut g = c.benchmark_group("btree_T");
    g.sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(format!("insert/T={T}"), |b| {
        b.iter_batched(
            OrderStatBTree::<usize, T>::new,
            |mut t| {
                for &k in ks {
                    t.insert(k);
                }
                black_box(t)
            },
            BatchSize::PerIteration,
        )
    });
    let mut t = OrderStatBTree::<usize, T>::new();
    for &k in ks {
        t.insert(k);
    }
    g.throughput(Throughput::Elements(BATCH as u64));
    g.bench_function(format!("rank/T={T}"), |b| {
        b.iter(|| probes.iter().map(|p| t.rank(p)).sum::<usize>())
    });
    g.bench_function(format!("select/T={T}"), |b| {
        b.iter(|| probes.iter().filter_map(|p| t.select(p % N)).sum::<usize>())
    });
    let mut rng = SplitMix64::new(7);
    g.bench_function(format!("mixed/T={T}"), |b| {
        b.iter(|| {
            let mut acc = 0;
            for i in 0..BATCH {
                match i % 4 {
                    0 => t.insert(rng.below(UNIVERSE)),
                    1 => {
                        let v = *t.select(rng.below(t.len())).unwrap();
                        t.remove(&v);
                    }
                    2 => acc += t.rank(&rng.below(UNIVERSE)),
                    _ => acc += *t.select(rng.below(t.len())).unwrap(),
                }
            }
            black_box(acc)
        })
    });
    g.finish();
}

fn bulk(c: &mut Criterion, sorted: &[usize]) {
    let mut g = c.benchmark_group("build_sorted");
    g.sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1))
        .throughput(Throughput::Elements(N as u64));
    g.bench_function("incremental_insert", |b| {
        b.iter_batched(
            OrderStatBTree::<usize, 16>::new,
            |mut t| {
                for &k in sorted {
                    t.insert(k);
                }
                black_box(t)
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("from_sorted", |b| {
        b.iter_batched(
            || sorted.to_vec(),
            |v| black_box(OrderStatBTree::<usize, 16>::from_sorted(v)),
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

fn all(c: &mut Criterion) {
    let mut rng = SplitMix64::new(1);
    let ks: Vec<usize> = (0..N).map(|_| rng.below(UNIVERSE)).collect();
    let probes: Vec<usize> = (0..BATCH).map(|_| rng.below(UNIVERSE)).collect();
    tune::<2>(c, &ks, &probes);
    tune::<4>(c, &ks, &probes);
    tune::<8>(c, &ks, &probes);
    tune::<16>(c, &ks, &probes);
    tune::<32>(c, &ks, &probes);
    tune::<64>(c, &ks, &probes);
    tune::<128>(c, &ks, &probes);
    let mut sorted = ks;
    sorted.sort_unstable();
    bulk(c, &sorted);
}

criterion_group!(benches, all);
criterion_main!(benches);
