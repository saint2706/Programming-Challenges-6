//! Criterion intervals for the operations that matter, at n = 65,536. `examples/scale.rs` covers
//! more sizes and memory; this is the tight confidence interval at a size where every operation
//! finishes in well under a millisecond.

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use persegtree::oracle::kth_by_sorting;
use persegtree::{FatTree, KthIndex, Persistent, SumI64, Version, WaveletMatrix};

const N: usize = 1 << 16;

fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
}

/// A deterministic stream of numbers below `n`.
fn stream(n: usize, seed: u64) -> impl Iterator<Item = usize> {
    let mut x = seed | 1;
    std::iter::from_fn(move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        Some(((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) % n as u64) as usize)
    })
}

fn updates_and_sums(c: &mut Criterion) {
    let zeros = vec![0i64; N];
    let mut group = c.benchmark_group("point-update-and-sum");
    configure(&mut group);
    group.bench_function("path-copy update", |b| {
        let mut t = Persistent::<SumI64>::from_values(&zeros, false);
        let mut v = t.first();
        let mut pos = stream(N, 1);
        b.iter(|| {
            v = t.update(v, pos.next().expect("infinite"), 7);
            black_box(v)
        })
    });
    group.bench_function("fat-node update", |b| {
        let mut t = FatTree::new(&zeros);
        let mut pos = stream(N, 1);
        b.iter(|| black_box(t.update(pos.next().expect("infinite"), 7)))
    });
    // Queries against 20,000 prebuilt versions.
    let mut path = Persistent::<SumI64>::from_values(&zeros, false);
    let mut fat = FatTree::new(&zeros);
    let mut v = path.first();
    for p in stream(N, 2).take(20_000) {
        v = path.update(v, p, 3);
        fat.update(p, 3);
    }
    let queries: Vec<(usize, usize, usize)> = stream(20_000, 3)
        .zip(stream(N / 2, 4))
        .zip(stream(N / 2, 5))
        .take(1024)
        .map(|((ver, a), b)| (ver + 1, a, a + b + 1))
        .collect();
    group.bench_function("path-copy historical sum", |b| {
        let mut i = 0;
        b.iter(|| {
            let (ver, l, r) = queries[i % queries.len()];
            i += 1;
            black_box(path.query(Version(ver as u32), l, r))
        })
    });
    group.bench_function("fat-node historical sum", |b| {
        let mut i = 0;
        b.iter(|| {
            let (ver, l, r) = queries[i % queries.len()];
            i += 1;
            black_box(fat.query(ver as u32, l, r))
        })
    });
    group.finish();
}

fn kth(c: &mut Criterion) {
    let a: Vec<i64> = stream(N, 7).take(N).map(|x| x as i64).collect();
    let persistent = KthIndex::new(&a);
    let mut sorted = a.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let ranks: Vec<u32> = a
        .iter()
        .map(|x| sorted.binary_search(x).unwrap() as u32)
        .collect();
    let wavelet = WaveletMatrix::new(&ranks);
    let queries: Vec<(usize, usize, usize)> = stream(N - 1000, 8)
        .zip(stream(1000, 9))
        .take(1024)
        .map(|(l, k)| (l, l + 1000, k))
        .collect();

    let mut group = c.benchmark_group("kth-smallest-range-1000");
    configure(&mut group);
    group.bench_function("persistent count tree", |b| {
        let mut i = 0;
        b.iter(|| {
            let (l, r, k) = queries[i % queries.len()];
            i += 1;
            black_box(persistent.kth_smallest(l, r, k))
        })
    });
    group.bench_function("wavelet matrix", |b| {
        let mut i = 0;
        b.iter(|| {
            let (l, r, k) = queries[i % queries.len()];
            i += 1;
            black_box(wavelet.kth_smallest(l, r, k))
        })
    });
    group.bench_function("sort the range", |b| {
        let mut i = 0;
        b.iter(|| {
            let (l, r, k) = queries[i % queries.len()];
            i += 1;
            black_box(kth_by_sorting(&a, l, r, k))
        })
    });
    group.finish();
}

criterion_group!(benches, updates_and_sums, kth);
criterion_main!(benches);
