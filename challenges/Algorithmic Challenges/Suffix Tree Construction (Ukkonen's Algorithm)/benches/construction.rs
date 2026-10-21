//! Construction, search and applications at 1,000,000 bytes, with criterion's statistics.
//!
//! Real texts come from `data/` (see `scripts/fetch_corpus.py`); a missing one is skipped. The
//! `examples/scale.rs` tables cover more sizes and memory.

use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use libsais::SuffixArrayConstruction;
use sufftree::apps::longest_common_substring;
use sufftree::array_index::{SuffixArrayIndex, longest_common_substring_array};
use sufftree::pwotd;
use sufftree::sais::suffix_array_u8;
use sufftree::satree::{sa_lcp_bytes, tree_of_words};
use sufftree::texts::uniform;
use sufftree::ukkonen::{HashChildren, ListChildren, Ukkonen};
use sufftree::{Bytes, SuffixTree};

const N: usize = 1_000_000;

fn texts() -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = ["dna", "english"]
        .into_iter()
        .filter_map(|name| {
            let all = std::fs::read(Path::new("data").join(name)).ok()?;
            Some((name.to_string(), all[..N.min(all.len())].to_vec()))
        })
        .collect();
    out.push(("uniform4".into(), uniform(N, 4, 1)));
    out
}

fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));
}

fn construct(c: &mut Criterion) {
    for (name, bytes) in texts() {
        let mut group = c.benchmark_group(format!("construct/{name}"));
        configure(&mut group);
        group.bench_function("SA-IS (own)", |b| {
            b.iter(|| suffix_array_u8(black_box(&bytes)))
        });
        group.bench_function("libsais", |b| {
            b.iter(|| {
                SuffixArrayConstruction::for_text(black_box(bytes.as_slice()))
                    .in_owned_buffer::<i32>()
                    .single_threaded()
                    .run()
                    .expect("libsais")
            })
        });
        group.bench_function("Ukkonen, sibling lists", |b| {
            b.iter(|| {
                let text = Bytes::new(bytes.as_slice());
                Ukkonen::<ListChildren>::build(&text).freeze(text)
            })
        });
        group.bench_function("Ukkonen, hash map", |b| {
            b.iter(|| {
                let text = Bytes::new(bytes.as_slice());
                Ukkonen::<HashChildren>::build(&text).freeze(text)
            })
        });
        group.bench_function("SA-IS + LCP + tree", |b| {
            b.iter(|| {
                let (sa, lcp) = sa_lcp_bytes(&bytes);
                SuffixTree::from_sa_lcp(Bytes::new(bytes.as_slice()), &sa, &lcp)
            })
        });
        group.bench_function("PWOTD", |b| {
            b.iter(|| pwotd::build(Bytes::new(bytes.as_slice())))
        });
        group.bench_function("WOTD (no partition)", |b| {
            b.iter(|| pwotd::build_with(Bytes::new(bytes.as_slice()), Some(0)))
        });
        group.finish();
    }
}

fn search(c: &mut Criterion) {
    for (name, bytes) in texts() {
        let (sa, lcp) = sa_lcp_bytes(&bytes);
        let tree = SuffixTree::from_sa_lcp(Bytes::new(bytes.as_slice()), &sa, &lcp);
        let index = SuffixArrayIndex::new(bytes.clone());
        let step = (bytes.len() - 16) / 1024;
        let pats: Vec<&[u8]> = (0..1024).map(|i| &bytes[i * step..i * step + 16]).collect();
        let mut group = c.benchmark_group(format!("search/{name}"));
        configure(&mut group);
        group.bench_function("suffix tree count", |b| {
            b.iter(|| pats.iter().map(|p| tree.count(black_box(p))).sum::<usize>())
        });
        group.bench_function("suffix array count", |b| {
            b.iter(|| {
                pats.iter()
                    .map(|p| index.count(black_box(p)))
                    .sum::<usize>()
            })
        });
        group.finish();
    }
}

fn applications(c: &mut Criterion) {
    let (_, bytes) = texts().into_iter().next().expect("a text");
    let mut group = c.benchmark_group("applications");
    configure(&mut group);
    let (sa, lcp) = sa_lcp_bytes(&bytes);
    let tree = SuffixTree::from_sa_lcp(Bytes::new(bytes.as_slice()), &sa, &lcp);
    let index = SuffixArrayIndex::new(bytes.clone());
    group.bench_function("longest repeat, tree scan", |b| {
        b.iter(|| black_box(tree.longest_repeated_substring()))
    });
    group.bench_function("longest repeat, LCP max", |b| {
        b.iter(|| black_box(index.longest_repeated_substring()))
    });
    let strings: Vec<Vec<u8>> = (0..3)
        .map(|i| bytes[i * 100_000..(i + 1) * 100_000].to_vec())
        .collect();
    group.bench_function("longest common substring of 3, tree", |b| {
        b.iter(|| black_box(longest_common_substring(&tree_of_words(&strings))))
    });
    group.bench_function("longest common substring of 3, array window", |b| {
        b.iter(|| black_box(longest_common_substring_array(&strings)))
    });
    group.finish();
}

criterion_group!(benches, construct, search, applications);
criterion_main!(benches);
