use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use merkle_tree::multi::verify_multi;
use merkle_tree::verify::verify_inclusion;
use merkle_tree::{Blake3Hasher, Build, MerkleTree, Sha256Hasher};
use std::hint::black_box;

/// 64-byte pseudo-random-ish entries (typical small log entry / transaction size).
fn leaves(n: usize) -> Vec<[u8; 64]> {
    (0..n)
        .map(|i| {
            let mut b = [0u8; 64];
            b[..8].copy_from_slice(&(i as u64).to_le_bytes());
            b[8..16]
                .copy_from_slice(&((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).to_le_bytes());
            b
        })
        .collect()
}

fn build(c: &mut Criterion) {
    let mut g = c.benchmark_group("build");
    g.sample_size(20);
    for n in [1_000usize, 10_000, 100_000, 1_000_000] {
        let d = leaves(n);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("sha256_seq", n), &d, |b, d| {
            b.iter(|| MerkleTree::<Sha256Hasher>::build(black_box(d), Build::Sequential).root())
        });
        g.bench_with_input(BenchmarkId::new("sha256_par", n), &d, |b, d| {
            b.iter(|| MerkleTree::<Sha256Hasher>::build(black_box(d), Build::Parallel).root())
        });
        g.bench_with_input(BenchmarkId::new("blake3_seq", n), &d, |b, d| {
            b.iter(|| MerkleTree::<Blake3Hasher>::build(black_box(d), Build::Sequential).root())
        });
        g.bench_with_input(BenchmarkId::new("blake3_par", n), &d, |b, d| {
            b.iter(|| MerkleTree::<Blake3Hasher>::build(black_box(d), Build::Parallel).root())
        });
    }
    g.finish();
}

fn build_small(c: &mut Criterion) {
    let mut g = c.benchmark_group("build_small");
    for n in [16usize, 256, 4_096] {
        let d = leaves(n);
        g.bench_with_input(BenchmarkId::new("sha256_seq", n), &d, |b, d| {
            b.iter(|| MerkleTree::<Sha256Hasher>::build(black_box(d), Build::Sequential).root())
        });
        g.bench_with_input(BenchmarkId::new("sha256_par", n), &d, |b, d| {
            b.iter(|| MerkleTree::<Sha256Hasher>::build(black_box(d), Build::Parallel).root())
        });
    }
    g.finish();
}

fn proofs(c: &mut Criterion) {
    let mut g = c.benchmark_group("inclusion");
    for n in [1_000usize, 100_000, 1_000_000] {
        let d = leaves(n);
        let t = MerkleTree::<Sha256Hasher>::build(&d, Build::Parallel);
        let root = t.root();
        let idx = n / 3;
        let leaf = t.leaf_hash(idx).unwrap();
        let proof = t.inclusion_proof(idx, n).unwrap();
        g.bench_with_input(BenchmarkId::new("generate", n), &n, |b, &n| {
            b.iter(|| t.inclusion_proof(black_box(idx), n).unwrap())
        });
        g.bench_with_input(BenchmarkId::new("verify", n), &n, |b, &n| {
            b.iter(|| {
                verify_inclusion::<Sha256Hasher>(&leaf, idx, n, black_box(&proof), &root).unwrap()
            })
        });
        // historical size (not a stored node -> O(log^2 n) path through range_hash)
        let hist = n * 3 / 5 + 1;
        g.bench_with_input(
            BenchmarkId::new("generate_historical", n),
            &hist,
            |b, &h| b.iter(|| t.inclusion_proof(black_box(idx.min(h - 1)), h).unwrap()),
        );
    }
    g.finish();
}

fn batch(c: &mut Criterion) {
    let n = 1_000_000usize;
    let d = leaves(n);
    let t = MerkleTree::<Sha256Hasher>::build(&d, Build::Parallel);
    let root = t.root();
    let mut g = c.benchmark_group("batch_vs_individual");
    for (label, k, stride) in [
        ("k=100 spread", 100usize, 9_000usize),
        ("k=1000 contiguous", 1000, 1),
    ] {
        let idx: Vec<usize> = (0..k).map(|i| 50_000 + i * stride).collect();
        let leaf_hashes: Vec<_> = idx.iter().map(|&i| (i, t.leaf_hash(i).unwrap())).collect();
        let multi = t.multi_proof(&idx).unwrap();
        let singles: Vec<_> = idx
            .iter()
            .map(|&i| t.inclusion_proof(i, n).unwrap())
            .collect();
        eprintln!(
            "[{label}] individual proofs: {} hashes, multi-proof: {} hashes",
            singles.iter().map(Vec::len).sum::<usize>(),
            multi.len()
        );
        g.bench_function(BenchmarkId::new("individual_verify", label), |b| {
            b.iter(|| {
                for ((i, lh), p) in leaf_hashes.iter().zip(&singles) {
                    verify_inclusion::<Sha256Hasher>(lh, *i, n, p, &root).unwrap();
                }
            })
        });
        g.bench_function(BenchmarkId::new("multi_verify", label), |b| {
            b.iter(|| {
                verify_multi::<Sha256Hasher>(black_box(&leaf_hashes), n, &multi, &root).unwrap()
            })
        });
    }
    g.finish();
}

criterion_group!(benches, build, build_small, proofs, batch);
criterion_main!(benches);
