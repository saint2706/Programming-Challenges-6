//! Property tests: the optimised tree against the RFC-transcribed oracle, over both hashers.

use merkle_tree::multi::verify_multi;
use merkle_tree::reference;
use merkle_tree::verify::{verify_consistency, verify_inclusion};
use merkle_tree::{Blake3Hasher, Build, CompactTree, Hasher, MerkleTree, Sha256Hasher};
use proptest::prelude::*;

fn data(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| format!("leaf-{i}").into_bytes()).collect()
}

fn check_all_sizes<H: Hasher>(max: usize) {
    let d = data(max);
    let full = MerkleTree::<H>::build(&d, Build::Sequential);
    for n in 0..=max {
        let root = reference::mth::<H, _>(&d[..n]);
        assert_eq!(full.root_at(n).unwrap(), root, "root size {n}");
        assert_eq!(
            MerkleTree::<H>::build(&d[..n], Build::Parallel).root(),
            root
        );
        for i in 0..n {
            let p = full.inclusion_proof(i, n).unwrap();
            assert_eq!(p, reference::path::<H, _>(i, &d[..n]), "path {i}/{n}");
            verify_inclusion::<H>(&H::leaf(&d[i]), i, n, &p, &root).unwrap();
        }
        for m in 1..=n {
            let p = full.consistency_proof(m, n).unwrap();
            assert_eq!(
                p,
                reference::consistency::<H, _>(m, &d[..n]),
                "consistency {m}->{n}"
            );
            verify_consistency::<H>(m, n, &reference::mth::<H, _>(&d[..m]), &root, &p).unwrap();
        }
    }
}

#[test]
fn exhaustive_small_sizes_sha256() {
    check_all_sizes::<Sha256Hasher>(40);
}

#[test]
fn exhaustive_small_sizes_blake3() {
    check_all_sizes::<Blake3Hasher>(24);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn large_random_trees_match_oracle(n in 1usize..1500, i_seed in any::<usize>(), m_seed in any::<usize>()) {
        let d = data(n);
        let t = MerkleTree::<Sha256Hasher>::build(&d, Build::Sequential);
        let root = reference::mth::<Sha256Hasher, _>(&d);
        prop_assert_eq!(t.root(), root);
        let i = i_seed % n;
        let p = t.inclusion_proof(i, n).unwrap();
        prop_assert_eq!(&p, &reference::path::<Sha256Hasher, _>(i, &d));
        prop_assert!(verify_inclusion::<Sha256Hasher>(&t.leaf_hash(i).unwrap(), i, n, &p, &root).is_ok());
        let m = m_seed % n + 1;
        let cp = t.consistency_proof(m, n).unwrap();
        prop_assert_eq!(&cp, &reference::consistency::<Sha256Hasher, _>(m, &d));
        let old = reference::mth::<Sha256Hasher, _>(&d[..m]);
        prop_assert!(verify_consistency::<Sha256Hasher>(m, n, &old, &root, &cp).is_ok());
    }

    #[test]
    fn parallel_build_is_identical(n in 0usize..20000) {
        let d = data(n);
        let a = MerkleTree::<Sha256Hasher>::build(&d, Build::Sequential);
        let b = MerkleTree::<Sha256Hasher>::build(&d, Build::Parallel);
        prop_assert_eq!(a.root(), b.root());
        if n > 0 {
            prop_assert_eq!(a.inclusion_proof(n / 2, n).unwrap(), b.inclusion_proof(n / 2, n).unwrap());
        }
    }

    #[test]
    fn incremental_equals_batch_at_every_size(n in 0usize..600) {
        let d = data(n);
        let mut c = CompactTree::<Sha256Hasher>::new();
        for (k, leaf) in d.iter().enumerate() {
            c.append(leaf);
            prop_assert_eq!(c.frontier().len(), (k + 1).count_ones() as usize);
            prop_assert_eq!(c.root(), reference::mth::<Sha256Hasher, _>(&d[..=k]));
        }
        prop_assert_eq!(c.size(), n);
    }

    #[test]
    fn frontier_resume_round_trips(n in 1usize..300, extra in 0usize..40) {
        let d = data(n + extra);
        let mut c = CompactTree::<Sha256Hasher>::new();
        for leaf in &d[..n] { c.append(leaf); }
        let mut resumed = CompactTree::<Sha256Hasher>::from_frontier(c.size(), c.frontier().to_vec()).unwrap();
        for leaf in &d[n..] { resumed.append(leaf); }
        prop_assert_eq!(resumed.root(), reference::mth::<Sha256Hasher, _>(&d));
    }

    #[test]
    fn multiproof_verifies_and_is_never_larger(
        n in 1usize..800,
        picks in prop::collection::btree_set(any::<usize>(), 1..40),
    ) {
        let d = data(n);
        let t = MerkleTree::<Sha256Hasher>::build(&d, Build::Sequential);
        let idx: Vec<usize> = picks
            .iter()
            .map(|p| p % n)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let leaves: Vec<_> = idx.iter().map(|&i| (i, t.leaf_hash(i).unwrap())).collect();
        let proof = t.multi_proof(&idx).unwrap();
        prop_assert!(verify_multi::<Sha256Hasher>(&leaves, n, &proof, &t.root()).is_ok());
        let separate: usize = idx.iter().map(|&i| t.inclusion_proof(i, n).unwrap().len()).sum();
        prop_assert!(proof.len() <= separate);
    }
}
