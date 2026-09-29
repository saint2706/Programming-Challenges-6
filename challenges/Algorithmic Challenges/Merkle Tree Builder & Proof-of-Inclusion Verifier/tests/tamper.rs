//! Every way to corrupt a proof, root, index or size must be rejected.

use merkle_tree::multi::verify_multi;
use merkle_tree::verify::{verify_consistency, verify_inclusion, verify_inclusion_data};
use merkle_tree::{Build, MerkleTree, ProofError, Sha256Hasher};

type H = Sha256Hasher;

fn tree(n: usize) -> (Vec<Vec<u8>>, MerkleTree<H>) {
    let d: Vec<Vec<u8>> = (0..n).map(|i| format!("entry {i}").into_bytes()).collect();
    let t = MerkleTree::<H>::build(&d, Build::Sequential);
    (d, t)
}

#[test]
fn inclusion_rejects_every_single_bit_flip_in_proof_leaf_and_root() {
    for n in [1usize, 2, 3, 7, 8, 13, 33] {
        let (_, t) = tree(n);
        let root = t.root();
        for i in 0..n {
            let p = t.inclusion_proof(i, n).unwrap();
            let leaf = t.leaf_hash(i).unwrap();
            assert!(verify_inclusion::<H>(&leaf, i, n, &p, &root).is_ok());
            for k in 0..p.len() {
                for bit in [0usize, 100, 255] {
                    let mut bad = p.clone();
                    bad[k][bit / 8] ^= 1 << (bit % 8);
                    assert!(
                        verify_inclusion::<H>(&leaf, i, n, &bad, &root).is_err(),
                        "n={n} i={i} k={k}"
                    );
                }
            }
            let mut bad_leaf = leaf;
            bad_leaf[0] ^= 1;
            assert!(verify_inclusion::<H>(&bad_leaf, i, n, &p, &root).is_err());
            let mut bad_root = root;
            bad_root[31] ^= 0x80;
            assert!(verify_inclusion::<H>(&leaf, i, n, &p, &bad_root).is_err());
        }
    }
}

#[test]
fn inclusion_rejects_wrong_index_size_and_proof_length() {
    for n in [2usize, 5, 8, 21] {
        let (_, t) = tree(n);
        let (_, bigger) = tree(n + 3);
        let root = t.root();
        for i in 0..n {
            let p = t.inclusion_proof(i, n).unwrap();
            let leaf = t.leaf_hash(i).unwrap();
            for wrong in (0..n).filter(|&w| w != i) {
                assert!(
                    verify_inclusion::<H>(&leaf, wrong, n, &p, &root).is_err(),
                    "n={n} {i}->{wrong}"
                );
            }
            // Claiming a different size with that size's *own* honest root must fail. (A
            // proof does not bind the size by itself -- a size-5 path for leaf 0 hashes
            // identically under a claimed size of 6 -- which is why real logs sign
            // (root, size) together in the tree head.)
            for wrong_size in (1..=n + 3).filter(|&s| s != n) {
                let wrong_root = bigger.root_at(wrong_size).unwrap();
                assert!(
                    verify_inclusion::<H>(&leaf, i, wrong_size, &p, &wrong_root).is_err(),
                    "size {n}->{wrong_size}"
                );
            }
            let mut long = p.clone();
            long.push([0u8; 32]);
            assert!(verify_inclusion::<H>(&leaf, i, n, &long, &root).is_err());
            if !p.is_empty() {
                assert!(verify_inclusion::<H>(&leaf, i, n, &p[..p.len() - 1], &root).is_err());
            }
        }
    }
}

#[test]
fn inclusion_out_of_range_and_data_variant() {
    let (d, t) = tree(5);
    assert_eq!(
        t.inclusion_proof(5, 5),
        Err(ProofError::IndexOutOfRange { index: 5, size: 5 })
    );
    assert_eq!(
        t.inclusion_proof(0, 6),
        Err(ProofError::SizeOutOfRange { size: 6, len: 5 })
    );
    let p = t.inclusion_proof(3, 5).unwrap();
    assert!(verify_inclusion_data::<H>(&d[3], 3, 5, &p, &t.root()).is_ok());
    assert!(verify_inclusion_data::<H>(b"entry 4", 3, 5, &p, &t.root()).is_err());
    assert!(verify_inclusion::<H>(&t.leaf_hash(0).unwrap(), 5, 5, &[], &t.root()).is_err());
}

#[test]
fn consistency_rejects_tampering() {
    let (_, t) = tree(40);
    for m in 1..37 {
        for n in [m + 1, 37] {
            let (r1, r2) = (t.root_at(m).unwrap(), t.root_at(n).unwrap());
            let p = t.consistency_proof(m, n).unwrap();
            assert!(verify_consistency::<H>(m, n, &r1, &r2, &p).is_ok());
            for k in 0..p.len() {
                let mut bad = p.clone();
                bad[k][7] ^= 0x10;
                assert!(
                    verify_consistency::<H>(m, n, &r1, &r2, &bad).is_err(),
                    "{m}->{n} k={k}"
                );
            }
            let (mut b1, mut b2) = (r1, r2);
            b1[0] ^= 1;
            b2[0] ^= 1;
            assert!(verify_consistency::<H>(m, n, &b1, &r2, &p).is_err());
            assert!(verify_consistency::<H>(m, n, &r1, &b2, &p).is_err());
            let r_next = t.root_at(n + 1).unwrap();
            assert!(verify_consistency::<H>(m, n + 1, &r1, &r_next, &p).is_err());
            let mut extra = p.clone();
            extra.push([0u8; 32]);
            assert!(verify_consistency::<H>(m, n, &r1, &r2, &extra).is_err());
        }
    }
}

#[test]
fn consistency_forked_history_is_rejected() {
    // Two logs share their first entries, but one rewrote entry 3: the honest 10->20 proof
    // must not link the rewritten history's 10-leaf root to the honest 20-leaf root.
    let (_, honest) = tree(20);
    let mut rewritten: Vec<Vec<u8>> = (0..20).map(|i| format!("entry {i}").into_bytes()).collect();
    rewritten[3] = b"rewritten".to_vec();
    let forged = MerkleTree::<H>::build(&rewritten, Build::Sequential);
    let p = honest.consistency_proof(10, 20).unwrap();
    assert!(
        verify_consistency::<H>(10, 20, &forged.root_at(10).unwrap(), &honest.root(), &p).is_err()
    );
}

#[test]
fn consistency_degenerate_arguments() {
    let (_, t) = tree(8);
    let r = t.root();
    assert!(t.consistency_proof(0, 8).is_err());
    assert!(t.consistency_proof(9, 8).is_err());
    assert!(t.consistency_proof(3, 9).is_err());
    assert!(t.consistency_proof(8, 8).unwrap().is_empty());
    assert!(verify_consistency::<H>(8, 8, &r, &r, &[]).is_ok());
    assert!(verify_consistency::<H>(8, 8, &r, &t.root_at(7).unwrap(), &[]).is_err());
    assert!(verify_consistency::<H>(4, 8, &t.root_at(4).unwrap(), &r, &[]).is_err());
}

#[test]
fn multiproof_rejects_tampering_and_bad_input() {
    let (_, t) = tree(50);
    let root = t.root();
    let idx = [2usize, 3, 17, 40, 49];
    let leaves: Vec<_> = idx.iter().map(|&i| (i, t.leaf_hash(i).unwrap())).collect();
    let p = t.multi_proof(&idx).unwrap();
    assert!(verify_multi::<H>(&leaves, 50, &p, &root).is_ok());
    for k in 0..p.len() {
        let mut bad = p.clone();
        bad[k][0] ^= 1;
        assert!(verify_multi::<H>(&leaves, 50, &bad, &root).is_err());
    }
    let mut bad_leaves = leaves.clone();
    bad_leaves[2].1[5] ^= 1;
    assert!(verify_multi::<H>(&bad_leaves, 50, &p, &root).is_err());
    assert!(verify_multi::<H>(&leaves, 51, &p, &root).is_err());
    assert!(verify_multi::<H>(&leaves[..4], 50, &p, &root).is_err());
    assert!(verify_multi::<H>(&leaves, 50, &p[1..], &root).is_err());
    let mut extra = p.clone();
    extra.push([1; 32]);
    assert!(verify_multi::<H>(&leaves, 50, &extra, &root).is_err());
    assert_eq!(t.multi_proof(&[]), Err(ProofError::InvalidIndices));
    assert_eq!(t.multi_proof(&[3, 3]), Err(ProofError::InvalidIndices));
    assert_eq!(t.multi_proof(&[4, 3]), Err(ProofError::InvalidIndices));
    assert_eq!(
        t.multi_proof(&[50]),
        Err(ProofError::IndexOutOfRange {
            index: 50,
            size: 50
        })
    );
}
