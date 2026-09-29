//! Known-answer tests. Vectors are the RFC 6962 / Certificate Transparency reference set
//! used by transparency-dev/merkle (`testonly/constants.go`, `rfc6962/rfc6962_test.go`).

use merkle_tree::verify::{verify_consistency, verify_inclusion};
use merkle_tree::{Build, CompactTree, MerkleTree, Sha256Hasher, hasher::Hasher};

fn hd(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn leaf_inputs() -> Vec<Vec<u8>> {
    [
        "",
        "00",
        "10",
        "2021",
        "3031",
        "40414243",
        "5051525354555657",
        "606162636465666768696a6b6c6d6e6f",
    ]
    .iter()
    .map(|h| hex::decode(h).unwrap())
    .collect()
}

/// Root hash for tree sizes 0..=8 over `leaf_inputs()`.
const ROOTS: [&str; 9] = [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
];

#[test]
fn primitive_hashes() {
    assert_eq!(Sha256Hasher::empty(), hd(ROOTS[0]));
    assert_eq!(
        Sha256Hasher::leaf(b""),
        hd("6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d")
    );
    assert_eq!(
        Sha256Hasher::leaf(b"L123456"),
        hd("395aa064aa4c29f7010acfe3f25db9485bbd4b91897b6ad7ad547639252b4d56")
    );
    // Upstream's node vector hashes the 4-byte strings "N123" and "N456" as if they were
    // child hashes; `node` takes 32-byte hashes, so pin the same prefix rule directly.
    use sha2::{Digest, Sha256};
    let want: [u8; 32] = Sha256::digest([&[1u8][..], b"N123", b"N456"].concat()).into();
    assert_eq!(
        want,
        hd("aa217fe888e47007fa15edab33c2b492a722cb106c64667fc2b044444de66bbb")
    );
}

#[test]
fn roots_for_every_prefix_size() {
    let d = leaf_inputs();
    for (n, want) in ROOTS.iter().enumerate() {
        for mode in [Build::Sequential, Build::Parallel] {
            let t = MerkleTree::<Sha256Hasher>::build(&d[..n], mode);
            assert_eq!(t.root(), hd(want), "size {n}");
        }
        // and via the full tree at historical size n
        let full = MerkleTree::<Sha256Hasher>::build(&d, Build::Sequential);
        assert_eq!(full.root_at(n).unwrap(), hd(want), "root_at({n})");
    }
}

#[test]
fn compact_tree_matches_each_prefix_root() {
    let mut c = CompactTree::<Sha256Hasher>::new();
    assert_eq!(c.root(), hd(ROOTS[0]));
    for (i, leaf) in leaf_inputs().iter().enumerate() {
        c.append(leaf);
        assert_eq!(c.root(), hd(ROOTS[i + 1]), "size {}", i + 1);
    }
}

/// Audit paths worked out by hand from the published node hashes (level 0 = leaves).
#[test]
fn hand_derived_inclusion_paths() {
    let d = leaf_inputs();
    let t = MerkleTree::<Sha256Hasher>::build(&d, Build::Sequential);
    // leaf 0 of 8: sibling leaf 1, node(2,3), node(4..8)
    let p0 = t.inclusion_proof(0, 8).unwrap();
    assert_eq!(
        p0,
        vec![
            hd("96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"),
            hd("5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e"),
            hd("6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4"),
        ]
    );
    // leaf 5 of 8: sibling leaf 4, node(6,7), node(0..4)
    let p5 = t.inclusion_proof(5, 8).unwrap();
    assert_eq!(
        p5,
        vec![
            hd("bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b"),
            hd("ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0"),
            hd("d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7"),
        ]
    );
    let root = hd(ROOTS[8]);
    verify_inclusion::<Sha256Hasher>(&t.leaf_hash(0).unwrap(), 0, 8, &p0, &root).unwrap();
    verify_inclusion::<Sha256Hasher>(&t.leaf_hash(5).unwrap(), 5, 8, &p5, &root).unwrap();
}

#[test]
#[allow(clippy::needless_range_loop)] // m and n are tree sizes, ROOTS is merely indexed by them
fn consistency_between_every_published_pair() {
    let d = leaf_inputs();
    let t = MerkleTree::<Sha256Hasher>::build(&d, Build::Sequential);
    for m in 1..=8 {
        for n in m..=8 {
            let p = t.consistency_proof(m, n).unwrap();
            verify_consistency::<Sha256Hasher>(m, n, &hd(ROOTS[m]), &hd(ROOTS[n]), &p)
                .unwrap_or_else(|e| panic!("{m}->{n}: {e}"));
        }
    }
    // spot-check a shape the RFC illustrates: 1 -> 8 is the 3 sibling roots
    assert_eq!(t.consistency_proof(1, 8).unwrap().len(), 3);
    assert_eq!(t.consistency_proof(6, 8).unwrap().len(), 3);
}
