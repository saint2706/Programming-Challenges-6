//! The classic Merkle second-preimage attack, with and without RFC 9162 domain separation.

use merkle_tree::insecure::NoDomainSha256;
use merkle_tree::verify::verify_inclusion_data;
use merkle_tree::{Build, Hasher, MerkleTree, Sha256Hasher};

/// Returns (forged tree has the real root, forged inclusion proof is accepted).
fn attack<H: Hasher>() -> (bool, bool) {
    let leaves: Vec<&[u8]> = vec![b"pay alice 5", b"pay bob 7", b"pay carol 1", b"pay dave 9"];
    let real = MerkleTree::<H>::build(&leaves, Build::Sequential);
    let root = real.root();

    // The attacker forges a 2-leaf tree whose "leaves" are the interior nodes over (a,b)
    // and (c,d), written out as 64-byte strings.
    let x = [H::leaf(leaves[0]), H::leaf(leaves[1])].concat();
    let y = [H::leaf(leaves[2]), H::leaf(leaves[3])].concat();
    let forged = MerkleTree::<H>::build(&[x.clone(), y.clone()], Build::Sequential);
    let same_root = forged.root() == root;

    // ...and can then "prove" that the 64-byte string x is leaf 0 of a 2-leaf tree with
    // the real root, using y's leaf hash as the single sibling.
    let proof = [H::leaf(&y)];
    let accepted = verify_inclusion_data::<H>(&x, 0, 2, &proof, &root).is_ok();
    (same_root, accepted)
}

#[test]
fn without_domain_separation_the_forgery_verifies() {
    assert_eq!(attack::<NoDomainSha256>(), (true, true));
}

#[test]
fn with_rfc9162_domain_separation_the_forgery_fails() {
    assert_eq!(attack::<Sha256Hasher>(), (false, false));
}

#[test]
fn leaf_and_node_hashes_never_collide_across_domains() {
    let a = Sha256Hasher::leaf(b"x");
    let b = Sha256Hasher::leaf(b"y");
    // Feeding a node's preimage (minus prefix) as leaf data must not reproduce the node.
    let as_leaf_data = [a.as_slice(), b.as_slice()].concat();
    assert_ne!(
        Sha256Hasher::leaf(&as_leaf_data),
        Sha256Hasher::node(&a, &b)
    );
}
