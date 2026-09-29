//! CLI: build a tree over the lines of a file, then prove and verify.

use clap::{Parser, Subcommand, ValueEnum};
use merkle_tree::hasher::Hasher;
use merkle_tree::insecure::NoDomainSha256;
use merkle_tree::verify::{verify_consistency, verify_inclusion_data};
use merkle_tree::{Blake3Hasher, Build, Hash, MerkleTree, Sha256Hasher};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Clone, Copy, ValueEnum)]
enum Algo {
    Sha256,
    Blake3,
}

#[derive(Parser)]
#[command(about = "RFC 9162 Merkle tree builder and proof verifier")]
struct Cli {
    /// Hash function (sha256 is RFC 6962/9162 compatible).
    #[arg(long, value_enum, default_value = "sha256", global = true)]
    hash: Algo,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the root over the lines of FILE.
    Root { file: PathBuf },
    /// Print the inclusion proof for line INDEX (0-based) of FILE.
    Prove { file: PathBuf, index: usize },
    /// Verify an inclusion proof for LEAF (raw text) against ROOT.
    Verify {
        root: String,
        index: usize,
        size: usize,
        leaf: String,
        /// Proof hashes (hex), deepest first.
        proof: Vec<String>,
    },
    /// Print the consistency proof from the first FIRST lines of FILE to all of FILE.
    Consistency { file: PathBuf, first: usize },
    /// Verify a consistency proof.
    VerifyConsistency {
        first: usize,
        second: usize,
        first_root: String,
        second_root: String,
        proof: Vec<String>,
    },
    /// Show the second-preimage attack on a hasher without 0x00/0x01 prefixes.
    Attack,
}

fn parse_hash(s: &str) -> Result<Hash, String> {
    let bytes = hex::decode(s).map_err(|e| format!("bad hex {s:?}: {e}"))?;
    bytes
        .try_into()
        .map_err(|_| format!("{s:?} is not 32 bytes"))
}

fn lines(file: &PathBuf) -> Result<Vec<Vec<u8>>, String> {
    let text = fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(text.lines().map(|l| l.as_bytes().to_vec()).collect())
}

fn hexes(proof: &[Hash]) -> String {
    proof.iter().map(hex::encode).collect::<Vec<_>>().join(" ")
}

fn run<H: Hasher>(cmd: Cmd) -> Result<(), String> {
    match cmd {
        Cmd::Root { file } => {
            let d = lines(&file)?;
            let t = MerkleTree::<H>::build(&d, Build::Parallel);
            println!("{} leaves, root {}", t.len(), hex::encode(t.root()));
        }
        Cmd::Prove { file, index } => {
            let d = lines(&file)?;
            let t = MerkleTree::<H>::build(&d, Build::Parallel);
            let p = t
                .inclusion_proof(index, t.len())
                .map_err(|e| e.to_string())?;
            println!("root  {}", hex::encode(t.root()));
            println!("size  {}", t.len());
            println!("proof {}", hexes(&p));
        }
        Cmd::Verify {
            root,
            index,
            size,
            leaf,
            proof,
        } => {
            let proof = proof
                .iter()
                .map(|s| parse_hash(s))
                .collect::<Result<Vec<_>, _>>()?;
            match verify_inclusion_data::<H>(
                leaf.as_bytes(),
                index,
                size,
                &proof,
                &parse_hash(&root)?,
            ) {
                Ok(()) => println!("VALID"),
                Err(e) => return Err(format!("INVALID: {e}")),
            }
        }
        Cmd::Consistency { file, first } => {
            let d = lines(&file)?;
            let t = MerkleTree::<H>::build(&d, Build::Parallel);
            let p = t
                .consistency_proof(first, t.len())
                .map_err(|e| e.to_string())?;
            println!(
                "first_root  {}",
                hex::encode(t.root_at(first).map_err(|e| e.to_string())?)
            );
            println!("second_root {}", hex::encode(t.root()));
            println!("proof       {}", hexes(&p));
        }
        Cmd::VerifyConsistency {
            first,
            second,
            first_root,
            second_root,
            proof,
        } => {
            let proof = proof
                .iter()
                .map(|s| parse_hash(s))
                .collect::<Result<Vec<_>, _>>()?;
            match verify_consistency::<H>(
                first,
                second,
                &parse_hash(&first_root)?,
                &parse_hash(&second_root)?,
                &proof,
            ) {
                Ok(()) => println!("VALID"),
                Err(e) => return Err(format!("INVALID: {e}")),
            }
        }
        Cmd::Attack => attack(),
    }
    Ok(())
}

fn attack() {
    let leaves: Vec<&[u8]> = vec![b"tx-a", b"tx-b", b"tx-c", b"tx-d"];
    println!("real tree: 4 leaves");
    forge::<NoDomainSha256>("no domain separation", &leaves);
    forge::<Sha256Hasher>("RFC 9162 (0x00/0x01)", &leaves);
}

fn forge<H: Hasher>(name: &str, leaves: &[&[u8]]) {
    let real = MerkleTree::<H>::build(leaves, Build::Sequential);
    // The attacker's 2-leaf "data" are the interior nodes of the real tree written out as
    // bytes: x = leafhash(a) || leafhash(b), y = leafhash(c) || leafhash(d).
    let x = [H::leaf(leaves[0]), H::leaf(leaves[1])].concat();
    let y = [H::leaf(leaves[2]), H::leaf(leaves[3])].concat();
    let forged = MerkleTree::<H>::build(&[x, y], Build::Sequential);
    let same = forged.root() == real.root();
    println!("  {name}: forged 2-leaf tree root == real root ? {same}");
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match cli.hash {
        Algo::Sha256 => run::<Sha256Hasher>(cli.cmd),
        Algo::Blake3 => run::<Blake3Hasher>(cli.cmd),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
