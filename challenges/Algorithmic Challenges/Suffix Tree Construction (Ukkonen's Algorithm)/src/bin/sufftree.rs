//! `sufftree`: build suffix trees with any of the algorithms and query them.

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use sufftree::apps::{longest_common_substring, matching_statistics, shortest_unique_substring};
use sufftree::array_index::longest_common_substring_array;
use sufftree::satree::{tree_of_bytes as sais_tree, tree_of_words};
use sufftree::ukkonen::{HashChildren, ListChildren, OnlineSuffixTree, tree_of_bytes as ukkonen};
use sufftree::{Bytes, SuffixTree, pwotd};

#[derive(Parser)]
#[command(
    version,
    about = "Suffix trees: Ukkonen, SA-IS + LCP, and top-down partitioning"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Algo {
    /// Ukkonen's online algorithm with sibling-list branching
    Ukkonen,
    /// Ukkonen's online algorithm with a hash map for branching
    UkkonenHash,
    /// Suffix array by SA-IS, then LCP, then the tree from the LCP intervals
    Sais,
    /// Partitioned write-only top-down (PWOTD)
    Pwotd,
    /// Write-only top-down without the partition phase
    Wotd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the tree of a file and print its size and build time.
    Build {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "ukkonen")]
        algo: Algo,
    },
    /// Count and locate patterns in a file.
    Find {
        file: PathBuf,
        patterns: Vec<String>,
        #[arg(long, value_enum, default_value = "ukkonen")]
        algo: Algo,
        /// Positions to print per pattern.
        #[arg(long, default_value_t = 10)]
        show: usize,
    },
    /// The longest substring that occurs at least twice.
    Lrs { file: PathBuf },
    /// The longest substring common to all the files.
    Lcs {
        files: Vec<PathBuf>,
        /// Use the suffix array and a sliding window instead of the generalized suffix tree.
        #[arg(long)]
        array: bool,
    },
    /// The shortest substring that occurs exactly once.
    Unique { file: PathBuf },
    /// For every position of QUERY, how much of it FILE contains.
    Ms { file: PathBuf, query: PathBuf },
    /// Feed a file to the online tree byte by byte, reporting PATTERN's occurrences as it grows.
    Online {
        file: PathBuf,
        pattern: String,
        /// Report after every this many bytes.
        #[arg(long, default_value_t = 1000)]
        every: usize,
    },
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut data))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(data)
}

fn build(algo: Algo, bytes: &[u8]) -> SuffixTree<Bytes> {
    match algo {
        Algo::Ukkonen => ukkonen::<ListChildren>(bytes),
        Algo::UkkonenHash => ukkonen::<HashChildren>(bytes),
        Algo::Sais => sais_tree(bytes),
        Algo::Pwotd => pwotd::build(Bytes::new(bytes)),
        Algo::Wotd => pwotd::build_with(Bytes::new(bytes), Some(0)),
    }
}

fn show(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).escape_debug().to_string()
}

fn run(cmd: Cmd) -> Result<(), String> {
    match cmd {
        Cmd::Build { file, algo } => {
            let bytes = read(&file)?;
            let t = Instant::now();
            let tree = build(algo, &bytes);
            let took = t.elapsed();
            println!(
                "{} bytes -> {} nodes ({} leaves), {:.1} B/byte in the node arrays, {:.3} s",
                bytes.len(),
                tree.node_count(),
                tree.leaf_count(),
                tree.heap_bytes() as f64 / bytes.len().max(1) as f64,
                took.as_secs_f64()
            );
        }
        Cmd::Find {
            file,
            patterns,
            algo,
            show: limit,
        } => {
            let bytes = read(&file)?;
            let tree = build(algo, &bytes);
            for p in &patterns {
                let at = tree.locate(p.as_bytes());
                let shown: Vec<String> = at.iter().take(limit).map(usize::to_string).collect();
                let more = if at.len() > limit { ", ..." } else { "" };
                println!(
                    "{p:?}: {} occurrences [{}{more}]",
                    at.len(),
                    shown.join(", ")
                );
            }
        }
        Cmd::Lrs { file } => {
            let bytes = read(&file)?;
            let (start, len) = sais_tree(&bytes).longest_repeated_substring();
            println!(
                "length {len} at {start}: {:?}",
                show(&bytes[start..start + len])
            );
        }
        Cmd::Lcs { files, array } => {
            if files.len() < 2 {
                return Err("need at least two files".into());
            }
            let strings = files.iter().map(read).collect::<Result<Vec<_>, _>>()?;
            let found = if array {
                longest_common_substring_array(&strings)
            } else {
                longest_common_substring(&tree_of_words(&strings))
            };
            match found {
                Some((j, start, len)) => println!(
                    "length {len}, e.g. {} at offset {start}: {:?}",
                    files[j].display(),
                    show(&strings[j][start..start + len])
                ),
                None => println!("the files share no substring"),
            }
        }
        Cmd::Unique { file } => {
            let bytes = read(&file)?;
            match shortest_unique_substring(&sais_tree(&bytes)) {
                Some((start, len)) => {
                    println!(
                        "length {len} at {start}: {:?}",
                        show(&bytes[start..start + len])
                    )
                }
                None => println!("empty text"),
            }
        }
        Cmd::Ms { file, query } => {
            let bytes = read(&file)?;
            let q = read(&query)?;
            let ms = matching_statistics(&sais_tree(&bytes), &q);
            let best = ms.iter().enumerate().max_by_key(|&(_, &l)| l);
            if let Some((i, &l)) = best {
                println!(
                    "longest match: {l} bytes at query offset {i}: {:?}",
                    show(&q[i..i + l])
                );
            }
            println!(
                "mean match length {:.2}",
                ms.iter().sum::<usize>() as f64 / ms.len().max(1) as f64
            );
        }
        Cmd::Online {
            file,
            pattern,
            every,
        } => {
            if every == 0 {
                return Err("--every must be at least 1".into());
            }
            let bytes = read(&file)?;
            let mut tree = OnlineSuffixTree::new();
            for (i, &b) in bytes.iter().enumerate() {
                tree.push(b);
                if (i + 1) % every == 0 || i + 1 == bytes.len() {
                    let at = tree.locate(pattern.as_bytes());
                    println!(
                        "after {:>9} bytes: {:>6} occurrences, {:>9} nodes",
                        i + 1,
                        at.len(),
                        tree.node_count()
                    );
                }
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
