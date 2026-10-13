//! `hld`: run a query script on a tree, or generate one.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use hld::script::{Structure, run_script};
use hld::shapes::{Rng, Shape, generate};

#[derive(Parser)]
#[command(
    version,
    about = "Tree path queries: classic HLD, globally balanced HLD, top tree"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a script (see `hld::script` for the format) and print one line per query.
    Run {
        file: PathBuf,
        /// naive, classic, balanced (no subtree queries) or toptree
        #[arg(long, default_value = "toptree", value_parser = parse_structure)]
        structure: Structure,
        /// Weights are on edges: edge lines are `u v w` and there is no weights line.
        #[arg(long)]
        edges: bool,
        #[arg(long, default_value_t = 0)]
        root: usize,
    },
    /// Print a random script for `run` to stdout.
    Gen {
        #[arg(long, default_value = "random", value_parser = parse_shape)]
        shape: Shape,
        #[arg(long, default_value_t = 1000)]
        n: usize,
        #[arg(long, default_value_t = 1000)]
        queries: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

fn parse_structure(s: &str) -> Result<Structure, String> {
    Structure::from_name(s)
        .ok_or_else(|| "expected naive, classic, balanced or toptree".to_string())
}

fn parse_shape(s: &str) -> Result<Shape, String> {
    Shape::from_name(s).ok_or_else(|| {
        let names: Vec<_> = Shape::ALL.iter().map(|s| s.name()).collect();
        format!("expected one of {}", names.join(", "))
    })
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Cmd::Run {
            file,
            structure,
            edges,
            root,
        } => {
            let text = match std::fs::read_to_string(&file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: cannot read {}: {e}", file.display());
                    return ExitCode::from(2);
                }
            };
            match run_script(&text, structure, edges, root) {
                Ok(lines) => {
                    for line in lines {
                        println!("{line}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(2)
                }
            }
        }
        Cmd::Gen {
            shape,
            n,
            queries,
            seed,
        } => {
            if n == 0 {
                eprintln!("error: --n must be at least 1");
                return ExitCode::from(2);
            }
            let tree = generate(shape, n, seed);
            let mut rng = Rng::new(seed ^ 0xDEAD_BEEF);
            let mut out = format!("{n}\n");
            let weights: Vec<String> = (0..n)
                .map(|_| (rng.below(2001) as i64 - 1000).to_string())
                .collect();
            let _ = writeln!(out, "{}", weights.join(" "));
            for (a, b) in &tree.edges {
                let _ = writeln!(out, "{a} {b}");
            }
            for _ in 0..queries {
                let (u, v) = (rng.below(n), rng.below(n));
                let _ = match rng.below(6) {
                    0 | 1 => writeln!(out, "path {u} {v}"),
                    2 => writeln!(out, "add {u} {v} {}", rng.below(21) as i64 - 10),
                    3 => writeln!(out, "set {u} {}", rng.below(2001) as i64 - 1000),
                    4 => writeln!(out, "sub {u}"),
                    _ => writeln!(out, "lca {u} {v}"),
                };
            }
            print!("{out}");
            ExitCode::SUCCESS
        }
    }
}
