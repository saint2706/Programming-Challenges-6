//! `lct`: run forest scripts on any structure, solve offline graph connectivity, and generate
//! inputs for both.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use lct::connectivity::{random_events, solve_offline, solve_rebuild};
use lct::script::{Structure, generate_script, parse_events, render_events, run_script};
use lct::shapes::Shape;

#[derive(Parser)]
#[command(
    version,
    about = "Dynamic forests: link-cut tree, Euler tour tree, UFO tree, offline connectivity"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a forest script (see `lct::script` for the commands) and print one line per command.
    Run {
        file: PathBuf,
        /// naive, lct, subtree-lct, ett or ufo
        #[arg(long, default_value = "subtree-lct", value_parser = parse_structure)]
        structure: Structure,
    },
    /// Print a random forest script for `run`.
    Gen {
        #[arg(long, default_value = "random", value_parser = parse_shape)]
        shape: Shape,
        #[arg(long, default_value_t = 1000)]
        n: usize,
        #[arg(long, default_value_t = 1000)]
        ops: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Answer an offline connectivity file (`add u v`, `del u v`, `conn u v`) with a link-cut tree.
    Connect {
        file: PathBuf,
        /// Also rebuild a union-find per query and check that the answers agree.
        #[arg(long)]
        verify: bool,
    },
    /// Print a random offline connectivity file for `connect`.
    GenEvents {
        #[arg(long, default_value_t = 1000)]
        n: usize,
        /// Live edges the graph hovers around.
        #[arg(long, default_value_t = 2000)]
        edges: usize,
        #[arg(long, default_value_t = 10000)]
        events: usize,
        #[arg(long, default_value_t = 30)]
        query_percent: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

fn parse_structure(s: &str) -> Result<Structure, String> {
    Structure::from_name(s).ok_or_else(|| {
        let names: Vec<_> = Structure::ALL.iter().map(|s| s.name()).collect();
        format!("expected one of {}", names.join(", "))
    })
}

fn parse_shape(s: &str) -> Result<Shape, String> {
    Shape::from_name(s).ok_or_else(|| {
        let names: Vec<_> = Shape::ALL.iter().map(|s| s.name()).collect();
        format!("expected one of {}", names.join(", "))
    })
}

fn read(file: &PathBuf) -> Result<String, ExitCode> {
    std::fs::read_to_string(file).map_err(|e| {
        eprintln!("error: cannot read {}: {e}", file.display());
        ExitCode::from(2)
    })
}

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {message}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Cmd::Run { file, structure } => {
            let text = match read(&file) {
                Ok(t) => t,
                Err(code) => return code,
            };
            match run_script(&text, structure) {
                Ok(lines) => {
                    lines.iter().for_each(|l| println!("{l}"));
                    ExitCode::SUCCESS
                }
                Err(e) => fail(e),
            }
        }
        Cmd::Gen {
            shape,
            n,
            ops,
            seed,
        } => {
            if n == 0 {
                return fail("--n must be at least 1");
            }
            print!("{}", generate_script(shape, n, ops, seed));
            ExitCode::SUCCESS
        }
        Cmd::Connect { file, verify } => {
            let text = match read(&file) {
                Ok(t) => t,
                Err(code) => return code,
            };
            let (n, events) = match parse_events(&text) {
                Ok(parsed) => parsed,
                Err(e) => return fail(e),
            };
            let solved = match solve_offline(n, &events) {
                Ok(s) => s,
                Err(e) => return fail(e),
            };
            for answer in &solved.answers {
                println!("{answer}");
            }
            eprintln!(
                "{} queries; forest: {} joins, {} swaps, {} non-tree edges dropped, {} tree deletions",
                solved.answers.len(),
                solved.tree_inserts,
                solved.swaps,
                solved.dropped,
                solved.tree_deletes
            );
            if verify {
                match solve_rebuild(n, &events) {
                    Ok(slow) if slow == solved.answers => eprintln!("verified against union-find"),
                    Ok(_) => return fail("answers differ from the union-find rebuild"),
                    Err(e) => return fail(e),
                }
            }
            ExitCode::SUCCESS
        }
        Cmd::GenEvents {
            n,
            edges,
            events,
            query_percent,
            seed,
        } => {
            if n < 2 || query_percent > 100 {
                return fail("--n must be at least 2 and --query-percent at most 100");
            }
            print!(
                "{}",
                render_events(n, &random_events(n, edges, events, query_percent, seed))
            );
            ExitCode::SUCCESS
        }
    }
}
