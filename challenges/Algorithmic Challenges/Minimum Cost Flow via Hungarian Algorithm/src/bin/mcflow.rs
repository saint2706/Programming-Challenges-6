//! `mcflow`: solve, compare, verify, generate and convert assignment and min-cost-flow instances.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use mcflow::dimacs;
use mcflow::flow::{self, FlowAlgorithm, Network, assignment_network};
use mcflow::instances;
use mcflow::verify::{certify, certify_flow, check_assignment, check_duals, check_potentials};
use mcflow::{Algorithm, Assignment, Dense, Sparse, solve_dense, solve_sparse};

#[derive(Parser)]
#[command(version, about = "Assignment and min-cost flow solvers")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solve an instance and print its optimal cost.
    Solve {
        file: PathBuf,
        /// hungarian, crouse, jv, lapmod, auction, ssp, cost-scaling or network-simplex
        #[arg(long, default_value = "crouse")]
        algo: Algorithm,
        /// Also print every matched pair (assignments) or arc flow (networks).
        #[arg(long)]
        show: bool,
    },
    /// Run every applicable algorithm on one instance and print a timing table.
    Compare { file: PathBuf },
    /// Solve, then prove the answer optimal (duals where the solver has them, and a negative-cycle
    /// search that trusts nothing).
    Verify {
        file: PathBuf,
        #[arg(long, default_value = "crouse")]
        algo: Algorithm,
    },
    /// Write a generated instance to stdout or `--out`.
    Gen {
        #[command(subcommand)]
        what: Gen,
        #[arg(long, global = true)]
        out: Option<PathBuf>,
    },
    /// Rewrite an instance in another format; `--to min` is the assignment-to-flow reduction.
    Convert {
        file: PathBuf,
        #[arg(long)]
        to: Format,
    },
}

#[derive(Subcommand)]
enum Gen {
    /// An assignment instance.
    Assignment {
        #[arg(value_enum)]
        family: Family,
        #[arg(long)]
        n: usize,
        /// Columns (default: same as --n); only for uniform.
        #[arg(long)]
        cols: Option<usize>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Costs are 0..=MAX_COST (uniform, sparse).
        #[arg(long, default_value_t = 1000)]
        max_cost: i64,
        /// Forbid this percentage of the entries of a dense instance (a feasible plant is kept).
        #[arg(long, default_value_t = 0)]
        forbid: u64,
    },
    /// A random feasible min-cost-flow network in DIMACS format.
    Flow {
        #[arg(long)]
        nodes: usize,
        #[arg(long)]
        arcs: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 100)]
        max_cap: i64,
        #[arg(long, default_value_t = 1000)]
        max_cost: i64,
        /// Allow negative arc costs.
        #[arg(long)]
        negative: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Family {
    Uniform,
    MacholWien,
    Geometric,
    Separable,
    /// Sparse planted-matching instance (DIMACS low-cost shape), written as `p asn`.
    Sparse,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    Matrix,
    Asn,
    Min,
}

enum Instance {
    Matrix(Dense),
    Asn(Sparse),
    Min(Network),
}

fn load(path: &Path) -> Result<Instance, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let head = text
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('c') && !l.starts_with('#'))
        .unwrap_or("");
    Ok(if head.starts_with("p min") {
        Instance::Min(dimacs::parse_min(&text)?)
    } else if head.starts_with("p asn") {
        Instance::Asn(dimacs::parse_asn(&text)?)
    } else {
        Instance::Matrix(dimacs::parse_matrix(&text)?)
    })
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}

fn ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1e3)
}

fn solve_assignment(
    inst: &Instance,
    algo: Algorithm,
) -> Result<(Assignment, Sparse, Duration), Box<dyn std::error::Error>> {
    match inst {
        Instance::Matrix(d) => {
            let (a, t) = timed(|| solve_dense(d, algo));
            Ok((a?, d.to_sparse(), t))
        }
        Instance::Asn(g) => {
            let (a, t) = timed(|| solve_sparse(g, algo));
            Ok((a?, g.clone(), t))
        }
        Instance::Min(_) => unreachable!("callers route networks elsewhere"),
    }
}

fn flow_algo(algo: Algorithm) -> Result<FlowAlgorithm, String> {
    algo.flow_algorithm().ok_or_else(|| {
        format!(
            "{algo} solves assignments only; a min-cost flow file takes ssp, cost-scaling or network-simplex"
        )
    })
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Solve { file, algo, show } => match load(&file)? {
            Instance::Min(net) => {
                let (f, t) =
                    timed(|| flow::solve(&net, flow_algo(algo)?).map_err(|e| e.to_string()));
                let f = f?;
                println!(
                    "cost {} ({} nodes, {} arcs, {algo}, {})",
                    f.cost,
                    net.nodes(),
                    net.arcs().len(),
                    ms(t)
                );
                if show {
                    for (k, (a, x)) in net.arcs().iter().zip(&f.flow).enumerate() {
                        if *x != 0 {
                            println!("arc {k}: {} -> {} carries {x}", a.from + 1, a.to + 1);
                        }
                    }
                }
            }
            inst => {
                let (a, g, t) = solve_assignment(&inst, algo)?;
                println!(
                    "cost {} ({}x{}, {algo}, {})",
                    a.cost,
                    g.rows(),
                    g.cols(),
                    ms(t)
                );
                if show {
                    for (i, j) in a.pairs() {
                        println!(
                            "row {i} -> column {j} (cost {})",
                            g.get(i, j).expect("matched edge exists")
                        );
                    }
                }
            }
        },
        Command::Compare { file } => compare(&load(&file)?)?,
        Command::Verify { file, algo } => match load(&file)? {
            Instance::Min(net) => {
                let f = flow::solve(&net, flow_algo(algo)?)?;
                certify_flow(&net, &f)?;
                if let Some(pi) = &f.potential {
                    check_potentials(&net, &f, pi)?;
                    println!(
                        "cost {}: optimal (negative-cycle search and potentials both agree)",
                        f.cost
                    );
                } else {
                    println!("cost {}: optimal (negative-cycle search)", f.cost);
                }
            }
            inst => {
                let (a, g, _) = solve_assignment(&inst, algo)?;
                check_assignment(&g, &a)?;
                certify(&g, &a)?;
                if a.duals.is_some() {
                    check_duals(&g, &a)?;
                    println!(
                        "cost {}: optimal (dual certificate and negative-cycle search agree)",
                        a.cost
                    );
                } else {
                    println!(
                        "cost {}: optimal (negative-cycle search; {algo} reports no duals)",
                        a.cost
                    );
                }
            }
        },
        Command::Gen { what, out } => {
            let text = match what {
                Gen::Assignment {
                    family,
                    n,
                    cols,
                    seed,
                    max_cost,
                    forbid,
                } => {
                    let dense = |d: Dense| {
                        let d = if forbid > 0 {
                            instances::forbid(&d, forbid, seed ^ 0xF0F0)
                        } else {
                            d
                        };
                        dimacs::write_matrix(&d)
                    };
                    match family {
                        Family::Uniform => {
                            dense(instances::uniform(n, cols.unwrap_or(n), max_cost, seed))
                        }
                        Family::MacholWien => dense(instances::machol_wien(n)),
                        Family::Geometric => dense(instances::geometric(n, 1000, seed)),
                        Family::Separable => dense(instances::separable(n, max_cost, 5, seed)),
                        Family::Sparse => dimacs::write_asn(&instances::dimacs_sparse(n, seed)),
                    }
                }
                Gen::Flow {
                    nodes,
                    arcs,
                    seed,
                    max_cap,
                    max_cost,
                    negative,
                } => dimacs::write_min(&instances::random_network(
                    nodes, arcs, max_cap, max_cost, negative, seed,
                )?),
            };
            match out {
                Some(path) => std::fs::write(path, text)?,
                None => print!("{text}"),
            }
        }
        Command::Convert { file, to } => {
            let text = match (load(&file)?, to) {
                (Instance::Matrix(d), Format::Matrix) => dimacs::write_matrix(&d),
                (Instance::Matrix(d), Format::Asn) => dimacs::write_asn(&d.to_sparse()),
                (Instance::Matrix(d), Format::Min) => {
                    dimacs::write_min(&assignment_network(&d.to_sparse())?)
                }
                (Instance::Asn(g), Format::Asn) => dimacs::write_asn(&g),
                (Instance::Asn(g), Format::Matrix) => dimacs::write_matrix(&g.to_dense()?),
                (Instance::Asn(g), Format::Min) => dimacs::write_min(&assignment_network(&g)?),
                (Instance::Min(net), Format::Min) => dimacs::write_min(&net),
                (Instance::Min(_), _) => {
                    return Err("a flow network cannot be turned back into an assignment".into());
                }
            };
            print!("{text}");
        }
    }
    Ok(())
}

fn compare(inst: &Instance) -> Result<(), Box<dyn std::error::Error>> {
    match inst {
        Instance::Min(net) => {
            println!("{:<16} {:>16} {:>12}", "algorithm", "cost", "time");
            for algo in FlowAlgorithm::ALL {
                let (f, t) = timed(|| flow::solve(net, algo));
                match f {
                    Ok(f) => println!("{:<16} {:>16} {:>12}", algo.name(), f.cost, ms(t)),
                    Err(e) => {
                        println!("{:<16} {:>16} {:>12}", algo.name(), format!("({e})"), ms(t))
                    }
                }
            }
        }
        inst => {
            println!("{:<16} {:>16} {:>12}", "algorithm", "cost", "time");
            for algo in Algorithm::ALL {
                match solve_assignment(inst, algo) {
                    Ok((a, _, t)) => println!("{:<16} {:>16} {:>12}", algo.name(), a.cost, ms(t)),
                    Err(e) => println!("{:<16} {:>16} {:>12}", algo.name(), format!("({e})"), "-"),
                }
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
