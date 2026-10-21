//! `pst`: ask historical range questions from the command line.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use persegtree::oracle::{distinct_by_hashing, kth_by_sorting};
use persegtree::{DistinctIndex, History, KthIndex, SumI64, WaveletMatrix};

#[derive(Parser)]
#[command(
    version,
    about = "Historical range queries on persistent segment trees"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The K-th smallest (0-based) of the numbers L..R (end exclusive) of a file.
    Kth {
        file: PathBuf,
        l: usize,
        r: usize,
        k: usize,
        #[arg(long, value_enum, default_value_t = Engine::Persistent)]
        engine: Engine,
    },
    /// How many distinct numbers the range L..R of a file holds.
    Distinct {
        file: PathBuf,
        l: usize,
        r: usize,
        /// Count by hashing the range instead of using the persistent index.
        #[arg(long)]
        brute_force: bool,
    },
    /// Replay a time-travel script and print every answer.
    ///
    /// Lines: `size N` (first), `set TIME POS VALUE`, `get TIME POS`, `sum TIME L R`,
    /// `forget TIME` (drop history older than TIME), `stats`. `#` starts a comment.
    History { script: PathBuf },
    /// Build every engine on a file and print what each costs in memory.
    Sizes { file: PathBuf },
}

#[derive(Clone, Copy, ValueEnum)]
enum Engine {
    /// Path-copying persistent count tree.
    Persistent,
    /// Static wavelet matrix.
    Wavelet,
    /// Sort the range (no index).
    Sort,
}

/// Reads whitespace- or comma-separated integers.
fn read_numbers(path: &Path) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    text.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<i64>()
                .map_err(|_| format!("{t:?} is not an integer").into())
        })
        .collect()
}

fn check_range(len: usize, l: usize, r: usize) -> Result<(), String> {
    if l > r || r > len {
        return Err(format!("range {l}..{r} does not fit {len} numbers"));
    }
    Ok(())
}

fn ranks(a: &[i64]) -> (Vec<u32>, Vec<i64>) {
    let mut sorted = a.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let ranks = a
        .iter()
        .map(|x| {
            sorted
                .binary_search(x)
                .expect("present in its own sorted copy") as u32
        })
        .collect();
    (ranks, sorted)
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Kth {
            file,
            l,
            r,
            k,
            engine,
        } => {
            let a = read_numbers(&file)?;
            check_range(a.len(), l, r)?;
            let answer = match engine {
                Engine::Persistent => KthIndex::new(&a).kth_smallest(l, r, k),
                Engine::Wavelet => {
                    let (rank, sorted) = ranks(&a);
                    WaveletMatrix::new(&rank)
                        .kth_smallest(l, r, k)
                        .map(|rk| sorted[rk as usize])
                }
                Engine::Sort => kth_by_sorting(&a, l, r, k),
            };
            match answer {
                Some(v) => println!("{v}"),
                None => return Err(format!("k = {k} is out of range for {} numbers", r - l).into()),
            }
        }
        Command::Distinct {
            file,
            l,
            r,
            brute_force,
        } => {
            let a = read_numbers(&file)?;
            check_range(a.len(), l, r)?;
            let n = if brute_force {
                distinct_by_hashing(&a, l, r)
            } else {
                DistinctIndex::new(&a).distinct(l, r)
            };
            println!("{n}");
        }
        Command::History { script } => replay(&std::fs::read_to_string(script)?)?,
        Command::Sizes { file } => {
            let a = read_numbers(&file)?;
            let kth = KthIndex::new(&a);
            let (rank, _) = ranks(&a);
            let wm = WaveletMatrix::new(&rank);
            println!("{} numbers, {} distinct", a.len(), kth.distinct_values());
            println!("{:<12} {:>12} {:>14}", "engine", "bytes", "bytes/number");
            let per = |b: usize| b as f64 / a.len().max(1) as f64;
            println!(
                "{:<12} {:>12} {:>14.1}",
                "persistent",
                kth.bytes(),
                per(kth.bytes())
            );
            println!(
                "{:<12} {:>12} {:>14.1}",
                "wavelet",
                wm.bytes(),
                per(wm.bytes())
            );
            println!("{:<12} {:>12} {:>14.1}", "plain array", a.len() * 8, 8.0);
        }
    }
    Ok(())
}

/// Largest `|value|` a script may set: sums of up to 2^22 such values stay far inside an `i64`.
const MAX_VALUE: u64 = 1 << 40;

fn replay(script: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut history: Option<History<SumI64>> = None;
    for (no, raw) in script.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let at = |msg: String| format!("line {}: {msg}", no + 1);
        let t: Vec<&str> = line.split_whitespace().collect();
        let num = |i: usize| -> Result<i64, String> {
            t.get(i)
                .ok_or_else(|| at(format!("`{}` needs more arguments", t[0])))?
                .parse::<i64>()
                .map_err(|_| at(format!("{:?} is not an integer", t[i])))
        };
        if t[0] == "size" {
            let n = usize::try_from(num(1)?).map_err(|_| at("size must be positive".into()))?;
            if n == 0 {
                return Err(at("size must be positive".into()).into());
            }
            history = Some(History::new(n));
            continue;
        }
        let h = history
            .as_mut()
            .ok_or_else(|| at("the script must start with `size N`".into()))?;
        let time =
            |i: usize| u64::try_from(num(i)?).map_err(|_| at("time must not be negative".into()));
        let pos = |i: usize| {
            usize::try_from(num(i)?).map_err(|_| at("positions must not be negative".into()))
        };
        let n = h.tree().len();
        match t[0] {
            "set" => {
                let (when, p) = (time(1)?, pos(2)?);
                if p >= n {
                    return Err(at(format!("position {p} out of range for size {n}")).into());
                }
                if when < h.latest_time() {
                    return Err(at(format!(
                        "time {when} is before the last change at {}",
                        h.latest_time()
                    ))
                    .into());
                }
                let value = num(3)?;
                if value.unsigned_abs() > MAX_VALUE {
                    return Err(at(format!("value {value} is beyond +-2^40")).into());
                }
                h.set(when, p, value);
            }
            "get" => {
                let p = pos(2)?;
                if p >= n {
                    return Err(at(format!("position {p} out of range for size {n}")).into());
                }
                match h.get_at(time(1)?, p) {
                    Some(v) => println!("get {} {p} = {v}", t[1]),
                    None => println!("get {} {p} = (history forgotten)", t[1]),
                }
            }
            "sum" => {
                let (l, r) = (pos(2)?, pos(3)?);
                if l > r || r > n {
                    return Err(at(format!("range {l}..{r} does not fit size {n}")).into());
                }
                match h.query_at(time(1)?, l, r) {
                    Some(v) => println!("sum {} {l}..{r} = {v}", t[1]),
                    None => println!("sum {} {l}..{r} = (history forgotten)", t[1]),
                }
            }
            "forget" => {
                let retired = h.forget_before(time(1)?);
                println!(
                    "forget {}: retired {retired} versions, {} retained",
                    t[1],
                    h.retained()
                );
            }
            "stats" => {
                let s = h.tree().stats();
                println!(
                    "stats: {} versions created, {} retained, {} live nodes, {} slots, {} bytes",
                    s.versions,
                    h.retained(),
                    s.live_nodes,
                    s.slots,
                    s.bytes
                );
            }
            other => return Err(at(format!("unknown command {other:?}")).into()),
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
