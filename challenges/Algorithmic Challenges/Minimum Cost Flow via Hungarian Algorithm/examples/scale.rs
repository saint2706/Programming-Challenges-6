//! Time and peak heap of every solver on every instance family, as markdown tables.
//!
//! `cargo run --release --example scale [dense|sparse|flow|tuning|all]` (default `all`).
//!
//! The parent process never solves anything. For each cell (solver x family x size) it starts
//! itself again with `--cell ...` and waits up to `TIMEOUT`, so a solver that blows up shows as
//! `DNF` instead of hanging the run, and each cell gets a fresh heap for the peak-memory figure.
//! A child builds the instance, then times `build + solve` from the matrix (for the flow solvers
//! that includes building the network; for the rival crates, building their matrix type) and
//! reports the median of a few repetitions when a run is short. Peak heap is the high-water mark of
//! live bytes above what the instance itself occupies, from a counting allocator.
//!
//! `skip` means the cell was not attempted because its input would not fit (dense solvers on a
//! sparse instance with more than 2^25 cells, flow solvers on more than 2^21 arcs).

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use mcflow::cost_scaling::{self, Options};
use mcflow::flow::{self, FlowAlgorithm, Network, assignment_network};
use mcflow::instances;
use mcflow::{Dense, Sparse, solve_dense, solve_sparse};

const TIMEOUT: Duration = Duration::from_secs(90);

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                let now =
                    LIVE.fetch_add(new_size - layout.size(), Relaxed) + new_size - layout.size();
                PEAK.fetch_max(now, Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

// ---------------------------------------------------------------------------------------------
// Child: one cell.
// ---------------------------------------------------------------------------------------------

enum Input {
    Dense(Dense),
    Sparse(Sparse),
    Network(Network),
}

fn build(family: &str, n: usize, seed: u64) -> Input {
    match family {
        "uniform" => Input::Dense(instances::uniform(n, n, 1000, seed)),
        "narrow" => Input::Dense(instances::uniform(n, n, 10, seed)),
        "geometric" => Input::Dense(instances::geometric(n, 1000, seed)),
        "separable" => Input::Dense(instances::separable(n, 1000, 5, seed)),
        "machol-wien" => Input::Dense(instances::machol_wien(n)),
        "sparse" => Input::Sparse(instances::dimacs_sparse(n, seed)),
        "flow" => {
            Input::Network(instances::random_network(n, 5 * n, 100, 1000, false, seed).unwrap())
        }
        "flow-neg" => {
            Input::Network(instances::random_network(n, 5 * n, 100, 1000, true, seed).unwrap())
        }
        other => panic!("unknown family {other}"),
    }
}

fn solve_cell(algo: &str, input: &Input) -> (String, String) {
    match (algo, input) {
        ("rival-lapjv", Input::Dense(d)) => {
            let m = ndarray_matrix(d);
            let (rows, _) = lapjv::lapjv(&m).unwrap();
            let cost: i64 = rows.iter().enumerate().map(|(i, &j)| d.get(i, j)).sum();
            (cost.to_string(), String::new())
        }
        ("rival-pathfinding", Input::Dense(d)) => {
            let m = pathfinding::matrix::Matrix::from_fn(d.rows(), d.cols(), |(i, j)| d.get(i, j));
            (
                pathfinding::kuhn_munkres::kuhn_munkres_min(&m)
                    .0
                    .to_string(),
                String::new(),
            )
        }
        ("hungarian-first", Input::Dense(d)) => (
            mcflow::hungarian::optimal_cost(d, mcflow::hungarian::TieBreak::First)
                .unwrap()
                .to_string(),
            String::new(),
        ),
        ("cs-plain", Input::Dense(d)) => cs(&assignment_network(&d.to_sparse()).unwrap(), false),
        ("cs-plain", Input::Sparse(g)) => cs(&assignment_network(g).unwrap(), false),
        ("cs-plain", Input::Network(net)) => cs(net, false),
        ("cost-scaling", Input::Network(net)) => cs(net, true),
        (name, Input::Network(net)) => {
            let algo = FlowAlgorithm::ALL
                .into_iter()
                .find(|a| a.name() == name)
                .unwrap();
            let f = flow::solve(net, algo).unwrap();
            let extra = if algo == FlowAlgorithm::NetworkSimplex {
                let (_, _, s) = mcflow::simplex::solve_with_stats(net).unwrap();
                format!("{} pivots", s.pivots)
            } else {
                String::new()
            };
            (f.cost.to_string(), extra)
        }
        (name, Input::Dense(d)) => (
            solve_dense(d, name.parse().unwrap())
                .unwrap()
                .cost
                .to_string(),
            String::new(),
        ),
        (name, Input::Sparse(g)) => (
            solve_sparse(g, name.parse().unwrap())
                .unwrap()
                .cost
                .to_string(),
            String::new(),
        ),
    }
}

fn cs(net: &Network, updates: bool) -> (String, String) {
    let opts = Options {
        price_updates: updates,
        ..Options::default()
    };
    let (flow, phases) = cost_scaling::solve_with_stats(net, opts).unwrap();
    let cost: i128 = net
        .arcs()
        .iter()
        .zip(&flow)
        .map(|(a, &x)| x as i128 * a.cost as i128)
        .sum();
    let relabels: u64 = phases.iter().map(|p| p.relabels).sum();
    (cost.to_string(), format!("{relabels} relabels"))
}

fn ndarray_matrix(d: &Dense) -> ndarray::Array2<f64> {
    ndarray::Array2::from_shape_fn((d.rows(), d.cols()), |(i, j)| d.get(i, j) as f64)
}

fn cell(args: &[String]) {
    let (algo, family, n, seed) = (
        &args[0],
        &args[1],
        args[2].parse().unwrap(),
        args[3].parse().unwrap(),
    );
    let input = build(family, n, seed);
    let baseline = LIVE.load(Relaxed);
    PEAK.store(baseline, Relaxed);
    let mut times = Vec::new();
    let mut last = (String::new(), String::new());
    let started = Instant::now();
    for _ in 0..7 {
        let t = Instant::now();
        last = solve_cell(algo, &input);
        times.push(t.elapsed());
        if started.elapsed() > Duration::from_millis(400) {
            break;
        }
    }
    times.sort();
    let median = times[times.len() / 2];
    let peak = PEAK.load(Relaxed).saturating_sub(baseline);
    println!(
        "ms={:.3} mb={:.1} cost={} {}",
        median.as_secs_f64() * 1e3,
        peak as f64 / 1048576.0,
        last.0,
        last.1
    );
}

// ---------------------------------------------------------------------------------------------
// Parent: tables.
// ---------------------------------------------------------------------------------------------

struct Cell {
    ms: f64,
    mb: f64,
    cost: String,
    extra: String,
}

fn run_cell(algo: &str, family: &str, n: usize, seed: u64) -> Option<Cell> {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--cell", algo, family, &n.to_string(), &seed.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if start.elapsed() > TIMEOUT {
            child.kill().ok();
            child.wait().ok();
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().last()?;
    let get = |key: &str| {
        line.split_whitespace()
            .find_map(|t| t.strip_prefix(key))
            .map(str::to_string)
    };
    let extra = line.splitn(4, ' ').nth(3).unwrap_or("").to_string();
    Some(Cell {
        ms: get("ms=")?.parse().ok()?,
        mb: get("mb=")?.parse().ok()?,
        cost: get("cost=")?,
        extra,
    })
}

fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else if ms >= 10.0 {
        format!("{ms:.0} ms")
    } else {
        format!("{ms:.2} ms")
    }
}

/// One table: rows are algorithms, columns are sizes; each entry is `time / peak MB`.
fn table(
    title: &str,
    show_extra: bool,
    family: &str,
    sizes: &[usize],
    algos: &[&str],
    seed: u64,
    skip: impl Fn(&str, usize) -> bool,
) {
    println!("\n### {title}\n");
    print!("| algorithm |");
    for n in sizes {
        print!(" n={n} |");
    }
    println!();
    print!("| --- |");
    for _ in sizes {
        print!(" ---: |");
    }
    println!();
    let mut costs: Vec<Vec<Option<String>>> = vec![vec![None; sizes.len()]; algos.len()];
    for (a, algo) in algos.iter().enumerate() {
        print!("| `{algo}` |");
        for (s, &n) in sizes.iter().enumerate() {
            if skip(algo, n) {
                print!(" skip |");
                continue;
            }
            match run_cell(algo, family, n, seed) {
                Some(c) => {
                    let extra = if show_extra && !c.extra.is_empty() {
                        format!(" ({})", c.extra)
                    } else {
                        String::new()
                    };
                    print!(" {} / {:.0} MB{extra} |", fmt_ms(c.ms), c.mb.max(1.0));
                    costs[a][s] = Some(c.cost);
                }
                None => print!(" DNF |"),
            }
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        println!();
    }
    for (s, &n) in sizes.iter().enumerate() {
        let mut seen: Vec<&String> = costs.iter().filter_map(|row| row[s].as_ref()).collect();
        seen.dedup();
        if seen.windows(2).any(|w| w[0] != w[1]) {
            println!("\n**costs disagree at n={n}: {seen:?}**");
        }
    }
}

fn dense_skip(algo: &str, n: usize) -> bool {
    let flow_based = matches!(
        algo,
        "ssp" | "cost-scaling" | "network-simplex" | "cs-plain"
    );
    (flow_based && n * n > 1 << 21) || (algo == "rival-lapjv" && n > 3000)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--cell") {
        return cell(&args[1..]);
    }
    let which = args.first().map_or("all", String::as_str);
    let dense_algos = [
        "hungarian",
        "crouse",
        "jv",
        "lapmod",
        "auction",
        "ssp",
        "cost-scaling",
        "network-simplex",
        "rival-lapjv",
        "rival-pathfinding",
    ];
    if matches!(which, "dense" | "all") {
        println!("## Dense assignment, time / peak heap above the instance (median, seed 1)");
        let sizes = [100, 300, 1000, 3000];
        table(
            "Uniform costs 0..1000",
            false,
            "uniform",
            &sizes,
            &dense_algos,
            1,
            dense_skip,
        );
        table(
            "Narrow costs 0..10 (many ties)",
            false,
            "narrow",
            &sizes,
            &dense_algos,
            1,
            dense_skip,
        );
        table(
            "Geometric (Euclidean distances)",
            false,
            "geometric",
            &sizes,
            &dense_algos,
            1,
            dense_skip,
        );
        table(
            "Separable a_i + b_j + noise",
            false,
            "separable",
            &sizes,
            &dense_algos,
            1,
            dense_skip,
        );
        table(
            "Machol-Wien c_ij = i*j",
            false,
            "machol-wien",
            &sizes,
            &dense_algos,
            1,
            dense_skip,
        );
    }
    if matches!(which, "sparse" | "all") {
        println!(
            "\n## Sparse assignment (DIMACS low-cost shape: degree 2*ceil(log2 n), costs 0..100)"
        );
        let sizes = [2048, 8192, 32768];
        let skip =
            |algo: &str, n: usize| matches!(algo, "hungarian" | "crouse" | "jv") && n * n > 1 << 25;
        let algos = [
            "hungarian",
            "crouse",
            "lapmod",
            "auction",
            "ssp",
            "cost-scaling",
            "network-simplex",
        ];
        table(
            "Sparse planted-matching instances",
            false,
            "sparse",
            &sizes,
            &algos,
            1,
            skip,
        );
    }
    if matches!(which, "flow" | "all") {
        println!(
            "\n## General min-cost flow (random feasible networks, 5 arcs per node, capacities 1..100)"
        );
        let sizes = [1000, 5000, 20000, 50000];
        let algos = ["ssp", "cost-scaling", "network-simplex"];
        table(
            "Non-negative costs 0..1000",
            true,
            "flow",
            &sizes,
            &algos,
            1,
            |_, _| false,
        );
        table(
            "Costs -250..1000 (negative cycles present)",
            true,
            "flow-neg",
            &sizes,
            &algos,
            1,
            |_, _| false,
        );
    }
    if matches!(which, "tuning" | "all") {
        tuning();
    }
}

/// The two knobs worth a table: cost scaling with and without its price update, and the auction's
/// epsilon-scaling factor.
fn tuning() {
    println!("\n## Tuning");
    table(
        "Cost scaling, price update off (`cs-plain`) vs on (`cost-scaling`), uniform dense",
        true,
        "uniform",
        &[100, 300, 1000],
        &["cs-plain", "cost-scaling"],
        1,
        |_, _| false,
    );
    table(
        "Hungarian tie-break on narrow costs 0..10: first minimal column (`hungarian-first`) vs a free one (`hungarian`)",
        false,
        "narrow",
        &[300, 1000, 3000],
        &["hungarian-first", "hungarian"],
        1,
        |_, _| false,
    );
    println!("\n### Auction: epsilon-scaling factor (bids, time), uniform 1000 / sparse 8192\n");
    println!(
        "| factor | uniform 1000 bids | uniform 1000 time | sparse 8192 bids | sparse 8192 time |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: |");
    let dense = instances::uniform(1000, 1000, 1000, 1).to_sparse();
    let sparse = instances::dimacs_sparse(8192, 1);
    for factor in [2, 4, 7, 10, 16, 32] {
        let measure = |g: &Sparse| {
            let t = Instant::now();
            let bids = mcflow::auction::count_bids(g, factor).unwrap();
            (bids, t.elapsed().as_secs_f64() * 1e3)
        };
        let (b1, t1) = measure(&dense);
        let (b2, t2) = measure(&sparse);
        println!(
            "| {factor} | {b1} | {} | {b2} | {} |",
            fmt_ms(t1),
            fmt_ms(t2)
        );
    }
}
