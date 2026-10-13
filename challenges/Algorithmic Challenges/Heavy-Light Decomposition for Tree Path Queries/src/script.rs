//! A tiny text format for trees plus a query script, shared by the CLI and the tests.
//!
//! ```text
//! # vertex-weighted (default)            # edge-weighted (--edges)
//! 5                                      5
//! 3 1 4 1 5          <- n weights        0 1 7       <- n-1 lines "u v w"
//! 0 1                <- n-1 lines "u v"  0 2 2
//! 0 2                                    2 3 9
//! 2 3                                    2 4 1
//! 2 4
//! path 1 3           # then commands, one per line; blank lines and '#' comments are skipped
//! add 1 3 10
//! set 2 100          # vertex 2 (edge index 2 in edge mode)
//! get 2
//! sub 2              # subtree aggregate
//! lca 1 4
//! kth 3 1            # 1st ancestor of 3
//! ```

use crate::{BalancedHld, TopTree};
use crate::{ClassicHld, EdgeTree, Layout, MaybeSubtree, NaiveTree, Stats, Tree};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Naive,
    Classic,
    Balanced,
    TopTree,
}

impl Structure {
    pub const ALL: [Structure; 4] = [
        Structure::Naive,
        Structure::Classic,
        Structure::Balanced,
        Structure::TopTree,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Structure::Naive => "naive",
            Structure::Classic => "classic",
            Structure::Balanced => "balanced",
            Structure::TopTree => "toptree",
        }
    }

    pub fn from_name(name: &str) -> Option<Structure> {
        Structure::ALL.into_iter().find(|s| s.name() == name)
    }
}

#[derive(Clone, Copy, Debug)]
enum Command {
    Path(usize, usize),
    Add(usize, usize, i64),
    Set(usize, i64),
    Get(usize),
    Sub(usize),
    Lca(usize, usize),
    Kth(usize, usize),
}

struct Parsed {
    n: usize,
    weights: Vec<i64>,
    edges: Vec<(usize, usize, i64)>,
    commands: Vec<(usize, Command)>,
}

fn parse_num<T: std::str::FromStr>(
    token: Option<&str>,
    what: &str,
    line: usize,
) -> Result<T, String> {
    let token = token.ok_or_else(|| format!("line {line}: missing {what}"))?;
    token
        .parse()
        .map_err(|_| format!("line {line}: {what} must be a number, got '{token}'"))
}

fn parse(input: &str, edge_weighted: bool) -> Result<Parsed, String> {
    let mut lines = input
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.split('#').next().unwrap_or("").trim()))
        .filter(|(_, l)| !l.is_empty());
    let (line, first) = lines
        .next()
        .ok_or("empty input: the first line is the vertex count")?;
    let n: usize = parse_num(first.split_whitespace().next(), "vertex count", line)?;
    if n == 0 {
        return Err(format!("line {line}: a tree needs at least one vertex"));
    }
    let mut weights = Vec::new();
    if !edge_weighted {
        let (line, row) = lines.next().ok_or("missing the line of vertex weights")?;
        weights = row
            .split_whitespace()
            .map(|t| parse_num(Some(t), "vertex weight", line))
            .collect::<Result<_, _>>()?;
        if weights.len() != n {
            return Err(format!(
                "line {line}: expected {n} weights, got {}",
                weights.len()
            ));
        }
    }
    let mut edges = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let (line, row) = lines.next().ok_or("fewer edge lines than n - 1")?;
        let mut t = row.split_whitespace();
        let (a, b) = (
            parse_num(t.next(), "edge endpoint", line)?,
            parse_num(t.next(), "edge endpoint", line)?,
        );
        let w = if edge_weighted {
            parse_num(t.next(), "edge weight", line)?
        } else {
            0
        };
        edges.push((a, b, w));
    }
    let vertex = |token: Option<&str>, line: usize| -> Result<usize, String> {
        let v: usize = parse_num(token, "vertex", line)?;
        if v >= n {
            Err(format!("line {line}: vertex {v} is out of range 0..{n}"))
        } else {
            Ok(v)
        }
    };
    let slots = if edge_weighted { n - 1 } else { n };
    let mut commands = Vec::new();
    for (line, row) in lines {
        let mut t = row.split_whitespace();
        let name = t.next().unwrap_or_default();
        let command = match name {
            "path" => Command::Path(vertex(t.next(), line)?, vertex(t.next(), line)?),
            "add" => Command::Add(
                vertex(t.next(), line)?,
                vertex(t.next(), line)?,
                parse_num(t.next(), "delta", line)?,
            ),
            "set" | "get" => {
                let id: usize = parse_num(t.next(), "index", line)?;
                if id >= slots {
                    return Err(format!(
                        "line {line}: index {id} is out of range 0..{slots}"
                    ));
                }
                if name == "set" {
                    Command::Set(id, parse_num(t.next(), "value", line)?)
                } else {
                    Command::Get(id)
                }
            }
            "sub" => Command::Sub(vertex(t.next(), line)?),
            "lca" => Command::Lca(vertex(t.next(), line)?, vertex(t.next(), line)?),
            "kth" => Command::Kth(vertex(t.next(), line)?, parse_num(t.next(), "k", line)?),
            other => return Err(format!("line {line}: unknown command '{other}'")),
        };
        if t.next().is_some() {
            return Err(format!("line {line}: unexpected extra tokens"));
        }
        commands.push((line, command));
    }
    Ok(Parsed {
        n,
        weights,
        edges,
        commands,
    })
}

/// One interface over a vertex-weighted structure and an [`EdgeTree`].
trait Uniform {
    fn path(&mut self, u: usize, v: usize) -> Stats;
    fn add(&mut self, u: usize, v: usize, delta: i64);
    fn set(&mut self, id: usize, value: i64);
    fn get(&mut self, id: usize) -> Option<i64>;
    fn sub(&mut self, v: usize) -> Option<Stats>;
    fn lca(&mut self, u: usize, v: usize) -> usize;
    fn kth(&mut self, v: usize, k: usize) -> Option<usize>;
}

struct VertexMode<S>(S);

impl<S: MaybeSubtree> Uniform for VertexMode<S> {
    fn path(&mut self, u: usize, v: usize) -> Stats {
        self.0.path_query(u, v)
    }
    fn add(&mut self, u: usize, v: usize, delta: i64) {
        self.0.path_add(u, v, delta);
    }
    fn set(&mut self, id: usize, value: i64) {
        self.0.set(id, value);
    }
    fn get(&mut self, id: usize) -> Option<i64> {
        self.0.get(id)
    }
    fn sub(&mut self, v: usize) -> Option<Stats> {
        self.0.maybe_subtree(v)
    }
    fn lca(&mut self, u: usize, v: usize) -> usize {
        self.0.lca(u, v)
    }
    fn kth(&mut self, v: usize, k: usize) -> Option<usize> {
        self.0.kth_ancestor(v, k)
    }
}

impl<S: MaybeSubtree> Uniform for EdgeTree<S> {
    fn path(&mut self, u: usize, v: usize) -> Stats {
        self.path_query(u, v)
    }
    fn add(&mut self, u: usize, v: usize, delta: i64) {
        self.path_add(u, v, delta);
    }
    fn set(&mut self, id: usize, value: i64) {
        self.set_edge(id, value);
    }
    fn get(&mut self, id: usize) -> Option<i64> {
        self.edge_weight(id)
    }
    fn sub(&mut self, v: usize) -> Option<Stats> {
        self.maybe_subtree(v)
    }
    fn lca(&mut self, u: usize, v: usize) -> usize {
        EdgeTree::lca(self, u, v)
    }
    fn kth(&mut self, v: usize, k: usize) -> Option<usize> {
        self.kth_ancestor(v, k)
    }
}

fn execute<U: Uniform>(
    engine: &mut U,
    commands: &[(usize, Command)],
    structure: Structure,
) -> Result<Vec<String>, String> {
    let show = |o: Option<usize>| o.map_or("none".to_string(), |x| x.to_string());
    let mut out = Vec::new();
    for &(line, command) in commands {
        match command {
            Command::Path(u, v) => out.push(engine.path(u, v).to_string()),
            Command::Add(u, v, d) => engine.add(u, v, d),
            Command::Set(id, w) => engine.set(id, w),
            Command::Get(id) => {
                out.push(engine.get(id).map_or("none".to_string(), |x| x.to_string()))
            }
            Command::Sub(v) => out.push(
                engine
                    .sub(v)
                    .ok_or_else(|| {
                        format!(
                            "line {line}: the {} structure has no subtree queries",
                            structure.name()
                        )
                    })?
                    .to_string(),
            ),
            Command::Lca(u, v) => out.push(engine.lca(u, v).to_string()),
            Command::Kth(v, k) => out.push(show(engine.kth(v, k))),
        }
    }
    Ok(out)
}

fn run_with<S: MaybeSubtree>(
    parsed: &Parsed,
    edge_weighted: bool,
    root: usize,
    structure: Structure,
) -> Result<Vec<String>, String> {
    if edge_weighted {
        let mut engine =
            EdgeTree::<S>::new(parsed.n, &parsed.edges, root).map_err(|e| e.to_string())?;
        execute(&mut engine, &parsed.commands, structure)
    } else {
        let pairs: Vec<(usize, usize)> = parsed.edges.iter().map(|&(a, b, _)| (a, b)).collect();
        let tree = Tree::new(parsed.n, &pairs).map_err(|e| e.to_string())?;
        let layout = Layout::new(&tree, root).map_err(|e| e.to_string())?;
        let mut engine = VertexMode(S::from_weights(layout, &parsed.weights));
        execute(&mut engine, &parsed.commands, structure)
    }
}

/// Parse `input` and run every command, returning one line per query. Errors carry line numbers.
pub fn run_script(
    input: &str,
    structure: Structure,
    edge_weighted: bool,
    root: usize,
) -> Result<Vec<String>, String> {
    let parsed = parse(input, edge_weighted)?;
    match structure {
        Structure::Naive => run_with::<NaiveTree>(&parsed, edge_weighted, root, structure),
        Structure::Classic => run_with::<ClassicHld>(&parsed, edge_weighted, root, structure),
        Structure::Balanced => run_with::<BalancedHld>(&parsed, edge_weighted, root, structure),
        Structure::TopTree => run_with::<TopTree>(&parsed, edge_weighted, root, structure),
    }
}
