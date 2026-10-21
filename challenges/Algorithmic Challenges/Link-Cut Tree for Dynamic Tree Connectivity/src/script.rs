//! A tiny text language for driving any structure from the command line, and generators for it.
//!
//! A script is a vertex count `n` on the first line, then one command per line (`#` starts a
//! comment). Vertices are `0..n`; every vertex starts with the value `0`.
//!
//! ```text
//! link u v      add edge u - v                      cut u v      remove edge u - v
//! conn u v      are u and v in one tree?
//! set v x       give v the value x                  get v        read it back
//! path u v      count/sum/min/max over the path     add u v d    add d along the path
//! sub p v       count/sum of v's side of edge p-v   comp v       count/sum of v's whole tree
//! ```
//!
//! A structure that cannot answer a command says so (`path` on an Euler tour tree, `sub` on a
//! path-only link-cut tree) instead of guessing.
//!
//! The offline connectivity format has `add u v`, `del u v` and `conn u v` lines; see
//! [`parse_events`].

use std::fmt::Write as _;

use crate::connectivity::Event;
use crate::shapes::{Rng, Shape, generate};
use crate::{
    EulerTourTree, Forest, NaiveForest, PathLct, PathOps, Stats, SubtreeLct, SubtreeOps, Sum,
    UfoTree, Weighted,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Naive,
    Lct,
    SubtreeLct,
    Ett,
    Ufo,
}

impl Structure {
    pub const ALL: [Structure; 5] = [
        Structure::Naive,
        Structure::Lct,
        Structure::SubtreeLct,
        Structure::Ett,
        Structure::Ufo,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Structure::Naive => "naive",
            Structure::Lct => "lct",
            Structure::SubtreeLct => "subtree-lct",
            Structure::Ett => "ett",
            Structure::Ufo => "ufo",
        }
    }

    pub fn from_name(name: &str) -> Option<Structure> {
        Structure::ALL.into_iter().find(|s| s.name() == name)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScriptError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ScriptError {}

fn err<T>(line: usize, message: impl Into<String>) -> Result<T, ScriptError> {
    Err(ScriptError {
        line,
        message: message.into(),
    })
}

/// One structure behind a uniform interface; `None` means "this structure cannot do that".
trait Engine {
    fn link(&mut self, u: usize, v: usize) -> bool;
    fn cut(&mut self, u: usize, v: usize) -> bool;
    fn connected(&mut self, u: usize, v: usize) -> bool;
    fn set(&mut self, _: usize, _: i64) -> Option<()> {
        None
    }
    fn get(&mut self, _: usize) -> Option<Option<i64>> {
        None
    }
    fn path(&mut self, _: usize, _: usize) -> Option<Option<Stats>> {
        None
    }
    fn path_add(&mut self, _: usize, _: usize, _: i64) -> Option<bool> {
        None
    }
    fn sub(&mut self, _: usize, _: usize) -> Option<Option<Sum>> {
        None
    }
    fn comp(&mut self, _: usize) -> Option<Sum> {
        None
    }
}

macro_rules! forest_engine {
    () => {
        fn link(&mut self, u: usize, v: usize) -> bool {
            Forest::link(self, u, v)
        }
        fn cut(&mut self, u: usize, v: usize) -> bool {
            Forest::cut(self, u, v)
        }
        fn connected(&mut self, u: usize, v: usize) -> bool {
            Forest::connected(self, u, v)
        }
    };
}

macro_rules! weighted_engine {
    () => {
        fn set(&mut self, v: usize, x: i64) -> Option<()> {
            Weighted::set(self, v, x);
            Some(())
        }
        fn get(&mut self, v: usize) -> Option<Option<i64>> {
            Some(Weighted::get(self, v))
        }
    };
}

macro_rules! path_engine {
    () => {
        fn path(&mut self, u: usize, v: usize) -> Option<Option<Stats>> {
            Some(PathOps::path_query(self, u, v))
        }
        fn path_add(&mut self, u: usize, v: usize, d: i64) -> Option<bool> {
            Some(PathOps::path_add(self, u, v, d))
        }
    };
}

macro_rules! subtree_engine {
    () => {
        fn sub(&mut self, p: usize, v: usize) -> Option<Option<Sum>> {
            Some(SubtreeOps::subtree_query(self, p, v))
        }
        fn comp(&mut self, v: usize) -> Option<Sum> {
            Some(SubtreeOps::component(self, v))
        }
    };
}

impl Engine for NaiveForest {
    forest_engine!();
    weighted_engine!();
    path_engine!();
    subtree_engine!();
}

impl Engine for PathLct {
    forest_engine!();
    weighted_engine!();
    path_engine!();
}

impl Engine for SubtreeLct {
    forest_engine!();
    weighted_engine!();
    path_engine!();
    subtree_engine!();
}

impl Engine for EulerTourTree {
    forest_engine!();
    weighted_engine!();
    subtree_engine!();
}

impl Engine for UfoTree {
    forest_engine!();
}

fn make_engine(structure: Structure, n: usize) -> Box<dyn Engine> {
    match structure {
        Structure::Naive => Box::new(NaiveForest::new(n)),
        Structure::Lct => Box::new(PathLct::new(n)),
        Structure::SubtreeLct => Box::new(SubtreeLct::new(n)),
        Structure::Ett => Box::new(EulerTourTree::new(n)),
        Structure::Ufo => Box::new(UfoTree::new(n)),
    }
}

type Lines<'a> = Box<dyn Iterator<Item = (usize, &'a str)> + 'a>;

fn parse_n(text: &str) -> Result<(usize, Lines<'_>), ScriptError> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.split('#').next().unwrap_or("").trim()))
        .filter(|(_, l)| !l.is_empty());
    let Some((line, first)) = lines.next() else {
        return err(1, "empty script: the first line must be the vertex count");
    };
    match first.parse::<usize>() {
        Ok(n) if n >= 1 => Ok((n, Box::new(lines))),
        _ => err(line, "the first line must be a vertex count of at least 1"),
    }
}

fn number<T: std::str::FromStr>(
    line: usize,
    word: Option<&str>,
    what: &str,
) -> Result<T, ScriptError> {
    match word.map(str::parse::<T>) {
        Some(Ok(x)) => Ok(x),
        Some(Err(_)) => err(line, format!("{what} is not a number")),
        None => err(line, format!("missing {what}")),
    }
}

fn vertex(line: usize, word: Option<&str>, n: usize) -> Result<usize, ScriptError> {
    let v: usize = number(line, word, "vertex")?;
    if v >= n {
        return err(line, format!("vertex {v} is out of range 0..{n}"));
    }
    Ok(v)
}

fn ok_or_rejected(name: &str, done: bool) -> String {
    format!("{name}: {}", if done { "ok" } else { "rejected" })
}

/// Run a script against `structure` and return one output line per command.
pub fn run_script(text: &str, structure: Structure) -> Result<Vec<String>, ScriptError> {
    let (n, lines) = parse_n(text)?;
    let mut engine = make_engine(structure, n);
    let mut out = Vec::new();
    for (line, l) in lines {
        let mut words = l.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let unsupported = || format!("{cmd}: not supported by {}", structure.name());
        let reply = match cmd {
            "link" | "cut" | "conn" | "path" => {
                let u = vertex(line, words.next(), n)?;
                let v = vertex(line, words.next(), n)?;
                match cmd {
                    "link" => ok_or_rejected(cmd, engine.link(u, v)),
                    "cut" => ok_or_rejected(cmd, engine.cut(u, v)),
                    "conn" => format!("conn: {}", engine.connected(u, v)),
                    _ => match engine.path(u, v) {
                        None => unsupported(),
                        Some(None) => "path: disconnected".to_string(),
                        Some(Some(s)) => format!("path: {s}"),
                    },
                }
            }
            "add" => {
                let u = vertex(line, words.next(), n)?;
                let v = vertex(line, words.next(), n)?;
                let d: i64 = number(line, words.next(), "delta")?;
                match engine.path_add(u, v, d) {
                    None => unsupported(),
                    Some(true) => "add: ok".to_string(),
                    Some(false) => "add: disconnected".to_string(),
                }
            }
            "set" => {
                let v = vertex(line, words.next(), n)?;
                let x: i64 = number(line, words.next(), "value")?;
                match engine.set(v, x) {
                    None => unsupported(),
                    Some(()) => "set: ok".to_string(),
                }
            }
            "get" => {
                let v = vertex(line, words.next(), n)?;
                match engine.get(v) {
                    None => unsupported(),
                    Some(None) => "get: neutral".to_string(),
                    Some(Some(x)) => format!("get: {x}"),
                }
            }
            "sub" => {
                let p = vertex(line, words.next(), n)?;
                let v = vertex(line, words.next(), n)?;
                match engine.sub(p, v) {
                    None => unsupported(),
                    Some(None) => "sub: not an edge".to_string(),
                    Some(Some(s)) => format!("sub: {s}"),
                }
            }
            "comp" => {
                let v = vertex(line, words.next(), n)?;
                match engine.comp(v) {
                    None => unsupported(),
                    Some(s) => format!("comp: {s}"),
                }
            }
            other => return err(line, format!("unknown command {other:?}")),
        };
        out.push(reply);
    }
    Ok(out)
}

/// Parse the offline connectivity format: `n`, then `add u v`, `del u v` and `conn u v` lines.
pub fn parse_events(text: &str) -> Result<(usize, Vec<Event>), ScriptError> {
    let (n, lines) = parse_n(text)?;
    let mut events = Vec::new();
    for (line, l) in lines {
        let mut words = l.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let u: usize = number(line, words.next(), "vertex")?;
        let v: usize = number(line, words.next(), "vertex")?;
        events.push(match cmd {
            "add" => Event::Add(u, v),
            "del" => Event::Remove(u, v),
            "conn" => Event::Query(u, v),
            other => return err(line, format!("unknown command {other:?}")),
        });
    }
    Ok((n, events))
}

/// Render events in the format [`parse_events`] reads.
pub fn render_events(n: usize, events: &[Event]) -> String {
    let mut out = format!("{n}\n");
    for &e in events {
        let _ = match e {
            Event::Add(u, v) => writeln!(out, "add {u} {v}"),
            Event::Remove(u, v) => writeln!(out, "del {u} {v}"),
            Event::Query(u, v) => writeln!(out, "conn {u} {v}"),
        };
    }
    out
}

/// A random valid script: build a tree of `shape`, then `ops` mixed commands that keep it a
/// forest (every cut is followed by a re-link elsewhere).
pub fn generate_script(shape: Shape, n: usize, ops: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ 0xDEAD_BEEF);
    let mut forest = SubtreeLct::new(n);
    let mut out = format!("{n}\n");
    let mut current = generate(shape, n, seed);
    for &(u, v) in &current {
        Forest::link(&mut forest, u, v);
        let _ = writeln!(out, "link {u} {v}");
    }
    for v in 0..n {
        let _ = writeln!(out, "set {v} {}", rng.range(-1000, 1000));
    }
    let mut emitted = 0;
    while emitted < ops {
        let (u, v) = (rng.below(n), rng.below(n));
        match rng.below(8) {
            0 | 1 => {
                let _ = writeln!(out, "path {u} {v}");
            }
            2 => {
                let _ = writeln!(out, "add {u} {v} {}", rng.range(-10, 11));
            }
            3 => {
                let _ = writeln!(out, "conn {u} {v}");
            }
            4 if !current.is_empty() => {
                let (a, b) = current[rng.below(current.len())];
                let _ = writeln!(out, "sub {a} {b}");
            }
            5 => {
                let _ = writeln!(out, "comp {u}");
            }
            _ if !current.is_empty() => {
                let k = rng.below(current.len());
                let (a, b) = current[k];
                Forest::cut(&mut forest, a, b);
                let _ = writeln!(out, "cut {a} {b}");
                let mut target = b;
                for _ in 0..8 {
                    let x = rng.below(n);
                    if !Forest::connected(&mut forest, a, x) {
                        target = x;
                        break;
                    }
                }
                Forest::link(&mut forest, a, target);
                current[k] = (a, target);
                emitted += 1;
                let _ = writeln!(out, "link {a} {target}");
            }
            _ => continue,
        }
        emitted += 1;
    }
    out
}
