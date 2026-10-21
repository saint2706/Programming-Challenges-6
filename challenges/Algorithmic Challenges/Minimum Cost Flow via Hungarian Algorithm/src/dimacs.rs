//! File formats: DIMACS min-cost flow (`p min`), DIMACS assignment (`p asn`), and a plain cost
//! matrix.
//!
//! * `p min N M`, `n ID SUPPLY`, `a FROM TO LOW CAP COST` (1-based ids). Only `LOW = 0` is
//!   supported; the lower-bound transformation is not implemented.
//! * `p asn N M`, `n ID` for each row node, `a ROW COLUMN COST`; the nodes not listed are columns.
//!   Rows and columns are numbered in the order of their ids.
//! * Matrix: one row per line, entries separated by commas or spaces; `x`, `inf` or an empty field
//!   is a forbidden edge.

use std::fmt::Write as _;

use crate::error::{Error, Result};
use crate::flow::Network;
use crate::matrix::{Cost, Dense, FORBIDDEN, Sparse};

fn bad(line: usize, msg: &str) -> Error {
    Error::Invalid(format!("line {line}: {msg}"))
}

fn num<T: std::str::FromStr>(tok: Option<&str>, line: usize, what: &str) -> Result<T> {
    tok.ok_or_else(|| bad(line, &format!("missing {what}")))?
        .parse()
        .map_err(|_| bad(line, &format!("{what} is not a number")))
}

pub fn parse_min(text: &str) -> Result<Network> {
    let mut net: Option<Network> = None;
    for (k, raw) in text.lines().enumerate() {
        let line = k + 1;
        let mut t = raw.split_whitespace();
        match t.next() {
            None | Some("c") => {}
            Some("p") => {
                if t.next() != Some("min") {
                    return Err(bad(line, "expected `p min N M`"));
                }
                let n: usize = num(t.next(), line, "node count")?;
                if n > crate::matrix::MAX_DIM {
                    return Err(Error::OutOfRange(format!("{n} nodes")));
                }
                net = Some(Network::new(n));
            }
            Some("n") => {
                let net = net.as_mut().ok_or_else(|| bad(line, "`n` before `p`"))?;
                let id: usize = num(t.next(), line, "node id")?;
                let b: i64 = num(t.next(), line, "supply")?;
                net.set_supply(
                    id.checked_sub(1)
                        .ok_or_else(|| bad(line, "ids start at 1"))?,
                    b,
                )?;
            }
            Some("a") => {
                let net = net.as_mut().ok_or_else(|| bad(line, "`a` before `p`"))?;
                let from: usize = num(t.next(), line, "tail")?;
                let to: usize = num(t.next(), line, "head")?;
                let low: i64 = num(t.next(), line, "lower bound")?;
                let cap: i64 = num(t.next(), line, "capacity")?;
                let cost: i64 = num(t.next(), line, "cost")?;
                if low != 0 {
                    return Err(Error::Unsupported("arc lower bounds other than 0"));
                }
                if from == 0 || to == 0 {
                    return Err(bad(line, "ids start at 1"));
                }
                net.add_arc(from - 1, to - 1, cap, cost)?;
            }
            Some(other) => return Err(bad(line, &format!("unknown line type {other:?}"))),
        }
    }
    net.ok_or_else(|| Error::Invalid("no `p min` line".into()))
}

pub fn write_min(net: &Network) -> String {
    let mut s = format!("p min {} {}\n", net.nodes(), net.arcs().len());
    for (v, &b) in net.supplies().iter().enumerate() {
        if b != 0 {
            let _ = writeln!(s, "n {} {b}", v + 1);
        }
    }
    for a in net.arcs() {
        let _ = writeln!(s, "a {} {} 0 {} {}", a.from + 1, a.to + 1, a.cap, a.cost);
    }
    s
}

pub fn parse_asn(text: &str) -> Result<Sparse> {
    let mut nodes = 0usize;
    let mut is_row: Vec<bool> = Vec::new();
    let mut arcs: Vec<(usize, usize, Cost)> = Vec::new();
    for (k, raw) in text.lines().enumerate() {
        let line = k + 1;
        let mut t = raw.split_whitespace();
        match t.next() {
            None | Some("c") => {}
            Some("p") => {
                if t.next() != Some("asn") {
                    return Err(bad(line, "expected `p asn N M`"));
                }
                nodes = num(t.next(), line, "node count")?;
                if nodes > 2 * crate::matrix::MAX_DIM {
                    return Err(Error::OutOfRange(format!("{nodes} nodes")));
                }
                is_row = vec![false; nodes];
            }
            Some("n") => {
                let id: usize = num(t.next(), line, "node id")?;
                *is_row
                    .get_mut(id.wrapping_sub(1))
                    .ok_or_else(|| bad(line, "node id out of range"))? = true;
            }
            Some("a") => {
                let u: usize = num(t.next(), line, "tail")?;
                let v: usize = num(t.next(), line, "head")?;
                let c: Cost = num(t.next(), line, "cost")?;
                if u == 0 || v == 0 || u > nodes || v > nodes {
                    return Err(bad(line, "node id out of range"));
                }
                arcs.push((u - 1, v - 1, c));
            }
            Some(other) => return Err(bad(line, &format!("unknown line type {other:?}"))),
        }
    }
    if is_row.is_empty() {
        return Err(Error::Invalid("no `p asn` line".into()));
    }
    let mut index = vec![0usize; nodes];
    let (mut rows, mut cols) = (0, 0);
    for (v, &r) in is_row.iter().enumerate() {
        if r {
            index[v] = rows;
            rows += 1;
        } else {
            index[v] = cols;
            cols += 1;
        }
    }
    let mut triplets = Vec::with_capacity(arcs.len());
    for (u, v, c) in arcs {
        match (is_row[u], is_row[v]) {
            (true, false) => triplets.push((index[u], index[v], c)),
            (false, true) => triplets.push((index[v], index[u], c)),
            _ => {
                return Err(Error::Invalid(format!(
                    "arc {} {} does not join a row to a column",
                    u + 1,
                    v + 1
                )));
            }
        }
    }
    Sparse::from_triplets(rows, cols, triplets)
}

pub fn write_asn(g: &Sparse) -> String {
    let (n, m) = (g.rows(), g.cols());
    let mut s = format!("p asn {} {}\n", n + m, g.nnz());
    for i in 0..n {
        let _ = writeln!(s, "n {}", i + 1);
    }
    for i in 0..n {
        for (j, c) in g.row(i) {
            let _ = writeln!(s, "a {} {} {c}", i + 1, n + j + 1);
        }
    }
    s
}

pub fn parse_matrix(text: &str) -> Result<Dense> {
    let mut rows: Vec<Vec<Cost>> = Vec::new();
    for (k, raw) in text.lines().enumerate() {
        let raw = raw.trim();
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        // Commas separate fields (an empty field is forbidden); without commas, whitespace does.
        let fields: Vec<&str> = if raw.contains(',') {
            raw.split(',').map(str::trim).collect()
        } else {
            raw.split_whitespace().collect()
        };
        let row = fields
            .into_iter()
            .map(|f| match f {
                "" | "x" | "X" | "inf" | "INF" => Ok(FORBIDDEN),
                v => v
                    .parse()
                    .map_err(|_| bad(k + 1, &format!("{v:?} is not a cost"))),
            })
            .collect::<Result<Vec<Cost>>>()?;
        rows.push(row);
    }
    Dense::from_rows(&rows)
}

pub fn write_matrix(d: &Dense) -> String {
    let mut s = String::new();
    for i in 0..d.rows() {
        let line: Vec<String> = d
            .row(i)
            .iter()
            .map(|&c| {
                if c == FORBIDDEN {
                    "x".into()
                } else {
                    c.to_string()
                }
            })
            .collect();
        s.push_str(&line.join(","));
        s.push('\n');
    }
    s
}
