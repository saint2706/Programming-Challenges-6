//! From a parsed `Select` to a plan: resolve columns to tables, push projections and predicates
//! into the scans, prune row groups with zone maps, and order the pipeline.
//!
//! The pipeline shape is fixed: scan the FROM table row group by row group, probe one hash table
//! per JOIN (each built from its own pruned, filtered scan), apply the residual filter, then
//! aggregate or project, sort, limit.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::Result;
use crate::format::{RowGroupMeta, Table};
use crate::sql::{BinOp, Expr, Select};
use crate::types::Value;

#[derive(Debug)]
pub struct Scan {
    pub table: String,
    pub path: PathBuf,
    /// Columns read from disk (projection pushdown).
    pub columns: Vec<String>,
    /// Conjuncts that mention only this table, evaluated right after the scan.
    pub filters: Vec<Expr>,
    /// Row groups that survive zone-map pruning.
    pub row_groups: Vec<usize>,
    pub total_row_groups: usize,
}

#[derive(Debug)]
pub struct Plan {
    pub query: Select,
    /// `scans[0]` is the FROM table; `scans[i]` for i > 0 is joined on `join_keys[i - 1]`.
    pub scans: Vec<Scan>,
    /// (probe-side column, build-side column)
    pub join_keys: Vec<(String, String)>,
    /// Conjuncts that span tables, evaluated after the joins.
    pub residual: Vec<Expr>,
}

pub fn conjuncts(e: Expr, out: &mut Vec<Expr>) {
    match e {
        Expr::Bin(a, BinOp::And, b) => {
            conjuncts(*a, out);
            conjuncts(*b, out);
        }
        e => out.push(e),
    }
}

/// `col op literal` (or `literal op col`), the only predicate shape a zone map can answer.
fn zone_predicate(e: &Expr) -> Option<(&str, BinOp, &Value)> {
    match e {
        Expr::Bin(a, op, b) if !matches!(op, BinOp::And | BinOp::Or) => match (&**a, &**b) {
            (Expr::Col(c), Expr::Lit(v)) => Some((c, *op, v)),
            (Expr::Lit(v), Expr::Col(c)) => Some((c, op.flip(), v)),
            _ => None,
        },
        _ => None,
    }
}

/// Can any row of a chunk with these min/max satisfy `col op v`?
pub fn may_match(min: &Value, max: &Value, op: BinOp, v: &Value) -> bool {
    use std::cmp::Ordering::*;
    if (min.dtype() == crate::DataType::Str) != (v.dtype() == crate::DataType::Str) {
        return true; // type mismatch: let the executor report it
    }
    let (lo, hi) = (min.total_cmp(v), max.total_cmp(v));
    match op {
        BinOp::Eq => lo != Greater && hi != Less,
        BinOp::Ne => !(lo == Equal && hi == Equal),
        BinOp::Lt => lo == Less,
        BinOp::Le => lo != Greater,
        BinOp::Gt => hi == Greater,
        BinOp::Ge => hi != Less,
        _ => true,
    }
}

fn prune(groups: &[RowGroupMeta], names: &[String], filters: &[Expr]) -> Vec<usize> {
    let preds: Vec<_> = filters
        .iter()
        .filter_map(zone_predicate)
        .filter_map(|(c, op, v)| names.iter().position(|n| n == c).map(|i| (i, op, v)))
        .collect();
    (0..groups.len())
        .filter(|&g| {
            preds.iter().all(|&(c, op, v)| {
                let m = &groups[g].chunks[c];
                may_match(&m.min, &m.max, op, v)
            })
        })
        .collect()
}

pub fn build(mut q: Select, catalog: &HashMap<String, PathBuf>) -> Result<Plan> {
    let tables: Vec<String> = std::iter::once(q.from.clone())
        .chain(q.joins.iter().map(|j| j.0.clone()))
        .collect();
    let mut opened = vec![];
    let mut owner: HashMap<String, usize> = HashMap::new();
    for (i, t) in tables.iter().enumerate() {
        let path = catalog
            .get(t)
            .ok_or_else(|| format!("unknown table {t:?}"))?
            .clone();
        let tab = Table::open(&path)?;
        for n in &tab.footer.names {
            if let Some(prev) = owner.insert(n.clone(), i) {
                return Err(format!(
                    "column {n:?} is in both {} and {t}; joined tables need distinct column names",
                    tables[prev]
                ));
            }
        }
        opened.push((path, tab));
    }
    // `t.col` must name the table that owns `col`; then the qualifier is dropped.
    let mut unqualify = |c: &mut String| -> Result<()> {
        if let Some((t, col)) = c.split_once('.') {
            let ti = tables
                .iter()
                .position(|x| x == t)
                .ok_or_else(|| format!("unknown table {t:?} in {c}"))?;
            match owner.get(col) {
                Some(&o) if o == ti => *c = col.to_string(),
                Some(&o) => {
                    return Err(format!("column {col:?} belongs to {}, not {t}", tables[o]));
                }
                None => return Err(format!("unknown column {c:?}")),
            }
        }
        Ok(())
    };
    if let Some(items) = &mut q.items {
        for (e, _) in items {
            e.map_columns(&mut unqualify)?;
        }
    }
    if let Some(f) = &mut q.filter {
        f.map_columns(&mut unqualify)?;
    }
    for e in q
        .group_by
        .iter_mut()
        .chain(q.order_by.iter_mut().map(|(e, _)| e))
    {
        e.map_columns(&mut unqualify)?;
    }
    for (_, a, b) in &mut q.joins {
        unqualify(a)?;
        unqualify(b)?;
    }
    let table_of = |c: &str| {
        owner
            .get(c)
            .copied()
            .ok_or_else(|| format!("unknown column {c:?}"))
    };

    // Every column the query touches, and which table it lives in.
    let mut used = vec![];
    match &q.items {
        Some(items) => items.iter().for_each(|(e, _)| e.columns(&mut used)),
        None => {
            for (_, t) in &opened {
                used.extend(t.footer.names.iter().cloned());
            }
        }
    }
    let aliases: Vec<&str> = q.items.iter().flatten().map(|(_, a)| a.as_str()).collect();
    for e in q.filter.iter().chain(&q.group_by) {
        e.columns(&mut used);
    }
    for (e, _) in &q.order_by {
        let mut cols = vec![];
        e.columns(&mut cols);
        for c in cols {
            if !aliases.contains(&c.as_str()) && !used.contains(&c) {
                used.push(c);
            }
        }
    }
    let mut join_keys = vec![];
    for (i, (t, a, b)) in q.joins.iter().enumerate() {
        let (ta, tb) = (table_of(a)?, table_of(b)?);
        let key = match (ta, tb) {
            (x, y) if y == i + 1 && x <= i => (a.clone(), b.clone()),
            (x, y) if x == i + 1 && y <= i => (b.clone(), a.clone()),
            _ => {
                return Err(format!(
                    "JOIN {t} ON {a} = {b}: one side must be a column of {t}, the other of an earlier table"
                ));
            }
        };
        for c in [&key.0, &key.1] {
            if !used.contains(c) {
                used.push(c.clone());
            }
        }
        join_keys.push(key);
    }
    for c in &used {
        table_of(c)?;
    }

    // Split WHERE into per-table conjuncts (pushed into scans) and residual ones.
    let mut parts = vec![];
    if let Some(f) = q.filter.clone() {
        conjuncts(f, &mut parts);
    }
    let mut pushed: Vec<Vec<Expr>> = vec![vec![]; tables.len()];
    let mut residual = vec![];
    for p in parts {
        let mut cols = vec![];
        p.columns(&mut cols);
        let owners: Vec<usize> = cols.iter().map(|c| table_of(c)).collect::<Result<_>>()?;
        match owners.first() {
            Some(&t) if owners.iter().all(|&o| o == t) => pushed[t].push(p),
            None => pushed[0].push(p), // constant predicate
            _ => residual.push(p),
        }
    }

    let mut scans = vec![];
    for (i, ((path, tab), filters)) in opened.into_iter().zip(pushed).enumerate() {
        let mut columns: Vec<String> = tab
            .footer
            .names
            .iter()
            .filter(|n| used.contains(n))
            .cloned()
            .collect();
        if columns.is_empty() {
            columns.push(tab.footer.names[0].clone()); // COUNT(*) still needs row counts
        }
        let row_groups = prune(&tab.footer.row_groups, &tab.footer.names, &filters);
        scans.push(Scan {
            table: tables[i].clone(),
            path,
            columns,
            filters,
            row_groups,
            total_row_groups: tab.footer.row_groups.len(),
        });
    }
    Ok(Plan {
        query: q,
        scans,
        join_keys,
        residual,
    })
}

fn list(v: &[Expr]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" AND ")
}

impl Plan {
    pub fn explain(&self) -> String {
        let q = &self.query;
        let mut lines = vec![];
        if let Some(n) = q.limit {
            lines.push(format!("Limit {n}"));
        }
        if !q.order_by.is_empty() {
            let keys: Vec<_> = q
                .order_by
                .iter()
                .map(|(e, d)| format!("{e}{}", if *d { " DESC" } else { "" }))
                .collect();
            lines.push(format!("Sort [{}]", keys.join(", ")));
        }
        let items = match &q.items {
            Some(i) => i
                .iter()
                .map(|(e, a)| format!("{e} AS {a}"))
                .collect::<Vec<_>>()
                .join(", "),
            None => "*".into(),
        };
        let agg = !q.group_by.is_empty() || q.items.iter().flatten().any(|(e, _)| e.has_agg());
        if agg {
            let g: Vec<_> = q.group_by.iter().map(ToString::to_string).collect();
            lines.push(format!("HashAggregate group=[{}] -> {items}", g.join(", ")));
        } else {
            lines.push(format!("Project {items}"));
        }
        if !self.residual.is_empty() {
            lines.push(format!("Filter {}", list(&self.residual)));
        }
        let mut out: Vec<String> = lines
            .iter()
            .enumerate()
            .map(|(d, l)| format!("{}{l}", "  ".repeat(d)))
            .collect();
        let depth = lines.len();
        let scan_line = |s: &Scan| {
            let f = if s.filters.is_empty() {
                String::new()
            } else {
                format!(" filter={}", list(&s.filters))
            };
            format!(
                "Scan {} columns=[{}]{f} row_groups={}/{}",
                s.table,
                s.columns.join(", "),
                s.row_groups.len(),
                s.total_row_groups
            )
        };
        for (i, (probe, build)) in self.join_keys.iter().enumerate().rev() {
            let d = depth + (self.join_keys.len() - 1 - i);
            out.push(format!("{}HashJoin {probe} = {build}", "  ".repeat(d)));
            out.push(format!(
                "{}{}",
                "  ".repeat(d + 1),
                scan_line(&self.scans[i + 1])
            ));
        }
        out.push(format!(
            "{}{}",
            "  ".repeat(depth + self.join_keys.len()),
            scan_line(&self.scans[0])
        ));
        out.join("\n")
    }
}
