//! The pivot itself.
//!
//! 1. Filter rows, encode every row/column field into small integer codes (`encode`).
//! 2. Pack each row's row-field codes into one `u64` and its column-field codes into another
//!    (mixed radix), and aggregate into a hash map keyed by that pair: one pass over the data.
//! 3. Build every subtotal and grand total by *merging* those finest cells into coarser ones
//!    (a prefix of the row key crossed with a prefix of the column key). The rows are never
//!    touched again, and a mean, a standard deviation or a distinct count of a total is exact,
//!    not an average of averages.
//! 4. Lay the items out (sort, top-N, subtotals, totals), finish each cell, evaluate calculated
//!    measures, and apply "show values as".

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::acc::Acc;
use crate::encode::{self, Encoded};
use crate::error::{Error, Result};
use crate::expr::{self, Expr};
use crate::filter;
use crate::spec::*;
use crate::table::{Column, Table};

pub use crate::fxhash::{Fx, FxMap};

type Cells = FxMap<(u64, u64), Vec<Acc>>;
/// Leaf keys sharing a prefix: `(key, codes)` pairs.
type Group = Vec<(u64, Vec<u32>)>;
/// One code at some level and the leaf keys beneath it.
type Node = (u32, Group);
type Cache = RefCell<FxMap<(u8, u8, u64, u64), Rc<Vec<Option<f64>>>>>;

#[derive(Clone)]
enum Source<'a> {
    Rows,
    Field(&'a Column),
    Calc(Expr, Vec<usize>),
}

struct Measure<'a> {
    label: String,
    agg: Agg,
    src: Source<'a>,
    weight: Option<&'a Column>,
    show_as: ShowAs,
    /// Position among the accumulators (calculated measures have none).
    acc: Option<usize>,
}

/// Everything both the pivot and the drill-through need to agree on.
pub struct Prepared {
    pub keep: Vec<bool>,
    pub row_encs: Vec<Encoded>,
    pub col_encs: Vec<Encoded>,
}

pub fn prepare(table: &Table, spec: &PivotSpec) -> Result<Prepared> {
    let blank = spec.options.blank_label.as_str();
    let mut keep = filter::mask(table, &spec.filters)?;
    for dim in spec.rows.iter().chain(&spec.cols).filter(|d| d.hide_blank) {
        let enc = encode::encode(table, dim, &keep, blank)?;
        if let Some(b) = enc.blank {
            for (k, code) in keep.iter_mut().zip(&enc.codes) {
                if *k && *code == b {
                    *k = false;
                }
            }
        }
    }
    let encode_all = |dims: &[Dim]| {
        dims.iter()
            .map(|d| encode::encode(table, d, &keep, blank))
            .collect::<Result<Vec<_>>>()
    };
    Ok(Prepared {
        row_encs: encode_all(&spec.rows)?,
        col_encs: encode_all(&spec.cols)?,
        keep,
    })
}

fn measures<'a>(table: &'a Table, spec: &PivotSpec) -> Result<Vec<Measure<'a>>> {
    let default = [Value {
        field: None,
        agg: Agg::Count,
        label: None,
        weight: None,
        calc: None,
        show_as: ShowAs::Normal,
    }];
    let values: &[Value] = if spec.values.is_empty() {
        &default
    } else {
        &spec.values
    };
    let mut out: Vec<Measure> = Vec::new();
    let mut n_acc = 0;
    for v in values {
        let mut label = v
            .label
            .clone()
            .unwrap_or_else(|| match (&v.calc, &v.field) {
                (Some(_), _) => format!("calc {}", out.len() + 1),
                (None, Some(f)) => format!("{}({f})", v.agg.name()),
                (None, None) => "count".to_string(),
            });
        let mut n = 2;
        let base = label.clone();
        while out.iter().any(|m| m.label == label) {
            label = format!("{base} ({n})");
            n += 1;
        }
        let (src, acc, weight) = if let Some(formula) = &v.calc {
            let e =
                expr::parse(formula).map_err(|e| Error::new(format!("measure '{label}': {e}")))?;
            let refs = e
                .vars()
                .iter()
                .map(|name| {
                    out.iter().position(|m| &m.label == name).ok_or_else(|| {
                        let have: Vec<&str> = out.iter().map(|m| m.label.as_str()).collect();
                        Error::new(format!("measure '{label}': the formula uses '{name}', which is not an earlier measure ({})", have.join(", ")))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            (Source::Calc(e, refs), None, None)
        } else {
            let acc = Some(n_acc);
            n_acc += 1;
            match &v.field {
                None if v.agg == Agg::Count => (Source::Rows, acc, None),
                None => {
                    return Err(Error::new(format!(
                        "measure '{label}': {} needs a field",
                        v.agg.name()
                    )));
                }
                Some(f) => {
                    let (_, col) = table.find(f)?;
                    if v.agg.needs_number() && !col.is_numeric() {
                        return Err(Error::new(format!(
                            "measure '{label}': cannot take the {} of '{f}': it is text (count and distinct work on text)",
                            v.agg.name()
                        )));
                    }
                    let weight = match (&v.weight, v.agg) {
                        (Some(w), Agg::WeightedMean) => {
                            let (_, wc) = table.find(w)?;
                            if !wc.is_numeric() {
                                return Err(Error::new(format!(
                                    "measure '{label}': the weight '{w}' is not a number"
                                )));
                            }
                            Some(wc)
                        }
                        (None, Agg::WeightedMean) => {
                            return Err(Error::new(format!(
                                "measure '{label}': weighted_mean needs a \"weight\" field"
                            )));
                        }
                        _ => None,
                    };
                    (Source::Field(col), acc, weight)
                }
            }
        };
        out.push(Measure {
            label,
            agg: v.agg,
            src,
            weight,
            show_as: v.show_as.clone(),
            acc,
        });
    }
    Ok(out)
}

/// `strides[i]` such that `key = sum(code_i * strides[i])` with the first field most significant.
fn strides(encs: &[Encoded], what: &str) -> Result<Vec<u64>> {
    let mut out = vec![1u64; encs.len()];
    for i in (0..encs.len().saturating_sub(1)).rev() {
        let card = encs[i + 1].labels.len().max(1) as u64;
        out[i] = out[i + 1].checked_mul(card).ok_or_else(|| {
            Error::new(format!(
                "too many {what} combinations: group some fields or use fewer"
            ))
        })?;
    }
    if let Some(first) = encs.first() {
        out[0]
            .checked_mul(first.labels.len().max(1) as u64)
            .ok_or_else(|| {
                Error::new(format!(
                    "too many {what} combinations: group some fields or use fewer"
                ))
            })?;
    }
    Ok(out)
}

struct Lane<'a> {
    k: usize,
    encs: &'a [Encoded],
    strides: Vec<u64>,
    dims: &'a [Dim],
}

impl Lane<'_> {
    fn code_at(&self, key: u64, level: usize) -> u32 {
        ((key / self.strides[level]) % self.encs[level].labels.len().max(1) as u64) as u32
    }

    /// The key of the first `depth` fields (what `key` becomes once the rest are dropped).
    fn prefix(&self, key: u64, depth: usize) -> u64 {
        if depth == 0 {
            0
        } else {
            key / self.strides[depth - 1]
        }
    }

    /// The key at `depth` for a list of codes.
    fn key_of(&self, codes: &[u32]) -> u64 {
        if codes.is_empty() {
            return 0;
        }
        let d = codes.len();
        let sum: u64 = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| c as u64 * self.strides[i])
            .sum();
        sum / self.strides[d - 1]
    }

    fn parent_key(&self, key: u64, depth: usize) -> u64 {
        if depth <= 1 {
            0
        } else {
            key / self.encs[depth - 1].labels.len().max(1) as u64
        }
    }
}

#[derive(Debug, Clone)]
struct Item {
    depth: usize,
    kind: HeaderKind,
    key: u64,
    codes: Vec<u32>,
}

struct Grid<'a> {
    maps: Vec<Vec<Cells>>,
    ms: &'a [Measure<'a>],
    cache: Cache,
}

impl Grid<'_> {
    /// Every measure of one cell, finished (calculated measures evaluated).
    fn base(&self, p: usize, q: usize, rk: u64, ck: u64) -> Rc<Vec<Option<f64>>> {
        let id = (p as u8, q as u8, rk, ck);
        if let Some(v) = self.cache.borrow().get(&id) {
            return Rc::clone(v);
        }
        let v = Rc::new(self.finish(self.maps[p][q].get(&(rk, ck))));
        self.cache.borrow_mut().insert(id, Rc::clone(&v));
        v
    }

    fn finish(&self, accs: Option<&Vec<Acc>>) -> Vec<Option<f64>> {
        let mut vals: Vec<Option<f64>> = Vec::with_capacity(self.ms.len());
        for m in self.ms {
            let v = match (&m.src, m.acc, accs) {
                (Source::Calc(e, refs), _, Some(_)) => {
                    let names: Vec<(&str, Option<f64>)> = refs
                        .iter()
                        .map(|&i| (self.ms[i].label.as_str(), vals[i]))
                        .collect();
                    e.eval(&|n| names.iter().find(|(l, _)| *l == n).and_then(|(_, v)| *v))
                }
                (_, Some(a), Some(accs)) => accs[a].finish(m.agg),
                _ => None,
            };
            vals.push(v);
        }
        vals
    }
}

/// Distinct leaf keys of one axis, as `(key, codes)`.
fn leaf_keys(axis: &Lane, keys: impl Iterator<Item = u64>) -> Vec<(u64, Vec<u32>)> {
    let mut v: Vec<u64> = keys.collect();
    v.sort_unstable();
    v.dedup();
    v.into_iter()
        .map(|key| (key, (0..axis.k).map(|i| axis.code_at(key, i)).collect()))
        .collect()
}

struct Layout<'a, 'g> {
    axis: &'a Lane<'a>,
    options: &'a Options,
    grid: &'a Grid<'g>,
    rows_axis: bool,
    measure_index: &'a dyn Fn(&str) -> Result<usize>,
}

impl Layout<'_, '_> {
    fn value_of(&self, codes: &[u32], j: usize) -> Option<f64> {
        let key = self.axis.key_of(codes);
        let d = codes.len();
        let accs = if self.rows_axis {
            self.grid.maps[d][0].get(&(key, 0))
        } else {
            self.grid.maps[0][d].get(&(0, key))
        };
        self.grid.finish(accs)[j]
    }

    fn sort_nodes(&self, level: usize, prefix: &[u32], nodes: &mut Vec<Node>) -> Result<()> {
        let dim = &self.axis.dims[level];
        let blank = self.axis.encs[level].blank;
        let value_key = |m: &str, code: u32| -> Result<Option<f64>> {
            let j = (self.measure_index)(m)?;
            let mut codes = prefix.to_vec();
            codes.push(code);
            Ok(self.value_of(&codes, j))
        };
        let by_value = |nodes: &mut Vec<Node>, m: &str, desc: bool| -> Result<()> {
            let mut keyed: Vec<(Option<f64>, Node)> = nodes
                .drain(..)
                .map(|n| Ok((value_key(m, n.0)?, n)))
                .collect::<Result<_>>()?;
            keyed.sort_by(|a, b| match (a.0, b.0) {
                (Some(x), Some(y)) => if desc {
                    y.total_cmp(&x)
                } else {
                    x.total_cmp(&y)
                }
                .then(a.1.0.cmp(&b.1.0)),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.1.0.cmp(&b.1.0),
            });
            nodes.extend(keyed.into_iter().map(|(_, n)| n));
            Ok(())
        };
        if let Some(top) = &dim.top {
            by_value(nodes, &top.measure, top.desc)?;
            nodes.truncate(top.n);
        }
        match &dim.sort {
            Sort::Label { desc } => nodes.sort_by(|a, b| {
                let (ab, bb) = (Some(a.0) == blank, Some(b.0) == blank);
                if ab != bb {
                    return ab.cmp(&bb); // blank always last
                }
                if *desc { b.0.cmp(&a.0) } else { a.0.cmp(&b.0) }
            }),
            Sort::Value { measure, desc } => by_value(nodes, measure, *desc)?,
        }
        Ok(())
    }

    fn walk(
        &self,
        level: usize,
        group: Group,
        prefix: &mut Vec<u32>,
        out: &mut Vec<Item>,
    ) -> Result<()> {
        let mut by_code: Vec<Node> = Vec::new();
        let mut index: HashMap<u32, usize> = HashMap::new();
        for entry in group {
            let code = entry.1[level];
            let slot = *index.entry(code).or_insert_with(|| {
                by_code.push((code, Vec::new()));
                by_code.len() - 1
            });
            by_code[slot].1.push(entry);
        }
        self.sort_nodes(level, prefix, &mut by_code)?;
        for (code, sub) in by_code {
            prefix.push(code);
            let depth = level + 1;
            if depth == self.axis.k {
                out.push(Item {
                    depth,
                    kind: HeaderKind::Leaf,
                    key: self.axis.key_of(prefix),
                    codes: prefix.clone(),
                });
            } else {
                let subtotal = Item {
                    depth,
                    kind: HeaderKind::Subtotal,
                    key: self.axis.key_of(prefix),
                    codes: prefix.clone(),
                };
                let on = self.options.subtotals;
                if on && self.options.subtotal_position == SubtotalPosition::Top {
                    out.push(subtotal.clone());
                }
                self.walk(depth, sub, prefix, out)?;
                if on && self.options.subtotal_position == SubtotalPosition::Bottom {
                    out.push(subtotal);
                }
            }
            prefix.pop();
        }
        Ok(())
    }

    fn items(&self, leaves: Vec<(u64, Vec<u32>)>, totals: bool) -> Result<Vec<Item>> {
        if self.axis.k == 0 {
            return Ok(vec![Item {
                depth: 0,
                kind: HeaderKind::Leaf,
                key: 0,
                codes: vec![],
            }]);
        }
        let mut out = Vec::new();
        if !leaves.is_empty() {
            self.walk(0, leaves, &mut Vec::new(), &mut out)?;
            if totals {
                out.push(Item {
                    depth: 0,
                    kind: HeaderKind::Total,
                    key: 0,
                    codes: vec![],
                });
            }
        }
        Ok(out)
    }
}

/// Items that are compared with each other by "difference from previous", running totals and rank:
/// same kind, same depth, same parent.
fn sibling_groups(axis: &Lane, items: &[Item]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut index: HashMap<(u8, usize, u64), usize> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        let kind = match it.kind {
            HeaderKind::Leaf => 0,
            HeaderKind::Subtotal => 1,
            HeaderKind::Total => 2,
        };
        let parent = axis.parent_key(it.key, it.depth);
        let slot = *index.entry((kind, it.depth, parent)).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[slot].push(i);
    }
    groups
}

fn ratio(num: Option<f64>, den: Option<f64>) -> Option<f64> {
    match (num, den) {
        (Some(n), Some(d)) if d != 0.0 => Some(n / d),
        _ => None,
    }
}

/// Add every used row to its cell's accumulator for one measure. The column's type is matched
/// once, outside the loop.
fn feed(keep: &[bool], cell_of: &[u32], cells: &mut [Vec<Acc>], a: usize, m: &Measure) {
    let agg = m.agg;
    let weight = m.weight;
    match &m.src {
        Source::Rows => {
            for (row, _) in keep.iter().enumerate().filter(|(_, k)| **k) {
                cells[cell_of[row] as usize][a].update(agg, row, Some(1.0), None, None);
            }
        }
        Source::Field(col) if agg.needs_number() => match col {
            Column::Int(v) | Column::Date(v) | Column::DateTime(v) => {
                for (row, x) in v.iter().enumerate().filter(|(r, _)| keep[*r]) {
                    let w = weight.and_then(|c| c.number(row));
                    cells[cell_of[row] as usize][a].update(agg, row, x.map(|x| x as f64), None, w);
                }
            }
            Column::Float(v) => {
                for (row, &x) in v.iter().enumerate().filter(|(r, _)| keep[*r]) {
                    let w = weight.and_then(|c| c.number(row));
                    let x = (!x.is_nan()).then_some(x);
                    cells[cell_of[row] as usize][a].update(agg, row, x, None, w);
                }
            }
            Column::Str { .. } => {} // refused when the measures were resolved
        },
        Source::Field(col) => {
            for (row, _) in keep.iter().enumerate().filter(|(_, k)| **k) {
                cells[cell_of[row] as usize][a].update(agg, row, None, col.identity(row), None);
            }
        }
        Source::Calc(..) => {}
    }
}

pub fn pivot(table: &Table, spec: &PivotSpec) -> Result<PivotResult> {
    let ms = measures(table, spec)?;
    let prep = prepare(table, spec)?;
    let (rk_n, ck_n) = (spec.rows.len(), spec.cols.len());
    let raxis = Lane {
        k: rk_n,
        encs: &prep.row_encs,
        strides: strides(&prep.row_encs, "row")?,
        dims: &spec.rows,
    };
    let caxis = Lane {
        k: ck_n,
        encs: &prep.col_encs,
        strides: strides(&prep.col_encs, "column")?,
        dims: &spec.cols,
    };
    let rows_used = prep.keep.iter().filter(|&&k| k).count();
    let mut warnings = Vec::new();
    if rows_used == 0 {
        warnings.push("no rows match the filters".to_string());
    }

    // 1. finest cells. First pass: which cell does each row belong to (one hash per row, and a
    // one-entry cache because real data is often clustered). Second pass: one tight loop per
    // measure over those cell numbers.
    let fresh = || -> Vec<Acc> {
        ms.iter()
            .filter(|m| m.acc.is_some())
            .map(|m| Acc::new(m.agg))
            .collect()
    };
    let mut index: FxMap<(u64, u64), u32> = FxMap::default();
    let mut keys: Vec<(u64, u64)> = Vec::new();
    let mut cell_of = vec![0u32; table.rows];
    let mut last = (u64::MAX, u64::MAX, 0u32);
    for (row, &used) in prep.keep.iter().enumerate() {
        if !used {
            continue;
        }
        let rk: u64 = prep
            .row_encs
            .iter()
            .zip(&raxis.strides)
            .map(|(e, s)| e.codes[row] as u64 * s)
            .sum();
        let ck: u64 = prep
            .col_encs
            .iter()
            .zip(&caxis.strides)
            .map(|(e, s)| e.codes[row] as u64 * s)
            .sum();
        let id = if (rk, ck) == (last.0, last.1) {
            last.2
        } else {
            *index.entry((rk, ck)).or_insert_with(|| {
                keys.push((rk, ck));
                (keys.len() - 1) as u32
            })
        };
        last = (rk, ck, id);
        cell_of[row] = id;
    }
    let mut cells: Vec<Vec<Acc>> = keys.iter().map(|_| fresh()).collect();
    for m in &ms {
        if let Some(a) = m.acc {
            feed(&prep.keep, &cell_of, &mut cells, a, m);
        }
    }
    let leaf: Cells = keys.into_iter().zip(cells).collect();

    // 2. every coarser level by merging
    let mut maps: Vec<Vec<Cells>> = (0..=rk_n)
        .map(|_| (0..=ck_n).map(|_| FxMap::default()).collect())
        .collect();
    let aggs: Vec<Agg> = ms
        .iter()
        .filter(|m| m.acc.is_some())
        .map(|m| m.agg)
        .collect();
    for (&(rk, ck), accs) in &leaf {
        for (p, level) in maps.iter_mut().enumerate() {
            for (q, cells) in level.iter_mut().enumerate() {
                if p == rk_n && q == ck_n {
                    continue;
                }
                let target = cells
                    .entry((raxis.prefix(rk, p), caxis.prefix(ck, q)))
                    .or_insert_with(fresh);
                for ((t, a), agg) in target.iter_mut().zip(accs).zip(&aggs) {
                    t.merge(*agg, a);
                }
            }
        }
    }
    let leaf_row_keys = leaf_keys(&raxis, leaf.keys().map(|k| k.0));
    let leaf_col_keys = leaf_keys(&caxis, leaf.keys().map(|k| k.1));
    maps[rk_n][ck_n] = leaf;
    let grid = Grid {
        maps,
        ms: &ms,
        cache: RefCell::new(FxMap::default()),
    };

    // 3. layout
    let index_of = |name: &str| -> Result<usize> {
        ms.iter().position(|m| m.label == name).ok_or_else(|| {
            let have: Vec<&str> = ms.iter().map(|m| m.label.as_str()).collect();
            Error::new(format!(
                "no measure called '{name}' (measures: {})",
                have.join(", ")
            ))
        })
    };
    let row_items = Layout {
        axis: &raxis,
        options: &spec.options,
        grid: &grid,
        rows_axis: true,
        measure_index: &index_of,
    }
    .items(leaf_row_keys, spec.options.totals_row)?;
    let col_items = Layout {
        axis: &caxis,
        options: &spec.options,
        grid: &grid,
        rows_axis: false,
        measure_index: &index_of,
    }
    .items(leaf_col_keys, spec.options.totals_col)?;
    if row_items.len().saturating_mul(col_items.len()) > spec.options.max_cells {
        return Err(Error::new(format!(
            "the result would have {} x {} cells, over the limit of {}: filter, group or use fewer fields",
            row_items.len(),
            col_items.len(),
            spec.options.max_cells
        )));
    }
    let row_groups = sibling_groups(&raxis, &row_items);
    let col_groups = sibling_groups(&caxis, &col_items);
    let position = |groups: &[Vec<usize>], i: usize| -> (usize, usize) {
        for (g, members) in groups.iter().enumerate() {
            if let Some(pos) = members.iter().position(|&m| m == i) {
                return (g, pos);
            }
        }
        (0, 0)
    };
    let row_pos: Vec<(usize, usize)> = (0..row_items.len())
        .map(|i| position(&row_groups, i))
        .collect();
    let col_pos: Vec<(usize, usize)> = (0..col_items.len())
        .map(|i| position(&col_groups, i))
        .collect();

    // 4. cells
    let mut cells: Vec<Vec<Vec<Option<f64>>>> = Vec::with_capacity(row_items.len());
    for (ri, r) in row_items.iter().enumerate() {
        let mut line = Vec::with_capacity(col_items.len());
        for (ci, c) in col_items.iter().enumerate() {
            let base = grid.base(r.depth, c.depth, r.key, c.key);
            let mut out = Vec::with_capacity(ms.len());
            for (j, m) in ms.iter().enumerate() {
                let v = base[j];
                let at = |p: usize, q: usize, rk: u64, ck: u64| grid.base(p, q, rk, ck)[j];
                let shown = match &m.show_as {
                    ShowAs::Normal => v,
                    ShowAs::PctGrand => ratio(v, at(0, 0, 0, 0)),
                    ShowAs::PctRow => ratio(v, at(r.depth, 0, r.key, 0)),
                    ShowAs::PctCol => ratio(v, at(0, c.depth, 0, c.key)),
                    ShowAs::PctParentRow => {
                        let (p, k) = if r.depth == 0 {
                            (0, 0)
                        } else {
                            (r.depth - 1, raxis.parent_key(r.key, r.depth))
                        };
                        ratio(v, at(p, c.depth, k, c.key))
                    }
                    ShowAs::PctParentCol => {
                        let (q, k) = if c.depth == 0 {
                            (0, 0)
                        } else {
                            (c.depth - 1, caxis.parent_key(c.key, c.depth))
                        };
                        ratio(v, at(r.depth, q, r.key, k))
                    }
                    ShowAs::Index => ratio(
                        v.zip(at(0, 0, 0, 0)).map(|(a, b)| a * b),
                        at(r.depth, 0, r.key, 0)
                            .zip(at(0, c.depth, 0, c.key))
                            .map(|(a, b)| a * b),
                    ),
                    ShowAs::Diff { along }
                    | ShowAs::PctDiff { along }
                    | ShowAs::Running { along }
                    | ShowAs::Rank { along, .. } => {
                        let rows = *along == Axis::Rows;
                        let (items, groups, pos, me) = if rows {
                            (&row_items, &row_groups, &row_pos, ri)
                        } else {
                            (&col_items, &col_groups, &col_pos, ci)
                        };
                        let (g, k) = pos[me];
                        let members = &groups[g];
                        let value_at = |idx: usize| -> Option<f64> {
                            let it = &items[idx];
                            if rows {
                                at(it.depth, c.depth, it.key, c.key)
                            } else {
                                at(r.depth, it.depth, r.key, it.key)
                            }
                        };
                        match &m.show_as {
                            ShowAs::Diff { .. } => (k > 0)
                                .then(|| v.zip(value_at(members[k - 1])).map(|(a, b)| a - b))
                                .flatten(),
                            ShowAs::PctDiff { .. } => {
                                if k == 0 {
                                    None
                                } else {
                                    let prev = value_at(members[k - 1]);
                                    ratio(v.zip(prev).map(|(a, b)| a - b), prev)
                                }
                            }
                            ShowAs::Running { .. } => {
                                v.map(|_| members[..=k].iter().filter_map(|&i| value_at(i)).sum())
                            }
                            ShowAs::Rank { desc, .. } => v.map(|x| {
                                let better = members
                                    .iter()
                                    .filter_map(|&i| value_at(i))
                                    .filter(|&y| if *desc { y > x } else { y < x })
                                    .count();
                                (better + 1) as f64
                            }),
                            _ => unreachable!(),
                        }
                    }
                };
                out.push(shown);
            }
            line.push(out);
        }
        cells.push(line);
    }

    let header = |axis: &Lane, it: &Item| Header {
        labels: (0..axis.k)
            .map(|i| {
                it.codes
                    .get(i)
                    .map(|&c| axis.encs[i].labels[c as usize].clone())
            })
            .collect(),
        kind: it.kind,
        depth: it.depth,
    };
    Ok(PivotResult {
        row_fields: spec.rows.iter().map(encode::title).collect(),
        col_fields: spec.cols.iter().map(encode::title).collect(),
        measures: ms
            .iter()
            .map(|m| MeasureInfo {
                label: m.label.clone(),
                agg: if matches!(m.src, Source::Calc(..)) {
                    "calc".to_string()
                } else {
                    m.agg.name().to_string()
                },
                show_as: m.show_as.name().to_string(),
                percent: m.show_as.is_percent(),
            })
            .collect(),
        rows: row_items.iter().map(|it| header(&raxis, it)).collect(),
        cols: col_items.iter().map(|it| header(&caxis, it)).collect(),
        cells,
        rows_used,
        rows_total: table.rows,
        warnings,
    })
}
