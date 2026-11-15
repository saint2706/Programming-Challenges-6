//! Vectorized execution: every operator works on a whole [`Batch`] (one row group) at a time,
//! and expressions evaluate column-at-a-time with scalars kept unbroadcast.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::Result;
use crate::format::Table;
use crate::plan::{self, Plan};
use crate::sql::{self, AggFn, BinOp, Expr};
use crate::types::{Batch, Column, DataType, Value};

/// Table name -> `.col` file.
pub type Catalog = HashMap<String, PathBuf>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub row_groups_read: usize,
    pub row_groups_total: usize,
    pub rows_scanned: usize,
}

#[derive(Debug)]
pub enum QueryResult {
    Rows(Batch, Stats),
    Explain(String),
}

pub fn query(catalog: &Catalog, sql_text: &str) -> Result<QueryResult> {
    let q = sql::parse(sql_text)?;
    let plan = plan::build(q, catalog)?;
    if plan.query.explain {
        return Ok(QueryResult::Explain(plan.explain()));
    }
    let (b, s) = run(plan)?;
    Ok(QueryResult::Rows(b, s))
}

// ---- expression evaluation ------------------------------------------------------------------

enum Val<'a> {
    Col(Cow<'a, Column>),
    Scalar(Value),
}

/// A numeric operand: a slice or a constant, so `x * 2` never allocates a column of 2s.
enum Src<'a, T> {
    S(&'a [T]),
    K(T),
}

impl<T: Copy> Src<'_, T> {
    #[inline]
    fn at(&self, i: usize) -> T {
        match self {
            Src::S(s) => s[i],
            Src::K(k) => *k,
        }
    }
}

fn map2<A: Copy, B: Copy, R>(a: &Src<A>, b: &Src<B>, n: usize, f: impl Fn(A, B) -> R) -> Vec<R> {
    (0..n).map(|i| f(a.at(i), b.at(i))).collect()
}

fn dtype(v: &Val) -> DataType {
    match v {
        Val::Col(c) => c.dtype(),
        Val::Scalar(s) => s.dtype(),
    }
}

fn floats<'a>(v: &'a Val) -> Result<Src<'a, f64>> {
    Ok(match v {
        Val::Col(c) => match &**c {
            Column::Float(x) => Src::S(x),
            _ => unreachable!("int columns are converted by the caller"),
        },
        Val::Scalar(s) => Src::K(s.as_f64().ok_or("arithmetic on a string")?),
    })
}

fn to_float_col(v: Val) -> Result<Val> {
    Ok(match v {
        Val::Col(c) => match &*c {
            Column::Int(x) => Val::Col(Cow::Owned(Column::Float(
                x.iter().map(|&v| v as f64).collect(),
            ))),
            Column::Float(_) => Val::Col(c),
            Column::Str(_) => return Err("arithmetic on a string column".into()),
        },
        s => s,
    })
}

fn ints<'a>(v: &'a Val) -> Src<'a, i64> {
    match v {
        Val::Col(c) => match &**c {
            Column::Int(x) => Src::S(x),
            _ => unreachable!(),
        },
        Val::Scalar(Value::Int(k)) => Src::K(*k),
        Val::Scalar(_) => unreachable!(),
    }
}

fn bools(v: Vec<bool>) -> Column {
    Column::Int(v.into_iter().map(i64::from).collect())
}

fn ord_ok(op: BinOp, o: Ordering) -> bool {
    match op {
        BinOp::Eq => o == Ordering::Equal,
        BinOp::Ne => o != Ordering::Equal,
        BinOp::Lt => o == Ordering::Less,
        BinOp::Le => o != Ordering::Greater,
        BinOp::Gt => o == Ordering::Greater,
        BinOp::Ge => o != Ordering::Less,
        _ => unreachable!(),
    }
}

fn scalar_bin(a: &Value, op: BinOp, b: &Value) -> Result<Value> {
    let one = |v: &Value| match v {
        Value::Int(x) => Column::Int(vec![*x]),
        Value::Float(x) => Column::Float(vec![*x]),
        Value::Str(s) => Column::Str(vec![s.clone()]),
    };
    let r = bin(
        Val::Col(Cow::Owned(one(a))),
        op,
        Val::Col(Cow::Owned(one(b))),
        1,
    )?;
    Ok(match r {
        Val::Col(c) => c.get(0),
        Val::Scalar(s) => s,
    })
}

fn bin<'a>(a: Val<'a>, op: BinOp, b: Val<'a>, n: usize) -> Result<Val<'a>> {
    use BinOp::*;
    if let (Val::Scalar(x), Val::Scalar(y)) = (&a, &b) {
        return scalar_bin(x, op, y).map(Val::Scalar); // constant folding
    }
    let (ta, tb) = (dtype(&a), dtype(&b));
    Ok(Val::Col(Cow::Owned(match op {
        And | Or => {
            let (ma, mb) = (mask_of(a, n)?, mask_of(b, n)?);
            bools(
                ma.iter()
                    .zip(&mb)
                    .map(|(&x, &y)| if op == And { x && y } else { x || y })
                    .collect(),
            )
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            if (ta == DataType::Str) != (tb == DataType::Str) {
                return Err(format!("cannot compare {ta:?} with {tb:?}"));
            }
            if ta == DataType::Str {
                bools(cmp_str(&a, &b, n, op))
            } else if ta == DataType::Int && tb == DataType::Int {
                let (x, y) = (ints(&a), ints(&b));
                bools(map2(&x, &y, n, |p, q| ord_ok(op, p.cmp(&q))))
            } else {
                let (a, b) = (to_float_col(a)?, to_float_col(b)?);
                let (x, y) = (floats(&a)?, floats(&b)?);
                bools(map2(&x, &y, n, |p, q| ord_ok(op, p.total_cmp(&q))))
            }
        }
        Add | Sub | Mul | Div => {
            if ta == DataType::Str || tb == DataType::Str {
                return Err(format!("operator {} on a string", op.symbol()));
            }
            if ta == DataType::Int && tb == DataType::Int && op != Div {
                let (x, y) = (ints(&a), ints(&b));
                let f = match op {
                    Add => i64::checked_add,
                    Sub => i64::checked_sub,
                    _ => i64::checked_mul,
                };
                let v: Option<Vec<i64>> = (0..n).map(|i| f(x.at(i), y.at(i))).collect();
                Column::Int(v.ok_or("integer overflow")?)
            } else {
                let (a, b) = (to_float_col(a)?, to_float_col(b)?);
                let (x, y) = (floats(&a)?, floats(&b)?);
                Column::Float(match op {
                    Add => map2(&x, &y, n, |p, q| p + q),
                    Sub => map2(&x, &y, n, |p, q| p - q),
                    Mul => map2(&x, &y, n, |p, q| p * q),
                    _ => map2(&x, &y, n, |p, q| p / q),
                })
            }
        }
    })))
}

fn cmp_str(a: &Val, b: &Val, n: usize, op: BinOp) -> Vec<bool> {
    fn s<'x>(v: &'x Val, i: usize) -> &'x str {
        match v {
            Val::Col(c) => match &**c {
                Column::Str(s) => &s[i],
                _ => unreachable!(),
            },
            Val::Scalar(Value::Str(s)) => s,
            _ => unreachable!(),
        }
    }
    (0..n).map(|i| ord_ok(op, s(a, i).cmp(s(b, i)))).collect()
}

fn eval<'a>(e: &Expr, b: &'a Batch) -> Result<Val<'a>> {
    let n = b.rows();
    Ok(match e {
        Expr::Col(c) => {
            let i = b.index(c).ok_or_else(|| format!("unknown column {c:?}"))?;
            Val::Col(Cow::Borrowed(&b.columns[i]))
        }
        Expr::Lit(v) => Val::Scalar(v.clone()),
        Expr::Bin(x, op, y) => bin(eval(x, b)?, *op, eval(y, b)?, n)?,
        Expr::Not(x) => Val::Col(Cow::Owned(bools(
            mask_of(eval(x, b)?, n)?.into_iter().map(|v| !v).collect(),
        ))),
        Expr::Neg(x) => bin(Val::Scalar(Value::Int(0)), BinOp::Sub, eval(x, b)?, n)?,
        Expr::Agg(..) => return Err(format!("aggregate {e} is not allowed here")),
    })
}

fn mask_of(v: Val, n: usize) -> Result<Vec<bool>> {
    Ok(match v {
        Val::Col(c) => match &*c {
            Column::Int(x) => x.iter().map(|&v| v != 0).collect(),
            Column::Float(x) => x.iter().map(|&v| v != 0.0).collect(),
            Column::Str(_) => return Err("a string is not a condition".into()),
        },
        Val::Scalar(s) => vec![s.as_f64().ok_or("a string is not a condition")? != 0.0; n],
    })
}

pub fn eval_column(e: &Expr, b: &Batch) -> Result<Column> {
    let n = b.rows();
    Ok(match eval(e, b)? {
        Val::Col(c) => c.into_owned(),
        Val::Scalar(Value::Int(v)) => Column::Int(vec![v; n]),
        Val::Scalar(Value::Float(v)) => Column::Float(vec![v; n]),
        Val::Scalar(Value::Str(v)) => Column::Str(vec![v; n]),
    })
}

fn filter_batch(b: Batch, preds: &[Expr]) -> Result<Batch> {
    if preds.is_empty() {
        return Ok(b);
    }
    let n = b.rows();
    let mut keep = vec![true; n];
    for p in preds {
        for (k, m) in keep.iter_mut().zip(mask_of(eval(p, &b)?, n)?) {
            *k &= m;
        }
    }
    if keep.iter().all(|&k| k) {
        return Ok(b);
    }
    Ok(Batch {
        names: b.names,
        columns: b.columns.iter().map(|c| c.filter(&keep)).collect(),
    })
}

// ---- hashing keys -----------------------------------------------------------------------------

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum K {
    I(i64),
    F(u64),
    S(String),
}

fn key_at(c: &Column, i: usize) -> K {
    match c {
        Column::Int(v) => K::I(v[i]),
        Column::Float(v) => K::F(if v[i] == 0.0 { 0 } else { v[i].to_bits() }), // -0.0 == 0.0
        Column::Str(v) => K::S(v[i].clone()),
    }
}

// ---- aggregation --------------------------------------------------------------------------------

struct Agg {
    f: AggFn,
    arg: Option<Expr>,
    dtype: Option<DataType>,
    count: Vec<i64>,
    isum: Vec<i64>,
    fsum: Vec<f64>,
    best: Vec<Option<Value>>,
}

impl Agg {
    fn grow(&mut self) {
        self.count.push(0);
        self.isum.push(0);
        self.fsum.push(0.0);
        self.best.push(None);
    }
    fn update(&mut self, ids: &[usize], b: &Batch) -> Result<()> {
        let Some(arg) = &self.arg else {
            ids.iter().for_each(|&g| self.count[g] += 1);
            return Ok(());
        };
        let col = eval_column(arg, b)?;
        if self.dtype.get_or_insert(col.dtype()) != &col.dtype() {
            return Err("aggregate argument changed type between batches".into());
        }
        if matches!(self.f, AggFn::Sum | AggFn::Avg) && col.dtype() == DataType::Str {
            return Err(format!("{:?} of a string column", self.f));
        }
        ids.iter().for_each(|&g| self.count[g] += 1);
        match (self.f, &col) {
            (AggFn::Sum | AggFn::Avg, Column::Int(v)) => {
                for (&g, &x) in ids.iter().zip(v) {
                    self.isum[g] = self.isum[g]
                        .checked_add(x)
                        .ok_or("integer overflow in SUM")?;
                }
            }
            (AggFn::Sum | AggFn::Avg, Column::Float(v)) => {
                ids.iter().zip(v).for_each(|(&g, &x)| self.fsum[g] += x);
            }
            (AggFn::Min | AggFn::Max, _) => {
                let want = if self.f == AggFn::Min {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
                for (i, &g) in ids.iter().enumerate() {
                    let better = match &self.best[g] {
                        None => true,
                        Some(cur) => match (&col, cur) {
                            (Column::Int(v), Value::Int(c)) => v[i].cmp(c) == want,
                            (Column::Float(v), Value::Float(c)) => v[i].total_cmp(c) == want,
                            (Column::Str(v), Value::Str(c)) => {
                                v[i].as_str().cmp(c.as_str()) == want
                            }
                            _ => unreachable!(),
                        },
                    };
                    if better {
                        self.best[g] = Some(col.get(i));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn finish(&self) -> Column {
        let groups = self.count.len();
        match self.f {
            AggFn::Count => Column::Int(self.count.clone()),
            AggFn::Sum if self.dtype == Some(DataType::Float) => Column::Float(self.fsum.clone()),
            AggFn::Sum => Column::Int(self.isum.clone()),
            AggFn::Avg => Column::Float(
                (0..groups)
                    .map(|g| {
                        let s = if self.dtype == Some(DataType::Int) {
                            self.isum[g] as f64
                        } else {
                            self.fsum[g]
                        };
                        s / self.count[g] as f64 // shortcut: no NULLs, so AVG of zero rows is NaN
                    })
                    .collect(),
            ),
            AggFn::Min | AggFn::Max => {
                let mut c = Column::empty(self.dtype.unwrap_or(DataType::Float));
                for b in &self.best {
                    c.push(b.clone().unwrap_or(Value::Float(f64::NAN))).unwrap();
                }
                c
            }
        }
    }
}

/// Replace group-by expressions and aggregates with references to the aggregate output.
fn rewrite(e: &Expr, group_by: &[Expr], aggs: &mut Vec<Expr>) -> Result<Expr> {
    if let Some(i) = group_by.iter().position(|g| g == e) {
        return Ok(Expr::Col(format!("#g{i}")));
    }
    Ok(match e {
        Expr::Agg(..) => {
            let j = aggs.iter().position(|a| a == e).unwrap_or_else(|| {
                aggs.push(e.clone());
                aggs.len() - 1
            });
            Expr::Col(format!("#a{j}"))
        }
        Expr::Col(c) => {
            return Err(format!(
                "column {c:?} must appear in GROUP BY or inside an aggregate"
            ));
        }
        Expr::Lit(_) => e.clone(),
        Expr::Bin(a, op, b) => Expr::bin(
            rewrite(a, group_by, aggs)?,
            *op,
            rewrite(b, group_by, aggs)?,
        ),
        Expr::Not(a) => Expr::Not(Box::new(rewrite(a, group_by, aggs)?)),
        Expr::Neg(a) => Expr::Neg(Box::new(rewrite(a, group_by, aggs)?)),
    })
}

// ---- sort -----------------------------------------------------------------------------------

fn cmp_at(c: &Column, i: usize, j: usize) -> Ordering {
    match c {
        Column::Int(v) => v[i].cmp(&v[j]),
        Column::Float(v) => v[i].total_cmp(&v[j]),
        Column::Str(v) => v[i].cmp(&v[j]),
    }
}

fn sort_indices(keys: &[(Column, bool)], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        for (c, desc) in keys {
            let o = cmp_at(c, i, j);
            if o != Ordering::Equal {
                return if *desc { o.reverse() } else { o };
            }
        }
        Ordering::Equal
    });
    idx
}

// ---- the pipeline ---------------------------------------------------------------------------

fn append(dst: &mut Batch, src: Batch) -> Result<()> {
    if dst.names.is_empty() {
        *dst = src;
        return Ok(());
    }
    for (d, s) in dst.columns.iter_mut().zip(src.columns) {
        d.append(s)?;
    }
    Ok(())
}

fn scan_all(s: &plan::Scan, stats: &mut Stats) -> Result<Batch> {
    let mut t = Table::open(&s.path)?;
    let cols: Vec<usize> = s
        .columns
        .iter()
        .map(|c| t.column_index(c))
        .collect::<Result<_>>()?;
    let mut out = Batch {
        names: s.columns.clone(),
        columns: cols
            .iter()
            .map(|&c| Column::empty(t.footer.types[c]))
            .collect(),
    };
    stats.row_groups_total += s.total_row_groups;
    for &rg in &s.row_groups {
        let b = t.read_group(rg, &cols)?;
        stats.row_groups_read += 1;
        stats.rows_scanned += b.rows();
        append(&mut out, filter_batch(b, &s.filters)?)?;
    }
    Ok(out)
}

struct HashSide {
    batch: Batch,
    probe_col: String,
    table: HashMap<K, Vec<u32>>,
}

fn probe(b: Batch, h: &HashSide) -> Result<Batch> {
    let pc = &b.columns[b.index(&h.probe_col).ok_or("join key missing")?];
    let (mut li, mut ri) = (vec![], vec![]);
    for i in 0..b.rows() {
        if let Some(rows) = h.table.get(&key_at(pc, i)) {
            for &r in rows {
                li.push(i);
                ri.push(r as usize);
            }
        }
    }
    let mut out = Batch::default();
    for (n, c) in b.names.iter().zip(&b.columns) {
        out.names.push(n.clone());
        out.columns.push(c.take(&li));
    }
    for (n, c) in h.batch.names.iter().zip(&h.batch.columns) {
        out.names.push(n.clone());
        out.columns.push(c.take(&ri));
    }
    Ok(out)
}

pub fn run(plan: Plan) -> Result<(Batch, Stats)> {
    let mut stats = Stats::default();
    let q = &plan.query;

    // Build one hash table per join.
    let mut sides = vec![];
    for (scan, (probe_col, build_col)) in plan.scans[1..].iter().zip(&plan.join_keys) {
        let batch = scan_all(scan, &mut stats)?;
        let kc = &batch.columns[batch.index(build_col).unwrap()];
        let mut table: HashMap<K, Vec<u32>> = HashMap::new();
        for i in 0..batch.rows() {
            table.entry(key_at(kc, i)).or_default().push(i as u32);
        }
        sides.push(HashSide {
            batch,
            probe_col: probe_col.clone(),
            table,
        });
    }
    // Join key types must agree, or every probe silently misses.
    for (s, (p, b)) in sides.iter().zip(&plan.join_keys) {
        let bt = s.batch.columns[s.batch.index(b).unwrap()].dtype();
        let owner = plan.scans.iter().find(|sc| sc.columns.contains(p)).unwrap();
        let t = Table::open(&owner.path)?;
        let pt = t.footer.types[t.column_index(p)?];
        if pt != bt {
            return Err(format!(
                "join key types differ: {p} is {pt:?}, {b} is {bt:?}"
            ));
        }
    }

    let items: Vec<(Expr, String)> = match &q.items {
        Some(i) => i.clone(),
        None => plan
            .scans
            .iter()
            .flat_map(|s| s.columns.iter().map(|c| (Expr::Col(c.clone()), c.clone())))
            .collect(),
    };
    // ORDER BY may name an output alias.
    let order: Vec<(Expr, bool)> = q
        .order_by
        .iter()
        .map(|(e, d)| match e {
            Expr::Col(c) => (
                items
                    .iter()
                    .find(|(_, a)| a == c)
                    .map_or(e.clone(), |(ie, _)| ie.clone()),
                *d,
            ),
            _ => (e.clone(), *d),
        })
        .collect();
    let aggregate = !q.group_by.is_empty() || items.iter().any(|(e, _)| e.has_agg());

    let mut agg_exprs = vec![];
    let (out_items, out_order) = if aggregate {
        let it: Vec<Expr> = items
            .iter()
            .map(|(e, _)| rewrite(e, &q.group_by, &mut agg_exprs))
            .collect::<Result<_>>()?;
        let or: Vec<Expr> = order
            .iter()
            .map(|(e, _)| rewrite(e, &q.group_by, &mut agg_exprs))
            .collect::<Result<_>>()?;
        (it, or)
    } else {
        (
            items.iter().map(|(e, _)| e.clone()).collect(),
            order.iter().map(|(e, _)| e.clone()).collect(),
        )
    };
    let mut aggs: Vec<Agg> = agg_exprs
        .iter()
        .map(|e| match e {
            Expr::Agg(f, a) => Agg {
                f: *f,
                arg: a.as_deref().cloned(),
                dtype: None,
                count: vec![],
                isum: vec![],
                fsum: vec![],
                best: vec![],
            },
            _ => unreachable!(),
        })
        .collect();
    let mut groups: HashMap<Vec<K>, usize> = HashMap::new();
    let mut group_cols: Vec<Column> = vec![];
    if aggregate && q.group_by.is_empty() {
        groups.insert(vec![], 0);
        aggs.iter_mut().for_each(Agg::grow);
    }

    // Project mode keeps the source columns the sort needs next to the output.
    let mut collected = Batch::default();
    let early_stop = q.order_by.is_empty() && !aggregate;

    let scan0 = &plan.scans[0];
    let mut t = Table::open(&scan0.path)?;
    let cols: Vec<usize> = scan0
        .columns
        .iter()
        .map(|c| t.column_index(c))
        .collect::<Result<_>>()?;
    stats.row_groups_total += scan0.total_row_groups;
    for &rg in &scan0.row_groups {
        let mut b = t.read_group(rg, &cols)?;
        stats.row_groups_read += 1;
        stats.rows_scanned += b.rows();
        b = filter_batch(b, &scan0.filters)?;
        for s in &sides {
            b = probe(b, s)?;
        }
        b = filter_batch(b, &plan.residual)?;
        if aggregate {
            let keys: Vec<Column> = q
                .group_by
                .iter()
                .map(|g| eval_column(g, &b))
                .collect::<Result<_>>()?;
            if group_cols.is_empty() {
                group_cols = keys.iter().map(|k| Column::empty(k.dtype())).collect();
            }
            let mut ids = Vec::with_capacity(b.rows());
            for i in 0..b.rows() {
                let key: Vec<K> = keys.iter().map(|k| key_at(k, i)).collect();
                let next = groups.len();
                let id = *groups.entry(key).or_insert_with(|| {
                    for (gc, k) in group_cols.iter_mut().zip(&keys) {
                        gc.push(k.get(i)).unwrap();
                    }
                    aggs.iter_mut().for_each(Agg::grow);
                    next
                });
                ids.push(id);
            }
            for a in &mut aggs {
                a.update(&ids, &b)?;
            }
        } else {
            let mut part = Batch::default();
            for (k, e) in out_items.iter().chain(&out_order).enumerate() {
                part.names.push(format!("#{k}"));
                part.columns.push(eval_column(e, &b)?);
            }
            append(&mut collected, part)?;
            if early_stop && q.limit.is_some_and(|l| collected.rows() >= l) {
                break;
            }
        }
    }

    let source = if aggregate {
        let mut src = Batch::default();
        for (i, c) in group_cols.into_iter().enumerate() {
            src.names.push(format!("#g{i}"));
            src.columns.push(c);
        }
        for (j, a) in aggs.iter().enumerate() {
            src.names.push(format!("#a{j}"));
            src.columns.push(a.finish());
        }
        let mut b = Batch::default();
        for (k, e) in out_items.iter().chain(&out_order).enumerate() {
            b.names.push(format!("#{k}"));
            b.columns.push(
                // GROUP BY over zero rows (e.g. every row group pruned): no groups, no key
                // columns were ever typed, so the answer is simply empty.
                if !q.group_by.is_empty() && groups.is_empty() {
                    Column::empty(DataType::Int)
                } else {
                    eval_column(e, &src)?
                },
            );
        }
        b
    } else {
        collected
    };

    let n = source.rows();
    let ni = out_items.len();
    let mut idx: Vec<usize> = if out_order.is_empty() {
        (0..n).collect()
    } else {
        let keys: Vec<(Column, bool)> = order
            .iter()
            .enumerate()
            .map(|(k, (_, d))| (source.columns[ni + k].clone(), *d))
            .collect();
        sort_indices(&keys, n)
    };
    if let Some(l) = q.limit {
        idx.truncate(l);
    }
    let mut out = Batch::default();
    for (k, (_, alias)) in items.iter().enumerate() {
        out.names.push(alias.clone());
        out.columns.push(match source.columns.get(k) {
            Some(c) => c.take(&idx),
            None => Column::Int(vec![]), // no rows reached a projection
        });
    }
    Ok((out, stats))
}
