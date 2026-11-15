//! Tokenizer and recursive-descent parser for:
//!
//! ```text
//! [EXPLAIN] SELECT item [, ...] FROM t [JOIN u ON a = b ...] [WHERE e] [GROUP BY e, ...]
//!   [ORDER BY e [ASC|DESC], ...] [LIMIT n]
//! item := * | expr [[AS] alias]
//! expr := OR / AND / NOT / comparison (= <> != < <= > >=, BETWEEN) / + - / * / unary - /
//!         literal | column | t.column | COUNT(*) | COUNT|SUM|AVG|MIN|MAX(expr) | (expr)
//! ```

use std::fmt;

use crate::Result;
use crate::types::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        use BinOp::*;
        match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Eq => "=",
            Ne => "<>",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "AND",
            Or => "OR",
        }
    }
    /// `a op b` == `b op.flip() a`, for moving the literal to the right-hand side.
    pub fn flip(self) -> Self {
        use BinOp::*;
        match self {
            Lt => Gt,
            Le => Ge,
            Gt => Lt,
            Ge => Le,
            o => o,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Col(String),
    Lit(Value),
    Bin(Box<Expr>, BinOp, Box<Expr>),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    /// `None` argument means `COUNT(*)`.
    Agg(AggFn, Option<Box<Expr>>),
}

impl Expr {
    pub fn bin(a: Expr, op: BinOp, b: Expr) -> Expr {
        Expr::Bin(Box::new(a), op, Box::new(b))
    }
    /// Apply `f` to every column name in place (used to resolve `table.col` qualifiers).
    pub fn map_columns(
        &mut self,
        f: &mut impl FnMut(&mut String) -> crate::Result<()>,
    ) -> crate::Result<()> {
        match self {
            Expr::Col(c) => f(c),
            Expr::Bin(a, _, b) => {
                a.map_columns(f)?;
                b.map_columns(f)
            }
            Expr::Not(a) | Expr::Neg(a) | Expr::Agg(_, Some(a)) => a.map_columns(f),
            _ => Ok(()),
        }
    }
    pub fn has_agg(&self) -> bool {
        match self {
            Expr::Agg(..) => true,
            Expr::Bin(a, _, b) => a.has_agg() || b.has_agg(),
            Expr::Not(a) | Expr::Neg(a) => a.has_agg(),
            _ => false,
        }
    }
    /// Every column the expression reads.
    pub fn columns(&self, out: &mut Vec<String>) {
        match self {
            Expr::Col(c) if !out.contains(c) => out.push(c.clone()),
            Expr::Bin(a, _, b) => {
                a.columns(out);
                b.columns(out);
            }
            Expr::Not(a) | Expr::Neg(a) => a.columns(out),
            Expr::Agg(_, Some(a)) => a.columns(out),
            _ => {}
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Col(c) => write!(f, "{c}"),
            Expr::Lit(Value::Str(s)) => write!(f, "'{}'", s.replace('\'', "''")),
            Expr::Lit(v) => write!(f, "{v}"),
            Expr::Bin(a, op, b) => write!(f, "({a} {} {b})", op.symbol()),
            Expr::Not(a) => write!(f, "NOT {a}"),
            Expr::Neg(a) => write!(f, "-{a}"),
            Expr::Agg(_, None) => write!(f, "COUNT(*)"),
            Expr::Agg(func, Some(a)) => write!(f, "{}({a})", format!("{func:?}").to_uppercase()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Select {
    pub explain: bool,
    /// `None` is `SELECT *`.
    pub items: Option<Vec<(Expr, String)>>,
    pub from: String,
    pub joins: Vec<(String, String, String)>, // (table, left column, right column)
    pub filter: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub order_by: Vec<(Expr, bool)>, // (key, descending)
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Kw(String),
    Num(String),
    Str(String),
    Sym(&'static str),
}

const KEYWORDS: &[&str] = &[
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "ASC", "DESC", "LIMIT", "AND", "OR", "NOT",
    "AS", "JOIN", "INNER", "ON", "BETWEEN", "EXPLAIN",
];
const SYMS: &[&str] = &[
    "<=", ">=", "<>", "!=", "(", ")", ",", "*", "+", "-", "/", "=", "<", ">", ".", ";",
];

fn lex(s: &str) -> Result<Vec<Tok>> {
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = vec![];
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '\'' {
            let mut v = String::new();
            i += 1;
            loop {
                match cs.get(i) {
                    None => return Err("unterminated string literal".into()),
                    Some('\'') if cs.get(i + 1) == Some(&'\'') => {
                        v.push('\'');
                        i += 2;
                    }
                    Some('\'') => {
                        i += 1;
                        break;
                    }
                    Some(&ch) => {
                        v.push(ch);
                        i += 1;
                    }
                }
            }
            out.push(Tok::Str(v));
        } else if c.is_ascii_digit() {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.') {
                i += 1;
            }
            out.push(Tok::Num(cs[st..i].iter().collect()));
        } else if c.is_alphabetic() || c == '_' || c == '"' {
            let quoted = c == '"';
            let st = if quoted { i + 1 } else { i };
            i = st;
            while i < cs.len()
                && (if quoted {
                    cs[i] != '"'
                } else {
                    cs[i].is_alphanumeric() || cs[i] == '_'
                })
            {
                i += 1;
            }
            let w: String = cs[st..i].iter().collect();
            if quoted {
                if i >= cs.len() {
                    return Err("unterminated quoted identifier".into());
                }
                i += 1;
                out.push(Tok::Ident(w));
            } else if KEYWORDS.contains(&w.to_uppercase().as_str()) {
                out.push(Tok::Kw(w.to_uppercase()));
            } else {
                out.push(Tok::Ident(w.to_lowercase()));
            }
        } else if let Some(sym) = SYMS
            .iter()
            .find(|s| cs[i..].starts_with(&s.chars().collect::<Vec<_>>()))
        {
            out.push(Tok::Sym(sym));
            i += sym.len();
        } else {
            return Err(format!("unexpected character {c:?}"));
        }
    }
    Ok(out)
}

struct Parser {
    t: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn kw(&mut self, k: &str) -> bool {
        if self.peek() == Some(&Tok::Kw(k.into())) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn sym(&mut self, s: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(x)) if *x == s) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect_kw(&mut self, k: &str) -> Result<()> {
        if self.kw(k) {
            Ok(())
        } else {
            Err(format!("expected {k} near {:?}", self.peek()))
        }
    }
    fn expect_sym(&mut self, s: &str) -> Result<()> {
        if self.sym(s) {
            Ok(())
        } else {
            Err(format!("expected {s:?} near {:?}", self.peek()))
        }
    }
    fn ident(&mut self) -> Result<String> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => {
                self.i += 1;
                Ok(s)
            }
            t => Err(format!("expected a name, found {t:?}")),
        }
    }
    /// `col` or `table.col`; the planner checks the qualifier against the column's table.
    fn column(&mut self) -> Result<String> {
        let a = self.ident()?;
        if self.sym(".") {
            Ok(format!("{a}.{}", self.ident()?))
        } else {
            Ok(a)
        }
    }

    fn expr(&mut self) -> Result<Expr> {
        let mut e = self.and()?;
        while self.kw("OR") {
            e = Expr::bin(e, BinOp::Or, self.and()?);
        }
        Ok(e)
    }
    fn and(&mut self) -> Result<Expr> {
        let mut e = self.not()?;
        while self.kw("AND") {
            e = Expr::bin(e, BinOp::And, self.not()?);
        }
        Ok(e)
    }
    fn not(&mut self) -> Result<Expr> {
        if self.kw("NOT") {
            Ok(Expr::Not(Box::new(self.not()?)))
        } else {
            self.cmp()
        }
    }
    fn cmp(&mut self) -> Result<Expr> {
        let a = self.add()?;
        if self.kw("BETWEEN") {
            let lo = self.add()?;
            self.expect_kw("AND")?;
            let hi = self.add()?;
            return Ok(Expr::bin(
                Expr::bin(a.clone(), BinOp::Ge, lo),
                BinOp::And,
                Expr::bin(a, BinOp::Le, hi),
            ));
        }
        let op = match self.peek() {
            Some(Tok::Sym("=")) => BinOp::Eq,
            Some(Tok::Sym("<>" | "!=")) => BinOp::Ne,
            Some(Tok::Sym("<")) => BinOp::Lt,
            Some(Tok::Sym("<=")) => BinOp::Le,
            Some(Tok::Sym(">")) => BinOp::Gt,
            Some(Tok::Sym(">=")) => BinOp::Ge,
            _ => return Ok(a),
        };
        self.i += 1;
        Ok(Expr::bin(a, op, self.add()?))
    }
    fn add(&mut self) -> Result<Expr> {
        let mut e = self.mul()?;
        loop {
            let op = if self.sym("+") {
                BinOp::Add
            } else if self.sym("-") {
                BinOp::Sub
            } else {
                return Ok(e);
            };
            e = Expr::bin(e, op, self.mul()?);
        }
    }
    fn mul(&mut self) -> Result<Expr> {
        let mut e = self.unary()?;
        loop {
            let op = if self.sym("*") {
                BinOp::Mul
            } else if self.sym("/") {
                BinOp::Div
            } else {
                return Ok(e);
            };
            e = Expr::bin(e, op, self.unary()?);
        }
    }
    fn unary(&mut self) -> Result<Expr> {
        if self.sym("-") {
            return Ok(match self.unary()? {
                Expr::Lit(Value::Int(v)) => Expr::Lit(Value::Int(-v)),
                Expr::Lit(Value::Float(v)) => Expr::Lit(Value::Float(-v)),
                e => Expr::Neg(Box::new(e)),
            });
        }
        self.primary()
    }
    fn primary(&mut self) -> Result<Expr> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.i += 1;
                if let Ok(v) = n.parse::<i64>() {
                    Ok(Expr::Lit(Value::Int(v)))
                } else {
                    n.parse::<f64>()
                        .map(|v| Expr::Lit(Value::Float(v)))
                        .map_err(|_| format!("bad number {n}"))
                }
            }
            Some(Tok::Str(s)) => {
                self.i += 1;
                Ok(Expr::Lit(Value::Str(s)))
            }
            Some(Tok::Sym("(")) => {
                self.i += 1;
                let e = self.expr()?;
                self.expect_sym(")")?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                let f = match name.as_str() {
                    "count" => Some(AggFn::Count),
                    "sum" => Some(AggFn::Sum),
                    "avg" => Some(AggFn::Avg),
                    "min" => Some(AggFn::Min),
                    "max" => Some(AggFn::Max),
                    _ => None,
                };
                if let (Some(f), Some(Tok::Sym("("))) = (f, self.t.get(self.i + 1)) {
                    self.i += 2;
                    let arg = if f == AggFn::Count && self.sym("*") {
                        None
                    } else {
                        let a = self.expr()?;
                        if a.has_agg() {
                            return Err("aggregates cannot be nested".into());
                        }
                        Some(Box::new(a))
                    };
                    self.expect_sym(")")?;
                    return Ok(Expr::Agg(f, arg));
                }
                Ok(Expr::Col(self.column()?))
            }
            t => Err(format!("expected an expression, found {t:?}")),
        }
    }
}

pub fn parse(sql: &str) -> Result<Select> {
    let mut p = Parser { t: lex(sql)?, i: 0 };
    let mut q = Select {
        explain: p.kw("EXPLAIN"),
        ..Default::default()
    };
    p.expect_kw("SELECT")?;
    if p.sym("*") {
        q.items = None;
    } else {
        let mut items = vec![];
        loop {
            let e = p.expr()?;
            let alias = if p.kw("AS") {
                p.ident()?
            } else if let Some(Tok::Ident(_)) = p.peek() {
                p.ident()?
            } else {
                match &e {
                    Expr::Col(c) => c.rsplit('.').next().unwrap_or(c).to_string(),
                    e => e.to_string(),
                }
            };
            items.push((e, alias));
            if !p.sym(",") {
                break;
            }
        }
        q.items = Some(items);
    }
    p.expect_kw("FROM")?;
    q.from = p.ident()?;
    loop {
        let inner = p.kw("INNER");
        if !p.kw("JOIN") {
            if inner {
                return Err("expected JOIN after INNER".into());
            }
            break;
        }
        let t = p.ident()?;
        p.expect_kw("ON")?;
        let a = p.column()?;
        p.expect_sym("=")?;
        let b = p.column()?;
        q.joins.push((t, a, b));
    }
    if p.kw("WHERE") {
        let f = p.expr()?;
        if f.has_agg() {
            return Err("aggregates are not allowed in WHERE".into());
        }
        q.filter = Some(f);
    }
    if p.kw("GROUP") {
        p.expect_kw("BY")?;
        loop {
            q.group_by.push(p.expr()?);
            if !p.sym(",") {
                break;
            }
        }
    }
    if p.kw("ORDER") {
        p.expect_kw("BY")?;
        loop {
            let e = p.expr()?;
            let desc = if p.kw("DESC") {
                true
            } else {
                p.kw("ASC");
                false
            };
            q.order_by.push((e, desc));
            if !p.sym(",") {
                break;
            }
        }
    }
    if p.kw("LIMIT") {
        match p.peek().cloned() {
            Some(Tok::Num(n)) => {
                p.i += 1;
                q.limit = Some(n.parse().map_err(|_| format!("bad LIMIT {n}"))?);
            }
            t => return Err(format!("expected a number after LIMIT, found {t:?}")),
        }
    }
    p.sym(";");
    if let Some(t) = p.peek() {
        return Err(format!("unexpected {t:?} after the end of the query"));
    }
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_and_between() {
        let q = parse("select a + b * 2 as x from t where c between 1 and 3 or not d = 'it''s'")
            .unwrap();
        let (e, alias) = &q.items.unwrap()[0];
        assert_eq!(alias, "x");
        assert_eq!(e.to_string(), "(a + (b * 2))");
        assert_eq!(
            q.filter.unwrap().to_string(),
            "(((c >= 1) AND (c <= 3)) OR NOT (d = 'it''s'))"
        );
    }

    #[test]
    fn full_shape() {
        let q = parse(
            "EXPLAIN SELECT g, count(*), sum(v) s FROM t JOIN u ON t.k = u.k2 \
             GROUP BY g ORDER BY s DESC, g LIMIT 5;",
        )
        .unwrap();
        assert!(q.explain);
        assert_eq!(q.joins, vec![("u".into(), "t.k".into(), "u.k2".into())]);
        assert!(q.order_by[0].1);
        assert_eq!(q.limit, Some(5));
        assert_eq!(q.items.unwrap()[2].1, "s");
    }

    #[test]
    fn errors() {
        for bad in [
            "select from t",
            "select a from",
            "select sum(sum(a)) from t",
            "select a from t where sum(a) > 1",
            "select a from t limit x",
            "select 'abc from t",
            "select a from t extra",
            "select a ? from t",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }
}
