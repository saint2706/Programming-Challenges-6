//! Calculated fields: a small arithmetic language over the other measures of a pivot, evaluated
//! once per cell (including subtotals and grand totals) after aggregation, as a spreadsheet does.
//!
//! `revenue / orders`, `([Total Sales] - cost) / [Total Sales] * 100`, `if(orders > 10, revenue, 0)`,
//! `round(a / b, 2)`. A name is letters, digits and `_`, or anything in square brackets.
//! A result is `None` (an empty cell) when an input is missing or the arithmetic is undefined
//! (division by zero, the square root of a negative number).
//!
//! `Expr::eval` is a plain tree walk over a parsed formula with a fixed set of functions: it runs
//! no code from the formula, only arithmetic, so a formula typed into the page is safe to evaluate.

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Op(Op),
    LParen,
    RParen,
    Comma,
}

fn lex(src: &str) -> Result<Vec<(Tok, usize)>> {
    let b: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let at = i;
        match c {
            ' ' | '\t' | '\n' => {
                i += 1;
                continue;
            }
            '(' => out.push((Tok::LParen, at)),
            ')' => out.push((Tok::RParen, at)),
            ',' => out.push((Tok::Comma, at)),
            '+' => out.push((Tok::Op(Op::Add), at)),
            '-' => out.push((Tok::Op(Op::Sub), at)),
            '*' => out.push((Tok::Op(Op::Mul), at)),
            '/' => out.push((Tok::Op(Op::Div), at)),
            '^' => out.push((Tok::Op(Op::Pow), at)),
            '<' | '>' | '=' | '!' => {
                let two = b.get(i + 1) == Some(&'=');
                let op = match (c, two) {
                    ('<', true) => Op::Le,
                    ('<', false) => Op::Lt,
                    ('>', true) => Op::Ge,
                    ('>', false) => Op::Gt,
                    ('=', _) => Op::Eq,
                    ('!', true) => Op::Ne,
                    _ => {
                        return Err(Error::new(format!(
                            "formula: unexpected '!' at position {}",
                            at + 1
                        )));
                    }
                };
                out.push((Tok::Op(op), at));
                if two || (c == '=' && b.get(i + 1) == Some(&'=')) {
                    i += 1;
                }
            }
            '[' => {
                let end = b[i + 1..].iter().position(|&x| x == ']').ok_or_else(|| {
                    Error::new(format!(
                        "formula: '[' at position {} is never closed",
                        at + 1
                    ))
                })?;
                let name: String = b[i + 1..i + 1 + end].iter().collect();
                out.push((Tok::Name(name), at));
                i += end + 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                    i += 1;
                }
                if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                    let mut j = i + 1;
                    if j < b.len() && (b[j] == '+' || b[j] == '-') {
                        j += 1;
                    }
                    if j < b.len() && b[j].is_ascii_digit() {
                        while j < b.len() && b[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                }
                let text: String = b[start..i].iter().collect();
                let n = text.parse::<f64>().map_err(|_| {
                    Error::new(format!(
                        "formula: bad number '{text}' at position {}",
                        start + 1
                    ))
                })?;
                out.push((Tok::Num(n), start));
                continue;
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                    i += 1;
                }
                out.push((Tok::Name(b[start..i].iter().collect()), start));
                continue;
            }
            other => {
                return Err(Error::new(format!(
                    "formula: unexpected '{other}' at position {}",
                    at + 1
                )));
            }
        }
        i += 1;
    }
    Ok(out)
}

struct Parser {
    toks: Vec<(Tok, usize)>,
    pos: usize,
}

fn binding(op: Op) -> (u8, bool) {
    // (precedence, right-associative)
    match op {
        Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::Eq | Op::Ne => (1, false),
        Op::Add | Op::Sub => (2, false),
        Op::Mul | Op::Div => (3, false),
        Op::Pow => (5, true),
    }
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(t, _)| t)
    }

    fn here(&self) -> usize {
        self.toks
            .get(self.pos)
            .map(|(_, p)| p + 1)
            .unwrap_or_else(|| self.toks.last().map(|(_, p)| p + 2).unwrap_or(1))
    }

    fn expr(&mut self, min_prec: u8) -> Result<Expr> {
        let mut lhs = self.unary()?;
        while let Some(Tok::Op(op)) = self.peek().cloned() {
            let (prec, right) = binding(op);
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let rhs = self.expr(if right { prec } else { prec + 1 })?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Some(Tok::Op(Op::Sub)) => {
                self.pos += 1;
                // binds looser than ^ so that -2^2 is -(2^2), as in a spreadsheet's math
                Ok(Expr::Neg(Box::new(self.expr(4)?)))
            }
            Some(Tok::Op(Op::Add)) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.atom(),
        }
    }

    fn atom(&mut self) -> Result<Expr> {
        let at = self.here();
        match self.toks.get(self.pos).cloned() {
            Some((Tok::Num(n), _)) => {
                self.pos += 1;
                Ok(Expr::Num(n))
            }
            Some((Tok::Name(name), _)) => {
                self.pos += 1;
                if self.peek() == Some(&Tok::LParen) {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RParen) {
                        loop {
                            args.push(self.expr(1)?);
                            match self.peek() {
                                Some(Tok::Comma) => self.pos += 1,
                                _ => break,
                            }
                        }
                    }
                    if self.peek() != Some(&Tok::RParen) {
                        return Err(Error::new(format!(
                            "formula: expected ')' at position {}",
                            self.here()
                        )));
                    }
                    self.pos += 1;
                    check_call(&name, args.len(), at)?;
                    Ok(Expr::Call(name.to_lowercase(), args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Some((Tok::LParen, _)) => {
                self.pos += 1;
                let inner = self.expr(1)?;
                if self.peek() != Some(&Tok::RParen) {
                    return Err(Error::new(format!(
                        "formula: expected ')' at position {}",
                        self.here()
                    )));
                }
                self.pos += 1;
                Ok(inner)
            }
            Some((tok, _)) => Err(Error::new(format!(
                "formula: unexpected {} at position {at}",
                describe(&tok)
            ))),
            None => Err(Error::new("formula: it ends where a value was expected")),
        }
    }
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Num(n) => format!("number {n}"),
        Tok::Name(n) => format!("name '{n}'"),
        Tok::Op(_) => "an operator".into(),
        Tok::LParen => "'('".into(),
        Tok::RParen => "')'".into(),
        Tok::Comma => "','".into(),
    }
}

fn check_call(name: &str, n: usize, at: usize) -> Result<()> {
    let ok = match name.to_lowercase().as_str() {
        "abs" | "sqrt" | "ln" | "log10" | "exp" | "floor" | "ceil" => n == 1,
        "round" => n == 1 || n == 2,
        "min" | "max" => n >= 1,
        "if" => n == 3,
        _ => {
            return Err(Error::new(format!(
                "formula: unknown function '{name}' at position {at}"
            )));
        }
    };
    if ok {
        Ok(())
    } else {
        Err(Error::new(format!(
            "formula: wrong number of arguments to {name} at position {at}"
        )))
    }
}

pub fn parse(src: &str) -> Result<Expr> {
    let toks = lex(src)?;
    if toks.is_empty() {
        return Err(Error::new("formula: it is empty"));
    }
    let mut p = Parser { toks, pos: 0 };
    let e = p.expr(1)?;
    if p.pos != p.toks.len() {
        return Err(Error::new(format!(
            "formula: unexpected {} at position {}",
            describe(&p.toks[p.pos].0),
            p.here()
        )));
    }
    Ok(e)
}

impl Expr {
    /// Every name the formula reads, in order of first use.
    pub fn vars(&self) -> Vec<String> {
        fn walk(e: &Expr, out: &mut Vec<String>) {
            match e {
                Expr::Num(_) => {}
                Expr::Var(v) => {
                    if !out.contains(v) {
                        out.push(v.clone());
                    }
                }
                Expr::Neg(a) => walk(a, out),
                Expr::Bin(_, a, b) => {
                    walk(a, out);
                    walk(b, out);
                }
                Expr::Call(_, args) => args.iter().for_each(|a| walk(a, out)),
            }
        }
        let mut out = Vec::new();
        walk(self, &mut out);
        out
    }

    /// `env` gives a name's value (`None` = missing). The result is `None` if undefined.
    pub fn eval(&self, env: &dyn Fn(&str) -> Option<f64>) -> Option<f64> {
        let v = match self {
            Expr::Num(n) => *n,
            Expr::Var(name) => env(name)?,
            Expr::Neg(a) => -a.eval(env)?,
            Expr::Bin(op, a, b) => {
                let (x, y) = (a.eval(env)?, b.eval(env)?);
                match op {
                    Op::Add => x + y,
                    Op::Sub => x - y,
                    Op::Mul => x * y,
                    Op::Div => {
                        if y == 0.0 {
                            return None;
                        }
                        x / y
                    }
                    Op::Pow => x.powf(y),
                    Op::Lt => f64::from(x < y),
                    Op::Le => f64::from(x <= y),
                    Op::Gt => f64::from(x > y),
                    Op::Ge => f64::from(x >= y),
                    Op::Eq => f64::from(x == y),
                    Op::Ne => f64::from(x != y),
                }
            }
            Expr::Call(name, args) => {
                if name == "if" {
                    let cond = args[0].eval(env)?;
                    return if cond != 0.0 {
                        args[1].eval(env)
                    } else {
                        args[2].eval(env)
                    };
                }
                let vals: Option<Vec<f64>> = args.iter().map(|a| a.eval(env)).collect();
                let vals = vals?;
                match name.as_str() {
                    "abs" => vals[0].abs(),
                    "sqrt" => {
                        if vals[0] < 0.0 {
                            return None;
                        }
                        vals[0].sqrt()
                    }
                    "ln" => {
                        if vals[0] <= 0.0 {
                            return None;
                        }
                        vals[0].ln()
                    }
                    "log10" => {
                        if vals[0] <= 0.0 {
                            return None;
                        }
                        vals[0].log10()
                    }
                    "exp" => vals[0].exp(),
                    "floor" => vals[0].floor(),
                    "ceil" => vals[0].ceil(),
                    "round" => {
                        let digits = vals.get(1).copied().unwrap_or(0.0);
                        let m = 10f64.powi(digits as i32);
                        (vals[0] * m).round() / m
                    }
                    "min" => vals.iter().copied().fold(f64::INFINITY, f64::min),
                    "max" => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    _ => return None,
                }
            }
        };
        v.is_finite().then_some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(src: &str) -> Option<f64> {
        parse(src).unwrap().eval(&|n| match n {
            "a" => Some(6.0),
            "b" => Some(3.0),
            "zero" => Some(0.0),
            "Total Sales" => Some(200.0),
            _ => None,
        })
    }

    #[test]
    fn precedence_and_associativity() {
        assert_eq!(ev("1 + 2 * 3"), Some(7.0));
        assert_eq!(ev("(1 + 2) * 3"), Some(9.0));
        assert_eq!(ev("10 - 4 - 3"), Some(3.0));
        assert_eq!(ev("2 ^ 3 ^ 2"), Some(512.0));
        assert_eq!(ev("-2 ^ 2"), Some(-4.0));
        assert_eq!(ev("8 / 4 / 2"), Some(1.0));
        assert_eq!(ev("--3"), Some(3.0));
        assert_eq!(ev("2 * -3"), Some(-6.0));
    }

    #[test]
    fn names_brackets_and_functions() {
        assert_eq!(ev("a / b"), Some(2.0));
        assert_eq!(ev("[Total Sales] / 4"), Some(50.0));
        assert_eq!(ev("round(a / 7, 2)"), Some(0.86));
        assert_eq!(ev("max(a, b, 10)"), Some(10.0));
        assert_eq!(ev("MIN(a, b)"), Some(3.0));
        assert_eq!(
            ev("abs(-5) + sqrt(16) + floor(1.9) + ceil(1.1)"),
            Some(12.0)
        );
        assert_eq!(ev("round(2.5)"), Some(3.0));
        assert_eq!(ev("exp(0) + ln(1) + log10(1000)"), Some(4.0));
    }

    #[test]
    fn comparisons_and_if() {
        assert_eq!(ev("a > b"), Some(1.0));
        assert_eq!(ev("a == b"), Some(0.0));
        assert_eq!(ev("a != b"), Some(1.0));
        assert_eq!(ev("a <= 6"), Some(1.0));
        assert_eq!(ev("if(a > b, a, b)"), Some(6.0));
        assert_eq!(ev("if(a < b, 1 / zero, 5)"), Some(5.0)); // the untaken branch is never evaluated
    }

    #[test]
    fn undefined_arithmetic_is_an_empty_cell_not_an_error() {
        assert_eq!(ev("a / zero"), None);
        assert_eq!(ev("sqrt(-1)"), None);
        assert_eq!(ev("ln(0)"), None);
        assert_eq!(ev("missing + 1"), None);
        assert_eq!(ev("10 ^ 1000"), None);
    }

    #[test]
    fn vars_in_order_of_first_use() {
        let e = parse("b / a + b * [Total Sales]").unwrap();
        assert_eq!(e.vars(), ["b", "a", "Total Sales"]);
    }

    #[test]
    fn syntax_errors_say_where() {
        for (src, needle) in [
            ("", "empty"),
            ("1 +", "ends where a value was expected"),
            ("(1 + 2", "expected ')'"),
            ("1 2", "unexpected number 2"),
            ("a $ b", "unexpected '$' at position 3"),
            ("foo(1)", "unknown function 'foo'"),
            ("round()", "wrong number of arguments"),
            ("[open", "never closed"),
            ("if(1,2)", "wrong number of arguments"),
            ("1 ! 2", "unexpected '!'"),
            ("* 2", "unexpected an operator at position 1"),
        ] {
            let e = parse(src).unwrap_err().to_string();
            assert!(e.contains(needle), "{src:?} -> {e}");
        }
    }

    #[test]
    fn scientific_notation() {
        assert_eq!(ev("1e3 + 2.5e-1"), Some(1000.25));
    }
}
