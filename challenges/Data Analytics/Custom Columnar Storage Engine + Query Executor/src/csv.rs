//! CSV import: RFC 4180 quoting, a header row, and per-column type inference
//! (all integers -> Int, all numbers -> Float, anything else -> Str).
//!
//! `import` makes two passes over the text: one to infer types, one to fill and write each row
//! group, so memory holds one row group of values rather than the whole table.

use std::path::Path;

use crate::Result;
use crate::format::{Footer, Writer};
use crate::types::{Batch, Column, DataType, Value};

/// Split one record. `"a""b"` is a quoted field with an escaped quote.
fn fields(line: &str, delim: char) -> Result<Vec<String>> {
    let mut out = vec![];
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match (quoted, ch) {
            (true, '"') if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            (true, '"') => quoted = false,
            (false, '"') if cur.is_empty() => quoted = true,
            (false, c) if c == delim => out.push(std::mem::take(&mut cur)),
            (_, c) => cur.push(c),
        }
    }
    if quoted {
        return Err("unterminated quoted field".into());
    }
    out.push(cur);
    Ok(out)
}

/// Records of the text; quoted fields may span lines, so physical lines are joined until the
/// quotes balance. Blank lines are skipped.
fn records(text: &str, delim: char) -> impl Iterator<Item = Result<Vec<String>>> + '_ {
    let mut lines = text.lines();
    std::iter::from_fn(move || {
        let mut rec = String::new();
        loop {
            match lines.next() {
                None if rec.is_empty() => return None,
                None => return Some(Err("unterminated quoted field at end of input".into())),
                Some(l) => {
                    if !rec.is_empty() {
                        rec.push('\n');
                    }
                    rec.push_str(l);
                }
            }
            if rec.matches('"').count() % 2 == 0 {
                if rec.trim().is_empty() {
                    rec.clear();
                    continue;
                }
                return Some(fields(&rec, delim));
            }
        }
    })
}

fn infer(text: &str, delim: char) -> Result<(Vec<String>, Vec<DataType>)> {
    let mut it = records(text, delim);
    let names = it.next().ok_or("empty CSV")??;
    let mut types = vec![DataType::Int; names.len()];
    for (i, r) in it.enumerate() {
        let r = r?;
        if r.len() != names.len() {
            return Err(format!(
                "row {} has {} fields, header has {}",
                i + 2,
                r.len(),
                names.len()
            ));
        }
        for (t, cell) in types.iter_mut().zip(&r) {
            if *t == DataType::Int && cell.parse::<i64>().is_err() {
                *t = DataType::Float;
            }
            if *t == DataType::Float && cell.parse::<f64>().is_err() {
                *t = DataType::Str;
            }
        }
    }
    Ok((names, types))
}

fn value(cell: String, t: DataType) -> Value {
    match t {
        DataType::Int => Value::Int(cell.parse().unwrap()),
        DataType::Float => Value::Float(cell.parse().unwrap()),
        DataType::Str => Value::Str(cell),
    }
}

/// Rows of the text in batches of `chunk` rows, typed by the inference pass.
fn batches(
    text: &str,
    delim: char,
    chunk: usize,
) -> Result<(
    Vec<String>,
    Vec<DataType>,
    impl Iterator<Item = Result<Batch>> + '_,
)> {
    let (names, types) = infer(text, delim)?;
    let mut it = records(text, delim).skip(1).peekable();
    let (n2, t2) = (names.clone(), types.clone());
    let iter = std::iter::from_fn(move || {
        it.peek()?;
        let mut cols: Vec<Column> = t2.iter().map(|&t| Column::empty(t)).collect();
        for r in it.by_ref().take(chunk) {
            let r = match r {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            for ((c, cell), &t) in cols.iter_mut().zip(r).zip(&t2) {
                c.push(value(cell, t)).unwrap();
            }
        }
        Some(Ok(Batch {
            names: n2.clone(),
            columns: cols,
        }))
    });
    Ok((names, types, iter))
}

/// Parse a whole CSV into memory (small inputs and tests).
pub fn parse(text: &str, delim: char) -> Result<Batch> {
    let (names, types, it) = batches(text, delim, usize::MAX)?;
    match it.into_iter().next() {
        Some(b) => b,
        None => Ok(Batch {
            names,
            columns: types.into_iter().map(Column::empty).collect(),
        }),
    }
}

/// Stream a CSV into a `.col` file. Returns the row count and the footer.
pub fn import(text: &str, delim: char, out: &Path, row_group: usize) -> Result<(usize, Footer)> {
    if row_group == 0 {
        return Err("row group size must be positive".into());
    }
    let (names, types, it) = batches(text, delim, row_group)?;
    let mut w = Writer::create(out, names, types)?;
    let mut rows = 0;
    for b in it {
        let b = b?;
        rows += b.rows();
        w.write_group(&b)?;
    }
    Ok((rows, w.finish()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoting_and_inference() {
        let b = parse(
            "id,price,name\n1,2.5,\"a, \"\"b\"\"\"\n\n2,3,\"x\ny\"\n",
            ',',
        )
        .unwrap();
        assert_eq!(b.columns[0], Column::Int(vec![1, 2]));
        assert_eq!(b.columns[1], Column::Float(vec![2.5, 3.0]));
        assert_eq!(
            b.columns[2],
            Column::Str(vec!["a, \"b\"".into(), "x\ny".into()])
        );
    }

    #[test]
    fn ragged_rows_are_rejected() {
        assert!(parse("a,b\n1\n", ',').unwrap_err().contains("row 2"));
        assert!(parse("a\n\"x\n", ',').is_err());
        assert!(parse("", ',').is_err());
    }

    #[test]
    fn header_only_gives_empty_columns() {
        let b = parse("a|b\n", '|').unwrap();
        assert_eq!(b.names, ["a", "b"]);
        assert_eq!(b.rows(), 0);
    }

    #[test]
    fn import_streams_row_groups() {
        let mut text = String::from("k,v\n");
        for i in 0..2500 {
            text.push_str(&format!("{i},{}\n", i % 3));
        }
        let p = std::env::temp_dir().join(format!("colstore-csv-{}.col", std::process::id()));
        let (rows, footer) = import(&text, ',', &p, 1000).unwrap();
        assert_eq!(rows, 2500);
        assert_eq!(
            footer.row_groups.iter().map(|g| g.rows).collect::<Vec<_>>(),
            [1000, 1000, 500]
        );
        let mut t = crate::format::Table::open(&p).unwrap();
        assert_eq!(t.read_chunk(2, 0).unwrap().get(499), Value::Int(2499));
    }
}
