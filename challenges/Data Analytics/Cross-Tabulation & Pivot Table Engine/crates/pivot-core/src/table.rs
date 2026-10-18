//! Columnar storage with type inference: integers, floats, dates, datetimes, and strings that are
//! dictionary-encoded (every distinct text stored once, rows hold a `u32` code).

use serde::Serialize;

use crate::csv::{self, Arena};
use crate::date;
use crate::error::{Error, Result};

/// Code for a missing string.
pub const NULL: u32 = u32::MAX;

const NULL_TOKENS: [&str; 8] = ["", "NA", "N/A", "NaN", "nan", "NULL", "null", "None"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Int,
    Float,
    Date,
    DateTime,
    Str,
}

#[derive(Debug, Clone)]
pub enum Column {
    Int(Vec<Option<i64>>),
    /// `NaN` marks a missing value.
    Float(Vec<f64>),
    /// Days since 1970-01-01.
    Date(Vec<Option<i64>>),
    /// Seconds since 1970-01-01T00:00:00 UTC.
    DateTime(Vec<Option<i64>>),
    Str {
        dict: Vec<String>,
        codes: Vec<u32>,
    },
}

impl Column {
    pub fn kind(&self) -> Kind {
        match self {
            Column::Int(_) => Kind::Int,
            Column::Float(_) => Kind::Float,
            Column::Date(_) => Kind::Date,
            Column::DateTime(_) => Kind::DateTime,
            Column::Str { .. } => Kind::Str,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Column::Int(v) | Column::Date(v) | Column::DateTime(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::Str { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_null(&self, i: usize) -> bool {
        match self {
            Column::Int(v) | Column::Date(v) | Column::DateTime(v) => v[i].is_none(),
            Column::Float(v) => v[i].is_nan(),
            Column::Str { codes, .. } => codes[i] == NULL,
        }
    }

    /// The value as a number: integers, floats, and dates as days or seconds. `None` for text or missing.
    pub fn number(&self, i: usize) -> Option<f64> {
        match self {
            Column::Int(v) | Column::Date(v) | Column::DateTime(v) => v[i].map(|x| x as f64),
            Column::Float(v) => (!v[i].is_nan()).then_some(v[i]),
            Column::Str { .. } => None,
        }
    }

    pub fn is_numeric(&self) -> bool {
        !matches!(self, Column::Str { .. })
    }

    /// An identity for the value, so distinct counting works on any column type.
    pub fn identity(&self, i: usize) -> Option<u64> {
        match self {
            Column::Int(v) | Column::Date(v) | Column::DateTime(v) => v[i].map(|x| x as u64),
            Column::Float(v) => (!v[i].is_nan()).then(|| (v[i] + 0.0).to_bits()),
            Column::Str { codes, .. } => (codes[i] != NULL).then_some(codes[i] as u64),
        }
    }

    /// The value as text, as it would be written back to a CSV.
    pub fn text(&self, i: usize) -> Option<String> {
        match self {
            Column::Int(v) => v[i].map(|x| x.to_string()),
            Column::Float(v) => (!v[i].is_nan()).then(|| format_float(v[i])),
            Column::Date(v) => v[i].map(format_date),
            Column::DateTime(v) => v[i].map(format_datetime),
            Column::Str { dict, codes } => {
                (codes[i] != NULL).then(|| dict[codes[i] as usize].clone())
            }
        }
    }
}

pub fn format_datetime(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    format!(
        "{} {:02}:{:02}:{:02}",
        format_date(days),
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

pub fn format_float(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{x:.0}")
    } else {
        format!("{x}")
    }
}

pub fn format_date(days: i64) -> String {
    let (y, m, d) = date::civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    pub delimiter: Option<u8>,
    pub trim: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            delimiter: None,
            trim: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    pub names: Vec<String>,
    pub columns: Vec<Column>,
    pub rows: usize,
}

#[derive(Debug, Serialize)]
pub struct FieldInfo {
    pub name: String,
    pub kind: Kind,
    pub nulls: usize,
    /// Distinct values (text columns only; others are counted on demand).
    pub distinct: Option<usize>,
    pub min: Option<String>,
    pub max: Option<String>,
}

impl Table {
    pub fn from_csv(bytes: &[u8], options: LoadOptions) -> Result<Table> {
        let parsed = csv::parse(bytes, options.delimiter, options.trim)?;
        let columns = parsed.columns.iter().map(infer).collect();
        Ok(Table {
            names: parsed.header,
            columns,
            rows: parsed.rows,
        })
    }

    pub fn find(&self, name: &str) -> Result<(usize, &Column)> {
        match self.names.iter().position(|n| n == name) {
            Some(i) => Ok((i, &self.columns[i])),
            None => Err(Error::new(format!(
                "no field called '{name}'. Fields: {}",
                self.names.join(", ")
            ))),
        }
    }

    pub fn describe(&self) -> Vec<FieldInfo> {
        self.names
            .iter()
            .zip(&self.columns)
            .map(|(name, col)| {
                let nulls = (0..col.len()).filter(|&i| col.is_null(i)).count();
                let (min, max) = extent(col);
                FieldInfo {
                    name: name.clone(),
                    kind: col.kind(),
                    nulls,
                    distinct: match col {
                        Column::Str { dict, .. } => Some(dict.len()),
                        _ => None,
                    },
                    min,
                    max,
                }
            })
            .collect()
    }
}

fn extent(col: &Column) -> (Option<String>, Option<String>) {
    match col {
        Column::Str { .. } => (None, None),
        _ => {
            let (mut lo, mut hi): (Option<usize>, Option<usize>) = (None, None);
            for i in 0..col.len() {
                let Some(x) = col.number(i) else { continue };
                if lo.is_none_or(|j| x < col.number(j).unwrap_or(f64::INFINITY)) {
                    lo = Some(i);
                }
                if hi.is_none_or(|j| x > col.number(j).unwrap_or(f64::NEG_INFINITY)) {
                    hi = Some(i);
                }
            }
            (lo.and_then(|i| col.text(i)), hi.and_then(|i| col.text(i)))
        }
    }
}

fn is_null_token(s: &str) -> bool {
    NULL_TOKENS.contains(&s)
}

/// A number that is really an identifier ("02134", "007") stays text.
fn has_leading_zero(s: &str) -> bool {
    let t = s.strip_prefix(['-', '+']).unwrap_or(s);
    t.len() > 1 && t.starts_with('0') && t.as_bytes()[1].is_ascii_digit()
}

fn infer(arena: &Arena) -> Column {
    let n = arena.len();
    let cells = || (0..n).map(|i| arena.get(i)).filter(|c| !is_null_token(c));
    let any = cells().next().is_some();
    if any {
        if cells().all(|c| c.parse::<i64>().is_ok() && !has_leading_zero(c)) {
            return Column::Int(
                (0..n)
                    .map(|i| parse_if(arena.get(i), |c| c.parse().ok()))
                    .collect(),
            );
        }
        if cells().all(|c| c.parse::<f64>().is_ok_and(f64::is_finite) && !has_leading_zero(c)) {
            return Column::Float(
                (0..n)
                    .map(|i| parse_if(arena.get(i), |c| c.parse::<f64>().ok()).unwrap_or(f64::NAN))
                    .collect(),
            );
        }
        if cells().all(|c| date::parse_date(c).is_some()) {
            return Column::Date(
                (0..n)
                    .map(|i| parse_if(arena.get(i), date::parse_date))
                    .collect(),
            );
        }
        if cells().all(|c| date::parse_datetime(c).is_some()) {
            return Column::DateTime(
                (0..n)
                    .map(|i| parse_if(arena.get(i), date::parse_datetime))
                    .collect(),
            );
        }
    }
    let mut dict: Vec<String> = Vec::new();
    let mut index: crate::fxhash::FxMap<&str, u32> = crate::fxhash::FxMap::default();
    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let cell = arena.get(i);
        if is_null_token(cell) {
            codes.push(NULL);
            continue;
        }
        let code = *index.entry(cell).or_insert_with(|| {
            dict.push(cell.to_string());
            (dict.len() - 1) as u32
        });
        codes.push(code);
    }
    Column::Str { dict, codes }
}

fn parse_if<T>(cell: &str, f: impl Fn(&str) -> Option<T>) -> Option<T> {
    if is_null_token(cell) { None } else { f(cell) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(csv: &str) -> Table {
        Table::from_csv(csv.as_bytes(), LoadOptions::default()).unwrap()
    }

    #[test]
    fn infers_each_kind() {
        let t = table(
            "i,f,d,dt,s\n1,1.5,2020-01-02,2020-01-02 03:04:05,x\n2,2,2021-12-31,2021-12-31T00:00,y\n,,,,\n",
        );
        let kinds: Vec<Kind> = t.columns.iter().map(Column::kind).collect();
        assert_eq!(
            kinds,
            [
                Kind::Int,
                Kind::Float,
                Kind::Date,
                Kind::DateTime,
                Kind::Str
            ]
        );
        assert_eq!(t.rows, 3);
        assert!(t.columns.iter().all(|c| c.is_null(2)));
    }

    #[test]
    fn mixed_columns_fall_back_to_text() {
        let t = table("a,b,c\n1,2020-01-01,1.5\nx,hello,2\n");
        let kinds: Vec<Kind> = t.columns.iter().map(Column::kind).collect();
        assert_eq!(kinds, [Kind::Str, Kind::Str, Kind::Float]);
    }

    #[test]
    fn leading_zeros_and_ids_stay_text() {
        let t = table("zip,n\n02134,10\n90210,20\n");
        assert_eq!(t.columns[0].kind(), Kind::Str);
        assert_eq!(t.columns[0].text(0).as_deref(), Some("02134"));
        assert_eq!(t.columns[1].kind(), Kind::Int);
    }

    #[test]
    fn null_tokens_and_question_marks() {
        // "?" is data in the Adult census (an unknown occupation), not a missing value
        let t = table("a,b\nx,1\n?,NA\nnull,3\n");
        assert_eq!(t.columns[0].text(1).as_deref(), Some("?"));
        assert!(t.columns[0].is_null(2));
        assert!(t.columns[1].is_null(1));
        assert_eq!(t.columns[1].kind(), Kind::Int);
    }

    #[test]
    fn dictionary_encoding_shares_codes() {
        let t = table("s\na\nb\na\nb\na\n");
        let Column::Str { dict, codes } = &t.columns[0] else {
            panic!("expected text")
        };
        assert_eq!(dict, &["a", "b"]);
        assert_eq!(codes, &[0, 1, 0, 1, 0]);
    }

    #[test]
    fn huge_integers_become_floats_and_inf_is_text() {
        let t = table("a,b\n99999999999999999999,inf\n1,2\n");
        assert_eq!(t.columns[0].kind(), Kind::Float);
        assert_eq!(t.columns[1].kind(), Kind::Str);
    }

    #[test]
    fn accessors() {
        let t = table("n,f,d,s\n3,2.5,2020-03-04,x\n,,,\n");
        assert_eq!(t.columns[0].number(0), Some(3.0));
        assert_eq!(t.columns[0].number(1), None);
        assert_eq!(t.columns[2].text(0).as_deref(), Some("2020-03-04"));
        assert_eq!(t.columns[3].number(0), None);
        assert_eq!(t.columns[1].identity(0), Some(2.5f64.to_bits()));
        assert_eq!(t.columns[3].identity(1), None);
        assert!(
            t.find("nope")
                .unwrap_err()
                .to_string()
                .contains("Fields: n, f, d, s")
        );
    }

    #[test]
    fn describe_reports_extent_and_nulls() {
        let d = table("n,s\n5,a\n2,b\n,a\n").describe();
        assert_eq!(
            (d[0].nulls, d[0].min.as_deref(), d[0].max.as_deref()),
            (1, Some("2"), Some("5"))
        );
        assert_eq!((d[1].distinct, d[1].min.clone()), (Some(2), None));
    }

    #[test]
    fn datetime_text_roundtrip() {
        let t = table("t\n2020-01-02 03:04:05\n2020-01-02 03:04:06\n");
        assert_eq!(t.columns[0].text(0).as_deref(), Some("2020-01-02 03:04:05"));
    }

    #[test]
    fn float_formatting_drops_trailing_zero() {
        assert_eq!(format_float(3.0), "3");
        assert_eq!(format_float(2.5), "2.5");
    }

    #[test]
    fn empty_and_all_null_columns() {
        let t = table("a,b\n,\n,\n");
        assert!(
            t.columns
                .iter()
                .all(|c| c.kind() == Kind::Str && c.is_null(0))
        );
        let t = table("a\n");
        assert_eq!(t.rows, 0);
    }
}
