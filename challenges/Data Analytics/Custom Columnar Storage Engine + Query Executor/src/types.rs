//! Column vectors and scalar values. No NULLs: every column is dense.

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Int,
    Float,
    Str,
}

impl DataType {
    pub fn tag(self) -> u8 {
        self as u8
    }
    pub fn from_tag(t: u8) -> crate::Result<Self> {
        match t {
            0 => Ok(Self::Int),
            1 => Ok(Self::Float),
            2 => Ok(Self::Str),
            _ => Err(format!("bad type tag {t}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Str(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dtype(&self) -> DataType {
        match self {
            Self::Int(_) => DataType::Int,
            Self::Float(_) => DataType::Float,
            Self::Str(_) => DataType::Str,
        }
    }
    pub fn empty(t: DataType) -> Self {
        match t {
            DataType::Int => Self::Int(vec![]),
            DataType::Float => Self::Float(vec![]),
            DataType::Str => Self::Str(vec![]),
        }
    }
    pub fn get(&self, i: usize) -> Value {
        match self {
            Self::Int(v) => Value::Int(v[i]),
            Self::Float(v) => Value::Float(v[i]),
            Self::Str(v) => Value::Str(v[i].clone()),
        }
    }
    /// Keep the rows where `mask` is true.
    pub fn filter(&self, mask: &[bool]) -> Self {
        fn f<T: Clone>(v: &[T], m: &[bool]) -> Vec<T> {
            v.iter()
                .zip(m)
                .filter(|(_, k)| **k)
                .map(|(x, _)| x.clone())
                .collect()
        }
        match self {
            Self::Int(v) => Self::Int(f(v, mask)),
            Self::Float(v) => Self::Float(f(v, mask)),
            Self::Str(v) => Self::Str(f(v, mask)),
        }
    }
    /// Gather rows by index (used by sort and join).
    pub fn take(&self, idx: &[usize]) -> Self {
        match self {
            Self::Int(v) => Self::Int(idx.iter().map(|&i| v[i]).collect()),
            Self::Float(v) => Self::Float(idx.iter().map(|&i| v[i]).collect()),
            Self::Str(v) => Self::Str(idx.iter().map(|&i| v[i].clone()).collect()),
        }
    }
    pub fn append(&mut self, other: Column) -> crate::Result<()> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.extend(b),
            (Self::Float(a), Self::Float(b)) => a.extend(b),
            (Self::Str(a), Self::Str(b)) => a.extend(b),
            (a, b) => return Err(format!("cannot append {:?} to {:?}", b.dtype(), a.dtype())),
        }
        Ok(())
    }
    pub fn push(&mut self, v: Value) -> crate::Result<()> {
        match (self, v) {
            (Self::Int(a), Value::Int(x)) => a.push(x),
            (Self::Float(a), Value::Float(x)) => a.push(x),
            (Self::Float(a), Value::Int(x)) => a.push(x as f64),
            (Self::Str(a), Value::Str(x)) => a.push(x),
            (c, v) => return Err(format!("cannot push {v} into a {:?} column", c.dtype())),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            Self::Str(_) => None,
        }
    }
    pub fn dtype(&self) -> DataType {
        match self {
            Self::Int(_) => DataType::Int,
            Self::Float(_) => DataType::Float,
            Self::Str(_) => DataType::Str,
        }
    }
    /// Total order: numbers compare numerically (NaN last), strings lexically, numbers before strings.
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Str(a), Self::Str(b)) => a.cmp(b),
            (Self::Str(_), _) => Ordering::Greater,
            (_, Self::Str(_)) => Ordering::Less,
            (a, b) => a.as_f64().unwrap().total_cmp(&b.as_f64().unwrap()),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v}"),
        }
    }
}

/// A set of equal-length named columns: one row group on disk, one vector batch in the executor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Batch {
    pub names: Vec<String>,
    pub columns: Vec<Column>,
}

impl Batch {
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, Column::len)
    }
    pub fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
}
