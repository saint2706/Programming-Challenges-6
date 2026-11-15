//! Column-chunk encodings. The writer tries every encoding that applies and keeps the smallest,
//! the same per-chunk choice Parquet writers make.
//!
//! | tag | name  | types | body                                                          |
//! |-----|-------|-------|---------------------------------------------------------------|
//! | 0   | plain | all   | i64 / f64 little-endian, strings as u32 length + bytes        |
//! | 1   | dict  | all   | plain-encoded distinct values, then bit-packed indices         |
//! | 2   | for   | int   | frame of reference: min, then bit-packed (v - min)             |
//! | 3   | rle   | int   | (value i64, run u32) pairs                                     |
//! | 4   | delta | int   | first value, then the FOR body of successive differences       |

use std::collections::HashMap;

use crate::Result;
use crate::types::Column;

pub const NAMES: [&str; 5] = ["plain", "dict", "for", "rle", "delta"];

// ---- byte helpers ---------------------------------------------------------------------------

pub fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
pub fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.len() as u32);
    b.extend_from_slice(s.as_bytes());
}

/// A bounds-checked cursor: a corrupt or truncated file is an error, never a panic.
pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or("unexpected end of data (truncated or corrupt file)")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    pub fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    pub fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.bytes(n)?.to_vec()).map_err(|_| "string is not UTF-8".into())
    }
}

// ---- bit packing ----------------------------------------------------------------------------

pub fn width(max: u64) -> u8 {
    (64 - max.leading_zeros()) as u8
}

/// Pack values of `w` bits each, least significant bit first.
pub fn pack(vals: impl Iterator<Item = u64>, w: u8, out: &mut Vec<u8>) {
    out.push(w);
    if w == 0 {
        return;
    }
    let (mut acc, mut bits) = (0u128, 0u32);
    for v in vals {
        acc |= (v as u128) << bits;
        bits += w as u32;
        while bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push(acc as u8);
    }
}

pub fn unpack(c: &mut Cursor, n: usize) -> Result<Vec<u64>> {
    let w = c.u8()? as u32;
    if w > 64 {
        return Err(format!("bit width {w} > 64"));
    }
    if w == 0 {
        return Ok(vec![0; n]);
    }
    let bytes = c.bytes((n * w as usize).div_ceil(8))?;
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let mut out = Vec::with_capacity(n);
    let (mut acc, mut bits, mut it) = (0u128, 0u32, bytes.iter());
    for _ in 0..n {
        while bits < w {
            acc |= (*it.next().unwrap() as u128) << bits;
            bits += 8;
        }
        out.push(acc as u64 & mask);
        acc >>= w;
        bits -= w;
    }
    Ok(out)
}

// ---- encoders -------------------------------------------------------------------------------

fn plain(col: &Column, out: &mut Vec<u8>) {
    match col {
        Column::Int(v) => v.iter().for_each(|x| put_u64(out, *x as u64)),
        Column::Float(v) => v.iter().for_each(|x| put_u64(out, x.to_bits())),
        Column::Str(v) => v.iter().for_each(|s| put_str(out, s)),
    }
}

fn for_body(v: &[i64], out: &mut Vec<u8>) {
    let min = v.iter().copied().min().unwrap_or(0);
    let max = v
        .iter()
        .map(|x| x.wrapping_sub(min) as u64)
        .max()
        .unwrap_or(0);
    put_u64(out, min as u64);
    pack(
        v.iter().map(|x| x.wrapping_sub(min) as u64),
        width(max),
        out,
    );
}

fn dict(col: &Column, out: &mut Vec<u8>) -> bool {
    fn build<K: std::hash::Hash + Eq, T: Clone>(
        v: &[T],
        key: impl Fn(&T) -> K,
    ) -> Option<(Vec<T>, Vec<u64>)> {
        let mut ids: HashMap<K, u64> = HashMap::new();
        let mut uniq = vec![];
        let mut codes = Vec::with_capacity(v.len());
        for x in v {
            let next = ids.len() as u64;
            let id = *ids.entry(key(x)).or_insert_with(|| {
                uniq.push(x.clone());
                next
            });
            if uniq.len() > v.len() / 2 + 1 {
                return None; // mostly distinct: a dictionary only adds bytes
            }
            codes.push(id);
        }
        Some((uniq, codes))
    }
    let (d, codes) = match col {
        Column::Int(v) => match build(v, |x| *x) {
            Some((u, c)) => (Column::Int(u), c),
            None => return false,
        },
        Column::Float(v) => match build(v, |x| x.to_bits()) {
            Some((u, c)) => (Column::Float(u), c),
            None => return false,
        },
        Column::Str(v) => match build(v, |x| x.clone()) {
            Some((u, c)) => (Column::Str(u), c),
            None => return false,
        },
    };
    put_u32(out, d.len() as u32);
    plain(&d, out);
    let w = width(d.len().saturating_sub(1) as u64);
    pack(codes.into_iter(), w, out);
    true
}

/// Encode a chunk with every applicable encoding and keep the smallest. Returns the bytes
/// (tag, row count, body) and the chosen tag.
pub fn encode(col: &Column) -> (Vec<u8>, u8) {
    let mut cands: Vec<(u8, Vec<u8>)> = vec![];
    let mut b = vec![];
    plain(col, &mut b);
    cands.push((0, b));
    let mut b = vec![];
    if dict(col, &mut b) {
        cands.push((1, b));
    }
    if let Column::Int(v) = col {
        let mut b = vec![];
        for_body(v, &mut b);
        cands.push((2, b));
        let mut b = vec![];
        let mut runs = 0u32;
        let mut i = 0;
        let mut body = vec![];
        while i < v.len() {
            let mut j = i;
            while j < v.len() && v[j] == v[i] {
                j += 1;
            }
            put_u64(&mut body, v[i] as u64);
            put_u32(&mut body, (j - i) as u32);
            runs += 1;
            i = j;
        }
        put_u32(&mut b, runs);
        b.extend(body);
        cands.push((3, b));
        if let Some(&first) = v.first() {
            let d: Vec<i64> = v.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect();
            let mut b = vec![];
            put_u64(&mut b, first as u64);
            for_body(&d, &mut b);
            cands.push((4, b));
        }
    }
    let (tag, body) = cands.into_iter().min_by_key(|(_, b)| b.len()).unwrap();
    let mut out = vec![tag];
    put_u32(&mut out, col.len() as u32);
    out.extend(body);
    (out, tag)
}

fn decode_plain(c: &mut Cursor, t: crate::DataType, n: usize) -> Result<Column> {
    Ok(match t {
        crate::DataType::Int => Column::Int((0..n).map(|_| c.i64()).collect::<Result<_>>()?),
        crate::DataType::Float => Column::Float((0..n).map(|_| c.f64()).collect::<Result<_>>()?),
        crate::DataType::Str => Column::Str((0..n).map(|_| c.string()).collect::<Result<_>>()?),
    })
}

fn decode_for(c: &mut Cursor, n: usize) -> Result<Vec<i64>> {
    let min = c.i64()?;
    Ok(unpack(c, n)?
        .into_iter()
        .map(|d| min.wrapping_add(d as i64))
        .collect())
}

pub fn decode(buf: &[u8], t: crate::DataType) -> Result<Column> {
    let mut c = Cursor::new(buf);
    let tag = c.u8()?;
    let n = c.u32()? as usize;
    let int_only = |what: &str| -> Result<()> {
        if t == crate::DataType::Int {
            Ok(())
        } else {
            Err(format!("{what} on a non-int column"))
        }
    };
    Ok(match tag {
        0 => decode_plain(&mut c, t, n)?,
        1 => {
            let k = c.u32()? as usize;
            let d = decode_plain(&mut c, t, k)?;
            let codes = unpack(&mut c, n)?;
            if codes.iter().any(|&i| i as usize >= k) {
                return Err("dictionary index out of range".into());
            }
            d.take(&codes.iter().map(|&i| i as usize).collect::<Vec<_>>())
        }
        2 => {
            int_only("FOR")?;
            Column::Int(decode_for(&mut c, n)?)
        }
        3 => {
            int_only("RLE")?;
            let runs = c.u32()?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..runs {
                let (x, len) = (c.i64()?, c.u32()? as usize);
                if v.len() + len > n {
                    return Err("RLE runs exceed the row count".into());
                }
                v.extend(std::iter::repeat_n(x, len));
            }
            if v.len() != n {
                return Err("RLE runs do not cover the row count".into());
            }
            Column::Int(v)
        }
        4 => {
            int_only("delta")?;
            let first = c.i64()?;
            let d = decode_for(&mut c, n.saturating_sub(1))?;
            let mut v = Vec::with_capacity(n);
            if n > 0 {
                v.push(first);
            }
            for x in d {
                v.push(v.last().unwrap().wrapping_add(x));
            }
            Column::Int(v)
        }
        _ => return Err(format!("unknown encoding tag {tag}")),
    })
}
