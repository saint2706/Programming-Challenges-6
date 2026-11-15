//! The `.col` file: `COL1` | column chunks, row group by row group | footer | footer length (u32) | `COL1`.
//!
//! The footer holds the schema and, for every row group and column, the chunk's byte range,
//! encoding and min/max (the zone map). A reader opens the footer once and then seeks straight
//! to the chunks a query needs, so unread columns and pruned row groups cost no I/O.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::Result;
use crate::encoding::{self, Cursor, put_str, put_u32, put_u64};
use crate::types::{Batch, Column, DataType, Value};

const MAGIC: &[u8; 4] = b"COL1";
pub const DEFAULT_ROW_GROUP: usize = 65_536;

#[derive(Clone, Debug)]
pub struct ChunkMeta {
    pub offset: u64,
    pub len: u64,
    pub encoding: u8,
    pub min: Value,
    pub max: Value,
}

#[derive(Clone, Debug)]
pub struct RowGroupMeta {
    pub rows: u32,
    pub chunks: Vec<ChunkMeta>,
}

#[derive(Clone, Debug)]
pub struct Footer {
    pub names: Vec<String>,
    pub types: Vec<DataType>,
    pub row_groups: Vec<RowGroupMeta>,
}

fn min_max(c: &Column) -> (Value, Value) {
    match c {
        Column::Int(v) => (
            Value::Int(*v.iter().min().unwrap()),
            Value::Int(*v.iter().max().unwrap()),
        ),
        Column::Float(v) => {
            let lo = v.iter().copied().min_by(f64::total_cmp).unwrap();
            let hi = v.iter().copied().max_by(f64::total_cmp).unwrap();
            (Value::Float(lo), Value::Float(hi))
        }
        Column::Str(v) => (
            Value::Str(v.iter().min().unwrap().clone()),
            Value::Str(v.iter().max().unwrap().clone()),
        ),
    }
}

fn put_value(b: &mut Vec<u8>, v: &Value) {
    b.push(v.dtype().tag());
    match v {
        Value::Int(x) => put_u64(b, *x as u64),
        Value::Float(x) => put_u64(b, x.to_bits()),
        Value::Str(s) => put_str(b, s),
    }
}

fn get_value(c: &mut Cursor) -> Result<Value> {
    Ok(match DataType::from_tag(c.u8()?)? {
        DataType::Int => Value::Int(c.i64()?),
        DataType::Float => Value::Float(c.f64()?),
        DataType::Str => Value::Str(c.string()?),
    })
}

/// Streaming writer: one `write_group` call per row group, then `finish` writes the footer.
pub struct Writer {
    w: BufWriter<File>,
    offset: u64,
    footer: Footer,
}

fn io(e: std::io::Error) -> String {
    e.to_string()
}

impl Writer {
    pub fn create(path: &Path, names: Vec<String>, types: Vec<DataType>) -> Result<Self> {
        let f = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);
        w.write_all(MAGIC).map_err(io)?;
        Ok(Self {
            w,
            offset: MAGIC.len() as u64,
            footer: Footer {
                names,
                types,
                row_groups: vec![],
            },
        })
    }

    pub fn write_group(&mut self, b: &Batch) -> Result<()> {
        let types: Vec<DataType> = b.columns.iter().map(Column::dtype).collect();
        if types != self.footer.types {
            return Err("row group schema differs from the file schema".into());
        }
        if b.rows() == 0 {
            return Ok(());
        }
        let mut chunks = vec![];
        for col in &b.columns {
            let (bytes, tag) = encoding::encode(col);
            let (min, max) = min_max(col);
            self.w.write_all(&bytes).map_err(io)?;
            chunks.push(ChunkMeta {
                offset: self.offset,
                len: bytes.len() as u64,
                encoding: tag,
                min,
                max,
            });
            self.offset += bytes.len() as u64;
        }
        self.footer.row_groups.push(RowGroupMeta {
            rows: b.rows() as u32,
            chunks,
        });
        Ok(())
    }

    pub fn finish(mut self) -> Result<Footer> {
        let footer = self.footer;
        let mut b = vec![];
        put_u32(&mut b, footer.names.len() as u32);
        for (name, t) in footer.names.iter().zip(&footer.types) {
            put_str(&mut b, name);
            b.push(t.tag());
        }
        put_u32(&mut b, footer.row_groups.len() as u32);
        for g in &footer.row_groups {
            put_u32(&mut b, g.rows);
            for c in &g.chunks {
                put_u64(&mut b, c.offset);
                put_u64(&mut b, c.len);
                b.push(c.encoding);
                put_value(&mut b, &c.min);
                put_value(&mut b, &c.max);
            }
        }
        self.w.write_all(&b).map_err(io)?;
        self.w
            .write_all(&(b.len() as u32).to_le_bytes())
            .map_err(io)?;
        self.w.write_all(MAGIC).map_err(io)?;
        self.w.flush().map_err(io)?;
        Ok(footer)
    }
}

/// Write `batch` as a `.col` file, cutting it into row groups of `row_group` rows.
pub fn write(path: &Path, batch: &Batch, row_group: usize) -> Result<Footer> {
    if row_group == 0 {
        return Err("row group size must be positive".into());
    }
    let types = batch.columns.iter().map(Column::dtype).collect();
    let mut w = Writer::create(path, batch.names.clone(), types)?;
    let n = batch.rows();
    for start in (0..n).step_by(row_group) {
        let idx: Vec<usize> = (start..(start + row_group).min(n)).collect();
        let part = Batch {
            names: batch.names.clone(),
            columns: batch.columns.iter().map(|c| c.take(&idx)).collect(),
        };
        w.write_group(&part)?;
    }
    w.finish()
}

/// An open `.col` file.
#[derive(Debug)]
pub struct Table {
    file: File,
    pub footer: Footer,
}

impl Table {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = file.seek(SeekFrom::End(0)).map_err(io)?;
        if size < 12 {
            return Err(format!("{}: too small to be a .col file", path.display()));
        }
        let mut tail = [0u8; 8];
        file.seek(SeekFrom::End(-8)).map_err(io)?;
        file.read_exact(&mut tail).map_err(io)?;
        if &tail[4..] != MAGIC {
            return Err(format!("{}: not a .col file (bad magic)", path.display()));
        }
        let flen = u32::from_le_bytes(tail[..4].try_into().unwrap()) as u64;
        if flen + 12 > size {
            return Err("footer length points outside the file".into());
        }
        file.seek(SeekFrom::Start(size - 8 - flen)).map_err(io)?;
        let mut buf = vec![0; flen as usize];
        file.read_exact(&mut buf).map_err(io)?;
        let mut c = Cursor::new(&buf);
        let ncols = c.u32()? as usize;
        let (mut names, mut types) = (vec![], vec![]);
        for _ in 0..ncols {
            names.push(c.string()?);
            types.push(DataType::from_tag(c.u8()?)?);
        }
        let mut row_groups = vec![];
        for _ in 0..c.u32()? {
            let rows = c.u32()?;
            let mut chunks = vec![];
            for _ in 0..ncols {
                let (offset, len, encoding) = (c.u64()?, c.u64()?, c.u8()?);
                if offset.checked_add(len).is_none_or(|e| e > size - 8 - flen) {
                    return Err("chunk byte range points outside the data section".into());
                }
                chunks.push(ChunkMeta {
                    offset,
                    len,
                    encoding,
                    min: get_value(&mut c)?,
                    max: get_value(&mut c)?,
                });
            }
            row_groups.push(RowGroupMeta { rows, chunks });
        }
        Ok(Self {
            file,
            footer: Footer {
                names,
                types,
                row_groups,
            },
        })
    }

    pub fn rows(&self) -> u64 {
        self.footer.row_groups.iter().map(|g| g.rows as u64).sum()
    }

    pub fn column_index(&self, name: &str) -> Result<usize> {
        self.footer
            .names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("no column {name:?}"))
    }

    /// Read and decode one column chunk.
    pub fn read_chunk(&mut self, rg: usize, col: usize) -> Result<Column> {
        let m = &self.footer.row_groups[rg].chunks[col];
        let mut buf = vec![0; m.len as usize];
        self.file
            .seek(SeekFrom::Start(m.offset))
            .map_err(|e| e.to_string())?;
        self.file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        let c = encoding::decode(&buf, self.footer.types[col])?;
        if c.len() != self.footer.row_groups[rg].rows as usize {
            return Err(format!(
                "chunk rg {rg} col {col} decoded to the wrong row count"
            ));
        }
        Ok(c)
    }

    /// Read the given columns (by index) of one row group as a batch.
    pub fn read_group(&mut self, rg: usize, cols: &[usize]) -> Result<Batch> {
        let mut b = Batch::default();
        for &c in cols {
            b.names.push(self.footer.names[c].clone());
            b.columns.push(self.read_chunk(rg, c)?);
        }
        Ok(b)
    }
}
