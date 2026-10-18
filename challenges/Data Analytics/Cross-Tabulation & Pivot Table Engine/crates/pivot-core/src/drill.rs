//! Drill-through: the source rows behind one cell of a pivot.
//!
//! A cell is named by the labels of its row path and column path as the result reported them (a
//! subtotal or grand total simply has a shorter path), so the page can send back exactly what it
//! showed. The rows are found with the same filters and the same encoding the pivot used, so the
//! count always agrees with the number a count-measure shows in that cell.

use serde::Serialize;

use crate::error::{Error, Result};
use crate::pivot;
use crate::spec::PivotSpec;
use crate::table::Table;

#[derive(Debug, Serialize)]
pub struct Drill {
    /// Every row behind the cell.
    pub total: usize,
    pub fields: Vec<String>,
    /// At most `limit` rows, in file order; `None` is a missing value.
    pub rows: Vec<Vec<Option<String>>>,
    /// The row numbers (0-based, in file order) of the rows returned.
    pub row_numbers: Vec<usize>,
}

fn codes_for(
    path: &[String],
    encs: &[crate::encode::Encoded],
    dims: &[crate::spec::Dim],
    axis: &str,
) -> Result<Vec<u32>> {
    if path.len() > dims.len() {
        return Err(Error::new(format!(
            "the {axis} path has {} labels but the pivot has {} {axis} fields",
            path.len(),
            dims.len()
        )));
    }
    path.iter()
        .enumerate()
        .map(|(i, label)| {
            encs[i]
                .labels
                .iter()
                .position(|l| l == label)
                .map(|p| p as u32)
                .ok_or_else(|| {
                    Error::new(format!(
                        "no '{label}' in {}",
                        crate::encode::title(&dims[i])
                    ))
                })
        })
        .collect()
}

pub fn drill(
    table: &Table,
    spec: &PivotSpec,
    row_path: &[String],
    col_path: &[String],
    limit: usize,
) -> Result<Drill> {
    let prep = pivot::prepare(table, spec)?;
    let rc = codes_for(row_path, &prep.row_encs, &spec.rows, "row")?;
    let cc = codes_for(col_path, &prep.col_encs, &spec.cols, "column")?;
    let mut total = 0;
    let mut row_numbers = Vec::new();
    for row in 0..table.rows {
        if !prep.keep[row] {
            continue;
        }
        let hit = rc
            .iter()
            .enumerate()
            .all(|(i, &c)| prep.row_encs[i].codes[row] == c)
            && cc
                .iter()
                .enumerate()
                .all(|(i, &c)| prep.col_encs[i].codes[row] == c);
        if hit {
            total += 1;
            if row_numbers.len() < limit {
                row_numbers.push(row);
            }
        }
    }
    let rows = row_numbers
        .iter()
        .map(|&r| table.columns.iter().map(|c| c.text(r)).collect())
        .collect();
    Ok(Drill {
        total,
        fields: table.names.clone(),
        rows,
        row_numbers,
    })
}
