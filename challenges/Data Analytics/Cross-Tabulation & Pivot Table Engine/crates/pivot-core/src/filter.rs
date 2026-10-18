//! Row filters ("page fields"): every filter must hold for a row to take part in the pivot.
//!
//! A missing value never satisfies a comparison, `in` or `contains`; it does satisfy `ne` and
//! `not_in` ("is not this"), and `is_null` / `not_null` test for it directly.

use serde_json::Value as Json;

use crate::date;
use crate::error::{Error, Result};
use crate::spec::{Filter, FilterOp};
use crate::table::{Column, Table};

/// One value of a filter, converted to what the column holds.
enum Operand {
    Num(f64),
    Text(String),
}

fn operand(col: &Column, field: &str, v: &Json) -> Result<Operand> {
    let bad = |what: &str| Error::new(format!("filter on '{field}': {v} is not {what}"));
    match col {
        Column::Str { .. } => Ok(Operand::Text(match v {
            Json::String(s) => s.clone(),
            Json::Number(n) => n.to_string(),
            Json::Bool(b) => b.to_string(),
            _ => return Err(bad("text")),
        })),
        Column::Int(_) | Column::Float(_) => match v {
            Json::Number(n) => Ok(Operand::Num(n.as_f64().ok_or_else(|| bad("a number"))?)),
            Json::String(s) => s
                .trim()
                .parse::<f64>()
                .map(Operand::Num)
                .map_err(|_| bad("a number")),
            _ => Err(bad("a number")),
        },
        Column::Date(_) => match v {
            Json::String(s) => date::parse_date(s.trim())
                .map(|d| Operand::Num(d as f64))
                .ok_or_else(|| bad("a date (YYYY-MM-DD)")),
            _ => Err(bad("a date (YYYY-MM-DD)")),
        },
        Column::DateTime(_) => match v {
            Json::String(s) => {
                let s = s.trim();
                date::parse_datetime(s)
                    .or_else(|| date::parse_date(s).map(|d| d * 86_400))
                    .map(|x| Operand::Num(x as f64))
                    .ok_or_else(|| bad("a date or datetime"))
            }
            _ => Err(bad("a date or datetime")),
        },
    }
}

fn cmp_to(col: &Column, i: usize, op: &Operand) -> Option<std::cmp::Ordering> {
    match (col, op) {
        (Column::Str { dict, codes }, Operand::Text(t)) => (codes[i] != crate::table::NULL)
            .then(|| dict[codes[i] as usize].as_str().cmp(t.as_str())),
        (_, Operand::Num(x)) => col.number(i).map(|v| v.total_cmp(x)),
        _ => None,
    }
}

fn list(f: &Filter) -> Result<&Vec<Json>> {
    f.values.as_ref().ok_or_else(|| {
        Error::new(format!(
            "filter on '{}': this operator needs a list in \"values\"",
            f.field
        ))
    })
}

fn single(f: &Filter) -> Result<&Json> {
    f.value.as_ref().ok_or_else(|| {
        Error::new(format!(
            "filter on '{}': this operator needs a \"value\"",
            f.field
        ))
    })
}

/// `mask[row]` is true when the row passes every filter.
pub fn mask(table: &Table, filters: &[Filter]) -> Result<Vec<bool>> {
    use std::cmp::Ordering::*;
    let mut keep = vec![true; table.rows];
    for f in filters {
        let (_, col) = table.find(&f.field)?;
        match f.op {
            FilterOp::IsNull => keep
                .iter_mut()
                .enumerate()
                .for_each(|(i, k)| *k &= col.is_null(i)),
            FilterOp::NotNull => keep
                .iter_mut()
                .enumerate()
                .for_each(|(i, k)| *k &= !col.is_null(i)),
            FilterOp::Eq
            | FilterOp::Ne
            | FilterOp::Gt
            | FilterOp::Ge
            | FilterOp::Lt
            | FilterOp::Le => {
                let target = operand(col, &f.field, single(f)?)?;
                for (i, k) in keep.iter_mut().enumerate() {
                    let ord = cmp_to(col, i, &target);
                    *k &= match (f.op, ord) {
                        (FilterOp::Ne, None) => true,
                        (_, None) => false,
                        (FilterOp::Eq, Some(o)) => o == Equal,
                        (FilterOp::Ne, Some(o)) => o != Equal,
                        (FilterOp::Gt, Some(o)) => o == Greater,
                        (FilterOp::Ge, Some(o)) => o != Less,
                        (FilterOp::Lt, Some(o)) => o == Less,
                        (_, Some(o)) => o != Greater,
                    };
                }
            }
            FilterOp::In | FilterOp::NotIn => {
                let targets = list(f)?
                    .iter()
                    .map(|v| operand(col, &f.field, v))
                    .collect::<Result<Vec<_>>>()?;
                for (i, k) in keep.iter_mut().enumerate() {
                    let hit = targets.iter().any(|t| cmp_to(col, i, t) == Some(Equal));
                    *k &= if f.op == FilterOp::In { hit } else { !hit };
                }
            }
            FilterOp::Between => {
                let bounds = list(f)?;
                if bounds.len() != 2 {
                    return Err(Error::new(format!(
                        "filter on '{}': between needs exactly two values",
                        f.field
                    )));
                }
                let (lo, hi) = (
                    operand(col, &f.field, &bounds[0])?,
                    operand(col, &f.field, &bounds[1])?,
                );
                for (i, k) in keep.iter_mut().enumerate() {
                    *k &= matches!(cmp_to(col, i, &lo), Some(Greater | Equal))
                        && matches!(cmp_to(col, i, &hi), Some(Less | Equal));
                }
            }
            FilterOp::Contains => {
                let needle = match single(f)? {
                    Json::String(s) => s.to_lowercase(),
                    other => other.to_string().to_lowercase(),
                };
                for (i, k) in keep.iter_mut().enumerate() {
                    *k &= col
                        .text(i)
                        .is_some_and(|t| t.to_lowercase().contains(&needle));
                }
            }
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::LoadOptions;
    use serde_json::json;

    fn run(
        op: FilterOp,
        field: &str,
        value: Option<Json>,
        values: Option<Vec<Json>>,
    ) -> Result<Vec<bool>> {
        let t = Table::from_csv(
            b"name,age,when,stamp\nann,30,2020-01-01,2020-01-01 10:00\nbob,,2020-06-15,2020-06-15 12:30\ncy,45,2021-02-03,2021-02-03 00:00\n,45,,\n",
            LoadOptions::default(),
        )
        .unwrap();
        mask(
            &t,
            &[Filter {
                field: field.into(),
                op,
                value,
                values,
            }],
        )
    }

    #[test]
    fn equality_and_membership_on_text() {
        assert_eq!(
            run(FilterOp::Eq, "name", Some(json!("bob")), None).unwrap(),
            [false, true, false, false]
        );
        assert_eq!(
            run(FilterOp::Ne, "name", Some(json!("bob")), None).unwrap(),
            [true, false, true, true]
        );
        assert_eq!(
            run(
                FilterOp::In,
                "name",
                None,
                Some(vec![json!("ann"), json!("cy")])
            )
            .unwrap(),
            [true, false, true, false]
        );
        assert_eq!(
            run(FilterOp::NotIn, "name", None, Some(vec![json!("ann")])).unwrap(),
            [false, true, true, true]
        );
    }

    #[test]
    fn comparisons_on_numbers_skip_missing_values() {
        assert_eq!(
            run(FilterOp::Gt, "age", Some(json!(30)), None).unwrap(),
            [false, false, true, true]
        );
        assert_eq!(
            run(FilterOp::Ge, "age", Some(json!(30)), None).unwrap(),
            [true, false, true, true]
        );
        assert_eq!(
            run(FilterOp::Lt, "age", Some(json!("45")), None).unwrap(),
            [true, false, false, false]
        );
        assert_eq!(
            run(FilterOp::Le, "age", Some(json!(45.0)), None).unwrap(),
            [true, false, true, true]
        );
        assert_eq!(
            run(
                FilterOp::Between,
                "age",
                None,
                Some(vec![json!(30), json!(40)])
            )
            .unwrap(),
            [true, false, false, false]
        );
    }

    #[test]
    fn null_tests() {
        assert_eq!(
            run(FilterOp::IsNull, "age", None, None).unwrap(),
            [false, true, false, false]
        );
        assert_eq!(
            run(FilterOp::NotNull, "name", None, None).unwrap(),
            [true, true, true, false]
        );
        assert_eq!(
            run(FilterOp::Ne, "age", Some(json!(45)), None).unwrap(),
            [true, true, false, false]
        );
    }

    #[test]
    fn dates_and_datetimes() {
        assert_eq!(
            run(FilterOp::Ge, "when", Some(json!("2020-06-15")), None).unwrap(),
            [false, true, true, false]
        );
        assert_eq!(
            run(
                FilterOp::Between,
                "when",
                None,
                Some(vec![json!("2020-01-01"), json!("2020-12-31")])
            )
            .unwrap(),
            [true, true, false, false]
        );
        assert_eq!(
            run(FilterOp::Gt, "stamp", Some(json!("2020-06-15")), None).unwrap(),
            [false, true, true, false]
        );
        assert_eq!(
            run(FilterOp::Lt, "stamp", Some(json!("2020-06-15 12:30")), None).unwrap(),
            [true, false, false, false]
        );
    }

    #[test]
    fn contains_is_case_insensitive() {
        assert_eq!(
            run(FilterOp::Contains, "name", Some(json!("N")), None).unwrap(),
            [true, false, false, false]
        );
    }

    #[test]
    fn filters_combine_with_and() {
        let t = Table::from_csv(b"a,b\n1,x\n2,x\n3,y\n", LoadOptions::default()).unwrap();
        let f = |field: &str, op, value| Filter {
            field: field.into(),
            op,
            value: Some(value),
            values: None,
        };
        let m = mask(
            &t,
            &[
                f("a", FilterOp::Ge, json!(2)),
                f("b", FilterOp::Eq, json!("x")),
            ],
        )
        .unwrap();
        assert_eq!(m, [false, true, false]);
        assert_eq!(mask(&t, &[]).unwrap(), [true, true, true]);
    }

    #[test]
    fn mistakes_are_explained() {
        let msg = |r: Result<Vec<bool>>| r.unwrap_err().to_string();
        assert!(
            msg(run(FilterOp::Eq, "nope", Some(json!(1)), None)).contains("no field called 'nope'")
        );
        assert!(msg(run(FilterOp::Gt, "age", Some(json!("abc")), None)).contains("not a number"));
        assert!(
            msg(run(FilterOp::Eq, "when", Some(json!("yesterday")), None)).contains("not a date")
        );
        assert!(msg(run(FilterOp::Eq, "age", None, None)).contains("needs a \"value\""));
        assert!(msg(run(FilterOp::In, "age", None, None)).contains("needs a list"));
        assert!(
            msg(run(FilterOp::Between, "age", None, Some(vec![json!(1)]))).contains("exactly two")
        );
        assert!(msg(run(FilterOp::Eq, "name", Some(json!([1])), None)).contains("not text"));
    }
}
