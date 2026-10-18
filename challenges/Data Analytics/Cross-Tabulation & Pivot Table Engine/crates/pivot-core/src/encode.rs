//! Turning a field into small integer codes, one per distinct group, in display order.
//!
//! Every dimension of a pivot becomes a `Vec<u32>` of codes plus the label of each code. Grouping
//! (date parts, equal-width bins, cut points, quantiles) happens here, so the rest of the engine
//! only ever sees codes: they pack into one integer key per row and column and hash cheaply.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use crate::date;
use crate::error::{Error, Result};
use crate::fxhash::Fx;
use crate::spec::{Dim, Group};
use crate::table::{Column, Table, format_date, format_datetime, format_float};

#[derive(Debug)]
pub struct Encoded {
    /// A code per table row (meaningful only for rows that pass the filters).
    pub codes: Vec<u32>,
    /// `labels[code]`; a trailing blank label is present only when some used row has no value.
    pub labels: Vec<String>,
    /// The code of a missing value, if any used row has one.
    pub blank: Option<u32>,
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const WEEKDAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/// How a dimension is titled in the result: `age (bins of 10)`, `date (month)`.
pub fn title(dim: &Dim) -> String {
    let how = match &dim.group {
        None => return dim.field.clone(),
        Some(Group::Year) => "year".to_string(),
        Some(Group::Quarter) => "quarter".into(),
        Some(Group::Month) => "month".into(),
        Some(Group::YearMonth) => "year-month".into(),
        Some(Group::YearQuarter) => "year-quarter".into(),
        Some(Group::Weekday) => "weekday".into(),
        Some(Group::Day) => "day".into(),
        Some(Group::Hour) => "hour".into(),
        Some(Group::Bin { width, .. }) => format!("bins of {}", format_float(*width)),
        Some(Group::Edges { .. }) => "bands".into(),
        Some(Group::Quantile { n }) => format!("{n} quantiles"),
    };
    format!("{} ({how})", dim.field)
}

/// A float's rank-preserving integer image, so floats can be sorted and keyed like integers.
fn order_key(x: f64) -> i64 {
    let b = (x + 0.0).to_bits() as i64;
    if b < 0 { b ^ i64::MAX } else { b }
}

fn from_order_key(k: i64) -> f64 {
    f64::from_bits((if k < 0 { k ^ i64::MAX } else { k }) as u64)
}

fn tidy(x: f64) -> String {
    format_float((x * 1e9).round() / 1e9)
}

pub fn encode(table: &Table, dim: &Dim, mask: &[bool], blank: &str) -> Result<Encoded> {
    let (_, col) = table.find(&dim.field)?;
    let n = table.rows;
    // The common case, text with no grouping, needs only a remap of the dictionary codes: no
    // per-row key list, no sort of the rows.
    if dim.group.is_none()
        && let Column::Str { dict, codes } = col
    {
        let mut used = vec![false; dict.len()];
        let mut has_blank = false;
        for (&c, &m) in codes.iter().zip(mask) {
            if m {
                if c == crate::table::NULL {
                    has_blank = true;
                } else {
                    used[c as usize] = true;
                }
            }
        }
        let mut order: Vec<usize> = (0..dict.len()).filter(|&d| used[d]).collect();
        order.sort_by(|&a, &b| {
            dict[a]
                .to_lowercase()
                .cmp(&dict[b].to_lowercase())
                .then_with(|| dict[a].cmp(&dict[b]))
        });
        let mut remap = vec![0u32; dict.len()];
        for (rank, &d) in order.iter().enumerate() {
            remap[d] = rank as u32;
        }
        let mut labels: Vec<String> = order.iter().map(|&d| dict[d].clone()).collect();
        let blank_code = labels.len() as u32;
        if has_blank {
            labels.push(blank.to_string());
        }
        let out = codes
            .iter()
            .zip(mask)
            .map(|(&c, &m)| match (m, c) {
                (false, _) => 0,
                (true, crate::table::NULL) => blank_code,
                (true, c) => remap[c as usize],
            })
            .collect();
        return Ok(Encoded {
            codes: out,
            labels,
            blank: has_blank.then_some(blank_code),
        });
    }
    let temporal = matches!(col, Column::Date(_) | Column::DateTime(_));
    let need_temporal = || {
        Error::new(format!(
            "cannot group '{}' by date part: it is {}, not a date",
            dim.field,
            if col.is_numeric() { "a number" } else { "text" }
        ))
    };
    let need_number = || Error::new(format!("cannot bin '{}': it is not a number", dim.field));
    let days_of = |i: usize| -> Option<i64> {
        match col {
            Column::Date(v) => v[i],
            Column::DateTime(v) => v[i].map(|s| s.div_euclid(86_400)),
            _ => None,
        }
    };

    type Label = Box<dyn Fn(i64) -> String>;
    let (keys, label): (Vec<Option<i64>>, Label) = match (&dim.group, col) {
        (None, Column::Str { dict, codes }) => {
            let mut sorted: Vec<usize> = (0..dict.len()).collect();
            sorted.sort_by(|&a, &b| {
                dict[a]
                    .to_lowercase()
                    .cmp(&dict[b].to_lowercase())
                    .then_with(|| dict[a].cmp(&dict[b]))
            });
            let mut rank = vec![0i64; dict.len()];
            for (r, &d) in sorted.iter().enumerate() {
                rank[d] = r as i64;
            }
            let names: Vec<String> = sorted.iter().map(|&d| dict[d].clone()).collect();
            (
                codes
                    .iter()
                    .map(|&c| (c != crate::table::NULL).then(|| rank[c as usize]))
                    .collect(),
                Box::new(move |k| names[k as usize].clone()),
            )
        }
        (None, Column::Int(v)) => (v.clone(), Box::new(|k| k.to_string())),
        (None, Column::Float(v)) => (
            v.iter()
                .map(|x| (!x.is_nan()).then(|| order_key(*x)))
                .collect(),
            Box::new(|k| format_float(from_order_key(k))),
        ),
        (None, Column::Date(v)) => (v.clone(), Box::new(format_date)),
        (None, Column::DateTime(v)) => (v.clone(), Box::new(format_datetime)),
        (Some(Group::Year), _) => {
            if !temporal {
                return Err(need_temporal());
            }
            (
                (0..n)
                    .map(|i| days_of(i).map(|d| date::civil_from_days(d).0))
                    .collect(),
                Box::new(|k| k.to_string()),
            )
        }
        (Some(Group::Quarter), _) => {
            if !temporal {
                return Err(need_temporal());
            }
            (
                (0..n)
                    .map(|i| days_of(i).map(|d| date::quarter(date::civil_from_days(d).1) as i64))
                    .collect(),
                Box::new(|k| format!("Q{k}")),
            )
        }
        (Some(Group::Month), _) => {
            if !temporal {
                return Err(need_temporal());
            }
            (
                (0..n)
                    .map(|i| days_of(i).map(|d| date::civil_from_days(d).1 as i64))
                    .collect(),
                Box::new(|k| MONTHS[k as usize - 1].to_string()),
            )
        }
        (Some(Group::YearMonth), _) => {
            if !temporal {
                return Err(need_temporal());
            }
            (
                (0..n)
                    .map(|i| {
                        days_of(i).map(|d| {
                            let (y, m, _) = date::civil_from_days(d);
                            y * 12 + m as i64 - 1
                        })
                    })
                    .collect(),
                Box::new(|k| format!("{:04}-{:02}", k.div_euclid(12), k.rem_euclid(12) + 1)),
            )
        }
        (Some(Group::YearQuarter), _) => {
            if !temporal {
                return Err(need_temporal());
            }
            (
                (0..n)
                    .map(|i| {
                        days_of(i).map(|d| {
                            let (y, m, _) = date::civil_from_days(d);
                            y * 4 + date::quarter(m) as i64 - 1
                        })
                    })
                    .collect(),
                Box::new(|k| format!("{:04} Q{}", k.div_euclid(4), k.rem_euclid(4) + 1)),
            )
        }
        (Some(Group::Weekday), _) => {
            if !temporal {
                return Err(need_temporal());
            }
            (
                (0..n)
                    .map(|i| days_of(i).map(|d| date::weekday(d) as i64))
                    .collect(),
                Box::new(|k| WEEKDAYS[k as usize].to_string()),
            )
        }
        (Some(Group::Day), _) => {
            if !temporal {
                return Err(need_temporal());
            }
            ((0..n).map(days_of).collect(), Box::new(format_date))
        }
        (Some(Group::Hour), Column::DateTime(v)) => (
            v.iter()
                .map(|s| s.map(|s| s.rem_euclid(86_400) / 3600))
                .collect(),
            Box::new(|k| format!("{k:02}:00")),
        ),
        (Some(Group::Hour), _) => {
            return Err(Error::new(format!(
                "cannot group '{}' by hour: it has no time of day",
                dim.field
            )));
        }
        (Some(Group::Bin { width, start }), _) => {
            if col.is_numeric() && !temporal {
                if !(*width > 0.0 && width.is_finite()) {
                    return Err(Error::new("a bin width must be a positive number"));
                }
                let (w, s) = (*width, start.unwrap_or(0.0));
                (
                    (0..n)
                        .map(|i| col.number(i).map(|x| ((x - s) / w).floor() as i64))
                        .collect(),
                    Box::new(move |k| {
                        format!(
                            "[{}, {})",
                            tidy(s + k as f64 * w),
                            tidy(s + (k + 1) as f64 * w)
                        )
                    }),
                )
            } else {
                return Err(need_number());
            }
        }
        (Some(Group::Edges { edges }), _) => {
            if !col.is_numeric() || temporal {
                return Err(need_number());
            }
            let mut cuts = edges.clone();
            if cuts.is_empty() || cuts.iter().any(|c| !c.is_finite()) {
                return Err(Error::new("bands need at least one finite cut point"));
            }
            cuts.sort_by(f64::total_cmp);
            cuts.dedup();
            let last = cuts.len();
            let names = cuts.clone();
            (
                (0..n)
                    .map(|i| {
                        col.number(i)
                            .map(|x| cuts.partition_point(|&c| c <= x) as i64)
                    })
                    .collect(),
                Box::new(move |k| {
                    let k = k as usize;
                    if k == 0 {
                        format!("< {}", tidy(names[0]))
                    } else if k == last {
                        format!(">= {}", tidy(names[last - 1]))
                    } else {
                        format!("[{}, {})", tidy(names[k - 1]), tidy(names[k]))
                    }
                }),
            )
        }
        (Some(Group::Quantile { n: bins }), _) => {
            if !col.is_numeric() || temporal {
                return Err(need_number());
            }
            if *bins < 2 {
                return Err(Error::new("quantile grouping needs at least 2 bins"));
            }
            let mut values: Vec<f64> = (0..n)
                .filter(|&i| mask[i])
                .filter_map(|i| col.number(i))
                .collect();
            values.sort_by(f64::total_cmp);
            let mut cuts: Vec<f64> = (1..*bins)
                .map(|j| quantile(&values, j as f64 / *bins as f64))
                .collect();
            cuts.dedup();
            let keys: Vec<Option<i64>> = (0..n)
                .map(|i| {
                    col.number(i)
                        .map(|x| cuts.partition_point(|&c| c < x) as i64)
                })
                .collect();
            // label each bin by the values that actually fall in it
            let mut extent: HashMap<i64, (f64, f64)> = HashMap::new();
            for i in (0..n).filter(|&i| mask[i]) {
                if let (Some(k), Some(x)) = (keys[i], col.number(i)) {
                    let e = extent.entry(k).or_insert((x, x));
                    e.0 = e.0.min(x);
                    e.1 = e.1.max(x);
                }
            }
            let mut order: Vec<i64> = extent.keys().copied().collect();
            order.sort_unstable();
            let names: HashMap<i64, String> = order
                .iter()
                .enumerate()
                .map(|(rank, k)| {
                    let (lo, hi) = extent[k];
                    (
                        *k,
                        format!("Q{} ({}\u{2013}{})", rank + 1, tidy(lo), tidy(hi)),
                    )
                })
                .collect();
            (
                keys,
                Box::new(move |k| names.get(&k).cloned().unwrap_or_default()),
            )
        }
    };

    Ok(dense(&keys, mask, &*label, blank))
}

/// Linear-interpolation quantile (the usual "type 7"): `sorted` must be ascending.
pub fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let h = (sorted.len() - 1) as f64 * p;
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo as f64)
}

/// Dense codes in ascending key order over the rows that are in use; a missing key gets the last code.
fn dense(
    keys: &[Option<i64>],
    mask: &[bool],
    label: &dyn Fn(i64) -> String,
    blank: &str,
) -> Encoded {
    let mut distinct: Vec<i64> = keys
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .filter_map(|(k, _)| *k)
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    // Most keys are small and dense (ranks of a dictionary, month numbers): a direct lookup
    // table beats any hash. Sparse keys (ids, floats) fall back to the fast hasher.
    let (lo, hi) = (
        distinct.first().copied().unwrap_or(0),
        distinct.last().copied().unwrap_or(0),
    );
    let span = (hi as i128 - lo as i128 + 1) as usize;
    let direct: Option<Vec<u32>> = (span <= distinct.len() * 4 + 1024).then(|| {
        let mut table = vec![0u32; span];
        for (i, &k) in distinct.iter().enumerate() {
            table[(k - lo) as usize] = i as u32;
        }
        table
    });
    let sparse: HashMap<i64, u32, BuildHasherDefault<Fx>> = if direct.is_none() {
        distinct
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i as u32))
            .collect()
    } else {
        HashMap::default()
    };
    let lookup = |k: i64| -> u32 {
        match &direct {
            Some(t) => t[(k - lo) as usize],
            None => sparse[&k],
        }
    };
    let mut labels: Vec<String> = distinct.iter().map(|&k| label(k)).collect();
    let blank_code = labels.len() as u32;
    let mut has_blank = false;
    let codes = keys
        .iter()
        .zip(mask)
        .map(|(k, &m)| match (m, k) {
            (false, _) => 0,
            (true, Some(k)) => lookup(*k),
            (true, None) => {
                has_blank = true;
                blank_code
            }
        })
        .collect();
    if has_blank {
        labels.push(blank.to_string());
    }
    Encoded {
        codes,
        labels,
        blank: has_blank.then_some(blank_code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Sort;
    use crate::table::LoadOptions;

    fn dim(field: &str, group: Option<Group>) -> Dim {
        Dim {
            field: field.into(),
            group,
            sort: Sort::default(),
            top: None,
            hide_blank: false,
        }
    }

    fn enc(csv: &str, d: Dim) -> Encoded {
        let t = Table::from_csv(csv.as_bytes(), LoadOptions::default()).unwrap();
        encode(&t, &d, &vec![true; t.rows], "(blank)").unwrap()
    }

    fn rows_of(e: &Encoded) -> Vec<&str> {
        e.codes
            .iter()
            .map(|&c| e.labels[c as usize].as_str())
            .collect()
    }

    #[test]
    fn text_sorts_case_insensitively_with_blank_last() {
        let e = enc(
            "s,n\nbanana,1\nApple,2\napple,3\n,4\ncherry,5\n",
            dim("s", None),
        );
        assert_eq!(e.labels, ["Apple", "apple", "banana", "cherry", "(blank)"]);
        assert_eq!(
            rows_of(&e),
            ["banana", "Apple", "apple", "(blank)", "cherry"]
        );
        assert_eq!(e.blank, Some(4));
    }

    #[test]
    fn numbers_sort_by_value_not_text() {
        let e = enc("n\n10\n9\n100\n-5\n", dim("n", None));
        assert_eq!(e.labels, ["-5", "9", "10", "100"]);
        let e = enc("f\n2.5\n-1.5\n0.25\n", dim("f", None));
        assert_eq!(e.labels, ["-1.5", "0.25", "2.5"]);
    }

    #[test]
    fn float_order_key_is_monotone_and_reversible() {
        let xs = [-1e300, -2.5, -0.0, 0.0, 1e-300, 3.5, 1e300];
        for w in xs.windows(2) {
            assert!(order_key(w[0]) <= order_key(w[1]), "{w:?}");
        }
        for x in xs {
            assert_eq!(from_order_key(order_key(x)), x + 0.0);
        }
    }

    #[test]
    fn date_parts() {
        let csv = "d\n2019-03-15\n2019-11-02\n2020-03-01\n";
        assert_eq!(
            enc(csv, dim("d", Some(Group::Year))).labels,
            ["2019", "2020"]
        );
        assert_eq!(
            enc(csv, dim("d", Some(Group::Month))).labels,
            ["Mar", "Nov"]
        );
        assert_eq!(
            enc(csv, dim("d", Some(Group::Quarter))).labels,
            ["Q1", "Q4"]
        );
        assert_eq!(
            enc(csv, dim("d", Some(Group::YearMonth))).labels,
            ["2019-03", "2019-11", "2020-03"]
        );
        assert_eq!(
            enc(csv, dim("d", Some(Group::YearQuarter))).labels,
            ["2019 Q1", "2019 Q4", "2020 Q1"]
        );
        assert_eq!(
            enc(csv, dim("d", Some(Group::Weekday))).labels,
            ["Fri", "Sat", "Sun"]
        );
        assert_eq!(enc(csv, dim("d", Some(Group::Day))).labels.len(), 3);
    }

    #[test]
    fn month_groups_across_years_in_calendar_order() {
        let e = enc(
            "d\n2020-12-01\n2019-02-01\n2021-12-31\n",
            dim("d", Some(Group::Month)),
        );
        assert_eq!(e.labels, ["Feb", "Dec"]);
        assert_eq!(rows_of(&e), ["Dec", "Feb", "Dec"]);
    }

    #[test]
    fn hour_needs_a_time() {
        let e = enc(
            "t\n2020-01-01 09:30\n2020-01-01 17:00\n2020-01-02 09:05\n",
            dim("t", Some(Group::Hour)),
        );
        assert_eq!(e.labels, ["09:00", "17:00"]);
        let t = Table::from_csv(b"d\n2020-01-01\n", LoadOptions::default()).unwrap();
        assert!(
            encode(&t, &dim("d", Some(Group::Hour)), &[true], "-")
                .unwrap_err()
                .to_string()
                .contains("no time of day")
        );
    }

    #[test]
    fn bins_have_clean_boundaries_and_sort_by_value() {
        let e = enc(
            "age\n17\n25\n30\n39\n40\n91\n",
            dim(
                "age",
                Some(Group::Bin {
                    width: 10.0,
                    start: None,
                }),
            ),
        );
        assert_eq!(
            e.labels,
            ["[10, 20)", "[20, 30)", "[30, 40)", "[40, 50)", "[90, 100)"]
        );
        assert_eq!(rows_of(&e)[3], "[30, 40)");
        let e = enc(
            "x\n0\n4\n5\n",
            dim(
                "x",
                Some(Group::Bin {
                    width: 5.0,
                    start: Some(-2.5),
                }),
            ),
        );
        assert_eq!(e.labels, ["[-2.5, 2.5)", "[2.5, 7.5)"]);
    }

    #[test]
    fn edges_make_open_ended_bands() {
        let e = enc(
            "x\n-3\n0\n5\n10\n99\n",
            dim(
                "x",
                Some(Group::Edges {
                    edges: vec![10.0, 0.0],
                }),
            ),
        );
        assert_eq!(e.labels, ["< 0", "[0, 10)", ">= 10"]);
        assert_eq!(rows_of(&e), ["< 0", "[0, 10)", "[0, 10)", ">= 10", ">= 10"]);
    }

    #[test]
    fn quantiles_label_the_observed_range_and_collapse_ties() {
        let e = enc(
            "x\n1\n2\n3\n4\n5\n6\n7\n8\n",
            dim("x", Some(Group::Quantile { n: 4 })),
        );
        assert_eq!(e.labels.len(), 4);
        assert!(e.labels[0].starts_with("Q1 (1\u{2013}2") && e.labels[3].ends_with("8)"));
        let e = enc(
            "x\n0\n0\n0\n0\n0\n0\n0\n9\n",
            dim("x", Some(Group::Quantile { n: 4 })),
        );
        assert!(e.labels.len() <= 2, "{:?}", e.labels);
    }

    #[test]
    fn blank_only_appears_when_a_used_row_is_blank() {
        let t = Table::from_csv(b"s,n\na,1\n,2\nb,3\n", LoadOptions::default()).unwrap();
        let e = encode(&t, &dim("s", None), &[true, false, true], "(blank)").unwrap();
        assert_eq!(e.labels, ["a", "b"]);
        assert_eq!(e.blank, None);
    }

    #[test]
    fn wrong_kind_of_grouping_is_explained() {
        let t = Table::from_csv(b"s,n\na,1\n", LoadOptions::default()).unwrap();
        let go = |f: &str, g| {
            encode(&t, &dim(f, Some(g)), &[true], "-")
                .unwrap_err()
                .to_string()
        };
        assert!(go("s", Group::Month).contains("it is text, not a date"));
        assert!(go("n", Group::Year).contains("it is a number, not a date"));
        assert!(
            go(
                "s",
                Group::Bin {
                    width: 1.0,
                    start: None
                }
            )
            .contains("not a number")
        );
        assert!(
            go(
                "n",
                Group::Bin {
                    width: 0.0,
                    start: None
                }
            )
            .contains("positive")
        );
        assert!(go("n", Group::Edges { edges: vec![] }).contains("at least one"));
        assert!(go("n", Group::Quantile { n: 1 }).contains("at least 2"));
    }

    #[test]
    fn titles() {
        assert_eq!(
            title(&dim(
                "age",
                Some(Group::Bin {
                    width: 10.0,
                    start: None
                })
            )),
            "age (bins of 10)"
        );
        assert_eq!(title(&dim("d", Some(Group::YearMonth))), "d (year-month)");
        assert_eq!(title(&dim("s", None)), "s");
    }

    #[test]
    fn quantile_function() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&xs, 0.0), 1.0);
        assert_eq!(quantile(&xs, 1.0), 4.0);
        assert_eq!(quantile(&xs, 0.5), 2.5);
        assert!((quantile(&xs, 0.25) - 1.75).abs() < 1e-12);
        assert!(quantile(&[], 0.5).is_nan());
    }
}
